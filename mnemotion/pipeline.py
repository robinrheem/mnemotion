"""Video generation pipeline using HunyuanVideo + FramePack."""

from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import (
    AutoPipelineForText2Image,
    HunyuanVideoFramepackPipeline,
    HunyuanVideoFramepackTransformer3DModel,
)
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel

from .config import Config, Scene


class VideoPipeline:
    """Generates videos using HunyuanVideo + FramePack (anti-drift)."""

    def __init__(self, config: Config, device: str = "cuda"):
        """Initialize pipeline with configuration."""
        self.config = config
        self.device = device
        self.is_flux = "flux" in config.image_model.lower()
        # Pipeline instances (lazy loaded)
        self.image_pipe = None
        self.framepack_pipe = None

    def _load_image_pipe(self) -> None:
        """Load image generation pipeline to GPU."""
        if self.image_pipe is not None:
            return
        print("Loading image model...")
        img_dtype = torch.bfloat16 if self.is_flux else torch.float16
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            self.config.image_model,
            torch_dtype=img_dtype,
        ).to(self.device)
        if self.config.style_lora is not None:
            print(f"Loading style LoRA: {self.config.style_lora}")
            self.image_pipe.load_lora_weights(str(self.config.style_lora))

    def _unload_image_pipe(self) -> None:
        """Unload image pipeline to free GPU memory."""
        if self.image_pipe is not None:
            del self.image_pipe
            self.image_pipe = None
            torch.cuda.empty_cache()
            print("Unloaded image model")

    def generate_anchor(self, scene: Scene) -> Image.Image:
        """Generate or load the anchor image for a scene."""
        if scene.anchor_image:
            return Image.open(scene.anchor_image).convert("RGB")
        self._load_image_pipe()
        # Use anchor_prompt for visual details, fall back to motion prompt
        prompt = scene.anchor_prompt or scene.prompt
        kwargs = {
            "prompt": prompt,
            "width": self.config.width,
            "height": self.config.height,
            "generator": (
                torch.Generator(self.device).manual_seed(scene.seed)
                if scene.seed
                else None
            ),
        }
        if not self.is_flux:
            kwargs["negative_prompt"] = scene.negative_prompt
        return self.image_pipe(**kwargs).images[0]

    def _load_framepack_pipe(self) -> None:
        """Load HunyuanVideo + FramePack pipeline to GPU."""
        if self.framepack_pipe is not None:
            return
        print("Loading HunyuanVideo + FramePack model...")
        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            self.config.framepack_model,
            torch_dtype=torch.bfloat16,
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder="feature_extractor",
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.framepack_pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            self.config.hunyuan_model,
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        self.framepack_pipe.enable_model_cpu_offload()
        self.framepack_pipe.vae.enable_tiling()
        self.framepack_pipe.vae.enable_slicing()
        self.framepack_pipe.transformer = torch.compile(
            self.framepack_pipe.transformer, mode="default"
        )
        print("Loaded HunyuanVideo + FramePack with CPU offload + torch.compile")

    @torch.inference_mode()
    def generate_clip(
        self,
        anchor: Image.Image,
        scene: Scene,
        last_image: Image.Image | None = None,
    ) -> list[Image.Image]:
        """Generate a video clip using HunyuanVideo + FramePack."""
        self._load_framepack_pipe()
        num_frames = int(scene.duration * self.config.fps)
        num_frames = max(17, num_frames)  # minimum 17 frames for FramePack
        # Load last_image if provided in scene config
        if last_image is None and scene.last_image:
            last_image = Image.open(scene.last_image).convert("RGB")
        result = self.framepack_pipe(
            image=anchor.resize((self.config.width, self.config.height)),
            last_image=(
                last_image.resize((self.config.width, self.config.height))
                if last_image
                else None
            ),
            prompt=scene.prompt,
            negative_prompt=scene.negative_prompt,
            height=self.config.height,
            width=self.config.width,
            num_frames=num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            sampling_type=self.config.framepack_sampling,
            generator=(
                torch.Generator(self.device).manual_seed(scene.seed)
                if scene.seed
                else None
            ),
        )
        return result.frames[0]

    def _to_uint8(self, frame: np.ndarray | Image.Image) -> np.ndarray:
        """Convert a frame to uint8 numpy array."""
        if isinstance(frame, Image.Image):
            return np.array(frame, dtype=np.uint8)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            return (frame * 255).clip(0, 255).astype(np.uint8)
        return frame.astype(np.uint8)

    def run(self) -> Path:
        """Run the full FramePack pipeline."""
        frame_count = 0
        prev_last_frame = None
        writer = imageio.get_writer(
            self.config.output,
            fps=self.config.fps,
            codec="libx264",
            quality=8,
        )
        try:
            for i, scene in enumerate(self.config.scenes):
                print(f"[{i + 1}/{len(self.config.scenes)}] {scene.prompt[:50]}...")
                # Get anchor image
                if scene.anchor_image:
                    anchor = Image.open(scene.anchor_image).convert("RGB")
                elif self.config.fresh_anchors or i == 0:
                    # Generate fresh anchor for this scene
                    anchor = self.generate_anchor(scene)
                    if not self.config.fresh_anchors:
                        self._unload_image_pipe()  # Free memory if only used for first scene
                else:
                    # Chain from previous scene's last frame
                    anchor = prev_last_frame
                # Load last_image if provided
                last_image = None
                if scene.last_image:
                    last_image = Image.open(scene.last_image).convert("RGB")
                frames = self.generate_clip(anchor, scene, last_image)
                for frame in frames:
                    writer.append_data(self._to_uint8(frame))
                    frame_count += 1
                prev_last_frame = Image.fromarray(self._to_uint8(frames[-1]))
            # Unload image pipe at end if we used fresh_anchors
            if self.config.fresh_anchors:
                self._unload_image_pipe()
        finally:
            writer.close()
        print(f"Wrote {frame_count} frames to {self.config.output}")
        return self.config.output
