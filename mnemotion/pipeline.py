"""Video generation pipeline using HunyuanVideo + FramePack."""

from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import (
    AutoPipelineForText2Image,
    FluxKontextPipeline,
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
        self.kontext_pipe = None
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

    def _load_kontext_pipe(self) -> None:
        """Load FLUX Kontext pipeline for character-consistent generation."""
        if self.kontext_pipe is not None:
            return
        print("Loading FLUX Kontext model...")
        self.kontext_pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        print("Loaded FLUX Kontext")

    def _unload_kontext_pipe(self) -> None:
        """Unload Kontext pipeline to free GPU memory."""
        if self.kontext_pipe is not None:
            del self.kontext_pipe
            self.kontext_pipe = None
            torch.cuda.empty_cache()
            print("Unloaded Kontext model")

    def generate_anchor(
        self, scene: Scene, reference: Image.Image | None = None
    ) -> Image.Image:
        """Generate or load the anchor image for a scene.

        Args:
            scene: Scene configuration with prompts
            reference: Optional reference image for character consistency (Kontext)
        """
        if scene.anchor_image:
            return Image.open(scene.anchor_image).convert("RGB")

        prompt = scene.anchor_prompt or scene.prompt
        generator = (
            torch.Generator(self.device).manual_seed(scene.seed) if scene.seed else None
        )

        if reference is not None and self.config.use_kontext:
            # Use Kontext for character-consistent generation
            return self._generate_anchor_kontext(prompt, reference, generator)
        else:
            # Use standard text-to-image
            return self._generate_anchor_t2i(scene, prompt, generator)

    def _generate_anchor_t2i(
        self, scene: Scene, prompt: str, generator: torch.Generator | None
    ) -> Image.Image:
        """Generate anchor using text-to-image (FLUX.1-dev)."""
        self._load_image_pipe()
        kwargs = {
            "prompt": prompt,
            "width": self.config.width,
            "height": self.config.height,
            "generator": generator,
        }
        if not self.is_flux:
            kwargs["negative_prompt"] = scene.negative_prompt
        return self.image_pipe(**kwargs).images[0]

    def _generate_anchor_kontext(
        self,
        prompt: str,
        reference: Image.Image,
        generator: torch.Generator | None,
    ) -> Image.Image:
        """Generate anchor using Kontext for character consistency."""
        # Unload other models to free memory before loading Kontext
        self._unload_image_pipe()
        self._unload_framepack_pipe()
        self._load_kontext_pipe()

        # Resize reference to target dimensions
        reference = reference.resize((self.config.width, self.config.height))

        result = self.kontext_pipe(
            image=reference,
            prompt=prompt,
            width=self.config.width,
            height=self.config.height,
            guidance_scale=2.5,
            num_inference_steps=28,
            generator=generator,
        )
        return result.images[0]

    def _unload_framepack_pipe(self) -> None:
        """Unload FramePack pipeline to free GPU memory."""
        if self.framepack_pipe is not None:
            del self.framepack_pipe
            self.framepack_pipe = None
            torch.cuda.empty_cache()
            print("Unloaded FramePack model")

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
        # torch.compile disabled - FramePack has dynamic indexing that inductor can't handle
        print("Loaded HunyuanVideo + FramePack with CPU offload")

    @torch.inference_mode()
    def generate_clip(
        self,
        anchor: Image.Image,
        scene: Scene,
        last_image: Image.Image | None = None,
    ) -> list[Image.Image]:
        """Generate a video clip using HunyuanVideo + FramePack."""
        # Unload image models to free memory for FramePack
        self._unload_image_pipe()
        self._unload_kontext_pipe()
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
        prev_anchor = None  # For Kontext chaining
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
                    # Generate anchor for this scene
                    # First scene: no reference (text-to-image)
                    # Subsequent scenes: use previous anchor as reference (Kontext)
                    anchor = self.generate_anchor(scene, reference=prev_anchor)
                    prev_anchor = anchor  # Chain for next scene
                    if not self.config.fresh_anchors:
                        self._unload_image_pipe()  # Free memory if only used for first scene
                else:
                    # Chain from previous scene's last frame (no Kontext)
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
            # Unload all pipes at end
            self._unload_image_pipe()
            self._unload_kontext_pipe()
            self._unload_framepack_pipe()
        finally:
            writer.close()
        print(f"Wrote {frame_count} frames to {self.config.output}")
        return self.config.output
