"""Video generation pipeline using image-to-video chaining."""

import subprocess
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, WanImageToVideoPipeline
from PIL import Image

from .config import Config, Scene


class VideoPipeline:
    """Generates videos by chaining anchor images through I2V models."""

    def __init__(self, config: Config, device: str = "cuda"):
        """Initialize pipeline with image and video generation models."""
        self.config = config
        self.device = device
        self.is_flux = "flux" in config.image_model.lower()
        self.reference_image: Image.Image | None = None
        self.ip_adapter_loaded = False
        # Validate IP-Adapter compatibility
        if (
            config.reference_image or config.use_first_frame_as_reference
        ) and self.is_flux:
            raise ValueError(
                "IP-Adapter requires SDXL. Set image_model to "
                "'stabilityai/stable-diffusion-xl-base-1.0' when using reference_image."
            )
        # Flux uses bfloat16, others use float16
        img_dtype = torch.bfloat16 if self.is_flux else torch.float16
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            config.image_model,
            torch_dtype=img_dtype,
        ).to(device)
        # Load IP-Adapter for character consistency (if reference provided upfront)
        if config.reference_image:
            self._load_ip_adapter()
            self.reference_image = Image.open(config.reference_image).convert("RGB")
            print(
                f"Loaded IP-Adapter with reference image, scale={config.ip_adapter_scale}"
            )
        self.video_pipe = WanImageToVideoPipeline.from_pretrained(
            config.video_model,
            torch_dtype=torch.float16,
        ).to(device)
        self.video_pipe.vae.enable_slicing()
        self.video_pipe.transformer = torch.compile(
            self.video_pipe.transformer,
            mode="reduce-overhead",
            fullgraph=True,
        )
        print("Compiled video transformer (first run will be slow)")

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter weights into the image pipeline."""
        if self.ip_adapter_loaded:
            return
        self.image_pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        self.image_pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)
        self.ip_adapter_loaded = True

    def generate_anchor(self, scene: Scene) -> Image.Image:
        """Generate or load the anchor image for a scene."""
        if scene.anchor_image:
            return Image.open(scene.anchor_image).convert("RGB")

        kwargs = {
            "prompt": scene.prompt,
            "width": self.config.width,
            "height": self.config.height,
            "generator": (
                torch.Generator(self.device).manual_seed(scene.seed)
                if scene.seed
                else None
            ),
        }
        # Flux doesn't support negative_prompt
        if not self.is_flux:
            kwargs["negative_prompt"] = scene.negative_prompt
        # IP-Adapter for character consistency
        if self.reference_image is not None:
            kwargs["ip_adapter_image"] = self.reference_image
        return self.image_pipe(**kwargs).images[0]

    @torch.inference_mode()
    def generate_clip(self, anchor: Image.Image, scene: Scene) -> list[Image.Image]:
        """Generate a video clip from an anchor image using I2V."""
        num_frames = int(scene.duration * self.config.fps)
        result = self.video_pipe(
            image=anchor.resize((self.config.width, self.config.height)),
            prompt=scene.prompt,
            negative_prompt=scene.negative_prompt,
            num_frames=num_frames,
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
        """Run the full pipeline: generate all scenes and concatenate."""
        clips: list[Path] = []
        anchor: Image.Image | None = None
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, scene in enumerate(self.config.scenes):
                print(
                    f"[{i + 1}/{len(self.config.scenes)}] Generating: {scene.prompt[:50]}..."
                )
                if anchor is None:
                    anchor = self.generate_anchor(scene)
                    # Use first frame as IP-Adapter reference for subsequent scenes
                    if (
                        self.config.use_first_frame_as_reference
                        and not self.ip_adapter_loaded
                    ):
                        self._load_ip_adapter()
                        self.reference_image = anchor
                        print(
                            f"Using first frame as IP-Adapter reference, scale={self.config.ip_adapter_scale}"
                        )
                frames = self.generate_clip(anchor, scene)
                # Convert frames to uint8 for video encoding
                frames_uint8 = [self._to_uint8(f) for f in frames]
                # Last frame becomes next anchor (convert back to PIL)
                anchor = Image.fromarray(frames_uint8[-1])
                clip_path = Path(tmpdir) / f"clip_{i:03d}.mp4"
                iio.imwrite(
                    clip_path,
                    frames_uint8,
                    fps=self.config.fps,
                    codec="libx264",
                )
                clips.append(clip_path)
            self._concat_clips(clips, self.config.output)
        return self.config.output

    def _concat_clips(self, clips: list[Path], output: Path) -> None:
        """Concatenate video clips using ffmpeg."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")
            concat_file = f.name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
