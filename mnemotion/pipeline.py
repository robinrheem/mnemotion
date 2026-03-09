"""Video generation pipeline using image-to-video chaining."""

import subprocess
import tempfile
from pathlib import Path

import imageio.v3 as iio
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

        # Flux uses bfloat16, others use float16
        img_dtype = torch.bfloat16 if self.is_flux else torch.float16

        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            config.image_model,
            torch_dtype=img_dtype,
        ).to(device)

        self.video_pipe = WanImageToVideoPipeline.from_pretrained(
            config.video_model,
            torch_dtype=torch.float16,
        ).to(device)

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

        return self.image_pipe(**kwargs).images[0]

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

                frames = self.generate_clip(anchor, scene)
                anchor = frames[-1]  # last frame becomes next anchor

                clip_path = Path(tmpdir) / f"clip_{i:03d}.mp4"
                iio.imwrite(
                    clip_path,
                    [frame for frame in frames],
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
