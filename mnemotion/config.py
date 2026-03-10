"""Configuration schema for video generation."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class Scene(BaseModel):
    """A single scene in the video generation pipeline."""

    prompt: str  # Motion/action description for video generation
    anchor_prompt: str | None = (
        None  # Visual details for anchor image (falls back to prompt)
    )
    negative_prompt: str = "blurry, low quality, distorted, NSFW"
    duration: float = 4.0  # seconds
    seed: int | None = None
    anchor_image: Path | None = None  # override first frame with file
    last_image: Path | None = None  # end frame for bidirectional generation


class Config(BaseModel):
    """Top-level configuration for video generation."""

    scenes: list[Scene]
    output: Path = Path("output.mp4")
    fps: int = 24
    width: int = 1280
    height: int = 720
    # Image model for anchor generation
    image_model: str = "black-forest-labs/FLUX.1-dev"
    # HunyuanVideo + FramePack settings
    framepack_model: str = "lllyasviel/FramePack_F1_I2V_HY_20250503"
    hunyuan_model: str = "hunyuanvideo-community/HunyuanVideo"
    framepack_sampling: str = "vanilla"  # "vanilla" or "inverted_anti_drifting"
    # Generation parameters
    guidance_scale: float = 6.0
    num_inference_steps: int = 30
    # If true, generate fresh anchor for each scene instead of chaining last frames
    fresh_anchors: bool = False
    # Style LoRA for consistent visual style
    style_lora: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
