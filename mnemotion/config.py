"""Configuration schema for video generation."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class Scene(BaseModel):
    """A single scene in the video generation pipeline."""

    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    duration: float = 4.0  # seconds
    seed: int | None = None
    anchor_image: Path | None = None  # override first frame


class Config(BaseModel):
    """Top-level configuration for video generation."""

    scenes: list[Scene]
    output: Path = Path("output.mp4")
    fps: int = 24
    width: int = 1280
    height: int = 720
    image_model: str = "black-forest-labs/FLUX.1-dev"
    video_model: str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
