"""CLI entrypoint for mnemotion."""

from pathlib import Path

import typer

from .config import Config
from .pipeline import VideoPipeline

app = typer.Typer()


@app.command()
def generate(
    config_path: Path = typer.Argument(..., help="Path to YAML config file"),
    device: str = typer.Option("cuda", help="Device to run on"),
):
    """Generate a video from a scene config file."""
    config = Config.from_yaml(config_path)
    pipeline = VideoPipeline(config, device=device)
    output = pipeline.run()
    print(f"Done: {output}")


if __name__ == "__main__":
    app()
