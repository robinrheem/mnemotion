# mnemotion

Memory-Persistent Long-Form Video Synthesis using HunyuanVideo + FramePack.

## Installation

```bash
uv sync
```

## Usage

```bash
# Generate a video from a scene config
uv run mnemotion generate example.yaml
```

## Evaluation with VBench

[VBench](https://github.com/Vchitect/VBench) is a comprehensive benchmark for video generation. Due to dependency conflicts (vbench requires `transformers==4.33.2`), it must be installed separately.

### Install VBench

```bash
uv tool install vbench
```

### Run Evaluation

```bash
# Evaluate a single dimension
vbench evaluate \
    --videos_path output.mp4 \
    --dimension subject_consistency \
    --mode custom_input \
    --output_path ./vbench_results

# Evaluate all 16 dimensions
for dim in subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality object_class multiple_objects human_action color spatial_relationship scene temporal_style appearance_style overall_consistency; do
    vbench evaluate --videos_path output.mp4 --dimension $dim --mode custom_input --output_path ./vbench_results
done
```

### Available Dimensions

| Dimension | Description | SOTA (Kling) |
|-----------|-------------|--------------|
| subject_consistency | Character looks same across frames | 0.983 |
| background_consistency | Background stays stable | 0.976 |
| temporal_flickering | Frame-to-frame jitter (higher = less flicker) | 0.993 |
| motion_smoothness | Natural motion quality | 0.994 |
| dynamic_degree | Amount of motion | 0.469 |
| aesthetic_quality | Visual appeal | 0.612 |
| imaging_quality | Technical image quality | 0.656 |
| object_class | Object recognition accuracy | 0.872 |
| multiple_objects | Multiple object handling | 0.681 |
| human_action | Human action recognition | 0.934 |
| color | Color accuracy | 0.899 |
| spatial_relationship | Spatial reasoning | 0.730 |
| scene | Scene recognition | 0.509 |
| temporal_style | Temporal style consistency | 0.242 |
| appearance_style | Visual style consistency | 0.196 |
| overall_consistency | Overall coherence | 0.264 |

SOTA reference scores from [VBench++ paper](https://arxiv.org/abs/2411.13503) (Kling model).
