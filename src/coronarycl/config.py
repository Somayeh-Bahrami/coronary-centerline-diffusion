"""Config loading. Every run is driven by a YAML config (see configs/default.yaml),
following the same pattern as configs/efficientnet.yaml / configs/vit.yaml in
the team's dermai-explainability repo.
"""

from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(requested: str = "auto") -> str:
    """Auto-detect device, matching dermai's cross-platform pattern:
    CUDA on Colab/PACE, MPS on Apple Silicon, CPU otherwise.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
