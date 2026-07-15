"""Step 1.2 — DRR generation: 2 views, correct camera geometry + motion sim.
COLAB ONLY — requires a CUDA GPU (DeepDRR / TIGRE cone-beam projection).
Does NOT run on Apple Silicon (M4) locally. Driven from
notebooks/colab_drr_generation.ipynb.

Generates 2 non-simultaneous cone-beam projections per case, with an
independent rigid perturbation (+/-10 deg rotation, +/-8mm translation)
applied to the second view only, following DeepCA's (Wang et al.,
WACV 2025) exact motion-simulation protocol. Saves the projection
matrix alongside every image.
"""

import numpy as np

MOTION_ROTATION_DEG = 10.0   # +/- range, second view only
MOTION_TRANSLATION_MM = 8.0  # +/- range, second view only


def sample_motion_perturbation(rng: np.random.Generator):
    """Sample a rigid perturbation matching DeepCA's protocol
    (+/-10 deg rotation, +/-8mm translation), applied only to view 2.
    """
    rot = rng.uniform(-MOTION_ROTATION_DEG, MOTION_ROTATION_DEG, size=3)
    trans = rng.uniform(-MOTION_TRANSLATION_MM, MOTION_TRANSLATION_MM, size=2)
    return rot, trans


def generate_case_projections(ct_path, rng):
    """Placeholder — implement with DeepDRR's Volume/Projector API in Colab.

    For each case, produce:
      - view_0: unperturbed cone-beam projection
      - view_1: cone-beam projection after applying sample_motion_perturbation()
      - projection matrices (source/detector pose) for BOTH views

    Returns:
        dict with keys: "images" (2, H, W), "poses" (list of 2 pose dicts)
    """
    raise NotImplementedError(
        "Run in Colab with DeepDRR installed (pip install -r requirements-colab.txt). "
        "See notebooks/colab_drr_generation.ipynb and README's DeepDRR usage example."
    )
