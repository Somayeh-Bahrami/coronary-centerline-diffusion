"""Step 1.2 -- DRR generation: 2 views, correct camera geometry + motion sim.
Requires a CUDA GPU (TIGRE's cone-beam projection) -- does NOT run on
Apple Silicon (M4) locally. Run on Kaggle Notebooks (GPU P100).

Pipeline history (see docs/work_breakdown.md Step 1.2 for full narrative):

1. Projection tool: TIGRE, not DeepDRR. TIGRE performs a direct
   HU-derived linear-attenuation-coefficient line integral (Beer-Lambert),
   with no material-classification step -- this avoids DeepDRR's 3-class
   (air/soft tissue/bone) decomposition, which was found to flatten
   contrast-enhanced vessel voxels toward generic soft tissue (measured
   vessel-vs-background contrast effect size: only 0.35).

2. Contrast representation: single, non-subtracted projections, NOT
   Digital Subtraction Angiography (DSA). An earlier version of this
   pipeline used DSA (pre/post-contrast subtraction) to boost contrast,
   but routine coronary angiography is normally acquired without
   background subtraction, since cardiac motion causes misregistration
   artifacts under subtraction. Single-shot projections keep training
   data closer to real deployment conditions, even though contrast is
   more modest as a result -- physically consistent with why 40-50% of
   real lesions fall into a visually ambiguous gray zone.

3. Projection matrix: TIGRE's internal camera-geometry convention could
   not be reliably derived analytically (two independent attempts each
   produced 20+ pixel errors against TIGRE's real Ax() output). Instead,
   the 3x4 projection matrix is calibrated empirically per view via a
   Direct Linear Transform against known synthetic marker points
   projected through TIGRE's real Ax(), validated to sub-pixel accuracy
   on held-out points. This also resolves an earlier feedback issue:
   poses are stored as a clean (2,3,4) float32 array, not a dict/object
   array (which causes PyTorch DataLoader issues).

4. Motion simulation: an independent rigid perturbation (+/-10 deg
   rotation, +/-8mm translation) applied to view 1 only, following
   DeepCA's (Wang et al., WACV 2025) protocol. Both alpha and beta
   angulation are passed to TIGRE's Ax() (an earlier version only
   applied alpha, silently ignoring beta).

Limitation, stated explicitly: both views are projections of the same
static CT volume -- there is no simulated cardiac motion/deformation
between the two views, only a rigid camera perturbation. This is a
simplified, "mostly static" approximation of real 2-view acquisition.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
import tigre

MOTION_ROTATION_DEG = 10.0
MOTION_TRANSLATION_MM = 8.0
VIEW0_ALPHA, VIEW0_BETA = 0.0, 0.0
VIEW1_ALPHA, VIEW1_BETA = 30.0, 10.0

SENSOR_SIZE = 512
PIXEL_SIZE_MM = 0.582
DSO = 750.0
DSD = 1020.0
MU_WATER = 0.02


def sample_motion_perturbation(rng: np.random.Generator):
    """Sample a rigid perturbation matching DeepCA's protocol
    (+/-10 deg rotation, +/-8mm translation), applied only to view 2.
    """
    rot = rng.uniform(-MOTION_ROTATION_DEG, MOTION_ROTATION_DEG, size=3)
    trans = rng.uniform(-MOTION_TRANSLATION_MM, MOTION_TRANSLATION_MM, size=3)
    return rot, trans


def _hu_to_mu(hu_volume: np.ndarray) -> np.ndarray:
    """Linear HU -> attenuation-coefficient proxy, clipped at 0 (air)."""
    return np.clip(MU_WATER * (1.0 + hu_volume / 1000.0), 0, None)


def _build_geometry(volume_shape, voxel_spacing):
    geo = tigre.geometry(mode="cone", nVoxel=np.array(
        volume_shape), default=True)
    geo.dVoxel = np.array(voxel_spacing, dtype=np.float32)
    geo.sVoxel = geo.dVoxel * geo.nVoxel
    geo.DSO = DSO
    geo.DSD = DSD
    geo.nDetector = np.array([SENSOR_SIZE, SENSOR_SIZE])
    geo.dDetector = np.array([PIXEL_SIZE_MM, PIXEL_SIZE_MM], dtype=np.float32)
    geo.sDetector = geo.dDetector * geo.nDetector
    return geo


def calibrate_projection_matrix(alpha, beta, offOrigin, n_points=10,
                                cal_shape=(50, 50, 50), cal_spacing=(4.0, 4.0, 4.0), seed=0):
    """Empirically determine the 3x4 projection matrix P for this exact
    (alpha, beta, offOrigin) by placing known markers in a small dummy
    volume, projecting with TIGRE's real Ax(), and solving via Direct
    Linear Transform (DLT). Sidesteps needing to know TIGRE's internal
    rotation convention -- uses real behavior only. Validated to
    sub-pixel accuracy on held-out points before adoption.
    """
    cal_geo = _build_geometry(cal_shape, cal_spacing)
    cal_geo.offOrigin = np.array(offOrigin, dtype=np.float32)
    angles = np.array([[alpha, beta, 0.0]], dtype=np.float32)

    rng_cal = np.random.default_rng(seed)
    center = np.array(cal_shape) / 2
    correspondences = []
    for _ in range(n_points):
        idx = rng_cal.integers(8, np.array(cal_shape) - 8)
        vol = np.zeros(cal_shape, dtype=np.float32)
        vol[tuple(idx)] = 1.0
        proj = tigre.Ax(vol, cal_geo, angles)[0]
        row, col = np.unravel_index(np.argmax(proj), proj.shape)
        point_mm = (idx - center) * np.array(cal_spacing)
        correspondences.append((point_mm, col, row))

    A = []
    for (X, x, y) in correspondences:
        X_h = np.array([*X, 1.0])
        A.append(np.concatenate([X_h, np.zeros(4), -x * X_h]))
        A.append(np.concatenate([np.zeros(4), X_h, -y * X_h]))
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    P = P / P[-1, -1]
    return P.astype(np.float32)


def generate_case_projections(img_path: Path, label_path: Path, rng: np.random.Generator) -> dict:
    """Generate the 2-view single-shot DRR pair for one case.

    Returns:
        dict with keys:
          "images": (2, H, W) single-shot intensity projections.
          "masks": (2, H, W) vessel-mask projections (ground truth for
              Step 2.1's baseline -- same geometry as "images").
          "poses": (2, 3, 4) float32 calibrated projection matrices.
    """
    nii = nib.load(str(img_path))
    hu_volume = nii.get_fdata().astype(np.float32)
    voxel_spacing = nii.header.get_zooms()[:3]

    vessel_mask = nib.load(str(label_path)).get_fdata() > 0.5
    if vessel_mask.shape != hu_volume.shape:
        raise RuntimeError(
            f"Shape mismatch: mask {vessel_mask.shape} vs CT {hu_volume.shape} -- "
            f"check img/label co-registration for this case."
        )

    mu_volume = _hu_to_mu(hu_volume)
    mu_mask = vessel_mask.astype(np.float32)

    geo = _build_geometry(hu_volume.shape, voxel_spacing)
    rot, trans = sample_motion_perturbation(rng)
    view_angles_deg = [(VIEW0_ALPHA, VIEW0_BETA),
                       (VIEW1_ALPHA + rot[0], VIEW1_BETA + rot[1])]

    images, masks, poses = [], [], []
    for view_idx, (alpha_deg, beta_deg) in enumerate(view_angles_deg):
        alpha = np.radians(alpha_deg)
        beta = np.radians(beta_deg)
        offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if view_idx == 1:
            offOrigin = np.array(trans, dtype=np.float32)
        geo.offOrigin = offOrigin

        angles = np.array([[alpha, beta, 0.0]], dtype=np.float32)
        images.append(tigre.Ax(mu_volume, geo, angles)[0])
        masks.append(tigre.Ax(mu_mask, geo, angles)[0])

        P = calibrate_projection_matrix(alpha, beta, offOrigin)
        poses.append(P)

    return {
        "images": np.stack(images),
        "masks": np.stack(masks),
        "poses": np.stack(poses),
    }


def generate_all(raw_dir: Path, out_dir: Path, case_ids=None, seed: int = 0):
    """Batch driver -- generates DRR projections + masks + poses for
    every case in raw_dir (or a specified subset via case_ids).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    img_files = sorted(raw_dir.glob("*.img.nii.gz"))
    if case_ids:
        case_ids = set(case_ids)
        img_files = [f for f in img_files if int(
            f.stem.split('.')[0]) in case_ids]

    for img_path in img_files:
        case_id = img_path.stem.split('.')[0]
        label_path = raw_dir / f"{case_id}.label.nii.gz"
        if not label_path.exists():
            print(f"SKIP case {case_id}: no matching label file")
            continue
        try:
            result = generate_case_projections(img_path, label_path, rng)
            np.savez(
                out_dir / f"case_{case_id}_projections.npz",
                images=result["images"], masks=result["masks"], poses=result["poses"],
            )
            print(
                f"case {case_id}: wrote {result['images'].shape}, poses {result['poses'].shape}")
        except Exception as e:
            print(f"FAILED case {case_id}: {e}")
