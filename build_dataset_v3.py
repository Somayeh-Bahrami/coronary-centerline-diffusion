"""Dataset v3 (Plan A) — single isotropic frame, GT regenerated after crop.

BUILDER VERSION 3.4.  Every .npz carries `builder_version`; refuse to mix
outputs from different builder versions in one directory (see --overwrite).

This builder REUSES the repo's validated modules rather than duplicating them,
so there is exactly one implementation of each piece of logic:

    src.coronarycl.centerline     _traversal_order, _classify_topology, LABEL_*
    src.coronarycl.preprocessing  pad_centerline, compute_centerline_norm_stats
    src.coronarycl.splits         make_case_level_split, write_splits

Order of operations (each step feeds the next, one coordinate frame throughout):

    ImageCAS <id>.label.nii.gz
      -> RCA / LCA split (3D connected components, side from NIfTI affine)
      -> 96 mm crop, bbox-centred, SHIFTED (not shrunk) at volume edges
      -> resample ONCE to isotropic (iso <= native min spacing: upsampling only)
      -> skeletonize  +  distance_transform_edt(sampling=iso)  -> radius in mm
      -> topology labels  +  DFS traversal ordering  (centerline.py)
      -> centerline mm relative to the isotropic volume centre = isocentre
      -> TIGRE projection of THAT SAME isotropic volume (DeepCA RENDER geometry)
      -> centroid-DLT pose calibration, twice: render pose and scanner pose
      -> hard gate, pad to fixed length (preprocessing.py), save

===========================================================================
v3.1 CORRECTIONS (verified against DeepCA data_simulation.py @ c01ab96)

P1-a  Motion translation is 2-AXIS by default (tz == 0), per DeepCA L191.
      --motion_3d is an explicitly NON-DeepCA option.
P1-b  P_scanner / P_render separated.  DeepCA renders view 2 with motion
      (L191/196) then reconstructs with NOMINAL geometry (L211/212: offOrigin
      reset to 0, angles reset to un-perturbed primary/secondary).  DSD/DSO are
      NOT reset there -- they are C-arm readouts, not patient motion.
          poses         -> nominal scanner geometry   (MODEL INPUT)
          poses_render  -> motion-carrying geometry   (VALIDATION ONLY)
      *** Projecting the GT through poses[1] no longer lands on images[1].
          That mismatch IS the task. ***
P1-c  mask_kept is a rejection criterion (--min_mask_kept).
P1-d  LCA/RCA named only when identifiable (exactly 2 components, separated on
      the affine L-R axis).  Merged / >2-component patients are skipped.
P2-e  DLT RMS measured on HELD-OUT markers.
P2-f  FOV gate is per-point, not a cube-side heuristic.
P2-g  Detector spacing corrected to DeepCA's [0.2779, 0.2789].

v3.2 CORRECTIONS (this file)

P1-h  *** THE 0.35 mm "PURE UPSAMPLING" CLAIM WAS FALSE. ***  508 of the 1000
      local ImageCAS cases have native in-plane spacing below 0.35 mm (min
      0.289 mm), so `zoom(order=0)` was DOWNsampling those cases and could
      break thin distal branches, altering topology and radius.
      Fixes, all three:
        * --iso auto (NEW DEFAULT) = min(ISO_MM_CAP, native spacing.min()) per
          case, so factors >= 1 on every axis and the claim is true by
          construction.  --iso 0.35 restores a fixed grid (not recommended).
        * --max_iso_voxels REJECTS an unaffordable grid instead of quietly
          coarsening it (an early draft clamped iso at a floor, which silently
          voided the guarantee again for sub-floor cases).
        * Topology is now GATED, not merely reported: the vessel mask is one
          connected component by construction, so its skeleton must be too.
          n_skel_components != 1 => REJECT.  The pre-resample component count
          is recorded alongside it, so a break can be attributed to the
          resample rather than the crop.

P1-i  NaN COULD EVADE DLT REJECTION.  Python's max() is order-dependent with
      NaN: max([0.5, nan]) == 0.5, so a NaN held-out RMS (fewer than 3 visible
      validation markers) was hidden whenever a finite value came first.  The
      bug was present at BOTH aggregation levels (per-view max(fr, fs) and the
      per-sample gate).  All RMS values now go into one flat array and the gate
      requires np.isfinite(...).all() before comparing the max.

P1-j  STALE OUTPUT COULD CONTAMINATE A RUN.  The builder reused ./dataset_v3
      with exist_ok=True and never checked it was empty, so a sample written by
      an older builder -- with motion baked into `poses` -- could survive a run
      that skipped or rejected it.  Now: default out_dir is ./dataset_v3_1, the
      builder REFUSES to start if .npz files are present (--overwrite clears
      them), and every sample is stamped with builder_version so a loader can
      assert on it.

P2-k  GATE FAILURE NOW EXITS NONZERO (SystemExit(1)), so a failed build cannot
      be followed by training in the same shell/notebook cell.

P2-l  geo.accuracy IS NOW SET EXPLICITLY (default 0.5) instead of inherited
      from TIGRE's default.
      *** DELIBERATE DIVERGENCE FROM DeepCA. ***  DeepCA sets accuracy = 1
      (data_simulation.py L137); TIGRE's default is 0.5
      (utilities/geometry_default.py L29/51/73).  The units are vx/sample, so
      accuracy=1 samples the ray every 1 voxel and accuracy=0.5 every half
      voxel -- DeepCA's value is COARSER, not finer.  Copying it here would be
      a fidelity win and an accuracy loss: our voxels are ~0.3 mm against
      DeepCA's ~0.75 mm, so a 1-voxel ray step is far likelier to step over a
      1-2 voxel distal branch and punch holes in a binary silhouette that is
      thresholded at > 1e-6.  We keep 0.5 and record the divergence.
      --tigre_accuracy 1.0 reproduces DeepCA exactly if you want that arm.

P2-m  SILHOUETTE CLIPPING IS CHECKED, AND RECORDED RATHER THAN REJECTED.
      (v3.2 rejected on it; the 20-patient pilot showed that was wrong -- see
      P1-o.  --reject_clipped restores the old behaviour.)

P2-n  --min_motion_effect NO LONGER REJECTS BY DEFAULT.  Rejecting samples
      whose motion happened to be small removed the low-motion tail of a
      distribution DeepCA samples uniformly over [-8,8] mm / [-10,10] deg, and
      that is a dataset bias introduced by the QC rather than a defect being
      caught.  It is now a DIAGNOSTIC, with two honest guards kept:
        * per-sample: rejected only when the SAMPLED motion was large
          (--motion_probe_trans_mm / --motion_probe_rot_deg) yet produced no
          measurable effect -- that combination is a wiring bug, not a small
          draw.
        * dataset-level: the gate fails if the mean effect is ~zero or the
          effect does not correlate with sampled motion magnitude.

v3.3 CORRECTIONS (from the first 20-patient CUDA pilot)

P1-o  *** THE CLIPPING GATE WAS MEASURING THE WRONG THING, AND THE FOV IS THE
      REAL CONSTRAINT. ***  The pilot rejected 13/38 samples (34%); all 13 were
      flagged "silhouette clipped".  The geometry explains it:

          detector FOV at isocentre = 512 * ~0.2784 * DSO/DSD
                                    = 115.4 mm (view 1 best) ... 99.2 mm (view 2 worst)
          a 96 mm cube rotated by th spans 96*(|cos th| + |sin th|)
                                    = 121 mm at 18 deg (DeepCA's MINIMUM primary
                                      angle), 131 mm at 30 deg, 136 mm at 42 deg
          largest never-clipping cube = 73-88 mm depending on view and angle

      So a 96 mm crop CANNOT fit DeepCA's detector at any angle they sample:
      clipping is geometrically guaranteed for any vessel that fills the crop,
      not an anomaly.  Shrinking the crop does not help -- it does not shrink
      the vessel, it just moves the loss from "clipped" to "mask_kept".

      More importantly the gate was wrong in principle.  Clinical coronary
      angiography clips vessels at the frame edge routinely, so a clipped
      silhouette is a REALISTIC input, not a defect.  What is genuinely
      disqualifying is a GT point with no support in EITHER view -- no model
      can place it.  So:
        * clipping           -> recorded diagnostic (--reject_clipped opts in)
        * --min_coverage 1.0 -> NEW hard gate: every GT point must be
                                on-detector in at least one view
        * --min_on_detector  -> 0.99 -> 0.95 (per view).  0.99 was arbitrary and
                                unreachable given the FOV arithmetic above.
        * --max_reject_frac  -> 0.02 -> 0.15.  0.02 had no data behind it; the
                                pilot puts the genuinely-broken rate near 10%.

      Related observation, worth a line in the report: 9 of the 13 pilot
      rejections were LCA and only 4 RCA.  The LCA tree (LAD + LCx) is larger
      and exceeds the FOV more often.  DeepCA trains on RCA ALONE (879 samples)
      -- this FOV limit is the likely reason, and is worth citing if the LCA
      yield stays low at scale.

      NOT changed: the mask_kept and single-skeleton gates.  4 of the 13
      rejections (1_LCA, 1_RCA, 4_LCA, 20_RCA) lost 7-16% of the vessel mask AND
      had the tree severed into 2-3 pieces by the 96 mm crop -- and in all four
      the pre-resample component count already equalled the post-resample count,
      so the crop did it, not the resampling.  Those are correct rejections:
      the ground truth really is incomplete.

v3.4 CORRECTIONS (from the first 200-patient run: 88/388 rejected, 22.7%)

P2-p  --min_adjacency NO LONGER REJECTS BY DEFAULT.  It was the SOLE reason for
      19 of the 88 rejections.  Adjacency measures how often consecutive ORDERED
      points are spatial neighbours -- but a DFS walk must jump every time it
      backtracks to a branch point, so adjacency FALLS as a tree gets more
      branched.  Rejecting on it therefore discards the most branched (most
      clinically interesting) trees, which is a dataset bias rather than a data
      defect: the GT is valid, it is merely harder to serialise.  If ordering
      quality hurts the model, the fix is a better ordering algorithm or an
      architecture that does not assume sequential adjacency -- not deleting the
      hard cases.  --reject_low_adjacency restores the old behaviour.
      (Same class of error as P2-n; worth checking any future gate for it.)

P2-q  THE GATE IS SPLIT INTO INTEGRITY vs YIELD, and the reject reasons are now
      tallied by category so the breakdown never has to be counted by hand.
        INTEGRITY  things that can only be wrong if the BUILDER is wrong (DLT
                   rms, downsampling, skeleton connectivity, union coverage,
                   radius sanity, motion wiring, hard failures).  A failure here
                   means do not train, full stop.
        YIELD      how much real data the QC discarded.  Those rejections are
                   individually justified -- incomplete GT, or vessels past
                   DeepCA's detector FOV -- so the limit is a STEP-CHANGE
                   DETECTOR, not a quality bar.
      --max_reject_frac had already been raised twice (0.02 -> 0.15) against
      pilot data; raising it a third time to obtain a green light would have
      been meaningless.  Instead --accept_yield makes the decision explicit and
      recorded.  Observed at 200 patients, with adjacency demoted: ~69/388
      (17.8%), of which the dominant causes are the 96 mm crop severing vessels
      (mask_kept + skeleton_split) and DeepCA's FOV (consistency + on_detector),
      the latter hitting LCA roughly twice as often as RCA.

NOT FIXED HERE (needs a decision or another file):
  * RAO/LAO labels are still not emitted, ON PURPOSE.  Mapping TIGRE's alpha to
    clinical RAO/LAO requires knowing TIGRE's rotation convention relative to
    patient anatomy -- exactly the thing this repo could not derive analytically
    (two attempts, 20+ px error) and works around with empirical DLT.  Emitting
    a guessed label would be worse than emitting none.  Raw primary/secondary
    angles are recorded; derive the clinical label only after establishing the
    convention against a known-orientation phantom.
  * src/coronarycl/dataset.py cannot read v3.2 output (different filenames, no
    vessel_masks, raw-mm unnormalised centerlines, and `poses` with new
    semantics).  See dataset_v3_1.py for a v3.2-compatible Dataset.
===========================================================================

Requires CUDA (TIGRE). Run from the repository root so `src.coronarycl`
imports resolve.

Usage:
    python build_dataset_v3.py --raw_dir DIR --out_dir ./dataset_v3_1 --n 20
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_dilation, center_of_mass,
                           distance_transform_edt, label as cc_label, zoom)
from skimage.morphology import skeletonize
import tigre

sys.path.insert(0, os.getcwd())          # repo root -> `src.coronarycl`
from src.coronarycl.centerline import (          # noqa: E402
    _traversal_order, _classify_topology,
    LABEL_ENDPOINT, LABEL_REGULAR, LABEL_BIFURCATION,
)
from src.coronarycl.preprocessing import (       # noqa: E402
    pad_centerline, compute_centerline_norm_stats,
)
from src.coronarycl.splits import (              # noqa: E402
    make_case_level_split, write_splits,
)

BUILDER_VERSION = "3.4"

# ---------------- DeepCA geometry (Wang et al., WACV 2025; data_simulation.py @ c01ab96) ----
DET_N = 512                                     # L146
DET_SPACING_RANGE = (0.2779, 0.2789)            # L147: 0.2779 + 0.001*rand
V1_DSD_RANGE, V2_DSD_RANGE = (970.0, 1010.0), (1050.0, 1070.0)   # L157, L190
V1_DSO_RANGE, V2_DSO_JITTER = (745.0, 785.0), 3.0                # L158, L190
V1_PRIMARY, V1_SECONDARY = (18.0, 42.0), (-8.0, 8.0)             # L161-162
V2_PRIMARY, V2_SECONDARY = (-8.0, 8.0), (18.0, 42.0)             # L193-194
MOTION_ROT_DEG, MOTION_TRANS_MM = 10.0, 8.0                      # L191, L196
DEEPCA_ACCURACY = 1.0                                            # L137 (we use 0.5, see P2-l)
CROP_MM_DEFAULT = 96.0
ISO_MM_CAP = 0.35            # never coarser than this; --iso auto goes finer as needed
MIN_COMPONENT_FRAC = 0.05
_CONN26 = np.ones((3, 3, 3), int)


# ======================= geometry / projection =======================
def sample_geometry(rng, motion_3d=False):
    """One DeepCA-style two-view acquisition, returned as TWO geometries.

    views_render   what TIGRE projects; view 2 carries the rigid motion.
    views_scanner  nominal C-arm geometry (same det spacing, same DSD/DSO,
                   motion removed) -- this is what the model is given.

    DeepCA fidelity: view 1 zero motion (L159/164); view 2 render offOrigin
    [tx,ty,0] (L191) and angles [pri+r1, sec+r2, 0] (L196); view 2 nominal
    offOrigin [0,0,0] (L211) and angles [pri, sec, 0] (L212), DSD/DSO kept.
    """
    det_sp = rng.uniform(*DET_SPACING_RANGE)
    dso1 = rng.uniform(*V1_DSO_RANGE)

    rot = rng.uniform(-MOTION_ROT_DEG, MOTION_ROT_DEG, size=2)
    trans = np.zeros(3, np.float64)
    trans[0] = rng.uniform(-MOTION_TRANS_MM, MOTION_TRANS_MM)
    trans[1] = rng.uniform(-MOTION_TRANS_MM, MOTION_TRANS_MM)
    if motion_3d:                       # NON-DeepCA
        trans[2] = rng.uniform(-MOTION_TRANS_MM, MOTION_TRANS_MM)

    v1 = dict(alpha=float(rng.uniform(*V1_PRIMARY)), beta=float(rng.uniform(*V1_SECONDARY)),
              DSD=float(rng.uniform(*V1_DSD_RANGE)), DSO=float(dso1),
              det_spacing=float(det_sp), offOrigin=np.zeros(3, np.float32))

    a_nom = float(rng.uniform(*V2_PRIMARY))
    b_nom = float(rng.uniform(*V2_SECONDARY))
    dsd2 = float(rng.uniform(*V2_DSD_RANGE))
    dso2 = float(dso1 + rng.uniform(-V2_DSO_JITTER, V2_DSO_JITTER))

    v2_render = dict(alpha=a_nom + float(rot[0]), beta=b_nom + float(rot[1]),
                     DSD=dsd2, DSO=dso2, det_spacing=float(det_sp),
                     offOrigin=np.array(trans, np.float32))
    v2_scanner = dict(alpha=a_nom, beta=b_nom, DSD=dsd2, DSO=dso2,
                      det_spacing=float(det_sp), offOrigin=np.zeros(3, np.float32))

    meta = dict(motion_rot_deg=rot.tolist(), motion_trans_mm=trans.tolist(),
                motion_3d=bool(motion_3d),
                motion_trans_norm_mm=float(np.linalg.norm(trans)),
                motion_rot_norm_deg=float(np.linalg.norm(rot)),
                v2_nominal_alpha=a_nom, v2_nominal_beta=b_nom,
                # Raw angles only. Clinical RAO/LAO is deliberately NOT derived
                # here -- see the module docstring.
                deepca_ref="data_simulation.py@c01ab96 L159/164 L191/196 L211/212")
    return [v1, v2_render], [dict(v1), v2_scanner], meta


def build_geo(shape, spacing, view, accuracy=0.5):
    geo = tigre.geometry(mode="cone", nVoxel=np.array(shape), default=True)
    geo.dVoxel = np.array(spacing, dtype=np.float32)
    geo.sVoxel = geo.dVoxel * geo.nVoxel
    geo.DSO, geo.DSD = view["DSO"], view["DSD"]
    geo.nDetector = np.array([DET_N, DET_N])
    geo.dDetector = np.array([view["det_spacing"]] * 2, dtype=np.float32)
    geo.sDetector = geo.dDetector * geo.nDetector
    geo.offOrigin = np.array(view["offOrigin"], dtype=np.float32)
    # P2-l: set explicitly rather than inheriting TIGRE's default. Units are
    # vx/sample, so SMALLER is finer. DeepCA uses 1.0; we default to 0.5.
    geo.accuracy = float(accuracy)
    return geo


def _blob_centre(proj):
    """Sub-pixel marker location: intensity-weighted centroid of the blob.
    Returns None when the marker is not visible (caller skips it).

    np.argmax returns the FIRST index of a flat-topped plateau -- its top-left
    corner -- a systematic multi-pixel bias.  That was the v1 pose error
    (DLT RMS 2.7-3.4 px vs 0.14-1.04 px with the centroid).
    """
    peak = proj.max()
    if not np.isfinite(peak) or peak <= 0:
        return None
    m = proj >= 0.5 * peak
    row, col = center_of_mass(proj * m)
    if not (np.isfinite(row) and np.isfinite(col)):
        row, col = np.unravel_index(np.argmax(proj), proj.shape)
    return float(col), float(row)


def project_points(P, pts_mm):
    """mm (relative to the volume centre) -> (col, row) pixel coordinates."""
    h = np.hstack([pts_mm, np.ones((len(pts_mm), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def calibrate_P(shape, spacing, view, n_fit=20, n_val=8, cal_n=64, seed=0, accuracy=0.5):
    """DLT against TIGRE's real Ax(), validated on HELD-OUT markers.

    The calibration phantom has the SAME physical extent (sVoxel) as the volume
    being projected, so P transfers exactly; only the sampling is coarser.

    P is fitted on `n_fit` markers; `n_val` further markers are drawn AFTER the
    fit (de-duplicated against it) and projected through the fitted P, so
    rms_val is a generalisation error rather than a training residual.

    rms_val is NaN when fewer than 3 validation markers were visible.  Callers
    MUST treat NaN as a failure -- see P1-i; do not funnel it through max().
    """
    sVoxel = np.array(shape) * np.array(spacing)
    cal_shape = (cal_n, cal_n, cal_n)
    cal_spacing = sVoxel / cal_n
    geo = build_geo(cal_shape, cal_spacing, view, accuracy=accuracy)
    angles = np.array([[np.radians(view["alpha"]), np.radians(view["beta"]), 0.0]], np.float32)

    rng = np.random.default_rng(seed)
    centre = np.array(cal_shape) / 2.0
    used = set()

    def _draw(n):
        out = []
        for _ in range(n):
            for _try in range(5):
                idx = rng.integers(6, np.array(cal_shape) - 6)
                key = tuple(int(v) for v in idx)
                if key not in used:
                    break
            used.add(key)
            vol = np.zeros(cal_shape, dtype=np.float32)
            vol[tuple(idx)] = 1.0
            centre_px = _blob_centre(tigre.Ax(vol, geo, angles)[0])
            if centre_px is None:
                continue                      # marker not visible -> skip it
            x, y = centre_px
            out.append(((idx + 0.5 - centre) * cal_spacing, x, y))
        return out

    fit = _draw(n_fit)
    if len(fit) < 6:
        raise RuntimeError("DLT calibration failed: too few visible fit markers")

    A = []
    for (X, x, y) in fit:
        Xh = np.array([*X, 1.0])
        A.append(np.concatenate([Xh, np.zeros(4), -x * Xh]))
        A.append(np.concatenate([np.zeros(4), Xh, -y * Xh]))
    _, _, Vt = np.linalg.svd(np.array(A))
    P = Vt[-1].reshape(3, 4)
    P = (P / P[-1, -1]).astype(np.float64)

    def _rms(corr):
        if len(corr) < 3:
            return float("nan")
        pts = np.array([c[0] for c in corr])
        obs = np.array([[c[1], c[2]] for c in corr])
        return float(np.sqrt(((project_points(P, pts) - obs) ** 2).sum(1).mean()))

    return P, _rms(fit), _rms(_draw(n_val))


def on_detector_fraction(uv):
    """Fraction of projected points inside the physical detector (P2-f)."""
    col, row = uv[:, 0], uv[:, 1]
    return float(((col >= 0) & (col < DET_N) & (row >= 0) & (row < DET_N)).mean())


def silhouette_clipped(binary):
    """P2-m: True when the rendered vessel touches a detector border, i.e. the
    silhouette is cut off even if every centerline point is on-detector."""
    return bool(binary[0].any() or binary[-1].any()
                or binary[:, 0].any() or binary[:, -1].any())


# ======================= vessel split / crop / resample =======================
def side_labels(affine, centroids, spacing):
    """Name exactly two components by anatomical side from the NIfTI axis codes.
    Returns (names, separation_mm), or (None, nan) if the affine has no L/R axis.
    Never called with one component -- side is not identifiable from one centroid.
    """
    try:
        codes = nib.aff2axcodes(affine)
    except Exception:
        return None, float("nan")
    ax = next((i for i, c in enumerate(codes) if c in ("L", "R")), None)
    if ax is None or len(centroids) != 2:
        return None, float("nan")
    vals = [c[ax] for c in centroids]
    sep_mm = abs(vals[0] - vals[1]) * float(spacing[ax])
    order = np.argsort(vals)
    # codes[ax]=="L": increasing index -> more Left, so LARGEST value is LCA.
    # codes[ax]=="R": increasing index -> more Right, so SMALLEST value is LCA.
    ranked = order[::-1] if codes[ax] == "L" else order
    names = [None] * len(vals)
    for rank, ci in enumerate(ranked):
        names[ci] = "LCA" if rank == 0 else "RCA"
    return names, sep_mm


def split_components(mask, affine, spacing, single_component="skip",
                     extra_components="skip", min_lr_sep_mm=10.0):
    """One sample per coronary system.  Returns (components, skip_reason).

    2 significant components -> LCA/RCA by affine L-R axis, if separated
    1 component              -> merged tree; side not identifiable -> skip
    >2 components            -> keeping the 2 largest discards anatomy -> skip
    """
    lab, n = cc_label(mask, structure=_CONN26)
    if n == 0:
        return [], "no components"
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    sig = [i for i in range(1, n + 1) if sizes[i] >= MIN_COMPONENT_FRAC * mask.sum()]
    sig = sorted(sig, key=lambda i: -sizes[i])
    if not sig:
        return [], "no significant components"

    if len(sig) == 1:
        if single_component != "keep":
            return [], "1 significant component (LCA/RCA merged); side not identifiable"
        i = sig[0]
        return [dict(name="MERGED", mask=(lab == i), n_vox=int(sizes[i]),
                     lr_sep_mm=float("nan"))], None

    if len(sig) > 2:
        frac3 = sizes[sig[2]] / mask.sum()
        if extra_components != "largest2":
            return [], (f"{len(sig)} significant components "
                        f"(3rd = {frac3:.1%} of mask) -- would discard anatomy")
        sig = sig[:2]

    keep = sig[:2]
    centroids = [np.array(center_of_mass(lab == i)) for i in keep]
    names, sep_mm = side_labels(affine, centroids, spacing)
    if names is None:
        return [], "no L/R axis in NIfTI affine; cannot name LCA/RCA"
    if not np.isfinite(sep_mm) or sep_mm < min_lr_sep_mm:
        return [], f"L-R centroid separation {sep_mm:.1f} mm < {min_lr_sep_mm} mm (ambiguous side)"
    return [dict(name=names[j], mask=(lab == i), n_vox=int(sizes[i]),
                 lr_sep_mm=float(sep_mm)) for j, i in enumerate(keep)], None


def crop_box(mask, shape, spacing, crop_mm):
    """crop_mm box on the vessel's bounding-box centre; shifted, never shrunk."""
    shape = np.array(shape)
    idx = np.argwhere(mask)
    centre = (idx.min(0) + idx.max(0)) / 2.0
    half = np.ceil((crop_mm / 2.0) / np.array(spacing)).astype(int)
    size = np.minimum(2 * half, shape)
    lo = np.round(centre).astype(int) - size // 2
    lo = np.clip(lo, 0, np.maximum(shape - size, 0))
    return lo, lo + size


def resolve_iso(spacing, iso_arg, iso_cap=ISO_MM_CAP):
    """P1-h: pick the isotropic grid for ONE case.

    'auto' -> min(iso_cap, native spacing.min()), with NO lower clamp, so zoom
    factors are >= 1 on every axis and `to_isotropic` really is pure upsampling.

    An earlier draft clamped this at a 0.25 mm floor, which silently voided the
    guarantee for any case finer than the floor (native [0.20,0.20,0.25] still
    gave factors [0.8,0.8,1.0]).  Grid size is bounded by
    rejecting oversized volumes (--max_iso_voxels) instead of by quietly
    degrading them: a build that cannot afford the grid should say so, not
    corrupt the mask.

    A fixed --iso value is honoured as given and may downsample; to_isotropic()
    reports which axes, and the QC gate fails on any of them.
    """
    if isinstance(iso_arg, str) and iso_arg.lower() == "auto":
        return float(min(iso_cap, float(np.min(spacing))))
    return float(iso_arg)


def to_isotropic(mask_crop, spacing, iso):
    """Nearest-neighbour resample to an isotropic grid.

    Returns (mask, downsampled_axes).  When iso <= min(spacing) every zoom
    factor is >= 1 and this is pure upsampling, so no thin vessel can be lost.
    That precondition is NOT automatic -- 508/1000 ImageCAS cases have in-plane
    spacing below 0.35 mm -- so the violated axes are returned and the caller
    gates on the resulting topology (P1-h).
    """
    factors = np.array(spacing, float) / float(iso)
    down = [int(a) for a in np.where(factors < 1.0)[0]]
    out = zoom(mask_crop.astype(np.uint8), factors, order=0,
               grid_mode=True, mode="nearest")
    return out.astype(bool), down


# ======================= centerline GT =======================
def centerline_from_iso(iso_mask, iso):
    """Skeleton + radius(mm) + topology + DFS order on the isotropic grid."""
    idx = np.argwhere(iso_mask)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, np.array(iso_mask.shape))
    sub = iso_mask[tuple(slice(l, h) for l, h in zip(lo, hi))]

    skel = skeletonize(sub)
    if skel.sum() < 20:
        return None
    dist_mm = distance_transform_edt(sub, sampling=(iso, iso, iso))   # radius in mm
    sub_coords = np.argwhere(skel)
    radii = dist_mm[skel]
    topo = _classify_topology(skel)
    order = _traversal_order(sub_coords, radii)
    n_comp = int(cc_label(skel, structure=_CONN26)[1])
    coords = (sub_coords + lo)[order]
    return coords, radii[order], topo[order], n_comp


def adjacency_fraction(coords):
    """QC: fraction of consecutive ordered pairs that are 26-connected."""
    if len(coords) < 2:
        return 1.0
    return float((np.abs(np.diff(coords, axis=0)).max(1) <= 1).mean())


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="./dataset_v3_1",
                    help="P1-j: versioned by default so v3.0/v3.1 output cannot mix")
    ap.add_argument("--overwrite", action="store_true",
                    help="P1-j: delete existing .npz/report files in out_dir before building")
    ap.add_argument("--n", type=int, default=10, help="number of PATIENTS")
    ap.add_argument("--crop_mm", type=float, default=CROP_MM_DEFAULT)
    ap.add_argument("--iso", default="auto",
                    help="P1-h: 'auto' = min(%.2f, native spacing.min()) per case "
                         "(guarantees upsampling); or a fixed value in mm, which may "
                         "DOWNsample -- not recommended" % ISO_MM_CAP)
    ap.add_argument("--max_iso_voxels", type=float, default=80e6,
                    help="P1-h: reject a sample whose isotropic grid exceeds this many "
                         "voxels, rather than silently coarsening it (~0.22 mm over a "
                         "96 mm cube). Raise it if you have the GPU memory.")
    ap.add_argument("--max_points", type=int, default=4000)
    ap.add_argument("--tol_px", type=float, default=2.0)
    ap.add_argument("--tigre_accuracy", type=float, default=0.5,
                    help="P2-l: TIGRE ray sampling, vx/sample (SMALLER is finer). "
                         "DeepCA uses %.1f; we default to 0.5 -- see module docstring."
                         % DEEPCA_ACCURACY)
    ap.add_argument("--min_consistency", type=float, default=0.95)
    ap.add_argument("--min_adjacency", type=float, default=0.95,
                    help="P2-p: ordering-adjacency threshold. DIAGNOSTIC by default; "
                         "pass --reject_low_adjacency to make it reject.")
    ap.add_argument("--reject_low_adjacency", action="store_true",
                    help="P2-p: reject samples below --min_adjacency. OFF by default: "
                         "a DFS walk must jump when it backtracks to a branch point, so "
                         "adjacency FALLS as a tree gets more branched. Rejecting on it "
                         "discards the most branched trees -- a dataset bias, not a data "
                         "defect (the GT is valid, just harder to serialise). At 200 "
                         "patients this was the SOLE reason for 19 of 88 rejections.")
    ap.add_argument("--min_mask_kept", type=float, default=0.95)
    ap.add_argument("--min_on_detector", type=float, default=0.95,
                    help="P2-f: minimum PER-VIEW fraction of centerline points inside "
                         "the 512^2 detector. Was 0.99, which was arbitrary: DeepCA's "
                         "FOV at isocentre is 99-115 mm while a 96 mm crop spans "
                         "121-136 mm once rotated, so partial per-view loss is expected.")
    ap.add_argument("--min_coverage", type=float, default=1.0,
                    help="P1-o: minimum fraction of GT points on-detector in AT LEAST "
                         "ONE view. A point invisible in both is unlearnable, so this "
                         "is the criterion that actually matters; keep it at 1.0.")
    ap.add_argument("--reject_clipped", action="store_true",
                    help="P2-m: reject when the silhouette touches a detector border. "
                         "OFF by default -- clinical angiography clips vessels at the "
                         "frame edge routinely and DeepCA's geometry makes it "
                         "unavoidable for a 96 mm crop, so it is a diagnostic.")
    ap.add_argument("--max_dlt_rms", type=float, default=1.0,
                    help="P2-e/P1-i: max HELD-OUT DLT rms in px; NaN always rejects")
    ap.add_argument("--require_single_skeleton", dest="require_single_skeleton",
                    action="store_true", default=True,
                    help="P1-h: reject when the skeleton is not one component (default on)")
    ap.add_argument("--allow_split_skeleton", dest="require_single_skeleton",
                    action="store_false")
    ap.add_argument("--motion_probe_trans_mm", type=float, default=4.0,
                    help="P2-n: above this sampled |translation|, a zero observed "
                         "motion effect is a wiring bug and rejects the sample")
    ap.add_argument("--motion_probe_rot_deg", type=float, default=5.0)
    ap.add_argument("--min_motion_effect", type=float, default=0.02,
                    help="P2-n: DIAGNOSTIC threshold; only enforced per-sample for "
                         "large-motion draws (see --motion_probe_*)")
    ap.add_argument("--min_lr_sep_mm", type=float, default=10.0)
    ap.add_argument("--single_component", choices=["skip", "keep"], default="skip")
    ap.add_argument("--extra_components", choices=["skip", "largest2"], default="skip")
    ap.add_argument("--dlt_fit", type=int, default=20)
    ap.add_argument("--dlt_val", type=int, default=8)
    ap.add_argument("--accept_yield", action="store_true",
                    help="P2-q: proceed when INTEGRITY passes but YIELD is over its "
                         "limit -- an explicit, recorded 'I read the breakdown and this "
                         "discard rate is acceptable'. Prefer this over raising "
                         "--max_reject_frac, which hides the decision.")
    ap.add_argument("--max_reject_frac", type=float, default=0.15,
                    help="Was 0.02, chosen with no data behind it. The 20-patient pilot "
                         "put the genuinely-broken rate at ~10%% (vessels the 96 mm crop "
                         "severs), so 0.02 could never be met.")
    ap.add_argument("--max_skip_frac", type=float, default=0.10)
    ap.add_argument("--motion_3d", action="store_true",
                    help="NON-DeepCA: also perturb out-of-plane translation (DeepCA L191 = 0)")
    ap.add_argument("--motion_2d", action="store_true",
                    help="DEPRECATED no-op: 2-axis motion is the default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- P1-j: refuse to build into a directory that already holds samples ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.npz"))
    if existing:
        if not args.overwrite:
            raise SystemExit(
                f"REFUSING TO RUN: {out} already contains {len(existing)} .npz file(s).\n"
                f"  Samples skipped or rejected by this build would survive from the\n"
                f"  earlier run -- possibly with incompatible pose semantics (pre-v3.1\n"
                f"  builds baked the simulated motion into `poses`).\n"
                f"  Re-run with --overwrite to clear them, or choose a fresh --out_dir.")
        for f in existing:
            f.unlink()
        for stale in ("pilot_report_v3.json", "norm_stats_v3.json", "case_splits_v3.json"):
            (out / stale).unlink(missing_ok=True)
        print(f"--overwrite: removed {len(existing)} stale .npz from {out}\n")
    if args.motion_2d:
        print("NOTE: --motion_2d is now the default and is ignored. "
              "Use --motion_3d for the non-DeepCA 3-axis variant.\n")

    # ---- patient-level 80/10/10 split, before anything else (splits.py) ----
    all_ids = sorted(int(os.path.basename(f).split(".")[0])
                     for f in glob.glob(os.path.join(args.raw_dir, "*.label.nii.gz")))
    splits = make_case_level_split(all_ids, val_frac=0.10, test_frac=0.10, seed=args.seed)
    write_splits(splits, out / "case_splits_v3.json")
    split_of = {c: s for s in ("train", "val", "test") for c in splits[s]}
    print(f"builder v{BUILDER_VERSION} | patient split over {len(all_ids)}: "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
          f"(both vessels share a split)")
    print(f"motion: {'3-axis (NON-DeepCA)' if args.motion_3d else '2-axis (DeepCA L191)'} | "
          f"poses = NOMINAL scanner geometry, poses_render = motion")
    print(f"iso: {args.iso} (max {args.max_iso_voxels/1e6:.0f}M voxels) | tigre accuracy "
          f"{args.tigre_accuracy} vx/sample (DeepCA {DEEPCA_ACCURACY})\n")

    rng = np.random.default_rng(args.seed)
    rows, times, failures, rejected, skipped, done = [], [], [], [], [], 0
    import collections
    reject_cat, reject_sole = collections.Counter(), collections.Counter()
    n_comp_hist = []
    for cid in all_ids:
        if done >= args.n:
            break
        lp = os.path.join(args.raw_dir, f"{cid}.label.nii.gz")
        if not os.path.exists(lp):
            continue
        t0 = time.time()
        nii = nib.load(lp)
        mask_full = np.asarray(nii.get_fdata()) > 0.5
        spacing = np.array(nii.header.get_zooms()[:3], float)
        if mask_full.sum() < 100:
            continue
        done += 1
        iso = resolve_iso(spacing, args.iso)

        _lab, _n = cc_label(mask_full, structure=_CONN26)
        _sz = np.bincount(_lab.ravel()); _sz[0] = 0
        n_comp_hist.append(int((_sz >= MIN_COMPONENT_FRAC * mask_full.sum()).sum()))

        comps, skip_why = split_components(
            mask_full, nii.affine, spacing,
            single_component=args.single_component,
            extra_components=args.extra_components,
            min_lr_sep_mm=args.min_lr_sep_mm)
        if skip_why:
            skipped.append((str(cid), skip_why))
            print(f"{cid:>12} [{split_of[cid]:>5}]: SKIPPED -- {skip_why}", flush=True)
            times.append(time.time() - t0)
            continue

        for comp in comps:
            sid = f"{cid}_{comp['name']}"
            lo, hi = crop_box(comp["mask"], mask_full.shape, spacing, args.crop_mm)
            cmask = comp["mask"][tuple(slice(l, h) for l, h in zip(lo, hi))]
            mask_kept = float(cmask.sum()) / float(comp["mask"].sum())
            if cmask.sum() < 50:
                failures.append((sid, "empty crop")); continue

            # P1-h: component count BEFORE the resample, so a topology break can
            # be attributed to the resample rather than the crop.
            n_comp_precrop = int(cc_label(cmask, structure=_CONN26)[1])
            iso_mask, down_axes = to_isotropic(cmask, spacing, iso)
            n_comp_postiso = int(cc_label(iso_mask, structure=_CONN26)[1])
            iso_shape = np.array(iso_mask.shape)
            iso_sp = np.array([iso] * 3)
            sVoxel = iso_shape * iso_sp

            if int(np.prod(iso_shape)) > args.max_iso_voxels:
                rejected.append((sid, f"isotropic grid {iso_shape.tolist()} = "
                                      f"{np.prod(iso_shape)/1e6:.1f}M voxels > "
                                      f"--max_iso_voxels {args.max_iso_voxels/1e6:.0f}M "
                                      f"at iso {iso:.3f} mm"))
                print(f"{sid:>12} [{split_of[cid]:>5}]: REJECTED -- grid too large "
                      f"({np.prod(iso_shape)/1e6:.1f}M voxels)", flush=True)
                continue

            cl = centerline_from_iso(iso_mask, iso)
            if cl is None:
                failures.append((sid, "skeleton too small")); continue
            coords, radii_mm, topo, n_comp = cl
            cl_mm = (coords + 0.5 - iso_shape / 2.0) * iso_sp
            adj = adjacency_fraction(coords)

            r_ok = bool(np.isfinite(radii_mm).all() and (radii_mm > 0).all())
            inside = float(iso_mask[coords[:, 0], coords[:, 1], coords[:, 2]].mean())
            n_end = int((topo == LABEL_ENDPOINT).sum())

            gt = np.concatenate([cl_mm, radii_mm[:, None], topo[:, None]], 1).astype(np.float32)
            try:
                padded, pmask = pad_centerline(gt, args.max_points)
            except ValueError:
                failures.append((sid, f"{len(gt)} points > --max_points {args.max_points}"))
                continue

            views_r, views_s, motion = sample_geometry(rng, motion_3d=args.motion_3d)
            images, P_scan, P_rend = [], [], []
            cons_r, cons_s, ond_r, ond_s = [], [], [], []
            rms_fit_all, rms_val_all, clipped = [], [], []
            inb_r, inb_s = [], []      # per-point on-detector masks, per view
            for k in range(2):
                vr, vs = views_r[k], views_s[k]
                geo = build_geo(iso_shape, iso_sp, vr, accuracy=args.tigre_accuracy)
                angles = np.array([[np.radians(vr["alpha"]), np.radians(vr["beta"]), 0.0]],
                                  np.float32)
                binary = tigre.Ax(iso_mask.astype(np.float32), geo, angles)[0] > 1e-6
                images.append(binary)
                clipped.append(silhouette_clipped(binary))          # P2-m

                Pr, fr, vr_rms = calibrate_P(iso_shape, iso_sp, vr, n_fit=args.dlt_fit,
                                             n_val=args.dlt_val, seed=args.seed + 10 * k,
                                             accuracy=args.tigre_accuracy)
                if k == 0:
                    Ps, fs, vs_rms = Pr, fr, vr_rms             # view 1 carries no motion
                else:
                    Ps, fs, vs_rms = calibrate_P(iso_shape, iso_sp, vs, n_fit=args.dlt_fit,
                                                 n_val=args.dlt_val, seed=args.seed + 97,
                                                 accuracy=args.tigre_accuracy)
                P_rend.append(Pr); P_scan.append(Ps)
                # P1-i: keep EVERY value flat. Never collapse with max() -- that
                # silently drops a NaN whenever a finite value comes first.
                rms_fit_all += [fr, fs]
                rms_val_all += [vr_rms, vs_rms]

                tolm = (binary_dilation(binary, iterations=int(args.tol_px))
                        if args.tol_px > 0 else binary)

                def _score(P):
                    uv = project_points(P, cl_mm)
                    col = np.round(uv[:, 0]).astype(int); row = np.round(uv[:, 1]).astype(int)
                    ib = (col >= 0) & (col < DET_N) & (row >= 0) & (row < DET_N)
                    hit = np.zeros(len(uv), bool)
                    hit[ib] = tolm[row[ib], col[ib]]
                    return float(hit.mean()), on_detector_fraction(uv), ib

                c_r, o_r, ib_r = _score(Pr); c_s, o_s, ib_s = _score(Ps)
                cons_r.append(c_r); ond_r.append(o_r); inb_r.append(ib_r)
                cons_s.append(c_s); ond_s.append(o_s); inb_s.append(ib_s)

            # P1-o: the disqualifying condition is a GT point with no support in
            # EITHER view -- no model can place it.  A point missing from one view
            # is still constrained by the other, and real angiography clips at the
            # frame edge routinely, so per-view clipping is realistic input, not a
            # defect (see P2-m note below).
            cover_r = float((inb_r[0] | inb_r[1]).mean())
            cover_s = float((inb_s[0] | inb_s[1]).mean())

            motion_effect = cons_r[1] - cons_s[1]
            rms_val_arr = np.asarray(rms_val_all, float)
            big_motion = (motion["motion_trans_norm_mm"] >= args.motion_probe_trans_mm
                          or motion["motion_rot_norm_deg"] >= args.motion_probe_rot_deg)

            # ---- PER-SAMPLE QC: reject before saving, never on a mean ----
            bad = []
            if min(cons_r) < args.min_consistency:
                bad.append(f"consistency {min(cons_r):.3f} < {args.min_consistency}")
            if args.reject_low_adjacency and adj < args.min_adjacency:   # P2-p
                bad.append(f"adjacency {adj:.3f} < {args.min_adjacency}")
            if not r_ok:
                bad.append("radius non-finite or <= 0")
            # P1-i: NaN can never pass this form of the check.
            if rms_val_arr.size == 0 or not np.isfinite(rms_val_arr).all():
                bad.append("held-out DLT rms is NaN (too few visible validation markers)")
            elif rms_val_arr.max() >= args.max_dlt_rms:
                bad.append(f"held-out DLT rms {rms_val_arr.max():.2f}px >= {args.max_dlt_rms}")
            if min(ond_r + ond_s) < args.min_on_detector:
                bad.append(f"on-detector {min(ond_r + ond_s):.3f} < {args.min_on_detector}")
            if min(cover_r, cover_s) < args.min_coverage:            # P1-o
                bad.append(f"union coverage {min(cover_r, cover_s):.3f} < "
                           f"{args.min_coverage} (points invisible in BOTH views)")
            if args.reject_clipped and any(clipped):                 # P2-m, opt-in
                bad.append(f"silhouette clipped at detector border (views "
                           f"{[i for i, c in enumerate(clipped) if c]})")
            if mask_kept < args.min_mask_kept:
                bad.append(f"mask_kept {mask_kept:.3f} < {args.min_mask_kept}")
            # P1-h: the mask is one component by construction, so its skeleton
            # must be too; anything else means the crop or the resample broke it.
            if args.require_single_skeleton and n_comp != 1:
                bad.append(f"skeleton has {n_comp} components "
                           f"(mask pre-resample {n_comp_precrop}, post {n_comp_postiso}"
                           f"{', DOWNSAMPLED axes ' + str(down_axes) if down_axes else ''})")
            # P2-n: a small motion draw producing a small effect is CORRECT and
            # must not be rejected; only a large draw with no effect is a bug.
            if big_motion and motion_effect < args.min_motion_effect:
                bad.append(f"motion effect {motion_effect:.3f} < {args.min_motion_effect} "
                           f"despite |t|={motion['motion_trans_norm_mm']:.1f}mm "
                           f"|r|={motion['motion_rot_norm_deg']:.1f}deg (wiring bug)")
            if bad:
                for _b in bad:                                   # P2-q: category tally
                    key = ("mask_kept" if "mask_kept" in _b else
                           "skeleton_split" if "skeleton has" in _b else
                           "adjacency" if "adjacency" in _b else
                           "union_coverage" if "union coverage" in _b else
                           "on_detector" if "on-detector" in _b else
                           "consistency" if "consistency" in _b else
                           "dlt_rms" if "DLT rms" in _b else
                           "grid_too_large" if "voxels" in _b else
                           "motion_wiring" if "motion effect" in _b else
                           "clipped" if "clipped" in _b else "other")
                    reject_cat[key] += 1
                    if len(bad) == 1:
                        reject_sole[key] += 1
                rejected.append((sid, "; ".join(bad)))
                print(f"{sid:>12} [{split_of[cid]:>5}]: REJECTED -- {'; '.join(bad)}", flush=True)
                continue

            np.savez_compressed(
                out / f"{sid}.npz",
                builder_version=BUILDER_VERSION,     # P1-j: assert on this in the loader
                images=np.stack(images).astype(np.uint8),
                # MODEL INPUT: nominal scanner geometry, motion NOT encoded.
                poses=np.stack(P_scan).astype(np.float32),
                # VALIDATION ONLY: carries the simulated motion.
                poses_render=np.stack(P_rend).astype(np.float32),
                centerline=padded, centerline_mask=pmask,   # RAW mm: x,y,z,radius,topology
                n_points=len(gt), patient_id=cid, vessel=comp["name"], split=split_of[cid],
                iso_mm=np.float32(iso), iso_shape=iso_shape.astype(np.int32),
                sVoxel=sVoxel.astype(np.float32), crop_lo=lo.astype(np.int32),
                native_spacing=spacing.astype(np.float32),
                mask_kept=np.float32(mask_kept),
                tigre_accuracy=np.float32(args.tigre_accuracy),
                geometry_render=json.dumps([{a: (b.tolist() if isinstance(b, np.ndarray) else b)
                                             for a, b in vw.items()} for vw in views_r]),
                geometry_scanner=json.dumps([{a: (b.tolist() if isinstance(b, np.ndarray) else b)
                                              for a, b in vw.items()} for vw in views_s]),
                motion=json.dumps(motion),
            )
            rows.append(dict(sample=sid, patient=cid, vessel=comp["name"], split=split_of[cid],
                             consistency=cons_r, consistency_scanner=cons_s,
                             motion_effect=round(motion_effect, 4),
                             motion_trans_mm=round(motion["motion_trans_norm_mm"], 2),
                             motion_rot_deg=round(motion["motion_rot_norm_deg"], 2),
                             coverage_render=round(cover_r, 4),
                             coverage_scanner=round(cover_s, 4),
                             clipped_views=[i for i, c in enumerate(clipped) if c],
                             on_det_render=[round(v, 4) for v in ond_r],
                             on_det_scanner=[round(v, 4) for v in ond_s],
                             dlt_rms_fit=[round(r, 3) for r in rms_fit_all],
                             dlt_rms_val=[round(r, 3) for r in rms_val_all],
                             n_points=int(len(gt)), adjacency=round(adj, 3), radius_ok=r_ok,
                             radius_mm=[round(float(radii_mm.min()), 2),
                                        round(float(radii_mm.max()), 2)],
                             skeleton_inside=round(inside, 4), n_endpoints=n_end,
                             n_skel_components=n_comp, n_mask_comp_precrop=n_comp_precrop,
                             n_mask_comp_postiso=n_comp_postiso,
                             iso_mm=round(iso, 4), downsampled_axes=down_axes,
                             iso_voxels=int(np.prod(iso_shape)),
                             mask_kept=round(mask_kept, 3),
                             lr_sep_mm=round(comp["lr_sep_mm"], 1)))
            print(f"{sid:>12} [{split_of[cid]:>5}]: cons {cons_r[0]:.3f}/{cons_r[1]:.3f} | "
                  f"scan {cons_s[0]:.3f}/{cons_s[1]:.3f} (dm {motion_effect:+.3f}) | "
                  f"rms_val {rms_val_arr.max():.2f}px | onDet {min(ond_r + ond_s):.3f} | "
                  f"iso {iso:.3f} ({iso_shape[0]}^3-ish) | pts {len(gt)} | adj {adj:.3f} | "
                  f"kept {mask_kept:.3f} | comps {n_comp}", flush=True)
        times.append(time.time() - t0)

    if not rows:
        print("No samples produced.")
        for s, why in skipped:
            print(f"  SKIPPED {s}: {why}")
        raise SystemExit(1)                                       # P2-k

    # ---- normalisation stats from TRAIN patients only (preprocessing.py) ----
    tr = [r for r in rows if r["split"] == "train"]
    if tr:
        mm_centerlines = {}
        for r in tr:
            z = np.load(out / f"{r['sample']}.npz")
            mm_centerlines[r["sample"]] = z["centerline"][z["centerline_mask"]]
        stats = compute_centerline_norm_stats(mm_centerlines, list(mm_centerlines))
        stats.update(n_train_samples=len(tr), builder_version=BUILDER_VERSION,
                     iso=args.iso, crop_mm=args.crop_mm, max_points=args.max_points)
        json.dump(stats, open(out / "norm_stats_v3.json", "w"), indent=2)
        print(f"\nnorm stats (train only, n={len(tr)}): "
              f"coord_std {np.round(stats['coord_std'], 2)} "
              f"radius {stats['radius_mean']:.2f}+/-{stats['radius_std']:.2f} mm")

    allc = np.array([c for r in rows for c in r["consistency"]])
    allcs = np.array([c for r in rows for c in r["consistency_scanner"]])
    rmsv = np.array([v for r in rows for v in r["dlt_rms_val"]], float)
    rmsf = np.array([v for r in rows for v in r["dlt_rms_fit"]], float)
    ond = np.array([v for r in rows for v in (r["on_det_render"] + r["on_det_scanner"])])
    adjall = np.array([r["adjacency"] for r in rows])
    npts = np.array([r["n_points"] for r in rows])
    kept = np.array([r["mask_kept"] for r in rows])
    dm = np.array([r["motion_effect"] for r in rows])
    mag = np.array([r["motion_trans_mm"] for r in rows])
    isos = np.array([r["iso_mm"] for r in rows])
    print("\n" + "=" * 76)
    print(f"builder v{BUILDER_VERSION} | patients {done} | samples {len(rows)} "
          f"({len(rows)/max(done,1):.1f} vessels/patient)")
    print(f"T3 consistency   mean {allc.mean():.3f}  min {allc.min():.3f}   [render pose, tol {args.tol_px}px]")
    print(f"   under scanner mean {allcs.mean():.3f}  min {allcs.min():.3f}   [motion NOT compensated -- the task]")
    print(f"T2 DLT rms       fit mean {rmsf.mean():.2f}px | HELD-OUT mean {rmsv.mean():.2f}px  max {rmsv.max():.2f}px")
    print(f"T1 skel inside   min {min(r['skeleton_inside'] for r in rows):.4f}")
    cov = np.array([r["coverage_render"] for r in rows] + [r["coverage_scanner"] for r in rows])
    nclip = sum(1 for r in rows if r["clipped_views"])
    print(f"on-detector      mean {ond.mean():.4f}  min {ond.min():.4f}  (per view, gate {args.min_on_detector})")
    print(f"union coverage   mean {cov.mean():.4f}  min {cov.min():.4f}  (>=1 view; gate {args.min_coverage})")
    print(f"silhouette clip  {nclip}/{len(rows)} kept samples touch a detector border "
          f"({'REJECTING' if args.reject_clipped else 'diagnostic only'})")
    print(f"ordering adj     mean {adjall.mean():.3f}  min {adjall.min():.3f}")
    print(f"radius sane      {sum(r['radius_ok'] for r in rows)}/{len(rows)} samples")
    print(f"points/vessel    mean {npts.mean():.0f}  max {npts.max()}  (max_points {args.max_points})")
    print(f"mask kept        mean {kept.mean():.3f}  min {kept.min():.3f}  (gate {args.min_mask_kept})")
    print(f"iso grid         {isos.min():.3f}-{isos.max():.3f} mm | max voxels "
          f"{max(r['iso_voxels'] for r in rows):,}")
    n_down = sum(1 for r in rows if r["downsampled_axes"])
    print(f"downsampled axes {n_down}/{len(rows)} samples "
          f"{'(EXPECT 0 with --iso auto)' if n_down else 'OK'}")
    print(f"skeleton comps   all == 1: {all(r['n_skel_components'] == 1 for r in rows)}")

    # ---- P2-n: motion is a DIAGNOSTIC, reported not silently filtered ----
    corr = float(np.corrcoef(mag, dm)[0, 1]) if len(rows) > 2 and mag.std() > 0 else float("nan")
    print(f"motion effect    mean {dm.mean():+.3f}  min {dm.min():+.3f}  max {dm.max():+.3f}")
    print(f"   |t| vs effect corr {corr:+.2f}   (expect clearly positive; "
          f"low-motion draws SHOULD show little effect)")
    if n_comp_hist:
        _h = np.array(n_comp_hist)
        print(f"vessel comps/pt  mean {_h.mean():.2f}  max {_h.max()}  "
              f"| ==1: {(_h == 1).sum()}  ==2: {(_h == 2).sum()}  >2: {(_h > 2).sum()}")
    for s, why in failures:
        print(f"  FAILURE {s}: {why}")
    per = float(np.mean(times))
    print(f"per-patient {per:.1f}s  ->  ETA 1000 patients: {per*1000/3600:.1f} h")

    n_attempt = len(rows) + len(rejected)
    reject_frac = len(rejected) / max(n_attempt, 1)
    skip_frac = len(skipped) / max(done, 1)
    print(f"rejected         {len(rejected)}/{n_attempt} samples "
          f"({reject_frac:.1%}; limit {args.max_reject_frac:.1%})")
    for s_, why in rejected:
        print(f"  REJECTED {s_}: {why}")
    print(f"skipped          {len(skipped)}/{done} patients "
          f"({skip_frac:.1%}; limit {args.max_skip_frac:.1%})")
    for s_, why in skipped:
        print(f"  SKIPPED {s_}: {why}")

    # Every SAVED sample already passed per-sample QC, so these minima are
    # guarantees. The gate checks the aggregate story: too much thrown away, or
    # motion that is not reaching the render at all.
    if reject_cat:
        print("\nrejection breakdown (reasons overlap; 'sole' = only reason):")
        for k, v in reject_cat.most_common():
            print(f"    {k:16s} {v:4d}   sole {reject_sole[k]:4d}")
        nl = sum(1 for s_, _ in rejected if "_LCA" in s_)
        print(f"    by vessel: LCA {nl}  RCA {len(rejected)-nl}"
              "   (a persistent LCA excess is the DeepCA FOV limit, not a bug)")

    # P2-q: the gate is split in two.
    #   INTEGRITY = things that can only be wrong if the BUILDER is wrong.
    #   YIELD     = how much real data the QC threw away.  Those rejections are
    #               individually justified (incomplete GT, vessels past the
    #               detector FOV), so this limit is a STEP-CHANGE DETECTOR, not a
    #               quality bar; raising it to chase a green light is meaningless.
    motion_alive = bool(dm.mean() >= args.min_motion_effect
                        and (np.isnan(corr) or corr > 0.0))
    integrity = {
        "held-out DLT rms finite and < %.2f px" % args.max_dlt_rms:
            bool(np.isfinite(rmsv).all() and rmsv.max() < args.max_dlt_rms),
        "no sample was downsampled":          n_down == 0,
        "every skeleton is one component":    all(r["n_skel_components"] == 1 for r in rows),
        "every GT point visible in >=1 view": bool(cov.min() >= args.min_coverage),
        "all radii finite and > 0":           all(r["radius_ok"] for r in rows),
        "motion reaches the render":          motion_alive,
        "no hard build failures":             not failures,
    }
    yield_ok = {
        "rejected <= %.0f%%" % (100*args.max_reject_frac): reject_frac <= args.max_reject_frac,
        "skipped  <= %.0f%%" % (100*args.max_skip_frac):   skip_frac <= args.max_skip_frac,
    }
    print("\nINTEGRITY (a failure here means the builder is wrong):")
    for k, v in integrity.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print("YIELD (how much real data the QC discarded):")
    for k, v in yield_ok.items():
        print(f"    [{'PASS' if v else 'over'}] {k}")
    ok = all(integrity.values()) and (all(yield_ok.values()) or args.accept_yield)
    if not motion_alive:
        print("  GATE NOTE: motion effect is ~zero across the whole build -- "
              "the render is probably not receiving the perturbation.")
    json.dump(rows, open(out / "pilot_report_v3.json", "w"), indent=2)
    print("saved pilot_report_v3.json + case_splits_v3.json + norm_stats_v3.json + per-sample npz")
    if all(integrity.values()) and not all(yield_ok.values()) and args.accept_yield:
        print("\nGATE PASSED with --accept_yield: integrity clean, yield over limit and\n"
              "explicitly accepted by the operator. The discard rate is recorded above.")
    elif all(integrity.values()) and not all(yield_ok.values()):
        print("\nGATE FAILED on YIELD ONLY — every integrity check passed, so the data\n"
              "that WAS written is sound. Read the breakdown above and decide whether the\n"
              "discarded fraction is acceptable; do not simply raise the limit.")
    else:
        print(("GATE PASSED — dataset v%s validated." % BUILDER_VERSION) if ok else
              "GATE FAILED on INTEGRITY — the builder is wrong; do not train.")
    print("=" * 76)
    if not ok:
        raise SystemExit(1)                                       # P2-k


if __name__ == "__main__":
    main()
