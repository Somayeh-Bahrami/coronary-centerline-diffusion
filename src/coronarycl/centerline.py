"""Step 1.1 — centerline + radius extraction from ImageCAS segmentations.
Runs locally on MacBook M4 (CPU-only, no GPU needed).

Classical skeletonization directly on ImageCAS's expert-annotated
segmentation mask — deterministic, no learned model needed since
ground-truth segmentation is already given.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import skeletonize

# Branch/topology label values (5th column of the output array).
LABEL_ENDPOINT = 0     # 1 skeleton neighbor  -- tip of a vessel branch
LABEL_REGULAR = 1      # 2 skeleton neighbors -- ordinary point along a branch
LABEL_BIFURCATION = 2  # 3+ skeleton neighbors -- branch point (vessel splits)

_NEIGHBOR_KERNEL = np.ones((3, 3, 3))
_NEIGHBOR_KERNEL[1, 1, 1] = 0  # don't count the voxel itself
# 26-connected offsets for the path-traversal below (separate from
# _NEIGHBOR_KERNEL, which is only used for the topology neighbor-count).
_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def _build_adjacency(coords: np.ndarray) -> list:
    """26-connected adjacency list over skeleton voxels, index-aligned
    with `coords` (adjacency[i] = indices j such that coords[i] and
    coords[j] are 26-connected neighbors)."""
    index_of = {tuple(c): i for i, c in enumerate(coords.astype(np.int64))}
    adjacency = [[] for _ in range(len(coords))]
    for i, c in enumerate(coords.astype(np.int64)):
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = index_of.get((c[0] + dx, c[1] + dy, c[2] + dz))
            if j is not None:
                adjacency[i].append(j)
    return adjacency


def _traversal_order(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Reorder skeleton points via DFS traversal over 26-connectivity,
    replacing np.argwhere()'s raster-scan order.

    Rationale: the 1D-UNet denoiser (Conv1d, strided pool/upsample) assumes
    array-adjacent points are spatially/topologically adjacent. Raster
    order violates this. Measured on a synthetic single-bifurcation test
    case: raster order has only 53.4% of consecutive pairs 26-connected
    (mean graph-geodesic hop distance 9.7, up to 37 hops between
    "consecutive" indices). DFS-preorder from a radius-max endpoint raises
    this to 98.6%, with residual jumps only at true bifurcation points
    (irreducible for any 1D serialization of a branching tree).

    Handles disconnected skeleton components (segmentation/skeletonization
    artifacts) by traversing each separately, largest first. Root of each
    component is the largest-radius endpoint (proxy for the proximal/
    ostium end, since coronary radius decreases distally); falls back to
    the largest-radius point overall if the component has no endpoint
    (e.g. a closed-loop artifact).

    Returns:
        order: (N,) int64 permutation such that coords[order],
        radii[order], branch_labels[order] are path-ordered.
    """
    n = len(coords)
    adjacency = _build_adjacency(coords)
    degree = np.array([len(nbrs) for nbrs in adjacency])

    unvisited = set(range(n))
    components = []
    while unvisited:
        start = next(iter(unvisited))
        stack, seen = [start], {start}
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adjacency[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(comp)
        unvisited -= seen
    components.sort(key=len, reverse=True)

    order = []
    for comp in components:
        endpoints = [i for i in comp if degree[i] <= 1]
        candidates = endpoints if endpoints else comp
        root = max(candidates, key=lambda i: radii[i])

        stack, seen = [root], {root}
        comp_order = []
        while stack:
            node = stack.pop()
            comp_order.append(node)
            for nb in sorted(adjacency[node]):  # sorted -> deterministic
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        order.extend(comp_order)

    order = np.array(order, dtype=np.int64)
    assert len(order) == n and len(set(order.tolist())) == n, \
        "traversal order lost or duplicated points -- bug"
    return order


def _classify_topology(skeleton: np.ndarray) -> np.ndarray:
    """For each skeleton voxel, count its 26-connected skeleton neighbors
    and classify it as an endpoint, regular point, or bifurcation.

    Returns an array of labels aligned with np.argwhere(skeleton) order.
    """
    neighbor_count = convolve(skeleton.astype(np.uint8), _NEIGHBOR_KERNEL,
                              mode="constant", cval=0)
    counts_at_skeleton = neighbor_count[skeleton]

    labels = np.full(counts_at_skeleton.shape, LABEL_REGULAR, dtype=np.int64)
    labels[counts_at_skeleton <= 1] = LABEL_ENDPOINT
    labels[counts_at_skeleton >= 3] = LABEL_BIFURCATION
    return labels


def extract_centerline(volume: np.ndarray, spacing=None) -> np.ndarray:
    """Skeletonize `volume` and return (N, 5): x, y, z (voxel index),
    radius, topology label -- path-ordered by `_traversal_order`.

    `spacing` is the (sx, sy, sz) voxel size in mm. When given, the
    distance transform is computed with physical sampling, so the radius
    column is in **mm**. When omitted the radius is in **voxels**, which
    is only correct for isotropic data -- see the note below.

    NOTE (bug fixed 2026-09): earlier revisions called
    distance_transform_edt(volume) with no `sampling`, producing radii in
    voxel units (recognisable as exact square roots of integers) that were
    later scaled by mean(spacing) in preprocessing.voxel_to_mm. That
    isotropic approximation is wrong for anisotropic ImageCAS data
    (typically 0.377 x 0.377 x 0.5 mm). Pass `spacing` to get true mm and
    do NOT rescale afterwards.
    """
    skeleton = skeletonize(volume)
    dist = distance_transform_edt(volume, sampling=spacing)
    coords = np.argwhere(skeleton)
    radii = dist[skeleton]
    branch_labels = _classify_topology(skeleton)

    order = _traversal_order(coords, radii)
    coords, radii, branch_labels = (
        coords[order], radii[order], branch_labels[order])

    return np.concatenate(
        [coords, radii[:, None], branch_labels[:, None]], axis=1
    )


def extract_case(label_path: Path, use_mm_radius: bool = True) -> np.ndarray:
    """Load one ImageCAS `<case>.label.nii.gz` segmentation and extract
    its centerline.

    With `use_mm_radius=True` (default) the radius column is in mm, taken
    from the NIfTI header's voxel spacing. Pass False only to reproduce
    the legacy voxel-radius output of the v1 dataset.
    """
    nii = nib.load(label_path)
    seg = nii.get_fdata() > 0.5
    spacing = nii.header.get_zooms()[:3] if use_mm_radius else None
    return extract_centerline(seg, spacing=spacing)


def extract_all(raw_dir: Path, out_dir: Path, case_ids=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(raw_dir.glob("*.label.nii.gz"))
    if case_ids:
        case_ids = set(case_ids)
        label_files = [f for f in label_files
                       if int(f.stem.split('.')[0]) in case_ids]

    for f in label_files:
        centerline = extract_case(f)
        out_path = out_dir / f"{f.stem.split('.')[0]}_centerline.npy"
        np.save(out_path, centerline)
        print(
            f"{f.stem}: {centerline.shape[0]} centerline points -> {out_path}")
