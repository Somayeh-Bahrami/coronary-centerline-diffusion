#!/usr/bin/env python3
"""Multi-seed DDIM evaluation in millimetres.

Sampler settings and checkpoint choice are tuned on VAL only. The canonical
sampler applies per-sample physical crop bounds to xyz and TRAIN-only observed
bounds to radius at every DDIM step. Curve metrics use a tree recovered from
GT skeleton neighbours and are explicitly labelled given/oracle-topology.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.coronarycl.dataset_v3_1 import (  # noqa: E402
    CoronaryCenterlineDatasetV31, list_samples)
from src.coronarycl.metrics import (  # noqa: E402
    crop_bounds_mm, evaluate_case, topology_tree_edges)
from src.coronarycl.models.diffusion import CenterlineDenoiser  # noqa: E402
from src.coronarycl.sampling import sample_ddim  # noqa: E402
from src.coronarycl.trainer import NoiseScheduler  # noqa: E402

DEFAULT_SEEDS = [104729, 130363, 155921]
METRIC_KEYS = [
    "chamfer_l2",
    "hd95_mm",
    "overlap@1.0mm",
    "overlap@2.0mm",
    "overlap@5.0mm",
    "edge_continuity_5x",
    "broken_edge_fraction_5x",
    "largest_connected_component_fraction_5x",
    "pred_edge_length_median_mm",
    "pred_edge_length_p95_mm",
    "pred_tree_length_mm",
    "gt_tree_length_mm",
    "tree_length_ratio",
    "out_of_crop_fraction",
    "mean_crop_excess_mm",
    "max_crop_excess_mm",
    "radius_mae_mm",
    "radius_out_of_train_range_fraction",
]


def comma_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("val", "test", "train"), default="val")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", type=comma_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke/debug only")
    parser.add_argument("--out", type=Path, default=Path("ddim_results.json"))
    parser.add_argument("--save-pred", type=Path)
    parser.add_argument(
        "--bounds", choices=("physical", "none"), default="physical")
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--null-cond", action="store_true")
    parser.add_argument("--mean-shape", action="store_true")
    parser.add_argument("--allow-test", action="store_true",
                        help="required for the one locked final TEST run")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_seed(base_seed, sample_id):
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def initial_noise(items, n_points, base_seed, device):
    """Per-sample noise independent of batching and sample ordering."""
    noise = torch.zeros(len(items), n_points, 4, device=device)
    for index, item in enumerate(items):
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(stable_sample_seed(base_seed, item["sample"]))
        count = int(item["n_points"])
        noise[index, :count] = torch.randn(
            count, 4, generator=generator, device=device)
    return noise


def training_radius_bounds_mm(data_dir):
    """Observed valid radius range from TRAIN only (never VAL/TEST)."""
    minimum = float("inf")
    maximum = float("-inf")
    count = 0
    for sample_id in list_samples(data_dir, "train"):
        with np.load(Path(data_dir) / f"{sample_id}.npz") as archive:
            mask = archive["centerline_mask"].astype(bool)
            radii = archive["centerline"][mask, 3].astype(np.float64)
        if len(radii):
            minimum = min(minimum, float(radii.min()))
            maximum = max(maximum, float(radii.max()))
            count += len(radii)
    if count == 0 or not 0 < minimum < maximum:
        raise RuntimeError("Invalid TRAIN-only radius range")
    return minimum, maximum


def normalized_physical_bounds(items, dataset, radius_bounds, device):
    lower = np.empty((len(items), 1, 4), dtype=np.float32)
    upper = np.empty_like(lower)
    for index, item in enumerate(items):
        xyz_lower, xyz_upper = crop_bounds_mm(
            item["sVoxel"].numpy(), item["iso_mm"])
        lower[index, 0, :3] = (
            xyz_lower - dataset.coord_mean) / dataset.coord_std
        upper[index, 0, :3] = (
            xyz_upper - dataset.coord_mean) / dataset.coord_std
        lower[index, 0, 3] = (
            radius_bounds[0] - dataset.radius_mean) / dataset.radius_std
        upper[index, 0, 3] = (
            radius_bounds[1] - dataset.radius_mean) / dataset.radius_std
    return (torch.from_numpy(lower).to(device),
            torch.from_numpy(upper).to(device))


def resample_arclength(points, count):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] <= 0:
        return np.repeat(points[:1], count, axis=0)
    target = np.linspace(0, cumulative[-1], count)
    return np.stack([
        np.interp(target, cumulative, points[:, axis])
        for axis in range(3)], axis=1)


def mean_shape_templates(data_dir, count=256):
    accumulated = defaultdict(list)
    for sample_id in list_samples(data_dir, "train"):
        with np.load(Path(data_dir) / f"{sample_id}.npz", allow_pickle=True) as z:
            mask = z["centerline_mask"].astype(bool)
            xyz = z["centerline"][mask, :3].astype(np.float64)
            vessel = str(z["vessel"])
        accumulated[vessel].append(resample_arclength(xyz, count))
    return {
        vessel: np.mean(np.stack(curves), axis=0)
        for vessel, curves in accumulated.items()}


def _metric_row(
    item, seed, pred_mm, gt_mm, radius_bounds, edges, lower, upper,
):
    pred_xyz, gt_xyz = pred_mm[:, :3], gt_mm[:, :3]
    metrics = evaluate_case(
        pred_xyz, gt_xyz, thresholds=(1.0, 2.0, 5.0),
        edges=edges, xyz_lower_mm=lower, xyz_upper_mm=upper)
    pred_radius = pred_mm[:, 3]
    metrics.update({
        "radius_mae_mm": float(np.mean(
            np.abs(pred_radius - gt_mm[:, 3]))),
        "radius_out_of_train_range_fraction": float(np.mean(
            (pred_radius < radius_bounds[0])
            | (pred_radius > radius_bounds[1]))),
    })
    return {
        "sample": item["sample"],
        "patient": str(item["patient_id"]),
        "vessel": item["vessel"],
        "seed": int(seed),
        "n_points": int(item["n_points"]),
        "n_topology_edges": int(len(edges)),
        "n_topology_components": int(metrics.pop("n_topology_components")),
        "out_of_crop_any": bool(metrics.pop("out_of_crop_any")),
        **metrics,
    }, edges, lower, upper


def aggregate_rows(rows, bootstrap_repeats):
    """Average seeds per vessel sample, then summarize all/LCA/RCA."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    sample_rows = []
    for sample_id, seed_rows in sorted(grouped.items()):
        first = seed_rows[0]
        sample_row = {
            "sample": sample_id,
            "patient": first["patient"],
            "vessel": first["vessel"],
            "n_seeds": len(seed_rows),
            "n_points": first["n_points"],
            "n_topology_edges": first["n_topology_edges"],
            "n_topology_components": first["n_topology_components"],
            "out_of_crop_any_rate": float(np.mean([
                row["out_of_crop_any"] for row in seed_rows])),
        }
        for key in METRIC_KEYS:
            sample_row[key] = float(np.mean([row[key] for row in seed_rows]))
        sample_rows.append(sample_row)

    groups = {"all": sample_rows}
    for vessel in ("LCA", "RCA"):
        groups[vessel] = [
            row for row in sample_rows if row["vessel"] == vessel]
    summary = {}
    for group_name, group_rows in groups.items():
        group_summary = {"n_samples": len(group_rows)}
        for key in METRIC_KEYS:
            values = np.asarray([row[key] for row in group_rows], dtype=float)
            finite = values[np.isfinite(values)]
            group_summary[key] = (
                float(finite.mean()) if len(finite) else float("nan"))
            group_summary[f"{key}_std"] = (
                float(finite.std()) if len(finite) else float("nan"))
            group_summary[f"{key}_median"] = (
                float(np.median(finite)) if len(finite) else float("nan"))
        summary[group_name] = group_summary

    # Patient bootstrap for the headline all-sample metrics. Vessels are first
    # averaged within patient, then patients receive equal weight.
    patient_rows = defaultdict(list)
    for row in sample_rows:
        patient_rows[row["patient"]].append(row)
    patients = sorted(patient_rows)
    rng = np.random.default_rng(301691)
    headline = [
        "chamfer_l2", "hd95_mm", "overlap@2.0mm",
        "edge_continuity_5x", "largest_connected_component_fraction_5x",
        "tree_length_ratio", "out_of_crop_fraction"]
    patient_ci = {}
    for key in headline:
        per_patient = np.asarray([
            np.mean([row[key] for row in patient_rows[patient]])
            for patient in patients], dtype=float)
        bootstrap = np.empty(bootstrap_repeats, dtype=float)
        for index in range(bootstrap_repeats):
            selected = rng.integers(0, len(per_patient), len(per_patient))
            bootstrap[index] = per_patient[selected].mean()
        patient_ci[key] = {
            "equal_patient_mean": float(per_patient.mean()),
            "bootstrap_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975))],
        }
    return sample_rows, summary, patient_ci


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _autocast(device, precision):
    if torch.device(device).type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _print_summary(summary):
    print(
        f"{'group':<6}{'n':>5}{'Chamfer':>10}{'HD95':>9}{'Ot2':>9}"
        f"{'continuity':>13}{'LCC':>9}{'length':>9}{'OOB':>9}")
    print("-" * 79)
    for name in ("all", "LCA", "RCA"):
        row = summary[name]
        print(
            f"{name:<6}{row['n_samples']:>5}"
            f"{row['chamfer_l2']:>10.2f}{row['hd95_mm']:>9.2f}"
            f"{row['overlap@2.0mm']:>9.3f}"
            f"{row['edge_continuity_5x']:>13.3f}"
            f"{row['largest_connected_component_fraction_5x']:>9.3f}"
            f"{row['tree_length_ratio']:>9.3f}"
            f"{row['out_of_crop_fraction']:>9.3f}")


def main():
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.ckpt:
        args.ckpt = args.ckpt.expanduser().resolve()
    if args.save_pred:
        args.save_pred = args.save_pred.expanduser().resolve()
    if not args.data.is_dir():
        raise FileNotFoundError(args.data)
    if args.batch <= 0 or args.bootstrap <= 0:
        raise ValueError("batch and bootstrap must be positive")
    seeds = list(args.seeds)
    if len(seeds) != len(set(seeds)) or len(seeds) == 0:
        raise ValueError("sampling seeds must be non-empty and unique")
    if args.split == "test" and not args.allow_test:
        raise PermissionError(
            "TEST is locked. Tune checkpoint/steps/guidance/bounds on VAL; "
            "pass --allow-test only once after the full protocol is frozen.")
    for output in (args.out, args.save_pred):
        if output and output.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output} exists; refusing to overwrite it")
    if not args.mean_shape and args.ckpt is None:
        raise ValueError("--ckpt is required unless --mean-shape is used")
    if args.mean_shape and args.null_cond:
        raise ValueError("mean-shape and null-conditioning modes conflict")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "bf16" and device == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support BF16")
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    radius_bounds = training_radius_bounds_mm(args.data)
    sample_ids = list_samples(args.data, args.split)
    if args.limit:
        sample_ids = sample_ids[:args.limit]
    dataset = CoronaryCenterlineDatasetV31(
        args.data, sample_ids=sample_ids, return_render_poses=False)
    cached_items = [dataset[index] for index in range(len(dataset))]
    order = sorted(
        range(len(sample_ids)),
        key=lambda index: cached_items[index]["n_points"])
    metric_context = {}
    for item in cached_items:
        count = int(item["n_points"])
        gt_mm = dataset.denormalize(
            item["centerline"].numpy())[:count, :4]
        edges = topology_tree_edges(gt_mm[:, :3], item["iso_mm"])
        if len(edges) != count - 1:
            raise RuntimeError(
                f"{item['sample']}: GT neighbour graph has {len(edges)} "
                f"tree edges for {count} nodes; expected one connected component")
        lower, upper = crop_bounds_mm(
            item["sVoxel"].numpy(), item["iso_mm"])
        metric_context[item["sample"]] = (gt_mm, edges, lower, upper)
    print(
        f"split={args.split}; samples={len(sample_ids)}; seeds={seeds}; "
        f"radius TRAIN range={radius_bounds[0]:.3f}..{radius_bounds[1]:.3f} mm")

    model = checkpoint = scheduler = None
    checkpoint_metadata = None
    if args.mean_shape:
        templates = mean_shape_templates(args.data)
    else:
        checkpoint = torch.load(
            args.ckpt, map_location="cpu", weights_only=True)
        model = CenterlineDenoiser(
            hidden_dim=int(checkpoint["hidden_dim"])).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        scheduler = NoiseScheduler(n_steps=1000, device=device)
        checkpoint_metadata = {
            "path": str(args.ckpt.expanduser().resolve()),
            "sha256": file_sha256(args.ckpt),
            "hidden_dim": int(checkpoint["hidden_dim"]),
            "step": int(checkpoint["step"]),
            "val_loss": checkpoint.get("val_loss"),
        }

    rows = []
    prediction_archive = {}
    for seed in ([seeds[0]] if args.mean_shape else seeds):
        completed = 0
        for start in range(0, len(order), args.batch):
            indices = order[start:start + args.batch]
            items = [cached_items[index] for index in indices]
            counts = [int(item["n_points"]) for item in items]
            padded_count = max(counts)
            padded_count += (-padded_count) % 4

            if args.mean_shape:
                predictions_mm = []
                for item, count in zip(items, counts):
                    xyz = resample_arclength(
                        templates[item["vessel"]], count)
                    radius = np.full(
                        (count, 1), dataset.radius_mean, dtype=np.float64)
                    predictions_mm.append(np.concatenate([xyz, radius], axis=1))
            else:
                images = torch.stack([item["images"] for item in items]).to(device)
                poses = torch.stack([item["poses"] for item in items]).to(device)
                if args.null_cond:
                    images = torch.zeros_like(images)
                    poses = torch.zeros_like(poses)
                mask = torch.zeros(
                    len(items), padded_count, dtype=torch.bool, device=device)
                for index, count in enumerate(counts):
                    mask[index, :count] = True
                noise = initial_noise(
                    items, padded_count, seed, device)
                lower = upper = None
                if args.bounds == "physical":
                    lower, upper = normalized_physical_bounds(
                        items, dataset, radius_bounds, device)
                with _autocast(device, args.precision):
                    normalized = sample_ddim(
                        model, scheduler, images, poses, mask, device,
                        initial_noise=noise, n_steps=args.steps,
                        guidance_scale=args.guidance,
                        x0_min=lower, x0_max=upper)
                denormalized = dataset.denormalize(
                    normalized.float().cpu().numpy())
                predictions_mm = [
                    denormalized[index, :count]
                    for index, count in enumerate(counts)]

            for item, count, pred_mm in zip(items, counts, predictions_mm):
                gt_mm, edges, lower_mm, upper_mm = metric_context[item["sample"]]
                row, edges, lower_mm, upper_mm = _metric_row(
                    item, seed, pred_mm, gt_mm, radius_bounds,
                    edges, lower_mm, upper_mm)
                rows.append(row)
                if args.save_pred:
                    sample_id = item["sample"]
                    prediction_archive[
                        f"pred__{seed}__{sample_id}"] = pred_mm.astype(np.float32)
                    prediction_archive.setdefault(
                        f"gt__{sample_id}", gt_mm.astype(np.float32))
                    prediction_archive.setdefault(
                        f"edges__{sample_id}", edges.astype(np.int32))
                    prediction_archive.setdefault(
                        f"bounds__{sample_id}",
                        np.stack([lower_mm, upper_mm]).astype(np.float32))
            completed += len(items)
            print(
                f"  seed {seed}: {completed}/{len(sample_ids)}", flush=True)

    sample_rows, summary, patient_ci = aggregate_rows(
        rows, args.bootstrap)
    _print_summary(summary)
    mode = (
        "mean_shape_control" if args.mean_shape else
        "null_condition_control" if args.null_cond else
        "conditional")
    payload = {
        "mode": mode,
        "checkpoint": checkpoint_metadata,
        "protocol": {
            "split": args.split,
            "ddim_steps": None if args.mean_shape else args.steps,
            "guidance": None if args.mean_shape else args.guidance,
            "seeds": ([seeds[0]] if args.mean_shape else seeds),
            "bounds": args.bounds,
            "radius_bounds_mm_from_train": list(radius_bounds),
            "precision": args.precision,
            "n_points_source": "ground truth",
            "topology_metrics": "given/oracle GT topology",
            "chamfer_convention": (
                "mean(pred->gt)+mean(gt->pred), Euclidean, unsquared"),
            "aggregation": (
                "seeds averaged per vessel sample; sample means stratified "
                "by vessel; headline CI gives equal patient weight"),
        },
        "summary": summary,
        "patient_bootstrap": patient_ci,
        "per_sample": sample_rows,
        "per_sample_seed": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    write_csv(args.out.with_suffix(".csv"), sample_rows)
    write_csv(args.out.with_name(args.out.stem + "_per_seed.csv"), rows)
    print(f"wrote {args.out}")
    if args.save_pred:
        args.save_pred.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_pred, **prediction_archive)
        print(f"wrote {args.save_pred}")


if __name__ == "__main__":
    main()
