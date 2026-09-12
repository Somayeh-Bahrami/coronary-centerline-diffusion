#!/usr/bin/env python3
"""Paired, patient-aware conditioning sensitivity on VAL only.

For every fixed evaluation seed, each target receives a donor from the same
vessel (LCA/RCA) but a different patient. Matched and shuffled passes share
the exact noisy centerline. Loss is reduced per sample, then vessels are
averaged within patient so each patient has equal weight. This avoids the old
singleton-batch, batch-weighting, and same-patient donor defects.

Run ``--shuffle-mode joint`` for full-conditioning sensitivity (donor images
and poses move together), and ``--shuffle-mode images`` to isolate image use
while retaining the target pose. These are diagnostics, not geometry metrics.
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

DEFAULT_SEEDS = [104729, 130363, 155921, 181081, 205019]
DIAGNOSTIC_TIMESTEPS = [0, 100, 200, 250, 400, 500, 600, 750, 800, 999]
PRIMARY_TIMESTEPS = [200, 250, 400, 500, 600, 750, 800]


def comma_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--ckpt", action="append", required=True, metavar="LABEL=PATH",
        help="repeat for each checkpoint, e.g. h384=/path/best.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seeds", type=comma_ints, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--timesteps", type=comma_ints, default=DIAGNOSTIC_TIMESTEPS)
    parser.add_argument(
        "--primary-timesteps", type=comma_ints,
        default=PRIMARY_TIMESTEPS)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--shuffle-mode", choices=("joint", "images"), default="joint")
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--limit", type=int, default=0,
                        help="balanced smoke-test sample limit")
    return parser.parse_args()


def parse_checkpoints(specifications):
    parsed = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected LABEL=PATH, got {specification!r}")
        label, raw_path = specification.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or label in parsed:
            raise ValueError(f"Invalid or duplicate label: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed[label] = path
    return parsed


def sample_parts(sample_id):
    patient, vessel = sample_id.rsplit("_", 1)
    if vessel not in {"LCA", "RCA"}:
        raise ValueError(f"Unexpected sample id: {sample_id}")
    return patient, vessel


def build_donor_map(sample_ids, seed):
    """Same-vessel cyclic derangement with a different-patient donor."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[sample_parts(sample_id)[1]].append(index)
    donors = [-1] * len(sample_ids)
    for vessel, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError(f"Need at least two {vessel} samples")
        cycle = np.asarray(indices, dtype=int)
        rng.shuffle(cycle)
        donor_cycle = np.roll(cycle, -1)
        for target, donor in zip(cycle.tolist(), donor_cycle.tolist()):
            target_patient, target_vessel = sample_parts(sample_ids[target])
            donor_patient, donor_vessel = sample_parts(sample_ids[donor])
            if target_patient == donor_patient or target_vessel != donor_vessel:
                raise RuntimeError("invalid conditioning donor map")
            donors[target] = donor
    if any(index < 0 for index in donors):
        raise RuntimeError("incomplete donor map")
    return donors


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def masked_mse_per_sample(prediction, target, mask):
    per_point = (prediction.float() - target.float()).square().mean(dim=-1)
    weights = mask.to(per_point.dtype)
    return ((per_point * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0))


def _autocast(device, precision):
    if torch.device(device).type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def evaluate_checkpoint(
    *, label, checkpoint_path, items, sample_ids, donor_maps, seeds,
    timesteps, batch_size, shuffle_mode, precision, device,
    model_class, scheduler_class,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True)
    hidden_dim = int(checkpoint["hidden_dim"])
    model = model_class(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    scheduler = scheduler_class(n_steps=1000, device=device)

    rows = []
    for seed in seeds:
        print(f"  {label}: evaluation seed {seed}", flush=True)
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(seed)
        donors = donor_maps[seed]
        for start in range(0, len(items), batch_size):
            targets = list(range(start, min(start + batch_size, len(items))))
            donor_indices = [donors[index] for index in targets]
            centerline = torch.stack(
                [items[i]["centerline"][..., :4] for i in targets]).to(device)
            mask = torch.stack(
                [items[i]["centerline_mask"] for i in targets]).to(device)
            images = torch.stack(
                [items[i]["images"] for i in targets]).to(device)
            poses = torch.stack(
                [items[i]["poses"] for i in targets]).to(device)
            donor_images = torch.stack(
                [items[i]["images"] for i in donor_indices]).to(device)
            donor_poses = torch.stack(
                [items[i]["poses"] for i in donor_indices]).to(device)
            shuffled_poses = donor_poses if shuffle_mode == "joint" else poses

            for timestep in timesteps:
                timestep_batch = torch.full(
                    (len(targets),), timestep,
                    dtype=torch.long, device=device)
                noise = torch.randn(
                    centerline.shape, generator=generator,
                    device=device, dtype=centerline.dtype)
                alpha_bar = scheduler.alpha_bars[timestep_batch].view(-1, 1, 1)
                noisy = (torch.sqrt(alpha_bar) * centerline
                         + torch.sqrt(1.0 - alpha_bar) * noise)
                with _autocast(device, precision):
                    matched_prediction = model(
                        noisy, timestep_batch, images, poses,
                        x0_self=None, node_mask=mask)
                    shuffled_prediction = model(
                        noisy, timestep_batch, donor_images, shuffled_poses,
                        x0_self=None, node_mask=mask)
                matched = masked_mse_per_sample(
                    matched_prediction, noise, mask).cpu().numpy()
                shuffled = masked_mse_per_sample(
                    shuffled_prediction, noise, mask).cpu().numpy()
                for local_index, target_index in enumerate(targets):
                    sample_id = sample_ids[target_index]
                    patient, vessel = sample_parts(sample_id)
                    rows.append({
                        "model": label,
                        "shuffle_mode": shuffle_mode,
                        "seed": seed,
                        "timestep": timestep,
                        "sample": sample_id,
                        "patient": patient,
                        "vessel": vessel,
                        "donor": sample_ids[donor_indices[local_index]],
                        "matched_mse": float(matched[local_index]),
                        "shuffled_mse": float(shuffled[local_index]),
                        "delta_mse": float(
                            shuffled[local_index] - matched[local_index]),
                    })

    metadata = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "hidden_dim": hidden_dim,
        "step": int(checkpoint["step"]),
        "val_loss": checkpoint.get("val_loss"),
    }
    del model, scheduler, checkpoint
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return rows, metadata


def patient_pairs(rows, label, timesteps, seed=None):
    grouped = defaultdict(list)
    for row in rows:
        if row["model"] != label or row["timestep"] not in timesteps:
            continue
        if seed is not None and row["seed"] != seed:
            continue
        grouped[row["patient"]].append(
            (row["matched_mse"], row["shuffled_mse"]))
    patients = sorted(grouped)
    matched = np.asarray([
        np.mean([pair[0] for pair in grouped[patient]])
        for patient in patients])
    shuffled = np.asarray([
        np.mean([pair[1] for pair in grouped[patient]])
        for patient in patients])
    return patients, matched, shuffled


def ratio_sensitivity(matched, shuffled):
    return float(100.0 * (shuffled.mean() / matched.mean() - 1.0))


def summarize(rows, labels, seeds, timesteps, primary_timesteps,
              bootstrap_repeats):
    primary = set(primary_timesteps)
    result = {"models": {}}
    patient_arrays = {}
    for label in labels:
        patients, matched, shuffled = patient_pairs(rows, label, primary)
        patient_arrays[label] = (patients, matched, shuffled)
        point = ratio_sensitivity(matched, shuffled)
        rng = np.random.default_rng(99173)
        bootstrap = np.empty(bootstrap_repeats, dtype=float)
        for index in range(bootstrap_repeats):
            selected = rng.integers(0, len(patients), len(patients))
            bootstrap[index] = ratio_sensitivity(
                matched[selected], shuffled[selected])
        per_seed = {}
        for seed in seeds:
            _, seed_matched, seed_shuffled = patient_pairs(
                rows, label, primary, seed=seed)
            per_seed[str(seed)] = ratio_sensitivity(
                seed_matched, seed_shuffled)
        per_timestep = {}
        for timestep in timesteps:
            _, matched_t, shuffled_t = patient_pairs(
                rows, label, {timestep})
            per_timestep[str(timestep)] = {
                "matched_mse": float(matched_t.mean()),
                "shuffled_mse": float(shuffled_t.mean()),
                "sensitivity_pct": ratio_sensitivity(
                    matched_t, shuffled_t),
            }
        result["models"][label] = {
            "n_patients": len(patients),
            "primary_sensitivity_pct": point,
            "patient_bootstrap_ci95_pct": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "absolute_delta_mse": float(shuffled.mean() - matched.mean()),
            "per_seed_sensitivity_pct": per_seed,
            "per_timestep": per_timestep,
        }
    if len(labels) == 2:
        first, second = labels
        p1, m1, s1 = patient_arrays[first]
        p2, m2, s2 = patient_arrays[second]
        if p1 != p2:
            raise RuntimeError("Models do not share validation patients")
        rng = np.random.default_rng(117191)
        bootstrap = np.empty(bootstrap_repeats, dtype=float)
        for index in range(bootstrap_repeats):
            selected = rng.integers(0, len(p1), len(p1))
            bootstrap[index] = (
                ratio_sensitivity(m1[selected], s1[selected])
                - ratio_sensitivity(m2[selected], s2[selected]))
        result["paired_model_difference"] = {
            "definition": f"{first} minus {second}, percentage points",
            "estimate": ratio_sensitivity(m1, s1) - ratio_sensitivity(m2, s2),
            "patient_bootstrap_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not repo.is_dir() or not data.is_dir():
        raise FileNotFoundError(f"repo={repo}, data={data}")
    sys.path.insert(0, str(repo))

    from src.coronarycl.dataset_v3_1 import (  # noqa: PLC0415
        CoronaryCenterlineDatasetV31, list_samples)
    from src.coronarycl.models.diffusion import (  # noqa: PLC0415
        CenterlineDenoiser)
    from src.coronarycl.trainer import NoiseScheduler  # noqa: PLC0415

    checkpoints = parse_checkpoints(args.ckpt)
    seeds = list(args.seeds)
    timesteps = sorted(set(args.timesteps))
    primary_timesteps = sorted(set(args.primary_timesteps))
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be distinct")
    if not args.limit and not 3 <= len(seeds) <= 5:
        raise ValueError("Use 3 to 5 fixed seeds for a full VAL run")
    if args.batch < 1 or args.bootstrap < 1:
        raise ValueError("batch and bootstrap must be positive")
    if not set(primary_timesteps).issubset(timesteps):
        raise ValueError("primary timesteps must be in diagnostic timesteps")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "bf16" and device == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support BF16")
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(f"device={device}; split=VAL only; shuffle={args.shuffle_mode}")

    sample_ids = list_samples(data, "val")
    if args.limit:
        grouped = defaultdict(list)
        for sample_id in sample_ids:
            grouped[sample_parts(sample_id)[1]].append(sample_id)
        each = max(2, args.limit // 2)
        sample_ids = sorted(grouped["LCA"][:each] + grouped["RCA"][:each])
    dataset = CoronaryCenterlineDatasetV31(
        data, sample_ids=sample_ids, return_render_poses=False)
    items = [dataset[index] for index in range(len(dataset))]
    donor_maps = {seed: build_donor_map(sample_ids, seed) for seed in seeds}
    print(
        f"samples={len(sample_ids)}; patients="
        f"{len({sample_parts(s)[0] for s in sample_ids})}; seeds={seeds}")

    rows = []
    checkpoint_metadata = {}
    for label, checkpoint_path in checkpoints.items():
        print(f"Evaluating {label}: {checkpoint_path}")
        model_rows, metadata = evaluate_checkpoint(
            label=label, checkpoint_path=checkpoint_path,
            items=items, sample_ids=sample_ids, donor_maps=donor_maps,
            seeds=seeds, timesteps=timesteps, batch_size=args.batch,
            shuffle_mode=args.shuffle_mode, precision=args.precision,
            device=device, model_class=CenterlineDenoiser,
            scheduler_class=NoiseScheduler)
        rows.extend(model_rows)
        checkpoint_metadata[label] = metadata

    summary = summarize(
        rows, list(checkpoints), seeds, timesteps,
        primary_timesteps, args.bootstrap)
    summary["protocol"] = {
        "split": "val",
        "sample_count": len(sample_ids),
        "patient_count": len({sample_parts(s)[0] for s in sample_ids}),
        "seeds": seeds,
        "diagnostic_timesteps": timesteps,
        "primary_timesteps": primary_timesteps,
        "shuffle_mode": args.shuffle_mode,
        "donors": "same vessel, different patient",
        "noise_pairing": "identical for matched and shuffled",
        "aggregation": (
            "per-sample masked MSE; vessels averaged within patient; "
            "equal patient weight"),
        "bootstrap_repeats": args.bootstrap,
        "note": "Evaluation seeds are not independent training seeds.",
    }
    summary["checkpoints"] = checkpoint_metadata
    summary["donor_maps"] = {
        str(seed): {
            sample_ids[index]: sample_ids[donor]
            for index, donor in enumerate(donors)}
        for seed, donors in donor_maps.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "conditioning_sensitivity.json"
    csv_path = args.out_dir / "conditioning_sensitivity_rows.csv"
    json_path.write_text(json.dumps({
        "summary": summary, "rows": rows}, indent=2))
    write_csv(csv_path, rows)
    print("PRIMARY RESULT (VAL only)")
    for label, model_summary in summary["models"].items():
        ci = model_summary["patient_bootstrap_ci95_pct"]
        print(
            f"  {label}: {model_summary['primary_sensitivity_pct']:.2f}% "
            f"(patient-bootstrap 95% CI {ci[0]:.2f} to {ci[1]:.2f})")
    print(f"wrote {json_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
