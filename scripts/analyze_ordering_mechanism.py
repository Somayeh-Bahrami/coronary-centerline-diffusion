"""CPU-only Stage 1.5 screen for the optimal-ordering hypothesis.

The analysis never permutes an existing prediction and calls that a model
improvement. Instead, it asks whether the graph edges that would become local
under the exact optimal ordering are currently the edges carrying the largest
breakage and tree-length inflation.

Four structural edge groups are evaluated:

* always_consecutive: consecutive under DFS and optimal ordering;
* rescued: non-consecutive under DFS, consecutive under optimal ordering;
* sacrificed: consecutive under DFS, non-consecutive under optimal ordering;
* always_nonconsecutive: non-consecutive under both orderings.

Two explicitly hypothetical length estimates are also reported. They are
screening diagnostics, not predictions of a retrained model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.coronarycl.dataset_v3_1 import list_samples  # noqa: E402
from src.coronarycl.edge_coherence import load_edge_cache  # noqa: E402


GROUPS = (
    "always_consecutive",
    "rescued",
    "sacrificed",
    "always_nonconsecutive",
)
EXPECTED_CHECKPOINT_SHA256 = (
    "d16ba2d76b026864ab2134fbdfae1f1e3c8588b28e3da355769c3623e80aede1"
)
EXPECTED_SEEDS = [104729, 130363, 155921, 181081, 205019]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_edges(edges) -> np.ndarray:
    array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    array = np.sort(array, axis=1)
    return array[np.lexsort((array[:, 1], array[:, 0]))]


def load_evaluation_package(package: Path, prediction_member: str,
                            result_member: str):
    if not package.is_file():
        raise FileNotFoundError(f"evaluation package not found: {package}")
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        for member in (prediction_member, result_member):
            if member not in names:
                raise KeyError(f"{member!r} is missing from {package}")
        result = json.loads(archive.read(result_member))
        prediction_bytes = archive.read(prediction_member)
    with np.load(io.BytesIO(prediction_bytes), allow_pickle=True) as source:
        predictions = {name: source[name] for name in source.files}
    return result, predictions


def validate_protocol(result):
    checkpoint = result["checkpoint"]
    protocol = result["protocol"]
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(
            f"wrong checkpoint: {checkpoint['sha256']} != "
            f"{EXPECTED_CHECKPOINT_SHA256}")
    if int(checkpoint["step"]) != 150000 or int(checkpoint["hidden_dim"]) != 384:
        raise AssertionError("expected the frozen h384 step-150000 checkpoint")
    if result["mode"] != "conditional" or protocol["split"] != "val":
        raise AssertionError("expected conditional validation predictions")
    if list(map(int, protocol["seeds"])) != EXPECTED_SEEDS:
        raise AssertionError("evaluation seeds do not match the frozen protocol")
    if int(protocol["ddim_steps"]) != 100 or float(protocol["guidance"]) != 2.0:
        raise AssertionError("expected DDIM=100 and guidance=2.0")
    if protocol["bounds"] != "physical":
        raise AssertionError("expected physical prediction bounds")


def load_permutations(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"permutation bank not found: {path}")
    with np.load(path, allow_pickle=True) as source:
        meta = json.loads(str(source["_meta"]))
        permutations = {
            name: source[name].astype(np.int64)
            for name in source.files if name != "_meta"
        }
    if meta["schema_version"] != "optimal_ordering_v1":
        raise AssertionError("unexpected permutation schema")
    return permutations, meta


def edge_groups(edges, permutation):
    edges = canonical_edges(edges)
    n_nodes = len(permutation)
    if not np.array_equal(np.sort(permutation), np.arange(n_nodes)):
        raise AssertionError("invalid optimal-order permutation")
    position = np.empty(n_nodes, dtype=np.int64)
    position[permutation] = np.arange(n_nodes)
    current = np.abs(edges[:, 0] - edges[:, 1]) == 1
    optimal = np.abs(position[edges[:, 0]] - position[edges[:, 1]]) == 1
    labels = np.empty(len(edges), dtype=object)
    labels[current & optimal] = "always_consecutive"
    labels[~current & optimal] = "rescued"
    labels[current & ~optimal] = "sacrificed"
    labels[~current & ~optimal] = "always_nonconsecutive"
    return labels


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else float("nan")


def finite_mean(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def bootstrap_patient_mean(patient_values, *, repetitions, seed):
    """Bootstrap the equal-patient mean of one scalar per patient."""
    values = np.asarray(list(patient_values.values()), dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n_patients": 0, "mean": float("nan"), "ci95": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    bootstrap = values[draws].mean(axis=1)
    return {
        "n_patients": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--edge-cache", required=True)
    parser.add_argument("--permutations", required=True)
    parser.add_argument("--evaluation-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prediction-member",
        default="results/final_val/conditional_predictions.npz")
    parser.add_argument(
        "--result-member",
        default="results/final_val/conditional.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = Path(args.data).resolve()
    permutation_path = Path(args.permutations).resolve()
    package = Path(args.evaluation_package).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "ordering_mechanism_report.json"
    sample_path = out / "ordering_mechanism_per_sample.csv"
    if (report_path.exists() or sample_path.exists()) and not args.overwrite:
        raise FileExistsError("outputs already exist; pass --overwrite to replace them")

    val_ids = list_samples(data, "val")
    if len(val_ids) != 177:
        raise AssertionError(f"expected 177 validation samples, found {len(val_ids)}")
    edge_map, edge_meta = load_edge_cache(
        args.edge_cache, data, required_ids=val_ids)
    permutations, permutation_meta = load_permutations(permutation_path)
    result, prediction_store = load_evaluation_package(
        package, args.prediction_member, args.result_member)
    validate_protocol(result)

    result_ids = {row["sample"] for row in result["per_sample"]}
    if result_ids != set(val_ids):
        raise AssertionError("evaluation sample IDs do not match the validation split")

    pooled = {
        group: defaultdict(float) for group in GROUPS
    }
    structural_counts = defaultdict(int)
    sample_rows = []
    patient_group = {
        group: defaultdict(lambda: defaultdict(float)) for group in GROUPS
    }
    patient_counterfactual = defaultdict(lambda: defaultdict(float))
    legacy_edge_mismatches = []

    for sample in val_ids:
        patient = sample.rsplit("_", 1)[0]
        vessel = sample.rsplit("_", 1)[1]
        if sample not in permutations:
            raise KeyError(f"permutation bank is missing {sample}")
        gt = np.asarray(prediction_store[f"gt__{sample}"], dtype=np.float64)
        stored_edges = canonical_edges(prediction_store[f"edges__{sample}"])
        cached_edges = canonical_edges(edge_map[sample])
        if not np.array_equal(stored_edges, cached_edges):
            stored_set = {tuple(edge) for edge in stored_edges.tolist()}
            cached_set = {tuple(edge) for edge in cached_edges.tolist()}
            legacy_edge_mismatches.append({
                "sample": sample,
                "stored_edge_count": len(stored_set),
                "canonical_edge_count": len(cached_set),
                "shared_edge_count": len(stored_set & cached_set),
                "stored_only_count": len(stored_set - cached_set),
                "canonical_only_count": len(cached_set - stored_set),
            })
        with np.load(data / f"{sample}.npz", allow_pickle=True) as packaged:
            mask = packaged["centerline_mask"].astype(bool)
            dataset_gt = packaged["centerline"][mask, :4].astype(np.float64)
        if gt.shape != dataset_gt.shape or not np.allclose(gt, dataset_gt, atol=1e-5):
            raise AssertionError(f"{sample}: saved GT differs from packaged dataset")

        edges = cached_edges
        labels = edge_groups(edges, permutations[sample])
        gt_lengths = np.linalg.norm(
            gt[edges[:, 0], :3] - gt[edges[:, 1], :3], axis=1)
        if np.any(gt_lengths <= 0):
            raise AssertionError(f"{sample}: non-positive GT edge length")
        for group in GROUPS:
            structural_counts[group] += int(np.sum(labels == group))

        for seed in EXPECTED_SEEDS:
            prediction = np.asarray(
                prediction_store[f"pred__{seed}__{sample}"], dtype=np.float64)
            if prediction.shape != gt.shape:
                raise AssertionError(f"{sample}, seed {seed}: prediction shape mismatch")
            pred_lengths = np.linalg.norm(
                prediction[edges[:, 0], :3] - prediction[edges[:, 1], :3],
                axis=1)
            broken = pred_lengths > 5.0 * gt_lengths
            excess = pred_lengths - gt_lengths
            positive_excess = np.maximum(excess, 0.0)

            for group in GROUPS:
                selected = labels == group
                count = int(selected.sum())
                if count == 0:
                    continue
                values = {
                    "edge_observations": count,
                    "pred_length_sum": float(pred_lengths[selected].sum()),
                    "gt_length_sum": float(gt_lengths[selected].sum()),
                    "broken_count": int(broken[selected].sum()),
                    "signed_excess_sum": float(excess[selected].sum()),
                    "positive_excess_sum": float(positive_excess[selected].sum()),
                }
                for key, value in values.items():
                    pooled[group][key] += value
                    patient_group[group][patient][key] += value
                sample_rows.append({
                    "sample": sample,
                    "patient": patient,
                    "vessel": vessel,
                    "seed": seed,
                    "group": group,
                    "n_edges": count,
                    "pred_gt_length_ratio": ratio(
                        values["pred_length_sum"], values["gt_length_sum"]),
                    "broken_edge_fraction_5x": ratio(
                        values["broken_count"], count),
                    "signed_excess_mm": values["signed_excess_sum"],
                    "positive_excess_mm": values["positive_excess_sum"],
                })

            actual_total = float(pred_lengths.sum())
            gt_total = float(gt_lengths.sum())
            always_good = labels == "always_consecutive"
            always_bad = labels == "always_nonconsecutive"
            current_good = np.isin(labels, ["always_consecutive", "sacrificed"])
            current_bad = np.isin(labels, ["rescued", "always_nonconsecutive"])
            good_reference = always_good if always_good.any() else current_good
            bad_reference = always_bad if always_bad.any() else current_bad
            good_factor = ratio(
                pred_lengths[good_reference].sum(), gt_lengths[good_reference].sum())
            bad_factor = ratio(
                pred_lengths[bad_reference].sum(), gt_lengths[bad_reference].sum())

            transfer_lengths = pred_lengths.copy()
            rescued = labels == "rescued"
            sacrificed = labels == "sacrificed"
            transfer_lengths[rescued] = gt_lengths[rescued] * good_factor
            transfer_lengths[sacrificed] = gt_lengths[sacrificed] * bad_factor
            perfect_rescue_lengths = pred_lengths.copy()
            perfect_rescue_lengths[rescued] = gt_lengths[rescued]

            target = patient_counterfactual[patient]
            target["actual_pred_sum"] += actual_total
            target["transfer_pred_sum"] += float(transfer_lengths.sum())
            target["perfect_rescue_pred_sum"] += float(perfect_rescue_lengths.sum())
            target["gt_sum"] += gt_total

    total_structural = sum(structural_counts.values())
    total_signed_excess = sum(
        pooled[group]["signed_excess_sum"] for group in GROUPS)
    total_positive_excess = sum(
        pooled[group]["positive_excess_sum"] for group in GROUPS)
    group_summary = {}
    for group in GROUPS:
        values = pooled[group]
        patient_ratios = {
            patient: ratio(v["pred_length_sum"], v["gt_length_sum"])
            for patient, v in patient_group[group].items()
        }
        patient_broken = {
            patient: ratio(v["broken_count"], v["edge_observations"])
            for patient, v in patient_group[group].items()
        }
        group_summary[group] = {
            "structural_edge_count": int(structural_counts[group]),
            "structural_edge_fraction": ratio(
                structural_counts[group], total_structural),
            "edge_observations_across_seeds": int(values["edge_observations"]),
            "pooled_pred_gt_length_ratio": ratio(
                values["pred_length_sum"], values["gt_length_sum"]),
            "pooled_broken_edge_fraction_5x": ratio(
                values["broken_count"], values["edge_observations"]),
            "signed_excess_mm_per_edge_observation": ratio(
                values["signed_excess_sum"], values["edge_observations"]),
            "positive_excess_mm_per_edge_observation": ratio(
                values["positive_excess_sum"], values["edge_observations"]),
            "fraction_of_total_signed_excess": ratio(
                values["signed_excess_sum"], total_signed_excess),
            "fraction_of_total_positive_excess": ratio(
                values["positive_excess_sum"], total_positive_excess),
            "equal_patient_length_ratio": bootstrap_patient_mean(
                patient_ratios,
                repetitions=args.bootstrap,
                seed=args.bootstrap_seed + GROUPS.index(group)),
            "equal_patient_broken_fraction": bootstrap_patient_mean(
                patient_broken,
                repetitions=args.bootstrap,
                seed=args.bootstrap_seed + 10 + GROUPS.index(group)),
        }

    actual_patient = {}
    transfer_patient = {}
    perfect_patient = {}
    for patient, values in patient_counterfactual.items():
        actual_patient[patient] = ratio(values["actual_pred_sum"], values["gt_sum"])
        transfer_patient[patient] = ratio(values["transfer_pred_sum"], values["gt_sum"])
        perfect_patient[patient] = ratio(
            values["perfect_rescue_pred_sum"], values["gt_sum"])
    actual_bootstrap = bootstrap_patient_mean(
        actual_patient, repetitions=args.bootstrap, seed=args.bootstrap_seed + 20)
    transfer_bootstrap = bootstrap_patient_mean(
        transfer_patient, repetitions=args.bootstrap, seed=args.bootstrap_seed + 21)
    perfect_bootstrap = bootstrap_patient_mean(
        perfect_patient, repetitions=args.bootstrap, seed=args.bootstrap_seed + 22)
    actual_mean = actual_bootstrap["mean"]
    transfer_mean = transfer_bootstrap["mean"]
    perfect_mean = perfect_bootstrap["mean"]

    transfer_movement = ratio(actual_mean - transfer_mean, actual_mean - 1.0)
    perfect_movement = ratio(actual_mean - perfect_mean, actual_mean - 1.0)
    rescued = group_summary["rescued"]
    sacrificed = group_summary["sacrificed"]
    rescued_carries_more = bool(
        rescued["pooled_broken_edge_fraction_5x"]
        > sacrificed["pooled_broken_edge_fraction_5x"]
        and rescued["positive_excess_mm_per_edge_observation"]
        > sacrificed["positive_excess_mm_per_edge_observation"]
    )
    screen_pass = bool(rescued_carries_more and transfer_movement >= 0.10)

    report = {
        "schema_version": "ordering_mechanism_stage1_5_v1",
        "protocol": {
            "split": "val",
            "n_samples": len(val_ids),
            "n_patients": len(patient_counterfactual),
            "seeds": EXPECTED_SEEDS,
            "broken_edge_threshold": "predicted edge length > 5 * GT edge length",
            "checkpoint": result["checkpoint"],
            "evaluation": result["protocol"],
            "bootstrap_repetitions": int(args.bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "provenance": {
            "dataset": str(data),
            "edge_cache_sha256": sha256_file(
                Path(args.edge_cache) / "edges_v1.npz"
                if Path(args.edge_cache).is_dir() else Path(args.edge_cache)),
            "permutation_bank_sha256": sha256_file(permutation_path),
            "evaluation_package_sha256": sha256_file(package),
            "permutation_metadata": permutation_meta,
            "canonical_topology_policy": (
                "All Stage 1.5 metrics use the edge cache built directly from "
                "packaged float32 GT. Legacy edges reconstructed after GT "
                "normalization/de-normalization are diagnostic only."
            ),
            "legacy_saved_edge_mismatch": {
                "affected_samples": len(legacy_edge_mismatches),
                "fraction_of_validation_samples": ratio(
                    len(legacy_edge_mismatches), len(val_ids)),
                "details": legacy_edge_mismatches,
            },
        },
        "groups": group_summary,
        "counterfactual": {
            "actual_tree_length_ratio": actual_bootstrap,
            "adjacency_transfer_tree_length_ratio": transfer_bootstrap,
            "perfect_rescue_only_tree_length_ratio": perfect_bootstrap,
            "adjacency_transfer_fraction_of_excess_removed": transfer_movement,
            "perfect_rescue_fraction_of_excess_removed": perfect_movement,
            "warning": (
                "Counterfactual values are CPU-only mechanism screens, not "
                "predictions of a retrained model."
            ),
        },
        "screening_decision": {
            "rescued_carries_more_per_edge_breakage_and_positive_excess_than_sacrificed":
                rescued_carries_more,
            "adjacency_transfer_removes_at_least_10pct_of_excess":
                bool(transfer_movement >= 0.10),
            "proceed_to_paired_50k_ordering_pilot": screen_pass,
        },
    }

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with sample_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    print("STAGE 1.5 COMPLETE")
    print(f"validated checkpoint: {result['checkpoint']['sha256']}")
    print(f"validation samples/patients: {len(val_ids)}/{len(patient_counterfactual)}")
    print(f"legacy saved-edge mismatches: {len(legacy_edge_mismatches)}/{len(val_ids)} "
          "(canonical raw-dataset edge cache used for all calculations)")
    print("\nedge transition groups")
    for group in GROUPS:
        values = group_summary[group]
        print(
            f"  {group:24s} edges={values['structural_edge_fraction']*100:6.2f}% "
            f"length_ratio={values['pooled_pred_gt_length_ratio']:7.3f} "
            f"broken={values['pooled_broken_edge_fraction_5x']*100:6.2f}% "
            f"positive_excess/edge={values['positive_excess_mm_per_edge_observation']:8.3f} mm")
    print("\ncounterfactual patient-equal tree-length ratio")
    print(f"  actual:             {actual_mean:.4f}")
    print(f"  adjacency transfer: {transfer_mean:.4f}")
    print(f"  perfect rescue:     {perfect_mean:.4f}")
    print(f"  transfer removes {100*transfer_movement:.2f}% of excess above 1.0")
    print(f"  perfect rescue removes {100*perfect_movement:.2f}% of excess above 1.0")
    print("\nscreening decision")
    print(json.dumps(report["screening_decision"], indent=2))
    print(f"\nreport: {report_path}")
    print(f"per-sample table: {sample_path}")
    if screen_pass:
        print("NEXT: validate/build the derived dataset, then run the paired 50k pilot.")
    else:
        print("NEXT: reject ordering before GPU training and proceed to Stage 2.")


if __name__ == "__main__":
    main()
