"""Recompute the manuscript's main validation table from the archived CSVs.

No inference is run. Reads the per-sample records in this directory, averages
them per patient, then across patients, and prints the reported values. Also
checks the paired-delta files against the per-patient differences.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARMS = {
    "dfs": "dfs_val.csv",
    "optimal": "optimal_val.csv",
    "branch": "ebt_val.csv",
}

DELTAS = {
    "optimal": ("ord_paired_deltas.csv", "optimal_minus_dfs"),
    "branch": ("br_paired_deltas.csv", "ebt_minus_dfs"),
}

TABLE = [
    ("chamfer_l2", "chamfer", 1.0, "{:>10.3f}"),
    ("broken_edge_fraction_5x", "broken%", 100.0, "{:>10.3f}"),
    ("largest_connected_component_fraction_5x", "LCC%", 100.0, "{:>10.2f}"),
    ("tree_length_ratio", "tree-len", 1.0, "{:>10.3f}"),
]

TOLERANCE = 5e-4


def load(name):
    with open(HERE / name, newline="") as handle:
        return list(csv.DictReader(handle))


def per_patient(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient"]].append(float(row[key]))
    return {patient: statistics.mean(v) for patient, v in grouped.items()}


def main():
    data = {arm: load(name) for arm, name in ARMS.items()}

    for arm, rows in data.items():
        seeds = {row["n_seeds"] for row in rows}
        patients = {row["patient"] for row in rows}
        print(f"{arm:<9} rows={len(rows):>4}  patients={len(patients):>3}  "
              f"n_seeds={sorted(seeds)}")
    print()

    header = f"{'arm':<9}" + "".join(f"{label:>10}" for _, label, _, _ in TABLE)
    print(header)
    for arm, rows in data.items():
        line = f"{arm:<9}"
        for key, _, scale, fmt in TABLE:
            line += fmt.format(statistics.mean(per_patient(rows, key).values()) * scale)
        print(line)
    print()

    print("paired per-patient deltas (recomputed vs archived)")
    failures = 0
    for arm, (fname, column) in DELTAS.items():
        archived = {row["metric"]: float(row[column]) for row in load(fname)}
        for key, label, _, _ in TABLE:
            base = per_patient(data["dfs"], key)
            other = per_patient(data[arm], key)
            shared = sorted(set(base) & set(other))
            recomputed = statistics.mean(other[p] - base[p] for p in shared)
            stored = archived[key]
            ok = abs(recomputed - stored) <= TOLERANCE
            failures += not ok
            print(f"  {arm:<8} {label:<9} recomputed={recomputed:+9.5f} "
                  f"archived={stored:+9.5f}  n={len(shared)}  "
                  f"{'OK' if ok else 'MISMATCH'}")

    print()
    print("all paired deltas reproduce" if not failures
          else f"{failures} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
