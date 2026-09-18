__generated_with = "0.24.0"

# %%
import marimo as mo

# %%
mo.md(r"""
# Stage 2 — frozen branch-token 50k run (VAL only)

This Molab notebook executes only the frozen Stage 2 EBT plan. It pins code,
builds the derived branch-token dataset without regenerating projections, runs
the required smoke gate, trains/resumes the one 50k EBT arm, and compares it
with the frozen DFS checkpoint on validation only. TEST is locked.
""")

# %%
# Cell 1 — immutable revision, paths, uploads, hashes, and helpers
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_URL = "https://github.com/Somayeh-Bahrami/coronary-centerline-diffusion"
# This revision fixes the direct-CLI ordering in ddim_eval.py.
EXPECTED_COMMIT = "121ea526f5307510b38e5719d9cefa38c4ae7dce"
EXPECTED_SAMPLES = 1694
EXPECTED_SPLITS = {"train": 1350, "val": 177, "test": 167}
EXPECTED_JSON_HASHES = {
    "case_splits_v3.json": "de82889a4c151461d1970d696a9065f9c8ebbc85d1e864d584730ed6d57bab61",
    "norm_stats_v3.json": "ff336f4d9c159a079abdec41797cb7af6dc71c503a6af7d75e87b1743287fa66",
    "pilot_report_v3.json": "f82e3c021ab4a8662323cacfe01b389c2483079bd004f33ea79acd08b9584d78",
}
EXPECTED_DFS_EDGE_SHA256 = "d2a2d1eec9c89fe572337300bcc333a7dc52c1e84aa68984c8d9f03ebee49b56"
EXPECTED_DFS_CHECKPOINT_SHA256 = "97bab28a99f2b89fb03ca68e987671bf267fd103f60a0cc320979a291929bddb"
SAMPLING_SEEDS = [104729, 130363, 155921, 181081, 205019]

# Leave blank for strict automatic upload discovery.
DFS_ZIP_OVERRIDE = ""
DFS_EDGE_OVERRIDE = ""
DFS_CHECKPOINT_OVERRIDE = ""

WORK = Path.cwd().resolve()
REPO = WORK / f"coronary_stage2_{EXPECTED_COMMIT}"
RUN_ROOT = WORK / "stage2_branch_token_50k"
CONFIG_ROOT = RUN_ROOT / "configs"
CKPT_ROOT = RUN_ROOT / "checkpoints"
RESULT_ROOT = RUN_ROOT / "results"
LOG_ROOT = RUN_ROOT / "logs"
PACKAGE_ROOT = RUN_ROOT / "package"
for _path in (RUN_ROOT, CONFIG_ROOT, CKPT_ROOT, RESULT_ROOT, LOG_ROOT, PACKAGE_ROOT):
    _path.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stream(command, log_path, cwd=None, append=False):
    command = [str(part) for part in command]
    log_path = Path(log_path)
    with log_path.open("a" if append else "w", encoding="utf-8") as log:
        header = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {' '.join(command)}\n"
        print(header, end="")
        log.write(header)
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"Command failed ({status}): {' '.join(command)}")


def safely_extract(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def dataset_root(extract_root):
    roots = [path.parent for path in Path(extract_root).rglob("*.npz")
             if path.name != "edges_v1.npz"]
    candidates = sorted({path for path in roots if (path / "norm_stats_v3.json").is_file()})
    assert len(candidates) == 1, f"Expected one dataset root, found {candidates}"
    return candidates[0]


def write_immutable_config(config, path):
    rendered = yaml.safe_dump(config, sort_keys=False)
    path = Path(path)
    if path.exists():
        assert yaml.safe_load(path.read_text()) == config, f"Config changed: {path}"
    else:
        path.write_text(rendered)
    return path


def patient_values(report, metric):
    grouped = {}
    for row in report["per_sample"]:
        value = float(row[metric])
        if np.isfinite(value):
            grouped.setdefault(str(row["patient"]), []).append(value)
    return {patient: float(np.mean(values)) for patient, values in grouped.items()}


def paired_delta(first, second, metric, repeats=10_000):
    left, right = patient_values(first, metric), patient_values(second, metric)
    patients = sorted(set(left) & set(right))
    assert len(patients) >= 90
    values = np.asarray([right[p] - left[p] for p in patients])
    rng = np.random.default_rng(20260916)
    bootstrap = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {"metric": metric, "ebt_minus_dfs": float(values.mean()),
            "ci95_low": float(np.quantile(bootstrap, .025)),
            "ci95_high": float(np.quantile(bootstrap, .975)), "patients": len(patients)}

# %%
# Cell 2 — explicit controls for expensive frozen stages
setup_button = mo.ui.run_button(label="1. Verify code and build EBT dataset")
smoke_button = mo.ui.run_button(label="2. Run required 20-step smoke test")
training_button = mo.ui.run_button(label="3. Train or exact-resume EBT 50k")
evaluation_button = mo.ui.run_button(label="4. Evaluate fixed VAL protocol")
package_button = mo.ui.run_button(label="5. Compare and package artifacts")
mo.vstack([setup_button, smoke_button, training_button, evaluation_button, package_button])

# %%
# Cell 3 — verify pinned code, uploaded DFS inputs, and derived EBT data
mo.stop(not setup_button.value, mo.md("Click **1. Verify code and build EBT dataset**."))

if not (REPO / ".git").is_dir():
    assert not REPO.exists(), f"Incomplete repository: {REPO}"
    run_stream(["git", "clone", REPO_URL, REPO], LOG_ROOT / "clone.log")
    run_stream(["git", "checkout", "--detach", EXPECTED_COMMIT], LOG_ROOT / "checkout.log", cwd=REPO)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert head == EXPECTED_COMMIT, (head, EXPECTED_COMMIT)
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()

# A fresh pinned clone must be syntactically valid before any expensive stage.
compile((REPO / "ddim_eval.py").read_text(), str(REPO / "ddim_eval.py"), "exec")

run_stream([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "marimo", "scipy", "nibabel", "scikit-image", "tqdm", "matplotlib",
            "pandas", "PyYAML", "pytest"], LOG_ROOT / "dependencies.log")

uploads = [path.resolve() for path in WORK.iterdir() if path.is_file()]
if DFS_ZIP_OVERRIDE:
    dfs_zip = Path(DFS_ZIP_OVERRIDE).expanduser().resolve()
else:
    zip_candidates = [p for p in uploads if p.suffix == ".zip"]
    assert len(zip_candidates) == 1, f"Upload exactly one DFS dataset ZIP: {zip_candidates}"
    dfs_zip = zip_candidates[0]
if DFS_EDGE_OVERRIDE:
    dfs_edge = Path(DFS_EDGE_OVERRIDE).expanduser().resolve()
else:
    matches = [p for p in uploads if p.name == "edges_v1.npz" and sha256_file(p) == EXPECTED_DFS_EDGE_SHA256]
    assert len(matches) == 1, f"Expected the frozen DFS edge cache: {matches}"
    dfs_edge = matches[0]
if DFS_CHECKPOINT_OVERRIDE:
    dfs_checkpoint = Path(DFS_CHECKPOINT_OVERRIDE).expanduser().resolve()
else:
    matches = [p for p in uploads if p.suffix == ".pt" and sha256_file(p) == EXPECTED_DFS_CHECKPOINT_SHA256]
    assert len(matches) == 1, f"Expected frozen DFS 50k checkpoint: {matches}"
    dfs_checkpoint = matches[0]

extract_root = RUN_ROOT / f"input_dfs_{sha256_file(dfs_zip)[:12]}"
if not extract_root.exists():
    extract_root.mkdir()
    safely_extract(dfs_zip, extract_root)
DFS_DATA = dataset_root(extract_root)
for name, digest in EXPECTED_JSON_HASHES.items():
    assert sha256_file(DFS_DATA / name) == digest
split_counts = {}
for npz_path in DFS_DATA.glob("*.npz"):
    with np.load(npz_path, allow_pickle=False) as sample:
        split = str(sample["split"])
    split_counts[split] = split_counts.get(split, 0) + 1
assert split_counts == EXPECTED_SPLITS, split_counts

DFS_EDGE = RUN_ROOT / "dfs_edges_v1.npz"
if not DFS_EDGE.exists():
    shutil.copy2(dfs_edge, DFS_EDGE)
assert sha256_file(DFS_EDGE) == EXPECTED_DFS_EDGE_SHA256
DFS_CHECKPOINT = RUN_ROOT / "dfs_50k_latest.pt"
if not DFS_CHECKPOINT.exists():
    shutil.copy2(dfs_checkpoint, DFS_CHECKPOINT)
assert sha256_file(DFS_CHECKPOINT) == EXPECTED_DFS_CHECKPOINT_SHA256

EBT_DATA = RUN_ROOT / "ds105_branch_token_v1"
EBT_EDGE = RUN_ROOT / "branch_token_edge_cache_v1"
if not EBT_DATA.exists():
    run_stream([sys.executable, REPO / "scripts/build_branch_token_dataset.py",
                "--source", DFS_DATA, "--source-edge-cache", DFS_EDGE,
                "--metadata-dir", DFS_DATA, "--out", EBT_DATA, "--edge-out", EBT_EDGE],
               LOG_ROOT / "build_ebt_dataset.log", cwd=REPO)
manifest = json.loads((EBT_DATA / "branch_token_manifest.json").read_text())
assert manifest["representation_version"] == "explicit_branch_token_tree_v1"
assert manifest["token_capacity"] == 2500
assert (EBT_EDGE / "edges_v1.npz").is_file()
run_stream([sys.executable, "-m", "pytest", "-q",
            "tests/test_branch_token_tree.py", "tests/test_build_branch_token_dataset.py",
            "tests/test_sampling.py", "tests/test_trainer.py", "tests/test_diffusion_integration.py",
            "tests/test_ddim_eval.py", "tests/test_sensitivity.py"],
           LOG_ROOT / "stage2_tests.log", cwd=REPO)

base_config = yaml.safe_load((REPO / "configs/h384_ordering_50k_dfs.yaml").read_text())
ebt_config = yaml.safe_load((REPO / "configs/h384_branch_token_stage2.yaml").read_text())
ebt_config["data"]["packaged_dir"] = str(EBT_DATA)
ebt_config["train"]["checkpoint_dir"] = str(CKPT_ROOT / "ebt" / "full")
ebt_config["eval"]["topology_edge_cache"] = str(EBT_EDGE)
left, right = deepcopy(base_config), deepcopy(ebt_config)
left["data"]["packaged_dir"] = right["data"]["packaged_dir"] = "<data>"
left["train"]["checkpoint_dir"] = right["train"]["checkpoint_dir"] = "<output>"
left["eval"]["topology_edge_cache"] = right["eval"]["topology_edge_cache"] = "<edges>"
assert left == right, "EBT config differs from frozen DFS config beyond paths"
FULL_CONFIG = write_immutable_config(ebt_config, CONFIG_ROOT / "ebt_50k.yaml")
SETUP_COMPLETE = True
print("SETUP PASSED; TEST WAS NOT ACCESSED")

# %%
# Cell 4 — required 20-step smoke gate
assert SETUP_COMPLETE
mo.stop(not smoke_button.value, mo.md("Click **2. Run required 20-step smoke test**."))
smoke = deepcopy(yaml.safe_load(FULL_CONFIG.read_text()))
smoke["train"].update({"checkpoint_dir": str(CKPT_ROOT / "ebt" / "smoke"),
                       "max_steps": 20, "warmup_steps": 2, "val_every": 10,
                       "checkpoint_every": 10, "milestone_every": 0, "max_hours": .5})
SMOKE_CONFIG = write_immutable_config(smoke, CONFIG_ROOT / "ebt_smoke.yaml")
SMOKE_CHECKPOINT = Path(smoke["train"]["checkpoint_dir"]) / "latest.pt"
if not SMOKE_CHECKPOINT.exists():
    run_stream([sys.executable, "-u", REPO / "train.py", "--config", SMOKE_CONFIG],
               LOG_ROOT / "smoke_ebt.log", cwd=REPO)
smoke_payload = torch.load(SMOKE_CHECKPOINT, map_location="cpu", weights_only=True)
assert int(smoke_payload["step"]) == 20 and smoke_payload["node_dim"] == 8
SMOKE_COMPLETE = True
print("EBT 20-STEP SMOKE PASSED")

# %%
# Cell 5 — train/resume the one frozen 50k EBT arm
assert SMOKE_COMPLETE
mo.stop(not training_button.value, mo.md("Click **3. Train or exact-resume EBT 50k**."))
EBT_CHECKPOINT = Path(ebt_config["train"]["checkpoint_dir"]) / "latest.pt"
if not EBT_CHECKPOINT.exists() or int(torch.load(EBT_CHECKPOINT, map_location="cpu", weights_only=True)["step"]) < 50_000:
    run_stream([sys.executable, "-u", REPO / "train.py", "--config", FULL_CONFIG],
               LOG_ROOT / "train_ebt.log", cwd=REPO, append=EBT_CHECKPOINT.exists())
payload = torch.load(EBT_CHECKPOINT, map_location="cpu", weights_only=True)
assert int(payload["step"]) == 50_000 and payload["stop_reason"] == "max_steps"
assert payload["node_dim"] == 8
TRAINING_COMPLETE = True
print("EBT 50K TRAINING COMPLETE")

# %%
# Cell 6 — fixed five-seed VAL-only evaluation and conditioning sensitivity
assert TRAINING_COMPLETE
mo.stop(not evaluation_button.value, mo.md("Click **4. Evaluate fixed VAL protocol**."))
REPORTS = {}
for label, checkpoint, data, edges in (("dfs", DFS_CHECKPOINT, DFS_DATA, DFS_EDGE),
                                       ("ebt", EBT_CHECKPOINT, EBT_DATA, EBT_EDGE)):
    output = RESULT_ROOT / f"{label}_val.json"
    predictions = RESULT_ROOT / f"{label}_predictions.npz"
    if not output.exists():
        run_stream([sys.executable, "-u", REPO / "ddim_eval.py", "--ckpt", checkpoint,
                    "--data", data, "--edge-cache", edges, "--split", "val", "--steps", "100",
                    "--guidance", "2.0", "--batch", "16", "--seeds", ",".join(map(str, SAMPLING_SEEDS)),
                    "--bootstrap", "10000", "--bounds", "physical", "--precision", "fp32",
                    "--out", output, "--save-pred", predictions],
                   LOG_ROOT / f"evaluate_{label}.log", cwd=REPO)
    report = json.loads(output.read_text())
    assert report["protocol"]["split"] == "val"
    assert report["protocol"]["seeds"] == SAMPLING_SEEDS
    REPORTS[label] = report

SENSITIVITY = {}
for label, checkpoint, data in (("dfs", DFS_CHECKPOINT, DFS_DATA), ("ebt", EBT_CHECKPOINT, EBT_DATA)):
    SENSITIVITY[label] = {}
    for mode in ("joint", "images"):
        output_dir = RESULT_ROOT / f"sensitivity_{label}_{mode}"
        output = output_dir / "conditioning_sensitivity.json"
        if not output.exists():
            run_stream([sys.executable, "-u", REPO / "cond_sensitivity.py", "--repo", REPO,
                        "--data", data, "--ckpt", f"{label}={checkpoint}", "--out-dir", output_dir,
                        "--batch", "16", "--seeds", ",".join(map(str, SAMPLING_SEEDS)),
                        "--bootstrap", "10000", "--shuffle-mode", mode, "--precision", "fp32"],
                       LOG_ROOT / f"sensitivity_{label}_{mode}.log", cwd=REPO)
        SENSITIVITY[label][mode] = json.loads(output.read_text())
EVALUATION_COMPLETE = True
print("VAL EVALUATION AND SENSITIVITY COMPLETE; TEST WAS NOT ACCESSED")

# %%
# Cell 7 — preregistered decision, reproducibility package, and downloads
assert EVALUATION_COMPLETE
mo.stop(not package_button.value, mo.md("Click **5. Compare and package artifacts**."))
METRICS = ["chamfer_l2", "hd95_mm", "overlap@1.0mm", "overlap@2.0mm", "overlap@5.0mm",
           "edge_continuity_5x", "broken_edge_fraction_5x", "largest_connected_component_fraction_5x",
           "tree_length_ratio", "radius_mae_mm", "radius_rmse_mm", "radius_bias_mm",
           "radius_correlation", "out_of_crop_fraction"]
summary_rows = []
for label in ("dfs", "ebt"):
    row = {"arm": label}
    for metric in METRICS:
        row[metric] = float(np.mean(list(patient_values(REPORTS[label], metric).values())))
    for mode in ("joint", "images"):
        row[f"sensitivity_{mode}_pct"] = float(SENSITIVITY[label][mode]["summary"]["models"][label]["primary_sensitivity_pct"])
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
deltas = pd.DataFrame([paired_delta(REPORTS["dfs"], REPORTS["ebt"], metric) for metric in METRICS])
dfs, ebt = summary_rows
lcc = deltas.loc[deltas["metric"] == "largest_connected_component_fraction_5x"].iloc[0]
tree_progress = (abs(dfs["tree_length_ratio"] - 1) - abs(ebt["tree_length_ratio"] - 1)) / max(abs(dfs["tree_length_ratio"] - 1), 1e-12)
checks = {
    "lcc_improves_beyond_patient_ci": bool(lcc["ci95_low"] > 0),
    "broken_edge_fraction_decreases": ebt["broken_edge_fraction_5x"] < dfs["broken_edge_fraction_5x"],
    "tree_length_moves_at_least_10pct_toward_one": tree_progress >= .10,
    "chamfer_degrades_no_more_than_5pct": ebt["chamfer_l2"] <= 1.05 * dfs["chamfer_l2"],
    "hd95_degrades_no_more_than_5pct": ebt["hd95_mm"] <= 1.05 * dfs["hd95_mm"],
    "radius_mae_degrades_no_more_than_5pct": ebt["radius_mae_mm"] <= 1.05 * dfs["radius_mae_mm"],
    "radius_rmse_degrades_no_more_than_5pct": ebt["radius_rmse_mm"] <= 1.05 * dfs["radius_rmse_mm"],
    "radius_correlation_drop_no_more_than_0.05": ebt["radius_correlation"] >= dfs["radius_correlation"] - .05,
    "joint_sensitivity_positive_and_at_least_80pct_of_dfs": ebt["sensitivity_joint_pct"] > 0 and ebt["sensitivity_joint_pct"] >= .8 * dfs["sensitivity_joint_pct"],
    "image_sensitivity_positive_and_at_least_80pct_of_dfs": ebt["sensitivity_images_pct"] > 0 and ebt["sensitivity_images_pct"] >= .8 * dfs["sensitivity_images_pct"],
}
decision = {"advance_to_200k": bool(all(checks.values())), "checks": checks,
            "tree_length_relative_progress_toward_one": float(tree_progress), "test_accessed": False}
summary.to_csv(RESULT_ROOT / "paired_summary.csv", index=False)
deltas.to_csv(RESULT_ROOT / "paired_patient_deltas.csv", index=False)
(RESULT_ROOT / "advancement_decision.json").write_text(json.dumps(decision, indent=2))

archive = RUN_ROOT / "stage2_branch_token_50k_reproducibility.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
    for path in [FULL_CONFIG, SMOKE_CONFIG, EBT_DATA / "branch_token_manifest.json",
                 RESULT_ROOT / "dfs_val.json", RESULT_ROOT / "ebt_val.json",
                 RESULT_ROOT / "paired_summary.csv", RESULT_ROOT / "paired_patient_deltas.csv",
                 RESULT_ROOT / "advancement_decision.json", EBT_CHECKPOINT, DFS_CHECKPOINT]:
        out.write(path, path.relative_to(RUN_ROOT))
    for path in LOG_ROOT.glob("*.log"):
        out.write(path, path.relative_to(RUN_ROOT))

mo.vstack([
    mo.md("## Stage 2 paired VAL results"), mo.ui.table(summary), mo.ui.table(deltas),
    mo.md(
        f"**Advance to 200k:** `{decision['advance_to_200k']}`\n\n"
        f"TEST accessed: `{decision['test_accessed']}`"
    ),
    mo.download(archive.read_bytes(), filename=archive.name, label="Download reproducibility package"),
    mo.download(EBT_CHECKPOINT.read_bytes(), filename="ebt_50k_latest.pt", label="Download EBT checkpoint"),
])
