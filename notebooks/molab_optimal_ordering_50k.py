__generated_with = "0.24.0"

# %%
import marimo as mo

# %%
mo.md(r"""
# Optimal-ordering paired 50k pilot (VAL only)

This notebook compares the current DFS serialization with the exact optimal
linear serialization. Both models start from random weights and use identical
code, seed, architecture, optimizer, schedule, precision, and evaluation.
Only centerline row ordering and its corresponding canonical edge cache differ.
TEST remains locked.
""")

# %%
# Cell 1 — environment, immutable revision, paths, hashes, and helpers
import hashlib
import json
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
EXPECTED_COMMIT = "e83db83b650dc99e837b5ea4a03f0fa46642e7b7"
EXPECTED_SAMPLES = 1694
EXPECTED_SPLITS = {"train": 1350, "val": 177, "test": 167}
EXPECTED_JSON_HASHES = {
    "case_splits_v3.json": "de82889a4c151461d1970d696a9065f9c8ebbc85d1e864d584730ed6d57bab61",
    "norm_stats_v3.json": "ff336f4d9c159a079abdec41797cb7af6dc71c503a6af7d75e87b1743287fa66",
    "pilot_report_v3.json": "f82e3c021ab4a8662323cacfe01b389c2483079bd004f33ea79acd08b9584d78",
}
EXPECTED_OPTIMAL_ARCHIVE_HASH = "b27e51c9496efc3a0333f191556158363d10a55c0c1a700f9eafab29ee9e5dc8"
EXPECTED_OPTIMAL_MANIFEST_HASH = "deb6aa2617b3e55d43d418fa321db006c278c2c7711af5e43955c04a9782d8ab"
EXPECTED_OPTIMAL_QA_HASH = "6b1ddb2e15603c04ca4f419003dd43238e2bb63c44a4fc45f411e34a75cdc40e"
EXPECTED_EDGE_HASHES = {
    "dfs": "d2a2d1eec9c89fe572337300bcc333a7dc52c1e84aa68984c8d9f03ebee49b56",
    "optimal": "fc8cae94302aff0cbe920a23c54af9a9b13542133448cf727a3643044a7c303f",
}
SAMPLING_SEEDS = [104729, 130363, 155921, 181081, 205019]
TRAINING_SEED = 20260911

# Optional exact uploaded paths. Leave blank for strict automatic discovery.
DFS_ZIP_OVERRIDE = ""
OPTIMAL_ZIP_OVERRIDE = ""
DFS_EDGE_OVERRIDE = ""
OPTIMAL_EDGE_OVERRIDE = ""

WORK = Path.cwd().resolve()
REPO = WORK / f"coronary_{EXPECTED_COMMIT[:7]}"
RUN_ROOT = WORK / "optimal_ordering_50k_pilot_s20260911"
CONFIG_ROOT = RUN_ROOT / "configs"
CKPT_ROOT = RUN_ROOT / "checkpoints"
RESULT_ROOT = RUN_ROOT / "results"
LOG_ROOT = RUN_ROOT / "logs"
VIZ_ROOT = RUN_ROOT / "visualizations"
for _path in (RUN_ROOT, CONFIG_ROOT, CKPT_ROOT, RESULT_ROOT, LOG_ROOT, VIZ_ROOT):
    _path.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def run_stream(command, log_path, cwd=None, append=False):
    command = [str(part) for part in command]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as log:
        header = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {' '.join(command)}\n"
        print(header, end="")
        log.write(header)
        process = subprocess.Popen(
            command, cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Command failed ({return_code}); see {log_path}")


def safely_extract(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None, f"Corrupt ZIP: {archive_path}"
        assert sum(item.file_size for item in archive.infolist()) < 100 * 2**30
        for item in archive.infolist():
            relative = Path(item.filename)
            mode = (item.external_attr >> 16) & 0xFFFF
            assert not relative.is_absolute() and ".." not in relative.parts
            assert not stat.S_ISLNK(mode), f"Symlink is not allowed: {relative}"
            target = (destination / relative).resolve()
            assert target == destination or destination in target.parents
        archive.extractall(destination)


def dataset_roots(search_root):
    roots = []
    for stats in Path(search_root).rglob("norm_stats_v3.json"):
        candidate = stats.parent.resolve()
        if REPO == candidate or REPO in candidate.parents:
            continue
        if len(list(candidate.glob("*.npz"))) == EXPECTED_SAMPLES:
            roots.append(candidate)
    return sorted(set(roots))


assert torch.cuda.is_available(), "Attach the Molab GPU and rerun this cell"
assert torch.cuda.is_bf16_supported(), "The selected GPU must support BF16"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
GPU_INFO = {
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
}
GPU_INFO

# %%
# Cell 2 — explicit controls for expensive stages
setup_button = mo.ui.run_button(label="1. Verify code and both datasets")
smoke_button = mo.ui.run_button(label="2. Run paired 20-step smoke tests")
benchmark_button = mo.ui.run_button(label="3. Benchmark both arms")
training_button = mo.ui.run_button(label="4. Train or exact-resume both 50k arms")
evaluation_button = mo.ui.run_button(label="5. Evaluate fixed VAL protocol")
package_button = mo.ui.run_button(label="6. Compare, visualize, and package")
mo.vstack([
    setup_button, smoke_button, benchmark_button,
    training_button, evaluation_button, package_button,
])

# %%
# Cell 3 — clone pinned code, locate four uploads, and enforce integrity gates
mo.stop(not setup_button.value, mo.md("Click **1. Verify code and both datasets**."))

if not (REPO / ".git").is_dir():
    assert not REPO.exists(), f"Incomplete repository path: {REPO}"
    run_stream(["git", "clone", REPO_URL, REPO], LOG_ROOT / "clone.log")
    run_stream(["git", "checkout", "--detach", EXPECTED_COMMIT],
               LOG_ROOT / "checkout.log", cwd=REPO)

REPO_HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
    capture_output=True, text=True).stdout.strip()
REPO_STATUS = subprocess.run(
    ["git", "status", "--porcelain"], cwd=REPO, check=True,
    capture_output=True, text=True).stdout.strip()
assert REPO_HEAD == EXPECTED_COMMIT and not REPO_STATUS

run_stream([
    sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "scipy", "nibabel", "scikit-image", "tqdm", "matplotlib",
    "pandas", "PyYAML", "pytest",
], LOG_ROOT / "dependencies.log")

_zip_candidates = [
    path.resolve() for path in WORK.glob("*.zip")
    if RUN_ROOT not in path.parents
]
if OPTIMAL_ZIP_OVERRIDE:
    _optimal_zip = Path(OPTIMAL_ZIP_OVERRIDE).expanduser().resolve()
else:
    _optimal_matches = [
        path for path in _zip_candidates
        if sha256_file(path) == EXPECTED_OPTIMAL_ARCHIVE_HASH
    ]
    assert len(_optimal_matches) == 1, (
        f"Expected one validated optimal ZIP; found {_optimal_matches}")
    _optimal_zip = _optimal_matches[0]


def is_valid_dfs_archive(path):
    if path == _optimal_zip:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if sum(name.endswith(".npz") for name in names) != EXPECTED_SAMPLES:
                return False
            for filename, expected in EXPECTED_JSON_HASHES.items():
                matches = [name for name in names if Path(name).name == filename]
                if len(matches) != 1 or sha256_bytes(archive.read(matches[0])) != expected:
                    return False
            return True
    except zipfile.BadZipFile:
        return False


if DFS_ZIP_OVERRIDE:
    _dfs_zip = Path(DFS_ZIP_OVERRIDE).expanduser().resolve()
    assert is_valid_dfs_archive(_dfs_zip)
else:
    _dfs_matches = [path for path in _zip_candidates if is_valid_dfs_archive(path)]
    assert len(_dfs_matches) == 1, (
        f"Expected one validated DFS dataset ZIP; found {_dfs_matches}")
    _dfs_zip = _dfs_matches[0]

_dfs_extract = RUN_ROOT / f"input_dfs_{sha256_file(_dfs_zip)[:12]}"
_optimal_extract = RUN_ROOT / f"input_optimal_{EXPECTED_OPTIMAL_ARCHIVE_HASH[:12]}"
if not dataset_roots(_dfs_extract):
    safely_extract(_dfs_zip, _dfs_extract)
if not dataset_roots(_optimal_extract):
    safely_extract(_optimal_zip, _optimal_extract)
_dfs_roots = dataset_roots(_dfs_extract)
_optimal_roots = dataset_roots(_optimal_extract)
assert len(_dfs_roots) == 1 and len(_optimal_roots) == 1
DFS_DATA = _dfs_roots[0]
OPTIMAL_DATA = _optimal_roots[0]
assert not (DFS_DATA / "optimal_order_manifest_v1.json").exists()
assert sha256_file(OPTIMAL_DATA / "optimal_order_manifest_v1.json") == EXPECTED_OPTIMAL_MANIFEST_HASH
assert sha256_file(OPTIMAL_DATA / "optimal_order_qa_v1.json") == EXPECTED_OPTIMAL_QA_HASH
for _root in (DFS_DATA, OPTIMAL_DATA):
    for _name, _digest in EXPECTED_JSON_HASHES.items():
        assert sha256_file(_root / _name) == _digest, (_root, _name)


def uploaded_edge_cache(override, expected_hash):
    if override:
        candidate = Path(override).expanduser().resolve()
        assert candidate.is_file() and sha256_file(candidate) == expected_hash
        return candidate
    matches = [
        path.resolve() for path in WORK.glob("*.npz")
        if sha256_file(path) == expected_hash
    ]
    assert len(matches) == 1, f"Expected one uploaded edge cache {expected_hash[:12]}"
    return matches[0]


DFS_EDGES = uploaded_edge_cache(DFS_EDGE_OVERRIDE, EXPECTED_EDGE_HASHES["dfs"])
OPTIMAL_EDGES = uploaded_edge_cache(OPTIMAL_EDGE_OVERRIDE, EXPECTED_EDGE_HASHES["optimal"])

for _module in list(sys.modules):
    if _module == "src" or _module.startswith("src."):
        del sys.modules[_module]
sys.path.insert(0, str(REPO))
from src.coronarycl.dataset_v3_1 import list_samples
from src.coronarycl.edge_coherence import load_edge_cache

DATASETS = {"dfs": DFS_DATA, "optimal": OPTIMAL_DATA}
EDGE_CACHES = {"dfs": DFS_EDGES, "optimal": OPTIMAL_EDGES}
for _arm in ("dfs", "optimal"):
    _counts = {
        split: len(list_samples(DATASETS[_arm], split))
        for split in ("train", "val", "test")
    }
    assert _counts == EXPECTED_SPLITS
    _all_ids = list_samples(DATASETS[_arm])
    _, _edge_meta = load_edge_cache(
        EDGE_CACHES[_arm], DATASETS[_arm], required_ids=_all_ids)
    assert int(_edge_meta["n_verified"]) == EXPECTED_SAMPLES

run_stream([sys.executable, "-m", "pytest", "-q"],
           LOG_ROOT / "pytest.log", cwd=REPO)
run_stream([sys.executable, "-m", "src.coronarycl.models.diffusion"],
           LOG_ROOT / "diffusion_self_test.log", cwd=REPO)
SETUP_COMPLETE = True
print("SETUP PASSED", REPO_HEAD)
print("DFS:", DFS_DATA, sha256_file(DFS_EDGES))
print("OPTIMAL:", OPTIMAL_DATA, sha256_file(OPTIMAL_EDGES))

# %%
# Cell 4 — materialize immutable paired configurations
assert SETUP_COMPLETE


def write_immutable_config(config, path):
    path = Path(path)
    rendered = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        assert yaml.safe_load(path.read_text()) == config, f"Config changed: {path}"
    else:
        path.write_text(rendered)
    return path


def resolved_config(arm):
    path = REPO / "configs" / f"h384_ordering_50k_{arm}.yaml"
    config = yaml.safe_load(path.read_text())
    config["data"]["packaged_dir"] = str(DATASETS[arm])
    config["train"]["checkpoint_dir"] = str(CKPT_ROOT / arm / "full")
    config["eval"]["topology_edge_cache"] = str(EDGE_CACHES[arm])
    return config


_dfs_full = resolved_config("dfs")
_optimal_full = resolved_config("optimal")
_left, _right = deepcopy(_dfs_full), deepcopy(_optimal_full)
for _config in (_left, _right):
    _config["data"]["packaged_dir"] = "<paired-data>"
    _config["train"]["checkpoint_dir"] = "<paired-output>"
    _config["eval"]["topology_edge_cache"] = "<paired-edges>"
assert _left == _right, "Paired configs differ beyond preregistered paths"

FULL_CONFIGS = {
    "dfs": write_immutable_config(_dfs_full, CONFIG_ROOT / "dfs_50k.yaml"),
    "optimal": write_immutable_config(
        _optimal_full, CONFIG_ROOT / "optimal_50k.yaml"),
}
CONFIGS_COMPLETE = True
print({key: str(value) for key, value in FULL_CONFIGS.items()})

# %%
# Cell 5 — paired 20-step end-to-end smoke tests
assert CONFIGS_COMPLETE
mo.stop(not smoke_button.value, mo.md("Click **2. Run paired 20-step smoke tests**."))

SMOKE_RESULTS = {}
for _arm, _full_path in FULL_CONFIGS.items():
    _config = yaml.safe_load(_full_path.read_text())
    _directory = CKPT_ROOT / _arm / "smoke"
    _config["train"].update({
        "checkpoint_dir": str(_directory), "max_steps": 20,
        "warmup_steps": 2, "val_every": 10, "checkpoint_every": 10,
        "milestone_every": 0, "max_hours": 0.5,
    })
    _path = write_immutable_config(_config, CONFIG_ROOT / f"{_arm}_smoke.yaml")
    _latest = _directory / "latest.pt"
    if not _latest.exists():
        run_stream([sys.executable, "-u", REPO / "train.py", "--config", _path],
                   LOG_ROOT / f"smoke_{_arm}.log", cwd=REPO)
    _payload = torch.load(_latest, map_location="cpu", weights_only=True)
    assert int(_payload["step"]) == 20 and _payload["stop_reason"] == "max_steps"
    assert _payload.get("prediction_type", "epsilon") == "epsilon"
    SMOKE_RESULTS[_arm] = {
        "train_loss": _payload["history"]["train_loss"][-1],
        "val_loss": _payload["history"]["val_loss"][-1],
    }
SMOKE_COMPLETE = True
print("PAIRED SMOKE PASSED", SMOKE_RESULTS)

# %%
# Cell 6 — paired 1,000-step speed benchmark
assert SMOKE_COMPLETE
mo.stop(not benchmark_button.value, mo.md("Click **3. Benchmark both arms**."))

BENCHMARK_RESULTS = {}
for _arm, _full_path in FULL_CONFIGS.items():
    _config = yaml.safe_load(_full_path.read_text())
    _directory = CKPT_ROOT / _arm / "benchmark"
    _config["train"].update({
        "checkpoint_dir": str(_directory), "max_steps": 1000,
        "warmup_steps": 10, "val_every": 1000, "checkpoint_every": 1000,
        "milestone_every": 0, "max_hours": 0.5,
    })
    _path = write_immutable_config(_config, CONFIG_ROOT / f"{_arm}_benchmark.yaml")
    _latest = _directory / "latest.pt"
    if not _latest.exists():
        run_stream([sys.executable, "-u", REPO / "train.py", "--config", _path],
                   LOG_ROOT / f"benchmark_{_arm}.log", cwd=REPO)
    _payload = torch.load(_latest, map_location="cpu", weights_only=True)
    assert int(_payload["step"]) == 1000
    _seconds = float(_payload["elapsed_seconds_total"])
    BENCHMARK_RESULTS[_arm] = {
        "seconds_1000": round(_seconds, 2),
        "projected_50k_hours": round(_seconds * 50 / 3600, 3),
    }
assert sum(row["projected_50k_hours"] for row in BENCHMARK_RESULTS.values()) <= 10.5
BENCHMARK_COMPLETE = True
print("BENCHMARK PASSED", BENCHMARK_RESULTS)

# %%
# Cell 7 — train or exact-resume both fixed 50,000-step arms
assert BENCHMARK_COMPLETE
mo.stop(not training_button.value, mo.md(
    "Click **4. Train or exact-resume both 50k arms**. If the session safety "
    "limit stops a run, click the same button again to resume exactly."))


def run_or_resume(arm, config_path):
    latest = CKPT_ROOT / arm / "full/latest.pt"
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if int(payload["step"]) == 50_000:
            assert payload["stop_reason"] == "max_steps"
            return latest
    run_stream([sys.executable, "-u", REPO / "train.py", "--config", config_path],
               LOG_ROOT / f"train_{arm}.log", cwd=REPO, append=latest.exists())
    payload = torch.load(latest, map_location="cpu", weights_only=True)
    assert int(payload["step"]) == 50_000, (
        f"{arm} stopped at step {payload['step']}; click button 4 again")
    assert payload["stop_reason"] == "max_steps"
    return latest


DFS_CHECKPOINT = run_or_resume("dfs", FULL_CONFIGS["dfs"])
OPTIMAL_CHECKPOINT = run_or_resume("optimal", FULL_CONFIGS["optimal"])
TRAINING_COMPLETE = True
print("BOTH 50K TRAINING ARMS COMPLETED")

# %%
# Cell 8 — five-seed VAL geometry evaluation and corrected sensitivity
assert TRAINING_COMPLETE
mo.stop(not evaluation_button.value, mo.md("Click **5. Evaluate fixed VAL protocol**."))

CHECKPOINTS = {"dfs": DFS_CHECKPOINT, "optimal": OPTIMAL_CHECKPOINT}
VAL_REPORTS = {}
for _arm in ("dfs", "optimal"):
    _json_path = RESULT_ROOT / f"{_arm}_val.json"
    _pred_path = RESULT_ROOT / f"{_arm}_predictions.npz"
    if not _json_path.exists() or not _pred_path.exists():
        assert not _json_path.exists() and not _pred_path.exists()
        run_stream([
            sys.executable, "-u", REPO / "ddim_eval.py",
            "--ckpt", CHECKPOINTS[_arm], "--data", DATASETS[_arm],
            "--edge-cache", EDGE_CACHES[_arm], "--split", "val",
            "--steps", "100", "--guidance", "2.0", "--batch", "16",
            "--seeds", ",".join(map(str, SAMPLING_SEEDS)),
            "--bootstrap", "10000", "--bounds", "physical",
            "--precision", "fp32", "--out", _json_path,
            "--save-pred", _pred_path,
        ], LOG_ROOT / f"evaluate_{_arm}.log", cwd=REPO)
    _report = json.loads(_json_path.read_text())
    assert _report["protocol"]["split"] == "val"
    assert _report["protocol"]["seeds"] == SAMPLING_SEEDS
    assert _report["protocol"]["topology_edge_source"] == "canonical edge cache"
    assert _report["protocol"]["topology_edge_cache_sha256"] == EXPECTED_EDGE_HASHES[_arm]
    assert _report["checkpoint"]["sha256"] == sha256_file(CHECKPOINTS[_arm])
    VAL_REPORTS[_arm] = _report

SENSITIVITY_REPORTS = {}
for _arm in ("dfs", "optimal"):
    SENSITIVITY_REPORTS[_arm] = {}
    for _mode in ("joint", "images"):
        _out_dir = RESULT_ROOT / f"sensitivity_{_arm}_{_mode}"
        _out_json = _out_dir / "conditioning_sensitivity.json"
        if not _out_json.exists():
            run_stream([
                sys.executable, "-u", REPO / "cond_sensitivity.py",
                "--repo", REPO, "--data", DATASETS[_arm],
                "--ckpt", f"{_arm}={CHECKPOINTS[_arm]}",
                "--out-dir", _out_dir, "--batch", "16",
                "--seeds", ",".join(map(str, SAMPLING_SEEDS)),
                "--bootstrap", "10000", "--shuffle-mode", _mode,
                "--precision", "fp32",
            ], LOG_ROOT / f"sensitivity_{_arm}_{_mode}.log", cwd=REPO)
        _sensitivity = json.loads(_out_json.read_text())
        assert _sensitivity["summary"]["protocol"]["split"] == "val"
        assert _sensitivity["summary"]["protocol"]["seeds"] == SAMPLING_SEEDS
        SENSITIVITY_REPORTS[_arm][_mode] = _sensitivity
EVALUATION_COMPLETE = True
print("VAL GEOMETRY AND SENSITIVITY COMPLETE; TEST WAS NOT ACCESSED")

# %%
# Cell 9 — preregistered paired decision, visualizations, and downloads
assert EVALUATION_COMPLETE
mo.stop(not package_button.value, mo.md("Click **6. Compare, visualize, and package**."))

METRICS = [
    "chamfer_l2", "hd95_mm", "overlap@1.0mm", "overlap@2.0mm",
    "overlap@5.0mm", "edge_continuity_5x", "broken_edge_fraction_5x",
    "largest_connected_component_fraction_5x", "tree_length_ratio",
    "radius_mae_mm", "radius_rmse_mm", "radius_bias_mm",
    "radius_correlation", "out_of_crop_fraction",
]


def patient_values(report, metric):
    grouped = {}
    for row in report["per_sample"]:
        value = float(row[metric])
        if np.isfinite(value):
            grouped.setdefault(str(row["patient"]), []).append(value)
    return {patient: float(np.mean(values)) for patient, values in grouped.items()}


def paired_delta(metric, repeats=10000):
    first = patient_values(VAL_REPORTS["dfs"], metric)
    second = patient_values(VAL_REPORTS["optimal"], metric)
    patients = sorted(set(first) & set(second))
    assert len(patients) >= 90, f"Too few finite paired patients for {metric}"
    values = np.asarray([second[key] - first[key] for key in patients])
    generator = np.random.default_rng(20260915)
    draws = generator.integers(0, len(values), size=(repeats, len(values)))
    bootstrap = values[draws].mean(axis=1)
    return {
        "metric": metric, "optimal_minus_dfs": float(values.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "patients": int(len(values)),
    }


SUMMARY_ROWS = []
for _arm in ("dfs", "optimal"):
    _row = {"arm": _arm}
    for _metric in METRICS:
        _row[_metric] = float(np.mean(list(patient_values(
            VAL_REPORTS[_arm], _metric).values())))
    for _mode in ("joint", "images"):
        _model = SENSITIVITY_REPORTS[_arm][_mode]["summary"]["models"][_arm]
        _row[f"sensitivity_{_mode}_pct"] = float(_model["primary_sensitivity_pct"])
    SUMMARY_ROWS.append(_row)
SUMMARY_TABLE = pd.DataFrame(SUMMARY_ROWS)
DELTA_ROWS = [paired_delta(metric) for metric in METRICS]
DELTA_TABLE = pd.DataFrame(DELTA_ROWS)

_dfs = SUMMARY_ROWS[0]
_optimal = SUMMARY_ROWS[1]
_lcc_delta = next(row for row in DELTA_ROWS
                  if row["metric"] == "largest_connected_component_fraction_5x")
_tree_progress = (
    (abs(_dfs["tree_length_ratio"] - 1.0)
     - abs(_optimal["tree_length_ratio"] - 1.0))
    / max(abs(_dfs["tree_length_ratio"] - 1.0), 1e-12)
)
ADVANCEMENT_CHECKS = {
    "lcc_improves_beyond_patient_ci": _lcc_delta["ci95_low"] > 0.0,
    "broken_edge_fraction_decreases": (
        _optimal["broken_edge_fraction_5x"] < _dfs["broken_edge_fraction_5x"]),
    "tree_length_moves_at_least_10pct_toward_one": _tree_progress >= 0.10,
    "chamfer_degrades_no_more_than_5pct": (
        _optimal["chamfer_l2"] <= 1.05 * _dfs["chamfer_l2"]),
    "hd95_degrades_no_more_than_5pct": (
        _optimal["hd95_mm"] <= 1.05 * _dfs["hd95_mm"]),
    "radius_mae_degrades_no_more_than_5pct": (
        _optimal["radius_mae_mm"] <= 1.05 * _dfs["radius_mae_mm"]),
    "radius_rmse_degrades_no_more_than_5pct": (
        _optimal["radius_rmse_mm"] <= 1.05 * _dfs["radius_rmse_mm"]),
    "radius_correlation_drop_no_more_than_0.05": (
        _optimal["radius_correlation"] >= _dfs["radius_correlation"] - 0.05),
    "joint_sensitivity_positive_and_at_least_80pct_of_dfs": (
        _optimal["sensitivity_joint_pct"] > 0
        and _optimal["sensitivity_joint_pct"] >= 0.8 * _dfs["sensitivity_joint_pct"]),
    "image_sensitivity_positive_and_at_least_80pct_of_dfs": (
        _optimal["sensitivity_images_pct"] > 0
        and _optimal["sensitivity_images_pct"] >= 0.8 * _dfs["sensitivity_images_pct"]),
}
ADVANCE_TO_200K = bool(all(ADVANCEMENT_CHECKS.values()))

SUMMARY_TABLE.to_csv(RESULT_ROOT / "paired_summary.csv", index=False)
DELTA_TABLE.to_csv(RESULT_ROOT / "paired_patient_deltas.csv", index=False)
(RESULT_ROOT / "advancement_decision.json").write_text(json.dumps({
    "advance_to_200k": ADVANCE_TO_200K,
    "checks": ADVANCEMENT_CHECKS,
    "tree_length_relative_progress_toward_one": float(_tree_progress),
    "training_seed_count": 1,
    "test_accessed": False,
}, indent=2))

for _arm in ("dfs", "optimal"):
    _directory = VIZ_ROOT / _arm
    if len(list(_directory.glob("*.png"))) < 3:
        run_stream([
            sys.executable, "-u", REPO / "visualize_predictions.py",
            "--pred", RESULT_ROOT / f"{_arm}_predictions.npz",
            "--results", RESULT_ROOT / f"{_arm}_val.json",
            "--out-dir", _directory, "--dpi", "200",
        ], LOG_ROOT / f"visualize_{_arm}.log", cwd=REPO)
VIZ_FILES = sorted(VIZ_ROOT.rglob("*.png"))

REPRO_MANIFEST = {
    "repo_commit": REPO_HEAD, "gpu": GPU_INFO,
    "datasets": {
        "dfs_json_hashes": EXPECTED_JSON_HASHES,
        "optimal_archive_sha256": EXPECTED_OPTIMAL_ARCHIVE_HASH,
        "optimal_manifest_sha256": EXPECTED_OPTIMAL_MANIFEST_HASH,
    },
    "edge_cache_sha256": EXPECTED_EDGE_HASHES,
    "checkpoints": {
        arm: {"path": str(CHECKPOINTS[arm]), "sha256": sha256_file(CHECKPOINTS[arm])}
        for arm in ("dfs", "optimal")
    },
    "protocol": {
        "split": "val", "ddim_steps": 100, "guidance": 2.0,
        "sampling_seeds": SAMPLING_SEEDS, "training_seed": TRAINING_SEED,
        "fixed_training_endpoint": 50000, "test_accessed": False,
    },
    "advancement_checks": ADVANCEMENT_CHECKS,
    "advance_to_200k": ADVANCE_TO_200K,
}
MANIFEST_PATH = RUN_ROOT / "reproducibility_manifest.json"
MANIFEST_PATH.write_text(json.dumps(REPRO_MANIFEST, indent=2))

REPORT_ARCHIVE = RUN_ROOT / "optimal_ordering_50k_reproducibility.zip"
with zipfile.ZipFile(REPORT_ARCHIVE, "w", zipfile.ZIP_DEFLATED) as _archive:
    for _source, _name in (
        (CONFIG_ROOT, "configs"), (RESULT_ROOT, "results"),
        (LOG_ROOT, "logs"), (VIZ_ROOT, "visualizations"),
    ):
        for _path in _source.rglob("*"):
            if _path.is_file() and not _path.name.endswith("_predictions.npz"):
                _archive.write(_path, Path(_name) / _path.relative_to(_source))
    _archive.write(MANIFEST_PATH, MANIFEST_PATH.name)
    for _relative in (
        "experiment_protocol.md", "ddim_eval.py", "cond_sensitivity.py",
        "src/coronarycl/models/diffusion.py", "src/coronarycl/sampling.py",
        "src/coronarycl/trainer.py", "src/coronarycl/metrics.py",
        "configs/h384_ordering_50k_dfs.yaml",
        "configs/h384_ordering_50k_optimal.yaml",
    ):
        _archive.write(REPO / _relative, Path("source") / _relative)

_display = [
    mo.md("## Paired 50k VAL results"),
    mo.ui.table(SUMMARY_TABLE.round(5)),
    mo.md("## Paired patient bootstrap: optimal minus DFS"),
    mo.ui.table(DELTA_TABLE.round(5)),
    mo.md(f"## Advance optimal ordering to 200k: **{ADVANCE_TO_200K}**"),
    mo.md(f"```json\n{json.dumps(ADVANCEMENT_CHECKS, indent=2)}\n```"),
]
for _figure in VIZ_FILES:
    _display.extend([mo.md(f"### `{_figure.parent.name}/{_figure.name}`"),
                     mo.image(_figure.read_bytes())])
_display.extend([
    mo.download(REPORT_ARCHIVE.read_bytes(), filename=REPORT_ARCHIVE.name,
                mimetype="application/zip", label="Download reproducibility report"),
    mo.download(DFS_CHECKPOINT.read_bytes(), filename="dfs_50k_latest.pt",
                mimetype="application/octet-stream", label="Download DFS checkpoint"),
    mo.download(OPTIMAL_CHECKPOINT.read_bytes(), filename="optimal_50k_latest.pt",
                mimetype="application/octet-stream", label="Download optimal checkpoint"),
])
mo.vstack(_display)
