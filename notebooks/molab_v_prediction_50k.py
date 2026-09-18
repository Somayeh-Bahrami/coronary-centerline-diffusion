__generated_with = "0.24.0"

# %%
import marimo as mo

# %%
mo.md(r"""
# Paired epsilon-versus-v 50k pilot (VAL only)

This notebook runs a controlled comparison from random weights. The two arms
differ only in prediction parameterization and checkpoint directory. Edge loss
is disabled, evaluation is fixed in advance, and TEST is never accessed.

Upload the validated 1,694-sample dataset ZIP (or extracted directory) through
the Molab sidebar, attach a BF16-capable GPU, and click the buttons in order.
""")

# %%
# Cell 1 — environment, immutable revision, paths, and safety helpers
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
EXPECTED_CODE_COMMIT = "c783b257fd18f947a07fa14cf32367ed7e31927a"
EXPECTED_SAMPLES = 1694
EXPECTED_SPLITS = {"train": 1350, "val": 177, "test": 167}
EXPECTED_DATASET_HASHES = {
    "case_splits_v3.json": "de82889a4c151461d1970d696a9065f9c8ebbc85d1e864d584730ed6d57bab61",
    "norm_stats_v3.json": "ff336f4d9c159a079abdec41797cb7af6dc71c503a6af7d75e87b1743287fa66",
    "pilot_report_v3.json": "f82e3c021ab4a8662323cacfe01b389c2483079bd004f33ea79acd08b9584d78",
}
SAMPLING_SEEDS = [104729, 130363, 155921, 181081, 205019]
TRAINING_SEED = 20260911

# Set this only if automatic dataset discovery is ambiguous.
DATASET_OVERRIDE = ""

WORK_ROOT = Path.cwd().resolve()
REPO_ROOT = WORK_ROOT / f"coronary_{EXPECTED_CODE_COMMIT[:7]}"
RUN_ROOT = WORK_ROOT / "v_prediction_50k_pilot_s20260911"
CONFIG_ROOT = RUN_ROOT / "configs"
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
RESULT_ROOT = RUN_ROOT / "results"
LOG_ROOT = RUN_ROOT / "logs"
VIZ_ROOT = RUN_ROOT / "visualizations"
for _directory in (
    RUN_ROOT, CONFIG_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, LOG_ROOT, VIZ_ROOT,
):
    _directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stream(command, log_path, cwd=None, append=False):
    command = [str(part) for part in command]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        header = f"[{stamp}] {' '.join(command)}\n"
        print(header, end="")
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}); see {log_path}")


assert torch.cuda.is_available(), "Attach the Molab GPU and rerun this cell"
assert torch.cuda.is_bf16_supported(), "The selected GPU must support BF16"
GPU_DESCRIPTION = {
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
}
GPU_DESCRIPTION

# %%
# Cell 2 — explicit controls for every expensive stage
setup_button = mo.ui.run_button(label="1. Verify code, dataset, and tests")
smoke_button = mo.ui.run_button(label="2. Run paired 20-step smoke tests")
benchmark_button = mo.ui.run_button(label="3. Benchmark both prediction types")
training_button = mo.ui.run_button(label="4. Train or exact-resume both 50k arms")
evaluation_button = mo.ui.run_button(label="5. Evaluate both arms on locked VAL")
package_button = mo.ui.run_button(label="6. Compare, visualize, and package")
mo.vstack([
    setup_button,
    smoke_button,
    benchmark_button,
    training_button,
    evaluation_button,
    package_button,
])

# %%
# Cell 3 — clone the pinned code and verify the exact dataset
mo.stop(not setup_button.value, mo.md("Click **1. Verify code, dataset, and tests**."))

if not (REPO_ROOT / ".git").is_dir():
    assert not REPO_ROOT.exists(), f"Incomplete repository path: {REPO_ROOT}"
    run_stream(["git", "clone", REPO_URL, REPO_ROOT], LOG_ROOT / "clone.log")
    run_stream(
        ["git", "checkout", "--detach", EXPECTED_CODE_COMMIT],
        LOG_ROOT / "checkout.log", cwd=REPO_ROOT,
    )

REPO_HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
    capture_output=True, text=True).stdout.strip()
REPO_STATUS = subprocess.run(
    ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
    capture_output=True, text=True).stdout.strip()
assert REPO_HEAD == EXPECTED_CODE_COMMIT
assert not REPO_STATUS, f"Pinned checkout is dirty:\n{REPO_STATUS}"

run_stream([
    sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "scipy", "nibabel", "scikit-image", "tqdm", "matplotlib",
    "pandas", "PyYAML", "pytest",
], LOG_ROOT / "dependencies.log")


def dataset_roots(search_root):
    roots = []
    for stats_path in Path(search_root).rglob("norm_stats_v3.json"):
        candidate = stats_path.parent.resolve()
        if REPO_ROOT == candidate or REPO_ROOT in candidate.parents:
            continue
        if len(list(candidate.glob("*.npz"))) == EXPECTED_SAMPLES:
            roots.append(candidate)
    return sorted(set(roots))


def safely_extract(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None, "Dataset archive is corrupted"
        assert sum(item.file_size for item in archive.infolist()) < 100 * 2**30
        for item in archive.infolist():
            relative = Path(item.filename)
            mode = (item.external_attr >> 16) & 0xFFFF
            assert not relative.is_absolute() and ".." not in relative.parts
            assert not stat.S_ISLNK(mode), f"Symlink is not allowed: {relative}"
            target = (destination / relative).resolve()
            assert target == destination or destination in target.parents
        archive.extractall(destination)


if DATASET_OVERRIDE:
    _dataset_input = Path(DATASET_OVERRIDE).expanduser().resolve()
    assert _dataset_input.exists()
    _dataset_candidates = (
        dataset_roots(_dataset_input) if _dataset_input.is_dir() else [])
    _dataset_archive = _dataset_input if _dataset_input.is_file() else None
else:
    _dataset_candidates = dataset_roots(WORK_ROOT)
    _dataset_archive = None

if not _dataset_candidates:
    if _dataset_archive is None:
        _archives = sorted(
            path.resolve() for path in WORK_ROOT.rglob("*.zip")
            if ("ds" in path.name.lower() or "dataset" in path.name.lower())
            and RUN_ROOT not in path.parents)
        assert len(_archives) == 1, (
            f"Expected one dataset ZIP; found {_archives}. Set DATASET_OVERRIDE.")
        _dataset_archive = _archives[0]
    _extraction_root = RUN_ROOT / f"dataset_{sha256_file(_dataset_archive)[:12]}"
    if not dataset_roots(_extraction_root):
        safely_extract(_dataset_archive, _extraction_root)
    _dataset_candidates = dataset_roots(_extraction_root)

assert len(_dataset_candidates) == 1, (
    f"Expected one dataset root; found {_dataset_candidates}")
DATASET_ROOT = _dataset_candidates[0]
for _name, _expected_hash in EXPECTED_DATASET_HASHES.items():
    assert sha256_file(DATASET_ROOT / _name) == _expected_hash, _name

for _module_name in list(sys.modules):
    if _module_name == "src" or _module_name.startswith("src."):
        del sys.modules[_module_name]
sys.path.insert(0, str(REPO_ROOT))

from src.coronarycl.dataset_v3_1 import list_samples

SPLIT_COUNTS = {
    split: len(list_samples(DATASET_ROOT, split))
    for split in ("train", "val", "test")
}
assert SPLIT_COUNTS == EXPECTED_SPLITS

run_stream(
    [sys.executable, "-m", "pytest", "-q"],
    LOG_ROOT / "pytest.log", cwd=REPO_ROOT)
run_stream(
    [sys.executable, "-m", "src.coronarycl.models.diffusion"],
    LOG_ROOT / "diffusion_self_test.log", cwd=REPO_ROOT)

SETUP_COMPLETE = True
print("Setup passed:", REPO_HEAD, DATASET_ROOT, SPLIT_COUNTS)

# %%
# Cell 4 — materialize immutable full, smoke, and benchmark configurations
assert SETUP_COMPLETE


def write_immutable_config(config, path):
    path = Path(path)
    rendered = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        assert yaml.safe_load(path.read_text()) == config, f"Config changed: {path}"
    else:
        path.write_text(rendered)
    return path


def resolved_arm_config(prediction_type):
    template = REPO_ROOT / "configs" / f"h384_{prediction_type}_50k.yaml"
    config = yaml.safe_load(template.read_text())
    assert config["train"]["prediction_type"] == prediction_type
    config["data"]["packaged_dir"] = str(DATASET_ROOT)
    config["train"]["checkpoint_dir"] = str(
        CHECKPOINT_ROOT / prediction_type / "full")
    return config


_epsilon_full = resolved_arm_config("epsilon")
_velocity_full = resolved_arm_config("v")
_normalized_epsilon = deepcopy(_epsilon_full)
_normalized_velocity = deepcopy(_velocity_full)
for _config in (_normalized_epsilon, _normalized_velocity):
    _config["train"].pop("prediction_type")
    _config["train"].pop("checkpoint_dir")
assert _normalized_epsilon == _normalized_velocity

FULL_CONFIGS = {
    "epsilon": write_immutable_config(
        _epsilon_full, CONFIG_ROOT / "epsilon_50k.yaml"),
    "v": write_immutable_config(
        _velocity_full, CONFIG_ROOT / "v_50k.yaml"),
}
CONFIGS_PREPARED = True
print({name: str(path) for name, path in FULL_CONFIGS.items()})

# %%
# Cell 5 — paired 20-step end-to-end smoke tests
assert CONFIGS_PREPARED
mo.stop(not smoke_button.value, mo.md("Click **2. Run paired 20-step smoke tests**."))

SMOKE_RESULTS = {}
for _prediction_type, _full_path in FULL_CONFIGS.items():
    _config = yaml.safe_load(_full_path.read_text())
    _directory = CHECKPOINT_ROOT / _prediction_type / "smoke"
    _config["train"].update({
        "checkpoint_dir": str(_directory),
        "max_steps": 20,
        "warmup_steps": 2,
        "val_every": 10,
        "checkpoint_every": 10,
        "milestone_every": 0,
        "max_hours": 0.5,
    })
    _path = write_immutable_config(
        _config, CONFIG_ROOT / f"{_prediction_type}_smoke.yaml")
    _latest = _directory / "latest.pt"
    if not _latest.exists():
        run_stream(
            [sys.executable, "-u", REPO_ROOT / "train.py", "--config", _path],
            LOG_ROOT / f"smoke_{_prediction_type}.log", cwd=REPO_ROOT)
    _payload = torch.load(_latest, map_location="cpu", weights_only=True)
    assert int(_payload["step"]) == 20
    assert _payload["stop_reason"] == "max_steps"
    assert _payload["prediction_type"] == _prediction_type
    assert _payload["run_signature"]["prediction_type"] == _prediction_type
    _expected_nonempty = "eps_loss" if _prediction_type == "epsilon" else "v_loss"
    assert _payload["history"][_expected_nonempty][-1] is not None
    SMOKE_RESULTS[_prediction_type] = {
        "prediction_loss": _payload["history"]["prediction_loss"][-1],
        "val_loss": _payload["history"]["val_loss"][-1],
    }
SMOKE_COMPLETE = True
SMOKE_RESULTS

# %%
# Cell 6 — paired 1,000-step speed benchmark and runtime gate
assert SMOKE_COMPLETE
mo.stop(not benchmark_button.value, mo.md("Click **3. Benchmark both prediction types**."))

BENCHMARK_RESULTS = {}
for _prediction_type, _full_path in FULL_CONFIGS.items():
    _config = yaml.safe_load(_full_path.read_text())
    _directory = CHECKPOINT_ROOT / _prediction_type / "benchmark"
    _config["train"].update({
        "checkpoint_dir": str(_directory),
        "max_steps": 1000,
        "warmup_steps": 10,
        "val_every": 1000,
        "checkpoint_every": 1000,
        "milestone_every": 0,
        "max_hours": 0.5,
    })
    _path = write_immutable_config(
        _config, CONFIG_ROOT / f"{_prediction_type}_benchmark.yaml")
    _latest = _directory / "latest.pt"
    if not _latest.exists():
        run_stream(
            [sys.executable, "-u", REPO_ROOT / "train.py", "--config", _path],
            LOG_ROOT / f"benchmark_{_prediction_type}.log", cwd=REPO_ROOT)
    _payload = torch.load(_latest, map_location="cpu", weights_only=True)
    assert int(_payload["step"]) == 1000
    _seconds = float(_payload["elapsed_seconds_total"])
    _projected_hours = _seconds * 50 / 3600
    BENCHMARK_RESULTS[_prediction_type] = {
        "seconds_for_1000": round(_seconds, 2),
        "projected_50k_hours": round(_projected_hours, 3),
    }

assert max(
    row["projected_50k_hours"] for row in BENCHMARK_RESULTS.values()
) <= 4.65, "One arm is projected to exceed the 4.65-hour safety gate"
assert sum(
    row["projected_50k_hours"] for row in BENCHMARK_RESULTS.values()
) <= 10.5, "Both sequential arms may exceed the Molab session budget"
BENCHMARK_COMPLETE = True
BENCHMARK_RESULTS

# %%
# Cell 7 — train or exact-resume both fixed 50,000-step arms
assert BENCHMARK_COMPLETE
mo.stop(not training_button.value, mo.md(
    "Click **4. Train or exact-resume both 50k arms**. If a runtime safety "
    "stop occurs, click the same button again to resume exactly."))


def run_or_resume_arm(prediction_type, config_path):
    latest = CHECKPOINT_ROOT / prediction_type / "full" / "latest.pt"
    if latest.exists():
        existing = torch.load(latest, map_location="cpu", weights_only=True)
        if int(existing["step"]) == 50_000:
            assert existing["stop_reason"] == "max_steps"
            assert existing["prediction_type"] == prediction_type
            return latest
    run_stream(
        [sys.executable, "-u", REPO_ROOT / "train.py", "--config", config_path],
        LOG_ROOT / f"train_{prediction_type}.log", cwd=REPO_ROOT,
        append=latest.exists())
    completed = torch.load(latest, map_location="cpu", weights_only=True)
    assert int(completed["step"]) == 50_000, (
        f"{prediction_type} stopped at {completed['step']}; click button 4 again")
    assert completed["stop_reason"] == "max_steps"
    assert completed["prediction_type"] == prediction_type
    return latest


EPSILON_CHECKPOINT = run_or_resume_arm("epsilon", FULL_CONFIGS["epsilon"])
VELOCITY_CHECKPOINT = run_or_resume_arm("v", FULL_CONFIGS["v"])
TRAINING_COMPLETE = True
print("Both fixed-endpoint 50k arms completed")

# %%
# Cell 8 — locked five-seed VAL evaluation at DDIM=100 and guidance=2
assert TRAINING_COMPLETE
mo.stop(not evaluation_button.value, mo.md("Click **5. Evaluate both arms on locked VAL**."))

_evaluation_jobs = {
    "epsilon": EPSILON_CHECKPOINT,
    "v": VELOCITY_CHECKPOINT,
}
VAL_REPORTS = {}
for _prediction_type, _checkpoint in _evaluation_jobs.items():
    _output_json = RESULT_ROOT / f"{_prediction_type}_50k_val.json"
    _output_predictions = RESULT_ROOT / f"{_prediction_type}_50k_predictions.npz"
    if not _output_json.exists() or not _output_predictions.exists():
        assert not _output_json.exists() and not _output_predictions.exists(), (
            "Partial evaluation outputs found; preserve them and use a new RUN_ROOT")
        run_stream([
            sys.executable, "-u", REPO_ROOT / "ddim_eval.py",
            "--ckpt", _checkpoint,
            "--data", DATASET_ROOT,
            "--split", "val",
            "--steps", "100",
            "--guidance", "2.0",
            "--batch", "16",
            "--seeds", ",".join(map(str, SAMPLING_SEEDS)),
            "--bootstrap", "10000",
            "--bounds", "physical",
            "--precision", "fp32",
            "--out", _output_json,
            "--save-pred", _output_predictions,
        ], LOG_ROOT / f"evaluate_{_prediction_type}.log", cwd=REPO_ROOT)
    _report = json.loads(_output_json.read_text())
    assert _report["protocol"]["split"] == "val"
    assert _report["protocol"]["seeds"] == SAMPLING_SEEDS
    assert _report["protocol"]["prediction_type"] == _prediction_type
    assert _report["checkpoint"]["prediction_type"] == _prediction_type
    assert _report["checkpoint"]["sha256"] == sha256_file(_checkpoint)
    VAL_REPORTS[_prediction_type] = _report
VAL_COMPLETE = True
print("VAL evaluation completed; TEST was not accessed")

# %%
# Cell 9 — paired patient-level comparison and graph-valid visualizations
assert VAL_COMPLETE
mo.stop(not package_button.value, mo.md("Click **6. Compare, visualize, and package**."))

_metric_keys = [
    "chamfer_l2",
    "hd95_mm",
    "overlap@2.0mm",
    "edge_continuity_5x",
    "largest_connected_component_fraction_5x",
    "tree_length_ratio",
    "out_of_crop_fraction",
]
_comparison_rows = []
for _prediction_type in ("epsilon", "v"):
    _patient_summary = VAL_REPORTS[_prediction_type]["patient_bootstrap"]
    _comparison_rows.append({
        "prediction": _prediction_type,
        **{
            key: float(_patient_summary[key]["equal_patient_mean"])
            for key in _metric_keys
        },
    })
COMPARISON_TABLE = pd.DataFrame(_comparison_rows).round(5)


def paired_patient_deltas(first_report, second_report, metric, repeats=10000):
    first = {row["sample"]: row for row in first_report["per_sample"]}
    second = {row["sample"]: row for row in second_report["per_sample"]}
    assert first.keys() == second.keys()
    by_patient = {}
    for sample_id in sorted(first):
        patient = str(first[sample_id]["patient"])
        assert patient == str(second[sample_id]["patient"])
        by_patient.setdefault(patient, []).append(
            float(second[sample_id][metric]) - float(first[sample_id][metric]))
    patient_delta = np.asarray([
        np.mean(by_patient[patient]) for patient in sorted(by_patient)
    ])
    generator = np.random.default_rng(20260911)
    indices = generator.integers(
        0, len(patient_delta), size=(repeats, len(patient_delta)))
    bootstrap = patient_delta[indices].mean(axis=1)
    return {
        "metric": metric,
        "v_minus_epsilon": float(patient_delta.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "patients": int(len(patient_delta)),
    }


DELTA_ROWS = [
    paired_patient_deltas(
        VAL_REPORTS["epsilon"], VAL_REPORTS["v"], metric)
    for metric in _metric_keys
]
DELTA_TABLE = pd.DataFrame(DELTA_ROWS).round(5)

_epsilon_summary = {
    key: VAL_REPORTS["epsilon"]["patient_bootstrap"][key]["equal_patient_mean"]
    for key in _metric_keys
}
_velocity_summary = {
    key: VAL_REPORTS["v"]["patient_bootstrap"][key]["equal_patient_mean"]
    for key in _metric_keys
}
ADVANCE_TO_FULL = bool(
    _velocity_summary["largest_connected_component_fraction_5x"]
    >= _epsilon_summary["largest_connected_component_fraction_5x"] + 0.02
    and abs(_velocity_summary["tree_length_ratio"] - 1.0)
    < abs(_epsilon_summary["tree_length_ratio"] - 1.0)
    and _velocity_summary["chamfer_l2"]
    <= 1.10 * _epsilon_summary["chamfer_l2"]
)

_ordered = sorted(
    VAL_REPORTS["epsilon"]["per_sample"], key=lambda row: row["chamfer_l2"])
_indices = [round((len(_ordered) - 1) * q) for q in (0.1, 0.5, 0.9)]
_samples = [_ordered[index]["sample"] for index in _indices]
for _prediction_type in ("epsilon", "v"):
    _viz_directory = VIZ_ROOT / _prediction_type
    _existing = sorted(_viz_directory.glob("*.png"))
    if len(_existing) < 3:
        _command = [
            sys.executable, "-u", REPO_ROOT / "visualize_predictions.py",
            "--pred", RESULT_ROOT / f"{_prediction_type}_50k_predictions.npz",
            "--results", RESULT_ROOT / f"{_prediction_type}_50k_val.json",
            "--out-dir", _viz_directory,
            "--dpi", "200",
        ]
        for _sample in _samples:
            _command.extend(["--sample", _sample])
        run_stream(
            _command, LOG_ROOT / f"visualize_{_prediction_type}.log",
            cwd=REPO_ROOT)
VIZ_FILES = sorted(VIZ_ROOT.rglob("*.png"))
assert len(VIZ_FILES) == 6
ANALYSIS_COMPLETE = True

COMPARISON_TABLE.to_csv(RESULT_ROOT / "paired_summary.csv", index=False)
DELTA_TABLE.to_csv(RESULT_ROOT / "paired_patient_deltas.csv", index=False)
(RESULT_ROOT / "paired_decision.json").write_text(json.dumps({
    "advance_to_full": ADVANCE_TO_FULL,
    "rule": {
        "lcc_v_minus_epsilon_min": 0.02,
        "tree_length_ratio_must_move_toward_one": True,
        "maximum_chamfer_relative_degradation": 0.10,
    },
    "aggregation": "equal patient mean",
    "training_seed_count": 1,
}, indent=2))

_display_items = [
    mo.md("## Fixed-endpoint VAL comparison"),
    mo.ui.table(COMPARISON_TABLE),
    mo.md(
        "## Paired patient bootstrap: v minus epsilon\n\n"
        "Positive favors v for overlap, continuity, and LCC. Negative favors "
        "v for Chamfer and HD95. Tree-length ratio should move toward 1."
    ),
    mo.ui.table(DELTA_TABLE),
    mo.md(f"## Advance to full multi-seed study: **{ADVANCE_TO_FULL}**"),
]
for _figure in VIZ_FILES:
    _display_items.extend([mo.md(f"### `{_figure.parent.name}/{_figure.name}`"),
                           mo.image(_figure.read_bytes())])
mo.vstack(_display_items)

# %%
# Cell 10 — reproducibility archive and checkpoint downloads
assert ANALYSIS_COMPLETE

REPRO_MANIFEST = {
    "repo_commit": REPO_HEAD,
    "gpu": GPU_DESCRIPTION,
    "dataset_json_sha256": {
        name: sha256_file(DATASET_ROOT / name)
        for name in EXPECTED_DATASET_HASHES
    },
    "checkpoints": {
        "epsilon_50k": {
            "path": str(EPSILON_CHECKPOINT),
            "sha256": sha256_file(EPSILON_CHECKPOINT),
        },
        "v_50k": {
            "path": str(VELOCITY_CHECKPOINT),
            "sha256": sha256_file(VELOCITY_CHECKPOINT),
        },
    },
    "protocol": {
        "split": "val",
        "ddim_steps": 100,
        "guidance": 2.0,
        "sampling_seeds": SAMPLING_SEEDS,
        "training_seed": TRAINING_SEED,
        "precision": "fp32",
        "bounds": "physical",
        "fixed_training_endpoint": 50000,
        "test_accessed": False,
    },
    "advance_to_full": ADVANCE_TO_FULL,
    "limitation": (
        "This pilot uses one training seed; patient bootstrap intervals do not "
        "measure variability across training seeds."
    ),
}
MANIFEST_PATH = RUN_ROOT / "reproducibility_manifest.json"
MANIFEST_PATH.write_text(json.dumps(REPRO_MANIFEST, indent=2))

REPORT_ARCHIVE = RUN_ROOT / "v_prediction_50k_reproducibility.zip"
with zipfile.ZipFile(REPORT_ARCHIVE, "w", zipfile.ZIP_DEFLATED) as _archive:
    for _source, _archive_root in (
        (CONFIG_ROOT, "configs"),
        (RESULT_ROOT, "results"),
        (LOG_ROOT, "logs"),
        (VIZ_ROOT, "visualizations"),
    ):
        for _path in _source.rglob("*"):
            if _path.is_file() and not _path.name.endswith("_predictions.npz"):
                _archive.write(
                    _path, Path(_archive_root) / _path.relative_to(_source))
    _archive.write(MANIFEST_PATH, MANIFEST_PATH.name)
    for _relative in (
        "src/coronarycl/models/diffusion.py",
        "src/coronarycl/prediction.py",
        "src/coronarycl/sampling.py",
        "src/coronarycl/trainer.py",
        "ddim_eval.py",
        "visualize_predictions.py",
        "configs/h384_epsilon_50k.yaml",
        "configs/h384_v_50k.yaml",
    ):
        _archive.write(REPO_ROOT / _relative, Path("source") / _relative)

mo.vstack([
    mo.md(
        "Download the report and both checkpoints. Keep TEST locked. Do not "
        "start a 200k run until this VAL report has been reviewed."
    ),
    mo.download(
        data=REPORT_ARCHIVE.read_bytes(),
        filename=REPORT_ARCHIVE.name,
        mimetype="application/zip",
        label="Download reproducibility report",
    ),
    mo.download(
        data=EPSILON_CHECKPOINT.read_bytes(),
        filename="epsilon_50k_latest.pt",
        mimetype="application/octet-stream",
        label="Download epsilon checkpoint",
    ),
    mo.download(
        data=VELOCITY_CHECKPOINT.read_bytes(),
        filename="v_50k_latest.pt",
        mimetype="application/octet-stream",
        label="Download v checkpoint",
    ),
])
