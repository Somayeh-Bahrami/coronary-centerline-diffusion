__generated_with = "0.24.0"

# %%
import marimo as mo

# %%
mo.md(r"""
# Group-balanced edge-coherence reproduction (VAL only)

This notebook reproduces the paired control/grouped fine-tuning experiment
from the frozen h384 step-150,000 checkpoint. It never evaluates TEST.

Upload the 1,694-sample dataset ZIP (or an extracted dataset directory) and
the reviewed `step_150000.pt` through the Molab sidebar. Then use the buttons
in order. Every expensive stage is button-gated and reuses verified completed
artifacts instead of overwriting them.
""")

# %%
# Cell 1 — environment, immutable code revision, and shared paths
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_URL = "https://github.com/Somayeh-Bahrami/coronary-centerline-diffusion"
EXPECTED_CODE_COMMIT = "31ffaf26e9e7d8e46ad61ace07c1a58afea1c0eb"
EXPECTED_BASELINE_SHA256 = (
    "d16ba2d76b026864ab2134fbdfae1f1e3c8588b28e3da355769c3623e80aede1"
)
EXPECTED_SPLITS = {"train": 1350, "val": 177, "test": 167}
SAMPLING_SEEDS = [104729, 130363, 155921, 181081, 205019]

# Set either override only when automatic discovery is ambiguous.
DATASET_OVERRIDE = ""
BASELINE_CHECKPOINT_OVERRIDE = ""

WORK_ROOT = Path.cwd().resolve()
REPO_ROOT = WORK_ROOT / f"coronary_{EXPECTED_CODE_COMMIT[:7]}"
RUN_ROOT = WORK_ROOT / "grouped_edge_repro_v1"
CONFIG_ROOT = RUN_ROOT / "configs"
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
RESULT_ROOT = RUN_ROOT / "results"
LOG_ROOT = RUN_ROOT / "logs"
EDGE_CACHE_ROOT = RUN_ROOT / "edge_cache"
for _directory in (
    RUN_ROOT, CONFIG_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, LOG_ROOT,
    EDGE_CACHE_ROOT,
):
    _directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stream(command, log_path, cwd=None):
    command = [str(part) for part in command]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
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
}
GPU_DESCRIPTION

# %%
# Cell 2 — explicit controls; no expensive stage runs without its button
setup_button = mo.ui.run_button(label="1. Verify inputs and prepare experiment")
smoke_button = mo.ui.run_button(label="2. Run paired 20-step smoke tests")
training_button = mo.ui.run_button(label="3. Run or resume both 30k arms")
evaluation_button = mo.ui.run_button(label="4. Evaluate frozen checkpoints on VAL")
diagnostic_button = mo.ui.run_button(label="5. Run focused edge diagnostics")
package_button = mo.ui.run_button(label="6. Build reproducibility report")
mo.vstack([
    setup_button,
    smoke_button,
    training_button,
    evaluation_button,
    diagnostic_button,
    package_button,
])

# %%
# Cell 3 — clone the pinned revision and locate immutable inputs
mo.stop(not setup_button.value, mo.md("Click **1. Verify inputs and prepare experiment**."))

if not (REPO_ROOT / ".git").is_dir():
    assert not REPO_ROOT.exists(), f"Incomplete repository path: {REPO_ROOT}"
    run_stream(
        ["git", "clone", REPO_URL, REPO_ROOT],
        LOG_ROOT / "clone.log",
    )
    run_stream(
        ["git", "checkout", "--detach", EXPECTED_CODE_COMMIT],
        LOG_ROOT / "checkout.log",
        cwd=REPO_ROOT,
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
        if len(list(candidate.glob("*.npz"))) == 1694:
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
    _dataset_candidates = dataset_roots(_dataset_input) if _dataset_input.is_dir() else []
    _dataset_archive = _dataset_input if _dataset_input.is_file() else None
else:
    _dataset_candidates = dataset_roots(WORK_ROOT)
    _dataset_archive = None

if not _dataset_candidates:
    if _dataset_archive is None:
        _archives = sorted(
            _path.resolve() for _path in WORK_ROOT.rglob("*.zip")
            if "ds" in _path.name.lower() and RUN_ROOT not in _path.parents)
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

if BASELINE_CHECKPOINT_OVERRIDE:
    _checkpoint_candidates = [Path(BASELINE_CHECKPOINT_OVERRIDE).expanduser().resolve()]
else:
    _checkpoint_candidates = sorted(WORK_ROOT.rglob("step_150000.pt"))
BASELINE_CHECKPOINTS = [
    _path for _path in _checkpoint_candidates
    if _path.is_file() and sha256_file(_path) == EXPECTED_BASELINE_SHA256
]
assert len(BASELINE_CHECKPOINTS) == 1, (
    "Upload the reviewed step_150000.pt or set BASELINE_CHECKPOINT_OVERRIDE")
BASELINE_CHECKPOINT = BASELINE_CHECKPOINTS[0]

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
INPUTS_VERIFIED = True
print("Inputs verified:", DATASET_ROOT, BASELINE_CHECKPOINT, SPLIT_COUNTS)

# %%
# Cell 4 — tests, full edge-cache verification, and resolved configurations
assert INPUTS_VERIFIED
run_stream(
    [sys.executable, "-m", "pytest", "-q"],
    LOG_ROOT / "pytest.log", cwd=REPO_ROOT)
run_stream(
    [sys.executable, "-m", "src.coronarycl.models.diffusion"],
    LOG_ROOT / "diffusion_self_test.log", cwd=REPO_ROOT)

run_stream([
    sys.executable, "-u", REPO_ROOT / "scripts/build_edge_cache.py",
    "--data", DATASET_ROOT, "--out", EDGE_CACHE_ROOT,
    "--splits", "train",
], LOG_ROOT / "build_edge_cache.log", cwd=REPO_ROOT)

from src.coronarycl.edge_coherence import load_edge_cache
from src.coronarycl.metrics import topology_tree_edges

TRAIN_IDS = list_samples(DATASET_ROOT, "train")
EDGE_MAP, EDGE_METADATA = load_edge_cache(
    EDGE_CACHE_ROOT, DATASET_ROOT, required_ids=TRAIN_IDS)
for _sample_id in TRAIN_IDS:
    with np.load(DATASET_ROOT / f"{_sample_id}.npz", allow_pickle=False) as _archive:
        _valid = _archive["centerline_mask"].astype(bool)
        _xyz = _archive["centerline"][_valid, :3]
        _iso_mm = float(_archive["iso_mm"])
    _expected_edges = topology_tree_edges(_xyz, _iso_mm)
    assert np.array_equal(EDGE_MAP[_sample_id], _expected_edges), _sample_id


def resolved_config(template_name, arm_name):
    config = yaml.safe_load((REPO_ROOT / "configs" / template_name).read_text())
    config["data"]["packaged_dir"] = str(DATASET_ROOT)
    config["train"]["init_checkpoint"] = str(BASELINE_CHECKPOINT)
    config["train"]["checkpoint_dir"] = str(CHECKPOINT_ROOT / arm_name)
    if arm_name == "grouped":
        config["train"]["edge_cache"] = str(EDGE_CACHE_ROOT)
    path = CONFIG_ROOT / f"{arm_name}.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        assert yaml.safe_load(path.read_text()) == config, f"Config changed: {path}"
    else:
        path.write_text(rendered)
    return path


CONTROL_CONFIG = resolved_config("h384_ft_control_30k.yaml", "control")
GROUPED_CONFIG = resolved_config("h384_ft_grouped_30k.yaml", "grouped")
PREPARED = True
print("Preparation passed; all", len(TRAIN_IDS), "TRAIN edge lists match metrics.py")

# %%
# Cell 5 — paired 20-step integration smoke tests
assert PREPARED
mo.stop(not smoke_button.value, mo.md("Click **2. Run paired 20-step smoke tests**."))

SMOKE_RESULTS = []
for _arm_name, _config_path in (("control", CONTROL_CONFIG), ("grouped", GROUPED_CONFIG)):
    _config = yaml.safe_load(_config_path.read_text())
    _smoke_directory = CHECKPOINT_ROOT / "smoke" / _arm_name
    _config["train"]["checkpoint_dir"] = str(_smoke_directory)
    _config["train"]["max_steps"] = 20
    _config["train"]["warmup_steps"] = 2
    _config["train"]["val_every"] = 10
    _config["train"]["checkpoint_every"] = 10
    _config["train"]["milestone_every"] = 0
    _smoke_config = CONFIG_ROOT / f"smoke_{_arm_name}.yaml"
    _smoke_config.write_text(yaml.safe_dump(_config, sort_keys=False))
    _latest = _smoke_directory / "latest.pt"
    if not _latest.exists():
        run_stream(
            [sys.executable, "-u", REPO_ROOT / "train.py", "--config", _smoke_config],
            LOG_ROOT / f"smoke_{_arm_name}.log", cwd=REPO_ROOT)
    _payload = torch.load(_latest, map_location="cpu", weights_only=True)
    assert _payload["step"] == 20 and _payload["stop_reason"] == "max_steps"
    if _arm_name == "grouped":
        assert _payload["run_signature"]["coh_kwargs"]["group_balanced"] is True
        assert _payload["history"]["coh_loss"][-1] > 0
    SMOKE_RESULTS.append({
        "arm": _arm_name,
        "val_loss": _payload["history"]["val_loss"][-1],
        "coh_loss": _payload["history"]["coh_loss"][-1],
    })
SMOKE_PASSED = True
SMOKE_RESULTS

# %%
# Cell 6 — paired 30,000-step fine-tuning with exact session resume
assert SMOKE_PASSED
mo.stop(not training_button.value, mo.md("Click **3. Run or resume both 30k arms**."))


def run_full_arm(arm_name, base_config):
    config = yaml.safe_load(Path(base_config).read_text())
    latest = CHECKPOINT_ROOT / arm_name / "latest.pt"
    run_config = Path(base_config)
    if latest.exists():
        existing = torch.load(latest, map_location="cpu", weights_only=True)
        if int(existing["step"]) == 30000 and existing["stop_reason"] == "max_steps":
            return latest
        config["train"]["resume_mode"] = "required"
        config["train"]["init_checkpoint"] = None
        run_config = CONFIG_ROOT / f"resume_{arm_name}.yaml"
        run_config.write_text(yaml.safe_dump(config, sort_keys=False))
    run_stream(
        [sys.executable, "-u", REPO_ROOT / "train.py", "--config", run_config],
        LOG_ROOT / f"train_{arm_name}.log", cwd=REPO_ROOT)
    completed = torch.load(latest, map_location="cpu", weights_only=True)
    assert int(completed["step"]) == 30000
    assert completed["stop_reason"] == "max_steps"
    return latest


CONTROL_CHECKPOINT = run_full_arm("control", CONTROL_CONFIG)
GROUPED_CHECKPOINT = run_full_arm("grouped", GROUPED_CONFIG)
TRAINING_COMPLETE = True
print("Both fine-tuning arms completed")

# %%
# Cell 7 — fixed five-seed VAL evaluation for baseline, control, and grouped
assert TRAINING_COMPLETE
mo.stop(not evaluation_button.value, mo.md("Click **4. Evaluate frozen checkpoints on VAL**."))

EVALUATION_JOBS = {
    "baseline_150k": BASELINE_CHECKPOINT,
    "control_30k": CONTROL_CHECKPOINT,
    "grouped_30k": GROUPED_CHECKPOINT,
}
EVALUATION_REPORTS = {}
for _label, _checkpoint_path in EVALUATION_JOBS.items():
    _output_json = RESULT_ROOT / f"{_label}.json"
    _output_predictions = RESULT_ROOT / f"{_label}_predictions.npz"
    if not _output_json.exists() or not _output_predictions.exists():
        assert not _output_json.exists() and not _output_predictions.exists()
        run_stream([
            sys.executable, "-u", REPO_ROOT / "ddim_eval.py",
            "--ckpt", _checkpoint_path,
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
        ], LOG_ROOT / f"evaluate_{_label}.log", cwd=REPO_ROOT)
    _report = json.loads(_output_json.read_text())
    assert _report["protocol"]["split"] == "val"
    assert _report["protocol"]["seeds"] == SAMPLING_SEEDS
    assert _report["checkpoint"]["sha256"] == sha256_file(_checkpoint_path)
    EVALUATION_REPORTS[_label] = _report
VAL_EVALUATION_COMPLETE = True
{
    _label: {
        "Chamfer_mm": _report["summary"]["all"]["chamfer_l2"],
        "HD95_mm": _report["summary"]["all"]["hd95_mm"],
        "LCC": _report["summary"]["all"]["largest_connected_component_fraction_5x"],
        "length_ratio": _report["summary"]["all"]["tree_length_ratio"],
    }
    for _label, _report in EVALUATION_REPORTS.items()
}

# %%
# Cell 8 — focused noised-GT and real-DDIM non-consecutive-edge diagnostics
assert VAL_EVALUATION_COMPLETE
mo.stop(not diagnostic_button.value, mo.md("Click **5. Run focused edge diagnostics**."))

FOCUSED_RESULT_ROOT = RESULT_ROOT / "focused_edge_diagnostic"
_focused_noised = FOCUSED_RESULT_ROOT / "grouped_noised_gt.json"
_focused_trajectory = FOCUSED_RESULT_ROOT / "grouped_trajectory.json"
if _focused_noised.exists() or _focused_trajectory.exists():
    assert _focused_noised.is_file() and _focused_trajectory.is_file(), (
        "A focused diagnostic stopped partially. Preserve that directory and "
        "choose a new RUN_ROOT before rerunning.")
else:
    run_stream([
        sys.executable, "-u", REPO_ROOT / "scripts/diagnose_grouped_focused.py",
        "--repo", REPO_ROOT,
        "--data", DATASET_ROOT,
        "--baseline", BASELINE_CHECKPOINT,
        "--control", CONTROL_CHECKPOINT,
        "--grouped", GROUPED_CHECKPOINT,
        "--out-dir", FOCUSED_RESULT_ROOT,
        "--batch", "8",
    ], LOG_ROOT / "focused_edge_diagnostic.log", cwd=REPO_ROOT)
for _result_name in ("grouped_noised_gt.json", "grouped_trajectory.json"):
    assert (FOCUSED_RESULT_ROOT / _result_name).is_file()
DIAGNOSTIC_COMPLETE = True
print("Focused diagnostics completed on VAL only")

# %%
# Cell 9 — compact reproducibility package (checkpoints stay in the sidebar)
assert DIAGNOSTIC_COMPLETE
mo.stop(not package_button.value, mo.md("Click **6. Build reproducibility report**."))

REPRO_MANIFEST = {
    "repo_commit": REPO_HEAD,
    "dataset_json_sha256": {
        name: sha256_file(DATASET_ROOT / name)
        for name in ("case_splits_v3.json", "norm_stats_v3.json", "pilot_report_v3.json")
    },
    "checkpoints": {
        _label: {"path": str(_path), "sha256": sha256_file(_path)}
        for _label, _path in EVALUATION_JOBS.items()
    },
    "protocol": {
        "split": "val",
        "ddim_steps": 100,
        "guidance": 2.0,
        "seeds": SAMPLING_SEEDS,
        "precision": "fp32",
        "bounds": "physical",
        "test_accessed": False,
    },
}
MANIFEST_PATH = RUN_ROOT / "reproducibility_manifest.json"
MANIFEST_PATH.write_text(json.dumps(REPRO_MANIFEST, indent=2))

REPORT_ARCHIVE = RUN_ROOT / "grouped_edge_reproducibility.zip"
with zipfile.ZipFile(REPORT_ARCHIVE, "w", zipfile.ZIP_DEFLATED) as _report_archive:
    for _source, _archive_root in (
        (CONFIG_ROOT, "configs"),
        (RESULT_ROOT, "results"),
        (LOG_ROOT, "logs"),
    ):
        for _path in _source.rglob("*"):
            if _path.is_file() and not _path.name.endswith("_predictions.npz"):
                _report_archive.write(
                    _path, Path(_archive_root) / _path.relative_to(_source))
    _report_archive.write(MANIFEST_PATH, MANIFEST_PATH.name)
    for _relative in (
        "src/coronarycl/models/diffusion.py",
        "src/coronarycl/sampling.py",
        "src/coronarycl/trainer.py",
        "src/coronarycl/edge_coherence.py",
        "src/coronarycl/edge_coherence_grouped.py",
        "ddim_eval.py",
        "scripts/diagnose_grouped_focused.py",
    ):
        _report_archive.write(
            REPO_ROOT / _relative, Path("source") / _relative)

mo.vstack([
    mo.md(
        "The report contains source, resolved configs, logs, metrics, diagnostics, "
        "and hashes. Download the three checkpoint files separately from the sidebar."
    ),
    mo.download(
        data=REPORT_ARCHIVE.read_bytes(),
        filename=REPORT_ARCHIVE.name,
        mimetype="application/zip",
        label="Download reproducibility report",
    ),
])
