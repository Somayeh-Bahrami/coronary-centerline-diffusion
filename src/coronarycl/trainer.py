"""Padding-aware diffusion training with deterministic validation and safe resume.

Validation loss is a noise-prediction diagnostic. Checkpoint and sampler
selection must still use physical geometry metrics on VAL; TEST stays locked
until the complete protocol is frozen.

``resume_mode`` is explicit: ``fresh`` refuses an existing ``latest.pt``;
``auto`` resumes it when present; and ``required`` insists it is present.
Both best/latest checkpoints contain full training state. Lightweight milestone
checkpoints retain weights for later VAL geometry selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .config import resolve_device
from .dataset_v3_1 import CoronaryCenterlineDatasetV31, list_samples
from .models.diffusion import CenterlineDenoiser

EVAL_TIMESTEPS = [0, 250, 500, 750, 999]
EVAL_SEED = 12345
CHECKPOINT_VERSION = 3


class NoiseScheduler:
    """Standard linear-beta DDPM noise schedule (Ho et al., 2020)."""

    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02,
                 device="cpu"):
        self.n_steps = int(n_steps)
        self.betas = torch.linspace(
            beta_start, beta_end, self.n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, timesteps):
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        noise = torch.randn_like(x0)
        noisy = (torch.sqrt(alpha_bar) * x0
                 + torch.sqrt(1.0 - alpha_bar) * noise)
        return noisy, noise


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Map every optimizer step to a deterministic epoch-shuffled batch."""

    def __init__(self, n_samples, batch_size, first_step, max_steps, seed):
        if n_samples <= 0 or batch_size <= 0:
            raise ValueError("n_samples and batch_size must be positive")
        if not 0 <= first_step <= max_steps:
            raise ValueError("require 0 <= first_step <= max_steps")
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        self.first_step = int(first_step)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.batches_per_epoch = math.ceil(self.n_samples / self.batch_size)

    def __len__(self):
        return self.max_steps - self.first_step

    def __iter__(self):
        global_step = self.first_step
        cached_epoch = None
        permutation = None
        while global_step < self.max_steps:
            epoch, batch_in_epoch = divmod(
                global_step, self.batches_per_epoch)
            if epoch != cached_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                permutation = torch.randperm(
                    self.n_samples, generator=generator).tolist()
                cached_epoch = epoch
            start = batch_in_epoch * self.batch_size
            stop = min(start + self.batch_size, self.n_samples)
            yield permutation[start:stop]
            global_step += 1


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
    }


def _restore_rng_state(state):
    if not state:
        raise RuntimeError("Exact-resume checkpoint has no RNG state")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["pos"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]])


def _autocast_context(device, precision):
    if torch.device(device).type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _make_lr_scheduler(optimizer, max_steps, warmup_steps, eta_min):
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, max_steps - warmup_steps),
        eta_min=eta_min,
    )
    if warmup_steps == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0,
        total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps])


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _dataset_fingerprint(packaged_dir, train_ids, val_ids):
    """Cheap stable identity for the data used by an exact resume."""
    root = Path(packaged_dir)
    manifest = {
        "train": list(train_ids),
        "val": list(val_ids),
        "sample_sizes": [
            (sample_id, (root / f"{sample_id}.npz").stat().st_size)
            for sample_id in list(train_ids) + list(val_ids)
        ],
    }
    for name in ("norm_stats_v3.json", "case_splits_v3.json",
                 "pilot_report_v3.json"):
        path = root / name
        if path.exists():
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_loss(
    model, scheduler, batch, device, fixed_t=None,
    cond_drop_prob=0.0, self_cond_p=0.5, return_parts=False,
):
    """Masked epsilon-MSE; optionally return numerator and node count."""
    centerline = batch["centerline"].to(device, non_blocking=True)
    mask = batch["centerline_mask"].to(device, non_blocking=True)
    images = batch["images"].to(device, non_blocking=True)
    poses = batch["poses"].to(device, non_blocking=True)
    x0 = centerline[..., :4]
    batch_size = x0.shape[0]

    if fixed_t is None and cond_drop_prob > 0.0:
        dropped = torch.rand(batch_size, device=device) < cond_drop_prob
        if dropped.any():
            images, poses = images.clone(), poses.clone()
            images[dropped] = 0.0
            poses[dropped] = 0.0

    timesteps = (
        torch.randint(0, scheduler.n_steps, (batch_size,), device=device)
        if fixed_t is None else
        torch.full((batch_size,), int(fixed_t), dtype=torch.long, device=device)
    )
    noisy, true_noise = scheduler.add_noise(x0, timesteps)

    x0_self = None
    use_self_conditioning = (
        fixed_t is None and self_cond_p > 0.0
        and torch.rand((), device=device).item() < self_cond_p)
    if use_self_conditioning:
        with torch.no_grad():
            first_noise = model(
                noisy, timesteps, images, poses,
                x0_self=None, node_mask=mask)
            alpha_bar = scheduler.alpha_bars[timesteps].view(-1, 1, 1)
            x0_self = (
                (noisy - torch.sqrt(1.0 - alpha_bar) * first_noise)
                / torch.sqrt(alpha_bar)
            )[..., :3].detach()

    predicted_noise = model(
        noisy, timesteps, images, poses,
        x0_self=x0_self, node_mask=mask)
    # Keep reduction in FP32 when the forward runs under BF16.
    per_point = F.mse_loss(
        predicted_noise.float(), true_noise.float(), reduction="none"
    ).mean(dim=-1)
    numerator = (per_point * mask.float()).sum()
    denominator = mask.sum(dtype=torch.float32).clamp_min(1.0)
    if return_parts:
        return numerator, denominator
    return numerator / denominator


@torch.no_grad()
def evaluate(model, scheduler, val_loader, device, precision="fp32"):
    """Point-weighted deterministic VAL loss over five fixed timesteps."""
    was_training = model.training
    model.eval()
    cuda_devices = (
        [torch.cuda.current_device()]
        if torch.device(device).type == "cuda" else [])
    total_numerator = 0.0
    total_denominator = 0.0
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(EVAL_SEED)
            if cuda_devices:
                torch.cuda.manual_seed_all(EVAL_SEED)
            for batch in val_loader:
                for timestep in EVAL_TIMESTEPS:
                    with _autocast_context(device, precision):
                        numerator, denominator = compute_loss(
                            model, scheduler, batch, device,
                            fixed_t=timestep, return_parts=True)
                    total_numerator += numerator.item()
                    total_denominator += denominator.item()
    finally:
        model.train(was_training)
    if total_denominator == 0:
        raise RuntimeError("Validation contains no valid centerline nodes")
    return total_numerator / total_denominator


def _validate_resume_signature(checkpoint, expected):
    actual = checkpoint.get("run_signature")
    if actual is None:
        raise RuntimeError(
            "latest.pt is an older partial checkpoint and cannot provide an "
            "exact resume. Use it only as init_checkpoint (fine-tuning), or "
            "start the clean run in a new checkpoint directory.")
    mismatches = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value}
    if mismatches:
        detail = ", ".join(
            f"{key}: checkpoint={old!r}, config={new!r}"
            for key, (old, new) in mismatches.items())
        raise RuntimeError(f"Exact-resume configuration mismatch: {detail}")


def _checkpoint_payload(
    *, raw_model, optimizer, lr_scheduler, step, last_val_loss,
    best_val_loss, early_stop_reference, checks_since_improvement,
    history, run_signature, config, elapsed_seconds_total,
    session_count, stop_reason,
):
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "full_training_state",
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": int(step),
        "val_loss": last_val_loss,
        "best_val_loss": float(best_val_loss),
        "early_stop_reference": float(early_stop_reference),
        "checks_since_improvement": int(checks_since_improvement),
        "hidden_dim": int(run_signature["hidden_dim"]),
        "history": history,
        "rng_state": _rng_state(),
        "run_signature": run_signature,
        "config": config,
        "elapsed_seconds_total": float(elapsed_seconds_total),
        "session_count": int(session_count),
        "stop_reason": stop_reason,
    }


def train(config, quick_test=False):
    """Train from scratch or resume an exact full-state checkpoint."""
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    device = resolve_device(train_cfg.get("device", "auto"))
    device_type = torch.device(device).type

    hidden_dim = int(train_cfg.get("hidden_dim", 384))
    batch_size = int(train_cfg.get("batch_size", 16))
    learning_rate = float(train_cfg.get("lr", 3e-4))
    max_steps = int(train_cfg.get("max_steps", 200_000))
    warmup_steps = int(train_cfg.get("warmup_steps", 2_000))
    eta_min_ratio = float(train_cfg.get("eta_min_ratio", 0.01))
    val_every = int(train_cfg.get("val_every", 1_000))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1_000))
    milestone_every = int(train_cfg.get("milestone_every", 25_000))
    log_every = int(train_cfg.get("log_every", 50))
    early_stopping = bool(train_cfg.get("early_stopping", False))
    early_stop_min_steps = int(
        train_cfg.get("early_stop_min_steps", 150_000))
    patience = int(train_cfg.get("patience", 30))
    min_delta = float(train_cfg.get("min_delta", 1e-4))
    max_hours = float(train_cfg.get("max_hours", 11.0))
    cond_drop_prob = float(train_cfg.get("cond_drop_prob", 0.1))
    self_cond_p = float(train_cfg.get("self_cond_p", 0.5))
    precision = str(train_cfg.get("precision", "bf16")).lower()
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    seed = int(train_cfg.get("seed", 20260911))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", True))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    compile_model = bool(train_cfg.get("compile", False))
    resume_mode = str(train_cfg.get("resume_mode", "auto")).lower()
    init_checkpoint = train_cfg.get("init_checkpoint")
    checkpoint_dir = Path(train_cfg.get(
        "checkpoint_dir", "outputs/h384_200k/checkpoints"))

    if precision not in {"fp32", "bf16"}:
        raise ValueError("train.precision must be fp32 or bf16")
    if resume_mode not in {"fresh", "auto", "required"}:
        raise ValueError("train.resume_mode must be fresh, auto, or required")
    if not 0 <= cond_drop_prob < 1 or not 0 <= self_cond_p <= 1:
        raise ValueError("conditioning probabilities are outside valid ranges")
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("require 0 <= warmup_steps < max_steps")
    if min(val_every, checkpoint_every, log_every, max_steps) <= 0:
        raise ValueError("step intervals and max_steps must be positive")
    if milestone_every < 0:
        raise ValueError("milestone_every must be zero (off) or positive")
    if max_hours <= 0:
        raise ValueError("max_hours must be positive")

    if quick_test:
        max_steps = min(max_steps, int(train_cfg.get("quick_steps", 20)))
        val_every = min(val_every, int(train_cfg.get("quick_val_every", 10)))
        checkpoint_every = min(checkpoint_every, val_every)
        log_every = min(log_every, val_every)
        warmup_steps = min(warmup_steps, max(0, max_steps // 4))
        milestone_every = 0
        early_stopping = False
        compile_model = False

    if precision == "bf16" and device_type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16")
    if precision == "bf16" and device_type != "cuda":
        print("WARNING: BF16 requested without CUDA; using FP32")
        precision = "fp32"
    if device_type != "cuda" and not quick_test:
        print("WARNING: full training without CUDA will be very slow")

    _seed_everything(seed)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_float32_matmul_precision(
            "high" if allow_tf32 else "highest")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    milestone_dir = checkpoint_dir / "milestones"
    if milestone_every:
        milestone_dir.mkdir(exist_ok=True)

    packaged_dir = data_cfg.get("packaged_dir", "data/processed/ds105_full")
    train_ids = list_samples(packaged_dir, "train")
    val_ids = list_samples(packaged_dir, "val")
    if quick_test:
        train_ids, val_ids = train_ids[:20], val_ids[:4]

    train_dataset = CoronaryCenterlineDatasetV31(
        packaged_dir, sample_ids=train_ids, return_render_poses=False)
    val_dataset = CoronaryCenterlineDatasetV31(
        packaged_dir, sample_ids=val_ids, return_render_poses=False)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and device_type == "cuda",
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1_000_000),
        **loader_kwargs)

    raw_model = CenterlineDenoiser(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        raw_model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    noise_scheduler = NoiseScheduler(n_steps=1000, device=device)
    lr_scheduler = _make_lr_scheduler(
        optimizer, max_steps, warmup_steps,
        learning_rate * eta_min_ratio)

    run_signature = {
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "lr": learning_rate,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "eta_min_ratio": eta_min_ratio,
        "val_every": val_every,
        "cond_drop_prob": cond_drop_prob,
        "self_cond_p": self_cond_p,
        "precision": precision,
        "grad_clip_norm": grad_clip_norm,
        "seed": seed,
        "allow_tf32": allow_tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "compile": compile_model,
        "dataset_fingerprint": _dataset_fingerprint(
            packaged_dir, train_ids, val_ids),
    }

    step = 0
    last_val_loss = None
    best_val_loss = float("inf")
    early_stop_reference = float("inf")
    checks_since_improvement = 0
    history = {"train_step": [], "train_loss": [], "val_step": [],
               "val_loss": [], "lr": []}
    elapsed_before = 0.0
    session_count = 1

    should_resume = latest_path.exists() and resume_mode in {"auto", "required"}
    if resume_mode == "fresh" and latest_path.exists():
        raise FileExistsError(
            f"{latest_path} exists; choose a new directory or resume it")
    if resume_mode == "required" and not latest_path.exists():
        raise FileNotFoundError(
            f"resume_mode=required but {latest_path} does not exist")
    if should_resume and init_checkpoint:
        raise ValueError("Cannot combine exact resume with init_checkpoint")

    if should_resume:
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=True)
        _validate_resume_signature(checkpoint, run_signature)
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        step = int(checkpoint["step"])
        last_val_loss = checkpoint.get("val_loss")
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        early_stop_reference = float(checkpoint.get(
            "early_stop_reference", best_val_loss))
        checks_since_improvement = int(checkpoint.get(
            "checks_since_improvement", 0))
        history = checkpoint.get("history", history)
        elapsed_before = float(checkpoint.get("elapsed_seconds_total", 0.0))
        session_count = int(checkpoint.get("session_count", 1)) + 1
        _restore_rng_state(checkpoint.get("rng_state"))
        print(f"Exact resume at step {step}; best VAL loss {best_val_loss:.6f}")
    elif init_checkpoint:
        initial = torch.load(
            Path(init_checkpoint), map_location="cpu", weights_only=True)
        initial_hidden = int(initial.get("hidden_dim", hidden_dim))
        if initial_hidden != hidden_dim:
            raise RuntimeError(
                f"init checkpoint hidden_dim={initial_hidden}; config={hidden_dim}")
        raw_model.load_state_dict(initial["model"], strict=True)
        print("Loaded weights only: this is fine-tuning, not exact resume")
    else:
        print("Fresh run from random weights")

    if step >= max_steps:
        print(f"Run already complete at step {step}")
        return best_path

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=DeterministicStepBatchSampler(
            len(train_dataset), batch_size, step, max_steps, seed + 10_000),
        generator=torch.Generator().manual_seed(seed + 2_000_000),
        **loader_kwargs)

    model = raw_model
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("train.compile=true requires torch.compile")
        model = torch.compile(raw_model, dynamic=False)

    session_started = time.time()
    stop_reason = "max_steps"
    last_checkpoint_step = step
    print(
        f"Training on {device}: train={len(train_ids)}, val={len(val_ids)}, "
        f"hidden={hidden_dim}, batch={batch_size}, lr={learning_rate:g}, "
        f"steps={max_steps}, warmup={warmup_steps}, precision={precision}, "
        f"max_hours_per_session={max_hours}")

    def elapsed_total():
        return elapsed_before + (time.time() - session_started)

    def save_latest(reason):
        payload = _checkpoint_payload(
            raw_model=raw_model, optimizer=optimizer,
            lr_scheduler=lr_scheduler, step=step,
            last_val_loss=last_val_loss, best_val_loss=best_val_loss,
            early_stop_reference=early_stop_reference,
            checks_since_improvement=checks_since_improvement,
            history=history, run_signature=run_signature, config=config,
            elapsed_seconds_total=elapsed_total(),
            session_count=session_count, stop_reason=reason)
        _atomic_torch_save(payload, latest_path)
        return payload

    for batch in train_loader:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            loss = compute_loss(
                model, noise_scheduler, batch, device,
                cond_drop_prob=cond_drop_prob, self_cond_p=self_cond_p)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()
        step += 1

        should_log = step % log_every == 0 or step == max_steps
        if should_log:
            loss_value = float(loss.detach())
            if not np.isfinite(loss_value):
                stop_reason = "non_finite_loss"
                save_latest(stop_reason)
                raise FloatingPointError(
                    f"Non-finite training loss at step {step}")
            history["train_step"].append(step)
            history["train_loss"].append(loss_value)

        # Check before starting a potentially long validation pass.
        if (step < max_steps
                and time.time() - session_started >= max_hours * 3600):
            stop_reason = "wall_clock_limit"
            print(f"Reached {max_hours} h at step {step}; saving exact state")
            save_latest(stop_reason)
            break

        should_stop_early = False
        if step % val_every == 0 or step == max_steps:
            last_val_loss = float(evaluate(
                model, noise_scheduler, val_loader, device, precision))
            history["val_step"].append(step)
            history["val_loss"].append(last_val_loss)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            current_train_loss = (
                history["train_loss"][-1] if history["train_loss"]
                else float(loss.detach()))
            print(
                f"step {step}: train_loss={current_train_loss:.6f}, "
                f"val_loss={last_val_loss:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}, "
                f"session_elapsed={time.time() - session_started:.1f}s")

            is_new_best = last_val_loss < best_val_loss
            if is_new_best:
                best_val_loss = last_val_loss
            if last_val_loss < early_stop_reference - min_delta:
                early_stop_reference = last_val_loss
                checks_since_improvement = 0
            elif step >= early_stop_min_steps:
                checks_since_improvement += 1

            payload = save_latest("running")
            last_checkpoint_step = step
            if is_new_best:
                _atomic_torch_save(payload, best_path)
                print(f"  -> new numerical best VAL loss: {best_val_loss:.6f}")
            if (early_stopping and step >= early_stop_min_steps
                    and checks_since_improvement >= patience):
                stop_reason = "early_stopping"
                should_stop_early = True
                print(
                    f"Early stopping at {step}: no >= {min_delta:g} "
                    f"improvement for {patience} checks")

        if milestone_every and step % milestone_every == 0:
            milestone = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_type": "model_only_milestone",
                "model": raw_model.state_dict(),
                "step": int(step),
                "val_loss": last_val_loss,
                "hidden_dim": hidden_dim,
                "run_signature": run_signature,
            }
            _atomic_torch_save(
                milestone, milestone_dir / f"step_{step:06d}.pt")

        if step % checkpoint_every == 0 and step != last_checkpoint_step:
            save_latest("running")
            last_checkpoint_step = step
        if should_stop_early or step >= max_steps:
            break

    save_latest(stop_reason)

    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(history["train_step"], history["train_loss"],
                  label="train loss", alpha=0.35)
        axis.plot(history["val_step"], history["val_loss"],
                  label="deterministic VAL loss", marker="o", markersize=3)
        axis.set(xlabel="Optimizer step", ylabel="Noise-prediction MSE",
                 title=f"Training curve (hidden_dim={hidden_dim})")
        axis.set_yscale("log")
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            checkpoint_dir / f"training_curve_{hidden_dim}.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass

    print(
        f"Stopped at step {step} ({stop_reason}); "
        f"best VAL loss={best_val_loss:.6f}")
    if not best_path.exists():
        raise RuntimeError("Training ended before the first validation")
    return best_path
