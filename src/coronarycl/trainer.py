"""Step 2.3 -- model training. Full-dataset run needs a CUDA GPU for
reasonable speed. Runs on Kaggle Notebooks (GPU P100). M4 is used only for
local unit tests (quick_test=True) on a tiny subset before committing
a full Kaggle run.

Training loop: standard diffusion noise-prediction objective over the
packaged dataset from Step 1.3, using CenterlineDenoiser
(src/coronarycl/models/diffusion.py). Validation uses a deterministic,
multi-timestep loss (averaged over a fixed set of timesteps, not one
random sample per check) -- an early naive version that validated on
one random timestep per check produced a val-loss curve too noisy to
give a trustworthy early-stopping signal.

Two things learned the hard way, both now built into this module
rather than left as open risks:

1. A good, steadily-decreasing training/validation loss does NOT by
   itself confirm the model can generate a coherent result, that
   requires the full reverse-diffusion sampling process (see
   generate_case_projections-style verification in evaluate.py /
   Step 3.1), which exercises the model very differently than the
   single-step noise-prediction loss this module tracks. Do not treat
   a low val_loss here as a substitute for that separate check.
2. When loading a saved checkpoint for evaluation elsewhere, always
   instantiate the model with the SAME hidden_dim used during this
   training run, and load with strict=True. A hidden_dim mismatch
   between training and evaluation was previously masked by
   non-strict loading, silently producing a model with partially
   uninitialized/mismatched weights and wildly incoherent output that
   was hard to distinguish from "the model just isn't very good",
   verify with model.input_proj (or similar) before trusting an
   evaluation result. This module records hidden_dim directly in every
   saved checkpoint specifically so this can't happen silently again.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import resolve_device
from .dataset import CoronaryCenterlineDataset
from .models.diffusion import CenterlineDenoiser

EVAL_TIMESTEPS = [0, 250, 500, 750, 999]


class NoiseScheduler:
    """Standard linear-beta DDPM noise schedule (Ho et al., 2020)."""

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.n_steps = n_steps
        self.betas = torch.linspace(
            beta_start, beta_end, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + \
            torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise


def compute_loss(model, scheduler, batch, device, fixed_t=None, cond_drop_prob=0.0, self_cond_p=0.5):
    centerline = batch["centerline"].to(device)
    mask = batch["centerline_mask"].to(device)
    images = batch["images"].to(device)
    poses = batch["poses"].to(device)
    x0 = centerline[:, :, :4]
    B = x0.shape[0]
    # Classifier-free guidance: during TRAINING (fixed_t is None) only, zero the
    # conditioning for a fraction of samples so the model also learns to denoise
    # unconditionally (Ho & Salimans, 2022). Never during eval -- eval must
    # measure the conditional loss.
    if fixed_t is None and cond_drop_prob > 0.0:
        drop = torch.rand(B, device=device) < cond_drop_prob
        if drop.any():
            images = images.clone()
            poses = poses.clone()
            images[drop] = 0.0
            poses[drop] = 0.0
    t = (torch.randint(0, scheduler.n_steps, (B,), device=device) if fixed_t is None
         else torch.full((B,), fixed_t, device=device, dtype=torch.long))
    x_t, true_noise = scheduler.add_noise(x0, t)
    # Self-conditioning (Chen et al. 2022): with prob self_cond_p (training
    # only), do a no-grad forward to get a predicted-x0, then feed its xyz
    # back as the 3DPQT query for the real (grad) forward -- a meaningful
    # query at every timestep instead of the near-random noisy position.
    x0_self = None
    if fixed_t is None and self_cond_p > 0.0 and torch.rand(1).item() < self_cond_p:
        with torch.no_grad():
            eps1 = model(x_t, t, images, poses, x0_self=None)
            abar = scheduler.alpha_bars[t].view(-1, 1, 1)
            x0_self = ((x_t - torch.sqrt(1 - abar) * eps1) /
                       torch.sqrt(abar))[:, :, :3].detach()
    pred_noise = model(x_t, t, images, poses, x0_self=x0_self)
    per_point_loss = F.mse_loss(
        pred_noise, true_noise, reduction="none").mean(dim=-1)
    return (per_point_loss * mask.float()).sum() / mask.float().sum()


@torch.no_grad()
def evaluate(model: CenterlineDenoiser, scheduler: NoiseScheduler,
             val_loader: DataLoader, device: str) -> float:
    """Deterministic, multi-timestep validation loss -- averages over
    EVAL_TIMESTEPS instead of one random sample per batch, removing
    most of the run-to-run measurement noise a naive validation loop
    would otherwise show.
    """
    model.eval()
    losses = [compute_loss(model, scheduler, b, device, fixed_t=t).item()
              for b in val_loader for t in EVAL_TIMESTEPS]
    model.train()
    return sum(losses) / len(losses)


def train(config: dict, quick_test: bool = False):
    """Train CenterlineDenoiser on the packaged dataset from Step 1.3.

    Args:
        config: parsed YAML config (see configs/default.yaml). Expected
            keys (all optional, with defaults matching the values
            validated across this project's actual training runs):
              data.packaged_dir, data.splits_file (dir containing
                  case_splits.json)
              train.hidden_dim (default 384 -- current best; started at
                  128, then 256, each shown to genuinely improve both
                  training loss and val-set Chamfer L2, not just the
                  former -- see docs/work_breakdown.md Step 2.3 for the
                  full comparison table)
              train.batch_size (default 16)
              train.lr (default 1e-3)
              train.max_steps (default 100000)
              train.patience (default 30)
              train.val_every (default 200)
              train.max_hours (default 12.0 -- hard wall-clock safety
                  cutoff, since full runs are billed against a limited
                  weekly GPU quota)
              train.checkpoint_dir (default "checkpoints")
        quick_test: if True, overrides max_steps/val_every to a tiny
            number of steps on a small subset -- for local M4
            sanity-checking only, not real training.

    Returns:
        Path to the best checkpoint saved during this run.
    """
    device = resolve_device(config.get("train", {}).get("device", "auto"))
    print(f"Training on device: {device}")
    if device != "cuda" and not quick_test:
        print("WARNING: no CUDA GPU detected. Full training should run on "
              "Kaggle (GPU P100), not locally. Pass quick_test=True for a "
              "local sanity check instead.")

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})

    hidden_dim = train_cfg.get("hidden_dim", 384)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 1e-3)
    max_steps = train_cfg.get("max_steps", 100000)
    patience = train_cfg.get("patience", 30)
    val_every = train_cfg.get("val_every", 200)
    max_hours = train_cfg.get("max_hours", 12.0)
    cond_drop_prob = train_cfg.get("cond_drop_prob", 0.1)
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))

    if quick_test:
        max_steps = min(max_steps, 20)
        val_every = min(val_every, 10)
        print(
            f"quick_test=True -- overriding to max_steps={max_steps}, val_every={val_every}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    packaged_dir = data_cfg.get("packaged_dir", "data/processed/packaged")
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    import json
    with open(splits_dir / "case_splits.json") as f:
        splits = json.load(f)

    train_ids = splits["train"][:20] if quick_test else splits["train"]
    val_ids = splits["val"][:4] if quick_test else splits["val"]

    train_dataset = CoronaryCenterlineDataset(packaged_dir, train_ids)
    val_dataset = CoronaryCenterlineDataset(packaged_dir, val_ids)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CenterlineDenoiser(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = NoiseScheduler(n_steps=1000, device=device)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.01)

    # Resume from a previous run's checkpoint, if one exists -- protects
    # against losing GPU-hours to a session interruption mid-run.
    start_step = 0
    best_val_loss = float("inf")
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        if ckpt.get("hidden_dim") != hidden_dim:
            raise RuntimeError(
                f"Existing checkpoint was trained with hidden_dim={ckpt.get('hidden_dim')}, "
                f"but this run is configured for hidden_dim={hidden_dim}. Refusing to resume "
                f"across a hidden_dim mismatch -- start a fresh checkpoint_dir instead."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:                                     # <-- new
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from step {start_step}, val_loss={best_val_loss:.4f}")
    else:
        print("No existing checkpoint found -- starting fresh.")

    train_losses, val_losses, val_steps = [], [], []
    steps_since_improvement = 0
    step = start_step
    start_time = time.time()

    print(f"Starting training: {len(train_ids)} train cases, batch_size={batch_size}, "
          f"lr={lr}, hidden_dim={hidden_dim}, patience={patience}, max_hours={max_hours}")

    while step < max_steps:
        for batch in train_loader:
            loss = compute_loss(model, scheduler, batch,
                                device, cond_drop_prob=cond_drop_prob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_losses.append(loss.item())
            step += 1

            if step % val_every == 0:
                val_loss = evaluate(model, scheduler, val_loader, device)
                val_losses.append(val_loss)
                val_steps.append(step)
                elapsed = time.time() - start_time
                print(f"step {step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, "
                      f"elapsed={elapsed:.1f}s")

                # hidden_dim is saved alongside the weights so a future
                # evaluation script can verify it's instantiating the
                # matching architecture before loading -- see module
                # docstring for why this matters.
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),          # <-- new
                    "step": step, "val_loss": val_loss, "hidden_dim": hidden_dim,
                }, latest_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_since_improvement = 0
                    torch.save({
                        "model": model.state_dict(), "step": step,
                        "val_loss": val_loss, "hidden_dim": hidden_dim,
                    }, checkpoint_dir / "best.pt")
                    print(f"  -> new best val_loss: {best_val_loss:.4f}")
                else:
                    steps_since_improvement += 1
                    if steps_since_improvement >= patience:
                        print(f"\nEarly stopping at step {step} -- no val improvement "
                              f"for {patience} checks.")
                        step = max_steps
                        break

                if elapsed > max_hours * 3600:
                    print(f"\nHit {max_hours}h wall-clock limit at step {step} -- "
                          f"stopping safely.")
                    step = max_steps
                    break

            if step >= max_steps:
                break

    print(f"\nTraining finished. Best val_loss: {best_val_loss:.4f}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="train loss", alpha=0.3)
        plt.plot(val_steps, val_losses, label="val loss", marker="o")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.yscale("log")
        plt.title(f"Training curve (hidden_dim={hidden_dim})")
        plt.savefig(checkpoint_dir / f"training_curve_{hidden_dim}.png")
        plt.close()
    except ImportError:
        pass  # plotting is a convenience, not required for training to succeed

    return checkpoint_dir / "best.pt"
