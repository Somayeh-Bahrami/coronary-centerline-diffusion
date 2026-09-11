"""Conditioning sensitivity of a TRAINED checkpoint.

Why this and not ablate_conditioning.py: that script trains two models and
compares them, which confounds (a) different weight trajectories, (b) a
4-sample validation set, and (c) shuffling the eval images as well as the
training images. On a 20-sample overfit it also cannot discriminate at all,
because a 12M-parameter model memorises 20 centerlines from the sequence
alone and gains nothing from the images.

This holds ONE set of weights fixed and changes only what the model is shown
at evaluation time:

    matched   : each sample with its own projections
    shuffled  : each sample with another sample's projections
                (torch.roll by 1 within the batch -- no sample keeps its own)

The diffusion noise is seeded identically for both passes, and the batch order
is identical, so the only difference between the two numbers is which images
the denoiser saw.

    sensitivity = (shuffled - matched) / matched

    ~0.00          the model ignores the projections. It has learned a
                   pose-agnostic average centerline. val_loss is meaningless
                   as a quality signal and reconstruction will not beat the
                   classical baseline. This is the collapse that hit v2 and v3.
    clearly > 0    the projections carry information the model is using.
                   How much is enough is empirical -- record the number and
                   compare it across runs rather than against a fixed cut-off.

Run:  python cond_sensitivity.py [path/to/best.pt]
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = str(Path(__file__).resolve().parent)
sys.path.insert(0, REPO)

from src.coronarycl.trainer import (                      # noqa: E402
    NoiseScheduler, compute_loss, EVAL_TIMESTEPS)
from src.coronarycl.models.diffusion import CenterlineDenoiser   # noqa: E402
from src.coronarycl.dataset_v3_1 import (                 # noqa: E402
    CoronaryCenterlineDatasetV31, list_samples)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/ckpt_v34/best.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "/root/ds105_full"
SEED = 12345
BATCH = 16

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=dev, weights_only=True)
print(f"checkpoint {CKPT}\n  step {ck['step']}  val_loss {ck['val_loss']:.4f}  "
      f"hidden_dim {ck['hidden_dim']}")

model = CenterlineDenoiser(hidden_dim=ck["hidden_dim"]).to(dev)
model.load_state_dict(ck["model"], strict=True)     # strict: see trainer docstring
model.eval()

sched = NoiseScheduler(n_steps=1000, device=dev)
val_ids = list_samples(DATA, "val")
ds = CoronaryCenterlineDatasetV31(DATA, sample_ids=val_ids, return_render_poses=False)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
print(f"  {len(ds)} val samples, {EVAL_TIMESTEPS} timesteps")


@torch.no_grad()
def sweep(shuffle_images: bool) -> float:
    torch.manual_seed(SEED)                 # identical noise in both passes
    if dev == "cuda":
        torch.cuda.manual_seed_all(SEED)
    losses = []
    for b in loader:
        if shuffle_images:
            if b["images"].shape[0] < 2:
                continue                    # a size-1 batch cannot be shuffled
            b = dict(b)
            b["images"] = torch.roll(b["images"], shifts=1, dims=0)
        for t in EVAL_TIMESTEPS:
            losses.append(compute_loss(model, sched, b, dev, fixed_t=t).item())
    return sum(losses) / len(losses)


matched = sweep(False)
shuffled = sweep(True)
sens = (shuffled - matched) / matched

print(f"\n  matched   {matched:.4f}")
print(f"  shuffled  {shuffled:.4f}")
print(f"  sensitivity  {sens:+.4f}  ({sens * 100:+.1f}%)")
print("\n  ~0.00 => the model is ignoring the projections (conditioning collapse).")
print("  Record this number alongside the run's Chamfer -- it is the single")
print("  cheapest early warning that a checkpoint is not worth evaluating.")
