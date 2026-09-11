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
def sweep(shuffle_images: bool, timesteps):
    """Mean loss per timestep. Seeded identically for both conditions, so the
    diffusion noise and batch order are the same and the only difference is
    which projections the denoiser saw."""
    torch.manual_seed(SEED)
    if dev == "cuda":
        torch.cuda.manual_seed_all(SEED)
    acc = {t: [] for t in timesteps}
    for b in loader:
        if shuffle_images:
            if b["images"].shape[0] < 2:
                continue                    # a size-1 batch cannot be shuffled
            b = dict(b)
            b["images"] = torch.roll(b["images"], shifts=1, dims=0)
        for t in timesteps:
            acc[t].append(compute_loss(model, sched, b, dev, fixed_t=t).item())
    return {t: sum(v) / len(v) for t, v in acc.items()}


# A per-timestep breakdown, not just the average.
#
# In eps-prediction, x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps:
#   t -> 0    abar -> 1, so x_t ~ x0 and eps is almost invisible in the input.
#             Recovering it is ill-posed, the loss is LARGE, and no amount of
#             conditioning helps -- expect ~0% sensitivity here, and read
#             nothing into it.
#   t -> 999  abar -> ~4e-5, so x_t ~ eps and the target is nearly copyable
#             from the input. The loss is TINY. Sensitivity there is measured
#             on a vanishing base, but it still matters for sampling: DDIM
#             computes x0 = (x_t - sqrt(1-abar)*eps)/sqrt(abar), dividing by
#             sqrt(abar) ~ 0.0064, so it amplifies any eps error by ~156x.
#
# The plain average over EVAL_TIMESTEPS is dominated by the t=0 term, where the
# metric is meaningless by construction, and so understates the model badly.
# Read the per-timestep column. A healthy model shows sensitivity RISING with t.
TS = sorted(set(list(EVAL_TIMESTEPS) + [100, 200, 400, 600, 800]))

m = sweep(False, TS)
s = sweep(True, TS)

print(f"\n  {'t':>5}{'matched':>10}{'shuffled':>10}{'sensitivity':>14}")
print("  " + "-" * 39)
for t in TS:
    d = (s[t] - m[t]) / m[t]
    star = "  <-- informative" if t >= 150 else "  (uninformative by construction)"
    print(f"  {t:>5}{m[t]:>10.4f}{s[t]:>10.4f}{d * 100:>13.1f}%{star}")

mid = [t for t in TS if t >= 150]
agg = (sum(s[t] for t in TS) - sum(m[t] for t in TS)) / sum(m[t] for t in TS)
mid_sens = (sum(s[t] for t in mid) - sum(m[t] for t in mid)) / sum(m[t] for t in mid)
print(f"\n  all timesteps    {agg * 100:+.1f}%   (dominated by t=0 -- do not quote this)")
print(f"  t >= 150 only    {mid_sens * 100:+.1f}%   <- the summary figure")
print("\n  Read the COLUMN, not just the summary. A healthy model shows")
print("  sensitivity rising with t. Flat ~0% across all t means the model is")
print("  ignoring the projections (the collapse that hit v2 and v3). ~0% at")
print("  t=0 alone is expected and says nothing either way.")
print("  This is a diagnostic, not a verdict -- only Chamfer is the verdict.")
