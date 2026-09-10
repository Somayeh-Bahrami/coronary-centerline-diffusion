"""Conditioning ablation: is the model actually using the X-ray projections?

Re-runs the SAME 1000-step overfit, but every __getitem__ swaps in the images
of a RANDOM other sample.  The pairing is re-randomised on every access, so
there is no consistent image->centerline mapping left to memorise.

Read it like this, comparing final train_loss against the matched run:

  shuffled loss CLEARLY WORSE   -> conditioning is live. Proceed.
  shuffled loss ~= matched      -> the model is ignoring the images and has
                                   learned the unconditional centerline prior.
                                   The image encoder / pose embedding is not
                                   reaching the denoiser. Do NOT start the long
                                   run: it would converge to a pose-agnostic
                                   average shape and never beat the baseline.

A deterministic offset (idx+k) would NOT work as an ablation -- that is just a
different but equally learnable mapping, and a 20-sample model memorises it.
Randomising per access is what removes the information.

Run:  python ablate_conditioning.py
"""
import sys

import numpy as np
import yaml

REPO = "/kaggle/working/repo"
sys.path.insert(0, REPO)

from src.coronarycl import dataset_v3_1 as D          # noqa: E402
from src.coronarycl.trainer import train              # noqa: E402

cls = D.CoronaryCenterlineDatasetV31
_orig_getitem = cls.__getitem__
_rng = np.random.default_rng(0)


def _shuffled_getitem(self, idx):
    item = _orig_getitem(self, idx)
    n = len(self)
    if n > 1:
        other = int(_rng.integers(n - 1))
        if other >= idx:                    # never pair a sample with itself
            other += 1
        item["images"] = _orig_getitem(self, other)["images"]
    return item


cls.__getitem__ = _shuffled_getitem
print("ABLATION ACTIVE: images randomly mismatched to centerlines on every access")

cfg = yaml.safe_load(open(f"{REPO}/configs/_overfit.yaml"))
cfg["train"]["checkpoint_dir"] = "/kaggle/working/ckpt_ablate"
train(cfg, quick_test=True)
