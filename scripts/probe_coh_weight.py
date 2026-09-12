"""Choose `coh_weight` from GRADIENT norms, not from loss values.

A ratio of loss magnitudes says nothing about how much each term moves the
parameters. This measures ||d coh / d theta|| / ||d eps / d theta|| on real
batches from the checkpoint the fine-tune will start from, then reports the
weight that puts the coherence term at a chosen share of the gradient.

Requires `detach_parts=False`, which returns the two losses still attached to
the autograd graph. With the default (detached) parts the ratio cannot be
computed at all.

Run from the repo root:

    python scripts/probe_coh_weight.py \\
        --ckpt .../checkpoints/prod/milestones/step_150000.pt \\
        --data .../ds105_full --edge_cache .../edge_cache --batches 8
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from src.coronarycl.dataset_v3_1 import (                       # noqa: E402
    CoronaryCenterlineDatasetV31, list_samples)
from src.coronarycl.edge_coherence import (                     # noqa: E402
    EdgeCollate, gradient_norm_ratio, load_edge_cache)
from src.coronarycl.models.diffusion import CenterlineDenoiser  # noqa: E402
from src.coronarycl.trainer import NoiseScheduler, compute_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--edge_cache", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--huber_beta", type=float, default=0.05)
    ap.add_argument("--targets", default="0.2,0.3,0.5",
                    help="desired coherence share of the gradient")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    hidden = int(ck["hidden_dim"])
    model = CenterlineDenoiser(hidden_dim=hidden).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.train()                       # same mode the training step uses
    print(f"checkpoint step {ck.get('step')}  hidden_dim {hidden}  device {dev}")

    train_ids = list_samples(args.data, "train")
    ds = CoronaryCenterlineDatasetV31(
        args.data, sample_ids=train_ids, return_render_poses=False)
    edge_map, meta = load_edge_cache(args.edge_cache, args.data,
                                     required_ids=train_ids)
    print(f"edge cache verified on {meta['n_verified']} train samples")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=EdgeCollate(edge_map),
                        generator=torch.Generator().manual_seed(args.seed))
    sched = NoiseScheduler(n_steps=1000, device=dev)
    coh_kwargs = {"huber_beta": args.huber_beta, "weight_by_abar": True}

    ratios, eps_g, coh_g, eps_v, coh_v = [], [], [], [], []
    for k, batch in enumerate(loader):
        if k >= args.batches:
            break
        model.zero_grad(set_to_none=True)
        # fp32 and detach_parts=False: the probe needs both losses on the graph
        total, eps_loss, coh_loss = compute_loss(
            model, sched, batch, dev,
            cond_drop_prob=0.1, self_cond_p=0.5,
            coh_weight=1.0, coh_kwargs=coh_kwargs,
            return_coh=True, detach_parts=False)
        r, ge, gc = gradient_norm_ratio(model, eps_loss, coh_loss)
        ratios.append(r); eps_g.append(ge); coh_g.append(gc)
        eps_v.append(float(eps_loss)); coh_v.append(float(coh_loss))
        print(f"  batch {k:>2}  eps {float(eps_loss):.5f} coh {float(coh_loss):.5f}"
              f" | grad eps {ge:.4e} coh {gc:.4e} | ratio {r:.4f}")

    if not ratios:
        sys.exit("no batches processed")
    med = statistics.median(ratios)
    print(f"\ngradient-norm ratio  median {med:.4f}  "
          f"min {min(ratios):.4f}  max {max(ratios):.4f}  (n={len(ratios)})")
    print(f"loss-value ratio     median {statistics.median(coh_v) / statistics.median(eps_v):.4f}"
          f"   <- NOT the number to tune on")
    print(f"\nhuber_beta = {args.huber_beta} (beta rescales the loss, so it is "
          f"entangled with the weight -- fix beta before choosing one)")
    print(f"\n{'target coherence share':>24}  {'coh_weight':>11}")
    for t in (float(x) for x in args.targets.split(",")):
        # want w*ratio / (1 + w*ratio) = t  ->  w = t / ((1-t) * ratio)
        print(f"{t:>24.2f}  {t / ((1.0 - t) * med):>11.3f}")
    print("\nSpread across batches matters: if min and max straddle a factor of "
          "two, one probe batch is not enough to fix the weight. Raise --batches.")


if __name__ == "__main__":
    main()
