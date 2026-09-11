"""DDIM sampling + Chamfer evaluation in millimetres.

This is the only measurement that answers whether Phase 1 worked. val_loss is
single-step noise-prediction MSE; it says nothing about whether the reverse
process produces a coherent centerline (see the trainer module docstring).

Self-contained on purpose: it uses only NoiseScheduler, CenterlineDenoiser,
dataset_v3_1 and eval_chamfer. It does NOT import sampling.py or evaluate.py,
so it cannot be broken by, or break, whatever those contain.

WHAT IS GIVEN TO THE MODEL AT GENERATION TIME
---------------------------------------------
Each sample is generated at its own GT n_points. That is consistent with the
architecture's stated design -- topology (column 4) is fixed/given and the
denoiser only predicts x, y, z, radius -- but it IS information from the ground
truth, and the report must say so. A stricter protocol would predict the node
count too, or generate at a fixed length; either changes the Chamfer number, so
do not silently compare across protocols.

Everything else the model sees is the nominal-geometry conditioning only:
`poses` (scanner geometry, motion removed) and `images`. `poses_render` is
never loaded -- return_render_poses=False.

PROTOCOL
--------
Tune n_steps / guidance on the VAL split. Report ONE number on TEST, once.
Choosing DDIM settings by test Chamfer is test-set fitting and invalidates the
comparison against the 22.78 mm baseline.

Usage:
    python ddim_eval.py --ckpt /root/ckpt_v34/best.pt --data /root/ds105_full \\
        --split val --steps 50 --guidance 1.0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parent)
sys.path.insert(0, REPO)

from eval_chamfer import evaluate as chamfer_evaluate, print_summary   # noqa: E402
from src.coronarycl.dataset_v3_1 import (                              # noqa: E402
    CoronaryCenterlineDatasetV31, list_samples)
from src.coronarycl.models.diffusion import CenterlineDenoiser         # noqa: E402
from src.coronarycl.trainer import NoiseScheduler                      # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/ckpt_v34/best.pt")
    p.add_argument("--data", default="/root/ds105_full")
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (of 1000)")
    p.add_argument("--guidance", type=float, default=1.0,
                   help="classifier-free guidance scale; 1.0 = none. The model was "
                        "trained with cond_drop_prob=0.1 so >1 is available.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N samples only (debug)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ddim_results.json")
    p.add_argument("--save_pred", default="", help="optional .npz of predictions")
    return p.parse_args()


@torch.no_grad()
def ddim_sample(model, sched, images, poses, node_mask, n_steps, guidance, device):
    """Deterministic DDIM (Song et al. 2021, eta=0) with self-conditioning
    carried across steps (Chen et al. 2022), matching how the model was trained.

    Returns x0 in NORMALISED units, shape (B, N, 4).
    """
    B, N = node_mask.shape
    x = torch.randn(B, N, 4, device=device)

    # evenly spaced subsequence of the 1000 training timesteps, high -> low
    ts = torch.linspace(sched.n_steps - 1, 0, n_steps, device=device).long()
    abar = sched.alpha_bars

    if guidance != 1.0:
        null_images = torch.zeros_like(images)   # matches compute_loss's CFG dropout
        null_poses = torch.zeros_like(poses)

    x0_self, x0_pred = None, None
    for i in range(n_steps):
        t = ts[i]
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)

        eps = model(x, t_b, images, poses, x0_self=x0_self, node_mask=node_mask)
        if guidance != 1.0:
            eps_u = model(x, t_b, null_images, null_poses,
                          x0_self=x0_self, node_mask=node_mask)
            eps = eps_u + guidance * (eps - eps_u)

        a_t = abar[t]
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x0_self = x0_pred[:, :, :3].detach()

        a_prev = abar[ts[i + 1]] if i + 1 < n_steps else torch.tensor(1.0, device=device)
        x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps

    return x0_pred


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=True)
    print(f"checkpoint {args.ckpt}")
    print(f"  step {ck['step']}  val_loss {ck['val_loss']:.4f}  hidden_dim {ck['hidden_dim']}")
    model = CenterlineDenoiser(hidden_dim=ck["hidden_dim"]).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    sched = NoiseScheduler(n_steps=1000, device=dev)
    ids = list_samples(args.data, args.split)
    if args.limit:
        ids = ids[:args.limit]
    ds = CoronaryCenterlineDatasetV31(args.data, sample_ids=ids,
                                      return_render_poses=False)   # normalised
    print(f"  {len(ids)} {args.split} samples | DDIM {args.steps} steps | "
          f"guidance {args.guidance} | device {dev}\n")

    # group by length so each batch pads to something close to its own max
    order = sorted(range(len(ids)), key=lambda i: int(np.load(
        Path(args.data) / f"{ids[i]}.npz")["n_points"]))

    samples, preds_out = [], {}
    for bstart in range(0, len(order), args.batch):
        idxs = order[bstart:bstart + args.batch]
        items = [ds[i] for i in idxs]
        npts = [it["n_points"] for it in items]
        N = max(npts)
        N += (-N) % 4                                   # v3.3 pads internally too

        images = torch.stack([it["images"][:, :, :] for it in items]).to(dev)
        poses = torch.stack([it["poses"] for it in items]).to(dev)
        mask = torch.zeros(len(items), N, dtype=torch.bool, device=dev)
        for k, n in enumerate(npts):
            mask[k, :n] = True

        x0 = ddim_sample(model, sched, images, poses, mask,
                         args.steps, args.guidance, dev)
        pred_mm = ds.denormalize(x0.cpu().numpy())      # -> millimetres

        for k, i in enumerate(idxs):
            it, n = items[k], npts[k]
            gt_mm = ds.denormalize(it["centerline"].numpy())[:n, :3]
            p_mm = pred_mm[k, :n, :3]
            samples.append((it["sample"], it["vessel"], p_mm, gt_mm))
            if args.save_pred:
                preds_out[it["sample"]] = np.stack([p_mm, gt_mm])
        print(f"  {len(samples):>4}/{len(ids)} sampled", flush=True)

    rows, summary = chamfer_evaluate(
        samples, thresholds=(1.0, 2.0, 5.0))            # matches configs eval:
    print()
    print_summary(summary, baseline_mm=22.78)

    json.dump({"checkpoint": args.ckpt, "step": int(ck["step"]),
               "split": args.split, "ddim_steps": args.steps,
               "guidance": args.guidance, "seed": args.seed,
               "n_points_source": "ground truth (topology given by design)",
               "summary": summary, "per_sample": rows},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    if args.save_pred:
        np.savez_compressed(args.save_pred, **preds_out)
        print(f"wrote {args.save_pred}")


if __name__ == "__main__":
    main()
