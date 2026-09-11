"""DDIM sampling + Chamfer evaluation in millimetres.

This is the only measurement that answers whether Phase 1 worked. val_loss is
single-step noise-prediction MSE; it says nothing about whether the reverse
process produces a coherent centerline (see the trainer module docstring).

Metrics come from the repo's OWN src/coronarycl/metrics.py, deliberately, so
every number here is directly comparable to the 22.78 mm classical baseline and
to anything already reported. Note that convention:

    chamfer_l2 = mean(d_pred->gt) + mean(d_gt->pred)      # a SUM, not an average

which is 2x the "symmetric mean nearest-neighbour distance" convention used in
much of the literature. That is fine internally, but if you ever compare against
a published figure (DeepCA's, say), check which convention that paper used
before putting the two numbers in the same table.

Sampling is self-contained: it does NOT import sampling.py or evaluate.py, so it
cannot be broken by, or break, whatever those contain.

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

from scipy.spatial import cKDTree                                      # noqa: E402
from src.coronarycl.metrics import chamfer_l2, overlap_metric          # noqa: E402
from src.coronarycl.dataset_v3_1 import (                              # noqa: E402
    CoronaryCenterlineDatasetV31, list_samples)
from src.coronarycl.models.diffusion import CenterlineDenoiser         # noqa: E402
from src.coronarycl.trainer import NoiseScheduler                      # noqa: E402


THRESH_KEYS = None          # set by score()


def hausdorff95(pred, gt):
    """95th-percentile symmetric distance, mm. Robust worst-case companion."""
    d_pg, _ = cKDTree(gt).query(pred)
    d_gp, _ = cKDTree(pred).query(gt)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def score(samples, thresholds=(1.0, 2.0, 5.0)):
    """samples: (sample_id, vessel, pred_mm, gt_mm) with padding ALREADY removed.

    Stratified by vessel because LCA and RCA differ in yield (82.2% vs 92.3%)
    and in branching, so a pooled mean hides which system actually fails.
    """
    global THRESH_KEYS
    THRESH_KEYS = [f"overlap@{d}mm" for d in thresholds]
    rows = []
    for sid, vessel, pred, gt in samples:
        r = {"sample": sid, "vessel": vessel, "n_pred": len(pred), "n_gt": len(gt),
             "chamfer_l2": chamfer_l2(pred, gt), "hd95_mm": hausdorff95(pred, gt)}
        for d in thresholds:
            r[f"overlap@{d}mm"] = overlap_metric(pred, gt, d)
        rows.append(r)

    keys = ["chamfer_l2", "hd95_mm"] + THRESH_KEYS
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(r["vessel"], []).append(r)

    summary = {}
    for name, rs in groups.items():
        st = {"n": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs], float)
            v = v[np.isfinite(v)]
            st[k] = float(v.mean()) if v.size else float("nan")
            st[k + "_std"] = float(v.std()) if v.size else float("nan")
        st["chamfer_median"] = float(np.median([r["chamfer_l2"] for r in rs]))
        summary[name] = st
    return rows, summary


def print_summary(summary, baseline_mm=None, baseline_label="classical epipolar"):
    order = [k for k in ("all", "LCA", "RCA") if k in summary]
    w = max(len(k) for k in order) + 2
    hdr = f"{'split':<{w}}{'n':>5}{'CD mm':>9}{'median':>9}{'HD95':>9}"
    hdr += "".join(f"{k.replace('overlap@', 'Ot '):>10}" for k in THRESH_KEYS)
    print(hdr); print("-" * len(hdr))
    for k in order:
        st = summary[k]
        line = (f"{k:<{w}}{st['n']:>5}{st['chamfer_l2']:>9.2f}"
                f"{st['chamfer_median']:>9.2f}{st['hd95_mm']:>9.2f}")
        line += "".join(f"{st[o]:>10.3f}" for o in THRESH_KEYS)
        print(line)
    print("\nchamfer_l2 convention: mean(pred->gt) + mean(gt->pred)  [a SUM]"
          "  -- src/coronarycl/metrics.py:36")
    if baseline_mm is not None:
        cd = summary["all"]["chamfer_l2"]
        print(f"{'BEATS' if cd < baseline_mm else 'does NOT beat'} the {baseline_label} "
              f"baseline ({cd:.2f} vs {baseline_mm:.2f} mm) -- same convention, "
              f"both from metrics.py.")


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

    rows, summary = score(samples, thresholds=(1.0, 2.0, 5.0))   # matches configs eval:
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
