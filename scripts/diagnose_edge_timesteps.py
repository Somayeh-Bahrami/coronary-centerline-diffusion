"""Paired VAL denoising diagnostic; no training and no TEST access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--coherence", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--patients", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.coronarycl.dataset_v3_1 import CoronaryCenterlineDatasetV31, list_samples
    from src.coronarycl.models.diffusion import CenterlineDenoiser
    from src.coronarycl.trainer import NoiseScheduler
    from src.coronarycl.metrics import topology_tree_edges, curve_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("Select the GPU runtime before running this diagnostic")
    if args.patients < 1 or args.batch < 1:
        raise ValueError("patients and batch must be positive")
    if args.out.exists():
        raise FileExistsError(f"Keep the existing result or choose a new output: {args.out}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ids = list_samples(args.data, "val")
    patients = sorted({s.rsplit("_", 1)[0] for s in ids})
    selected = set(np.random.default_rng(104729).choice(
        patients, min(args.patients, len(patients)), replace=False).tolist())
    ids = [s for s in ids if s.rsplit("_", 1)[0] in selected]
    ds = CoronaryCenterlineDatasetV31(args.data, sample_ids=ids, return_render_poses=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    edges = {}
    for sid in ids:
        with np.load(args.data / f"{sid}.npz", allow_pickle=False) as z:
            gt = z["centerline"][z["centerline_mask"].astype(bool), :3]
            edges[sid] = topology_tree_edges(gt, float(z["iso_mm"]))
    std = torch.tensor(ds.coord_std, device="cuda", dtype=torch.float32)
    schedule = NoiseScheduler(n_steps=1000, device="cuda")
    times = [0, 100, 250, 400, 500, 600, 750, 900, 999]
    rows, metadata = [], {}
    print(f"VAL only: {len(selected)} patients, {len(ids)} vessel samples", flush=True)
    for arm in ("baseline", "control", "coherence"):
        path = getattr(args, arm)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = CenterlineDenoiser(hidden_dim=int(ck["hidden_dim"])).cuda().float().eval()
        model.load_state_dict(ck["model"], strict=True)
        metadata[arm] = {"sha256": digest(path), "step": int(ck["step"]), "path": str(path)}
        del ck
        with torch.inference_mode():
            for t in times:
                for batch in loader:
                    x = batch["centerline"][..., :4].cuda().float()
                    mask = batch["centerline_mask"].cuda()
                    images, poses = batch["images"].cuda(), batch["poses"].cuda()
                    noise = torch.empty_like(x)
                    for b, sid in enumerate(batch["sample"]):
                        seed = int.from_bytes(hashlib.sha256(
                            f"104729:{sid}:{t}".encode()).digest()[:8], "little") % (2**63-1)
                        noise[b] = torch.randn(x[b].shape, device="cuda",
                            generator=torch.Generator(device="cuda").manual_seed(seed))
                    a = schedule.alpha_bars[t].float()
                    xt = a.sqrt() * x + (1-a).sqrt() * noise
                    ts = torch.full((len(x),), t, dtype=torch.long, device="cuda")
                    eps0 = model(xt, ts, images, poses, x0_self=None, node_mask=mask)
                    xhat0 = (xt - (1-a).sqrt() * eps0.float()) / a.sqrt()
                    eps1 = model(xt, ts, images, poses,
                        x0_self=xhat0[..., :3], node_mask=mask)
                    for mode, eps in (("off", eps0), ("on", eps1)):
                        xhat = (xt - (1-a).sqrt() * eps.float()) / a.sqrt()
                        if not torch.isfinite(xhat).all():
                            raise RuntimeError(f"Nonfinite reconstruction: {arm}, t={t}")
                        for b, sid in enumerate(batch["sample"]):
                            gt, pred = x[b, mask[b]], xhat[b, mask[b]]
                            e = torch.as_tensor(edges[sid], dtype=torch.long, device="cuda")
                            dg = gt[e[:,1], :3] - gt[e[:,0], :3]
                            dp = pred[e[:,1], :3] - pred[e[:,0], :3]
                            raw = F.smooth_l1_loss(dp, dg, beta=0.05).item()
                            geom = curve_metrics((pred[:,:3]*std).cpu().numpy(),
                                (gt[:,:3]*std).cpu().numpy(), edges[sid])
                            rows.append({"arm": arm, "t": t, "self_cond": mode,
                                "sample": sid, "patient": sid.rsplit("_",1)[0],
                                "abar": a.item(), "edge_huber_raw": raw,
                                "edge_huber_weighted": raw*a.item(),
                                "edge_vector_error_mm": ((dp-dg)*std).norm(dim=-1).mean().item(),
                                "xyz_error_mm": ((pred[:,:3]-gt[:,:3])*std).norm(dim=-1).mean().item(),
                                "eps_mse": F.mse_loss(eps[b,mask[b]].float(), noise[b,mask[b]]).item(),
                                "lcc": geom["largest_connected_component_fraction_5x"],
                                "length_ratio": geom["tree_length_ratio"]})
                print(f"Completed {arm}, t={t}", flush=True)
        del model
        torch.cuda.empty_cache()
    summary = []
    keys = ["eps_mse", "edge_huber_raw", "edge_huber_weighted",
            "edge_vector_error_mm", "xyz_error_mm", "lcc", "length_ratio"]
    for arm in metadata:
        for t in times:
            for mode in ("off", "on"):
                group = [r for r in rows if (r["arm"],r["t"],r["self_cond"])==(arm,t,mode)]
                result = {"arm":arm,"t":t,"self_cond":mode}
                for k in keys:
                    result[k] = float(np.mean([np.mean([r[k] for r in group if r["patient"]==pid]) for pid in selected]))
                summary.append(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol": {"split":"val", "precision":"fp32",
        "patients":sorted(selected),"samples":ids,"seed":104729,"timesteps":times,
        "beta_normalized":0.05,"aggregation":"equal patient mean",
        "note":"Exploratory paired GT-noising diagnostic; not DDIM rollout or training-seed replication"},
        "checkpoints":metadata,"summary":summary,"rows":rows}, indent=2))
    print(f"DIAGNOSTIC COMPLETE: {args.out}", flush=True)


if __name__ == "__main__":
    main()
