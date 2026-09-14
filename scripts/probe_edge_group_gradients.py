"""TRAIN-only paired gradient probe for consecutive/nonconsecutive GT edges.

Reports group-mean losses and each group's ACTUAL contribution to the current
all-edge loss. No optimizer, no checkpoint writes, and no automatic tuning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("repo", "data", "edge-cache", "baseline", "coherence", "out"):
        ap.add_argument("--"+k, type=Path, required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU")
    if args.out.exists():
        raise FileExistsError(f"Preserve the existing report: {args.out}")
    if min(args.batches, args.batch) < 1:
        raise ValueError("Batch settings must be positive")
    sys.path.insert(0, str(args.repo.resolve()))
    from src.coronarycl.dataset_v3_1 import CoronaryCenterlineDatasetV31, list_samples
    from src.coronarycl.edge_coherence import EdgeCollate, load_edge_cache, x0_from_eps, gather_edge_deltas
    from src.coronarycl.models.diffusion import CenterlineDenoiser
    from src.coronarycl.trainer import NoiseScheduler

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ids = list_samples(args.data, "train")
    ds = CoronaryCenterlineDatasetV31(args.data, sample_ids=ids, return_render_poses=False)
    edge_map, meta = load_edge_cache(args.edge_cache, args.data, required_ids=ids)
    scheduler = NoiseScheduler(n_steps=1000, device="cuda")
    rows, checkpoints = [], {}
    reference_batches = []
    for arm in ("baseline", "coherence"):
        path = getattr(args, arm)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = CenterlineDenoiser(hidden_dim=int(ck["hidden_dim"])).cuda().float().train()
        model.load_state_dict(ck["model"], strict=True)
        checkpoints[arm] = {"sha256": sha(path), "step": int(ck["step"])}
        del ck
        params = [p for p in model.parameters() if p.requires_grad]
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
            collate_fn=EdgeCollate(edge_map), generator=torch.Generator().manual_seed(20260911))
        for k, batch in enumerate(loader):
            if k >= args.batches:
                break
            # Reset per batch so model construction and previous forwards cannot
            # change timestep/noise/dropout/self-conditioning pairing between arms.
            torch.manual_seed(104729+k)
            torch.cuda.manual_seed_all(104729+k)
            sample_ids = list(batch["sample"])
            if arm == "baseline":
                reference_batches.append(sample_ids)
            else:
                assert sample_ids == reference_batches[k]
            x = batch["centerline"][..., :4].cuda().float()
            mask = batch["centerline_mask"].cuda()
            images, poses = batch["images"].cuda(), batch["poses"].cuda()
            dropped = torch.rand(len(x), device="cuda") < .1
            images, poses = images.clone(), poses.clone()
            images[dropped], poses[dropped] = 0, 0
            t = torch.randint(0,1000,(len(x),),device="cuda")
            xt, noise = scheduler.add_noise(x,t)
            a = scheduler.alpha_bars[t]
            use_self = bool(torch.rand((),device="cuda") < .5)
            self_x = None
            if use_self:
                with torch.no_grad():
                    first = model(xt,t,images,poses,x0_self=None,node_mask=mask)
                    self_x = x0_from_eps(xt,first,a)[...,:3].detach()
            eps = model(xt,t,images,poses,x0_self=self_x,node_mask=mask)
            eps_loss = ((eps-noise).square().mean(-1)*mask).sum()/mask.sum()
            xhat = x0_from_eps(xt,eps,a)
            e, em = batch["edge_index"].cuda(), batch["edge_mask"].cuda()
            consecutive = em & ((e[...,1]-e[...,0]).abs()==1)
            nonconsecutive = em & ~consecutive
            assert nonconsecutive.any(), "No nonconsecutive edges in this batch"
            error = F.smooth_l1_loss(gather_edge_deltas(xhat,e),
                gather_edge_deltas(x,e), beta=.05, reduction="none").mean(-1)
            weighted = error*a[:,None]
            total_count = em.sum(1).clamp_min(1)
            losses = {"epsilon":eps_loss}
            for name, group in (("consecutive",consecutive),("nonconsecutive",nonconsecutive)):
                # Contributions sum EXACTLY to the implemented all-edge mean.
                losses[name+"_contribution"] = ((weighted*group).sum(1)/total_count).mean()
                counts = group.sum(1)
                valid = counts > 0
                losses[name+"_group_mean"] = ((weighted*group).sum(1)[valid]/counts[valid]).mean()
            losses["edge_total"] = ((weighted*em).sum(1)/total_count).mean()
            assert torch.allclose(losses["edge_total"], losses["consecutive_contribution"]+losses["nonconsecutive_contribution"])
            grads = {}
            for name, loss in losses.items():
                g = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                grads[name] = [v.detach() if v is not None else None for v in g]
            def dot(left,right):
                return sum((a*b).sum() for a,b in zip(left,right) if a is not None and b is not None)
            norms = {name:float(dot(g,g).sqrt()) for name,g in grads.items()}
            assert all(np.isfinite(v) for v in norms.values()) and norms["epsilon"]>0
            cosines = {}
            for name in grads:
                if name != "epsilon":
                    cosines[name+"_vs_epsilon"] = float(dot(grads[name],grads["epsilon"]))/max(norms[name]*norms["epsilon"],1e-30)
            cosines["consecutive_vs_nonconsecutive"] = float(dot(grads["consecutive_contribution"],grads["nonconsecutive_contribution"]))/max(norms["consecutive_contribution"]*norms["nonconsecutive_contribution"],1e-30)
            row = {"arm":arm,"batch":k,"samples":sample_ids,"timesteps":t.cpu().tolist(),
                "self_conditioning":use_self,"dropped":dropped.cpu().tolist(),
                "nonconsecutive_edge_fraction":float((nonconsecutive.sum(1)/total_count).mean()),
                "losses":{name:float(loss.detach()) for name,loss in losses.items()},
                "gradient_norms":norms,"gradient_cosines":cosines,
                "weighted_gradient_ratios_to_epsilon":{name:.2*v/norms["epsilon"] for name,v in norms.items() if name!="epsilon"}}
            rows.append(row)
            print(f"{arm} batch {k+1}/{args.batches}: w*grad/eps consecutive={row['weighted_gradient_ratios_to_epsilon']['consecutive_contribution']:.3f}, nonconsecutive={row['weighted_gradient_ratios_to_epsilon']['nonconsecutive_contribution']:.3f}",flush=True)
            del grads, losses, eps, xhat, error, weighted, g
        del model, params
        torch.cuda.empty_cache()
    summary = {}
    for arm in checkpoints:
        group = [r for r in rows if r["arm"]==arm]
        summary[arm] = {field:{key:{"median":float(np.median([r[field][key] for r in group])),
            "min":float(min(r[field][key] for r in group)),"max":float(max(r[field][key] for r in group))}
            for key in group[0][field]} for field in ("losses","gradient_norms","gradient_cosines","weighted_gradient_ratios_to_epsilon")}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"protocol":{"split":"train","precision":"fp32","beta":.05,
        "current_weight":.2,"batches":args.batches,"batch_size":args.batch,"cache_verified":meta["n_verified"],
        "note":"Group-mean gradients are hypothetical reweightings; contribution gradients are the actual current-loss decomposition. Norms are not additive; inspect cosines."},
        "checkpoints":checkpoints,"summary":summary,"rows":rows},indent=2))
    print(f"GROUP GRADIENT PROBE COMPLETE: {args.out}",flush=True)


if __name__ == "__main__":
    main()
