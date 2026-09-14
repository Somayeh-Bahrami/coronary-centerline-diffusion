"""Observe the canonical DDIM sampler without changing its computations.

Use the patient/sample selection from diagnose_edge_timesteps.py. Record
unbounded and bounded x0 estimates, not noisy x_t, at fixed sampling stages.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def observe_sampler(sampler, keep, kwargs):
    """Line tracing is observational; assert the installed sampler layout."""
    inner = inspect.unwrap(sampler)
    lines, start = inspect.getsourcelines(inner)
    before = [start+i for i, s in enumerate(lines) if s.strip() == "if lower is not None:"]
    after = [start+i for i, s in enumerate(lines) if s.strip() == "x0_self = x0_hat[..., :3].detach()"]
    if len(before) != 1 or len(after) != 1:
        raise RuntimeError("Sampler source changed; tracing anchors require review")
    records = {}

    def trace(frame, event, arg):
        if frame.f_code is not inner.__code__:
            return None
        if event == "line" and frame.f_lineno in (before[0], after[0]):
            loc = frame.f_locals
            index = int(loc["index"])
            if index in keep:
                stage = "raw" if frame.f_lineno == before[0] else "bounded"
                records[index, stage] = (
                    int(loc["timestep"]), loc["x0_hat"].detach().float().cpu().clone())
        return trace

    if sys.gettrace() is not None:
        raise RuntimeError("Run as a standalone subprocess outside a debugger")
    try:
        sys.settrace(trace)
        output = sampler(**kwargs)
    finally:
        sys.settrace(None)
    assert len(records) == len(keep)*2, "Missing trajectory snapshots"
    return output, records


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "data", "reference", "baseline", "control", "coherence", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU")
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    sys.path.insert(0, str(args.repo.resolve()))
    from src.coronarycl.dataset_v3_1 import CoronaryCenterlineDatasetV31, list_samples
    from src.coronarycl.models.diffusion import CenterlineDenoiser
    from src.coronarycl.trainer import NoiseScheduler
    from src.coronarycl.sampling import sample_ddim
    from src.coronarycl.metrics import topology_tree_edges, curve_metrics, evaluate_case, crop_bounds_mm
    from ddim_eval import initial_noise, normalized_physical_bounds, training_radius_bounds_mm

    ref = json.loads(args.reference.read_text())
    assert ref["protocol"]["split"] == "val"
    ids = ref["protocol"]["samples"]
    assert set(ids).issubset(set(list_samples(args.data, "val")))
    seed = int(ref["protocol"]["seed"])
    for arm in ("baseline", "control", "coherence"):
        assert sha(getattr(args, arm)) == ref["checkpoints"][arm]["sha256"], arm
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ds = CoronaryCenterlineDatasetV31(args.data, sample_ids=ids, return_render_poses=False)
    items = sorted([ds[i] for i in range(len(ds))], key=lambda x: x["n_points"])
    context = {}
    for item in items:
        n = int(item["n_points"])
        gt = ds.denormalize(item["centerline"].numpy())[:n, :4]
        e = topology_tree_edges(gt[:, :3], item["iso_mm"])
        lo, hi = crop_bounds_mm(item["sVoxel"].numpy(), item["iso_mm"])
        context[item["sample"]] = gt, e, lo, hi
    radius_bounds = training_radius_bounds_mm(args.data)
    scheduler = NoiseScheduler(n_steps=1000, device="cuda")
    keep = {0, 10, 25, 40, 50, 60, 75, 90, 99}
    rows, checks, predictions = [], {}, {}
    print(f"VAL: {len(ids)} vessels; same samples as reference; seed={seed}", flush=True)
    for arm in ("baseline", "control", "coherence"):
        ck = torch.load(getattr(args, arm), map_location="cpu", weights_only=True)
        model = CenterlineDenoiser(hidden_dim=int(ck["hidden_dim"])).cuda().float().eval()
        model.load_state_dict(ck["model"], strict=True)
        del ck
        for offset in range(0, len(items), args.batch):
            batch = items[offset:offset+args.batch]
            counts = [int(i["n_points"]) for i in batch]
            padded = max(counts)
            padded += (-padded) % 4
            mask = torch.arange(padded, device="cuda")[None, :] < torch.tensor(counts, device="cuda")[:, None]
            lo, hi = normalized_physical_bounds(batch, ds, radius_bounds, "cuda")
            kwargs = dict(model=model, scheduler=scheduler,
                images=torch.stack([i["images"] for i in batch]).cuda(),
                poses=torch.stack([i["poses"] for i in batch]).cuda(),
                node_mask=mask, device="cuda",
                initial_noise=initial_noise(batch, padded, seed, "cuda"),
                n_steps=100, guidance_scale=2.0, x0_min=lo, x0_max=hi)
            output, snapshots = observe_sampler(sample_ddim, keep, kwargs)
            if offset == 0:
                plain = sample_ddim(**kwargs)
                delta = float((output-plain).abs().max())
                assert torch.equal(output, plain), f"Tracing changed sampler output: {delta}"
                checks[arm] = {"traced_vs_plain_max_delta":delta, "exact_equal":True}
                print(f"PASS {arm}: traced output equals original sampler exactly", flush=True)
                del plain
            for (index, stage), (t, values) in sorted(snapshots.items()):
                mm = ds.denormalize(values.numpy())
                for b, item in enumerate(batch):
                    sid = item["sample"]
                    gt, e, low, high = context[sid]
                    pred = mm[b, :counts[b], :3]
                    assert np.isfinite(pred).all()
                    geom = curve_metrics(pred, gt[:, :3], e)
                    spatial = evaluate_case(pred, gt[:, :3], thresholds=(2.0,5.0),
                        edges=e, xyz_lower_mm=low, xyz_upper_mm=high)
                    error = (pred[e[:,1]]-pred[e[:,0]]) - (gt[e[:,1],:3]-gt[e[:,0],:3])
                    rows.append({"arm":arm,"iteration":index+1,"t":t,"stage":stage,
                        "sample":sid,"patient":str(item["patient_id"]),
                        "edge_vector_error_mm":float(np.linalg.norm(error,axis=1).mean()),
                        **{k:float(spatial[k]) for k in ("chamfer_l2","hd95_mm","overlap@2.0mm","out_of_crop_fraction")},
                        "lcc":geom["largest_connected_component_fraction_5x"],
                        "length_ratio":geom["tree_length_ratio"]})
                    if index==99 and stage=="bounded":
                        predictions[f"{arm}__{sid}"] = pred
                        predictions[f"gt__{sid}"] = gt[:,:3]
            print(f"{arm}: {min(offset+args.batch,len(items))}/{len(items)} vessels", flush=True)
        del model
        torch.cuda.empty_cache()
    metrics = ["edge_vector_error_mm","chamfer_l2","hd95_mm","overlap@2.0mm","out_of_crop_fraction","lcc","length_ratio"]
    summary = []
    for arm in checks:
        for index in sorted(keep):
            for stage in ("raw","bounded"):
                group = [r for r in rows if (r["arm"],r["iteration"],r["stage"])==(arm,index+1,stage)]
                patients = sorted({r["patient"] for r in group})
                summary.append({"arm":arm,"iteration":index+1,"t":group[0]["t"],"stage":stage,
                    **{k:float(np.mean([np.mean([r[k] for r in group if r["patient"]==pid]) for pid in patients])) for k in metrics}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol":{"split":"val","seed":seed,"samples":ids,
        "ddim_steps":100,"guidance":2.0,"bounds":"physical","precision":"fp32",
        "aggregation":"equal patient mean","sampler_sha256":sha(inspect.getfile(inspect.unwrap(sample_ddim))),
        "note":"raw snapshots precede clipping within the bounded production trajectory; not an unbounded rollout"},
        "checkpoints":ref["checkpoints"],"equivalence_checks":checks,"summary":summary,"rows":rows},indent=2))
    np.savez_compressed(args.out.with_suffix(".npz"), **predictions)
    print(f"TRAJECTORY DIAGNOSTIC COMPLETE: {args.out}", flush=True)


if __name__ == "__main__":
    main()
