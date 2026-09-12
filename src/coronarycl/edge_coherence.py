"""Edge-coherence auxiliary loss on the REAL vessel graph.

Supersedes the earlier index-adjacent version, which was wrong: it penalised
DFS-consecutive pairs, while the failing gate (edge_continuity_5x,
largest_connected_component_fraction_5x) is computed on real graph edges. In
the inspected seed ~66% of broken edges joined nodes that are graph-adjacent
but NOT array-adjacent, so the index-adjacent term missed two thirds of the
failures it was meant to fix. Optimise the same edge set the gate measures.

The packaged centerline array does NOT carry the edge list: index 3 is radius,
index 4 is a topology LABEL. The edges must come from the same extraction the
metrics already use -- see PLUMBING below.

SCOPE CLAIM, CORRECTED
----------------------
The null-conditioned control (coherence 0.957 / 0.254 / 3.48 vs conditional
0.954 / 0.247 / 3.49) rules out CONDITIONING as the lever. It does NOT separate
the objective from the architecture. Leaving diffusion.py and sampling.py
unchanged for the first experiment is a scoping decision to keep one variable
moving -- not evidence that the architecture is innocent.

NO HARD CLAMP
-------------
An earlier draft clamped x0_hat to +/-4 sigma. That zeroes the gradient on
exactly the escaped nodes the loss exists to pull back (verified:
d/dx of clamp(x,-4,4)^2 at x=6 is 0). Huber on the residual bounds the
gradient magnitude at high t without ever zeroing it.

WEIGHT SELECTION
----------------
A ratio of loss VALUES says nothing about the ratio of GRADIENTS. Use
`gradient_norm_ratio()` below on a handful of real batches and pick the weight
from that. Changing the weight mid-run makes it a different experiment.

PLUMBING
--------
Edges are variable-count per sample, so they need the same pad+mask treatment
as nodes:
    edge_index (B, E_max, 2) long   -- node indices into the padded centerline
    edge_mask  (B, E_max)     bool  -- True = real edge
Precompute once per sample with the SAME function the metrics use, cache to a
sidecar keyed by sample id, return from the dataset, and pad in a collate_fn
(`collate_edges` below). Do not re-derive the graph in the training loop.
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
def x0_from_eps(x_t, eps_hat, abar_t):
    """Standard DDPM inversion, in fp32. abar_t (B,); x_t / eps_hat (B, N, C).

    fp32 is not optional: at t=999, sqrt(abar) ~ 0.0064 and the division
    amplifies everything ~156x. bf16 has ~3 decimal digits of mantissa.
    """
    a = abar_t.float().view(-1, 1, 1)
    return (x_t.float() - torch.sqrt(1.0 - a) * eps_hat.float()) / torch.sqrt(a)


def gather_edge_deltas(x, edge_index):
    """x (B, N, C) -> (B, E, 3) displacement along each edge, xyz only."""
    i = edge_index[..., 0].unsqueeze(-1).expand(-1, -1, 3)
    j = edge_index[..., 1].unsqueeze(-1).expand(-1, -1, 3)
    xi = torch.gather(x[..., :3], 1, i)
    xj = torch.gather(x[..., :3], 1, j)
    return xj - xi


def edge_coherence_loss(x0_hat, x0_true, edge_index, edge_mask, abar_t,
                        huber_beta=0.05, weight_by_abar=True,
                        hinge_k=0.0, hinge_weight=0.0):
    """Displacement matching on the real vessel graph.

    Base term: Huber( (x0_hat[j]-x0_hat[i]) - (x0_gt[j]-x0_gt[i]) ).
    Matching the GT displacement (rather than driving displacements to zero)
    is what keeps long, legitimate edges long.

    Optional hinge (`hinge_weight` > 0) adds a one-sided penalty on edges
    longer than `hinge_k` x their GT length. It is ADDITIVE ONLY: on its own a
    hinge is satisfiable by a wrong configuration -- it reaches zero loss and
    stops caring while the tree is still badly scaled -- so it must never
    replace the base term.

    `huber_beta` is in NORMALISED units, not millimetres. With coord_std ~22 mm
    and a GT edge of ~0.47 mm (0.021 sigma), beta=0.5 is ~11 mm: every realistic
    residual then sits in the quadratic branch, so the Huber behaves as a scaled
    MSE and contributes no outlier robustness. That is not the same as being
    inert -- the loss still works, it just is not doing the job Huber was chosen
    for. beta=0.05 (~1.1 mm) is a PROPOSAL for this experiment, chosen so normal
    edges stay quadratic and broken ones cross into the linear branch. It has not
    been validated on real batches; treat it as a hyperparameter, and note that
    beta also rescales the loss magnitude (PyTorch divides by beta in the
    quadratic branch), so it is entangled with coh_weight -- fix beta first.

    Aggregation is per-sample mean THEN batch mean, so a 2,300-node tree does
    not outweigh a 300-node one.

    Computed in fp32 regardless of the autocast dtype: x0 reconstruction
    divides by sqrt(abar) ~ 0.0064 at high t, which bf16 cannot carry.

    Returns a scalar; 0-valued (graph-connected) when a batch has no edges.
    """
    x0_hat = x0_hat.float()
    x0_true = x0_true.float()

    d_hat = gather_edge_deltas(x0_hat, edge_index)          # (B, E, 3)
    d_gt = gather_edge_deltas(x0_true, edge_index)
    m = edge_mask.float()                                   # (B, E)

    per_edge = F.smooth_l1_loss(d_hat, d_gt, beta=huber_beta,
                                reduction="none").mean(dim=-1)      # (B, E)

    if hinge_weight > 0.0:
        over = (d_hat.norm(dim=-1) - hinge_k * d_gt.norm(dim=-1)).clamp(min=0.0)
        per_edge = per_edge + hinge_weight * over ** 2

    per_edge = per_edge * m
    if weight_by_abar:
        per_edge = per_edge * abar_t.float().view(-1, 1)

    # per-sample mean, then batch mean
    n_edges = m.sum(dim=1)                                  # (B,)
    valid = n_edges > 0
    if not bool(valid.any()):
        return x0_hat.sum() * 0.0
    per_sample = per_edge.sum(dim=1)[valid] / n_edges[valid]
    return per_sample.mean()


# --------------------------------------------------------------------------
def collate_edges(edge_lists, device=None):
    """list of (E_k, 2) int arrays -> (B, E_max, 2) long, (B, E_max) bool.

    Padded entries point at node 0 and are masked out, so they are gathered
    safely and contribute nothing.
    """
    B = len(edge_lists)
    E_max = max(1, max(len(e) for e in edge_lists))
    idx = torch.zeros(B, E_max, 2, dtype=torch.long)
    msk = torch.zeros(B, E_max, dtype=torch.bool)
    for b, e in enumerate(edge_lists):
        if len(e):
            t = torch.as_tensor(e, dtype=torch.long)
            idx[b, : len(e)] = t
            msk[b, : len(e)] = True
    if device is not None:
        idx, msk = idx.to(device), msk.to(device)
    return idx, msk


class EdgeCollate:
    """collate_fn that attaches padded graph edges, keyed on batch["sample"].

    A class, not a closure, so DataLoader workers can pickle it under any start
    method. Because it looks edges up by sample id, `dataset_v3_1.py` needs no
    changes at all.
    """

    def __init__(self, edge_map):
        self.edge_map = edge_map

    def __call__(self, items):
        from torch.utils.data._utils.collate import default_collate
        batch = default_collate(items)
        lists, missing = [], []
        for it in items:
            e = self.edge_map.get(it["sample"])
            if e is None:
                missing.append(it["sample"])
            lists.append(e if e is not None else [])
        if missing:
            raise KeyError(f"edge cache is missing {len(missing)} samples, "
                           f"e.g. {missing[:5]}")
        batch["edge_index"], batch["edge_mask"] = collate_edges(lists)
        return batch


def load_edge_cache(cache_path, packaged_dir, required_ids):
    """Load edges_v1.npz and PROVE it belongs to this dataset.

    Sample count alone cannot detect a stale cache. This re-hashes the actual
    GT coordinates, in packaged order, for every required sample and compares
    against the fingerprint recorded at build time. A cache built from other
    data, or from the same data in a different point order, is rejected here
    rather than silently mis-indexing the loss.
    """
    import hashlib
    import json
    from pathlib import Path as _Path

    import numpy as np

    cache_path = _Path(cache_path)
    if cache_path.is_dir():
        cache_path = cache_path / "edges_v1.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"edge cache not found: {cache_path}")
    if _Path(cache_path).parent.resolve() == _Path(packaged_dir).resolve():
        raise RuntimeError(
            f"{cache_path} sits inside the dataset directory. list_samples() "
            f"globs *.npz there and reads z['split'], which this file lacks. "
            f"Move the cache to its own directory.")

    with np.load(cache_path, allow_pickle=True) as z:
        meta = json.loads(str(z["_meta"]))
        edge_map = {k: z[k] for k in z.files if k != "_meta"}

    recorded = {r["sample"]: r for r in meta["rows"]}
    root = _Path(packaged_dir)
    checked = 0
    for sid in required_ids:
        if sid not in edge_map:
            raise KeyError(f"edge cache has no entry for {sid}")
        with np.load(root / f"{sid}.npz", allow_pickle=True) as z:
            m = z["centerline_mask"].astype(bool)
            gt = z["centerline"][m][:, :3].astype(np.float64)
        fp = hashlib.sha256(
            np.ascontiguousarray(gt, dtype=np.float64).tobytes()).hexdigest()[:16]
        if recorded[sid]["fingerprint"] != fp:
            raise RuntimeError(
                f"{sid}: edge cache fingerprint {recorded[sid]['fingerprint']} "
                f"does not match the dataset ({fp}). The cache was built from "
                f"different data or a different point ordering -- rebuild it.")
        e = edge_map[sid]
        if len(e) and (e.max() >= len(gt) or e.min() < 0):
            raise RuntimeError(f"{sid}: cached edge index out of range")
        checked += 1
    meta["n_verified"] = checked
    return edge_map, meta


def gradient_norm_ratio(model, eps_loss, coh_loss):
    """||d coh / d theta|| / ||d eps / d theta||, on one real batch.

    Pick `coh_weight` so that weight * ratio lands where you want it (0.2-0.5
    is a reasonable starting band). This is the number to use -- NOT the ratio
    of the loss values, which says nothing about the gradients.
    """
    def gnorm(loss):
        g = torch.autograd.grad(loss, [p for p in model.parameters()
                                       if p.requires_grad],
                                retain_graph=True, allow_unused=True)
        return torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)).item()
    ge, gc = gnorm(eps_loss), gnorm(coh_loss)
    return gc / max(ge, 1e-12), ge, gc


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------
def _tests():
    """Every perturbation below must CHANGE the edge deltas.

    An earlier version of these tests used `x0 + 0.05` -- a uniform
    translation. Edge deltas are differences, so a constant shift cancels
    exactly and the tests compared 1e-17 against 5e-22, i.e. pure numerical
    noise, while appearing to pass. Perturbations here are per-node.
    """
    torch.manual_seed(0)
    B, N = 4, 120
    node_mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        node_mask[b, : 60 + 15 * b] = True

    x0 = torch.cumsum(torch.randn(B, N, 4) * 0.021, dim=1)     # ~0.021 sigma edges
    abar = torch.full((B,), 0.9)

    edge_lists = []
    for b in range(B):
        n = int(node_mask[b].sum())
        e = [(k, k + 1) for k in range(n - 1)]
        e += [(5, 40), (12, 55), (20, n - 2)]        # graph-adjacent, array-distant
        edge_lists.append(e)
    ei, em = collate_edges(edge_lists)
    print(f"edges/sample {[len(e) for e in edge_lists]}, padded to {ei.shape[1]}")

    # -- 0. a uniform shift must be INVISIBLE (documents the old test bug) ----
    shifted = edge_coherence_loss(x0 + 0.05, x0, ei, em, abar).item()
    assert shifted < 1e-12, shifted
    print(f"uniform +0.05 shift -> {shifted:.2e}  (invisible by construction: "
          f"this is why the old weighting/hinge tests were vacuous)")

    # -- 1. perfect prediction -----------------------------------------------
    assert edge_coherence_loss(x0.clone(), x0, ei, em, abar).item() < 1e-12
    print("perfect prediction -> 0")

    # -- 2. monotone in PER-NODE noise ---------------------------------------
    prev, g = -1.0, torch.Generator().manual_seed(1)
    for j in (0.0, 0.005, 0.02, 0.05, 0.2):
        pert = x0 + torch.randn(x0.shape, generator=g) * j
        l = edge_coherence_loss(pert, x0, ei, em, abar).item()
        assert l > prev, (j, l, prev)
        prev = l
    print(f"monotone in per-node noise: OK (last {prev:.3e})")

    # -- 3. abar weighting, with a perturbation that actually moves edges -----
    noisy = x0 + torch.randn(x0.shape, generator=torch.Generator().manual_seed(2)) * 0.05
    lo = edge_coherence_loss(noisy, x0, ei, em, torch.full((B,), 4e-5)).item()
    hi = edge_coherence_loss(noisy, x0, ei, em, torch.full((B,), 0.99)).item()
    assert lo > 1e-8 and hi > 1e-8, (lo, hi)          # both must be REAL values
    assert hi / lo > 1e4
    print(f"abar weighting: t~999 {lo:.3e} vs t~0 {hi:.3e}  ({hi/lo:.0f}x) -- both non-trivial")

    # -- 4. hinge must ADD when an edge is genuinely over-long ---------------
    stretched = x0.clone()
    stretched[:, 40:, :3] += 0.8                      # edge (5,40) and (39,40) stretch
    base = edge_coherence_loss(stretched, x0, ei, em, abar,
                               hinge_k=3.0, hinge_weight=0.0).item()
    both = edge_coherence_loss(stretched, x0, ei, em, abar,
                               hinge_k=3.0, hinge_weight=0.1).item()
    assert base > 1e-6 and both > base * 1.0001, (base, both)
    print(f"hinge adds on over-long edges: base {base:.4e} -> +hinge {both:.4e}")

    # -- 5. padded edges cannot contribute, with an IMPERFECT prediction ------
    pert = x0 + torch.randn(x0.shape, generator=torch.Generator().manual_seed(3)) * 0.03
    l_ref = edge_coherence_loss(pert, x0, ei, em, abar).item()
    ei_junk = ei.clone()
    ei_junk[~em] = torch.randint(0, N, ei_junk[~em].shape)     # garbage in the PAD only
    l_junk = edge_coherence_loss(pert, x0, ei_junk, em, abar).item()
    assert abs(l_junk - l_ref) < 1e-9, (l_ref, l_junk)
    print(f"padded-edge isolation: {l_ref:.6e} vs {l_junk:.6e} with random pad indices")

    # -- 6. per-sample then batch mean: a big tree must not dominate ---------
    #    sample 3 has ~2x the edges of sample 0; give ONLY sample 0 the error.
    only0 = x0.clone()
    only0[0] += torch.randn(x0[0].shape, generator=torch.Generator().manual_seed(4)) * 0.1
    only3 = x0.clone()
    only3[3] += torch.randn(x0[3].shape, generator=torch.Generator().manual_seed(4)) * 0.1
    l0 = edge_coherence_loss(only0, x0, ei, em, abar).item()
    l3 = edge_coherence_loss(only3, x0, ei, em, abar).item()
    print(f"same error on small vs large tree: {l0:.4e} vs {l3:.4e}  "
          f"(ratio {max(l0,l3)/min(l0,l3):.2f}, would be ~2x with flat batch mean)")
    assert max(l0, l3) / min(l0, l3) < 1.35

    # -- 7. gradient survives on a far escapee -------------------------------
    far = x0.clone(); far[:, 40, :3] += 12.0
    far = far.requires_grad_(True)
    edge_coherence_loss(far, x0, ei, em, abar).backward()
    g40 = far.grad[:, 40, :3].abs().max().item()
    assert g40 > 1e-6
    print(f"gradient at a 12-sigma escapee: {g40:.5f} (non-zero -- no hard clamp)")

    # -- 8. beta: what Huber actually buys is a BOUNDED GRADIENT ------------
    #    Loss VALUE is larger for small beta on an outlier (PyTorch scales the
    #    quadratic branch by 1/beta), so asserting on the value is misleading.
    #    The property that matters: as a node escapes further, the gradient
    #    saturates under small beta and keeps growing under large beta.
    def esc_grad(dist, beta):
        z = x0.clone(); z[:, 40, :3] += dist
        z = z.requires_grad_(True)
        edge_coherence_loss(z, x0, ei, em, abar, huber_beta=beta).backward()
        return z.grad[:, 40, :3].abs().max().item()

    g_small = esc_grad(5.0, 0.05) , esc_grad(20.0, 0.05)
    g_large = esc_grad(5.0, 100.0), esc_grad(20.0, 100.0)
    r_small = g_small[1] / g_small[0]
    r_large = g_large[1] / g_large[0]
    assert r_small < 1.05, f"beta=0.05 gradient not saturated: ratio {r_small}"
    assert r_large > 3.0, f"beta=100 gradient did not grow: ratio {r_large}"
    print(f"escapee gradient 5->20 sigma: beta 0.05 ratio {r_small:.3f} (saturated) "
          f"| beta 100 ratio {r_large:.2f} (unbounded)")

    # quadratic branch scales as 1/beta when residuals are small
    small = x0 + torch.randn(x0.shape, generator=torch.Generator().manual_seed(5)) * 0.005
    b1 = edge_coherence_loss(small, x0, ei, em, abar, huber_beta=0.05).item()
    b2 = edge_coherence_loss(small, x0, ei, em, abar, huber_beta=0.10).item()
    assert abs(b1 / b2 - 2.0) < 0.15, (b1, b2, b1 / b2)
    print(f"quadratic-branch 1/beta scaling: {b1/b2:.3f} (expect ~2.0)")

    # -- 9. bf16 inputs must agree with fp32 to a stated tolerance -----------
    lb = edge_coherence_loss(noisy.bfloat16(), x0.bfloat16(), ei, em,
                             abar.bfloat16()).item()
    lf = edge_coherence_loss(noisy, x0, ei, em, abar).item()
    rel = abs(lb - lf) / lf
    assert rel < 0.02, f"bf16 vs fp32 differ by {rel:.2%}"
    print(f"bf16 {lb:.5e} vs fp32 {lf:.5e}  (rel {rel:.2%} < 2%)")

    # -- 10. x0_from_eps at HIGH t, where sqrt(abar) ~ 0.0064 ----------------
    #    This is the reconstruction the whole auxiliary loss depends on.
    betas = torch.linspace(1e-4, 0.02, 1000)
    abars = torch.cumprod(1.0 - betas, dim=0)
    for t in (0, 250, 500, 750, 999):
        a = abars[t].expand(B)
        eps = torch.randn(B, N, 4, generator=torch.Generator().manual_seed(6 + t))
        av = a.view(-1, 1, 1)
        x_t = torch.sqrt(av) * x0 + torch.sqrt(1 - av) * eps
        rec = x0_from_eps(x_t, eps, a)
        err = (rec - x0).abs().max().item()
        assert torch.isfinite(rec).all(), f"t={t}: non-finite reconstruction"
        assert err < 2e-3, f"t={t}: round-trip error {err}"
        # and in bf16, which is what the trainer actually produces
        rec_b = x0_from_eps(x_t.bfloat16(), eps.bfloat16(), a.bfloat16())
        assert torch.isfinite(rec_b).all(), f"t={t}: non-finite in bf16"
        print(f"  t={t:>3} sqrt(abar)={abars[t].sqrt():.4f}  "
              f"fp32 err {err:.2e}  bf16 err {(rec_b - x0).abs().max():.2e}")
    print("x0_from_eps verified across the full timestep range")

    print("\nall tests passed (every test asserts)")


if __name__ == "__main__":
    _tests()
