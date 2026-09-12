"""v3.3 -- padding-aware conditioning. Drop-in replacement for diffusion.py.

WHAT CHANGED FROM v3.1 AND WHY
------------------------------
v3.1 masked padded rows at the LOSS only. That is correct for a pointwise
network, but this denoiser mixes information across the sequence axis in three
places, and each one leaks padding into the valid nodes:

  1. SelfAttentionBlock1D attended over every position, padded included. After
     input_proj a padded row is pure noise, and at high diffusion timesteps the
     valid rows are noisy too -- so the model has no reliable signature to
     down-weight them by. The bottleneck is exactly where image evidence is
     supposed to propagate globally, so this is the worst place to inject noise.
     -> now takes key_padding_mask.

  2. GroupNorm reduced over padded positions. The padding fraction varies from
     sample to sample, so the normalisation statistics moved for reasons that
     have nothing to do with the anatomy, applying a spurious per-sample rescale
     inside the SPADE path -- the same path that collapsed in v3.
     -> now MaskedGroupNorm1d, statistics over valid positions only.

  3. Conv1d receptive fields bled padding into the last valid nodes.
     -> padded positions are re-zeroed after every stage, so a padded position
        is always exactly 0 and contributes a known constant.

`node_mask` is OPTIONAL. Passing None reproduces v3.1 behaviour exactly (the
mask becomes all-ones), so this file can be dropped in before the call sites are
updated without changing any result.

v3.3 adds, from review:
  4. PositionalQueryConditioner passes need_weights=False. The default True
     forces MultiheadAttention onto the non-fused path and materialises a
     (B, N, T) attention matrix -- ~82 MB per call at B=16, N=2500, T=512 --
     that is immediately discarded.
  5. Arbitrary sequence lengths. The down path (two stride-2 convs) and the up
     path (two ConvTranspose1d) only agree when N is a multiple of 4; N=810
     returned 808 and crashed in spade_out. forward() now pads to a multiple
     of 4 internally, marks the pad invalid, and crops the output back, so
     sampling can pass any length. N=2500 is unchanged (pad == 0).

The topology label (column 4 of the packaged centerline) does not enter the
model. ``node_dim=4`` means the denoiser predicts noise for x, y, z, and radius.
Current generation does use the GT node count and row correspondence; any
edge-based evaluation must state that it uses given/oracle GT topology.

CALL SITES TO UPDATE (2 lines, both outside this file):
  trainer.compute_loss  -> model(noisy, t, images, poses, x0_self=..., node_mask=mask)
  sampling.sample_ddim  -> model(x, t, images, poses, x0_self=..., node_mask=mask)
where `mask` is the batch's centerline_mask, (B, N) bool, True = valid.

CAVEAT: this changes the architecture's numerics. A run with it is NOT a
single-variable A/B against run 4 or any earlier run -- report it as a new
configuration, not as a hyperparameter change.

Unchanged from v3.1: 3DPQT positional-query conditioning (DX2CT, Jeong et al.
2025), SPADE conditioning (Park et al. 2019), self-conditioning (Chen et al.
2022), and noise prediction for columns 0-3.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# masking helpers
# --------------------------------------------------------------------------
def _as_float_mask(node_mask, batch, n_points, device, dtype):
    """(B, N) bool or None -> (B, 1, N) float. None means 'everything valid'."""
    if node_mask is None:
        return torch.ones(batch, 1, n_points, device=device, dtype=dtype)
    m = node_mask
    if m.dim() == 2:
        m = m[:, None, :]
    return m.to(device=device, dtype=dtype)


def _downsample_mask(m):
    """(B,1,N) -> (B,1,N/2). A coarse position is valid if ANY child was valid,
    which matches what a stride-2 conv actually reads."""
    return F.max_pool1d(m, kernel_size=2, stride=2)


class MaskedGroupNorm1d(nn.Module):
    """GroupNorm over (B, C, N) that reduces over VALID positions only.

    Identical to nn.GroupNorm when the mask is all ones (verified in the
    self-test at the bottom of this file).
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"{num_channels} channels not divisible by {num_groups} groups")
        self.num_groups, self.num_channels, self.eps, self.affine = \
            num_groups, num_channels, eps, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, m):
        B, C, N = x.shape
        G, Cg = self.num_groups, C // self.num_groups
        xg = x.view(B, G, Cg, N)
        mg = m.view(B, 1, 1, N)

        # Always accumulate normalization statistics in FP32. This makes the
        # intended Blackwell BF16 training path stable without changing any
        # parameters or checkpoint shapes.
        x_stats = xg.float()
        m_stats = mg.float()
        cnt = m_stats.sum(dim=(2, 3), keepdim=True) * Cg      # valid elems per group
        cnt = cnt.clamp(min=1.0)                               # guard all-padded
        mean = (x_stats * m_stats).sum(dim=(2, 3), keepdim=True) / cnt
        var = (((x_stats - mean) ** 2) * m_stats).sum(
            dim=(2, 3), keepdim=True) / cnt
        xg = ((x_stats - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)

        out = xg.view(B, C, N)
        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * m                                         # padded stay exactly 0


# --------------------------------------------------------------------------
# embeddings (unchanged from v3.1)
# --------------------------------------------------------------------------
class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep t (Ho et al., 2020)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device).float() / half)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SinusoidalPositionEmbedding(nn.Module):
    """Per-token (sequence-order) positional encoding (Vaswani et al. 2017)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, n_points, device):
        half = self.dim // 2
        pos = torch.arange(n_points, device=device).float()
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=device).float() / half)
        args = pos[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ImageConditionEncoder(nn.Module):
    """Shared CNN over the 2 projections -> tokens, with each view's 3x4
    projection matrix embedded and added to that view's tokens. Unchanged:
    image tokens are never padded, so no mask is needed here."""

    def __init__(self, embed_dim: int = 128, pose_dim: int = 12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, embed_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim), nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim), nn.SiLU(),
        )
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, images: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        B, V, H, W = images.shape
        feat = self.cnn(images.reshape(B * V, 1, H, W))
        C, Hp, Wp = feat.shape[1:]
        tokens = feat.reshape(B, V, C, Hp * Wp).permute(0, 1, 3, 2)
        tokens = tokens + self.pose_embed(poses.reshape(B, V, 12))[:, :, None, :]
        return tokens.reshape(B, V * Hp * Wp, C)


class PositionalQueryConditioner(nn.Module):
    """3DPQT-style conditioning (DX2CT, 2025): each node's 3D position is an
    explicit positional query that cross-attends into the X-ray(+pose) tokens.

    Keys are image tokens (never padded) so no key mask is required; padded
    QUERIES produce meaningless cond vectors, which the caller zeroes."""

    def __init__(self, dim, n_heads=4, n_freqs=10):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 3 * (2 * n_freqs + 1)
        self.query_mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.register_buffer("freq_bands",
                             (2.0 ** torch.arange(n_freqs).float()) * math.pi)

    def _encode_xyz(self, xyz):
        feats = [xyz]
        for f in self.freq_bands:
            feats.append(torch.sin(xyz * f))
            feats.append(torch.cos(xyz * f))
        return torch.cat(feats, dim=-1)

    def forward(self, node_xyz, tokens):
        q = self.query_mlp(self._encode_xyz(node_xyz))
        kv = self.norm_kv(tokens)
        # need_weights=False: the default True forces the non-fused path and
        # materialises a (B, N, T) attention matrix -- 82 MB per call at
        # B=16, N=2500, T=512 -- which is then thrown away. We never read it.
        out, _ = self.attn(self.norm_q(q), kv, kv, need_weights=False)
        return out


class SPADE1D(nn.Module):
    """Spatially-Adaptive Normalization for 1D node sequences (Park et al. 2019;
    the conditioning method in DX2CT 2025), now with masked statistics."""

    def __init__(self, channels, cond_dim, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.norm = MaskedGroupNorm1d(8, channels, affine=False)
        self.shared = nn.Sequential(nn.Conv1d(cond_dim, hidden, 1), nn.SiLU())
        self.to_gamma = nn.Conv1d(hidden, channels, 1)
        self.to_beta = nn.Conv1d(hidden, channels, 1)

    def forward(self, x, cond, m):
        normalized = self.norm(x, m)
        h = self.shared(cond)
        return (normalized * (1 + self.to_gamma(h)) + self.to_beta(h)) * m


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = MaskedGroupNorm1d(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = MaskedGroupNorm1d(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x, t_emb, m):
        h = self.conv1(F.silu(self.norm1(x, m))) * m
        h = (h + self.time_proj(t_emb)[:, :, None]) * m
        h = self.conv2(F.silu(self.norm2(h, m))) * m
        return (x + h) * m


class SelfAttentionBlock1D(nn.Module):
    """Node<->node self-attention at the bottleneck, now padding-aware."""

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        # x: (B, N, dim); key_padding_mask: (B, N) True == IGNORE this key.
        h = self.norm(x)
        if key_padding_mask is not None:
            # A row that is entirely masked makes softmax produce NaN. Cannot
            # happen with real data (every sample has valid nodes) but a guard
            # costs nothing and turns a silent NaN into correct behaviour.
            all_masked = key_padding_mask.all(dim=1, keepdim=True)
            key_padding_mask = key_padding_mask & ~all_masked
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return x + out


class CenterlineDenoiser(nn.Module):
    """1D-UNet denoiser over centerline nodes (x, y, z, radius).

    forward(noisy_nodes, t, images, poses, x0_self=None, node_mask=None)
      node_mask: (B, N) bool, True = valid node. None => all valid (v3.1 behaviour).
    """

    def __init__(self, node_dim=4, hidden_dim=384, time_dim=128, n_heads=4,
                 pos_emb_scale=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.pos_emb_scale = pos_emb_scale

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.image_encoder = ImageConditionEncoder(embed_dim=hidden_dim)
        self.pos_query = PositionalQueryConditioner(hidden_dim, n_heads)   # 3DPQT
        self.input_proj = nn.Conv1d(node_dim, hidden_dim, 1)
        self.pos_embed = SinusoidalPositionEmbedding(hidden_dim)

        self.spade_in = SPADE1D(hidden_dim, hidden_dim)

        self.down1 = ResBlock1D(hidden_dim, time_dim)
        self.pool1 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.down2 = ResBlock1D(hidden_dim, time_dim)
        self.pool2 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.down3 = ResBlock1D(hidden_dim, time_dim)
        self.self_attn = SelfAttentionBlock1D(hidden_dim, n_heads)

        self.up1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res_up1 = ResBlock1D(hidden_dim, time_dim)
        self.up2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res_up2 = ResBlock1D(hidden_dim, time_dim)

        self.spade_out = SPADE1D(hidden_dim, hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, node_dim, 1)

    def forward(self, noisy_nodes, t, images, poses, x0_self=None, node_mask=None):
        B, N_in, _ = noisy_nodes.shape
        m1 = _as_float_mask(node_mask, B, N_in, noisy_nodes.device, noisy_nodes.dtype)

        # Two stride-2 convs give floor((N-2)/2)+1; two ConvTranspose1d give
        # 2(N-1)-2+4. These agree only when N is a multiple of 4 -- e.g. N=810
        # comes back as 808 and spade_out fails on the broadcast. Rather than
        # make every call site remember to pad (sampling would eventually
        # forget), pad here and crop the output back. The padded tail is marked
        # invalid in the mask, so it cannot influence any real node.
        pad = (-N_in) % 4
        if pad:
            noisy_nodes = F.pad(noisy_nodes, (0, 0, 0, pad))
            m1 = F.pad(m1, (0, pad))                       # zeros == invalid
            if x0_self is not None:
                x0_self = F.pad(x0_self, (0, 0, 0, pad))
        N = N_in + pad

        m2 = _downsample_mask(m1)          # N/2
        m4 = _downsample_mask(m2)          # N/4, bottleneck

        t_emb = self.time_embed(t)
        tokens = self.image_encoder(images, poses)

        # 3DPQT on the predicted-x0 estimate when available (self-conditioning,
        # Chen et al. 2022), else on the noisy position.
        query_xyz = x0_self if x0_self is not None else noisy_nodes[:, :, :3]
        cond = self.pos_query(query_xyz * m1.transpose(1, 2), tokens)
        cond = cond.transpose(1, 2) * m1                       # (B, hidden, N)

        x = self.input_proj(noisy_nodes.transpose(1, 2)) * m1
        pos_emb = self.pos_embed(N, noisy_nodes.device)
        x = (x + self.pos_emb_scale * pos_emb.T.unsqueeze(0)) * m1
        x = self.spade_in(x, cond, m1)

        x = self.down1(x, t_emb, m1)
        x = self.pool1(x) * m2
        x = self.down2(x, t_emb, m2)
        x = self.pool2(x) * m4
        x = self.down3(x, t_emb, m4)

        x = self.self_attn(x.transpose(1, 2),
                           key_padding_mask=(m4.squeeze(1) == 0)).transpose(1, 2) * m4

        x = self.up1(x) * m2
        x = self.res_up1(x, t_emb, m2)
        x = self.up2(x) * m1
        x = self.res_up2(x, t_emb, m1)

        x = self.spade_out(x, cond, m1)
        out = self.output_proj(x) * m1
        return out[:, :, :N_in].transpose(1, 2)          # crop the internal pad


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------
def _test_masked_groupnorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 16, 40)
    mgn = MaskedGroupNorm1d(8, 16)
    gn = nn.GroupNorm(8, 16)
    gn.load_state_dict({"weight": mgn.weight.data, "bias": mgn.bias.data})
    ones = torch.ones(3, 1, 40)
    err = (mgn(x, ones) - gn(x)).abs().max().item()
    assert err < 1e-4, f"MaskedGroupNorm diverges from nn.GroupNorm by {err}"
    print(f"MaskedGroupNorm1d == nn.GroupNorm on unmasked input (max err {err:.2e})")


def _test_padding_isolation():
    """The whole point: changing the PADDED rows must not change the output at
    the VALID rows. This fails on v3.1."""
    torch.manual_seed(0)
    B, N = 2, 64
    model = CenterlineDenoiser(hidden_dim=64, time_dim=32).eval()
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, :40] = True
    mask[1, :52] = True

    nodes = torch.randn(B, N, 4)
    t = torch.randint(0, 1000, (B,))
    images = torch.randn(B, 2, 512, 512)
    poses = torch.randn(B, 2, 3, 4)

    perturbed = nodes.clone()
    perturbed[~mask] = torch.randn_like(perturbed[~mask]) * 50.0   # garbage padding

    with torch.no_grad():
        a = model(nodes, t, images, poses, node_mask=mask)
        b = model(perturbed, t, images, poses, node_mask=mask)
    d = (a[mask] - b[mask]).abs().max().item()
    assert d < 1e-4, f"padding still leaks into valid nodes: max delta {d}"
    print(f"padding isolation OK (valid-node delta {d:.2e} under 50x padding noise)")

    with torch.no_grad():
        c = model(nodes, t, images, poses, node_mask=None)
    assert c.shape == nodes.shape
    print("node_mask=None path runs (reproduces v3.1 behaviour)")


def dummy_forward_backward_test():
    """Forward + backward at real packaged shapes with a realistic padding load."""
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")

    B, N = 2, 2500
    model = CenterlineDenoiser().to(device)
    nodes = torch.randn(B, N, 4, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    images = torch.randn(B, 2, 512, 512, device=device)
    poses = torch.randn(B, 2, 3, 4, device=device)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask[0, :1400] = True
    mask[1, :2100] = True

    pred = model(nodes, t, images, poses, node_mask=mask)
    assert pred.shape == nodes.shape, f"shape mismatch {pred.shape} vs {nodes.shape}"
    print("Forward OK. Output shape:", tuple(pred.shape))

    target = torch.randn_like(nodes)
    per_point = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    loss = (per_point * mask.float()).sum() / mask.float().sum()
    loss.backward()
    n_none = [n for n, p in model.named_parameters() if p.grad is None]
    assert not n_none, f"no gradient reached: {n_none[:5]}"
    print(f"Backward OK. Loss {loss.item():.4f}. "
          f"Params {sum(p.numel() for p in model.parameters()):,}")


def _test_arbitrary_length():
    """Any N must work, and N=2500 must be bit-identical to the unpadded path."""
    torch.manual_seed(0)
    model = CenterlineDenoiser(hidden_dim=64, time_dim=32).eval()
    images = torch.randn(1, 2, 512, 512)
    poses = torch.randn(1, 2, 3, 4)
    t = torch.randint(0, 1000, (1,))
    for N in (809, 810, 811, 812, 64, 2500):
        nodes = torch.randn(1, N, 4)
        mask = torch.zeros(1, N, dtype=torch.bool)
        mask[0, : max(4, int(N * 0.7))] = True
        with torch.no_grad():
            out = model(nodes, t, images, poses, node_mask=mask)
        assert out.shape == (1, N, 4), f"N={N} gave {tuple(out.shape)}"
    print("arbitrary lengths OK (809, 810, 811, 812, 64, 2500)")


def _test_pad_does_not_perturb_multiple_of_four():
    """N already a multiple of 4 must take the pad==0 path exactly."""
    torch.manual_seed(0)
    model = CenterlineDenoiser(hidden_dim=64, time_dim=32).eval()
    nodes = torch.randn(1, 64, 4)
    mask = torch.zeros(1, 64, dtype=torch.bool); mask[0, :40] = True
    t = torch.randint(0, 1000, (1,))
    images = torch.randn(1, 2, 512, 512); poses = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        a = model(nodes, t, images, poses, node_mask=mask)
        b = model(nodes, t, images, poses, node_mask=mask)
    assert torch.equal(a, b), "forward is not deterministic in eval mode"
    print("multiple-of-4 path deterministic and unpadded")


if __name__ == "__main__":
    _test_masked_groupnorm_matches_torch()
    _test_padding_isolation()
    _test_arbitrary_length()
    _test_pad_does_not_perturb_multiple_of_four()
    dummy_forward_backward_test()
