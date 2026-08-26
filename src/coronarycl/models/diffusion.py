"""Step 2.2 -- conditional diffusion model: design & implementation.
Develop/debug locally on M4 via PyTorch MPS backend (small batch).
Full-scale training happens in Step 2.3 (Kaggle).

v3 -- conditioning overhaul aimed at the conditioning-collapse wall.
Two changes, both adapted from DX2CT (Jeong et al., 2025, 3D CT recon
from bi/mono-planar X-rays -- the same 2D-X-ray -> 3D conditioning
problem we face):

  1. 3D Positional-Query conditioning (their "3DPQT"): instead of using
     the node's hidden features as the cross-attention query, we use the
     node's CURRENT 3D position as an explicit positional query that
     attends into the X-ray(+pose) tokens. This makes the 3D->2D lookup
     explicit and learnable, rather than hoping generic cross-attention
     discovers epipolar geometry on its own.

  2. SPADE conditioning (spatially-adaptive normalization) instead of
     additive injection (x = x + cond). DX2CT show plainly that simple
     concatenation/addition "did not fully utilize semantic information,
     leading to sub-optimal results," and that SPADE is better. The
     per-node conditioning now modulates the denoiser's normalization
     scale/bias, which is much harder for the model to ignore than an
     added tensor (our prior failure mode).

Conditioning data unchanged: still (images, poses) already in the
packaged .npz -- no data-prep / DRR / centerline / baseline changes.
Geometry (poses) still flows in via the pose-embedded X-ray tokens.

Topology (column 4) is fixed/given -- the denoiser predicts noise only
for columns 0-3 (x, y, z, radius). Padded rows are excluded from the
loss (masking at the loss level), not inside the architecture.

v3.1 -- adds SELF-CONDITIONING. v3 (SPADE + 3DPQT) collapsed to 0.0
conditioning sensitivity because the 3DPQT query ran on the *noisy* node
position (near-random at high timesteps), feeding SPADE garbage; SPADE
being invasive, the model learned to zero it out. Fix: the query now runs
on the predicted-x0 estimate (Chen et al. 2022, self-conditioning), which
is a meaningful position at every timestep. `forward` takes an optional
`x0_self`; when given, 3DPQT queries on it, else on the noisy position.
Training draws x0_self via a no-grad pass with prob 0.5 (see
trainer.compute_loss); sampling carries it across DDIM steps (see
sampling.sample_ddim).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep t (Ho et al., 2020)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half,
                                            device=t.device).float() / half
        )
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
        return emb  # (n_points, hidden_dim)


class ImageConditionEncoder(nn.Module):
    """CNN encoder for the 2 conditioning projections. Each view is encoded
    with a shared CNN to a compact spatial map, flattened into tokens.
    Each view's projection matrix (3x4) is embedded and added to that view's
    tokens, so the geometry (epipolar relationship) rides along with the
    image features -- the 3DPQT below then queries into these tokens by 3D
    position.
    """

    def __init__(self, embed_dim: int = 128, pose_dim: int = 12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.GroupNorm(
                4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.GroupNorm(
                8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GroupNorm(
                8, 64), nn.SiLU(),
            nn.Conv2d(64, embed_dim, 4, stride=2, padding=1), nn.GroupNorm(
                8, embed_dim), nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 4, stride=2,
                      padding=1), nn.GroupNorm(8, embed_dim), nn.SiLU(),
        )
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, embed_dim), nn.SiLU(
            ), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, images: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """images: (B, 2, H, W), poses: (B, 2, 3, 4) -> (B, T, embed_dim), T = 2*(H/32)*(W/32)."""
        B, V, H, W = images.shape
        x = images.reshape(B * V, 1, H, W)
        feat = self.cnn(x)
        C, Hp, Wp = feat.shape[1:]
        tokens = feat.reshape(B, V, C, Hp * Wp).permute(0, 1, 3, 2)
        pose_flat = poses.reshape(B, V, 12)
        pose_tok = self.pose_embed(pose_flat)
        tokens = tokens + pose_tok[:, :, None, :]
        return tokens.reshape(B, V * Hp * Wp, C)


class PositionalQueryConditioner(nn.Module):
    """3DPQT-style conditioning (DX2CT, 2025). Each node's CURRENT 3D
    position becomes an explicit positional query (NeRF-style Fourier
    features + MLP) that cross-attends into the X-ray(+pose) tokens, and
    returns a per-node, position-aware conditioning vector.

    This is the learnable, position-grounded replacement for generic
    node-feature cross-attention: the model is told *where* each point is
    in 3D and asked to fetch the matching X-ray evidence.
    """

    def __init__(self, dim, n_heads=4, n_freqs=10):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 3 * (2 * n_freqs + 1)  # xyz + sin/cos at n_freqs bands
        self.query_mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.register_buffer(
            "freq_bands", (2.0 ** torch.arange(n_freqs).float()) * math.pi)

    def _encode_xyz(self, xyz):
        # xyz: (B, N, 3) -> (B, N, 3*(2F+1))
        feats = [xyz]
        for f in self.freq_bands:
            feats.append(torch.sin(xyz * f))
            feats.append(torch.cos(xyz * f))
        return torch.cat(feats, dim=-1)

    def forward(self, node_xyz, tokens):
        # node_xyz: (B, N, 3) ; tokens: (B, T, dim) -> (B, N, dim)
        q = self.query_mlp(self._encode_xyz(node_xyz))
        kv = self.norm_kv(tokens)
        out, _ = self.attn(self.norm_q(q), kv, kv)
        return out


class SPADE1D(nn.Module):
    """Spatially-Adaptive Normalization for 1D node sequences (Park et al.
    2019; used as the diffusion conditioning method in DX2CT, 2025). The
    per-node conditioning produces a per-node scale (gamma) and bias (beta)
    that modulate the (affine-free) group-normalized features. Much harder
    to ignore than additive injection -- targets our conditioning collapse.
    """

    def __init__(self, channels, cond_dim, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.norm = nn.GroupNorm(8, channels, affine=False)
        self.shared = nn.Sequential(nn.Conv1d(cond_dim, hidden, 1), nn.SiLU())
        self.to_gamma = nn.Conv1d(hidden, channels, 1)
        self.to_beta = nn.Conv1d(hidden, channels, 1)

    def forward(self, x, cond):
        # x: (B, C, N) ; cond: (B, cond_dim, N)
        normalized = self.norm(x)
        h = self.shared(cond)
        return normalized * (1 + self.to_gamma(h)) + self.to_beta(h)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SelfAttentionBlock1D(nn.Module):
    """Node<->node self-attention at the bottleneck (global receptive field)."""

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x)
        out, _ = self.attn(h, h, h)
        return x + out


class CenterlineDenoiser(nn.Module):
    """1D-UNet denoiser over centerline nodes (x, y, z, radius).

    v3 conditioning: per-node position-aware conditioning (PositionalQuery
    Conditioner / 3DPQT) injected via SPADE at full resolution, at both the
    input and the output stage of the UNet. Bottleneck self-attention kept
    for global node<->node reasoning. Sequence positional embedding kept
    (down-scaled) for list-order. The old additive cross-attention
    conditioning is replaced by the SPADE path.
    """

    def __init__(self, node_dim=4, hidden_dim=384, time_dim=128, n_heads=4,
                 pos_emb_scale=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.pos_emb_scale = pos_emb_scale

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(
            ), nn.Linear(time_dim, time_dim),
        )
        self.image_encoder = ImageConditionEncoder(embed_dim=hidden_dim)
        self.pos_query = PositionalQueryConditioner(
            hidden_dim, n_heads)   # 3DPQT
        self.input_proj = nn.Conv1d(node_dim, hidden_dim, 1)
        self.pos_embed = SinusoidalPositionEmbedding(hidden_dim)

        # SPADE inject (input)
        self.spade_in = SPADE1D(hidden_dim, hidden_dim)

        # Level 1 (N)
        self.down1 = ResBlock1D(hidden_dim, time_dim)
        self.pool1 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        # Level 2 (N/2)
        self.down2 = ResBlock1D(hidden_dim, time_dim)
        self.pool2 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        # Level 3 / bottleneck (N/4)
        self.down3 = ResBlock1D(hidden_dim, time_dim)
        self.self_attn = SelfAttentionBlock1D(hidden_dim, n_heads)

        self.up1 = nn.ConvTranspose1d(
            hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res_up1 = ResBlock1D(hidden_dim, time_dim)
        self.up2 = nn.ConvTranspose1d(
            hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res_up2 = ResBlock1D(hidden_dim, time_dim)

        # SPADE inject (output)
        self.spade_out = SPADE1D(hidden_dim, hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, node_dim, 1)

    def forward(self, noisy_nodes, t, images, poses, x0_self=None):
        t_emb = self.time_embed(t)
        # (B, T, hidden)
        tokens = self.image_encoder(images, poses)

        # 3DPQT: per-node, position-aware conditioning from X-ray tokens
        # (B, N, hidden)
        # Self-conditioning (Chen et al. 2022): query 3DPQT on the predicted-x0
        # estimate when available (meaningful position at every timestep), else
        # the noisy position -- this is what SPADE needs to stop collapsing.
        query_xyz = x0_self if x0_self is not None else noisy_nodes[:, :, :3]
        cond = self.pos_query(query_xyz, tokens)
        # (B, hidden, N)
        cond = cond.transpose(1, 2)

        x = self.input_proj(noisy_nodes.transpose(
            1, 2))                   # (B, hidden, N)
        pos_emb = self.pos_embed(noisy_nodes.shape[1], noisy_nodes.device)
        # sequence-order hint
        x = x + self.pos_emb_scale * pos_emb.T.unsqueeze(0)
        # SPADE conditioning
        x = self.spade_in(x, cond)

        x = self.down1(x, t_emb)
        x = self.pool1(x)
        x = self.down2(x, t_emb)
        x = self.pool2(x)
        x = self.down3(x, t_emb)

        x_tok = x.transpose(1, 2)
        x_tok = self.self_attn(x_tok)
        x = x_tok.transpose(1, 2)

        x = self.up1(x)
        x = self.res_up1(x, t_emb)
        x = self.up2(x)
        x = self.res_up2(x, t_emb)

        # SPADE conditioning
        x = self.spade_out(x, cond)
        return self.output_proj(x).transpose(1, 2)


def dummy_forward_backward_test():
    """Sanity check: forward + backward on dummy data at real packaged shapes:
    centerline (B, 2884, 5), images (B, 2, 512, 512), poses (B, 2, 3, 4).
    Runs on M4 CPU/MPS without CUDA. DoD for Step 2.2.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on device: {device}")

    B, N = 2, 2884
    model = CenterlineDenoiser().to(device)

    noisy_nodes = torch.randn(B, N, 4, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    images = torch.randn(B, 2, 512, 512, device=device)
    poses = torch.randn(B, 2, 3, 4, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    pred_noise = model(noisy_nodes, t, images, poses)
    assert pred_noise.shape == noisy_nodes.shape, f"Shape mismatch: {pred_noise.shape} vs {noisy_nodes.shape}"
    print("Forward pass OK. Output shape:", pred_noise.shape)

    target_noise = torch.randn_like(noisy_nodes)
    per_point_loss = F.mse_loss(
        pred_noise, target_noise, reduction="none").mean(dim=-1)
    loss = (per_point_loss * mask.float()).sum() / mask.float().sum()
    loss.backward()

    print(f"Backward pass OK. Loss: {loss.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    dummy_forward_backward_test()
