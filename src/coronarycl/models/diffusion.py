"""Step 2.2 -- conditional diffusion model: design & implementation.
Develop/debug locally on M4 via PyTorch MPS backend (small batch).
Full-scale training happens in Step 2.3 (Kaggle).

Centerline-native diffusion architecture, following AortaDiff's
(2025, arXiv:2507.13404) centerline-diffusion design: a denoiser over
centerline nodes (x, y, z, radius), conditioned on both 2D projections
and their projection matrices via cross-attention, so the model can
reason about epipolar geometry between the two views.

Topology (branch structure, column 4 of the packaged centerline array)
is treated as fixed and given -- the denoiser only predicts noise for
columns 0-3 (x, y, z, radius), never touching the topology label.
Padded rows (per centerline_mask, see Step 1.3) are excluded from the
training loss but still flow through the network -- masking is applied
at the loss level, not inside the architecture, matching standard
practice for padded-sequence diffusion models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep t,
    following the original DDPM (Ho et al., 2020) formulation.
    """

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


class ImageConditionEncoder(nn.Module):
    """CNN encoder for the 2 conditioning projections. Each view is
    encoded independently with a shared CNN, downsampling to a compact
    spatial feature map, then flattened into a set of tokens for
    cross-attention. Each view's projection matrix (3x4 = 12 numbers,
    flattened) is embedded and added to that view's tokens, so the
    model can reason about epipolar geometry between the two views --
    this is our own conditioning-mechanism design, adapting AortaDiff's
    ViT-on-3D-volume conditioning to our 2D-projection setup.
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
        """
        Args:
            images: (B, 2, H, W) -- the 2 conditioning projections.
            poses: (B, 2, 3, 4) -- calibrated projection matrices.

        Returns:
            (B, T, embed_dim) conditioning tokens, T = 2 * (H/32) * (W/32).
        """
        B, V, H, W = images.shape
        x = images.reshape(B * V, 1, H, W)
        feat = self.cnn(x)
        C, Hp, Wp = feat.shape[1:]
        tokens = feat.reshape(B, V, C, Hp * Wp).permute(0, 1, 3, 2)

        pose_flat = poses.reshape(B, V, 12)
        pose_tok = self.pose_embed(pose_flat)
        tokens = tokens + pose_tok[:, :, None, :]

        return tokens.reshape(B, V * Hp * Wp, C)


class CrossAttentionBlock(nn.Module):
    """Centerline nodes attend to image conditioning tokens."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        out, _ = self.attn(q, kv, kv)
        return x + out


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


class CenterlineDenoiser(nn.Module):
    """1D-UNet denoiser over centerline nodes (x, y, z, radius),
    conditioned on both 2D projections via cross-attention, following
    AortaDiff's (arXiv:2507.13404) centerline-diffusion design.
    """

    def __init__(self, node_dim: int = 4, hidden_dim: int = 128,
                 time_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(
            ), nn.Linear(time_dim, time_dim),
        )

        self.image_encoder = ImageConditionEncoder(embed_dim=hidden_dim)

        self.input_proj = nn.Conv1d(node_dim, hidden_dim, 1)

        self.down1 = ResBlock1D(hidden_dim, time_dim)
        self.pool = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.down2 = ResBlock1D(hidden_dim, time_dim)

        self.cross_attn = CrossAttentionBlock(hidden_dim, n_heads)

        self.up = nn.ConvTranspose1d(
            hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.up1 = ResBlock1D(hidden_dim, time_dim)

        self.output_proj = nn.Conv1d(hidden_dim, node_dim, 1)

    def forward(self, noisy_nodes: torch.Tensor, t: torch.Tensor,
                images: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_nodes: (B, N, 4) noisy (x, y, z, radius) at step t.
                N must be even (pooled/upsampled by factor 2 here).
            t: (B,) diffusion timestep.
            images: (B, 2, H, W) the 2 conditioning projections.
            poses: (B, 2, 3, 4) projection matrices.

        Returns:
            predicted noise, (B, N, 4), same shape as noisy_nodes.
        """
        t_emb = self.time_embed(t)
        cond_tokens = self.image_encoder(images, poses)

        x = noisy_nodes.transpose(1, 2)
        x = self.input_proj(x)

        x = self.down1(x, t_emb)
        x = self.pool(x)
        x = self.down2(x, t_emb)

        x_tok = x.transpose(1, 2)
        x_tok = self.cross_attn(x_tok, cond_tokens)
        x = x_tok.transpose(1, 2)

        x = self.up(x)
        x = self.up1(x, t_emb)

        out = self.output_proj(x)
        return out.transpose(1, 2)


def dummy_forward_backward_test():
    """Sanity check: forward + backward pass on a small dummy batch,
    matching the REAL packaged-data shapes (see src/coronarycl/dataset.py):
    centerline (B, 2884, 5), images (B, 2, 512, 512), poses (B, 2, 3, 4).
    Should run on M4 CPU/MPS without CUDA. This is the DoD for Step 2.2.
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
