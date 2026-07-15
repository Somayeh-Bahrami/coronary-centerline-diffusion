"""Step 2.2 — conditional diffusion model: design & implementation.
Develop/debug locally on M4 via PyTorch MPS backend (small batch).
Full-scale training happens in Step 2.3 (Colab).

Centerline-native diffusion architecture, following AortaDiff's
(2025, arXiv:2507.13404) centerline-diffusion design: a denoiser over
centerline nodes (x, y, z, radius), conditioned on both 2D projections
and their projection matrices via cross-attention, so the model can
reason about epipolar geometry between the two views.

Topology (branch structure) is treated as fixed/given -- diffusion
denoises node positions and radii only, not tree connectivity.
"""

import torch
import torch.nn as nn


class ImageConditionEncoder(nn.Module):
    """CNN encoder for a single 2D projection view. Produces a feature
    map used for cross-attention conditioning in the denoiser.
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()
        # TODO: replace with a real backbone (e.g. small ResNet).
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (B, C, H', W')


class CenterlineDenoiser(nn.Module):
    """Denoises noisy centerline nodes (x, y, z, radius) at diffusion
    step t, conditioned on both views' image features + projection
    matrices via cross-attention.

    TODO: implement as a 1D-UNet or GNN over centerline nodes (see
    AortaDiff, arXiv:2507.13404, for the reference design this follows).
    """

    def __init__(self, node_dim: int = 4, cond_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageConditionEncoder(out_channels=cond_dim)
        # TODO: cross-attention block(s) between node embeddings and
        # image features, conditioned on projection-matrix embeddings.
        self.node_proj = nn.Linear(node_dim, cond_dim)

    def forward(self, noisy_nodes, t, views, proj_matrices):
        """
        Args:
            noisy_nodes: (B, N, 4) noisy (x, y, z, radius) at step t.
            t: (B,) diffusion timestep.
            views: (B, 2, 1, H, W) the 2 conditioning projections.
            proj_matrices: (B, 2, ...) projection geometry for each view.

        Returns:
            predicted noise, same shape as noisy_nodes.
        """
        raise NotImplementedError(
            "Architecture TODO — see docs/work_breakdown.md Step 2.2. "
            "Debug locally on M4 (MPS) with a small dummy batch first."
        )


def dummy_forward_smoke_test():
    """Sanity check: forward pass on a tiny dummy batch. Should run on
    M4 CPU/MPS without CUDA. This is the DoD for Step 2.2 — a clean run
    here (once forward() is implemented) satisfies it.
    """
    from ..config import resolve_device

    device = resolve_device()
    model = CenterlineDenoiser().to(device)
    noisy_nodes = torch.randn(2, 64, 4, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    views = torch.randn(2, 2, 1, 256, 256, device=device)
    proj_matrices = torch.randn(2, 2, 12, device=device)
    out = model(noisy_nodes, t, views, proj_matrices)
    assert out.shape == noisy_nodes.shape
    print("Smoke test passed on", device)
