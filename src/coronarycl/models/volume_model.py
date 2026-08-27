"""Stage 1 (Option B) v3: X-ray -> 3D vessel volume, geometry-aware
back-projection conditioning (DX2CT-style diffusion).

v3 changes vs v2 (for 128-cubed training):
  - 4-level U-Net (was 3): input res -> res/16 bottleneck. res must be
    divisible by 16 (64, 128 both OK).
  - base widened default 32.
  - optional gradient checkpointing (use_checkpoint=True) to fit 128^3 on 16 GB.
Geometry unchanged and validated: P maps center-origin mm -> pixel(col,row);
mm = svoxel*((idx+0.5)/res - 0.5); projective grid_sample (views 30 deg apart).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SENSOR = 512  # projection-matrix pixel space (drr.py SENSOR_SIZE)


class SinusoidalTime(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim

    def forward(self, t):
        h = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(h,
                      device=t.device).float() / h)
        a = t.float()[:, None] * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class XrayFeat(nn.Module):
    """One X-ray (B,1,H,W) -> full-frame feature map (B,C,H,W)."""

    def __init__(self, C=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, C, 3, 1, 1))

    def forward(self, x): return self.net(x)


def backproject(feat, P, svoxel, res):
    """feat (B,C,Hf,Wf); P (B,3,4) center-mm->pixel(col,row); svoxel (B,3) mm.
    Returns per-voxel sampled features (B,C,res,res,res)."""
    B, C, Hf, Wf = feat.shape
    dev = feat.device
    q = torch.arange(res, device=dev).float()
    gi, gj, gk = torch.meshgrid(q, q, q, indexing="ij")
    coord = torch.stack([gi, gj, gk], -1).reshape(-1, 3)
    frac = (coord + 0.5) / res - 0.5
    mm = frac[None] * svoxel[:, None, :]
    Xh = torch.cat([mm, torch.ones(B, mm.shape[1], 1, device=dev)], -1)
    pix = torch.einsum("bij,bnj->bni", P, Xh)
    z = pix[..., 2:3].clamp(min=1e-3)
    col = pix[..., 0:1] / z
    row = pix[..., 1:2] / z
    gx = 2 * col / (SENSOR - 1) - 1
    gy = 2 * row / (SENSOR - 1) - 1
    grid = torch.cat([gx, gy], -1).view(B, res, res * res, 2)
    samp = F.grid_sample(feat, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)
    return samp.view(B, C, res, res, res)


class FiLMResBlock3D(nn.Module):
    """3D conv block, timestep-only FiLM (X-ray info enters via input channels)."""

    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.film = nn.Linear(cond, cout * 2)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        g, b = self.film(c)[:, :, None, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + g) + b
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class BackProjVolumeDenoiser(nn.Module):
    """Predicts x0. Input = noisy volume + back-projected features (both views).
    4 levels -> res must be divisible by 16. use_checkpoint trades compute for
    memory (needed at 128^3)."""

    def __init__(self, base=32, feat_c=16, time_dim=128, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.time = nn.Sequential(SinusoidalTime(
            time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.xray = XrayFeat(feat_c)
        cin = 1 + 2 * feat_c
        self.in_conv = nn.Conv3d(cin, base, 3, padding=1)
        cond = time_dim
        self.d1 = FiLMResBlock3D(base,   base,   cond)
        self.p1 = nn.Conv3d(base,   base,   4, 2, 1)
        self.d2 = FiLMResBlock3D(base,   base*2, cond)
        self.p2 = nn.Conv3d(base*2, base*2, 4, 2, 1)
        self.d3 = FiLMResBlock3D(base*2, base*4, cond)
        self.p3 = nn.Conv3d(base*4, base*4, 4, 2, 1)
        self.d4 = FiLMResBlock3D(base*4, base*8, cond)
        self.p4 = nn.Conv3d(base*8, base*8, 4, 2, 1)
        self.mid = FiLMResBlock3D(base*8, base*8, cond)
        self.u4 = nn.ConvTranspose3d(base*8, base*8, 4, 2, 1)
        self.r4 = FiLMResBlock3D(base*16, base*4, cond)
        self.u3 = nn.ConvTranspose3d(base*4, base*4, 4, 2, 1)
        self.r3 = FiLMResBlock3D(base*8,  base*2, cond)
        self.u2 = nn.ConvTranspose3d(base*2, base*2, 4, 2, 1)
        self.r2 = FiLMResBlock3D(base*4,  base,   cond)
        self.u1 = nn.ConvTranspose3d(base,   base,   4, 2, 1)
        self.r1 = FiLMResBlock3D(base*2,  base,   cond)
        self.out = nn.Conv3d(base, 1, 1)

    def _blk(self, blk, x, c):
        if self.use_checkpoint and self.training:
            return checkpoint(blk, x, c, use_reentrant=False)
        return blk(x, c)

    def forward(self, vol, t, images, poses, svoxel):
        B, _, res, _, _ = vol.shape
        c = self.time(t)
        f0 = self.xray(images[:, 0:1])
        f1 = self.xray(images[:, 1:2])
        b0 = backproject(f0, poses[:, 0], svoxel, res)
        b1 = backproject(f1, poses[:, 1], svoxel, res)
        x = self.in_conv(torch.cat([vol, b0, b1], 1))
        d1 = self._blk(self.d1, x, c)
        x = self.p1(d1)
        d2 = self._blk(self.d2, x, c)
        x = self.p2(d2)
        d3 = self._blk(self.d3, x, c)
        x = self.p3(d3)
        d4 = self._blk(self.d4, x, c)
        x = self.p4(d4)
        x = self._blk(self.mid, x, c)
        x = self.u4(x)
        x = self._blk(self.r4, torch.cat([x, d4], 1), c)
        x = self.u3(x)
        x = self._blk(self.r3, torch.cat([x, d3], 1), c)
        x = self.u2(x)
        x = self._blk(self.r2, torch.cat([x, d2], 1), c)
        x = self.u1(x)
        x = self._blk(self.r1, torch.cat([x, d1], 1), c)
        return self.out(x)


if __name__ == "__main__":
    for R in (32, 64):
        m = BackProjVolumeDenoiser(base=32, use_checkpoint=False)
        vol = torch.randn(1, 1, R, R, R)
        t = torch.randint(0, 1000, (1,))
        im = torch.randn(1, 2, 128, 128)
        P = torch.randn(1, 2, 3, 4)
        sv = torch.rand(1, 3)*160+40
        out = m(vol, t, im, P, sv)
        n = sum(p.numel() for p in m.parameters())
        print(f"R={R}: OK {tuple(out.shape)}, params {n:,}")
