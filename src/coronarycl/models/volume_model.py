"""Stage 1 (Option B) v2: X-ray -> 3D vessel volume with GEOMETRY-AWARE
back-projection conditioning (X2CT-GAN / DX2CT style).

Why this replaces the global-FiLM model: a single global X-ray vector cannot
carry depth, so the model collapsed to an "average" vessel volume (cross-case
Dice ratio 1.06, output sensitivity 0.003 -- ignores the X-rays). Here every
3D voxel is projected into each X-ray via the calibrated projection matrix P
and samples that view's 2D feature map, so a voxel that lands on a vessel in
BOTH views receives vessel evidence in both -- the model can finally locate
structure in depth.

Geometry (from drr.py, validated numerically):
  P maps center-origin mm -> pixel(col,row);  mm = svoxel*((q+0.5)/res - 0.5)
  where q is the voxel index and svoxel (mm extent) is stored per case.
  Views are 30 deg apart (not orthogonal) -> true projective grid_sample,
  NOT X2CT broadcast tiling.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """One X-ray (B,1,H,W) -> full-frame feature map (B,C,H,W). Full-frame so
    normalized [-1,1] grid coords line up with the projection's field of view."""

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
    gi, gj, gk = torch.meshgrid(
        q, q, q, indexing="ij")          # (res,res,res)
    coord = torch.stack([gi, gj, gk], -1).reshape(-1,
                                                  # (res^3, 3) order i,j,k
                                                  3)
    # centered [-.5,.5]
    frac = (coord + 0.5) / res - 0.5
    mm = frac[None] * svoxel[:, None, :]                         # (B,res^3,3)
    # (B,res^3,4)
    Xh = torch.cat([mm, torch.ones(B, mm.shape[1], 1, device=dev)], -1)
    pix = torch.einsum("bij,bnj->bni", P, Xh)                   # (B,res^3,3)
    z = pix[..., 2:3].clamp(min=1e-3)
    col = pix[..., 0:1] / z
    row = pix[..., 1:2] / z
    # grid x <- col (width)
    gx = 2 * col / (SENSOR - 1) - 1
    # grid y <- row (height)
    gy = 2 * row / (SENSOR - 1) - 1
    grid = torch.cat([gx, gy], -1).view(B, res, res * res, 2)
    samp = F.grid_sample(feat, grid, mode="bilinear",
                         # (B,C,res,res^2)
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
    """Predicts x0 (clean volume). Input channels = noisy volume + back-projected
    features from both views. res must be divisible by 8."""

    def __init__(self, base=24, feat_c=16, time_dim=128):
        super().__init__()
        self.time = nn.Sequential(SinusoidalTime(
            time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.xray = XrayFeat(feat_c)
        cin = 1 + 2 * feat_c
        self.in_conv = nn.Conv3d(cin, base, 3, padding=1)
        cond = time_dim
        self.d1 = FiLMResBlock3D(base, base, cond)
        self.p1 = nn.Conv3d(base, base, 4, 2, 1)
        self.d2 = FiLMResBlock3D(base, base * 2, cond)
        self.p2 = nn.Conv3d(base * 2, base * 2, 4, 2, 1)
        self.d3 = FiLMResBlock3D(base * 2, base * 4, cond)
        self.p3 = nn.Conv3d(base * 4, base * 4, 4, 2, 1)
        self.mid = FiLMResBlock3D(base * 4, base * 4, cond)
        self.u3 = nn.ConvTranspose3d(base * 4, base * 4, 4, 2, 1)
        self.r3 = FiLMResBlock3D(base * 8, base * 2, cond)
        self.u2 = nn.ConvTranspose3d(base * 2, base * 2, 4, 2, 1)
        self.r2 = FiLMResBlock3D(base * 4, base, cond)
        self.u1 = nn.ConvTranspose3d(base, base, 4, 2, 1)
        self.r1 = FiLMResBlock3D(base * 2, base, cond)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, vol, t, images, poses, svoxel):
        B, _, res, _, _ = vol.shape
        c = self.time(t)
        f0 = self.xray(images[:, 0:1])
        f1 = self.xray(images[:, 1:2])
        b0 = backproject(f0, poses[:, 0], svoxel, res)
        b1 = backproject(f1, poses[:, 1], svoxel, res)
        x = self.in_conv(torch.cat([vol, b0, b1], 1))
        d1 = self.d1(x, c)
        x = self.p1(d1)
        d2 = self.d2(x, c)
        x = self.p2(d2)
        d3 = self.d3(x, c)
        x = self.p3(d3)
        x = self.mid(x, c)
        x = self.u3(x)
        x = self.r3(torch.cat([x, d3], 1), c)
        x = self.u2(x)
        x = self.r2(torch.cat([x, d2], 1), c)
        x = self.u1(x)
        x = self.r1(torch.cat([x, d1], 1), c)
        return self.out(x)


if __name__ == "__main__":
    for R in (64, 128):
        m = BackProjVolumeDenoiser(base=24)
        vol = torch.randn(2, 1, R, R, R)
        t = torch.randint(0, 1000, (2,))
        im = torch.randn(2, 2, 128, 128)
        P = torch.randn(2, 2, 3, 4)
        sv = torch.rand(2, 3) * 160 + 40
        out = m(vol, t, im, P, sv)
        n = sum(p.numel() for p in m.parameters())
        print(f"R={R}: forward OK {tuple(out.shape)}, params {n:,}")
