"""Stage 1 : conditional 3D-UNet diffusion denoiser.
Input: noisy 3D vessel volume + timestep + 2 X-rays. Output: predicted noise.
The 2 X-rays are encoded to a global conditioning vector and injected via FiLM
at every 3D block (+ the timestep). Dense per-voxel supervision on the volume
is what avoids the conditioning collapse we hit with the sparse point model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTime(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim

    def forward(self, t):
        h = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(h,
                      device=t.device).float() / h)
        a = t.float()[:, None] * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class XrayEncoder(nn.Module):
    """2 X-rays (B,2,H,W) -> global conditioning vector (B, cond_dim)."""

    def __init__(self, cond_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),   # H/2
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),  # H/4
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(
                8, 128), nn.SiLU(),  # H/8
            nn.Conv2d(128, cond_dim, 4, 2, 1), nn.GroupNorm(
                8, cond_dim), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x): return self.net(x).flatten(
        1)                  # (B, cond_dim)


class FiLMResBlock3D(nn.Module):
    """3D conv block modulated by (timestep + X-ray) conditioning via FiLM."""

    def __init__(self, cin, cout, cond):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.film = nn.Linear(cond, cout * 2)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        g, b = self.film(c)[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + g) + b
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class ConditionalVolumeDenoiser(nn.Module):
    def __init__(self, base=24, cond_dim=128, time_dim=128):
        super().__init__()
        self.time = nn.Sequential(SinusoidalTime(
            time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.xray = XrayEncoder(cond_dim)
        cond = time_dim + cond_dim
        self.in_conv = nn.Conv3d(1, base, 3, padding=1)
        self.d1 = FiLMResBlock3D(base, base, cond)
        self.p1 = nn.Conv3d(base, base, 4, 2, 1)
        self.d2 = FiLMResBlock3D(base, base*2, cond)
        self.p2 = nn.Conv3d(base*2, base*2, 4, 2, 1)
        self.d3 = FiLMResBlock3D(base*2, base*4, cond)
        self.p3 = nn.Conv3d(base*4, base*4, 4, 2, 1)
        self.mid = FiLMResBlock3D(base*4, base*4, cond)
        self.u3 = nn.ConvTranspose3d(base*4, base*4, 4, 2, 1)
        self.r3 = FiLMResBlock3D(base*8, base*2, cond)
        self.u2 = nn.ConvTranspose3d(base*2, base*2, 4, 2, 1)
        self.r2 = FiLMResBlock3D(base*4, base, cond)
        self.u1 = nn.ConvTranspose3d(base, base, 4, 2, 1)
        self.r1 = FiLMResBlock3D(base*2, base, cond)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, vol, t, images):
        c = torch.cat([self.time(t), self.xray(images)], -1)
        x0 = self.in_conv(vol)
        d1 = self.d1(x0, c)
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
        m = ConditionalVolumeDenoiser(base=24)
        vol = torch.randn(1, 1, R, R, R)
        t = torch.randint(0, 1000, (1,))
        im = torch.randn(1, 2, 128, 128)
        try:
            out = m(vol, t, im)
            n = sum(p.numel() for p in m.parameters())
            print(f"R={R}: forward OK {tuple(out.shape)}, params {n:,}")
        except Exception as e:
            print(f"R={R}: FAILED -> {e}")
