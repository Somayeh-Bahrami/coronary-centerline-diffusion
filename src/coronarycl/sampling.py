"""Step 3.1 -- reverse-diffusion sampling.

sample_ddpm: original DDPM ancestral sampler (Ho et al., 2020). RETAINED
FOR REFERENCE ONLY -- it diverges on this data: over 1000 stochastic steps
the reverse trajectory's per-point ||x_t|| inflates to ~3x the target and
max|x_t| blows up to ~40+, giving inflated, outlier-ridden centerlines
(overfit Chamfer ~135mm even at train_loss ~0.04). Diagnosed by norm-tracing
the reverse loop.

sample_ddim: deterministic DDIM (Song et al., 2021) with predicted-x0
clipping (Ho et al. 2020; Nichol & Dhariwal 2021) and optional
classifier-free guidance (Ho & Salimans, 2022). Dropping the per-step
stochastic term + clipping predicted-x0 to the normalized data range fixes
the divergence: overfit 135 -> ~27mm, full-val 37.9 -> 31.8mm on the same
weights, no retraining. Fewer steps reduce accumulation further.

clip=2.0, n_steps=50 were selected on the overfit-case diagnostic; for
reported numbers, confirm/sweep on held-out cases.
"""

import torch


@torch.no_grad()
def sample_ddpm(model, sched, images, poses, n_points, device, seed=None):
    """Original ancestral sampler -- diverges on this data. Prefer sample_ddim."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    B = images.shape[0]
    x_t = torch.randn(B, n_points, 4, device=device)
    for t_step in reversed(range(sched.n_steps)):
        t = torch.full((B,), t_step, device=device, dtype=torch.long)
        pred_noise = model(x_t, t, images, poses)
        a, ab, b = sched.alphas[t_step], sched.alpha_bars[t_step], sched.betas[t_step]
        mean = (1 / torch.sqrt(a)) * \
            (x_t - (b / torch.sqrt(1 - ab)) * pred_noise)
        x_t = mean + torch.sqrt(b) * \
            torch.randn_like(x_t) if t_step > 0 else mean
    model.train()
    return x_t


@torch.no_grad()
def sample_ddim(model, sched, images, poses, n_points, device,
                seed=None, clip=2.0, n_steps=50, guidance_scale=1.0):
    """Deterministic DDIM + predicted-x0 clipping + optional CFG.

    clip: clamp predicted-x0 to [-clip, clip] in normalized space each step.
        None disables (trajectory then amplifies bad estimates -> ~226mm).
    n_steps: DDIM steps, evenly strided over the scheduler's training steps.
        Fewer = less accumulation. None uses all.
    guidance_scale: 1.0 = plain conditional (one model call). >1.0 adds an
        unconditional pass (zeroed conditioning) and extrapolates
        pred = uncond + scale*(cond - uncond). Only meaningful if trained
        with conditioning dropout (train.cond_drop_prob > 0).
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    ab, T = sched.alpha_bars, sched.n_steps
    steps = list(reversed(range(T))) if n_steps is None \
        else list(reversed(range(0, T, max(1, T // n_steps))))
    B = images.shape[0]
    x_t = torch.randn(B, n_points, 4, device=device)
    x0_prev = None  # self-conditioning: previous step's predicted-x0 xyz
    for i, t_step in enumerate(steps):
        t = torch.full((B,), t_step, device=device, dtype=torch.long)
        pred_cond = model(x_t, t, images, poses, x0_self=x0_prev)
        if guidance_scale != 1.0:
            pred_uncond = model(x_t, t, torch.zeros_like(
                images), torch.zeros_like(poses), x0_self=x0_prev)
            pred_noise = pred_uncond + guidance_scale * \
                (pred_cond - pred_uncond)
        else:
            pred_noise = pred_cond
        abt = ab[t_step]
        x0_hat = (x_t - torch.sqrt(1 - abt) * pred_noise) / torch.sqrt(abt)
        if clip is not None:
            x0_hat = x0_hat.clamp(-clip, clip)
        x0_prev = x0_hat[:, :, :3]  # carry predicted-x0 to next step's query
        t_prev = steps[i + 1] if i + 1 < len(steps) else -1
        ab_prev = ab[t_prev] if t_prev >= 0 else torch.tensor(
            1.0, device=device)
        x_t = torch.sqrt(ab_prev) * x0_hat + \
            torch.sqrt(1 - ab_prev) * pred_noise
    model.train()
    return x_t
