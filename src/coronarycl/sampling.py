"""Canonical, padding-aware DDIM sampling for centerline generation."""

from __future__ import annotations

import torch


def ddim_timesteps(training_steps, sampling_steps, device="cpu"):
    """Exactly ``sampling_steps`` unique indices, including T-1 and zero."""
    training_steps = int(training_steps)
    sampling_steps = int(sampling_steps)
    if not 2 <= sampling_steps <= training_steps:
        raise ValueError(
            f"sampling_steps must be in [2, {training_steps}]")
    timesteps = torch.linspace(
        training_steps - 1, 0, sampling_steps, device=device
    ).round().long()
    if len(torch.unique(timesteps)) != sampling_steps:
        raise RuntimeError("DDIM timestep construction produced duplicates")
    return timesteps


def _as_bound(value, reference):
    return torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype)


@torch.no_grad()
def sample_ddim(
    model,
    scheduler,
    images,
    poses,
    node_mask,
    device,
    *,
    seed=None,
    initial_noise=None,
    n_steps=50,
    guidance_scale=1.0,
    x0_min=None,
    x0_max=None,
):
    """Deterministic eta=0 DDIM with CFG and componentwise x0 bounds.

    ``node_mask`` is mandatory and reaches every model call. Padded rows stay
    zero throughout the reverse process. ``x0_min`` and ``x0_max`` may be any
    tensors broadcastable to ``(B,N,4)``. Production evaluation supplies
    per-sample crop bounds for xyz and TRAIN-only radius bounds. Exactly one of
    ``seed`` or ``initial_noise`` may be supplied; the evaluator uses explicit
    per-sample noise so results do not depend on batch composition.
    """
    if node_mask is None:
        raise ValueError("node_mask is required; implicit padding is unsafe")
    if guidance_scale <= 0:
        raise ValueError("guidance_scale must be positive")
    if seed is not None and initial_noise is not None:
        raise ValueError("provide seed or initial_noise, not both")
    timesteps = ddim_timesteps(
        scheduler.n_steps, n_steps, device=device)

    was_training = model.training
    model.eval()
    try:
        images = images.to(device)
        poses = poses.to(device)
        mask = node_mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[0] != images.shape[0]:
            raise ValueError("node_mask must have shape (B,N)")
        batch_size, n_points = mask.shape
        mask_float = mask.unsqueeze(-1).to(dtype=images.dtype)

        if initial_noise is None:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=torch.device(device))
                generator.manual_seed(int(seed))
            x_t = torch.randn(
                batch_size, n_points, 4,
                device=device, dtype=images.dtype, generator=generator)
        else:
            x_t = torch.as_tensor(
                initial_noise, device=device, dtype=images.dtype)
            if x_t.shape != (batch_size, n_points, 4):
                raise ValueError(
                    "initial_noise must have shape "
                    f"{(batch_size, n_points, 4)}, got {tuple(x_t.shape)}")
        x_t = x_t * mask_float

        lower = upper = None
        if (x0_min is None) != (x0_max is None):
            raise ValueError("x0_min and x0_max must be supplied together")
        if x0_min is not None:
            lower = _as_bound(x0_min, x_t)
            upper = _as_bound(x0_max, x_t)
            try:
                torch.broadcast_shapes(
                    x_t.shape, lower.shape, upper.shape)
            except RuntimeError as error:
                raise ValueError(
                    "x0 bounds are not broadcastable to (B,N,4)") from error
            if torch.any(lower >= upper):
                raise ValueError("every x0_min must be smaller than x0_max")

        null_images = null_poses = None
        if guidance_scale != 1.0:
            null_images = torch.zeros_like(images)
            null_poses = torch.zeros_like(poses)

        x0_self = None
        x0_hat = None
        for index, timestep in enumerate(timesteps):
            timestep_batch = torch.full(
                (batch_size,), int(timestep),
                device=device, dtype=torch.long)
            epsilon_conditional = model(
                x_t, timestep_batch, images, poses,
                x0_self=x0_self, node_mask=mask)
            if guidance_scale == 1.0:
                epsilon = epsilon_conditional
            else:
                epsilon_unconditional = model(
                    x_t, timestep_batch, null_images, null_poses,
                    x0_self=x0_self, node_mask=mask)
                epsilon = epsilon_unconditional + guidance_scale * (
                    epsilon_conditional - epsilon_unconditional)
            epsilon = epsilon * mask_float

            alpha_t = scheduler.alpha_bars[timestep]
            x0_hat = (
                x_t - torch.sqrt(1.0 - alpha_t) * epsilon
            ) / torch.sqrt(alpha_t)
            if lower is not None:
                x0_hat = torch.maximum(
                    torch.minimum(x0_hat, upper), lower)
            x0_hat = x0_hat * mask_float
            x0_self = x0_hat[..., :3].detach()

            alpha_previous = (
                scheduler.alpha_bars[timesteps[index + 1]]
                if index + 1 < len(timesteps)
                else torch.ones((), device=device, dtype=x_t.dtype)
            )
            x_t = (
                torch.sqrt(alpha_previous) * x0_hat
                + torch.sqrt(1.0 - alpha_previous) * epsilon
            ) * mask_float

        return x0_hat
    finally:
        model.train(was_training)
