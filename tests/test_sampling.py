import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.sampling import ddim_timesteps, sample_ddim
from src.coronarycl.trainer import NoiseScheduler


class ZeroDenoiser(torch.nn.Module):
    def forward(self, noisy, timestep, images, poses, x0_self=None,
                node_mask=None):
        del timestep, images, poses, x0_self
        output = torch.zeros_like(noisy)
        return output * node_mask.unsqueeze(-1)


class RecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noisy, timestep, images, poses, x0_self=None,
                node_mask=None):
        self.calls.append({
            "images": images.detach().clone(),
            "poses": poses.detach().clone(),
            "self_conditioned": x0_self is not None,
            "mask": node_mask.detach().clone(),
        })
        return torch.zeros_like(noisy) * node_mask.unsqueeze(-1)


def inputs():
    images = torch.zeros(2, 2, 16, 16)
    poses = torch.zeros(2, 2, 3, 4)
    mask = torch.tensor([
        [True, True, True, False, False, False],
        [True, True, True, True, True, False],
    ])
    return images, poses, mask


def test_ddim_timesteps_are_exact_and_include_endpoints():
    steps = ddim_timesteps(1000, 50)
    assert len(steps) == len(torch.unique(steps)) == 50
    assert int(steps[0]) == 999
    assert int(steps[-1]) == 0


def test_sampling_is_repeatable_and_preserves_padding_and_mode():
    model = ZeroDenoiser().train()
    scheduler = NoiseScheduler(n_steps=20)
    images, poses, mask = inputs()
    first = sample_ddim(
        model, scheduler, images, poses, mask, "cpu",
        seed=17, n_steps=5)
    second = sample_ddim(
        model, scheduler, images, poses, mask, "cpu",
        seed=17, n_steps=5)
    other = sample_ddim(
        model, scheduler, images, poses, mask, "cpu",
        seed=18, n_steps=5)
    assert torch.equal(first, second)
    assert not torch.equal(first[mask], other[mask])
    assert torch.equal(first[~mask], torch.zeros_like(first[~mask]))
    assert model.training


def test_componentwise_bounds_are_enforced():
    model = ZeroDenoiser().eval()
    scheduler = NoiseScheduler(n_steps=20)
    images, poses, mask = inputs()
    lower = torch.tensor([-0.5, -1.0, -1.5, -2.0]).view(1, 1, 4)
    upper = torch.tensor([0.5, 1.0, 1.5, 2.0]).view(1, 1, 4)
    sampled = sample_ddim(
        model, scheduler, images, poses, mask, "cpu",
        seed=19, n_steps=5, x0_min=lower, x0_max=upper)
    for channel in range(4):
        valid = sampled[..., channel][mask]
        assert torch.all(valid >= lower[0, 0, channel])
        assert torch.all(valid <= upper[0, 0, channel])


def test_cfg_uses_joint_null_conditioning_and_sampler_self_conditions():
    model = RecordingDenoiser()
    scheduler = NoiseScheduler(n_steps=20)
    images, poses, mask = inputs()
    sample_ddim(
        model, scheduler, images + 1.0, poses + 1.0, mask, "cpu",
        seed=23, n_steps=5, guidance_scale=2.0)
    assert len(model.calls) == 10
    for index in range(0, len(model.calls), 2):
        conditional, unconditional = model.calls[index:index + 2]
        assert torch.count_nonzero(conditional["images"]) > 0
        assert torch.count_nonzero(conditional["poses"]) > 0
        assert torch.count_nonzero(unconditional["images"]) == 0
        assert torch.count_nonzero(unconditional["poses"]) == 0
        assert torch.equal(conditional["mask"], mask)
        assert torch.equal(unconditional["mask"], mask)
        expected_self_conditioning = index > 0
        assert conditional["self_conditioned"] == expected_self_conditioning
        assert unconditional["self_conditioned"] == expected_self_conditioning


def test_sampling_rejects_missing_mask_and_conflicting_noise_sources():
    model = ZeroDenoiser()
    scheduler = NoiseScheduler(n_steps=20)
    images, poses, mask = inputs()
    with pytest.raises(ValueError, match="node_mask is required"):
        sample_ddim(model, scheduler, images, poses, None, "cpu", n_steps=5)
    with pytest.raises(ValueError, match="seed or initial_noise"):
        sample_ddim(
            model, scheduler, images, poses, mask, "cpu", seed=1,
            initial_noise=torch.zeros(2, 6, 4), n_steps=5)
    empty = mask.clone()
    empty[0] = False
    with pytest.raises(ValueError, match="at least one valid node"):
        sample_ddim(model, scheduler, images, poses, empty, "cpu", n_steps=5)
    nonfinite = torch.zeros(2, 6, 4)
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sample_ddim(
            model, scheduler, images, poses, mask, "cpu",
            initial_noise=nonfinite, n_steps=5)


def test_zero_epsilon_ddim_matches_closed_form_solution():
    model = ZeroDenoiser()
    scheduler = NoiseScheduler(n_steps=20)
    images, poses, mask = inputs()
    initial = torch.randn(2, 6, 4) * mask.unsqueeze(-1)
    sampled = sample_ddim(
        model, scheduler, images, poses, mask, "cpu",
        initial_noise=initial, n_steps=5)
    expected = initial / scheduler.alpha_bars[-1].sqrt()
    assert torch.allclose(sampled, expected, atol=1e-6, rtol=1e-6)
