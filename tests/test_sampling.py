import sys
from pathlib import Path

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
