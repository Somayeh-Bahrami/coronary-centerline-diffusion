import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.models.diffusion import CenterlineDenoiser


def small_inputs():
    torch.manual_seed(31)
    nodes = torch.randn(2, 11, 4)
    timesteps = torch.tensor([100, 700])
    images = torch.randn(2, 2, 32, 32)
    poses = torch.randn(2, 2, 3, 4)
    mask = torch.tensor([
        [True] * 7 + [False] * 4,
        [True] * 9 + [False] * 2,
    ])
    return nodes, timesteps, images, poses, mask


def test_arbitrary_length_output_and_padding_isolation():
    model = CenterlineDenoiser(hidden_dim=32, time_dim=16).eval()
    nodes, timesteps, images, poses, mask = small_inputs()
    perturbed = nodes.clone()
    perturbed[~mask] = torch.randn_like(perturbed[~mask]) * 100
    with torch.no_grad():
        first = model(nodes, timesteps, images, poses, node_mask=mask)
        second = model(perturbed, timesteps, images, poses, node_mask=mask)
    assert first.shape == nodes.shape
    assert torch.equal(first[~mask], torch.zeros_like(first[~mask]))
    assert torch.allclose(first[mask], second[mask], atol=1e-5, rtol=1e-5)


def test_images_and_nominal_poses_are_connected_to_valid_outputs():
    model = CenterlineDenoiser(hidden_dim=32, time_dim=16).eval()
    nodes, timesteps, images, poses, mask = small_inputs()
    images.requires_grad_()
    poses.requires_grad_()
    output = model(nodes, timesteps, images, poses, node_mask=mask)
    output[mask].square().mean().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert poses.grad is not None and poses.grad.abs().sum() > 0
