import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.trainer import (
    DeterministicStepBatchSampler,
    NoiseScheduler,
    _make_lr_scheduler,
    _validate_resume_signature,
    compute_loss,
)
from src.coronarycl.prediction import training_target


class _ZeroModel(torch.nn.Module):
    def forward(
        self, x_t, timesteps, images, poses, x0_self=None, node_mask=None
    ):
        return torch.zeros_like(x_t)


class _SelfConditionSpy(torch.nn.Module):
    def __init__(self, scheduler, prediction_type):
        super().__init__()
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.calls = 0
        self.expected_x0_self = None

    def forward(
        self, x_t, timesteps, images, poses, x0_self=None, node_mask=None
    ):
        self.calls += 1
        if self.calls == 1:
            alpha_bar = self.scheduler.alpha_bars[timesteps].view(-1, 1, 1)
            if self.prediction_type == "epsilon":
                expected_x0 = x_t / torch.sqrt(alpha_bar)
            else:
                expected_x0 = torch.sqrt(alpha_bar) * x_t
            self.expected_x0_self = expected_x0[..., :3]
            assert x0_self is None
        else:
            torch.testing.assert_close(x0_self, self.expected_x0_self)
        return torch.zeros_like(x_t)


def _batch():
    generator = torch.Generator().manual_seed(808)
    centerline = torch.randn(2, 7, 5, generator=generator)
    mask = torch.tensor([
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True],
    ])
    centerline[~mask] = 0.0
    return {
        "centerline": centerline,
        "centerline_mask": mask,
        "images": torch.zeros(2, 2, 8, 8),
        "poses": torch.zeros(2, 2, 3, 4),
    }


def test_step_batch_sampler_resume_is_exact_suffix():
    full = list(DeterministicStepBatchSampler(
        n_samples=11, batch_size=4, first_step=0, max_steps=9, seed=71))
    resumed = list(DeterministicStepBatchSampler(
        n_samples=11, batch_size=4, first_step=4, max_steps=9, seed=71))
    assert resumed == full[4:]
    assert [len(batch) for batch in full[:3]] == [4, 4, 3]


def test_warmup_cosine_reaches_peak_then_eta_min():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    scheduler = _make_lr_scheduler(
        optimizer, max_steps=200, warmup_steps=20, eta_min=3e-6)
    initial = optimizer.param_groups[0]["lr"]
    for _ in range(20):
        optimizer.step()
        scheduler.step()
    peak = optimizer.param_groups[0]["lr"]
    for _ in range(180):
        optimizer.step()
        scheduler.step()
    final = optimizer.param_groups[0]["lr"]
    assert abs(initial - 1.5e-5) < 1e-15
    assert abs(peak - 3e-4) < 1e-12
    assert abs(final - 3e-6) < 1e-12


@pytest.mark.parametrize("prediction_type", ["epsilon", "v"])
def test_compute_loss_uses_selected_prediction_target(prediction_type):
    batch = _batch()
    scheduler = NoiseScheduler(n_steps=20)
    x0 = batch["centerline"][..., :4]
    timestep = 7

    torch.manual_seed(991)
    noise = torch.randn_like(x0)
    alpha_bar = scheduler.alpha_bars[
        torch.full((x0.shape[0],), timestep, dtype=torch.long)
    ]
    target = training_target(x0, noise, alpha_bar, prediction_type)
    per_point = target.square().mean(dim=-1)
    expected = (
        per_point * batch["centerline_mask"].float()
    ).sum() / batch["centerline_mask"].sum()

    torch.manual_seed(991)
    actual = compute_loss(
        _ZeroModel(), scheduler, batch, "cpu",
        fixed_t=timestep, prediction_type=prediction_type,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v"])
def test_self_conditioning_uses_x0_for_selected_target(prediction_type):
    scheduler = NoiseScheduler(n_steps=20)
    model = _SelfConditionSpy(scheduler, prediction_type)

    compute_loss(
        model, scheduler, _batch(), "cpu",
        self_cond_p=1.0, prediction_type=prediction_type,
    )

    assert model.calls == 2


def test_resume_signature_defaults_old_checkpoint_to_epsilon():
    checkpoint = {"run_signature": {"hidden_dim": 384}}

    _validate_resume_signature(
        checkpoint, {"hidden_dim": 384, "prediction_type": "epsilon"}
    )

    with pytest.raises(RuntimeError, match="prediction_type"):
        _validate_resume_signature(
            checkpoint, {"hidden_dim": 384, "prediction_type": "v"}
        )
