import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coronarycl.trainer import (
    DeterministicStepBatchSampler,
    _make_lr_scheduler,
)


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
