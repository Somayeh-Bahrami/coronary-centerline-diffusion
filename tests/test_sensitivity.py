import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cond_sensitivity import (
    build_donor_map,
    masked_mse_per_sample,
    sample_parts,
)


def test_donors_match_vessel_but_not_patient():
    samples = [
        "1_LCA", "1_RCA", "2_LCA", "2_RCA",
        "3_LCA", "3_RCA", "4_LCA", "4_RCA",
    ]
    donors = build_donor_map(samples, seed=104729)
    for target_index, donor_index in enumerate(donors):
        target_patient, target_vessel = sample_parts(samples[target_index])
        donor_patient, donor_vessel = sample_parts(samples[donor_index])
        assert target_patient != donor_patient
        assert target_vessel == donor_vessel


def test_masked_mse_is_reduced_per_sample_not_per_batch():
    prediction = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0], [100.0, 100.0]],
        [[2.0, 2.0], [100.0, 100.0], [100.0, 100.0]],
    ])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = masked_mse_per_sample(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor([1.0, 4.0]))
