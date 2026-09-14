import pytest
import torch

from src.coronarycl.prediction import (
    model_output_to_x0_epsilon,
    training_target,
    validate_prediction_type,
)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v"])
def test_prediction_round_trip(prediction_type):
    generator = torch.Generator().manual_seed(123)
    x0 = torch.randn(4, 37, 4, generator=generator)
    epsilon = torch.randn(4, 37, 4, generator=generator)
    alpha_bar = torch.tensor([0.999, 0.8, 0.2, 0.001])

    expanded = alpha_bar[:, None, None]
    x_t = (
        torch.sqrt(expanded) * x0
        + torch.sqrt(1.0 - expanded) * epsilon
    )

    target = training_target(
        x0, epsilon, alpha_bar, prediction_type
    )
    recovered_x0, recovered_epsilon = model_output_to_x0_epsilon(
        x_t, target, alpha_bar, prediction_type
    )

    assert torch.allclose(recovered_x0, x0, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        recovered_epsilon, epsilon, atol=2e-5, rtol=2e-5
    )


def test_epsilon_target_is_unchanged():
    x0 = torch.randn(2, 8, 4)
    epsilon = torch.randn_like(x0)
    alpha_bar = torch.tensor([0.9, 0.1])

    target = training_target(x0, epsilon, alpha_bar, "epsilon")

    assert target is epsilon


def test_invalid_prediction_type_is_rejected():
    with pytest.raises(ValueError):
        validate_prediction_type("noise")
