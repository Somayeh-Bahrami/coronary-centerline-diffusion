"""Conversions between epsilon-, x0-, and v-prediction."""

from __future__ import annotations

import torch


VALID_PREDICTION_TYPES = {"epsilon", "v"}


def validate_prediction_type(prediction_type: str) -> str:
    prediction_type = str(prediction_type).lower()
    if prediction_type not in VALID_PREDICTION_TYPES:
        raise ValueError(
            f"prediction_type must be one of "
            f"{sorted(VALID_PREDICTION_TYPES)}, got {prediction_type!r}"
        )
    return prediction_type


def _expand_alpha_bar(alpha_bar: torch.Tensor, reference: torch.Tensor):
    alpha_bar = torch.as_tensor(
        alpha_bar,
        device=reference.device,
        dtype=reference.dtype,
    )
    while alpha_bar.ndim < reference.ndim:
        alpha_bar = alpha_bar.unsqueeze(-1)
    return alpha_bar


def training_target(
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Return the requested diffusion training target."""
    prediction_type = validate_prediction_type(prediction_type)

    if prediction_type == "epsilon":
        return epsilon

    alpha_bar = _expand_alpha_bar(alpha_bar, x0)
    sqrt_alpha = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar)

    return sqrt_alpha * epsilon - sqrt_one_minus_alpha * x0


def model_output_to_x0_epsilon(
    x_t: torch.Tensor,
    model_output: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a model output into both x0 and epsilon."""
    prediction_type = validate_prediction_type(prediction_type)
    alpha_bar = _expand_alpha_bar(alpha_bar, x_t)

    sqrt_alpha = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar)

    if prediction_type == "epsilon":
        epsilon = model_output
        x0 = (
            x_t - sqrt_one_minus_alpha * epsilon
        ) / sqrt_alpha
    else:
        velocity = model_output
        x0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * velocity
        epsilon = (
            sqrt_one_minus_alpha * x_t
            + sqrt_alpha * velocity
        )

    return x0, epsilon
