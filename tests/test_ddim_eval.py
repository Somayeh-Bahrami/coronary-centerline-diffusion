import pytest

from ddim_eval import checkpoint_prediction_type


def test_legacy_checkpoint_defaults_to_epsilon():
    assert checkpoint_prediction_type({"run_signature": {}}) == "epsilon"


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"prediction_type": "v"},
        {"run_signature": {"prediction_type": "v"}},
        {
            "prediction_type": "v",
            "run_signature": {"prediction_type": "v"},
        },
    ],
)
def test_checkpoint_prediction_type_accepts_consistent_v(checkpoint):
    assert checkpoint_prediction_type(checkpoint) == "v"


def test_checkpoint_prediction_type_rejects_conflict():
    checkpoint = {
        "prediction_type": "v",
        "run_signature": {"prediction_type": "epsilon"},
    }
    with pytest.raises(RuntimeError, match="inconsistent"):
        checkpoint_prediction_type(checkpoint)


def test_checkpoint_prediction_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="prediction_type"):
        checkpoint_prediction_type({"prediction_type": "x0"})
