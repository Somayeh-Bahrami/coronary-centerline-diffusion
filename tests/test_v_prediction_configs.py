from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_paired_50k_configs_change_only_prediction_and_output_directory():
    epsilon = _load("h384_epsilon_50k.yaml")
    velocity = _load("h384_v_50k.yaml")

    assert epsilon["train"]["prediction_type"] == "epsilon"
    assert velocity["train"]["prediction_type"] == "v"
    assert epsilon["train"]["checkpoint_dir"] != velocity["train"][
        "checkpoint_dir"
    ]

    normalized_epsilon = deepcopy(epsilon)
    normalized_velocity = deepcopy(velocity)
    for config in (normalized_epsilon, normalized_velocity):
        config["train"].pop("prediction_type")
        config["train"].pop("checkpoint_dir")

    assert normalized_epsilon == normalized_velocity


def test_paired_50k_protocol_is_locked_to_val():
    for name in ("h384_epsilon_50k.yaml", "h384_v_50k.yaml"):
        config = _load(name)
        assert config["train"]["max_steps"] == 50_000
        assert config["train"]["coh_weight"] == 0.0
        assert config["train"]["init_checkpoint"] is None
        assert config["eval"] == {
            "split": "val",
            "bounds": "physical",
            "ddim_steps": 100,
            "guidance": 2.0,
            "sampling_seeds": [
                104729, 130363, 155921, 181081, 205019
            ],
            "precision": "fp32",
        }
