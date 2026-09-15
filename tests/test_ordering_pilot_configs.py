from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DFS_PATH = ROOT / "configs/h384_ordering_50k_dfs.yaml"
OPTIMAL_PATH = ROOT / "configs/h384_ordering_50k_optimal.yaml"


def load(path):
    return yaml.safe_load(path.read_text())


def test_paired_configs_have_only_preregistered_path_differences():
    dfs, optimal = load(DFS_PATH), load(OPTIMAL_PATH)
    assert dfs["data"]["packaged_dir"].endswith("ds105_full")
    assert optimal["data"]["packaged_dir"].endswith("ds105_optimal_v1")
    assert dfs["eval"]["topology_edge_cache"].endswith("edge_cache")
    assert optimal["eval"]["topology_edge_cache"].endswith("ordering_edge_cache_v1")
    assert dfs["train"]["checkpoint_dir"] != optimal["train"]["checkpoint_dir"]

    left, right = deepcopy(dfs), deepcopy(optimal)
    for config in (left, right):
        config["data"]["packaged_dir"] = "<paired-dataset>"
        config["train"]["checkpoint_dir"] = "<paired-output>"
        config["eval"]["topology_edge_cache"] = "<paired-canonical-edges>"
    assert left == right


def test_pilot_is_frozen_epsilon_50k_from_random_initialization():
    for config in (load(DFS_PATH), load(OPTIMAL_PATH)):
        train = config["train"]
        assert train["prediction_type"] == "epsilon"
        assert train["hidden_dim"] == 384
        assert train["max_steps"] == 50_000
        assert train["warmup_steps"] == 500
        assert train["seed"] == 20260911
        assert train["init_checkpoint"] is None
        assert train["coh_weight"] == 0.0
        assert train["edge_cache"] is None
        assert config["eval"]["split"] == "val"
        assert config["eval"]["sampling_seeds"] == [
            104729, 130363, 155921, 181081, 205019
        ]
