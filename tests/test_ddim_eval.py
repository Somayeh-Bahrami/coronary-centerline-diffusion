import pytest

from ddim_eval import (
    checkpoint_node_dim,
    checkpoint_prediction_type,
    decoded_topology_edges,
    edge_cache_file,
    initial_noise,
    radius_metrics,
)
from src.coronarycl.branch_token_tree import encode_branch_tokens


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


def test_edge_cache_file_accepts_directory_or_npz(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "edges_v1.npz"
    archive.write_bytes(b"fixture")
    assert edge_cache_file(cache) == archive.resolve()
    assert edge_cache_file(archive) == archive.resolve()


def test_edge_cache_file_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="canonical edge cache"):
        edge_cache_file(tmp_path / "missing")


def test_radius_metrics_report_physical_errors_and_correlation():
    result = radius_metrics([1.5, 2.5, 3.5], [1.0, 2.0, 3.0])
    assert result["radius_mae_mm"] == pytest.approx(0.5)
    assert result["radius_rmse_mm"] == pytest.approx(0.5)
    assert result["radius_bias_mm"] == pytest.approx(0.5)
    assert result["radius_correlation"] == pytest.approx(1.0)


def test_radius_correlation_is_nan_for_constant_values():
    result = radius_metrics([2.0, 2.0], [1.0, 2.0])
    assert result["radius_correlation"] != result["radius_correlation"]


def test_checkpoint_node_dim_uses_frozen_run_signature():
    assert checkpoint_node_dim({"run_signature": {"node_dim": 8}}) == 8
    assert checkpoint_node_dim({"run_signature": {}}) == 4


def test_decoded_topology_edges_never_accepts_oracle_edges():
    capacity, n = 8, 6
    centerline = __import__("numpy").zeros((capacity, 5), dtype="float32")
    mask = __import__("numpy").array([True] * n + [False] * 2)
    expected = __import__("numpy").array(
        [[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], dtype="int32"
    )
    tokens = encode_branch_tokens(centerline, mask, expected)

    actual = decoded_topology_edges(tokens[:n], mask[:n], token_capacity=capacity)

    assert {tuple(sorted(map(int, edge))) for edge in actual} == {
        tuple(sorted(map(int, edge))) for edge in expected
    }


def test_initial_noise_preserves_seed_and_supports_eight_channels():
    items = [{"sample": "fixture", "n_points": 3}]
    first = initial_noise(items, 5, 17, "cpu", node_dim=8)
    second = initial_noise(items, 5, 17, "cpu", node_dim=8)

    assert first.shape == (1, 5, 8)
    assert __import__("torch").equal(first, second)
    assert __import__("torch").count_nonzero(first[:, 3:]) == 0
