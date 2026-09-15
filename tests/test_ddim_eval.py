import pytest

from ddim_eval import checkpoint_prediction_type, edge_cache_file, radius_metrics


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
