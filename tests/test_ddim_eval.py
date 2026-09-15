import pytest

from ddim_eval import checkpoint_prediction_type, edge_cache_file


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
