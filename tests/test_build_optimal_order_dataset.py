import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_optimal_order_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_optimal_order_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_arrays(n=6, padded=8):
    centerline = np.zeros((padded, 5), dtype=np.float32)
    centerline[:n] = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
    mask = np.zeros(padded, dtype=bool)
    mask[:n] = True
    return {
        "builder_version": np.asarray("3.4"),
        "centerline": centerline,
        "centerline_mask": mask,
        "n_points": np.asarray(n),
        "split": np.asarray("train"),
        "images": np.zeros((2, 4, 4), dtype=np.uint8),
        "poses": np.zeros((2, 3, 4), dtype=np.float32),
    }


def test_transform_preserves_nodes_fields_padding_and_graph():
    source = sample_arrays()
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], dtype=np.int32)
    result = MODULE.optimal_tree_ordering(6, edges)
    output, remapped, row = MODULE.transform_sample(
        source, result.permutation, edges, "fixture"
    )
    inverse = np.empty(6, dtype=int)
    inverse[result.permutation] = np.arange(6)
    assert np.array_equal(output["centerline"][:6][inverse], source["centerline"][:6])
    assert np.count_nonzero(output["centerline"][6:]) == 0
    assert np.array_equal(output["images"], source["images"])
    assert np.array_equal(output["poses"], source["poses"])
    assert row["optimal_consecutive_edges"] == result.max_consecutive_edges
    restored = {
        tuple(sorted((int(result.permutation[a]), int(result.permutation[b]))))
        for a, b in remapped
    }
    assert restored == {tuple(edge) for edge in edges.tolist()}


def test_transform_rejects_incomplete_permutation():
    source = sample_arrays(n=3, padded=4)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    try:
        MODULE.transform_sample(source, np.array([0, 1, 1]), edges, "bad")
    except AssertionError as error:
        assert "complete permutation" in str(error)
    else:
        raise AssertionError("invalid permutation was accepted")


def test_transform_rejects_nonzero_padding():
    source = sample_arrays(n=3, padded=4)
    source["centerline"][3, 0] = 1
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    try:
        MODULE.transform_sample(source, np.arange(3), edges, "bad")
    except AssertionError as error:
        assert "padding" in str(error)
    else:
        raise AssertionError("nonzero padding was accepted")


def test_required_metadata_list_is_complete():
    assert MODULE.REQUIRED_METADATA == (
        "case_splits_v3.json",
        "norm_stats_v3.json",
        "pilot_report_v3.json",
    )
