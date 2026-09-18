import importlib.util
from pathlib import Path

import numpy as np

from src.coronarycl.branch_token_tree import decode_branch_tokens


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_branch_token_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_branch_token_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _edges(edges):
    return {tuple(sorted(map(int, edge))) for edge in np.asarray(edges)}


def test_transform_preserves_non_centerline_fields_and_builds_decodable_tokens():
    n, capacity = 6, 8
    centerline = np.zeros((capacity, 5), dtype=np.float32)
    centerline[:n] = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
    mask = np.zeros(capacity, dtype=bool)
    mask[:n] = True
    source = {
        "builder_version": np.asarray("3.4"),
        "centerline": centerline,
        "centerline_mask": mask,
        "n_points": np.asarray(n),
        "split": np.asarray("train"),
        "images": np.zeros((2, 4, 4), dtype=np.uint8),
        "poses": np.zeros((2, 3, 4), dtype=np.float32),
    }
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], dtype=np.int32)

    output, row = MODULE.transform_sample(source, edges, "fixture")

    assert output["centerline"].shape == (capacity, 8)
    assert np.array_equal(output["centerline"][:n, :4], centerline[:n, :4])
    assert np.count_nonzero(output["centerline"][n:]) == 0
    assert np.array_equal(output["images"], source["images"])
    assert np.array_equal(output["poses"], source["poses"])
    assert row["representation_version"] == "explicit_branch_token_tree_v1"
    assert row["token_capacity"] == capacity
    assert _edges(decode_branch_tokens(output["centerline"], mask)) == _edges(edges)
