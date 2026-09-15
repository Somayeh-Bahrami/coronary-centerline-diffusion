import itertools

import numpy as np
import pytest

from src.coronarycl.optimal_ordering import (
    count_consecutive_tree_edges,
    optimal_tree_ordering,
)


def _brute_force_optimum(n, edges):
    return max(
        count_consecutive_tree_edges(order, edges)
        for order in itertools.permutations(range(n))
    )


def _random_tree(n, rng):
    return np.asarray(
        [(node, int(rng.integers(0, node))) for node in range(1, n)],
        dtype=np.int64,
    )


def test_path_keeps_every_edge_consecutive():
    edges = np.asarray([(node, node + 1) for node in range(11)])
    result = optimal_tree_ordering(12, edges)
    assert result.max_consecutive_edges == 11
    assert count_consecutive_tree_edges(result.permutation, edges) == 11


@pytest.mark.parametrize("degree", [3, 5, 8])
def test_star_has_exactly_two_consecutive_edges(degree):
    edges = np.asarray([(0, leaf) for leaf in range(1, degree + 1)])
    result = optimal_tree_ordering(degree + 1, edges)
    assert result.max_consecutive_edges == 2
    assert count_consecutive_tree_edges(result.permutation, edges) == 2


def test_dp_matches_brute_force_on_small_random_trees():
    rng = np.random.default_rng(20260915)
    for n in range(2, 9):
        for _ in range(8):
            edges = _random_tree(n, rng)
            result = optimal_tree_ordering(n, edges)
            assert result.max_consecutive_edges == _brute_force_optimum(n, edges)


def test_result_is_root_invariant_in_objective():
    rng = np.random.default_rng(7)
    edges = _random_tree(200, rng)
    values = {
        optimal_tree_ordering(200, edges, root=root).max_consecutive_edges
        for root in [0, 17, 91, 199]
    }
    assert len(values) == 1


def test_result_is_a_complete_permutation_and_selected_degree_is_two():
    rng = np.random.default_rng(19)
    edges = _random_tree(100, rng)
    result = optimal_tree_ordering(100, edges)
    assert np.array_equal(np.sort(result.permutation), np.arange(100))
    degree = np.bincount(result.selected_edges.ravel(), minlength=100)
    assert degree.max() <= 2


@pytest.mark.parametrize(
    "n, edges, message",
    [
        (3, [(0, 1)], "N-1"),
        (3, [(0, 1), (0, 1)], "duplicate"),
        (3, [(0, 1), (1, 3)], "outside"),
        (3, [(0, 0), (0, 1)], "self-loop"),
    ],
)
def test_invalid_trees_are_rejected(n, edges, message):
    with pytest.raises(ValueError, match=message):
        optimal_tree_ordering(n, edges)

