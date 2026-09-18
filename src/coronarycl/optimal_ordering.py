"""Exact constructive linear ordering for a tree.

The objective is to maximize the number of real tree edges whose endpoints
are adjacent in the returned node permutation. For a tree, this is equivalent
to selecting a maximum-cardinality subset of edges with selected degree at
most two. The selected subgraph is a linear forest (a disjoint union of
paths), and concatenating those paths realizes every selected edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimalOrdering:
    """Result of exact tree linearization."""

    permutation: np.ndarray
    selected_edges: np.ndarray
    max_consecutive_edges: int


def _canonical_tree_edges(n_nodes: int, edges) -> np.ndarray:
    """Validate and canonicalize a connected tree edge list."""
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be positive")

    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if edge_array.size:
        if edge_array.min() < 0 or edge_array.max() >= n:
            raise ValueError("edge index is outside [0, n_nodes)")
        if np.any(edge_array[:, 0] == edge_array[:, 1]):
            raise ValueError("self-loops are not allowed")
        edge_array = np.sort(edge_array, axis=1)
        edge_array = edge_array[np.lexsort(
            (edge_array[:, 1], edge_array[:, 0]))]
        if len({tuple(edge) for edge in edge_array.tolist()}) != len(edge_array):
            raise ValueError("duplicate edges are not allowed")

    if len(edge_array) != n - 1:
        raise ValueError(
            f"expected a tree with N-1 edges; got N={n}, E={len(edge_array)}")

    adjacency = [[] for _ in range(n)]
    for left, right in edge_array:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if len(seen) != n:
        raise ValueError("edges do not form one connected tree")
    return edge_array


def count_consecutive_tree_edges(permutation, edges) -> int:
    """Count tree edges realized by adjacent positions in a permutation."""
    order = np.asarray(permutation, dtype=np.int64)
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_set = {tuple(sorted((int(left), int(right))))
                for left, right in edge_array}
    return sum(
        tuple(sorted((int(left), int(right)))) in edge_set
        for left, right in zip(order[:-1], order[1:])
    )


def optimal_tree_ordering(n_nodes: int, edges, *, root: int = 0) -> OptimalOrdering:
    """Return an exact maximum-consecutive-edge ordering of a tree.

    Dynamic programming chooses a maximum-cardinality degree-at-most-two
    subgraph. Because the input is a tree, every selected component is a path.
    The paths are then oriented and concatenated deterministically.
    """
    n = int(n_nodes)
    edge_array = _canonical_tree_edges(n, edges)
    root = int(root)
    if root < 0 or root >= n:
        raise ValueError("root is outside [0, n_nodes)")
    if n == 1:
        return OptimalOrdering(
            permutation=np.array([0], dtype=np.int64),
            selected_edges=np.empty((0, 2), dtype=np.int64),
            max_consecutive_edges=0,
        )

    adjacency = [[] for _ in range(n)]
    for left, right in edge_array:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    for neighbors in adjacency:
        neighbors.sort()

    parent = np.full(n, -2, dtype=np.int64)
    parent[root] = -1
    traversal = []
    stack = [root]
    while stack:
        node = stack.pop()
        traversal.append(node)
        for neighbor in reversed(adjacency[node]):
            if neighbor == parent[node]:
                continue
            if parent[neighbor] != -2:
                raise ValueError("cycle detected while rooting tree")
            parent[neighbor] = node
            stack.append(neighbor)

    children = [[] for _ in range(n)]
    for node in range(n):
        if parent[node] >= 0:
            children[int(parent[node])].append(node)

    # dp[node, p] is the optimum below node when its parent edge is selected
    # (p=1) or not selected (p=0). Selecting a parent edge consumes one of the
    # node's two available selected-edge slots.
    dp = np.zeros((n, 2), dtype=np.int64)
    for node in reversed(traversal):
        base = sum(int(dp[child, 0]) for child in children[node])
        gains = sorted(
            (1 + int(dp[child, 1]) - int(dp[child, 0]), child)
            for child in children[node]
        )
        gains.reverse()
        for parent_selected in (0, 1):
            capacity = 2 - parent_selected
            positive = [gain for gain, _ in gains[:capacity] if gain > 0]
            dp[node, parent_selected] = base + sum(positive)

    selected = []
    backtrack = [(root, 0)]
    while backtrack:
        node, parent_selected = backtrack.pop()
        gains = sorted(
            (1 + int(dp[child, 1]) - int(dp[child, 0]), child)
            for child in children[node]
        )
        gains.reverse()
        capacity = 2 - parent_selected
        selected_children = {
            child for gain, child in gains[:capacity] if gain > 0
        }
        for child in reversed(children[node]):
            is_selected = child in selected_children
            if is_selected:
                selected.append(tuple(sorted((node, child))))
            backtrack.append((child, int(is_selected)))

    selected_array = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected_array.size == 0:
        selected_array = np.empty((0, 2), dtype=np.int64)
    optimum = int(dp[root, 0])
    if len(selected_array) != optimum:
        raise AssertionError("DP backtracking did not recover the optimum")

    selected_adjacency = [[] for _ in range(n)]
    for left, right in selected_array:
        selected_adjacency[int(left)].append(int(right))
        selected_adjacency[int(right)].append(int(left))
    if max(map(len, selected_adjacency), default=0) > 2:
        raise AssertionError("selected subgraph is not a linear forest")

    paths = []
    unseen = set(range(n))
    while unseen:
        component_seed = min(unseen)
        component = []
        component_seen = {component_seed}
        component_stack = [component_seed]
        while component_stack:
            node = component_stack.pop()
            component.append(node)
            for neighbor in selected_adjacency[node]:
                if neighbor not in component_seen:
                    component_seen.add(neighbor)
                    component_stack.append(neighbor)
        unseen -= component_seen

        endpoints = sorted(
            node for node in component if len(selected_adjacency[node]) <= 1)
        if not endpoints:
            raise AssertionError("a selected component unexpectedly contains a cycle")
        start = endpoints[0]
        path = []
        previous = -1
        node = start
        while True:
            path.append(node)
            candidates = [neighbor for neighbor in selected_adjacency[node]
                          if neighbor != previous]
            if not candidates:
                break
            if len(candidates) != 1:
                raise AssertionError("selected component is not a path")
            previous, node = node, candidates[0]
        if len(path) != len(component):
            raise AssertionError("path reconstruction lost selected nodes")
        paths.append(path)

    # The original DFS indices provide a deterministic tie-break without
    # changing the exact objective. Each path is oriented toward its smaller
    # endpoint and path components are ordered by their smallest node index.
    paths.sort(key=lambda path: min(path))
    permutation = np.asarray(
        [node for path in paths for node in path], dtype=np.int64)
    if len(permutation) != n or len(np.unique(permutation)) != n:
        raise AssertionError("constructed ordering is not a permutation")

    achieved = count_consecutive_tree_edges(permutation, edge_array)
    if achieved != optimum:
        raise AssertionError(
            f"constructed ordering achieved {achieved}, expected {optimum}")

    return OptimalOrdering(
        permutation=permutation,
        selected_edges=selected_array,
        max_consecutive_edges=optimum,
    )

