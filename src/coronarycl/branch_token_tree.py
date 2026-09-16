"""Frozen Stage-2 explicit branch-token tree representation."""

from __future__ import annotations

import numpy as np


def _validate_tree(mask, edges):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n < 1 or not mask[:n].all() or mask[n:].any():
        raise ValueError("valid-node mask must be a non-empty contiguous prefix")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape != (n - 1, 2):
        raise ValueError("edges must have shape (N-1, 2)")
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge index out of range")

    neighbours = [[] for _ in range(n)]
    for left, right in edges:
        left, right = int(left), int(right)
        if left == right:
            raise ValueError("self-loop")
        neighbours[left].append(right)
        neighbours[right].append(left)

    seen, stack = {0}, [0]
    while stack:
        node = stack.pop()
        for other in neighbours[node]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    if len(seen) != n:
        raise ValueError("edges are not a connected tree")
    return n, neighbours


def encode_branch_tokens(centerline, mask, edges):
    """Return (M,8): xyzr plus four frozen normalized topology targets."""
    centerline = np.asarray(centerline)
    mask = np.asarray(mask, dtype=bool)
    n, neighbours = _validate_tree(mask, edges)
    capacity = len(mask)
    scale = max(1, capacity - 1)
    out = np.zeros((capacity, 8), dtype=np.float32)
    out[:n, :4] = centerline[:n, :4]

    records = {}
    next_branch_id = 1

    def emit_branch(start, previous, parent_branch_id, parent_attach_index):
        nonlocal next_branch_id
        branch_id = next_branch_id
        next_branch_id += 1
        nodes, current, prior = [start], start, previous

        while True:
            children = sorted(node for node in neighbours[current] if node != prior)
            if len(children) != 1:
                break
            prior, current = current, children[0]
            nodes.append(current)

        for within_index, node in enumerate(nodes):
            records[node] = (
                branch_id, parent_branch_id, parent_attach_index, within_index
            )

        terminal = nodes[-1]
        for child in sorted(node for node in neighbours[terminal] if node != prior):
            emit_branch(child, terminal, branch_id, len(nodes) - 1)

    emit_branch(0, None, 0, 0)
    if set(records) != set(range(n)):
        raise AssertionError("branch construction did not assign every valid node once")

    for node in range(n):
        out[node, 4:] = np.asarray(records[node], dtype=np.float32) / scale
    return out


def decode_branch_tokens(tokens, mask):
    """Recover edges solely from decoded tokens and the valid-node mask."""
    tokens = np.asarray(tokens)
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if tokens.shape != (len(mask), 8):
        raise ValueError("tokens must have shape (M,8)")
    if n < 1 or not mask[:n].all() or mask[n:].any():
        raise ValueError("valid-node mask must be a non-empty contiguous prefix")

    scale = max(1, len(mask) - 1)
    labels = np.rint(tokens[:n, 4:] * scale).astype(np.int64)
    groups = {}
    for node, (branch_id, parent_id, attach_index, within_index) in enumerate(labels):
        if branch_id < 1:
            continue
        groups.setdefault(int(branch_id), []).append(
            (int(within_index), node, int(parent_id), int(attach_index))
        )

    accepted = {}
    root = groups.get(1, [])
    if not root or any(row[2] != 0 for row in root):
        return np.empty((0, 2), dtype=np.int32)

    def valid_group(rows):
        order = [row[0] for row in rows]
        return len(set(order)) == len(order) and all(row[2:] == rows[0][2:] for row in rows)

    if not valid_group(root):
        return np.empty((0, 2), dtype=np.int32)
    accepted[1] = sorted(root)

    changed = True
    while changed:
        changed = False
        for branch_id in sorted(groups):
            if branch_id in accepted:
                continue
            rows = groups[branch_id]
            if not valid_group(rows):
                continue
            parent_id, attach_index = rows[0][2], rows[0][3]
            parent = accepted.get(parent_id)
            if parent is None:
                continue
            attachment = next((row for row in parent if row[0] == attach_index), None)
            if attachment is None:
                continue
            accepted[branch_id] = sorted(rows)
            changed = True

    edges = []
    for branch_id, rows in accepted.items():
        for left, right in zip(rows, rows[1:]):
            edges.append((left[1], right[1]))
        if branch_id != 1:
            parent = accepted[rows[0][2]]
            attachment = next(row for row in parent if row[0] == rows[0][3])
            edges.append((attachment[1], rows[0][1]))
    return np.asarray(edges, dtype=np.int32).reshape(-1, 2)
