import numpy as np

from src.coronarycl.branch_token_tree import (
    decode_branch_tokens,
    encode_branch_tokens,
)


def _edge_set(edges):
    return {tuple(sorted(map(int, edge))) for edge in np.asarray(edges)}


def test_branch_tokens_round_trip_preserves_geometry_padding_and_tree():
    capacity, n = 8, 6
    centerline = np.zeros((capacity, 5), dtype=np.float32)
    centerline[:n, :4] = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    centerline[:n, 4] = np.arange(n, dtype=np.float32)
    mask = np.zeros(capacity, dtype=bool)
    mask[:n] = True
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], dtype=np.int32)

    tokens = encode_branch_tokens(centerline, mask, edges)

    assert tokens.shape == (capacity, 8)
    assert np.array_equal(tokens[:n, :4], centerline[:n, :4])
    assert np.count_nonzero(tokens[n:]) == 0

    decoded_edges = decode_branch_tokens(tokens, mask)

    # Decoder receives only predicted tokens and valid-node mask, never GT edges.
    assert _edge_set(decoded_edges) == _edge_set(edges)


def test_decoder_discards_unresolvable_branch_and_keeps_root_component():
    capacity, n = 8, 6
    centerline = np.zeros((capacity, 5), dtype=np.float32)
    mask = np.zeros(capacity, dtype=bool)
    mask[:n] = True
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], dtype=np.int32)

    tokens = encode_branch_tokens(centerline, mask, edges)
    tokens[2, 5] = 1.0  # parent_branch_id = 7; that branch does not exist

    decoded_edges = decode_branch_tokens(tokens, mask)

    assert _edge_set(decoded_edges) == {(0, 1), (1, 3), (3, 4), (3, 5)}
