from __future__ import annotations

import numpy as np
import pytest

from preprocessing.packed12 import decode_packed12, packed12_byte_count


def _pack(values: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.uint16).reshape(-1)
    if np.any(samples > 4095):
        raise ValueError("test samples exceed 12 bits")
    pair_count, has_last = divmod(samples.size, 2)
    packed = np.empty(packed12_byte_count(samples.size), dtype=np.uint8)
    if pair_count:
        pairs = samples[: pair_count * 2].reshape(-1, 2)
        triples = packed[: pair_count * 3].reshape(-1, 3)
        triples[:, 0] = pairs[:, 0] & 0xFF
        triples[:, 1] = ((pairs[:, 0] >> 8) & 0x0F) | (
            (pairs[:, 1] & 0x0F) << 4
        )
        triples[:, 2] = pairs[:, 1] >> 4
    if has_last:
        packed[-2] = samples[-1] & 0xFF
        packed[-1] = (samples[-1] >> 8) & 0x0F
    return packed


def test_decode_preserves_all_nibble_boundaries_and_order() -> None:
    expected = np.asarray(
        [0, 1, 15, 16, 255, 256, 2047, 2048, 3840, 4094, 4095, 1234],
        dtype=np.uint16,
    )
    assert np.array_equal(decode_packed12(_pack(expected)), expected)


def test_decode_supports_matlab_style_odd_final_sample() -> None:
    expected = np.asarray([4095, 0, 2748], dtype=np.uint16)
    assert packed12_byte_count(expected.size) == 5
    assert np.array_equal(
        decode_packed12(_pack(expected), sample_count=expected.size), expected
    )


@pytest.mark.parametrize("sample_count", [-1, 1.5, True])
def test_byte_count_rejects_invalid_sample_counts(sample_count) -> None:
    with pytest.raises(ValueError):
        packed12_byte_count(sample_count)


def test_decode_rejects_partial_or_mismatched_groups() -> None:
    with pytest.raises(ValueError):
        decode_packed12(np.zeros(2, dtype=np.uint8))
    with pytest.raises(ValueError):
        decode_packed12(np.zeros(3, dtype=np.uint8), sample_count=1)
