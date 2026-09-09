"""Authoritative packed unsigned-12 little-endian decoding primitives.

The acquisition format stores two samples in three bytes.  The first sample
uses byte 0 plus the low nibble of byte 1; the second uses the high nibble of
byte 1 plus byte 2.  Keep this module limited to byte decoding so all TIFF,
MIP and denoising paths share exactly the same sample order.
"""

from __future__ import annotations

import numpy as np


BITS_PER_SAMPLE = 12


def packed12_byte_count(sample_count: int) -> int:
    """Return the number of bytes used by MATLAB-style packed ubit12 values."""
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 0
    ):
        raise ValueError("sample_count 必须是非负整数。")
    return (sample_count * BITS_PER_SAMPLE + 7) // 8


def decode_packed12(
    packed: np.ndarray,
    *,
    sample_count: int | None = None,
) -> np.ndarray:
    """Decode packed little-endian unsigned-12 samples without reordering.

    When ``sample_count`` is omitted, the input must contain complete
    three-byte/two-sample groups.  Supplying an odd ``sample_count`` also
    permits the final sample to be represented by its two low-order bytes.
    """
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 1:
        raise ValueError("packed-12 输入必须是一维 uint8 数组。")
    if sample_count is None:
        if values.size % 3:
            raise ValueError("packed-12 字节数必须是 3 的整数倍。")
        sample_count = values.size // 3 * 2
    elif (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 0
    ):
        raise ValueError("sample_count 必须是非负整数。")

    expected_bytes = packed12_byte_count(sample_count)
    if values.size != expected_bytes:
        raise ValueError(
            f"{sample_count} 个 packed-12 样本应为 {expected_bytes} 字节，"
            f"实际为 {values.size} 字节。"
        )

    pair_count, has_last_sample = divmod(sample_count, 2)
    decoded = np.empty(sample_count, dtype=np.uint16)
    if pair_count:
        triples = values[: pair_count * 3].reshape(-1, 3)
        decoded[: pair_count * 2 : 2] = triples[:, 0].astype(np.uint16) | (
            (triples[:, 1] & 0x0F).astype(np.uint16) << 8
        )
        decoded[1 : pair_count * 2 : 2] = (
            (triples[:, 1] >> 4).astype(np.uint16)
            | (triples[:, 2].astype(np.uint16) << 4)
        )
    if has_last_sample:
        tail = values[pair_count * 3 :]
        decoded[-1] = np.uint16(tail[0]) | (
            np.uint16(tail[1] & 0x0F) << np.uint16(8)
        )
    return decoded
