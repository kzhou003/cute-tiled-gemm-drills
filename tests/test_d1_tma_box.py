import numpy as np
import pytest

from drill import load
from tests.golden import golden

m = load("d1_tma_box")
G = golden("d1_tma_box")
D = m.TmaDesc


def codes(desc):
    return [e.split(":")[0] for e in m.validate(desc)]


def test_running_example_is_valid():
    # B tile：64×64 half，K 连续，行 stride 128 字节，128B swizzle
    assert m.validate(D(2, (64, 64), (128,), (64, 64), swizzle="128B")) == []


@pytest.mark.parametrize("desc,want", [
    (D(2, (64, 64), (128,), (64,)), ["E_RANK"]),
    (D(2, (1,) * 6, (16,) * 5, (1,) * 6), ["E_RANK"]),
    (D(2, (64, 64), (128,), (64, 64), base_addr=8), ["E_ALIGN"]),
    (D(2, (0, 64), (128,), (64, 64)), ["E_GDIM"]),
    (D(2, (2 ** 32 + 1, 64), (2 ** 33 + 16,), (64, 64)), ["E_GDIM"]),
    (D(2, (60, 64), (120,), (64, 64)), ["E_GSTRIDE_ALIGN"]),
    (D(2, (64, 64), (2 ** 40,), (64, 64)), ["E_GSTRIDE_MAX"]),
    (D(2, (64, 512), (128,), (64, 257)), ["E_BOX"]),
    (D(2, (64, 64), (128,), (64, 64), elem_strides=(1, 9)), ["E_ESTRIDE"]),
    (D(2, (64, 64), (128,), (4, 64)), ["E_INNER_16B"]),
    (D(2, (64, 64), (128,), (64, 64), swizzle="256B"), ["E_SWIZZLE_MODE"]),
    (D(2, (64, 64), (128,), (64, 64), swizzle="64B"), ["E_SWIZZLE_SPAN"]),
    (D(1, (100,), (), (8,), base_addr=8), ["E_ALIGN", "E_INNER_16B"]),
], ids=lambda x: "" if isinstance(x, list) else None)
def test_validate_rules(desc, want):
    assert codes(desc) == want


def test_expect_tx_counts_every_box_element():
    assert m.expect_tx_bytes(D(2, (64, 64), (128,), (64, 64), swizzle="128B")) == 8192
    # 越界也算：box 64×64 落在 40×70 的边上，字节数不变
    assert m.expect_tx_bytes(D(2, (40, 70), (80,), (64, 64))) == 8192
    d = D(2, (64, 64), (128,), (64, 64), elem_strides=(1, 2))
    assert m.elements_per_dim(d) == (64, 32)
    assert m.expect_tx_bytes(d) == 4096
    assert m.elements_per_dim(D(4, (64, 64), (256,), (64, 5), elem_strides=(1, 2))) == (64, 3)


def test_swizzle_byte_is_cute_swizzle_B43():
    assert m.swizzle_byte(128, "128B") == 128 ^ 16
    assert m.swizzle_byte(7 * 128 + 5, "128B") == 7 * 128 + (5 ^ (7 << 4))
    assert m.swizzle_byte(3 * 128, "32B") == 3 * 128 ^ 16
    assert m.swizzle_byte(1000, "none") == 1000
    for mode in ("32B", "64B", "128B"):
        assert all(m.swizzle_byte(m.swizzle_byte(x, mode), mode) == x for x in range(2048))


def test_smem_byte_offset():
    d = D(2, (64, 64), (128,), (64, 64), swizzle="128B")
    assert m.smem_byte_offset(d, (0, 0)) == 0
    assert m.smem_byte_offset(d, (0, 1)) == 144
    assert m.smem_byte_offset(d, (8, 1)) == 128
    assert m.smem_byte_offset(d, (3, 7)) == 1014


@pytest.mark.parametrize("t", G["tiles"][:3], ids=lambda t: t["mode"])
def test_tile_element_map_matches_gmma_swizzle_atoms(t):
    assert m.tile_element_map(t["rows"], t["inner"], 2, t["mode"]) == t["offsets"]


def _gmem(rows, cols, ld_elems, dtype=np.uint16):
    a = (np.arange(rows * ld_elems, dtype=np.int64) % 60000 + 1).astype(dtype).reshape(rows, ld_elems)
    return a, a.view(np.uint8).reshape(-1)


@pytest.mark.parametrize("swz", ["none", "32B", "64B", "128B"])
def test_box_load_round_trips_through_swizzle(swz):
    inner = {"none": 64, "32B": 16, "64B": 32, "128B": 64}[swz]
    a, raw = _gmem(64, 64, 64)
    d = D(2, (64, 64), (128,), (inner, 32), swizzle=swz)
    x0 = 64 - inner if inner < 64 else 0
    smem = m.box_load(raw, d, (x0, 16))
    assert smem.dtype == np.uint8 and len(smem) == m.expect_tx_bytes(d)
    box = m.unswizzle(d, smem, np.uint16)
    assert box.shape == (32, inner)
    assert (box == a[16:48, x0:x0 + inner]).all()


def test_box_load_fills_out_of_bounds_with_zero():
    a, raw = _gmem(70, 40, 40)
    d = D(2, (40, 70), (80,), (32, 32), swizzle="64B")
    box = m.unswizzle(d, m.box_load(raw, d, (16, 50)), np.uint16)
    want = np.zeros((32, 32), np.uint16)
    want[:20, :24] = a[50:70, 16:40]
    assert (box == want).all()
    neg = m.unswizzle(d, m.box_load(raw, d, (-8, -4)), np.uint16)
    assert (neg[:4, :] == 0).all() and (neg[:, :8] == 0).all()
    assert (neg[4:, 8:] == a[0:28, 0:24]).all()


def test_box_load_with_element_strides():
    a, raw = _gmem(64, 64, 64)
    d = D(2, (64, 64), (128,), (64, 16), elem_strides=(1, 2))
    box = m.unswizzle(d, m.box_load(raw, d, (0, 0)), np.uint16)
    assert box.shape == (8, 64)
    assert (box == a[0:16:2, :]).all()


def test_box_load_rejects_invalid_descriptor():
    _, raw = _gmem(64, 64, 64)
    with pytest.raises(ValueError):
        m.box_load(raw, D(2, (64, 64), (128,), (64, 64), swizzle="64B"), (0, 0))


def test_document_examples():
    a = D(2, (64, 64), (128,), (16, 32), swizzle="32B")
    assert m.validate(a) == [] and m.expect_tx_bytes(a) == 1024
    assert codes(D(2, (64, 64), (100,), (16, 512), swizzle="32B", base_addr=4)) == ["E_ALIGN", "E_GSTRIDE_ALIGN", "E_BOX"]
    d = D(2, (64, 64), (128,), (16, 32), elem_strides=(1, 3))
    assert m.elements_per_dim(d) == (16, 11) and m.expect_tx_bytes(d) == 352
    assert [m.swizzle_byte(o, "128B") for o in (0, 16, 128, 144, 256, 1152)] == [0, 16, 144, 128, 288, 1168]
    b = D(2, (64, 256), (128,), (64, 64), swizzle="128B")
    assert [m.smem_byte_offset(b, (0, r)) for r in range(8)] == [0, 144, 288, 432, 576, 720, 864, 1008]


def test_swizzle_never_splits_a_16_byte_chunk_or_leaves_its_128_byte_row():
    for mode in ("32B", "64B", "128B"):
        for off in range(0, 4096):
            s = m.swizzle_byte(off, mode)
            assert s % 16 == off % 16 and s // 128 == off // 128
