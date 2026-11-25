import pytest

from drill import load
from tests.golden import golden

m = load("a2_copy_tv")
G = {v["name"]: v for v in golden("a2_copy_tv")["variants"]}

VARIANTS = {
    "base": ((32, 4), (4, 1), (1, 4)),
    "colthr": ((32, 4), (1, 32), (1, 4)),
    "t8x16_v4x1": ((8, 16), (16, 1), (4, 1)),
    "t16x8_v2x2": ((16, 8), (8, 1), (2, 2)),
}


@pytest.mark.parametrize("name", VARIANTS)
def test_golden_variants_tile_32x16(name):
    (TM, TK), _, (VM, VK) = VARIANTS[name]
    assert (TM * VM, TK * VK) == (32, 16)
    assert G[name]["tiler"] == "(_32,_16)"


def test_thread_coord_inverts_thread_layout():
    assert m.thread_coord((32, 4), (4, 1), 5) == (1, 1)
    assert m.thread_coord((32, 4), (4, 1), 127) == (31, 3)
    assert m.thread_coord((32, 4), (1, 32), 5) == (5, 0)
    assert m.thread_coord((8, 16), (16, 1), 17) == (1, 1)


@pytest.mark.parametrize("bad", [((4, 4), (2, 1)), ((4, 4), (4, 2)), ((2, 2), (1, 1))])
def test_non_bijective_thread_layout_raises(bad):
    with pytest.raises(ValueError):
        m.thread_coord(bad[0], bad[1], 0)


@pytest.mark.parametrize("name", VARIANTS)
def test_copy_coords_match_real_partition_S(name):
    got = m.copy_coords(*VARIANTS[name])
    V = len(G[name]["threads"][0])
    for t, want in enumerate(G[name]["threads"]):
        assert [tuple(x) for x in got[t * V:(t + 1) * V]] == [tuple(x) for x in want], f"thread {t}"


def test_document_example_thread_5():
    cs = m.copy_coords((32, 4), (4, 1), (1, 4))
    assert cs[20:24] == [(1, 4), (1, 5), (1, 6), (1, 7)]


def test_values_are_numbered_colex():
    # (2,2) 的值按列主序编号：v1 在 v0 下面，不在右边。
    cs = m.copy_coords((16, 8), (8, 1), (2, 2))
    assert cs[0:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]


@pytest.mark.parametrize("name", VARIANTS)
def test_tv_index_map_matches_real_get_layoutS_TV(name):
    assert m.tv_index_map(*VARIANTS[name]) == G[name]["tv_values"]


@pytest.mark.parametrize("name", VARIANTS)
def test_every_element_copied_exactly_once(name):
    cs = m.copy_coords(*VARIANTS[name])
    assert sorted(cs) == [(r, c) for r in range(32) for c in range(16)]


def test_byte_addr():
    assert m.byte_addr(0, 0) == 0
    assert m.byte_addr(1, 4) == (64 + 4) * 2
    assert m.byte_addr(1, 4, m0=32, k0=16, ld=64) == ((33) * 64 + 20) * 2
    assert m.byte_addr(2, 3, ld=128, elem_bytes=4) == (2 * 128 + 3) * 4


@pytest.mark.parametrize("name,w", [("base", 4), ("colthr", 4), ("t8x16_v4x1", 1), ("t16x8_v2x2", 1)])
def test_vector_width_per_variant(name, w):
    assert m.vector_elems(*VARIANTS[name]) == w


def test_misaligned_origin_shrinks_vector():
    assert m.vector_elems(*VARIANTS["base"], k0=2) == 2
    s = m.copy_stats(*VARIANTS["base"], k0=2)
    assert s["instr_per_thread"] == 2
    assert s["sectors_per_warp"] == [24, 24, 24, 24]
    assert s["total_sectors"] == 96


def test_stats_two_kinds_of_contiguity():
    base = m.copy_stats(*VARIANTS["base"])
    assert (base["vec_elems"], base["vec_bytes"], base["instr_per_thread"]) == (4, 8, 1)
    assert base["sectors_per_warp"] == [8, 8, 8, 8]
    assert base["total_sectors"] == 32 and base["efficiency"] == pytest.approx(1.0)
    col = m.copy_stats(*VARIANTS["colthr"])          # 线程内部连续，线程之间不连续
    assert col["instr_per_thread"] == 1 and col["total_sectors"] == 128
    assert col["efficiency"] == pytest.approx(0.25)
    v41 = m.copy_stats(*VARIANTS["t8x16_v4x1"])       # 线程之间连续，线程内部不连续
    assert v41["instr_per_thread"] == 4 and v41["total_sectors"] == 32
    v22 = m.copy_stats(*VARIANTS["t16x8_v2x2"])       # 两种都不好
    assert v22["instr_per_thread"] == 4 and v22["sectors_per_warp"] == [16, 16, 16, 16]
    assert v22["total_sectors"] == 64


def test_rank_variants():
    assert m.rank_variants(VARIANTS) == ["base", "colthr", "t8x16_v4x1", "t16x8_v2x2"]
