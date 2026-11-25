import pytest

from drill import load
from tests.golden import golden

m = load("a3_smem_swizzle")
G = golden("a3_smem_swizzle")
SW = {tuple(s["sw"]): s for s in G["swizzled"]}


def test_plain_map_matches_cute():
    assert m.smem_map(None) == G["plain"]["offsets"]


@pytest.mark.parametrize("sw", list(SW), ids=str)
def test_swizzled_map_matches_cute_composition(sw):
    assert m.smem_map(sw) == SW[sw]["offsets"]


def test_swizzle_bit_formula_examples():
    # Sw<2,2,2>：bit [4,6) 右移 2 位异或到 bit [2,4)。
    assert m.swizzle(16, 2, 2, 2) == 20
    assert m.swizzle(48, 2, 2, 2) == 60
    assert m.swizzle(3, 2, 2, 2) == 3
    # 负的 S：左移
    assert m.swizzle(4, 1, 2, -2) == 4 ^ 16      # bit 2 左移 2 位异或到 bit 4
    assert m.swizzle(16, 1, 2, -2) == 16


@pytest.mark.parametrize("bad", [(-1, 2, 2), (2, -1, 2), (3, 3, 2), (2, 2, 1)])
def test_invalid_swizzle_params_raise(bad):
    with pytest.raises(ValueError):
        m.swizzle(0, *bad)


def test_swizzle_is_an_involution():
    for x in range(512):
        assert m.swizzle(m.swizzle(x, 2, 2, 2), 2, 2, 2) == x


@pytest.mark.parametrize("sw", [None] + list(SW), ids=str)
def test_every_swizzle_is_a_bijection(sw):
    assert m.is_bijection(m.smem_map(sw))


def test_is_bijection_detects_collisions():
    assert m.is_bijection([2, 0, 1])
    assert not m.is_bijection([0, 0, 1])
    assert not m.is_bijection([0, 2])


def test_chunk_contiguity_depends_on_M():
    # Sw<2,2,2> 只保留 2^2 = 4 个元素连续：8 个 half 的 ldmatrix 行被拆开。
    assert m.chunk_contiguous((2, 2, 2), 4)
    assert not m.chunk_contiguous((2, 2, 2), 8)
    assert m.chunk_contiguous((1, 3, 3), 8)
    assert m.chunk_contiguous(None, 8)


def test_wavefront_model_basics():
    assert m.wavefronts([4 * i for i in range(32)], 4) == 1          # 32 个 bank 各一个 word
    assert m.wavefronts([128 * i for i in range(32)], 4) == 32       # 全落在 bank 0
    assert m.wavefronts([0] * 32, 4) == 1                             # 广播
    assert m.wavefronts([16 * i for i in range(8)], 16) == 1          # 8 个 16 字节行，一个 phase
    with pytest.raises(ValueError):
        m.wavefronts([2], 4)
    with pytest.raises(ValueError):
        m.wavefronts([0], 2)


def test_access_patterns_have_the_right_shape():
    assert len(m.copy_store_addrs(None)) == 128
    assert m.copy_store_addrs(None)[5] == 2 * (16 * 1 + 4)
    assert m.copy_store_addrs(None, "colthr")[5] == 2 * (16 * 5)
    assert m.ldmatrix_addrs(None, 8, 8) == [2 * (16 * (8 + i) + 8) for i in range(8)]
    assert m.lds32_addrs(None)[5] == 2 * (16 * 1 + 2)


def test_breaking_a_vector_raises():
    with pytest.raises(ValueError):
        m.ldmatrix_addrs((2, 2, 2), 0, 0)


def test_conflict_report_plain():
    assert m.conflict_report(None) == {"store_base": 8, "store_colthr": 32, "ldmatrix": 16, "lds32": 16}


def test_document_swizzle_222_does_not_fix_conflicts():
    r = m.conflict_report((2, 2, 2))
    assert r == {"store_base": 8, "store_colthr": 32, "ldmatrix": None, "lds32": 16}


def test_swizzle_133_fixes_ldmatrix_and_lds32():
    r = m.conflict_report((1, 3, 3))
    assert r["ldmatrix"] == 8 and r["lds32"] == 8 and r["store_base"] == 8


@pytest.mark.parametrize("pattern,want", [("store_base", (0, 0, 0)), ("store_colthr", (2, 2, 4)),
                                          ("ldmatrix", (1, 3, 3)), ("lds32", (1, 3, 3))])
def test_find_min_swizzle(pattern, want):
    assert m.find_min_swizzle(pattern) == want


def test_find_min_swizzle_returns_none_when_search_space_too_small():
    assert m.find_min_swizzle("ldmatrix", max_B=1, max_M=2, max_S=2) is None
