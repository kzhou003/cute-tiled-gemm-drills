import numpy as np
import pytest

from drill import load
from tests.golden import golden

m = load("a4_mma_fragments")
G = golden("a4_mma_fragments")


def test_group_and_thread_in_group():
    assert (m.group_id(5), m.thread_in_group(5)) == (1, 1)
    assert (m.group_id(31), m.thread_in_group(31)) == (7, 3)
    with pytest.raises(ValueError):
        m.group_id(32)


def test_ptx_examples():
    assert m.ptx_a(5, 0) == (1, 2)
    assert m.ptx_a(5, 2) == (9, 2)
    assert m.ptx_a(5, 5) == (1, 11)
    assert m.ptx_b(5, 3) == (1, 11)
    assert m.ptx_c(5, 3) == (9, 3)
    with pytest.raises(ValueError):
        m.ptx_b(0, 4)


@pytest.mark.parametrize("which", ["A", "B", "C"])
def test_atom_tv_matches_mma_traits(which):
    assert m.atom_tv(which) == G["atom"][which]


@pytest.mark.parametrize("which", ["A", "B", "C"])
def test_tiled_tv_matches_get_layoutX_TV(which):
    assert m.tiled_tv(which) == G["tiled"][which]


@pytest.mark.parametrize("which,rows", [("A", 16), ("B", 8), ("C", 16)])
def test_atom_covers_every_element(which, rows):
    tv = m.atom_tv(which)
    assert sorted(set(tv)) == list(range(len(set(tv))))
    if which != "A":
        assert len(tv) == len(set(tv)) == rows * (16 if which != "C" else 8)


def test_registers_and_fragment_shapes():
    assert m.registers_per_thread() == {"A": 4, "B": 4, "C": 8}
    want = {k: v.replace("_", "") for k, v in G["fragment_shape"].items()}
    for w in "ABC":
        assert str(m.fragment_shape(w)).replace(" ", "") == want[w]


def test_holders_of_a_element_are_replicated_across_n_warps():
    # A 的 (9,2)：wn = 0 和 wn = 1 两个 warp 都要（线程 5 和 69 的第 2 个值）。
    assert sorted(m.holders("A", 9, 2)) == [(5, 2), (69, 2)]
    # C 的每个元素只有一个持有者
    assert m.holders("C", 5, 7) == [(23, 1)]
    for r in range(0, 32, 7):
        for c in range(0, 32, 5):
            assert len(m.holders("C", r, c)) == 1


def test_warp_mma_matches_numpy():
    rng = np.random.default_rng(0)
    A, B, C0 = rng.standard_normal((16, 16)), rng.standard_normal((8, 16)), rng.standard_normal((16, 8))
    a = np.array([[A[m.ptx_a(l, i)] for i in range(8)] for l in range(32)])
    b = np.array([[B[m.ptx_b(l, i)] for i in range(4)] for l in range(32)])
    c = np.array([[C0[m.ptx_c(l, i)] for i in range(4)] for l in range(32)])
    out = m.warp_mma(a, b, c)
    want = C0 + A @ B.T
    for l in range(32):
        for i in range(4):
            assert out[l][i] == pytest.approx(want[m.ptx_c(l, i)])


@pytest.mark.parametrize("K", [16, 64])
def test_cta_tile_gemm_accumulates_in_registers(K):
    rng = np.random.default_rng(K)
    A, B = rng.standard_normal((32, K)), rng.standard_normal((32, K))
    C, st = m.cta_tile_gemm(A, B)
    assert np.allclose(C, A @ B.T)
    assert st == {"mma": 8 * (K // 16), "k_slices": K // 16, "acc_stores": 1024}


def test_cta_tile_gemm_rejects_bad_shapes():
    with pytest.raises(ValueError):
        m.cta_tile_gemm(np.zeros((32, 10)), np.zeros((32, 10)))
    with pytest.raises(ValueError):
        m.cta_tile_gemm(np.zeros((16, 16)), np.zeros((32, 16)))


def test_valid_order():
    assert m.valid_order(m.MAPPINGS)
    assert not m.valid_order(["copy_tv", "cta_tile", "smem_layout", "mma_operand_tv", "accumulator_tv"])
    assert not m.valid_order(m.MAPPINGS[:-1])
    assert not m.valid_order(m.MAPPINGS + ["cta_tile"])


@pytest.mark.parametrize("t", [0, 5, 37, 127])
def test_thread_fragments_match_real_partitions(t):
    part = golden("a5_assemble_gemm")          # partition_A/B/C of identity tensors, real CuTe
    for which, key in (("A", "mma_A"), ("B", "mma_B"), ("C", "mma_C")):
        tv = m.tiled_tv(which)
        got = [(tv[t + 128 * v] % 32, tv[t + 128 * v] // 32) for v in range(8)]
        assert got == [tuple(x) for x in part[key][t]], which


def test_replication_counts():
    for r in range(0, 32, 5):
        for k in range(0, 16, 3):
            assert len(m.holders("A", r, k)) == 2
            assert len(m.holders("B", r, k)) == 2
    assert m.holders("B", 1, 2) == [(5, 0), (37, 0)]
