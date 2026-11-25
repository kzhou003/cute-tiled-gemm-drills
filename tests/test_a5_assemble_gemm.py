import numpy as np
import pytest

from drill import load
from tests.golden import golden

m = load("a5_assemble_gemm")
G = golden("a5_assemble_gemm")


def _ab(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-3, 4, (64, 64)).astype(float), rng.integers(-3, 4, (64, 64)).astype(float)


def test_given_copy_matches_real_tiled_copy():
    for t in range(128):
        assert [tuple(x) for x in m.copy_coords(t)] == [tuple(x) for x in G["copy"][t]]


def test_given_tiled_tv_matches_real_partitions():
    for which, key in (("A", "mma_A"), ("B", "mma_B"), ("C", "mma_C")):
        tv = m.tiled_tv(which)
        for t in range(128):
            got = [(tv[t + 128 * v] % 32, tv[t + 128 * v] // 32) for v in range(8)]
            assert got == [tuple(x) for x in G[key][t]], (which, t)


def test_given_smem_offset_matches_swizzle_222():
    for r in range(32):
        assert [m.smem_offset(r, c) for c in range(16)] == G["smem_sw222"][r]


def test_wavefronts_model():
    assert m.wavefronts([(4 * i, 4) for i in range(32)]) == 1
    assert m.wavefronts([(0, 4), (128, 4)]) == 2            # 同一 bank 两个 word
    assert m.wavefronts([(0, 4), (0, 4)]) == 1              # 广播
    assert m.wavefronts([(16 * i, 16) for i in range(32)]) == 4   # 512 字节至少 4 个 wavefront
    assert m.wavefronts([]) == 0


@pytest.mark.parametrize("swizzle", [True, False, (1, 3, 3)], ids=str)
def test_simulated_gemm_is_correct(swizzle):
    A, B = _ab(1)
    C, st = m.simulate_gemm(A, B, swizzle)
    assert np.allclose(C, A @ B.T)


def test_every_c_element_written_exactly_once():
    A, B = _ab(2)
    _, st = m.simulate_gemm(A, B)
    assert (st["c_writes"] == 1).all()


def test_stats_per_cta():
    A, B = _ab(3)
    _, st = m.simulate_gemm(A, B)
    for bm in range(2):
        for bn in range(2):
            s = st[(bm, bn)]
            assert s["mma"] == 32
            assert s["regs"] == {"A": 4, "B": 4, "C": 8}
            assert s["gmem_sectors"] == 256          # 每个 warp 每条 copy 8 个 sector × 4 warp × 2 × 4 K 片
            assert s["smem_store_wavefronts"] == 64


def test_swizzle_222_does_not_help_loads_but_133_does():
    A, B = _ab(4)
    load = {}
    for sw in (False, True, (1, 3, 3)):
        _, st = m.simulate_gemm(A, B, sw)
        load[str(sw)] = st[(0, 0)]["smem_load_wavefronts"]
    assert load["False"] == 256 and load["True"] == 256
    assert load["(1, 3, 3)"] == 128


def test_rejects_other_shapes():
    with pytest.raises(ValueError):
        m.simulate_gemm(np.zeros((32, 64)), np.zeros((64, 64)))


def test_trace_c_5_7_k_50():
    tr = m.trace(5, 7, 50)
    assert tr["cta"] == (0, 0) and tr["k_tile"] == 3
    assert tr["A"]["tile"] == (5, 2) and tr["A"]["copy"] == (20, 2)
    assert tr["A"]["smem"] == G["smem_sw222"][5][2]
    assert tr["B"]["tile"] == (7, 2) and tr["B"]["copy"] == (28, 2)
    assert tr["C"] == {"tile": (5, 7), "acc": (23, 1)}


def _holders(key, r, c):
    return sorted((t, v) for t in range(128) for v, x in enumerate(G[key][t]) if tuple(x) == (r, c))


@pytest.mark.parametrize("mnk", [(5, 7, 50), (40, 3, 0), (63, 63, 63), (17, 44, 21)])
def test_trace_agrees_with_real_partitions(mnk):
    mm, nn, kk = mnk
    tr = m.trace(mm, nn, kk)
    r, c = tr["A"]["tile"]
    assert sorted(tr["A"]["mma"]) == _holders("mma_A", r, c)
    r, c = tr["B"]["tile"]
    assert sorted(tr["B"]["mma"]) == _holders("mma_B", r, c)
    t, v = tr["A"]["copy"]
    assert tuple(G["copy"][t][v]) == tr["A"]["tile"]
    t, v = tr["C"]["acc"]
    assert tuple(G["mma_C"][t][v]) == tr["C"]["tile"]


def test_trace_rejects_out_of_range():
    with pytest.raises(ValueError):
        m.trace(64, 0, 0)


def test_trace_document_example_k3():
    assert m.trace(5, 7, 3) == {
        "cta": (0, 0), "k_tile": 0,
        "A": {"tile": (5, 3), "copy": (20, 3), "smem": 87, "mma": [(21, 1), (85, 1)]},
        "B": {"tile": (7, 3), "copy": (28, 3), "smem": 127, "mma": [(29, 1), (61, 1)]},
        "C": {"tile": (5, 7), "acc": (23, 1)}}


def test_wavefronts_count_bytes_not_only_conflicts():
    assert m.wavefronts([(8 * i, 8) for i in range(32)]) == 2      # 256 字节，无冲突也要 2 轮
    assert m.wavefronts([(128 * i, 4) for i in range(32)]) == 32
