import random

import pytest

from drill import load
from tests.golden import golden

m = load("a1_cta_tiling")
G = golden("a1_cta_tiling")["cases"]


def test_num_tiles_rounds_up():
    assert m.num_tiles(64, 32) == 2
    assert m.num_tiles(70, 32) == 3
    assert m.num_tiles(1, 32) == 1
    assert m.num_tiles(40, 16) == 3


@pytest.mark.parametrize("bad", [(0, 32), (64, 0), (-1, 4), (8, -2)])
def test_num_tiles_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        m.num_tiles(*bad)


@pytest.mark.parametrize("case", G, ids=lambda c: f"{c['M']}x{c['N']}x{c['K']}")
def test_cta_grid_matches_cute(case):
    assert tuple(m.cta_grid(case["M"], case["N"], case["BM"], case["BN"])) == tuple(case["grid"])
    assert m.num_tiles(case["K"], case["BK"]) == case["k_tiles"]


@pytest.mark.parametrize("case", G, ids=lambda c: f"{c['M']}x{c['N']}x{c['K']}")
def test_c_tile_coords_match_local_tile(case):
    for e in case["c"]:
        got = m.c_tile_coords(*e["pid"], case["BM"], case["BN"])
        assert [tuple(x) for x in got] == [(r, c) for r, c, _ in e["coords"]], e["pid"]


@pytest.mark.parametrize("case", G, ids=lambda c: f"{c['M']}x{c['N']}x{c['K']}")
def test_a_and_b_tile_coords_match_local_tile(case):
    for e in case["a"]:
        got = m.a_tile_coords(e["pid_m"], e["k"], case["BM"], case["BK"])
        assert [tuple(x) for x in got] == [(r, c) for r, c, _ in e["coords"]]
    for e in case["b"]:
        got = m.b_tile_coords(e["pid_n"], e["k"], case["BN"], case["BK"])
        assert [tuple(x) for x in got] == [(r, c) for r, c, _ in e["coords"]]


def test_a_view_does_not_depend_on_pid_n():
    # 同一行的两个 CTA 读同一块 A：a_tile_coords 根本没有 pid_n 参数，这里确认 C 视图确实不同。
    assert m.c_tile_coords(0, 0, 32, 32) != m.c_tile_coords(0, 1, 32, 32)
    assert m.a_tile_coords(0, 2, 32, 16)[0] == (0, 32)


@pytest.mark.parametrize("case", G, ids=lambda c: f"{c['M']}x{c['N']}x{c['K']}")
def test_predicate_matches_elem_less(case):
    for e in case["c"]:
        coords = [(r, c) for r, c, _ in e["coords"]]
        assert m.predicate(coords, (case["M"], case["N"])) == [bool(p) for _, _, p in e["coords"]]
    for e in case["a"]:
        coords = [(r, c) for r, c, _ in e["coords"]]
        assert m.predicate(coords, (case["M"], case["K"])) == [bool(p) for _, _, p in e["coords"]]


def test_ragged_tile_keeps_full_static_shape():
    # 最后一块不缩小：70×50 的 CTA (2,1) 仍有 32×32 个坐标，只是谓词屏蔽掉越界部分。
    cs = m.c_tile_coords(2, 1, 32, 32)
    assert len(cs) == 32 * 32
    live = sum(m.predicate(cs, (70, 50)))
    assert live == (70 - 64) * (50 - 32)


@pytest.mark.parametrize("case", G, ids=lambda c: f"{c['M']}x{c['N']}x{c['K']}")
def test_view_offset_is_pure_translation(case):
    # 真实 CuTe：行主序 C 上 CTA (0,1) 视图的 4×4 角落的偏移。
    N = case["N"]
    want = case["view01_offsets_4x4"]
    got = [m.view_offset((r, c), (0, 32), N) for r in range(4) for c in range(4)]
    assert got == want


def test_verify_cover_on_running_examples():
    assert m.verify_cover(64, 64, 32, 32)
    assert m.verify_cover(70, 50, 32, 32)


def test_verify_cover_random_shapes():
    rng = random.Random(1)
    for _ in range(30):
        M, N = rng.randint(1, 90), rng.randint(1, 90)
        BM, BN = rng.choice([8, 16, 32]), rng.choice([8, 16, 32])
        assert m.verify_cover(M, N, BM, BN)


def test_view_offset_matches_numpy_slicing():
    import numpy as np
    A = np.arange(64 * 64).reshape(64, 64)
    flat = A.reshape(-1)
    for (pm, pn) in [(0, 1), (1, 0), (1, 1)]:
        for r, c in [(0, 0), (5, 7), (31, 31)]:
            assert flat[m.view_offset((r, c), (32 * pm, 32 * pn), 64)] == A[32 * pm:32 * pm + 32, 32 * pn:32 * pn + 32][r, c]


def test_gemm_built_only_from_views_and_predicates():
    import numpy as np
    rng = np.random.default_rng(0)
    M, N, K, BM, BN, BK = 70, 50, 40, 32, 32, 16
    A, B = rng.standard_normal((M, K)), rng.standard_normal((N, K))
    C = np.zeros((M, N))
    gm, gn = m.cta_grid(M, N, BM, BN)
    for pm in range(gm):
        for pn in range(gn):
            cs = m.c_tile_coords(pm, pn, BM, BN)
            for k in range(m.num_tiles(K, BK)):
                acols = [c for (_, c), ok in zip(m.a_tile_coords(pm, k, BM, BK), m.predicate(m.a_tile_coords(pm, k, BM, BK), (M, K))) if ok]
                ks = sorted(set(acols))
                for (r, c), ok in zip(cs, m.predicate(cs, (M, N))):
                    if ok:
                        C[r, c] += A[r, ks] @ B[c, ks]
    assert np.allclose(C, A @ B.T)
