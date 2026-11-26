"""F1 · SIMT CuTe kernel: host part vs real CuTe (oracle/f1_simt_cute_gemm.cpp), then the .cu on a GPU."""
from collections import Counter

import pytest

from drill import load
from tests.cu_helpers import run_gpu, typecheck
from tests.golden import golden

m = load("f1_simt_cute_gemm")
G = golden("f1_simt_cute_gemm")
THREADS = sorted(int(t) for t in G["threads"])


@pytest.mark.parametrize("t", THREADS)
def test_copy_coords_match_local_partition(t):
    assert m.copy_coords(t) == [tuple(x) for x in G["threads"][str(t)]["copy"]]


@pytest.mark.parametrize("t", THREADS)
def test_c_coords_match_local_partition_column_major(t):
    assert m.c_coords(t) == [tuple(x) for x in G["threads"][str(t)]["c"]]


@pytest.mark.parametrize("t", THREADS)
def test_smem_rows_read_match_projected_partition(t):
    assert m.a_rows(t) == G["threads"][str(t)]["a_rows"]
    assert m.b_rows(t) == G["threads"][str(t)]["b_rows"]


def test_thread_37_worked_example():
    assert m.copy_coords(37) == [(4, 5), (36, 5), (68, 5), (100, 5)]
    assert m.c_coords(37)[:3] == [(5, 2), (21, 2), (37, 2)]
    assert m.a_rows(37) == [5, 21, 37, 53, 69, 85, 101, 117]


def test_every_tile_element_copied_by_exactly_one_thread():
    seen = Counter(c for t in range(m.THREADS) for c in m.copy_coords(t))
    assert len(seen) == m.BM * m.BK and set(seen.values()) == {1}


def test_every_output_accumulated_by_exactly_one_thread():
    seen = Counter(c for t in range(m.THREADS) for c in m.c_coords(t))
    assert len(seen) == m.BM * m.BN and set(seen.values()) == {1}


def test_thread_out_of_range_raises():
    for f in (m.copy_coords, m.c_coords, m.a_rows, m.b_rows):
        with pytest.raises(ValueError):
            f(256)
        with pytest.raises(ValueError):
            f(-1)


def test_cta_grid_and_divisibility():
    assert m.cta_grid(2048, 1024) == (16, 8)
    for bad in ((100, 128), (128, 0), (0, 128)):
        with pytest.raises(ValueError):
            m.cta_grid(*bad)


def test_counting_formulas():
    assert m.global_bytes(256, 256, 16) == 4 * 2 * (128 + 128) * 8 * 2
    assert m.global_bytes(4096, 4096, 4096) == 32 * 32 * 512 * 4096
    assert m.ffma_per_thread(4096) == 64 * 4096
    assert m.smem_loads_per_ffma() == pytest.approx(0.25)
    with pytest.raises(ValueError):
        m.global_bytes(128, 128, 12)


def test_cu_typechecks():
    typecheck("f1_simt_cute_gemm")


@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (2048, 1024, 512)])
def test_gpu_matches_reference(shape):
    (res,) = run_gpu("f1_simt_cute_gemm", *shape, 5)
    assert res["ok"], res
