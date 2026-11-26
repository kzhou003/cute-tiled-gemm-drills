"""F2 · first Tensor Core GEMM: copy / SMEM / MMA partitions vs real CuTe (Config<128,128,64,1,2,4>)."""
from collections import Counter

import pytest

from drill import load
from tests.cu_helpers import run_gpu, typecheck
from tests.golden import golden

m = load("f2_tensorcore_gemm")
G = golden("f2_tensorcore_gemm")


@pytest.mark.parametrize("t", sorted(int(t) for t in G["copy"]))
def test_copy_coords_match_partition_S(t):
    assert m.copy_coords(t) == [tuple(x) for x in G["copy"][str(t)]]


def test_copy_starts_are_16_byte_chunks():
    for t in range(m.THREADS):
        starts = m.copy_starts(t)
        assert len(starts) == 4 and all(k % 8 == 0 for _, k in starts)
        assert m.copy_coords(t) == [(r, k + v) for r, k in starts for v in range(8)]


@pytest.mark.parametrize("t", sorted(int(t) for t in G["mma_c"]))
def test_mma_c_coords_match_partition_C(t):
    assert m.mma_c_coords(t) == [tuple(x) for x in G["mma_c"][str(t)]]


def test_smem_offset_table_matches_swizzled_layout():
    table = G["smem_offset"]
    for r in range(m.BM):
        assert [m.smem_offset(r, k) for k in range(m.BK)] == table[r], f"row {r}"


def test_smem_offset_out_of_tile_raises():
    for r, k in ((128, 0), (0, 64), (-1, 0)):
        with pytest.raises(ValueError):
            m.smem_offset(r, k)


def test_ldmatrix_rows_hit_eight_different_bank_groups():
    # ldmatrix reads the same 16 B chunk of 8 consecutive rows: with the 128 B swizzle they must
    # land in 8 different 16 B bank groups (conflict free).
    for r0 in range(0, m.BM, 8):
        for chunk in range(8):
            groups = {m.bank_group(r0 + i, 8 * chunk) for i in range(8)}
            assert len(groups) == 8, (r0, chunk)


def test_copy_and_mma_cover_tiles_exactly_once():
    cp = Counter(c for t in range(m.THREADS) for c in m.copy_coords(t))
    assert len(cp) == m.BM * m.BK and set(cp.values()) == {1}
    mm = Counter(c for t in range(m.THREADS) for c in m.mma_c_coords(t))
    assert len(mm) == m.BM * m.BN and set(mm.values()) == {1}


def test_counts():
    assert m.smem_bytes(128, 128, 64, 1) == G["smem_bytes"] == 32768
    assert m.smem_bytes(128, 128, 64, 3) == 98304
    assert m.mma_per_ktile(128, 128, 64) == 8 * 16 * 4


def test_cu_typechecks():
    typecheck("f2_tensorcore_gemm")


@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (4096, 4096, 4096)])
def test_gpu_matches_reference(shape):
    (res,) = run_gpu("f2_tensorcore_gemm", *shape, 10)
    assert res["ok"], res
