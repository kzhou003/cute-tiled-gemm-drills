"""F3 · multi-stage cp.async pipeline + ldmatrix: the main loop's event trace and its invariants."""
import os

import pytest

from drill import load
from tests.cu_helpers import run_gpu, typecheck

m = load("f3_pipelined_gemm")

# sgemm_sm80.cu's prologue + main loop with K_PIPE_MAX = 3 and 5 k-tiles, event by event
TRACE_5_3 = [
    ("load", 0, 0), ("load", 1, 1), ("wait", 1),
    ("load", 2, 2), ("compute", 0, 0), ("wait", 1),
    ("load", 3, 0), ("compute", 1, 1), ("wait", 1),
    ("load", 4, 1), ("compute", 2, 2), ("wait", 1),
    ("load", 4, 2), ("compute", 3, 0), ("wait", 1),
    ("load", 4, 0), ("compute", 4, 1), ("wait", 1),
]


def test_trace_matches_sgemm_sm80_loop():
    assert m.trace(5, 3) == TRACE_5_3


@pytest.mark.parametrize("stages", [2, 3, 4, 5])
@pytest.mark.parametrize("k_tiles", [1, 2, 3, 5, 8, 13])
def test_invariants_for_many_shapes(k_tiles, stages):
    ev = m.trace(k_tiles, stages)
    assert m.check_trace(ev, stages) is True
    computes = [e for e in ev if e[0] == "compute"]
    assert [e[1] for e in computes] == list(range(k_tiles))
    assert all(e[2] == e[1] % stages for e in computes)
    assert m.redundant_loads(k_tiles, stages) == stages - 1
    assert all(e[1] == stages - 2 for e in ev if e[0] == "wait")


def test_trace_rejects_bad_arguments():
    with pytest.raises(ValueError):
        m.trace(0, 3)
    with pytest.raises(ValueError):
        m.trace(4, 1)


def test_check_trace_catches_compute_before_wait():
    ev = list(TRACE_5_3)
    del ev[2]                       # drop the prologue wait: k-tile 0 is used before it has landed
    with pytest.raises(AssertionError, match="rule 2"):
        m.check_trace(ev, 3)


def test_check_trace_catches_out_of_order_compute():
    ev = list(TRACE_5_3)
    ev[4], ev[7] = ("compute", 1, 1), ("compute", 0, 0)
    with pytest.raises(AssertionError, match="rule 1"):
        m.check_trace(ev, 3)


def test_check_trace_catches_overwriting_unused_data():
    ev = list(TRACE_5_3)
    ev[3] = ("load", 2, 1)          # slot 1 still holds k-tile 1, not computed yet
    with pytest.raises(AssertionError, match="rule 3"):
        m.check_trace(ev, 3)


def test_stage_formulas():
    assert m.smem_bytes(128, 128, 64, 3) == 98304
    assert m.max_stages(128, 128, 64) == 3                     # 99 KB per CTA on the RTX 5090
    assert m.max_stages(128, 128, 64, limit=227 * 1024) == 7   # H100
    assert m.max_stages(64, 64, 64) == 6
    with pytest.raises(ValueError):
        m.max_stages(512, 512, 128)
    assert m.wait_arg(3) == 1 and m.wait_arg(2) == 0
    assert m.inflight_bytes(128, 128, 64, 3) == 65536


def test_cu_typechecks():
    typecheck("f3_pipelined_gemm")


@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (2048, 1024, 3072), (4096, 4096, 4096)])
def test_gpu_matches_reference(shape):
    (res,) = run_gpu("f3_pipelined_gemm", *shape, 10)
    assert res["ok"], res
    floor = float(os.environ.get("DRILL_F3_MIN_TFLOPS", "0"))
    if floor and shape == (4096, 4096, 4096):
        assert res["tflops"] >= floor, f"{res['tflops']} TFLOPS < DRILL_F3_MIN_TFLOPS={floor}"
