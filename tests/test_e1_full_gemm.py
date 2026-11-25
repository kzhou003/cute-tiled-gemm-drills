import numpy as np
import pytest

from drill import load
from tests.golden import golden

m = load("e1_full_gemm")


def _rand(*shape, seed=0):
    return np.random.default_rng(seed + sum(shape)).standard_normal(shape)


def test_given_sw128_is_the_gmma_128b_swizzle_atom():
    t = next(x for x in golden("d1_tma_box")["tiles"] if x["mode"] == "128B" and x["inner"] == 64)
    assert [m.sw128(r * 64 + k) for k in range(64) for r in range(64)] == t["offsets"]


def test_pipeline_state_advance():
    st = m.PipelineState(3)
    seq = []
    for _ in range(7):
        seq.append((st.index, st.phase, st.count))
        st.advance()
    assert seq == [(0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3), (1, 1, 4), (2, 1, 5), (0, 0, 6)]
    one = m.PipelineState(1)
    one.advance()
    assert (one.index, one.phase, one.count) == (0, 1, 1)


def test_pipeline_state_matches_real_cutlass():
    g = golden("d2_mbarrier_pipeline")
    st, seq = m.PipelineState.producer_start(4), []
    for _ in range(10):
        seq.append([st.index, st.phase, st.count])
        st.advance()
    assert seq == g["producer_4"]


def test_pipeline_start_states():
    p, c = m.PipelineState.producer_start(4), m.PipelineState.consumer_start(4)
    assert (p.stages, p.index, p.phase, p.count) == (4, 0, 1, 0)
    assert (c.stages, c.index, c.phase, c.count) == (4, 0, 0, 0)


def test_tma_box_load_swizzles_and_zero_fills():
    src = np.arange(100 * 70, dtype=float).reshape(100, 70) + 1
    smem = np.full(2 * 64 * 64, -1.0)
    nbytes = m.tma_box_load(src, 64, 32, 64, 64, smem, 4096)
    assert nbytes == 64 * 64 * m.ELEM_BYTES
    assert (smem[:4096] == -1).all()                        # 只写自己的 stage
    tile = m.smem_read(smem, 4096, 64, 64)
    want = np.zeros((64, 64))
    want[:36, :38] = src[64:100, 32:70]
    assert (tile == want).all()
    assert smem[4096 + m.sw128(1 * 64 + 0)] == src[65, 32]


@pytest.mark.parametrize("MNK,stages", [((128, 128, 512), 3), ((256, 128, 256), 4), ((200, 150, 100), 2),
                                         ((64, 64, 64), 1), ((130, 260, 70), 4)])
def test_run_gemm_is_correct(MNK, stages):
    M, N, K = MNK
    A, B = _rand(M, K), _rand(N, K, seed=1)
    C, rep = m.run_gemm(A, B, m.Cfg(M, N, K, stages=stages))
    assert C.shape == (M, N)
    assert np.allclose(C, A @ B.T)


def test_report_running_example():
    A, B = _rand(128, 512), _rand(128, 512, seed=1)
    _, r = m.run_gemm(A, B, m.Cfg(128, 128, 512, stages=3))
    assert r["end"] == pytest.approx(10.6)
    assert r["trigger"] == pytest.approx(9.6)
    assert r["tma_busy"] == pytest.approx(8.0)
    assert r["mma_busy"] == pytest.approx(4.8)
    assert r["overlap"] == pytest.approx(4.1)


def test_one_stage_cannot_overlap_and_more_stages_help():
    A, B = _rand(128, 512), _rand(128, 512, seed=1)
    ends = {}
    for s in (1, 2, 3, 4):
        _, r = m.run_gemm(A, B, m.Cfg(128, 128, 512, stages=s))
        ends[s] = r["end"]
        if s == 1:
            assert r["overlap"] == pytest.approx(0.0)
        else:
            assert r["overlap"] > 3
    assert ends[1] == pytest.approx(18.3)
    assert ends[1] > ends[2] > ends[3] == pytest.approx(ends[4])


def test_start_time_shifts_everything():
    A, B = _rand(128, 128), _rand(128, 128, seed=1)
    _, r0 = m.run_gemm(A, B, m.Cfg(128, 128, 128))
    _, r1 = m.run_gemm(A, B, m.Cfg(128, 128, 128), start=5.0, grid_ready=5.0)
    assert r1["end"] == pytest.approx(r0["end"] + 5.0)


def test_pdl_wait_blocks_only_the_dependent_operand():
    A, B = _rand(128, 256), _rand(128, 256, seed=1)
    C, r = m.run_gemm(A, B, m.Cfg(128, 128, 256), start=0.0, grid_ready=3.0, pdl_wait=True)
    assert np.allclose(C, A @ B.T)
    _, r_late = m.run_gemm(A, B, m.Cfg(128, 128, 256), start=3.0, grid_ready=3.0)
    assert r["end"] < r_late["end"]                       # 预取 B 的收益


@pytest.mark.parametrize("pdl,spans", [(False, [(0, 6.6), (9.6, 16.2)]), (True, [(0, 6.6), (6.1, 13.0)])])
def test_chain_with_and_without_pdl(pdl, spans):
    X, W1, W2 = _rand(128, 256), _rand(256, 256, seed=1), _rand(128, 256, seed=2)
    Y, sp = m.chain(X, W1, W2, m.Cfg(128, 256, 256), m.Cfg(128, 128, 256), pdl)
    assert np.allclose(Y, (X @ W1.T) @ W2.T)
    assert np.allclose(np.array(sp, dtype=float), np.array(spans, dtype=float))
    if pdl:
        assert sp[1][0] < sp[0][1]                         # 后继在前驱结束前启动


def _ctx(cfg, pdl=False):
    A, B = _rand(cfg.M, cfg.K), _rand(cfg.N, cfg.K, seed=1)
    C = np.zeros((cfg.M, cfg.N))
    return m.Ctx(cfg, A, B, C, 0, 0, np.zeros(cfg.stages * cfg.BM * cfg.BK), np.zeros(cfg.stages * cfg.BN * cfg.BK),
                 [m.Barrier(1) for _ in range(cfg.stages)], [m.Barrier(m.NUM_CONSUMERS) for _ in range(cfg.stages)],
                 [np.zeros((cfg.BM // m.NUM_CONSUMERS, cfg.BN)) for _ in range(m.NUM_CONSUMERS)], pdl_wait=pdl)


def _ops(gen, ctx):
    out = []
    for op in gen:
        kind = op[0]
        if kind == "wait":
            bar = op[1]
            which = "empty" if any(bar is b for b in ctx.empty) else "full"
            idx = next(i for i, b in enumerate(ctx.empty if which == "empty" else ctx.full) if b is bar)
            out.append(("wait", which, idx, op[2]))
        elif kind == "arrive_expect_tx":
            out.append(("expect", next(i for i, b in enumerate(ctx.full) if b is op[1]), op[2]))
        elif kind == "tma":
            out.append(("tma", op[2], op[4]))
        elif kind == "arrive":
            out.append(("arrive", next(i for i, b in enumerate(ctx.empty) if b is op[1])))
        else:
            out.append((kind,))
    return out


def test_producer_op_sequence():
    cfg = m.Cfg(128, 64, 256, BN=64, stages=3)          # A box 16 KB, B box 8 KB：好区分
    ctx = _ctx(cfg)
    ops = _ops(m.producer(ctx), ctx)
    assert ops[0] == ("compute",)
    body = ops[1:]
    A, B = 128 * 64 * 2, 64 * 64 * 2
    for k in range(4):
        s, phase = k % 3, (1 if k < 3 else 0)
        assert body[4 * k:4 * k + 4] == [("wait", "empty", s, phase), ("expect", s, A + B), ("tma", A, s), ("tma", B, s)]


def test_producer_pdl_prefetches_weights_before_grid_wait():
    cfg = m.Cfg(128, 64, 256, BN=64, stages=3)
    ctx = _ctx(cfg, pdl=True)
    ops = _ops(m.producer(ctx), ctx)
    gw = ops.index(("grid_wait",))
    before = [o for o in ops[:gw] if o[0] == "tma"]
    assert before == [("tma", 64 * 64 * 2, s) for s in range(3)]          # 只有 B（权重）
    assert [o[2] for o in ops[:gw] if o[0] == "expect"] == [128 * 64 * 2 + 64 * 64 * 2] * 3


def test_consumer_op_sequence_and_single_trigger():
    cfg = m.Cfg(128, 64, 128, BN=64, stages=2)
    for wg in range(m.NUM_CONSUMERS):
        ctx = _ctx(cfg)
        ops = _ops(m.consumer(ctx, wg), ctx)
        kinds = [o[0] for o in ops]
        assert kinds[0] == "compute"
        assert ops[1] == ("wait", "full", 0, 0) and ops[3] == ("arrive", 0)
        assert ops[4] == ("wait", "full", 1, 0) and ops[6] == ("arrive", 1)
        assert kinds.count("trigger") == (1 if wg == 0 else 0)
        assert kinds[-1] == "compute"                                       # epilogue 最后
        if wg == 0:
            assert kinds.index("trigger") == len(kinds) - 2
