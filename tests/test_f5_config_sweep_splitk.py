"""F5 · config sweep and split-K: the cost model on the host, then every config + cuBLAS on the GPU."""
import pytest

from drill import load
from tests.cu_helpers import run_gpu, typecheck

m = load("f5_config_sweep_splitk")
C = {(c.name, c.splits): c for c in m.CONFIGS}


def test_smem_and_legality():
    assert m.smem_bytes(C["128x128x64_s3_w2x4", 1]) == 98304
    assert m.legal(C["128x128x64_s3_w2x4", 1], 4096, 4096, 4096) == (True, "")
    assert m.legal(C["128x128x64_s3_w2x4_splitk", 5], 1024, 1024, 16384) == (False, "divisibility")
    assert m.legal(C["128x256x64_s2_w2x4", 1], 1000, 1024, 1024) == (False, "divisibility")
    big = m.Cfg("256x256x64_s3", 256, 256, 64, 3, 2, 4)
    assert m.legal(big, 4096, 4096, 4096) == (False, "smem")      # 196608 B > 99 KB


def test_prediction_by_hand_4096_cube():
    # 1024 CTAs -> ceil(1024/170) = 7 rounds; 64 k-tiles x (4096 + 150) clk + 1500 clk fill
    want = 7 * (64 * (2 * 128 * 128 * 64 / 512 + 150) + 1500) / 2410
    assert m.predict_us(C["128x128x64_s3_w2x4", 1], 4096, 4096, 4096) == pytest.approx(want)


def test_prediction_by_hand_split_k():
    # 64 x 2 = 128 CTAs (one round), 128 k-tiles each; plus (1 + 2) x M x N x 4 B of zeroing + atomics
    main = (128 * (4096 + 150) + 1500) / 2410
    split = 3 * 1024 * 1024 * 4 / 1792e3
    got = m.predict_us(C["128x128x64_s3_w2x4_splitk", 2], 1024, 1024, 16384)
    assert got == pytest.approx(main + split)


def test_predict_illegal_raises():
    with pytest.raises(ValueError):
        m.predict_us(C["128x128x64_s3_w2x4_splitk", 5], 1024, 1024, 16384)


def test_rank_contains_only_legal_sorted():
    r = m.rank(1024, 1024, 16384)
    assert [x[2] for x in r] == sorted(x[2] for x in r)
    assert ("128x128x64_s3_w2x4_splitk", 5) not in [(n, s) for n, s, _ in r]
    assert len(r) == len(m.CONFIGS) - 1


def test_big_square_problem_does_not_split():
    assert m.best(8192, 8192, 8192).splits == 1


def test_few_tiles_long_k_prefers_split_and_wins_big():
    r = m.rank(1024, 1024, 16384)
    plain = next(u for n, s, u in r if (n, s) == ("128x128x64_s3_w2x4", 1))
    assert m.best(1024, 1024, 16384).splits > 1
    assert plain / r[0][2] > 1.4


def _measured(order):
    out = []
    for i, (name, splits, _) in enumerate(order):
        out.append({"kernel": name, "splits": splits, "M": 1024, "N": 1024, "K": 16384, "ms": 0.1 * (i + 1)})
    return out


def test_compare_rank_correlation():
    r = m.rank(1024, 1024, 16384)
    same = m.compare(_measured(r))
    assert same["spearman"] == pytest.approx(1.0) and len(same["pairs"]) == len(r)
    rev = m.compare(_measured(r[::-1]))
    assert rev["spearman"] < -0.9          # ties in the prediction keep it from being exactly -1
    extra = _measured(r[:1]) + [{"kernel": "cublas_bf16_fp32", "baseline": True, "ms": 0.01},
                                {"kernel": "x", "skipped": True}]
    assert m.compare(extra)["spearman"] is None


def test_cu_typechecks():
    typecheck("f5_config_sweep_splitk")


@pytest.mark.parametrize("shape", [(4096, 4096, 4096), (1024, 1024, 16384), (2048, 512, 8192)])
def test_gpu_all_configs_correct_and_compare(shape):
    lines = run_gpu("f5_config_sweep_splitk", *shape, 10)
    kernels = [l for l in lines if not l.get("skipped")]
    assert kernels and all(l["ok"] for l in kernels), [l for l in kernels if not l["ok"]]
    res = m.compare(lines)
    print(f"\n{shape}: spearman(pred, measured) = {res['spearman']}")
    for name, splits, pred, meas in sorted(res["pairs"], key=lambda p: p[3]):
        print(f"  {name:28s} S={splits}  predicted {pred:8.1f} us   measured {meas:8.1f} us")
    base = [l for l in lines if l.get("baseline")]
    if base:
        best = min(l["ms"] for l in kernels if not l.get("baseline"))
        print(f"  best / cuBLAS = {base[0]['ms'] / best:.2f}")
