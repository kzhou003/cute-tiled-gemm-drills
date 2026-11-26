"""F4 · ragged M / N: predicated loads, masked BF16 epilogue."""
import math

import numpy as np
import pytest

from drill import load
from tests.cu_helpers import run_gpu, typecheck
from tests.golden import golden

m = load("f4_epilogue_residue")
G = golden("f4_epilogue_residue")


def test_grid_rounds_up_and_waste():
    assert m.cta_grid(1000, 760) == (8, 6) == tuple(G["grid"])
    assert m.cta_grid(128, 128) == (1, 1)
    assert m.tile_waste(128, 256) == 0.0
    assert m.tile_waste(1000, 760) == pytest.approx(1 - 760000 / (48 * 128 * 128))
    assert m.tile_waste(129, 1) == pytest.approx(1 - 129 / (2 * 128 * 128))
    with pytest.raises(ValueError):
        m.cta_grid(0, 5)


@pytest.mark.parametrize("p", G["load_predicates"], ids=lambda p: f"bm{p['bm']}_t{p['t']}")
def test_load_predicates_match_identity_partition(p):
    assert m.load_predicates(p["bm"], p["t"], G["M"]) == p["pred"]


def test_store_masks_match_identity_partition_counts():
    for c in G["store_counts"]:
        mask = m.store_mask(c["bm"], c["bn"], G["M"], G["N"])
        assert mask.shape == (128, 128) and mask.dtype == bool
        assert int(mask.sum()) == c["live"]


@pytest.mark.parametrize("M,N", [(1000, 760), (129, 1), (128, 128), (300, 513)])
def test_store_masks_cover_output_exactly_once(M, N):
    gm, gn = m.cta_grid(M, N)
    cover = np.zeros((gm * 128, gn * 128), dtype=int)
    for bm in range(gm):
        for bn in range(gn):
            cover[bm * 128:(bm + 1) * 128, bn * 128:(bn + 1) * 128] += m.store_mask(bm, bn, M, N)
    assert (cover[:M, :N] == 1).all() and cover.sum() == M * N


def test_bf16_round_to_nearest_even():
    one = np.float32(1.0)
    ulp = 2.0 ** -7                                   # BF16 spacing just above 1.0
    xs = np.array([1.0, 1.0 + ulp / 2, 1.0 + 1.5 * ulp, 1.0 + ulp / 4, -2.5, 3.140625], dtype=np.float32)
    got = m.bf16_round(xs)
    assert got.dtype == np.float32
    assert got.tolist() == [1.0, 1.0, 1.0 + 2 * ulp, 1.0, -2.5, 3.140625]   # ties go to even mantissa
    assert (got.view(np.uint32) & 0xFFFF == 0).all()
    assert math.isnan(m.bf16_round(np.array([np.nan], dtype=np.float32))[0])
    assert m.bf16_round(np.array([one], dtype=np.float32))[0] == one


def test_bf16_round_random_values_land_on_bf16_grid():
    x = np.random.default_rng(0).standard_normal(10000).astype(np.float32) * 100
    r = m.bf16_round(x)
    assert (r.view(np.uint32) & 0xFFFF == 0).all()
    assert np.max(np.abs(r - x) / np.abs(x)) <= 2.0 ** -8


def test_epilogue_alpha_beta_and_beta_zero_ignores_c():
    acc = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    c = np.array([4.0, np.nan, 1.0], dtype=np.float32)
    out = m.epilogue(acc, np.array([4.0, 2.0, 1.0], dtype=np.float32), 1.5, -0.5)
    assert out.tolist() == [-0.5, 2.0, -5.0]
    clean = m.epilogue(acc, c, 2.0, 0.0)              # β = 0: C is never read, NaN in C is harmless
    assert clean.tolist() == [2.0, 4.0, -6.0]


def test_max_rel_err_rule():
    assert m.max_rel_err([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert m.max_rel_err([0.5], [0.0]) == pytest.approx(0.5)          # scale is max(1, |ref|)
    assert m.max_rel_err([110.0], [100.0]) == pytest.approx(0.1)
    assert m.max_rel_err([float("nan")], [1.0]) == float("inf")


def test_cu_typechecks():
    typecheck("f4_epilogue_residue")


@pytest.mark.parametrize("shape", [(1000, 760, 1024), (129, 1, 64), (4096, 4096, 4096), (384, 640, 192)])
def test_gpu_matches_reference(shape):
    (res,) = run_gpu("f4_epilogue_residue", *shape, 5, 1.5, -0.5)
    assert res["ok"], res
