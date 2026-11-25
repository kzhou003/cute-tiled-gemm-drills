import json
from dataclasses import replace
from pathlib import Path

import pytest

from drill import load
from tests.golden import golden

m = load("c2_compile_checks")
G = golden("c2_compile_checks")["snippets"]
E = m.RUNNING_EXAMPLE


def _tup(x):
    return tuple(_tup(e) for e in x) if isinstance(x, list) else x


SWEEP = {k: _tup(v) for k, v in json.loads(
    (Path(__file__).parent.parent / "oracle" / "c2_sweep.json").read_text()).items()}
DESCS = {**m.SNIPPETS, **SWEEP}


def codes(cfg):
    return [(v.code, v.layer) for v in m.check_config(cfg)]


@pytest.mark.parametrize("name", sorted(G))
def test_predict_compiles_matches_real_compiler(name):
    ok, msg = m.predict_compiles(DESCS[name])
    assert ok == G[name]["compiles"]
    assert msg == G[name]["message"]


def test_golden_covers_every_snippet():
    assert set(DESCS) == set(G)
    assert len(G) == 15 + 48


def test_composition_rules():
    assert m.composition_check(((4, 8), (8, 1)), ((2, 4), (1, 2))) == []
    assert m.composition_check(((4, 6), (1, 5)), (3, 6)) == ["Stride Divisibility Condition"]
    assert m.composition_check(((4, 6), (1, 5)), (3, 2)) == ["Shape Divisibility Condition"]
    assert m.composition_check(((4, 6), (1, 4)), (3, 6)) == []          # lhs 先 coalesce 成 24:1
    assert m.composition_check(((4, 6), (1, 5)), (3, None)) == []       # 动态值：跳过
    assert m.composition_check(((4, 6), (1, 5)), (3, 0)) == []          # stride 0：广播
    assert m.composition_check(((4, 6), (1, 5)), ((3, 3), (2, 6))) == [
        "Shape Divisibility Condition", "Stride Divisibility Condition"]


def test_composition_messages_are_deduplicated():
    assert m.composition_check(((4, 6), (1, 5)), ((3, 3), (6, 6))) == ["Stride Divisibility Condition"]


def test_swizzle_rules_in_order():
    assert m.swizzle_check(2, 2, 2) == []
    assert m.swizzle_check(3, 3, 2) == ["abs(SShift) must be more than BBits."]
    assert m.swizzle_check(-1, -1, 0) == ["MBase must be positive.", "BBits must be positive."]
    assert m.swizzle_check(2, 1, -3) == []


def test_tiled_copy_rules():
    assert m.tiled_copy_check(64, 16, 1, 128, 4) == []
    assert m.tiled_copy_check(128, 16, 1, 128, 4) == ["TiledCopy uses too few vals for selected CopyAtom"]
    assert m.tiled_copy_check(64, 16, 32, 48, 4) == ["TiledCopy uses too few thrs for selected CopyAtom"]


def test_tile_to_shape_rules():
    assert m.tile_to_shape_check((8, 64), (128, 64)) == []
    assert m.tile_to_shape_check((8, 64), (12, 64)) == ["tile_to_shape: block shape does not divide the target shape."]
    assert m.tile_to_shape_check((8, 64, 2), (12, 64)) == ["Rank of layout must be <= rank of target shape."]
    assert m.tile_to_shape_check((8,), (16, 3)) == []                   # block 补 1
    assert m.tile_to_shape_check((8,), (12, None)) == []                # 有未知值：不检查


def test_running_example_is_clean():
    assert m.check_config(E) == []


@pytest.mark.parametrize("change,want", [
    (dict(copy_atom_bits=128), [("copy_vals", "cute"), ("vector_bytes", "cutlass")]),
    (dict(M=70, N=50, K=40), [("ragged_tile", "runtime")]),
    (dict(swizzle=(3, 3, 2)), [("swizzle_params", "cute")]),
    (dict(threads=256), [("copy_threads", "cutlass"), ("mma_threads", "cutlass")]),
    (dict(use_ldmatrix=True), [("ldmatrix", "cutlass")]),
    (dict(use_ldmatrix=True, swizzle=(1, 3, 3)), []),
    (dict(stages=8, BM=128, BN=128, BK=32), [("smem_capacity", "cutlass"), ("ragged_tile", "runtime")]),
    (dict(K=62), [("alignment", "cutlass"), ("ragged_tile", "runtime")]),
    (dict(swizzle=(2, 1, 2)), [("swizzle_vector", "cutlass")]),
    (dict(BM=48), [("copy_tile", "cutlass"), ("mma_tile", "cutlass"), ("ragged_tile", "runtime")]),
    (dict(swizzle=(3, 3, 3), BM=16), [("copy_tile", "cutlass"), ("mma_tile", "cutlass"), ("swizzle_span", "cutlass")]),
    (dict(swizzle=(3, 3, 3)), []),
], ids=lambda x: str(x) if isinstance(x, dict) else "")
def test_check_config_finds_each_problem(change, want):
    assert codes(replace(E, **change)) == want


def test_violations_have_messages_and_unique_codes():
    vs = m.check_config(replace(E, copy_atom_bits=128, M=70, threads=256))
    assert len({v.code for v in vs}) == len(vs)
    assert all(isinstance(v.message, str) and v.message for v in vs)


def test_smem_capacity_boundary():
    assert m.check_config(replace(E, stages=24)) == []                       # 24 × 2048 B = 48 KB
    assert [(v.code, v.layer) for v in m.check_config(replace(E, stages=25))] == [("smem_capacity", "cutlass")]
