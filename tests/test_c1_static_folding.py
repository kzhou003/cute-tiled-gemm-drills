import random

import pytest

from drill import load
from tests.golden import golden

m = load("c1_static_folding")
G = golden("c1_static_folding")
ASM = golden("c1_asm")
C, V = m.C, m.V


def test_layout_expr_structure():
    e = m.layout_expr((2, "d"), (V("a"), V("b")))
    assert e == ("+", ("*", V("a"), C(2)), ("*", V("b"), V("d")))
    assert m.layout_expr((3,), (V("x"),)) == ("*", V("x"), C(3))
    with pytest.raises(ValueError):
        m.layout_expr((1, 2), (V("x"),))


def test_idx2crd_colex_last_mode_not_wrapped():
    i = V("i")
    assert m.idx2crd_expr(i, (4, 8, 2)) == (("%", i, C(4)), ("%", ("//", i, C(4)), C(8)), ("//", i, C(32)))
    assert m.idx2crd_expr(i, (5,)) == (i,)
    assert m.idx2crd_expr(i, ("a", 3)) == (("%", i, V("a")), ("//", i, V("a")))
    vals = [m.evaluate(x, {"i": 70}) for x in m.idx2crd_expr(i, (4, 8, 2))]
    assert vals == [2, 1, 2]


def _env(t, v, k, dyn=False):
    e = {"tid": t, "v": v, "k": k}
    if dyn:
        e.update(bm=32, bk=16, ld=64)
    return e


@pytest.mark.parametrize("static", [True, False])
def test_copy_offsets_match_real_cute(static):
    e = m.copy_offset_expr(static)
    table = G["static_offsets"] if static else G["dynamic_offsets"]
    for k in range(4):
        for t in range(128):
            got = [m.evaluate(e, _env(t, v, k, not static)) for v in range(4)]
            assert got == table[k][t], (k, t)


@pytest.mark.parametrize("static", [True, False])
def test_simplified_offsets_still_match_real_cute(static):
    e = m.simplify(m.copy_offset_expr(static))
    table = G["static_offsets"] if static else G["dynamic_offsets"]
    for k in range(4):
        for t in range(0, 128, 3):
            assert [m.evaluate(e, _env(t, v, k, not static)) for v in range(4)] == table[k][t]


def test_static_offsets_lose_every_mul_div_mod():
    raw = m.copy_offset_expr(True)
    assert m.count_ops(raw) == {"+": 5, "*": 7, "%": 3, "//": 3}
    assert m.count_ops(m.fold(raw)) == {"+": 4, "*": 4, "%": 3, "//": 2}
    assert m.count_ops(m.simplify(raw)) == {"+": 4, "<<": 4, "&": 3, ">>": 2}


def test_dynamic_offsets_keep_mul_div_mod():
    assert m.count_ops(m.simplify(m.copy_offset_expr(False))) == {
        "+": 4, "*": 4, "%": 1, "&": 2, "<<": 1, ">>": 1, "//": 1}


def test_fold_identities_and_constant_chains():
    x = V("x")
    assert m.fold(("*", ("*", ("*", x, C(2)), C(4)), C(1))) == ("*", x, C(8))
    assert m.fold(("+", ("+", x, C(3)), C(4))) == ("+", x, C(7))
    assert m.fold(("//", ("//", x, C(4)), C(8))) == ("//", x, C(32))
    assert m.fold(("*", C(3), x)) == ("*", x, C(3))
    assert m.fold(("+", C(0), x)) == x
    assert m.fold(("*", x, C(0))) == C(0)
    assert m.fold(("%", x, C(1))) == C(0)
    assert m.fold(("//", C(0), x)) == C(0)
    assert m.fold(("%", C(0), x)) == C(0)
    assert m.fold(("//", x, C(1))) == x
    assert m.fold(("+", ("*", C(2), C(3)), C(4))) == C(10)


def test_strength_reduce_only_powers_of_two():
    x = V("x")
    assert m.strength_reduce(("*", x, C(8))) == ("<<", x, C(3))
    assert m.strength_reduce(("//", x, C(4))) == (">>", x, C(2))
    assert m.strength_reduce(("%", x, C(32))) == ("&", x, C(31))
    assert m.strength_reduce(("*", x, C(6))) == ("*", x, C(6))
    assert m.strength_reduce(("%", x, C(1))) == ("%", x, C(1))
    assert m.strength_reduce(("*", x, V("n"))) == ("*", x, V("n"))


def test_count_ops_dedups_structurally_equal_subtrees():
    x = V("x")
    sub = ("*", x, C(3))
    assert m.count_ops(("+", sub, sub)) == {"+": 1, "*": 1}
    assert m.count_ops(("+", ("*", C(2), C(3)), x)) == {"+": 1}
    assert m.count_ops(C(5)) == {}


def test_coalesce_flat():
    assert m.coalesce_flat((4, 32, 4, 1), (128, 1, 32, 0)) == ((4, 128), (128, 1))
    assert m.coalesce_flat((2, 4), (1, 2)) == ((8,), (1,))
    assert m.coalesce_flat((1, 1), (3, 4)) == ((1,), (0,))
    assert m.coalesce_flat((32, 16), (1, 32)) == ((512,), (1,))
    assert m.coalesce_flat((4, 8), (8, 1)) == ((4, 8), (8, 1))
    assert m.coalesce_flat((1, 4, 1, 2), (7, 2, 9, 8)) == ((8,), (2,))


def test_coalesce_removes_identity_that_folding_cannot():
    i = V("i")
    plain = m.simplify(m.layout_index_expr(i, (32, 16), (1, 32)))
    assert plain != i                                   # 表达式化简看不出这是恒等函数
    assert m.simplify(m.layout_index_expr(i, (32, 16), (1, 32), coalesce=True)) == i


def test_hoist_k_static_and_dynamic():
    base, step = m.hoist(m.copy_offset_expr(True), "k")
    assert step == C(16)
    assert "k" not in m.pretty(base)
    for t, v in [(0, 0), (5, 3), (127, 1)]:
        b = m.evaluate(base, {"tid": t, "v": v})
        assert [b + 16 * k for k in range(4)] == [G["static_offsets"][k][t][v] for k in range(4)]
    assert m.count_ops(base) == {"+": 3, "<<": 3, "&": 3, ">>": 2}
    _, dstep = m.hoist(m.copy_offset_expr(False), "k")
    assert dstep == V("bk")


def test_hoist_rejects_nonlinear():
    x, k = V("x"), V("k")
    with pytest.raises(ValueError):
        m.hoist(("*", k, k), "k")
    with pytest.raises(ValueError):
        m.hoist(("+", x, ("//", k, C(3))), "k")
    assert m.hoist(("+", x, ("*", k, C(4))), "k") == (x, C(4))
    assert m.hoist(x, "k") == (x, C(0))


def _rand_expr(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return C(rng.choice([0, 1, 2, 3, 4, 6, 8, 16, 32])) if rng.random() < 0.5 else V(rng.choice("abc"))
    op = rng.choice(["+", "*", "//", "%"])
    a = _rand_expr(rng, depth - 1)
    b = C(rng.choice([1, 2, 3, 4, 8, 16])) if op in ("//", "%") else _rand_expr(rng, depth - 1)
    return (op, a, b)


def test_simplify_preserves_value_on_random_expressions():
    rng = random.Random(7)
    for _ in range(300):
        e = _rand_expr(rng, 4)
        s = m.simplify(e)
        for _ in range(3):
            env = {n: rng.randint(0, 200) for n in "abc"}
            assert m.evaluate(s, env) == m.evaluate(e, env)


def test_golden_real_compiler_folds_static_layouts():
    f = ASM["functions"]
    assert f["offset_static"]["mul_div"] == 0
    assert f["offset_dynamic"]["mul_div"] > 0
