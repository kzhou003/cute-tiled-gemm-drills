import pytest

from drill import load
from tests.golden import golden

m = load("b2_vector_width")
G = golden("b2_vector_width")
L, Sw = m.Layout, m.Swizzled

g = L((32, 16), (64, 1))
s = L((32, 16), (16, 1))
LAYOUTS = {
    "grow": g, "gk60": L((32, 16), (60, 1)), "gcol": L((32, 16), (1, 64)),
    "srow": s, "ssw111": Sw(1, 1, 1, s), "ssw222": Sw(2, 2, 2, s), "ssw333": Sw(3, 3, 3, s),
    "spad18": L((32, 16), (18, 1)), "smmajor": L((32, 16), (1, 32)),
}
PAIRS = {
    "grow_splain": ("grow", "srow"), "grow_ssw111": ("grow", "ssw111"), "grow_ssw222": ("grow", "ssw222"),
    "grow_ssw333": ("grow", "ssw333"), "grow_spad18": ("grow", "spad18"), "grow_smmajor": ("grow", "smmajor"),
    "gk60_ssw333": ("gk60", "ssw333"), "gcol_smmajor": ("gcol", "smmajor"),
    "ssw333_ssw333": ("ssw333", "ssw333"), "ssw222_ssw333": ("ssw222", "ssw333"),
}
GP = {p["name"]: p for p in G["pairs"]}


def _pair(name):
    a, b = PAIRS[name]
    return LAYOUTS[a], LAYOUTS[b]


def test_given_layout_reprs_match_cute():
    for name, (a, b) in PAIRS.items():
        assert repr(LAYOUTS[a]) == GP[name]["a"] and repr(LAYOUTS[b]) == GP[name]["b"]


@pytest.mark.parametrize("name", PAIRS)
def test_max_common_vector_matches_real_cute(name):
    assert m.max_common_vector(*_pair(name)) == GP[name]["mcv"]


@pytest.mark.parametrize("name", [n for n in PAIRS if GP[n]["mcl"] is not None])
def test_max_common_layout_matches_real_cute(name):
    assert repr(m.max_common_layout(*_pair(name))) == GP[name]["mcl"]


@pytest.mark.parametrize("sw,cap", [((1, 1, 1), 2), ((2, 2, 2), 4), ((3, 3, 3), 8)])
def test_swizzled_argument_order_does_not_matter(sw, cap):
    x = Sw(*sw, s)
    assert m.max_common_vector(g, x) == m.max_common_vector(x, g) == cap


def test_two_swizzles():
    assert m.max_common_vector(Sw(3, 3, 3, s), Sw(3, 3, 3, s)) == 512
    assert m.max_common_vector(Sw(2, 2, 2, s), Sw(3, 3, 3, s)) == 4


@pytest.mark.parametrize("name", ["grow_splain", "grow_ssw222", "grow_ssw333", "gcol_smmajor"])
def test_common_layout_elements_are_contiguous_on_both_sides(name):
    a, b = _pair(name)
    cl = m.max_common_layout(a, b)
    idx = [cl(i) for i in range(cl.size())]
    for side in (a, b):
        vals = [side(i) for i in idx]
        assert vals == list(range(vals[0], vals[0] + len(vals)))


@pytest.mark.parametrize("name", ["grow_splain", "grow_ssw111", "grow_ssw222", "grow_ssw333", "grow_smmajor",
                                  "gcol_smmajor", "ssw333_ssw333", "ssw222_ssw333"])
def test_brute_force_agrees_with_algebra(name):
    a, b = _pair(name)
    assert m.brute_vector(a, b, 512) == m.max_common_vector(a, b)


def test_padding_contiguous_but_not_aligned():
    pad = LAYOUTS["spad18"]
    assert m.max_common_vector(g, pad) == 16
    assert m.brute_vector(g, pad, 512) == 2
    assert m.max_alignment(pad) == 2
    assert m.auto_vector_bits(g, pad) == 32


@pytest.mark.parametrize("name", [a["name"] for a in G["alignment"]])
def test_max_alignment_matches_real_cute(name):
    want = next(a for a in G["alignment"] if a["name"] == name)
    assert repr(LAYOUTS[name]) == want["layout"]
    assert m.max_alignment(LAYOUTS[name]) == want["max_alignment"]


@pytest.mark.parametrize("name", PAIRS)
def test_auto_vector_bits_matches_copy_hpp_formula(name):
    assert m.auto_vector_bits(*_pair(name)) == GP[name]["auto_vector_bits"]


def _brute_aligned_bits(a, b, cap_bits=128):
    # 带对齐要求的暴力定义：两边对齐的 n 元素块（n·16 位 ≤ cap），并且两边各自按 n 对齐。
    n = m.brute_vector(a, b, 512)
    n = min(n, cap_bits // 16)
    while n > 1 and (m.max_alignment(a) % n or m.max_alignment(b) % n):
        n //= 2
    return n * 16


@pytest.mark.parametrize("name", [n for n in PAIRS if n not in ("ssw333_ssw333",)])
def test_aligned_brute_force_equals_auto_vectorization(name):
    a, b = _pair(name)
    assert _brute_aligned_bits(a, b) == m.auto_vector_bits(a, b)


def test_vector_and_alignment_bytes():
    assert (m.vector_bytes(16), m.vector_bytes(3), m.vector_bytes(4), m.vector_bytes(2, elem_bytes=4)) == (16, 2, 8, 8)
    assert m.alignment_bytes([0, 8, 24]) == 8
    assert m.alignment_bytes([0, 0]) == 16
    assert m.alignment_bytes([4, 16]) == 4
    assert m.alignment_bytes([6]) == 2


def test_copy_atom_names():
    assert m.copy_atom(8) == "UniversalCopy<uint64_t>"
    assert m.copy_atom(2) == "UniversalCopy<uint16_t>"
    assert m.copy_atom(16, async_=True) == "SM80_CP_ASYNC_CACHEALWAYS<uint128_t>"
    assert m.copy_atom(4, async_=True) == "SM80_CP_ASYNC_CACHEALWAYS<uint32_t>"
    for bad, a in [(3, False), (32, False), (2, True), (12, True)]:
        with pytest.raises(ValueError):
            m.copy_atom(bad, a)


def test_copy_plans_for_the_running_example():
    assert m.copy_plan(g, Sw(2, 2, 2, s), 4) == {"mcv": 4, "vec": 4, "bytes": 8, "atom": "UniversalCopy<uint64_t>"}
    assert m.copy_plan(g, Sw(3, 3, 3, s), 8, async_=True) == {
        "mcv": 8, "vec": 8, "bytes": 16, "atom": "SM80_CP_ASYNC_CACHEALWAYS<uint128_t>"}
    pad = m.copy_plan(g, LAYOUTS["spad18"], 8)
    assert (pad["mcv"], pad["bytes"], pad["atom"]) == (16, 4, "UniversalCopy<uint32_t>")
    off = m.copy_plan(g, Sw(3, 3, 3, s), 8, base_offsets_bytes=(4,), async_=True)
    assert (off["bytes"], off["atom"]) == (4, "SM80_CP_ASYNC_CACHEALWAYS<uint32_t>")
    k60 = m.copy_plan(LAYOUTS["gk60"], Sw(3, 3, 3, s), 8, async_=True)
    assert (k60["mcv"], k60["bytes"], k60["atom"]) == (8, 8, "SM80_CP_ASYNC_CACHEALWAYS<uint64_t>")
    assert m.copy_plan(g, LAYOUTS["smmajor"], 8)["bytes"] == 2
