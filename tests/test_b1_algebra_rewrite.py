import pytest

from drill import load
from tests.golden import golden

m = load("b1_algebra_rewrite")
G = golden("b1_algebra_rewrite")
L = m.Layout
gA = L((64, 64), (64, 1))
THR, VAL = L((32, 4), (4, 1)), L((1, 4))
VARIANTS = {
    "base": (L((32, 4), (4, 1)), L((1, 4))),
    "colthr": (L((32, 4), (1, 32)), L((1, 4))),
    "t8x16_v4x1": (L((8, 16), (16, 1)), L((4, 1))),
    "t16x8_v2x2": (L((16, 8), (8, 1)), L((2, 2))),
}


# ---- the given core is pinned to real CuTe (these pass with or without your code) ----

def test_given_core_matches_cute_printouts():
    env = {"composition((4,8):(8,1), 8:2)": lambda: m.composition(L((4, 8), (8, 1)), L(8, 2)),
           "complement(4:2, 24)": lambda: m.complement(L(4, 2), 24),
           "coalesce((2,(1,6)):(1,(6,2)))": lambda: m.coalesce(L((2, (1, 6)), (1, (6, 2)))),
           "blocked_product(4:1, (2,3):(1,2))": lambda: m.blocked_product(L(4), L((2, 3))),
           "raked_product(4:1, (2,3):(1,2))": lambda: m.raked_product(L(4), L((2, 3))),
           "logical_divide(24:1, 4:2)": lambda: m.logical_divide(L(24), L(4, 2)),
           "right_inverse((4,8):(8,1))": lambda: m.right_inverse(L((4, 8), (8, 1))),
           "zipped_divide((8,6):(1,8), (4,3))": lambda: m.zipped_divide(L((8, 6)), (4, 3))}
    for c in G["core"]:
        assert repr(env[c["expr"]]()) == c["got"], c["expr"]


def test_given_zipped_divide_matches_cute():
    assert repr(m.zipped_divide(gA, (32, 16))) == G["zipped_divide"]


# ---- local_tile ----

def test_local_tile_view_is_zipped_divide_then_slice():
    tile, off = m.local_tile_view(gA, (32, 16), (1, 2))
    assert repr(tile) == G["local_tile_1_2"]["layout"]
    assert off == G["local_tile_1_2"]["offset"] == 2080


def test_local_tile_offsets_for_every_cta_coord():
    for r in G["rest"]:
        _, off = m.local_tile_view(gA, (32, 16), tuple(r["coord"]))
        assert off == r["offset"]


@pytest.mark.parametrize("bm,k", [(0, 0), (1, 2), (0, 3), (1, 1)])
def test_algebraic_local_tile_equals_hand_formula(bm, k):
    tile, off = m.local_tile_view(gA, (32, 16), (bm, k))
    assert m.tile_map(tile, off) == m.hand_local_tile(bm, k)


def test_hand_local_tile_formula():
    h = m.hand_local_tile(1, 2)
    assert len(h) == 512
    assert h[0] == 32 * 64 + 32 and h[1] == 33 * 64 + 32 and h[32] == 32 * 64 + 33


def test_tile_map_adds_offset():
    assert m.tile_map(L((2, 2), (10, 1)), 100) == [100, 110, 101, 111]


# ---- Copy TV ----

@pytest.mark.parametrize("name", VARIANTS)
def test_make_layout_tv_matches_cute(name):
    want = next(t for t in G["tv"] if t["name"] == name)
    mn, tv, tiler = m.make_layout_tv(*VARIANTS[name])
    assert repr(mn) == want["layout_mn"]
    assert repr(tv) == want["layout_tv"]
    assert tuple(tiler) == (32, 16) and want["tiler"] == "(_32,_16)"
    assert [tv(i) for i in range(tv.size())] == want["tv_values"]


def test_algebraic_copy_tv_equals_hand_formula():
    mn, tv, tiler = m.make_layout_tv(THR, VAL)
    dom = [(t, v) for t in range(128) for v in range(4)]
    assert m.same_function(lambda x: m.tv_coord(tv, tiler, *x), lambda x: m.hand_copy_coord(*x), dom)
    assert m.tv_coord(tv, tiler, 5, 0) == (1, 4)


def test_owner_inverts_tv():
    mn, tv, tiler = m.make_layout_tv(THR, VAL)
    for t in range(0, 128, 7):
        for v in range(4):
            assert m.owner(mn, *m.tv_coord(tv, tiler, t, v)) == (t, v)
    assert m.owner(mn, 1, 6) == (5, 2)


def test_hand_copy_coord():
    assert m.hand_copy_coord(5, 3) == (1, 7)
    assert m.hand_copy_coord(127, 0) == (31, 12)


# ---- SMEM ----

def test_smem_layout_map_matches_cute_composition():
    base = L((32, 16), (16, 1))
    assert m.smem_layout_map((2, 2, 2), base) == G["smem"]


def test_algebraic_smem_equals_hand_formula():
    base = L((32, 16), (16, 1))
    got = m.smem_layout_map((2, 2, 2), base)
    assert got == [m.hand_smem_offset(i % 32, i // 32) for i in range(512)]
    assert m.hand_smem_offset(1, 0) == 20


def test_same_function():
    assert m.same_function(lambda x: x * 2, lambda x: x + x, range(10))
    assert not m.same_function(lambda x: x, lambda x: -x, range(1, 3))


def test_local_tile_views_tile_A_exactly_once():
    seen = []
    for bm in range(2):
        for k in range(4):
            tile, off = m.local_tile_view(gA, (32, 16), (bm, k))
            seen += m.tile_map(tile, off)
    assert sorted(seen) == list(range(64 * 64))


def test_layout_tv_is_right_inverse_of_layout_mn():
    mn, tv, tiler = m.make_layout_tv(THR, VAL)
    for i in range(512):
        t, v = i % 128, i // 128
        off = tv((t, v))
        assert mn((off % 32, off // 32)) == t + 128 * v
    assert mn((31, 15)) == 511
    assert tv((5, 2)) == 193
