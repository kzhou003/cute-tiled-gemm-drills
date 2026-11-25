import pytest

from drill import load

m = load("e2_config_search")
G = m.H100_SXM
C = m.Config


def test_smem_and_acc_regs():
    c = C(128, 128, 64, 4)
    assert m.smem_bytes(c) == 4 * 256 * 64 * 2 + 8192 + 64 == 139328
    assert m.acc_regs_per_thread(c) == 64
    assert m.acc_regs_per_thread(C(128, 128, 64, 4, "pingpong")) == 128
    assert m.acc_regs_per_thread(C(128, 256, 64, 4)) == 128


@pytest.mark.parametrize("cfg,want", [
    (C(128, 128, 64, 4), []),
    (C(64, 128, 64, 4), ["wgmma_m"]),
    (C(64, 128, 64, 4, "pingpong"), []),
    (C(128, 256, 64, 4, "pingpong"), ["acc_regs"]),
    (C(256, 256, 64, 4), ["smem", "acc_regs"]),
    (C(128, 128, 64, 7), ["smem"]),
    (C(128, 128, 32, 4), ["swizzle"]),
    (C(128, 128, 32, 4, "cooperative", 64), []),
    (C(128, 128, 64, 1), ["stages"]),
    (C(128, 264, 64, 4), ["wgmma_n", "tma_box"]),
], ids=str)
def test_violations(cfg, want):
    assert m.violations(cfg, G) == want


def test_register_rule_uses_gpu_budget():
    small = m.GPU(132, 232448, 32768, 3.35e6, 989e6, 0.6)
    assert m.violations(C(128, 128, 64, 4), small) == ["regs"]


def test_estimate_running_example():
    e = m.estimate(C(128, 128, 64, 4), G, 4096, 4096, 4096)
    assert e["tiles"] == 1024 and e["waves"] == 8
    assert e["wave_eff"] == pytest.approx(1024 / (8 * 132))
    assert e["step"] == pytest.approx(0.2799, abs=1e-4)
    assert e["t_wave"] == pytest.approx(19.9859, abs=1e-3)
    assert e["total"] == pytest.approx(159.887, abs=1e-2)
    assert e["tflops"] == pytest.approx(859.6, abs=0.1)


def test_too_few_stages_expose_latency():
    totals = {s: m.estimate(C(128, 128, 64, s), G, 4096, 4096, 4096)["total"] for s in (2, 3, 4, 6)}
    assert totals[2] == pytest.approx(216.5, abs=0.05)
    assert totals[3] == pytest.approx(totals[4]) == pytest.approx(totals[6])


def test_narrow_swizzle_slows_mma():
    t = {sw: m.estimate(C(128, 128, 64, 4, "cooperative", sw), G, 4096, 4096, 4096)["total"] for sw in (32, 64, 128)}
    assert t[32] > t[64] > t[128]
    assert t[128] == pytest.approx(159.887, abs=1e-2)


def test_pingpong_waves_count_two_tiles_per_sm():
    e = m.estimate(C(128, 128, 64, 4, "pingpong"), G, 4096, 4096, 4096)
    assert e["waves"] == 4
    assert e["wave_eff"] == pytest.approx(1024 / (4 * 132 * 2))


def test_l2_reuse_lowers_load_time():
    big = m.estimate(C(128, 128, 64, 2), G, 4096, 4096, 4096)
    one_row = m.estimate(C(128, 128, 64, 2), G, 128, 4096, 4096)      # 一行 tile：A panel 全部共享
    assert one_row["step"] < big["step"]


def test_default_space_size():
    assert len(m.default_space()) == 3 * 3 * 3 * 6 * 2 * 3


def test_search_ranks_legal_configs():
    r = m.search(m.default_space(), G, 4096, 4096, 4096)
    assert len(r) == 357
    assert all(m.violations(c, G) == [] for c, _ in r)
    totals = [e["total"] for _, e in r]
    assert totals == sorted(totals)
    assert r[0][0] == C(64, 256, 64, 2, "pingpong", 128)
    assert r[0][1]["tflops"] == pytest.approx(906.7, abs=0.1)


def test_search_breaks_ties_by_config_fields():
    r = m.search(m.default_space(), G, 4096, 4096, 4096)
    top = [c for c, e in r if e["total"] == pytest.approx(r[0][1]["total"], abs=1e-9)]
    assert top == sorted(top)
    assert [c.stages for c in top[:4]] == [2, 3, 4, 5]


def test_small_problem_prefers_small_tiles():
    r = m.search(m.default_space(), G, 512, 512, 8192)
    best, e = r[0]
    assert (best.BM, best.BN) == (64, 64) and e["tiles"] == 64 and e["waves"] == 1


def test_multiple_violations_are_all_reported_in_order():
    assert m.violations(C(128, 128, 512, 2), G) == ["tma_box", "smem"]
    assert m.violations(C(256, 128, 64, 4), G) == []
    assert m.violations(C(256, 128, 64, 4, "pingpong"), G) == ["acc_regs"]


def test_pingpong_beats_cooperative_on_the_running_example():
    pp = m.estimate(C(128, 128, 64, 4, "pingpong"), G, 4096, 4096, 4096)["total"]
    assert pp == pytest.approx(151.6, abs=0.05)
    assert pp < m.estimate(C(128, 128, 64, 4), G, 4096, 4096, 4096)["total"]
