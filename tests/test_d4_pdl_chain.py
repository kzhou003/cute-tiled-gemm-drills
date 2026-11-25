from dataclasses import replace

import pytest

from drill import load

m = load("d4_pdl_chain")
f = frozenset


def test_kernel_and_trigger_time():
    p, c = m.two_gemm_chain()
    assert m.kernel_time(p) == 26 and m.kernel_time(c) == 25
    assert m.trigger_time(p) == 22                    # mainloop 结束后 trigger
    assert m.trigger_time(replace(p, trigger_after=None)) == 26
    assert m.trigger_time(replace(p, trigger_after=0)) == 2


@pytest.mark.parametrize("mode,spans,total", [
    ("stream", [(0, 26), (29, 54)], 54),
    ("graph", [(0, 26), (27, 52)], 52),
    ("pdl", [(0, 26), (22.5, 47.5)], 47.5),
    ("graph_pdl", [(0, 26), (22.5, 47.5)], 47.5),
])
def test_timeline_modes(mode, spans, total):
    t = m.timeline(m.two_gemm_chain(), mode)
    assert t["spans"] == spans and t["total"] == total


def test_pdl_phases_wait_for_the_full_producer_not_its_trigger():
    ph = m.timeline(m.two_gemm_chain(), "pdl")["phases"]
    assert ph[0] == [("setup", 0, 2), ("mainloop", 2, 22), ("epilogue", 22, 26)]
    # 后继在 22.5 启动；mainloop 前 wait，要等前驱结束 26 + 刷出 0.5
    assert ph[1] == [("setup", 22.5, 24.5), ("prefetch_w2", 24.5, 29.5), ("mainloop", 29.5, 44.5), ("epilogue", 44.5, 47.5)]
    early = m.timeline(m.two_gemm_chain(consumer_wait="setup"), "pdl")["phases"][1]
    assert early[0] == ("setup", 26.5, 28.5)


def test_no_trigger_falls_back_to_gap():
    ks = m.two_gemm_chain(trigger_after_main=False)
    assert m.timeline(ks, "pdl")["total"] == m.timeline(ks, "stream")["total"] == 54
    assert m.timeline(ks, "graph_pdl")["total"] == m.timeline(ks, "graph")["total"] == 52


def test_wait_position_changes_time():
    totals = {w: m.timeline(m.two_gemm_chain(consumer_wait=w), "pdl")["total"]
              for w in ("setup", "prefetch_w2", "mainloop")}
    assert totals == {"setup": 51.5, "prefetch_w2": 49.5, "mainloop": 47.5}


def test_wait_after_all_phases():
    p, c = m.two_gemm_chain()
    c2 = replace(c, phases=c.phases[:1], wait_before=1)      # setup 在 24.5 结束，wait 拖到 26.5
    t = m.timeline([p, c2], "pdl")
    assert t["spans"][1] == (22.5, 26.5)


def test_check_finds_read_before_wait():
    assert m.check(m.two_gemm_chain(consumer_wait="none")) == ["read-before-wait: gemm2.mainloop reads H"]
    for w in ("setup", "prefetch_w2", "mainloop"):
        assert m.check(m.two_gemm_chain(consumer_wait=w)) == []


def test_check_finds_write_before_wait_and_ignores_trigger():
    p = m.Kernel("k1", (m.Phase("main", 5, f({"A"}), f({"B"})),), trigger_after=0)
    c = m.Kernel("k2", (m.Phase("a", 1, writes=f({"A", "B"})), m.Phase("b", 1, f({"B"}))), wait_before=None)
    assert m.check([p, c]) == ["write-before-wait: k2.a writes A", "write-before-wait: k2.a writes B",
                               "read-before-wait: k2.b reads B"]
    assert m.check([replace(p, trigger_after=None), replace(c, wait_before=0)]) == []


def test_check_three_kernel_chain_checks_each_adjacent_pair():
    p = m.Kernel("k1", (m.Phase("w", 1, writes=f({"X"})),), trigger_after=0)
    q = m.Kernel("k2", (m.Phase("r", 1, f({"X"}), f({"Y"})),), trigger_after=0, wait_before=0)
    r = m.Kernel("k3", (m.Phase("r", 1, f({"Y", "X"})),))
    assert m.check([p, q, r]) == ["read-before-wait: k3.r reads Y"]


def test_independent_operands_and_latest_wait():
    p, c = m.two_gemm_chain()
    assert m.independent_operands(p, c) == ["W2"]
    assert m.latest_wait(p, c) == 2
    q = m.Kernel("q", (m.Phase("x", 1, f({"Z"})),))
    assert m.latest_wait(p, q) == 1


def test_best_trigger_and_contention():
    ks = m.two_gemm_chain()
    assert m.best_trigger(ks) == (0, 44.5)
    assert m.best_trigger(ks, contention=0.5) == (1, 47.5)
    assert m.best_trigger(ks, contention=1.0) == (1, 48.0)


def test_contention_only_stretches_the_producer_end():
    t = m.timeline(m.two_gemm_chain(), "pdl", contention=0.5)
    assert t["spans"] == [(0, 27.75), (22.5, 47.5)]
    assert m.timeline(m.two_gemm_chain(), "stream", contention=0.5)["total"] == 54


def test_speedup():
    ks = m.two_gemm_chain()
    assert m.speedup(ks, "stream", "pdl") == pytest.approx(54 / 47.5)
    assert m.speedup(ks, "graph", "graph_pdl") == pytest.approx(52 / 47.5)


def test_unknown_mode():
    with pytest.raises(ValueError):
        m.timeline(m.two_gemm_chain(), "async")


def test_wait_waits_for_end_not_trigger():
    p, c = m.two_gemm_chain()
    slow = replace(p, phases=p.phases[:2] + (replace(p.phases[2], us=30.0),))     # 前驱 trigger 后还要跑 30
    ph = m.timeline([slow, c], "pdl")["phases"][1]
    assert ph[2][1] == 52.5                    # mainloop 等到 2 + 20 + 30 + 0.5


def test_write_hazard_before_wait():
    p, c = m.two_gemm_chain()
    c2 = replace(c, phases=(replace(c.phases[0], writes=frozenset({"X"})),) + c.phases[1:])
    assert m.check([p, c2]) == ["write-before-wait: gemm2.setup writes X"]


def test_trigger_position_never_changes_correctness():
    for w in ("setup", "prefetch_w2", "mainloop", "none"):
        base = m.check(m.two_gemm_chain(consumer_wait=w))
        for ta in (None, 0, 1, 2):
            p, c = m.two_gemm_chain(consumer_wait=w)
            assert m.check([replace(p, trigger_after=ta), c]) == base
