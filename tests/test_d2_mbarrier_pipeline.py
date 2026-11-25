import pytest

from drill import load
from tests.golden import golden

m = load("d2_mbarrier_pipeline")
G = golden("d2_mbarrier_pipeline")


def test_new_barrier_parity_rule():
    b = m.MBarrier(1)
    assert b.try_wait(1) is True          # “上一个 phase”在新 barrier 上视为已完成
    assert b.try_wait(0) is False
    b.arrive()
    assert b.completed == 1
    assert b.try_wait(0) is True and b.try_wait(1) is False


def test_phase_needs_both_arrivals_and_bytes():
    b = m.MBarrier(2)
    b.arrive_and_expect_tx(1024)
    assert b.completed == 0
    b.arrive()
    assert b.completed == 0                # 字节还没到
    b.complete_tx(512)
    assert b.completed == 0
    b.complete_tx(512)
    assert b.completed == 1
    assert b.pending == 2                  # 下一个 phase 重新计数


def test_expect_tx_before_arrive():
    b = m.MBarrier(1)
    b.arrive_and_expect_tx(64)
    assert b.completed == 0 and b.tx == 64
    b.complete_tx(64)
    assert b.completed == 1


def test_too_many_arrivals_is_a_protocol_error():
    b = m.MBarrier(1)
    with pytest.raises(m.ProtocolError):
        b.arrive(2)


def test_pipeline_state_increment_wraps_and_flips():
    st = m.PipelineState(3)
    seen = []
    for _ in range(7):
        seen.append(st.as_tuple())
        st.increment()
    assert seen == [(0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3), (1, 1, 4), (2, 1, 5), (0, 0, 6)]


def test_single_stage_flips_every_step():
    st = m.PipelineState(1)
    st.increment()
    assert st.as_tuple() == (0, 1, 1)
    st.increment()
    assert st.as_tuple() == (0, 0, 2)


@pytest.mark.parametrize("S,start,n", [(4, 0, 9), (3, 2, 5), (5, 4, 13), (2, 1, 1), (4, 3, 0)])
def test_advance_equals_repeated_increment(S, start, n):
    a = m.PipelineState(S, start, 1, 0)
    b = m.PipelineState(S, start, 1, 0)
    a.advance(n)
    for _ in range(n):
        b.increment()
    assert a.as_tuple() == b.as_tuple()


def test_producer_start_state():
    assert m.producer_start_state(4).as_tuple() == (0, 1, 0)


def test_running_example_timeline():
    r = m.simulate(4, 8, 700, 200)
    assert r.load_issue == [0, 0, 0, 0, 900, 1100, 1300, 1500]
    assert r.load_done == [700, 700, 700, 700, 1600, 1800, 2000, 2200]
    assert r.mma_start == [700, 900, 1100, 1300, 1600, 1800, 2000, 2200]
    assert r.mma_end == [900, 1100, 1300, 1500, 1800, 2000, 2200, 2400]
    assert r.total == 2400
    assert r.steady_interval == pytest.approx(225.0)


@pytest.mark.parametrize("S", [1, 2, 3, 4, 5, 6])
def test_steady_interval_matches_formula(S):
    r = m.simulate(S, 12, 700, 200)
    assert r.steady_interval == pytest.approx(m.predicted_interval(S, 700, 200))


def test_formulas():
    assert m.predicted_interval(4, 700, 200) == pytest.approx(225.0)
    assert m.predicted_interval(5, 700, 200) == pytest.approx(200.0)
    assert m.min_stages(700, 200) == 5
    assert m.min_stages(400, 200) == 3
    assert m.min_stages(100, 200) == 2


def test_more_stages_stop_helping_after_min_stages():
    assert m.simulate(5, 12, 700, 200).total == m.simulate(6, 12, 700, 200).total == 3100
    assert m.simulate(4, 12, 700, 200).total == 3300


def test_short_loop_has_no_steady_interval():
    assert m.simulate(4, 4, 700, 200).steady_interval is None


def test_bug_producer_phase0_deadlocks():
    with pytest.raises(m.DeadlockError):
        m.simulate(4, 8, 700, 200, bug="producer_phase0")


def test_bug_consumer_phase1_reads_dirty_data():
    with pytest.raises(m.RaceError):
        m.simulate(4, 8, 700, 200, bug="consumer_phase1")


def test_bug_no_release_deadlocks_after_filling_the_ring():
    with pytest.raises(m.DeadlockError):
        m.simulate(4, 8, 700, 200, bug="no_release")
    # 环没转满时不会暴露
    assert m.simulate(4, 4, 700, 200, bug="no_release").total == 1500


def test_bug_skip_acquire_overwrites_a_busy_stage():
    with pytest.raises(m.RaceError):
        m.simulate(4, 8, 700, 200, bug="skip_acquire")


def _walk(st, n):
    out = []
    for _ in range(n):
        out.append(list(st.as_tuple()))
        st.increment()
    return out


def test_pipeline_state_matches_real_cutlass():
    assert _walk(m.producer_start_state(4), 10) == G["producer_4"]
    assert _walk(m.PipelineState(4), 10) == G["consumer_4"]
    assert _walk(m.PipelineState(3), 7) == G["consumer_3"]
    assert _walk(m.PipelineState(1), 3) == G["consumer_1"]


@pytest.mark.parametrize("case", G["advance"], ids=lambda c: f"S{c['stages']}_{c['start']}_{c['n']}")
def test_advance_matches_real_cutlass(case):
    st = m.PipelineState(case["stages"], case["start"][0], case["start"][1], 0)
    st.advance(case["n"])
    assert list(st.as_tuple()) == case["got"]
