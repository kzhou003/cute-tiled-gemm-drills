import random

import pytest

from drill import load

m = load("d3_warp_specialization")


def test_launch_regs_round_down_to_8():
    assert m.launch_regs_per_thread(384) == 168
    assert m.launch_regs_per_thread(256) == 256
    assert m.launch_regs_per_thread(640) == 96      # 65536/640 = 102.4 → 96


def test_validate_setmaxnreg():
    assert m.validate_setmaxnreg(40) == [] and m.validate_setmaxnreg(232) == []
    assert [e.split(":")[0] for e in m.validate_setmaxnreg(41)] == ["E_NREG_MULT8"]
    assert [e.split(":")[0] for e in m.validate_setmaxnreg(16)] == ["E_NREG_RANGE"]
    assert [e.split(":")[0] for e in m.validate_setmaxnreg(264)] == ["E_NREG_RANGE"]
    assert [e.split(":")[0] for e in m.validate_setmaxnreg(257)] == ["E_NREG_MULT8", "E_NREG_RANGE"]


def test_cutlass_register_plan_uses_the_whole_budget():
    p = m.register_plan()
    assert p == dict(threads=384, launch_regs=168, launch_total=64512, final_total=64512,
                     freed=16384, needed=16384, ok=True, errors=[])


def test_heavy_register_plan_24_240():
    p = m.register_plan(1, 24, 2, 240)
    assert p["ok"] and p["freed"] == p["needed"] == 18432


@pytest.mark.parametrize("args,codes", [
    ((1, 40, 2, 240), ["E_BUDGET"]),
    ((1, 180, 2, 232), ["E_NREG_MULT8", "E_DEALLOC_ABOVE_LAUNCH", "E_BUDGET"]),
    ((1, 41, 2, 160), ["E_NREG_MULT8", "E_ALLOC_BELOW_LAUNCH"]),
])
def test_register_plan_errors(args, codes):
    p = m.register_plan(*args)
    assert not p["ok"]
    assert [e.split(":")[0] for e in p["errors"]] == codes


def test_accumulator_registers():
    assert m.accumulator_regs(128, 128, 256) == 64      # cooperative：两个 WG 分一个 tile
    assert m.accumulator_regs(128, 128, 128) == 128     # ping-pong：一个 WG 一个 tile
    assert m.accumulator_regs(64, 128, 128) == 64


def _seg(s):
    return [(x.wg, x.kind, x.tile, x.start, x.end) for x in s.segments]


def test_pingpong_running_example_timeline():
    s = m.schedule_pingpong(4, 10, 4)
    assert _seg(s) == [(1, "mma", 0, 0, 10), (1, "epi", 0, 10, 14), (2, "mma", 1, 10, 20), (1, "mma", 2, 20, 30),
                       (2, "epi", 1, 20, 24), (1, "epi", 2, 30, 34), (2, "mma", 3, 30, 40), (2, "epi", 3, 40, 44)]
    assert s.makespan == 44 and s.tensor_core_busy == 40
    assert s.utilization == pytest.approx(40 / 44)


def test_cooperative_timeline():
    s = m.schedule_cooperative(2, 10, 4)
    assert _seg(s) == [(1, "mma", 0, 0, 10), (2, "mma", 0, 0, 10), (1, "epi", 0, 10, 12), (2, "epi", 0, 10, 12),
                       (1, "mma", 1, 12, 22), (2, "mma", 1, 12, 22), (1, "epi", 1, 22, 24), (2, "epi", 1, 22, 24)]
    assert m.schedule_cooperative(4, 10, 4).makespan == 48


def test_pingpong_hides_epilogue_when_it_is_short():
    assert m.schedule_pingpong(4, 10, 4).makespan < m.schedule_cooperative(4, 10, 4).makespan


def test_cooperative_wins_when_epilogue_is_long():
    assert m.schedule_cooperative(4, 10, 25).makespan == 90
    assert m.schedule_pingpong(4, 10, 25).makespan == 110


def test_steady_per_tile():
    assert m.steady_per_tile("cooperative", 10, 4) == 12
    assert m.steady_per_tile("pingpong", 10, 4) == 10
    assert m.steady_per_tile("pingpong", 10, 25) == 25
    with pytest.raises(ValueError):
        m.steady_per_tile("serial", 1, 1)


def test_pingpong_invariants_on_random_inputs():
    rng = random.Random(3)
    for _ in range(50):
        n, M, E = rng.randint(1, 12), rng.uniform(1, 20), rng.uniform(0.5, 30)
        s = m.schedule_pingpong(n, M, E)
        mma = sorted((x for x in s.segments if x.kind == "mma"), key=lambda x: x.start)
        assert all(a.end <= b.start + 1e-9 for a, b in zip(mma, mma[1:]))          # Tensor Core 不重叠
        for wg in (1, 2):
            own = sorted((x for x in s.segments if x.wg == wg), key=lambda x: x.start)
            assert all(a.end <= b.start + 1e-9 for a, b in zip(own, own[1:]))      # 一个 WG 同一时刻只做一件事
        assert s.tensor_core_busy == pytest.approx(n * M)
        if n >= 6:
            per = (s.makespan - m.schedule_pingpong(n - 2, M, E).makespan) / 2
            assert per == pytest.approx(m.steady_per_tile("pingpong", M, E), rel=1e-6)


@pytest.mark.parametrize("E,pp,coop", [(4, 84, 96), (10, 90, 120), (16, 138, 144)])
def test_eight_tiles_makespans(E, pp, coop):
    assert m.schedule_pingpong(8, 10, E).makespan == pytest.approx(pp)
    assert m.schedule_cooperative(8, 10, E).makespan == pytest.approx(coop)


def test_three_consumer_warp_groups_cap_at_160():
    assert m.launch_regs_per_thread(512) == 128
    assert m.register_plan(1, 24, 3, 160)["ok"]
    assert [e.split(":")[0] for e in m.register_plan(1, 24, 3, 168)["errors"]] == ["E_BUDGET"]


def test_empty_schedule():
    s = m.schedule_pingpong(0, 10, 4)
    assert s.segments == [] and s.makespan == 0 and s.utilization == 0
