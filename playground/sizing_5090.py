#!/usr/bin/env python3
"""The sizing model behind the knowledge pages "配置决策" and "BM、BN、BK 各管什么" (RTX 5090, BF16 in,
FP32 accumulate, BF16 out, A and B K-contiguous).

    python3 playground/sizing_5090.py                 # every example of the two pages
    python3 playground/sizing_5090.py 2048 512 8192   # your own shape: classification + candidates
    python3 playground/sizing_5090.py --check         # assert the numbers printed in the pages

It is deliberately simple. Time of the busiest SM:

    t_k        = 2·BM·BN·BK / 512 clk                       one k-tile on one SM's Tensor Cores
    main loop  = (CTAs that SM runs in sequence) · (k-tiles per CTA) · t_k
    total      = max(main loop, DRAM lower bound) + 1500 clk fill per sequential CTA + reduction
    reduction  = split-K only: workspace S·M·N·4 B written once and read once at DRAM speed + 1 µs launch

Not modelled: L2 bandwidth (not published for the 5090), epilogue overlap, bank conflicts, clocks
below boost. Use it to drop obviously bad configurations and keep 2–3 candidates, then measure
(Track F, F5 does exactly that).
"""
from __future__ import annotations

import math
import sys

SMS = 170
CLK_GHZ = 2.41
TC_FLOP_PER_CLK = 512           # BF16 dense, FP32 accumulate, per SM per clock
DRAM_GBPS = 1792
SMEM_PER_CTA = 99 * 1024
FILL_CLK = 1500                 # one DRAM round trip before the first k-tile can start
REDUCE_LAUNCH_US = 1.0
PEAK_TFLOPS = TC_FLOP_PER_CLK * SMS * CLK_GHZ / 1e3   # 209.7

SHAPES = {
    "P1 大方阵": (8192, 8192, 8192),
    "P2 中等方阵": (4096, 4096, 4096),
    "P3 tile 少、K 长": (1024, 1024, 16384),
    "P4 小 GEMM": (1024, 256, 1024),
    "P5 decode": (16, 4096, 4096),
    "P6 K 很短": (8192, 8192, 256),
}


def us(clk: float) -> float:
    return clk / (CLK_GHZ * 1e3)


def ideal_us(M, N, K) -> float:
    return 2 * M * N * K / (PEAK_TFLOPS * 1e6)


def dram_us(M, N, K) -> float:
    return (M * K + N * K + M * N) * 2 / (DRAM_GBPS * 1e3)


def t_k(BM, BN, BK=64) -> float:
    return 2 * BM * BN * BK / TC_FLOP_PER_CLK


def classify(M, N, K, BM=128, BN=128, BK=64) -> dict:
    I = 2 * M * N * K / ((M * K + N * K + M * N) * 2)
    T = math.ceil(M / BM) * math.ceil(N / BN)
    kt = math.ceil(K / BK)
    if min(M, N) <= 64 and I < 117:
        cls = "E 瘦长（decode）"
    elif kt <= 8:
        cls = "F K 很短"
    elif ideal_us(M, N, K) < 5:
        cls = "D 小"
    elif T < SMS:
        cls = "C tile 少、K 长"
    elif T < 4 * SMS or T / (math.ceil(T / SMS) * SMS) < 0.9:
        cls = "B 波量化"
    else:
        cls = "A 大而方"
    return {"I": I, "T": T, "kt": kt, "class": cls}


def data_parallel(M, N, K, BM, BN, BK=64) -> dict:
    T = math.ceil(M / BM) * math.ceil(N / BN)
    kt = math.ceil(K / BK)
    seq = math.ceil(T / SMS)
    main = us(seq * kt * t_k(BM, BN, BK))
    total = max(main, dram_us(M, N, K)) + seq * us(FILL_CLK)
    return {"ctas": T, "seq": seq, "kt": kt, "wave_eff": T / (seq * SMS), "main": main, "extra": 0.0, "total": total}


def split_k(M, N, K, BM, BN, S, BK=64) -> dict:
    T = math.ceil(M / BM) * math.ceil(N / BN)
    kt = math.ceil(K / BK)
    ctas = T * S
    seq = math.ceil(ctas / SMS)
    per = math.ceil(kt / S)
    main = us(seq * per * t_k(BM, BN, BK))
    ws = S * M * N * 4
    extra = 2 * ws / (DRAM_GBPS * 1e3) + REDUCE_LAUNCH_US if S > 1 else 0.0
    total = max(main, dram_us(M, N, K)) + seq * us(FILL_CLK) + extra
    return {"ctas": ctas, "seq": seq, "kt": per, "main": main, "extra": extra, "total": total, "workspace_MB": ws / 2**20}


def stream_k(M, N, K, BM, BN, BK=64) -> dict:
    T = math.ceil(M / BM) * math.ceil(N / BN)
    kt = math.ceil(K / BK)
    units = math.ceil(T * kt / SMS)
    main = us(units * t_k(BM, BN, BK))
    total = max(main, dram_us(M, N, K)) + us(FILL_CLK)
    return {"ctas": SMS, "units": units, "main": main, "extra": 0.0, "total": total}


def tflops(M, N, K, total_us) -> float:
    return 2 * M * N * K / (total_us * 1e6)


def bk_sweep(BM=128, BN=128, K=1024, smem=SMEM_PER_CTA, latency=1500, overhead=150, bks=(16, 32, 64, 128), max_stages=8):
    """Main-loop time of one tile vs BK: rounds × (max(compute, latency / (stages − 1)) + overhead)."""
    ideal = 2 * BM * BN * K / TC_FLOP_PER_CLK
    rows = []
    for BK in bks:
        stage = (BM + BN) * BK * 2
        stages = min(max_stages, smem // stage)
        rounds = K // BK
        comp = t_k(BM, BN, BK)
        if stages >= 2:
            per = max(comp, latency / (stages - 1)) + overhead
        else:
            per = comp + latency + overhead
        total = rounds * per
        rows.append({"BK": BK, "stage_KB": stage / 1024, "stages": stages, "rounds": rounds, "compute": comp,
                     "per_round": per, "total": total, "over_ideal": total / ideal - 1})
    return rows


def tile_table(BK=64):
    out = []
    for BM, BN in ((64, 32), (64, 64), (128, 64), (128, 128), (128, 256), (256, 256)):
        stage = (BM + BN) * BK * 2
        out.append({"tile": f"{BM}x{BN}", "acc": BM * BN, "acc_4w": BM * BN // 128, "acc_8w": BM * BN // 256,
                    "stage_KB": stage / 1024, "max_stages": SMEM_PER_CTA // stage, "intensity": BM * BN / (BM + BN)})
    return out


# ------------------------------------------------------------------ printing

def show(title, rows, cols):
    print(f"\n{title}")
    print("  " + "  ".join(f"{c:>12s}" for c in cols))
    for r in rows:
        print("  " + "  ".join(f"{(f'{r[c]:.1f}' if isinstance(r[c], float) else str(r[c])):>12s}" for c in cols))


def examples():
    print(f"RTX 5090 model: {SMS} SMs, {CLK_GHZ} GHz, {PEAK_TFLOPS:.1f} TFLOP/s BF16, {DRAM_GBPS} GB/s")
    show("六个形状", [{"name": n, "ideal_us": ideal_us(*s), "dram_us": dram_us(*s), **classify(*s)} for n, s in SHAPES.items()],
         ["name", "ideal_us", "dram_us", "T", "kt", "class"])
    show("第 2 步：tile 候选（BK = 64）", tile_table(), ["tile", "acc", "acc_4w", "acc_8w", "stage_KB", "max_stages", "intensity"])
    rows = []
    for name, (M, N, K), (BM, BN) in (("P1", SHAPES["P1 大方阵"], (128, 128)), ("P1", SHAPES["P1 大方阵"], (128, 256)),
                                      ("P2", SHAPES["P2 中等方阵"], (128, 128)), ("P2", SHAPES["P2 中等方阵"], (128, 64)),
                                      ("P2", SHAPES["P2 中等方阵"], (128, 256)), ("P3", SHAPES["P3 tile 少、K 长"], (128, 128))):
        r = data_parallel(M, N, K, BM, BN)
        rows.append({"problem": name, "tile": f"{BM}x{BN}", "ctas": r["ctas"], "seq": r["seq"], "wave_eff": 100 * r["wave_eff"], "main_us": r["main"]})
    show("第 3 步：数 tile、算波（主循环不含填充）", rows, ["problem", "tile", "ctas", "seq", "wave_eff", "main_us"])
    M, N, K = SHAPES["P3 tile 少、K 长"]
    rows = []
    for S in (1, 2, 3, 4, 5, 8, 13):
        r = split_k(M, N, K, 128, 128, S)
        rows.append({"plan": "不切" if S == 1 else f"S={S}", "ctas": r["ctas"], "work": f"{r['seq']}x{r['kt']}", "main": r["main"],
                     "extra": r["extra"], "total": r["total"], "TFLOPs": tflops(M, N, K, r["total"])})
    r = stream_k(M, N, K, 128, 128)
    rows.append({"plan": "Stream-K", "ctas": r["ctas"], "work": f"{r['units']} units", "main": r["main"], "extra": 0.0,
                 "total": r["total"], "TFLOPs": tflops(M, N, K, r["total"])})
    r = data_parallel(M, N, K, 64, 64)
    rows.append({"plan": "64x64 不切", "ctas": r["ctas"], "work": f"{r['seq']}x{r['kt']}", "main": r["main"], "extra": 0.0,
                 "total": r["total"], "TFLOPs": tflops(M, N, K, r["total"])})
    show("4.5 P3：split-K / Stream-K", rows, ["plan", "ctas", "work", "main", "extra", "total", "TFLOPs"])
    rows = []
    for name, (M, N, K) in (("P2", SHAPES["P2 中等方阵"]), ("P6", SHAPES["P6 K 很短"])):
        rows.append({"problem": name, "plan": "不切", "total": data_parallel(M, N, K, 128, 128)["total"]})
        rows.append({"problem": name, "plan": "S=2", "total": split_k(M, N, K, 128, 128, 2)["total"]})
        if name == "P2":
            rows.append({"problem": name, "plan": "Stream-K", "total": stream_k(M, N, K, 128, 128)["total"]})
    show("4.8 什么时候切 K 反而更慢（128x128）", rows, ["problem", "plan", "total"])
    M, N, K = SHAPES["P5 decode"]
    rows = [{"plan": "128x128", **pick(data_parallel(M, N, K, 128, 128))}, {"plan": "16x64", **pick(data_parallel(M, N, K, 16, 64))},
            {"plan": "16x32", **pick(data_parallel(M, N, K, 16, 32))}, {"plan": "16x64 S=4", **pick(split_k(M, N, K, 16, 64, 4))}]
    show("P5 decode", rows, ["plan", "ctas", "total"])
    show("bk_sweep：128x128, K = 1024, 99 KB", bk_sweep(), ["BK", "stage_KB", "stages", "rounds", "compute", "per_round", "total", "over_ideal"])
    show("bk_sweep：227 KB（H100）", bk_sweep(smem=232448, bks=(64, 128)), ["BK", "stages", "over_ideal"])
    show("bk_sweep：64x32 tile", bk_sweep(64, 32, bks=(32, 64, 128)), ["BK", "stage_KB", "stages", "over_ideal"])


def pick(r):
    return {"ctas": r["ctas"], "total": r["total"]}


def custom(M, N, K):
    c = classify(M, N, K)
    print(f"{M} x {N} x {K}: I = {c['I']:.0f} FLOP/B, T(128x128) = {c['T']}, kt = {c['kt']} -> {c['class']}")
    print(f"  ideal {ideal_us(M, N, K):.1f} us, DRAM lower bound {dram_us(M, N, K):.1f} us")
    rows = []
    tiles = [(128, 128), (128, 64), (64, 64), (64, 32)]
    if min(M, N) <= 64:            # decode-like: let BM hug M (the model has no L2, so small tiles only here)
        tiles += [(16, 64), (16, 32)]
    for BM, BN in tiles:
        r = data_parallel(M, N, K, BM, BN)
        rows.append({"plan": f"{BM}x{BN} DP", "ctas": r["ctas"], "total": r["total"], "TFLOPs": tflops(M, N, K, r["total"])})
        r = stream_k(M, N, K, BM, BN)
        rows.append({"plan": f"{BM}x{BN} Stream-K", "ctas": r["ctas"], "total": r["total"], "TFLOPs": tflops(M, N, K, r["total"])})
        for S in (2, 3, 4, 5, 8):
            if math.ceil(K / 64) // S >= 4:
                r = split_k(M, N, K, BM, BN, S)
                rows.append({"plan": f"{BM}x{BN} S={S}", "ctas": r["ctas"], "total": r["total"], "TFLOPs": tflops(M, N, K, r["total"])})
    rows.sort(key=lambda r: (round(r["total"], 1), -int(r["plan"].split("x")[0])))
    show("candidates (best first; no L2 model: among near-ties prefer the bigger tile)", rows[:12], ["plan", "ctas", "total", "TFLOPs"])


def check():
    """The numbers the knowledge pages print, recomputed."""
    def near(a, b, tol=0.6):
        assert abs(a - b) <= tol, (a, b)
    P1, P2, P3, P4, P5, P6 = SHAPES.values()
    near(ideal_us(*P1), 5242), near(dram_us(*P1), 225), near(ideal_us(*P2), 655), near(dram_us(*P2), 56)
    near(ideal_us(*P3), 164), near(dram_us(*P3), 39), near(dram_us(*P5), 18.9, 0.05), near(dram_us(*P6), 80)
    near(data_parallel(*P1, 128, 128)["main"], 5439), near(data_parallel(*P1, 128, 256)["main"], 5656)
    near(data_parallel(*P2, 128, 128)["main"], 761), near(data_parallel(*P2, 128, 64)["main"], 707)
    near(data_parallel(*P2, 128, 256)["main"], 870), near(data_parallel(*P3, 128, 128)["main"], 435)
    for S, total in ((1, 436), (2, 229), (3, 309), (4, 239), (5, 202), (8, 259), (13, 235)):
        near(split_k(*P3, 128, 128, S)["total"], total)
    near(stream_k(*P3, 128, 128)["total"], 166), near(data_parallel(*P3, 64, 64)["total"], 219)
    near(data_parallel(*P2, 128, 128)["total"], 766), near(split_k(*P2, 128, 128, 2)["total"], 866)
    near(stream_k(*P2, 128, 128)["total"], 657), near(data_parallel(*P6, 128, 128)["total"], 186)
    near(split_k(*P6, 128, 128, 2)["total"], 797)
    near(data_parallel(*P5, 128, 128)["total"], 109), near(data_parallel(*P5, 16, 64)["total"], 19.5, 0.05)
    near(data_parallel(*P5, 16, 32)["total"], 19.5, 0.05), near(split_k(*P5, 16, 64, 4)["total"], 22.3, 0.05)
    near(data_parallel(*P1, 128, 128)["total"], 5454), near(stream_k(*P1, 128, 128)["total"], 5244)
    near(data_parallel(*P2, 128, 64)["total"], 715)
    near(stream_k(*P4, 64, 64)["total"], 3.6, 0.05), near(data_parallel(*P4, 64, 32)["total"], 4.0, 0.05)
    assert [round(100 * r["over_ideal"], 1) for r in bk_sweep()] == [14.6, 7.3, 3.7, 20.1]
    assert [round(100 * r["over_ideal"], 1) for r in bk_sweep(smem=232448, bks=(64, 128))] == [3.7, 1.8]
    assert [round(100 * r["over_ideal"], 1) for r in bk_sweep(64, 32, bks=(32, 64, 128))] == [58.6, 29.3, 14.6]
    assert [r["max_stages"] for r in tile_table()] == [8, 6, 4, 3, 2, 1]
    print("all page numbers reproduced")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--check"]:
        check()
    elif len(a) == 3:
        custom(*map(int, a))
    else:
        examples()
