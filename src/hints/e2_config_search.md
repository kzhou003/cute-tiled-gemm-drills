lede: CuTe 能告诉你一个配置合不合法、能生成它的代码，但不会替你选 tile、stage 数、warp group 安排和 swizzle。本题把 Hopper 上的硬约束（wgmma 形状、TMA box、SMEM 容量、寄存器）写成检查器，再写一个带 L2 复用、wave 量化和流水线延迟的代价模型，在 972 个配置里排序，并和 CUTLASS 常用的选择对照：模型能筛掉不合法的配置、指出趋势，最终选择仍然要实测。
# E2 · 选配置：CTA tile、stage 数、cooperative / pingpong 与 swizzle，在硬件约束下用代价模型排序

## 题面速览

- **背景知识**：[Layout Algebra 与编译](../knowledge/algebra_compile.html) 里“编译期能检查什么、不能决定什么”，以及 [真实代码导读](../knowledge/real_code.html)。
- **在体系里的位置**：CUTLASS 的 CollectiveBuilder（[[cutlass:include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl|sm90_gmma_builder.inl]]）用 SMEM 容量减去 epilogue 的 carveout，再除以每个 stage 的字节数得到 stage 数；kernel 用 `LoadRegisterRequirement = 40 / MmaRegisterRequirement = 232` 重新分配寄存器（D3）。tile 形状和 schedule 则由用户在 CUTLASS 的例子或 profiler 里手选。
- **你要实现**：`smem_bytes`、`acc_regs_per_thread`、`violations`、`estimate`、`search`、`default_space`。常量、`GPU`、`H100_SXM` 和 `Config` 已经给出。
- **关键规则**：
  1. 合法性按固定顺序检查：`wgmma_m`、`wgmma_n`、`swizzle`、`tma_box`、`smem`、`acc_regs`、`regs`、`stages`，返回所有违反的规则。
  2. cooperative 两个 WG 合算一个 tile（每个 WG 64 行起，所以 BM 要是 128 的倍数），累加器 `BM·BN / 256` 个寄存器每线程；pingpong 每个 WG 持有整个 tile，`BM·BN / 128`。
  3. 每个 k-tile 的时间 `step = max(t_mma_k, t_load_k, (lat + t_load_k) / stages)`：stage 太少时延迟藏不住。
  4. L2 复用：一个 wave 里同时运行的 tile 共享 A、B panel，只有不同的 panel 才从 HBM 读。
  5. 排序键是 `(total, config)`：并列时按配置字段升序，结果确定。

**验收要点**：`smem_bytes(128×128×64, 4) = 139328`、累加器寄存器 64 / 128 · 每条规则各一个违反的例子 · 多条违反按顺序全报 · regs 规则读 GPU 的寄存器文件 · 128×128×64×4 cooperative 的 tiles / waves / step / total / TFLOP/s · pingpong 的 waves 与 total · 2 个 stage 比 4 个慢（216.5 vs 159.9 µs）· 窄 swizzle 更慢 · L2 复用降低 step · 搜索空间 972 个、合法 357 个、排序与第一名、并列时的次序 · 小问题选小 tile。

::: callout info 第一次出现的术语
- **wgmma**（warpgroup MMA）：Hopper 的 tensor core 指令，一个 warp group 一次算 M = 64、N ∈ {8, 16, …, 256} 的块。
- **cooperative / pingpong**：CUTLASS sm90 的两种 warp-specialized kernel schedule（D3）。
- **wave**：GPU 一次能同时放下的 CTA 数（这里每个 SM 一个）。tiles 不是 SM 数的整数倍时，最后一个 wave 不满，叫 **wave quantization**（波量化）损失。
- **carveout**：从 SMEM 里预留给 epilogue 等其他用途的字节数，剩下的才给主循环的 stage。
- **roofline**：时间取“算力限制”和“带宽限制”中较大的一个。
:::

## 直觉与类比

::: callout analogy 给一间工作室排班
SM 是一间工作室，SMEM 是工作台面积，寄存器是每个工人手边的抽屉。tile 越大，一次加工的件越多、来回搬料的次数越少（算术强度高），但工作台要放得下 stages 份料（smem），工人的抽屉要放得下手上的半成品（累加器寄存器）。stage 数是台面上同时摆几份料：太少，工人做完一份要等下一份从仓库运来（延迟藏不住）；太多，台面放不下。pingpong 像两个工人共用一台机器：一个在打包（epilogue）时另一个用机器，机器不闲着，但每个人的抽屉要放下整件半成品。最后，100 份订单分给 132 间工作室，有 32 间闲着；1024 份订单要 8 轮，最后一轮只开 100 间：这就是 wave 量化。
:::

## 图解：一个配置的生死与快慢

::: flow Config → violations → estimate → 排名
cfg cw=220 rh=100 nw=180
n cfg 0 0.5 "Config" "tile · stages · 调度" k=dark
n shp 1 0 "形状约束" "wgmma · swizzle · box" k=todo
n res 1 1 "资源约束" "smem · 寄存器 · stages" k=todo
n kt 2 0 "每个 k-tile" "t_mma · t_load" k=acc
n pl 2 1 "流水线" "延迟 / stages" k=vio
n wv 3 0.5 "wave" "ceil(tiles / 并发)" k=teal
n rk 4 0.5 "排序" "(total, config)" k=pass
e cfg -> shp
e cfg -> res
e shp -> kt
e res -> pl
e kt -> wv
e pl -> wv
e wv -> rk
cap: 约束是硬的：违反一条就不参加排序。代价模型是软的：它只负责把明显差的配置排到后面，相近的配置之间的先后不可信。
:::

## 逐步演示

::: steps 4096³ 上的 128×128×64、stages = 4：从合法性到 159.9 µs
::: flow
cfg cw=180 rh=80 nw=150
n sm 0 0 "smem" "139328 B" k=acc
n ar 1 0 "acc_regs" "64 / 线程" k=vio
n kt 2 0 "k-tile" "t_mma 0.2799" k=teal
n w 3 0 "8 waves" "eff 0.970" k=warn
n pp 4 0 "pingpong" "151.6 µs" k=pass
n s2 2 1 "2 stages" "216.5 µs" k=todo
e sm -> ar
e ar -> kt
e kt -> w
e w -> pp
e kt -> s2 dash
:::
--- step SMEM {hl=sm}
`4 × (128 + 128) × 64 × 2 + 8192 + 4 × 16 = 139328` 字节 < 232448。按同样的公式，同一个 tile 最多能放 `(232448 − 8192) / (32768 + 16) = 6` 个 stage；7 个 stage 是 237680 字节，违反 `smem`。
--- step 寄存器 {hl=sm,ar}
cooperative 每个 MMA 线程持有 128·128 / 256 = 64 个 FP32 累加器，加上 40 个其他寄存器 = 104 ≤ 232。256×256 cooperative 要 256 个，违反 `acc_regs`；256×128 cooperative 正好 128 + 40 = 168，合法，但同一个 tile 换成 pingpong 就是 256，不合法。
--- step 一个 k-tile {hl=ar,kt}
`t_mma_k = 2·128·128·64 / (989e6 / 132) = 0.2799` µs。1024 个 tile、每 wave 132 个，同一个 wave 里只有 `ceil(132/32) = 5` 个不同的 A panel 和 32 个不同的 B panel，复用系数 `(5·128 + 32·128) / (132·256) = 0.140`，`t_load_k = 0.181` µs（每个 CTA 只分到 1/132 的 HBM 带宽）。`step = max(0.2799, 0.181, (0.6 + 0.181)/4 = 0.195) = 0.2799`：计算受限。
--- step stage 太少 {hl=kt,s2}
同一个 tile 只给 2 个 stage：`(0.6 + 0.181) / 2 = 0.3905`，超过了 `t_mma_k`，step 变成 0.3905，总时间从 159.9 涨到 216.5 µs。这就是 D2、E1 里看到的“延迟藏不住”。
--- step wave 量化 {hl=kt,w}
1024 / 132 = 7.76，要 8 个 wave，效率 1024 / 1056 = 0.970。`t_wave = 0.6 + 0.181 + 64·0.2799 + t_epi = 19.99` µs，`total = 8 × 19.99 = 159.887` µs，859.6 TFLOP/s。
--- step 换成 pingpong {hl=w,pp}
每个 SM 同时处理两个 tile，wave 数降到 4，但两个 WG 共用 tensor core：`t_wave = max(2·64·0.2799, 64·0.2799) + …`，总时间 151.6 µs，比 cooperative 快 5%：多出来的是 epilogue 和 load 尾巴被另一个 WG 的 MMA 藏住了。这和 D3 的结论一致：epilogue 不超过 MMA 的两倍时 pingpong 不慢。
:::

在整个默认搜索空间（972 个配置，合法 357 个）里，模型的第一名是 `64×256×64、pingpong、2 stages`（906.7 TFLOP/s，前几名并列，只有 stage 数不同）。常见的 CUTLASS 选择 `128×128×64 cooperative 4 stages` 排第 43，`128×256×64 cooperative` 排第 36，`64×128×64 pingpong` 排第 19。模型认为它们差距不大，排名对这些细节非常敏感：这正是“最后一步要实测”的原因。

## 易错点

- **cooperative 只要求 BM % 64。** 两个 WG 各 64 行起步，BM 必须是 128 的倍数。测试 `test_violations` 里 64×128 cooperative 报 `wgmma_m`、pingpong 合法。
- **只返回第一个违反的规则。** 128×128×512 同时违反 `tma_box` 和 `smem`，测试要求两个都报、按固定顺序。
- **累加器按 tile 总量算，不分 schedule。** pingpong 每个 WG 持有整个 tile，寄存器翻倍。测试 `test_smem_and_acc_regs`。
- **t_load 用整块 HBM 带宽。** 132 个 CTA 同时读，每个 CTA 只分到 1/132（再乘复用系数）。不这样做的话所有配置都被判成计算受限，`test_too_few_stages_expose_latency` 会失败。
- **swizzle 只当合法性检查。** 窄 swizzle 合法但慢（MMA 读 SMEM 有 bank 冲突），不乘系数时 32B swizzle 会和 128B 并列，排序变得没有意义。测试 `test_narrow_swizzle_slows_mma`。
- **排序不稳定。** 很多配置的 total 完全相同（pingpong 在计算受限时与 stage 数无关），只按 total 排序时结果依赖输入顺序。测试 `test_search_breaks_ties_by_config_fields`。

## 对照文档

> “不要误解为：CuTe 会自动搜索所有 CTA tile、warp 数、pipeline stage、swizzle 和 scheduler 参数，并保证找到全局最优 kernel。它提供的是可组合的表达、静态推导、合法性检查和代码生成路径；性能搜索通常还需要人工设计搜索空间、autotuning 和实机 profiling。”

本题把这句话拆成三步：`violations` 是“合法性检查”（C2 的 CUTLASS 层），`estimate` 是“代价模型”（排除明显差的），`search` 是“搜索空间”。缺的那一步是实机 profiling：模型里 64×256 pingpong 和 128×128 pingpong 只差 0.01 µs，真实 GPU 上谁快取决于模型没写进去的东西（L2 命中率的细节、指令发射、epilogue 和 TMA store 的争用）。

## 上机验证

```bash
DRILL_IMPL=solutions python3 -c "
from drill import load
m = load('e2_config_search')
r = m.search(m.default_space(), m.H100_SXM, 4096, 4096, 4096)
for c, e in r[:5]: print(c, round(e['total'], 2), round(e['tflops'], 1))"
```

`H100_SXM` 是示例规格（132 个 SM、每 CTA 最多 232448 字节 SMEM、3.35 TB/s、989 TFLOP/s dense FP16）。真机上可以用 CUTLASS profiler（`cutlass_profiler --operation=gemm --m=4096 --n=4096 --k=4096`）对同样的候选测时间，看模型的排名和实测差多少（本机没有 GPU，未运行）。RTX 5090 的规格完全不同（170 个 SM、每 SM 128 KB 级的 SMEM、没有 WGMMA），要换一套 `GPU(...)` 参数和约束才有意义。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
violations(Config(128, 128, 512, 2), H100_SXM)       # ['tma_box', 'smem']
violations(Config(256, 256, 64, 2), H100_SXM)        # ['acc_regs']
e = estimate(Config(128, 128, 64, 4), H100_SXM, 4096, 4096, 4096)
e["waves"], round(e["step"], 4), round(e["total"], 3)            # (8, 0.2799, 159.887)
estimate(Config(128, 128, 64, 2), H100_SXM, 4096, 4096, 4096)["total"]       # 216.502…
r = search(default_space(), H100_SXM, 4096, 4096, 4096)
len(default_space()), len(r)                          # (972, 357)
search(default_space(), H100_SXM, 512, 512, 8192)[0][0]        # 64×64×64, stages=5, pingpong
```

#### 关键问题

1. 为什么 pingpong 的累加器寄存器是 cooperative 的两倍？这对 tile 大小意味着什么？
2. 为什么 stage 数在计算受限时“够了就不再有用”？在什么问题形状下会重新变得重要？
3. 512×512×8192 这种 M、N 小、K 大的问题，为什么模型选小 tile？真实世界里还有什么办法（提示：split-K、stream-K）？
:::

::: hint 2 思路拆解与常见坑
- `smem_bytes` 和 `acc_regs_per_thread`：照公式写。
- `violations`：八个 if，按顺序 append。
- `estimate`：先算 tiles、k_tiles、W = min(tiles, sms)、复用系数；再算 `t_mma_k`、`t_load_k`、`step`、`t_epi`；最后按 schedule 算 waves 和 `t_wave`。
- `search`：过滤 → 估算 → 按 `(total, config)` 排序（`Config` 是 `order=True` 的 dataclass，可以直接比较）。
- `default_space`：`itertools.product` 六个维度。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. cooperative 两个 WG 平分一个 tile 的累加器，pingpong 每个 WG 持有一个完整 tile。同样 232 个寄存器的预算，pingpong 的 tile 面积只能是 cooperative 的一半（例如 64×256 或 128×128 对 128×256）。
2. stage 只用来藏延迟：只要 `(lat + t_load) / stages ≤ max(t_mma, t_load)`，再加 stage 只占 SMEM，不缩短 step。延迟相对计算变大时（K 小的 tile、带宽受限的形状、或 TMA 延迟有抖动）它又重要了。
3. 只有 (512/BM)·(512/BN) 个 tile：大 tile 只能开很少几个 CTA，大部分 SM 空着（wave 效率极低）。小 tile 让 64 个 CTA 同时跑。真实世界还会用 split-K / stream-K 把 K 方向也切开，让更多 SM 参与，最后再归约。
:::
:::

## 参考解

::: solution solutions/e2_config_search.py
{{code: solutions/e2_config_search.py}}
:::

## 面试官追问 & 对照源码

- CUTLASS 的 `StageCountAutoCarveout` 怎样从 SMEM 容量算出 stage 数（[[cutlass:include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl|sm90_gmma_builder.inl]]）？它和本题的 `smem_bytes` 有什么不同？
- 本模型没有 cluster：2×1 的 cluster 加 TMA multicast 能把 A 或 B 的 HBM 读取减半。把它加进模型，排名会怎样变？
- L2 复用系数假设 tile 按行主序发射。CUTLASS 的 tile scheduler 有 swizzle（raster order）选项：它改变的是哪一项？
- 如果要做一个 autotuner：先用 C2 的 `check_config` 和本题的 `violations` 过滤，再用 `estimate` 取前 N 个去实测。N 取多少合适？依据是什么？
