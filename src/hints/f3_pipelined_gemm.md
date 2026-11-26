lede: 在 F2 上加三样东西，就得到 CuTe 教程 sgemm\_sm80 的完整结构：SMEM 开 3 级环形缓冲，算第 k 块时第 k+2 块已经在路上（cp.async 的 commit / wait\_group）；SMEM → 寄存器改用ldmatrix.x4；寄存器双缓冲，算第 kb 个 k16 时先把第 kb+1 个读进来。主机部分用整数模拟主循环的每一次 load / wait / compute，检查三条不变量——这是这道题最容易写错、也最难在 GPU 上调试的部分。
# F3 · 多级 cp.async 流水线 + ldmatrix + 寄存器双缓冲

## 题面速览

- 背景知识：[时间维度](../knowledge/time_dimension.html)（流水线的思路）；[配置决策](../knowledge/config.html) 第 6 步（stage 数怎么定）；[真实代码导读](../knowledge/real_code.html) 的 sgemm\_sm80 部分。上机准备见 [F1](../hints/f1_simt_cute_gemm.html)。
- 在体系里的位置：逐行对照 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu#L132-L305|sgemm_sm80.cu 的预取和主循环]]。区别：BF16 + FP32 累加、8 个 warp（2×4）、C 是 FP32 行主序。
- 你要写：

1. `mine/f3_pipelined_gemm.cu` 的一段 `DRILL_TODO` ⑥⑦：预取、主循环。①–⑤（tile、拷贝分工、MMA 分工、ldmatrix 的 retile）已经给出。
2. `mine/f3_pipelined_gemm.py` ：`trace` 、`check_trace` 和五个算式函数。

- 配置：`Config<128,128,64,3,2,4>` ：3 × 32 KB = 96 KB 动态 SMEM（要 `cudaFuncSetAttribute` 打开 > 48 KB），256 线程，每线程 64 个累加器。

测试检查：`trace(5, 3)` 和 sgemm\_sm80 的循环逐事件一致 · 24 种 (k\_tiles, stages) 组合下不变量全部成立、每个 k-tile 恰好算一次、多余的 load 正好 stages − 1 个 · `check_trace` 能抓住三种错误（没等就算、顺序错、覆盖了没用过的数据）·stage 算式（99 KB 下 128×128 最多 3 级，227 KB 下 7 级）· `.cu` 类型检查 · GPU 上三个形状对拍（可选：`DRILL_F3_MIN_TFLOPS=100` 加速度下限）。

## 直觉与类比

::: callout analogy 三个工作台轮流用
F2 的小组只有一张工作台：搬料时冲压机停，冲压时叉车停。F3 有三张：冲压机在用 1 号台的料时，2 号台的料已经到了，3 号台正在卸货。每用完一张台子，就让叉车去给它装“三轮之后”要用的料。工人从台子上取料也不再一件件拿，而是一次抱一摞（ldmatrix.x4），而且在冲这一摞的同时，先把下一摞抱到手边（寄存器双缓冲）。
:::

## 图解：一圈主循环里发生了什么

::: flow 一圈主循环里三级 SMEM 各在做什么（STAGES = 3）
cfg cw=230 rh=100 nw=180
n rd 0 0 "slot read" "正在被 ldmatrix 读" k=vio
n nx 1 0 "slot read+1" "已到齐，下一圈用" k=pass
n wr 2 0 "slot write" "cp.async 正在写" k=teal
n rg 0 1 "寄存器" "fragment 双缓冲" k=acc
n g 2 1 "全局 A, B" "k_tile_next" k=dark
e rd -> rg "ldmatrix"
e g -> wr
cap: kb = 0 时发下一级 load（写进 write 那一级）并把 read / write 往前推；kb = KB−1 时 `cp_async_wait<STAGES-2>()` + `__syncthreads()`，确认下一圈要读的那一级到齐。
:::

## 逐步演示：用 trace(5, 3) 走一遍

::: steps k_tiles = 5，STAGES = 3
::: flow
cfg cw=170 rh=80 nw=145
n pf 0 0 "预取" "tile 0、1" k=dark
n w 1 0 "wait(1)" "tile 0 到齐" k=warn
n r0 2 0 "第 0 圈" "load 2，算 0" k=acc
n r12 3 0 "第 1–2 圈" "load 3、4" k=vio
n r34 4 0 "第 3–4 圈" "重复 load 4" k=teal
e pf -> w
e w -> r0
e r0 -> r12
e r12 -> r34
:::
--- step 预取 STAGES − 1 = 2 级 {hl=pf}
`load(0 → slot 0)`、`load(1 → slot 1)`，每次一个 commit group。count 从 5 减到 3，k_next 走到 2。
--- step 等第 0 级 {hl=w}
`cp_async_wait<1>()`：允许最近的 1 组还在飞，所以第 0 组（tile 0）一定到了。`__syncthreads()` 之后读 tile 0 的第一个 k16 进寄存器。
--- step 第 0 圈 {hl=r0}
kb = 0：先发 `load(2 → slot 2)`，再 write = 0、read = 1；然后算 tile 0（fragment 来自 slot 0）。kb = 3：`wait<1>` 确认 tile 1 到齐，切到 slot 1。
--- step 第 1、2 圈 {hl=r12}
第 1 圈 `load(3 → slot 0)`——slot 0 里的 tile 0 刚在第 0 圈用完，第 0 圈末尾的 `__syncthreads()` 保证所有 warp 都读完了；算 tile 1。第 2 圈 `load(4 → slot 1)`，算 tile 2。
--- step 第 3、4 圈：尾巴上的多余 load {hl=r34}
count 已经 ≤ 0，k_next 停在 4，所以又发了两次 `load(4)`（写进 slot 2、slot 0）。它们不会再被读，多余但无害——这正是 `redundant_loads = STAGES − 1`。第 3 圈算 tile 3（slot 0），第 4 圈算 tile 4（slot 1）。一共 5 圈 = k_tiles。
:::

## 从小例子开始：先在草稿纸上手算

| 量 | 算式 | 128×128×64，3 级 |
|---|---|---|
| 每级 SMEM | (BM + BN)·BK·2 B | 32 KB |
| 总 SMEM | × STAGES | 96 KB（5090 每 CTA 上限 99 KB） |
| 稳态在途字节 | (STAGES − 1) × 每级 | 64 KB（远大于每 SM 约 7 KB 的需求） |
| `cp_async_wait` 的参数 | STAGES − 2 | 1 |
| 每个 k-tile 的 ldmatrix.x4 | 每 warp 每 k16：A 4 条（64 行 × 16）+ B 2 条（32 列 × 16） | 每 warp 24 条、每 CTA 192 条 |
| 每个 k-tile 的计算 | 2·128·128·64 / 512 FLOP/clk | 4096 clk ≫ 1500 clk 的访存延迟 |

最后一行说明为什么 2–3 级就够：一个 k-tile 算 4096 个时钟，提前一级就能把 1500 个时钟的延迟完全藏住（[配置决策](../knowledge/config.html) 第 6 步）。

## 上机验证

```
./check.sh f3 -s
DRILL_F3_MIN_TFLOPS=100 ./check.sh f3 -s            # 给 4096³ 加一个速度下限
python3 tools/cu_drill.py run mine/f3_pipelined_gemm.cu 8192 8192 8192
```

本仓库在 5090 上的实测（参考解，10 次平均）：1024³ 时 F3 约 70 TFLOPS、F2 约 47；2048³ 起两者几乎一样（2048³ 约 95、4096³ 约 107、8192³ 约 111 TFLOPS），cuBLAS 在 4096³ 约 218。原因值得想一想：F2 只用 32 KB SMEM、每线程 128 个寄存器（`nvcc --resource-usage`），一个 SM 能同时驻留 2 个 CTA（寄存器先到上限：128 × 256 = 半个寄存器堆），一个 CTA 在等访存时另一个在算——它靠**占用率**隐藏延迟；F3 的 3 级 stage 占 96 KB、每线程 146 个寄存器，一个 SM 只放得下 1 个 CTA，靠**流水线**隐藏延迟。大问题上两种办法效果相当，小问题（CTA 不够填满所有 SM 的驻留位）上只有流水线管用。两者离 cuBLAS 的差距主要在 epilogue（FP32 逐元素写回）、tile 光栅化和更深的软件流水，F5 会和 cuBLAS 并排比较。如果结果错了，最常见的原因：wait 的参数写成了 STAGES − 1（允许的在飞组数太多，读到还没到的数据）；read / write 推进的位置放到了 `gemm` 之后；漏了 kb = KB − 1 处的 `__syncthreads()` 。先用主机部分的 `check_trace` 检查你的循环逻辑。

## 提示

::: hint 1 先在 Python 里把循环跑对
`trace()` 就是把 .cu 里的循环逐行翻译成 Python：同样的 count、k\_next、read、write 变量，同样的分支。测试里给出了 `trace(5, 3)` 的完整期望值。Python 版对了，再把同样的结构写回 .cu。
:::

::: hint 2 思路拆解
.cu 的 TODO 分三段：预取 `for s in 0..STAGES-2` （copy A、copy B、`cp_async_fence()` 、count−−、条件 k\_next++）；`cp_async_wait<STAGES-2>()` + `__syncthreads()` + 把第 0 个 k16 读进 `tXrA(_,_,0)` 、`tXrB(_,_,0)` ；主循环 `while (k_tile_count > -(STAGES-1))` 套 `for kb` ，kb == KB−1 时切 `tXsA_p` 并 wait + sync，每步先 copy `kb+1` 的 fragment，kb == 0 时发 load 并推进 read/write，最后 `gemm(mma, tCrA(_,_,kb), tCrB(_,_,kb), tCrC)` 。
:::

::: hint 3 关键问题的答案
为什么 kb == 0 时就可以覆盖 write 那一级，不怕别的 warp 还在读？

write 那一级就是上一圈刚用完的那一级（`write = read` 之前的 read）。上一圈 kb == KB−1 处有一个

`__syncthreads()` ，在那之后所有 warp 读的都是新的 read 级——旧的那一级已经没人读了。这个 barrier 同时承担了“下一级到齐”和“旧的一级可以覆盖”两件事，这就是为什么一圈只需要一个 barrier。
:::

## 参考解

::: solution solutions/f3_pipelined_gemm.py
{{code: solutions/f3_pipelined_gemm.py}}
:::

::: solution solutions/f3_pipelined_gemm.cu
{{code: solutions/f3_pipelined_gemm.cu}}
:::

## 面试官追问 & 对照源码

- 把 STAGES 改成 2 和 4，`trace` 和 `check_trace` 都还成立吗？在 5090 上各快多少？（[配置决策](../knowledge/config.html) 第 6 步的预测：128×128 时 2 级就基本够。）
- 尾巴上那 STAGES − 1 次多余的 load 能省掉吗？CUTLASS 的 collective mainloop 是怎么处理的（提示：给 copy 加谓词）？
- 这个结构里 producer 和 consumer 是同一批 warp。Hopper 的 warp specialization（D3）把它们拆开，用 mbarrier 代替 `cp_async_wait` + `__syncthreads` （D2）——5090 有 TMA 和 mbarrier，但没有 `setmaxnreg` 和 WGMMA，这种拆分在 5090 上还值吗？
