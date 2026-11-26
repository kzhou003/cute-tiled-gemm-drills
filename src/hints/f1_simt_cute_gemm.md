lede: Track F 的第一题：写一个真正在 GPU 上跑的 CuTe kernel。先不碰 Tensor Core，用`make_tensor` 、`local_tile` 、`local_partition` 、`copy` 、`gemm` 这五个最基本的 API 写一个SIMT（FFMA）的 tiled GEMM，在 RTX 5090 上对拍、计时。主机部分把 kernel 里每次 partition的结果先用整数算出来，并和真实 CuTe 对拍——没有 GPU 也能确认自己理解对了。
# F1 · 第一个真实 CuTe kernel：local_tile + local_partition + SMEM，用 CUDA core 算

## 题面速览

- 背景知识：[从零设计 Tiled GEMM](../knowledge/from_scratch.html) 的 ① CTA tile 和 ② 线程分工；[执行模型](../knowledge/execution.html)（线程按 warp 执行、SMEM 按 CTA 分配）。
- 在体系里的位置：结构照着 CUTLASS 的 [[cutlass:examples/cute/tutorial/sgemm_1.cu#L44-L246|sgemm_1.cu]] （`gemm_device` ）和它的 TN 配置 [[cutlass:examples/cute/tutorial/sgemm_1.cu#L300-L350|gemm_tn]] 。区别：输入是 BF16（搬进 SMEM 时转成 float），C 是行主序 FP32。
- 你要写两个文件：

1. `mine/f1_simt_cute_gemm.cu` ：把三个 `DRILL_TODO` 换成代码——① `local_tile` 切出 gA / gB / gC；②`local_partition` 分出搬运分工和计算分工，建累加器；③ K 循环和写回。
2. `mine/f1_simt_cute_gemm.py` ：`cta_grid` 、`copy_coords` 、`c_coords` 、`a_rows` 、`b_rows` 和三个计数函数。

- 关键规则（.py 的 docstring 有完整版）：

1. CTA tile (128, 128, 8)，256 线程；A、B 都按 K 连续，C 按 N 连续。
2. 搬运布局 `(32,8):(8,1)` ：线程 t 拿 `(t // 8 + 32·i, t % 8)` ，i = 0..3。
3. 计算布局 `(16,16):(1,16)` ：线程 t 累加 `(t % 16 + 16·a, t // 16 + 16·b)` ，a、b = 0..7，列主序。

测试检查：五个线程的 `local_partition` 结果和真实 CuTe 完全一致 · 每个元素恰好被一个线程搬、被一个线程累加 · 计数公式 · `.cu` 在没有 GPU 时通过 clang 的 CUDA 类型检查 · 有 GPU 时两个形状对拍通过。

## 上机准备（Track F 全部题目通用）

| 在哪台机器 | 能跑什么 | 怎么跑 |
|---|---|---|
| 没有 GPU 的机器（例如这台 Mac） | 主机部分的全部测试；`.cu` 的类型检查：用 clang 的 CUDA 前端做 `-fsyntax- only` ，完整实例化 CuTe 模板、检查所有 `static_assert` ，但不生成机器码 | `./check.sh f1` （GPU 测试会显示 skipped） |
| 你的 RTX 5090 机器 | 上面全部，再加 nvcc 编译、对拍、计时 | 装 CUDA ≥ 12.8；`tools/fetch_headers.sh` 取固定版本的 CUTLASS；`./check.sh f1 -s` （`-s` 让 pytest 打印每次运行的 JSON） |

- 单独编译、运行一个文件：`python3 tools/cu_drill.py run mine/f1_simt_cute_gemm.cu 2048 2048 2048` ；只做类型检查：`python3 tools/cu_drill.py check mine/f1_simt_cute_gemm.cu` 。
- 默认 `-arch=sm_120` （RTX 5090）。别的卡用 `CUTE_DRILL_ARCH=sm_86 ./check.sh f1` 之类；F1–F5 只用到 sm\_80 的指令（cp.async、ldmatrix、mma.sync），Ampere 以后的卡都能跑。
- 每个程序最后打印一行 JSON：`ok` （误差 ≤ 容差）、`max_rel_err` 、`ms` 、`tflops` 。参考结果由 `gpu/common.cuh` 里一个“一个线程算一个输出”的朴素 kernel 给出。
- `mine/*.cu` 里没写的地方是 `DRILL_TODO("…")` ：一个必然失败的 `static_assert` ，所以没写完时编译器会把每个 TODO 的说明都列出来。

## 直觉与类比

::: callout analogy 先用最普通的工人把流水线跑通
Tensor Core 是专用冲压机，CUDA core 是普通机床。F1 只用普通机床，但车间的组织方式已经是最终的样子：订单（C）按 128×128 切给小组（CTA）；每轮从仓库搬一小箱料（A、B 各 128×8）上工作台（SMEM），搬的时候 256 个工人按“搬运表”（`ThrA` ）分工；然后每个工人按“加工表”（`ThrC` ）从工作台取料，在自己的工具台（寄存器）上累加 8×8 = 64 个零件。后面四题只是把机床换成冲压机、把一次一箱换成流水线。
:::

## 图解：一个 k-tile 里数据怎么走

::: flow F1 的一个 k-tile（BK = 8）
cfg cw=200 rh=86 nw=165
n g 0 0.5 "全局 A, B" "local_tile → (128,8)" k=dark
n cp 1 0.5 "搬运分工" "ThrA (32,8)" k=acc
n s 2 0.5 "SMEM (float)" "BF16 → float" k=teal
n th 3 0 "计算分工" "ThrC (16,16)" k=vio
n acc 3 1 "累加器 8×8" "64 个 float" k=pass
n c 4 0.5 "全局 C" "K 走完才写" k=dark
e g -> cp
e cp -> s
e s -> th
e th -> acc "FFMA"
e acc -> c
cap: 两个 `__syncthreads()` 分别保证“大家都搬完了才开始算”和“大家都算完了才开始覆盖 SMEM”。
:::

## 逐步演示：kernel 的三段 TODO

::: steps 从全局张量到写回
::: flow
cfg cw=170 rh=80 nw=145
n t 0 0 "全局张量" "make_tensor" k=dark
n lt 1 0 "① CTA tile" "local_tile" k=acc
n lp 2 0 "② 分工" "local_partition" k=teal
n kl 3 0 "③ K 循环" "copy / gemm" k=vio
n wb 4 0 "写回" "copy(tCrC, tCgC)" k=pass
e t -> lt
e lt -> lp
e lp -> kl
e kl -> wb
:::
--- step 全局张量：只是指针加 layout {hl=t}
`make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}))` 不搬任何数据，它只是说“A 是一个 M×K、K 方向 stride 为 1 的张量”。动态的 M、N、K 和静态的 `Int<1>` 可以混用：静态的部分在编译期就被折叠掉（Track C）。
--- step ① local_tile：这个 CTA 看哪一块 {hl=lt}
`local_tile(mA, make_shape(BM, BN, BK), make_coord(blockIdx.x, blockIdx.y, _), Step<_1, X, _1>{})`。`Step<_1, X, _1>` 说“A 只用 tiler 的 M、K 两维”；坐标里的 `_` 说“K 方向不固定，保留成最后一维”。结果 gA 的形状是 (128, 8, K/8)：第三维就是 k-tile 编号。
--- step ② local_partition：每个线程拿哪几个 {hl=lp}
`local_partition(gA, ThrA{}, threadIdx.x)` 把 (128,8) 按 ThrA 的形状 (32,8) 切块，线程 t 在每一块里拿 ThrA⁻¹(t) 那个位置——4 个元素，形状 (4, 1, k)。计算分工用投影：`local_partition(sA, ThrC{}, t, Step<_1, X>{})` 只按 ThrC 的“行”去分 sA，所以线程 t 拿到 8 整行（8 × BK）。
--- step ③ K 循环：搬、等、算、等 {hl=kl}
`copy(tAgA(_, _, kt), tAsA)` 逐元素搬（BF16 → float 的转换就发生在这里），`__syncthreads()`，`gemm(tCsA, tCsB, tCrC)` 做 (8,8) += (8,8)·(8,8)ᵀ 的外积累加，全是 FFMA，再 `__syncthreads()`。
--- step 写回 {hl=wb}
`copy(tCrC, tCgC)`：每个线程把自己的 64 个累加器写到 C 的 64 个位置。它们按 16 的步长散开，所以每个 warp 的一条 store 写的是 32 个不连续的 float——这是 F1 慢的原因之一。
:::

## 从小例子开始：先在草稿纸上手算

只看一个线程：t = 37。

| 问题 | 怎么算 | 结果 |
|---|---|---|
| 在 ThrA 里的坐标 | (37 // 8, 37 % 8) | (4, 5) |
| 搬 A tile 的哪 4 个元素 | (4 + 32i, 5) | (4,5) (36,5) (68,5) (100,5) |
| 在 ThrC 里的坐标 | (37 % 16, 37 // 16) | (5, 2) |
| 累加 C tile 的哪些元素 | (5 + 16a, 2 + 16b) | (5,2) (21,2) … (117,2)，(5,18) … 共 64 个 |
| 计算时读 sA 的哪几行 | 5 + 16a | 5, 21, 37, …, 117 |

再算一个 CTA 的量：每个 k-tile 搬 (128 + 128) × 8 × 2 B = 4 KB，做 2 × 128 × 128 × 8 = 262144 FLOP，强度 64 FLOP/B（和 Tensor Core 版的 128×128 tile 一样——强度只看 BM、BN，见 [BM、BN、BK 各管什么](../knowledge/tile_dims.html)）。

## 上机验证

```
./check.sh f1 -s                                         # 在 5090 上：主机部分 + 类型检查 + nvcc 编译、对拍、计时
python3 tools/cu_drill.py run mine/f1_simt_cute_gemm.cu 4096 4096 4096
```

期望：`ok: true` 。`max_rel_err` 在这里通常正好是 0：`gpu/common.cuh` 把输入取在 1/64 的网格上，乘积和部分和在 FP32 里都是精确的，所以累加顺序不同也不会产生舍入差；换成随机输入时会在 1e-6 量级。速度只有 FP32 峰值（约 105 TFLOPS）的一小部分——每次 FFMA 要从 SMEM 读 0.25 个 float，每个 k-tile 两次 `__syncthreads()` ，还没有任何流水线。记下这个数，F2、F3 之后对比。

## 提示

::: hint 1 先让 shape 对上
在 kernel 开头加 `if (thread0()) { print(gA); print("\n"); print(tAgA); print("\n"); }` （CuTe 的 `print` 在设备上也能用），先看形状：gA 应该是 `(_128,_8,K/8)` ，tAgA 是 `(_4,_1,K/8)` ，tCsA 是 `(_8,_8)` ，tCrC 是 `(_8,_8)` 。主机部分的 `copy_coords` / `c_coords` 就是这些张量里每个元素的坐标。
:::

::: hint 2 思路拆解
① 三次 `local_tile` ，只差 `Step` ：A 是 `Step<_1, X, _1>` ，B 是 `Step<X, _1, _1>` ，C 是 `Step<_1, _1, X>` 。② 搬运用 `local_partition(g, Thr, threadIdx.x)` ；计算用带投影的版本：sA 按 ThrC 的行（`Step<_1, X>` ）、sB 按列（`Step<X, _1>` ）、gC 两维都用（`Step<_1, _1>` ）；累加器 `make_tensor_like(tCgC)` 后 `clear` 。③ 循环次数是 `size<2>(tAgA)` 。
:::

::: hint 3 关键问题的答案
为什么需要两个 \_\_syncthreads？

第一个：所有线程的 copy 都写进 SMEM 之后才能开始算（别的线程搬的数据你也要用）。第二个：所有线程都算完之后，下一轮才能覆盖 SMEM。去掉第二个，快的 warp 会在慢的 warp 还在读 sA 时把它覆盖掉——结果偶尔错、而且和运行次数有关，是最难查的那一类 bug。
:::

## 参考解

::: solution solutions/f1_simt_cute_gemm.py
{{code: solutions/f1_simt_cute_gemm.py}}
:::

::: solution solutions/f1_simt_cute_gemm.cu
{{code: solutions/f1_simt_cute_gemm.cu}}
:::

## 面试官追问 & 对照源码

- 把 ThrC 换成 `(16,16):(16,1)` （行主序）会怎样？`c_coords` 怎么变？写回时一个 warp 的 32 次 store 落在几个 128 B 的 cache line 上？
- sA 存 float（每个 k-tile 4 KB）和存 BF16（2 KB）各有什么好处？如果存 BF16，`gemm` 里每次 FFMA 前要多做什么？
- [[cutlass:examples/cute/tutorial/sgemm_2.cu|sgemm_2.cu]] 把同样的计算改成 `TiledMMA<UniversalFMA>` + `TiledCopy` 。对照它，F1 的哪几行会被换掉？这正是 F2 要做的事。
