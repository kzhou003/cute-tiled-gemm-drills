lede: 最后一题把“配置决策”从纸面搬到 GPU 上：同一个模板化的流水线 kernel 实例化成 5 种 tile /stage / warp 配置，再加 split-K（grid 的第三维切 K，FP32 atomicAdd 归约），在三个形状上逐个对拍、计时，并跑 cuBLAS 当基准。上机之前，先用主机部分的代价模型给这些配置排好序；上机之后，用秩相关系数看模型错在哪里。
# F5 · 在 5090 上扫配置：tile、stage、warp 排列、split-K，和 cuBLAS 对比

## 题面速览

- 背景知识：[配置决策](../knowledge/config.html)（9 步流程，尤其第 4 步 split-K 和第 8 步“估时间、留候选、实测”）；[怎样定 CTA / tile / warp 的数量](../knowledge/sizing.html)；上机准备见 [F1](../hints/f1_simt_cute_gemm.html)。
- 给出的部分：`gpu/pipelined_mainloop.cuh` （就是 F3 的主循环，模板化了）、`gpu/tiled_gemm_config.hpp`（`Config<BM,BN,BK,Stages,WarpM,WarpN>` ）、扫描用的 `run<Cfg>()` 和 `main` 。
- 你要写：

1. `mine/f5_config_sweep_splitk.cu` 的三个 `DRILL_TODO` ：① split-K 的子问题（指针偏移 + 更薄的 K）；②归约（S = 1 直接写，S > 1 atomicAdd）；③ cuBLAS 基准（行主序 → 列主序的参数换算）。
2. `mine/f5_config_sweep_splitk.py` ：`smem_bytes` 、`legal` 、`predict_us` 、`rank` 、`best` 、`compare` 。

- 关键规则：

1. 第 z 个 split 处理 K 的 `[z·Kc, (z+1)·Kc)` ：A、B 的指针各往后挪 `z·Kc` 个元素，形状 (M, Kc)、(N, Kc)，leading dimension 仍然是 K。
2. S > 1 时 C 先清零，每个 CTA 对自己的 64 个累加器做 `atomicAdd` ——结果不确定（浮点加法顺序不同），但在 1e-3 的容差内。
3. 模型（docstring 有完整公式）：最忙的 SM 顺序跑 ceil(CTA / 170) 个 CTA；每个 k-tile = 计算 + 150 clk 固定开销；每个 CTA 1500 clk 填充；显存下限；split-K 额外的 (1 + S)·M·N·4 字节。

测试检查：合法性（整除、SMEM）· 预测公式的两个手算值 · 大问题的最优是不切 K、tile 少 K 长时最优比 128×128 不切快 40% 以上 · `compare` 的秩相关 · `.cu` 类型检查 · GPU 上三个形状所有配置对拍通过，并打印“预测 vs 实测”。

## 直觉与类比

::: callout analogy 先排班，再上工，最后复盘
厂长拿到一批订单，先在纸上排班：哪种小组规模、几张工作台、要不要把一单拆给几个小组分头做（split-K）。排完上工，记下每种排法真正花了多久，再和隔壁大厂（cuBLAS）比。最后复盘：纸面排名和实际排名差在哪——是没算中转仓（L2）的拥堵，还是低估了小组之间交接（atomic 归约）的成本。
:::

## 图解：split-K 在 kernel 里只改两处

::: flow split-K 在 kernel 里只改两处
cfg cw=215 rh=90 nw=175
n gr 0 0.5 "grid (M/BM, N/BN, S)" "z = blockIdx.z" k=dark
n sub 1 0.5 "子问题" "A + z·Kc，形状 Kc" k=acc
n ml 2 0.5 "流水线主循环" "和 F3 完全一样" k=vio
n s1 3 0 "S == 1" "copy 写回" k=pass
n s2 3 1 "S > 1" "atomicAdd" k=warn
e gr -> sub
e sub -> ml
e ml -> s1
e ml -> s2
cap: 主循环完全不用改：对它来说，第 z 段 K 就是一个 K = Kc 的普通问题，只是 A、B 的行之间仍然隔着原来的 K。
:::

## 逐步演示：1024 × 1024 × 16384 上的四种做法

::: steps 1024 × 1024 × 16384 上的四种做法
::: flow
cfg cw=180 rh=80 nw=150
n a 0 0 "128×128 S=1" "64 CTA" k=dark
n b 1 0 "128×128 S=2" "128 CTA" k=acc
n c 2 0 "64×64 S=4" "1024 CTA" k=vio
n d 3 0 "实测" "排名对吗？" k=pass
e a -> b
e b -> c
e c -> d
:::
--- step 128×128，不切 K：只用 64 个 SM {hl=a}
64 个 CTA，每个走 256 个 k-tile。模型：256 × (4096 + 150) + 1500 ≈ 109 万 clk ≈ 452 µs。170 个 SM 里 106 个闲着。
--- step 128×128，S = 2 {hl=b}
128 个 CTA，每个 128 个 k-tile：主循环减半到约 226 µs，再加约 7 µs 的清零和原子加。这是模型的第一名。
--- step 64×64，S = 4 {hl=c}
1024 个 CTA，分 7 批（ceil(1024 / 170)）；每个 64 个 k-tile × (1024 + 150) clk。模型几乎和 S = 2 并列——但它没有算 L2 流量：64×64 的 tile 让同一块 A、B 被读的次数翻倍。
--- step 上机看排名 {hl=d}
测试会打印每个配置的“预测 µs / 实测 µs”和 Spearman 秩相关。相关性高，说明模型抓住了 SM 利用率这个主要矛盾；某几个配置排错了，通常是模型没建 L2 带宽、epilogue 或寄存器 spill（128×256 每线程 128 个累加器）。
:::

## 从小例子开始：先在草稿纸上手算

| 问题 | 算式 | 结果 |
|---|---|---|
| 4096³、128×128×64 S=1 的预测 | 1024 CTA → 最忙 SM 7 个；7 × (64 × 4246 + 1500)/ 2410 | ≈ 793.7 µs |
| 1024²×16384、S=2 的预测 | 1 × (128 × 4246 + 1500) / 2410 + 3 × 4 MB / 1.792 GB/ms | ≈ 226.1 + 7.0 µs |
| S = 5 合不合法（K = 16384） | 16384 / 5 不是 64 的倍数 | 不合法（"divisibility"） |
| 128×256×64、2 级的 SMEM | 2 × 384 × 64 × 2 | 96 KB，合法 |
| cuBLAS 的参数 | Cᵀ(N×M) = B·Aᵀ：`gemm(T, N, m=N, n=M, k=K, B, ldb=K, A, lda=K, C, ldc=N)` |  |

## 上机验证

```
./check.sh f5 -s                                            # 三个形状，打印所有配置和 cuBLAS
python3 tools/cu_drill.py run mine/f5_config_sweep_splitk.cu 8192 8192 8192   # 只看最后一行
```

记录三件事：①每个形状上最快的配置是不是模型的前三名；②最好的配置达到 cuBLAS 的多少（cuBLAS 在 5090 上会用 sm\_120 专门调过的 kernel，差距主要来自 epilogue、tile 调度和更深的软件流水）；③split-K 在哪个形状上赢、在哪个形状上输，和 [配置决策](../knowledge/config.html) 4.7 节的“三个禁区”对不对得上。

## 提示

::: hint 1 先把 Python 模型写对
`predict_us` 的每一项都在 docstring 里给了公式，测试里有两个手算值（4096³ S=1、1024²×16384 S=2）。先让这两个数对上，再看 `rank` 的排序是否符合直觉。
:::

::: hint 2 思路拆解
① `int kc = K / splits; int z = blockIdx.z;` ，`make_tensor(make_gmem_ptr(A + size_t(z) * kc), make_shape(M, kc), make_stride(K, Int<1>{}))` ，B 同理。② `if (splits == 1) copy(tCrC, tCgC); else for` `(i…) atomicAdd(&tCgC(i), tCrC(i));` —— `tCgC(i)` 是全局内存里那个 float 的引用，取地址就是 `float*` 。③`cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)` 。
:::

::: hint 3 关键问题的答案
为什么 leading dimension 不能也换成 Kc？

A 在内存里仍然是 M×K 的一整块：第 m 行从 `A + m·K` 开始。第 z 段只是每一行里的 `[z·Kc, (z+1)·Kc)` 这一截。指针挪 `z·Kc` 让每行的起点对上，stride 仍然是 K 才能跳到下一行。如果写成 Kc，第 1 行就会读到第 0 行后半截的数据。
:::

## 参考解

::: solution solutions/f5_config_sweep_splitk.py
{{code: solutions/f5_config_sweep_splitk.py}}
:::

::: solution solutions/f5_config_sweep_splitk.cu
{{code: solutions/f5_config_sweep_splitk.cu}}
:::

## 面试官追问 & 对照源码

- 把 atomicAdd 换成 parallel split-K（每个 split 写自己的 workspace，再起一个归约 kernel），哪个形状上更快、哪个更慢？结果是否变成确定性的？
- 实现一个最简单的 Stream-K：开 170 个 persistent CTA，每个 CTA 领连续的 ceil(T·kt / 170) 个 k-tile 单元。和 split-K 比，最大的实现难点在哪（提示：一个 CTA 可能跨两个输出 tile 的边界）？参考 [[cutlass:include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp|sm90_tile_scheduler_stream_k.hpp]] 。
- 进阶：CuTe DSL（Python）。同样的 Ampere 风格 GEMM 有官方的 Python 写法 [[cutlass:examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py|tensorop_gemm.py]] ，5090 专用的 TMA 版本是 [[cutlass:examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py|blackwell_geforce dense_gemm.py]] 。在 5090 机器上 `pip install nvidia-cutlass-dsl` 后可以直接运行；对照本题，找出它们的 tile、stage、TiledCopy、TiledMMA 分别写在哪里。（这台 Mac 上装不了 CuTe DSL 的编译器库，所以这部分没有放进测试。）
