lede: 写 CUDA 代码时，你写的是“一个线程”的程序；硬件执行时，按 32 个线程一组的 warp 发射指令；warp 固定在 SM 的某个 SMSP 上，CTA 固定在某个 SM 上。这一页把这几层粒度一次讲清楚：SIMT 和 SIMD 的区别、lane 什么时候不满、哪些指令必须整个 warp 一起执行、为什么不能让单个线程调用 `mma` 、一个 SM 能同时发射和驻留多少 warp 与 CTA，以及每条指令去 TensorCore、FP32/INT32 lane 还是别的单元，它们怎样在同一个 SMSP 上同时工作。
# 执行模型：thread、warp、SMSP、CTA 怎样跑起来

## 先看全貌：每一层的粒度

| 粒度 | 规模 | 你在这一层决定什么 | 例子 |
|---|---|---|---|
| 线程内部 | 一个线程一次处理多个数据 | 向量化、ILP | 一次搬 16 B（8 个 BF16）；`HFMA2` 一条算 2 个 |
| 线程（thread） | 1 个 | 数据：读哪个地址、走哪条分支 | 你写的每一行 CUDA C++ |
| warp | 32 个线程 | 执行：硬件发射、计费的单位；集体指令的单位 | `mma.sync` 、`ldmatrix` 、`__shfl_sync` |
| warp group | 4 个 warp | Hopper 的 `wgmma` （5090 上没有） |  |
| CTA | ≤ 1024 个线程 | 协作：SMEM 共享、`__syncthreads()` 、谁搬谁算 | TMA 一个线程发起，搬整个 CTA 要用的 tile |
| cluster | 多个 CTA | DSMEM、TMA multicast（5090 上基本不用） |  |

一句话：写代码以线程为单位，算性能以 warp 为单位，设计分工以 CTA 为单位。

## SIMT：按线程写，按 warp 执行

```
int t = blockIdx.x * blockDim.x + threadIdx.x;
float x = A[t];          // 每个线程读自己的地址
if (x > 0) x = x * 2.f;  // 每个线程可以走自己的分支
C[t] = x;
```

每个线程有自己的寄存器、自己的地址、自己的分支。编译后每一行变成一条 SASS 指令（`LDG` 、`FSETP` 、`FMUL` 、`STG` ），调度员发一次，这条指令就同时覆盖 warp 里的 32 个线程。这叫 SIMT（Single Instruction, Multiple Threads）。

它在执行层面和 SIMD 很像（一条指令、32 条 lane 同时做），但编程方式不同：

|  | SIMD（例如 CPU 的 AVX） | SIMT（GPU 的 warp） |
|---|---|---|
| 怎么写 | 显式写向量指令：一次操作 8 个 float | 写普通的标量代码，每个线程一份 |
| 分支 | 程序员自己算掩码、手动合并结果 | 硬件自动加掩码、自动汇合 |
| 访存 | 通常要求连续地址；gather / scatter 是专门的慢指令 | 每个线程各给一个地址；地址连续时硬件自动合并 |
| 线程状态 | 只有一个控制流 | 每个线程有自己的寄存器；从 Volta 起还有自己的 PC，发散的路径可以交错推进 |

所以：写代码时是 SIMT，硬件执行时像 SIMD。

### 分支：线程可以各干各的，但要轮流执行

```
if (lane < 16)  x = a * b;   // 路径 1
else            x = a + b;   // 路径 2
```

```
第 1 拍：发 FMUL，只有 lane 0–15 生效，16–31 被屏蔽、空着
第 2 拍：发 FADD，只有 lane 16–31 生效，0–15 空着
之后：两组重新汇合，继续一起执行
```

结果是对的，只是两条路径的时间加在了一起——这就是分支发散（divergence）。数据可以细到每个线程，代价按 warp 算。

### lane 什么时候不满

| 原因 | 例子 | 特点 |
|---|---|---|
| 按线程编号分支 | `if (lane < 16)` 、`if (threadIdx.x == 0)` | 写代码时就知道 |
| 按数据分支 | `if (x[i] > 0)` 、softmax 里的 mask | 取决于运行时的数据 |
| 边界处理 | `if (i < N)` ，最后一个 warp 只有一部分落在 N 以内 | 只有边缘的 warp 不满（A1 的谓词） |
| block 大小不是 32 的倍数 | `blockDim = 100` → 4 个 warp，最后一个只有 4 个有效 lane | 这个 warp 从头到尾都不满：block 大小取 32 的倍数 |
| 循环次数不同 | `for (j = 0; j < len[i]; j++)` | 先跑完的线程等慢的线程 |
| 线程提前退出 | 一部分线程先 `return` | 剩下的继续执行，lane 不满 |

还要区分两种浪费：

- lane 不满：这一拍发了指令，但只有一部分 lane 在干活。
- 发射槽空着：这一拍这个 SMSP 根本没有 ready 的 warp（比如都在等内存），一条指令都没发。这种浪费更大，靠更多 warp、多级流水线和 ILP 解决。

用 Nsight Compute 看 Avg. Active Threads Per Warp（32 = 满）和 branch efficiency。

## warp 级集体指令：为什么不能让一个线程单独调用 `mma`

有些指令本身就是 warp 级的，单个线程做不了：

| 指令 | 作用 | 为什么必须整个 warp 一起 |
|---|---|---|
| `mma.sync` | Tensor Core 矩阵乘 | 32 个线程各交一部分 fragment，硬件合起来算一块 16×8×16 |
| `ldmatrix` / `stmatrix` | SMEM ↔ 寄存器 | 32 个 lane 各给一个行地址，数据按 mma 的布局分回各线程 |
| `__shfl_sync` | warp 内交换寄存器 | 本身就是 lane 之间的通信（可以用 mask 指定一部分 lane，但 mask 里的必须都到） |
| `__ballot_sync` 、`__any_sync` | warp 内投票 | 同上 |

```
if (lane == 0)   { mma(...); }   // ✗ 未定义行为：结果错误或卡死
if (warp_id == 0){ mma(...); }   // ✓ warp 0 的 32 个线程一起执行
```

原因有两个：

- 操作数分散在 32 个线程手里：一条 `m16n8k16` 的 A 有 256 个元素，每个线程寄存器里只有 8 个。少一个线程，矩阵就缺一块。
- PTX 的规定：`mma.sync.aligned` 的 `.aligned` 要求 warp 内所有线程执行同一条 `mma` ，`.sync` 要求它们在这里汇合。

可以把它想成一条 SIMD 指令，只是它的“向量寄存器”被拆成 32 份放在 32 个线程手里。CuTe 的 TV layout 就是这座桥：每个线程的代码里调用 `gemm(mma, tCrA, tCrB, tCrC)` ，`tCrA` 是这个线程持有的那一份 fragment，背后执行的是 warp 级的 `mma.sync` 。

Tensor Core 最小的操作单位就是一个 warp 发一条 `mma` 。单个线程当然也能做矩阵乘，但那是在 CUDA core 上用 `FFMA` 一个一个元素算：5090 上比 Tensor Core 慢约一半，数据中心卡上差得多（H100 约 16 倍）。

## 想让线程做不同的事：按 warp 分工

warp 之间走不同分支，没有任何代价：每个 warp 独立调度，不存在“屏蔽 lane”的问题。所以分工的粒度应该是 warp。warp specialization 就是这样做的：

```
int warp_id = threadIdx.x / 32;
if (warp_id == 0) {
    // producer warp：只负责搬数据
    if (cute::elect_one_sync()) {    // warp 里选 1 个线程
        tma_load(...);                // ✓ TMA 本来就是单线程发起
    }
} else {
    // consumer warp：整个 warp 一起算
    ldmatrix(...);  mma(...);         // ✓ 32 个线程都在
}
```

| 操作 | 谁来发起 | 说明 |
|---|---|---|
| 普通算术、load / store（`FFMA` 、`LDG` ） | 每个线程各自 | 可以分支，但按 warp 发射，被屏蔽的 lane 空转 |
| `mma.sync` 、`ldmatrix` 、`stmatrix` | 整个 warp | 不能只让一个线程执行 |
| `wgmma` （Hopper） | 整个 warp group（4 个 warp） | 5090 上没有 |
| TMA（`cp.async.bulk.tensor` ） | 1 个线程 | 一条指令搬整块 tile；常配合 `elect_one_sync()` |
| mbarrier 的 `arrive` /`expect_tx` | 每个线程或 1 个线程，看协议 | D2 里的到达计数就是在数这个 |

一句话：线程决定数据，warp 决定执行。 要让线程做差别很大的事，就把它们分到不同的 warp。

## 执行 vs 驻留：一个 SM 里到底有几个 warp、几个 CTA

说“一个 SM 有多少 warp”时，要先分清两个不同的数：同一时钟在发射指令的，和同时驻留在 SM 上的。

|  | 每个 SMSP | 每个 SM（4 个 SMSP） |
|---|---|---|
| 同一时钟能发射的 | 1 个 warp = 32 线程 | 4 个 warp = 128 线程 |
| 同时驻留的上限（5090，cc 12.0） | 12 个 warp = 384 线程 | 48 个 warp = 1536 线程 |
| 对照：H100 驻留上限 | 16 个 warp | 64 个 warp = 2048 线程 |

- 发射：每个 SMSP 有 1 个 warp scheduler 和 32 条 FP32/INT32 lane，每时钟从本 SMSP 的 ready warp 里挑 1 个发一条指令。4 × 32 = 128 就是“128 个 CUDA core”的来历。
- 驻留：驻留 warp 的寄存器常驻在寄存器堆里，切换 warp 不花时间。某个 warp 在等内存时，调度员下一拍就换另一个 ready 的 warp，所以驻留数比发射数多 12 倍。这就是 latency hiding 的硬件基础。

### 为什么每个 SMSP 最多驻留 12 个 warp

NVIDIA 公开的是每 SM 48 个 warp（cc 12.0 的 1536 线程）；warp 按编号固定分到 4 个 SMSP，所以每个最多 12 个。卡住它的是硬件 warp 槽位：调度器要为每个驻留 warp 保存一份执行状态（PC、分支收敛状态、谓词、barrier 状态、记录哪些寄存器还在等结果的 scoreboard），这些槽位数量是固定的。槽位满了，即使寄存器和 SMEM 还有空，也放不下新的 warp。

为什么 GeForce 卡不做 64 个槽位（以下是推断，NVIDIA 没有公开原因）：

|  | 每 SM 寄存器 | 驻留上限 | 满占用时每线程寄存器 |
|---|---|---|---|
| RTX 5090 / 4090 / 3090 | 64K | 48 warp | 65536 ÷ 1536 ≈ 42 |
| H100 / A100 | 64K | 64 warp | 65536 ÷ 2048 = 32 |

寄存器堆一样大。要真正驻留 64 个 warp，每线程只能用 ≤ 32 个寄存器，大多数正经 kernel 做不到（GEMM 常用 100 个以上）。48 个槽位对应 ≈ 42 个寄存器 / 线程，更接近常见 kernel 能达到的占用率；省下的槽位状态和调度逻辑就是芯片面积。

实际写 kernel 时，48 很少是先碰到的限制：每线程 64 个寄存器时只能驻留 32 个 warp（每 SMSP 8 个），128 个时只有 16 个（每 SMSP 4 个）；SMEM 用满时每 SM 只能放 1 个 CTA。只有寄存器 ≤ 42、SMEM 用得少的 kernel（elementwise、reduction）才会撞上槽位上限。

### warp 固定在一个 SMSP 上，不能跨两个

一个 warp 从创建到结束都固定在同一个 SMSP 上，不跨、不迁移：

- 它的 32 个线程跑在这个 SMSP 的 32 条 lane 上。warp 宽度定成 32，正好就是一个 SMSP 的 lane 数。
- 它的寄存器在这个 SMSP 那份寄存器堆里（16384 个 / SMSP）。寄存器在哪，warp 就只能在哪执行，这是它不能迁移的原因。
- 它只用这个 SMSP 的调度员和这个 SMSP 的那 1 个 Tensor Core。有些单元不到 32 宽（例如每 SMSP 只有 1 个 SFU），一条 warp 指令会分几个时钟发完，但仍在同一个 SMSP 里完成。

想让一条“操作”用满整个 SM，就要 4 个 warp 一起上：5090 上单个 warp 的 `mma.sync` 只能用 1/4 的 Tensor Core 算力，所以每 CTA 至少 4 个 warp。Hopper 的 WGMMA 也是这个思路：一个 warp group = 4 个 warp，各在自己的 SMSP 上，一起发一条 WGMMA，同时用上 4 个 Tensor Core。看起来像跨了 SMSP，实际是 4 个各自固定的 warp 在协作。

### 一个 CTA 只在一个 SM 上，但一个 SM 可以同时驻留多个 CTA

5090 上每 SM 最多 32 个 CTA，实际能放几个取四个上限的最小值：

```
每 SM 驻留 CTA 数 = min( 1536 / 每CTA线程数,
                       65536 / (每CTA线程数 × 每线程寄存器),
                       100 KB / 每CTA的SMEM,
                       32 )
```

| CTA 配置 | 线程限制 | 寄存器限制 | SMEM 限制 | 结果 |
|---|---|---|---|---|
| 128 线程、32 寄存器、0 SMEM（elementwise） | 12 | 16 | ∞ | 12 个（48 warp，占满） |
| 128 线程、72 寄存器、48 KB（[端到端例子](../knowledge/e2e.html) 方案 ④） | 12 | 7 | 2 | 2 个 |
| 256 线程、128 寄存器、96 KB（8192³GEMM） | 6 | 2 | 1 | 1 个 |
| 1024 线程、32 寄存器 | 1 | 2 | — | 1 个（1536 放不下 2 个 1024） |

- 调度不区分 CTA：每个 CTA 的 warp 各自分到 4 个 SMSP 上；调度员在本 SMSP 的所有驻留 warp 里挑 ready 的，同一时钟发射的 4 个 warp 可能来自不同 CTA。一个 CTA 在等内存时，另一个 CTA 的 warp 可以顶上。
- 资源彼此隔离：每个 CTA 有自己的 SMEM 分配，看不到别的 CTA 的（cluster 的 DSMEM 例外，5090 上基本不用）；`__syncthreads()` 只同步本 CTA；寄存器按 warp 分配。
- 生命周期：CTA 一旦开始，就在这个 SM 上跑到结束，不拆、不迁移；结束释放资源后，GigaThread 再往这个 SM 派下一个。
- 取舍：每 SM 放多个小 CTA，延迟容易藏，但 tile 小、复用低；放 1 个大 CTA，复用高，但只能靠多级 stage 和 ILP 藏延迟。GEMM 通常每 SM 1–2 个 CTA。

::: callout key 数量关系一页记住
SM 有 4 个 SMSP → 每时钟最多 4 个 warp（128 线程）在发射 → 同时最多驻留 48 个 warp（1536 线程）、32 个 CTA → 实际驻留数由槽位、寄存器、SMEM、CTA 数四个上限里最紧的那个决定。warp 固定在一个 SMSP，CTA 固定在一个 SM；一个 SMSP 上可以有多个 warp，一个 SM 上可以有多个 CTA。
:::

## 指令去哪个单元：Tensor Core、FP32/INT32 lane 和其他

### Tensor Core 的指令（5090 上就是 `mma.sync` 一族）

| PTX | 典型 SASS | 用途 | 5090 |
|---|---|---|---|
| `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16` | BF16/FP16 矩阵乘加（本套练习的 atom） | 有 |
| `mma.sync … m16n8k8` （tf32/ f16） | `HMMA.1688…` | TF32、小 K 版本 | 有 |
| `mma.sync … m16n8k32` （s8/ u8） | `IMMA.16832…` | INT8 | 有 |
| `mma.sync … .kind::f8f6f4m16n8k32` | FP8/FP6/FP4 专用助记符，以 `cuobjdump` 为准 | FP8 / FP6 / FP4 | 有（CUDA ≥12.8） |
| `mma.sync … .kind::mxf4nvf4.block_scale m16n8k64` | 同上 | 带 scale factor 的 NVFP4 / MXFP4 | 有（`sm_120a` ） |
| `mma.sp.sync …` | `HMMA.SP…` | 2:4 稀疏 | 有 |
| `wmma.mma` （旧的 C++ WMMA API） | 编译成 `HMMA` | 同一个硬件 | 有 |
| `wgmma.mma_async` | `HGMMA` 等 | Hopper 的 warp group 级 MMA | 无 |
| `tcgen05.mma` | `UTC…MMA` | B200 的 MMA，累加器在 TMEM | 无 |

容易混淆的一点：`ldmatrix` （`LDSM` ）和 `stmatrix` （`STSM` ）不是 Tensor Core 指令，而是 LD/ST 单元的访存指令，只是取出来的布局正好是 mma 要的 fragment。

### FP32 / INT32 lane 的指令（每 SMSP 32 条，每 SM 128 条）

| 类别 | SASS 例子 | 在 GEMM 里干什么 |
|---|---|---|
| FP32 算术 | `FFMA` 、`FADD` 、`FMUL` 、`FMNMX` 、`FSETP` 、`FSEL` | epilogue 的 α·acc + β·C、bias、激活；softmax |
| 打包半精度 | `HFMA2` 、`HADD2` 、`HMUL2` （含`.BF16` ） | 一条指令算 2 个 FP16/BF16 |
| 整数乘加 | `IMAD` 、`IMAD.WIDE` | 地址计算（行 × stride + 列） |
| 整数加 / 逻辑 /移位 | `IADD3` 、`LOP3` 、`SHF` 、`LEA` 、`ISETP` 、`IMNMX` | 地址、循环计数、边界谓词；swizzle 的 XOR 就是 `LOP3` |
| 类型转换 | `F2FP.BF16.F32.PACK_AB` 、`F2F` 、`I2F` 、`F2I` | 累加器 FP32 → BF16 输出（部分转换走 XU 管线） |

Nsight Compute 里这一类分成几条管线：`FFMA` 、`IMAD` 在 fma 管线，`IADD3` 、`LOP3` 、`ISETP` 在 alu 管线。白皮书说 Blackwell 的 128 条 lane 都能做 FP32 或 INT32；INT32 的实际吞吐建议自己测。

### 其他单元

| 单元 | SASS | 说明 |
|---|---|---|
| SFU（每 SMSP 1 个） | `MUFU.EX2` 、`RCP` 、`RSQ` 、`LG2` 、`SIN` 、`COS` | 超越函数，吞吐约为 FMA 的 1/8 |
| FP64 | `DFMA` 、`DADD` 、`DMUL` | GeForce 上很少，约 FP32 的 1/64（推断） |
| LD/ST（LSU） | `LDG` / `STG` 、`LDS` / `STS` 、`LDSM` / `STSM` 、`LDGSTS` 、`ATOM` / `RED` | 所有由线程逐个给地址的访存 |
| TMA | `cp.async.bulk.tensor` （SASS 里是 `UTMALDG` 一类） | 单线程发起，独立的搬运硬件执行 |
| 控制 / uniform datapath | `BRA` 、`BAR` 、`S2R` 、`ULDC` 、`UIADD3` | 分支、同步、warp 内统一的标量计算 |

一个 GEMM kernel 里各阶段的指令：

| 阶段 | 主要指令 | 单元 |
|---|---|---|
| 主循环计算 | `HMMA` | Tensor Core |
| 主循环喂数据 | `LDSM` 、`LDGSTS` 或 TMA | LSU / TMA |
| 主循环地址和循环控制 | `IMAD` 、`IADD3` 、`LOP3` 、`ISETP` 、`BAR` | FP32/INT32 lane、ALU |
| epilogue | `FFMA` / `FMUL` 、`F2FP.BF16` 、`STS` / `STG` | FP32 lane、LSU |

速率对比（每 SM 每时钟，5090）：Tensor Core BF16 512 FLOP；FP32 `FFMA` 128 条 × 2 = 256 FLOP；SFU 约 16 个结果。数据中心卡上 Tensor Core 和 FP32 的差距大得多（H100 约 4096 : 256），5090 只有 2 倍，所以 epilogue、softmax 这类 FP32 工作在 5090 上相对没那么容易成为瓶颈。

自己确认：`cuobjdump -sass your.cubin | grep -E "HMMA|IMMA|FFMA|IMAD|LDSM|MUFU"` 数指令条数（C1 就是这样统计的）；Nsight Compute 看 `pipe_tensor` 、`pipe_fma` 、`pipe_alu` 、`pipe_xu` 、`pipe_lsu` 的利用率。

## 同一个 SMSP 里，CUDA core 和 Tensor Core 怎样同时工作

没有静态分配。 真正静态分配的只有两样：warp 落在哪个 SMSP、寄存器分给谁。计算单元是逐条指令动态派发的：

```
每个 SMSP：
  调度员每个时钟挑 1 个 ready 的 warp，发 1 条指令
      │
      ├─ FFMA / IMAD           → fma 管线（32 条 FP32/INT32 lane）
      ├─ IADD3 / LOP3 / ISETP  → alu 管线
      ├─ HMMA（mma.sync）      → tensor 管线（这个 SMSP 的 1 个 Tensor Core）
      ├─ LDSM / LDS / LDGSTS   → LSU（访存）
      └─ MUFU                  → SFU
```

1. 大家共享的是“发射槽”：每个 SMSP 每时钟只能发 1 条指令，发给哪个管线由指令类型决定。
2. 各个管线可以同时忙：管线是流水化的。一条 `HMMA` 发出去后 Tensor Core 要吃约 32 个时钟（5090 上的推算值），这期间调度员继续发 `LDSM` 、`IMAD` 、`FFMA` （同一个 warp 或别的 warp 的都行）。所以 CUDA core 和 Tensor Core 在同一个 SMSP 上是并行工作的。
3. Tensor Core 由本 SMSP 的所有 warp 轮流用：一个 SMSP 只有 1 个 Tensor Core，几个 warp 都发 `HMMA` 时在 tensor 管线里排队、交替执行。

算一下 GEMM 主循环：每个 warp 每个 k16 步发 3 条 `ldmatrix.x4` + 4 条 `HMMA` + 少量地址计算，约 10 个发射槽；Tensor Core 却要算 4 × 32 = 128 个时钟。发射槽很宽裕，瓶颈在 Tensor Core 本身，剩下的时间正好用来发下一步的 `ldmatrix` 和下一级的加载。

要让 Tensor Core 不停，需要足够多互相独立的 `HMMA` ：

- 同一个 warp 内：一个 64×32 的 warp tile 有 16 个 atom，对应 16 组互不依赖的累加器，可以连续发。
- 同一个 SMSP 上的其他 warp：例如 8192³ 的配置里每个 SMSP 有 2 个 warp，一个在等 `ldmatrix` 时另一个顶上来。

反过来，连续几条 `HMMA` 都累加到同一组寄存器上，后一条就得等前一条写完——Tensor Core 有空也用不上，所以累加器要分开放。

::: callout key 这一页记住
1. 写代码按线程（SIMT），执行按 warp：一条指令覆盖 32 条 lane，分支发散时轮流执行。2. `mma` 、`ldmatrix` 是 warp 级集体指令，不能只让一个线程调用；TMA 是单线程发起的。要分工，就按

warp 分。

3. 一个 SM 每时钟最多发 4 个 warp（128 线程），最多驻留 48 个 warp、32 个 CTA；实际驻留数由槽位、寄存

器、SMEM、CTA 数里最紧的那个决定。

4. warp 固定在一个 SMSP，CTA 固定在一个 SM；一个 SM 上可以同时有多个 CTA，调度员不区分 warp 属于哪

个 CTA。

5. 每条指令按类型去 Tensor Core、fma、alu、LSU、SFU；各管线并行，共享的是每 SMSP 每时钟 1 条的发

射槽。
:::
