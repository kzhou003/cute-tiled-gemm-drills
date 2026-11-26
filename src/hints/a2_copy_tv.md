lede: 设计 Tiled GEMM 的第二个问题：128 个线程怎样协作把一个 32×16 的 A tile 从 global memory 搬走。本题从 ThrLayout = (32,4):(4,1) 和 ValLayout = (1,4) 出发，自己推出 (thread, value) → (m, k) 的 Copy TV 映射，和真实 make_tiled_copy 逐项对拍，再数两种连续性：一个 warp 碰到几个 sector、一个线程能发多宽的向量访问。
# A2 · 线程怎样搬数据：从 ThrLayout / ValLayout 推出 Copy TV 映射，数 sector 与向量宽度

## 题面速览

- **背景知识**：先读 [从零设计](../knowledge/from_scratch.html) 的“② 线程怎么搬”。
- **在体系里的位置**：真实 kernel 用 `make_tiled_copy(copy_atom, thr_layout, val_layout)`（[[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|copy_atom.hpp]]）。它内部做的正是本题：`raked_product(thr, val)` 得到 (m,k) → (thread, value)，再用 `right_inverse` 反过来得到 TV layout。SM80 教程的设置见 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_tn|sgemm_sm80.cu]]（16×8 个线程，每线程 8 个 half，128 位 `cp.async`）。
- **你要实现**：`thread_coord`、`copy_coords`、`tv_index_map`、`byte_addr`、`vector_elems`、`copy_stats`、`rank_variants`。
- **关键规则**：
  1. 线程布局是“线程坐标 → 线程号”；搬运需要它的逆“线程号 → 线程坐标”，所以必须是双射。
  2. raked：线程 t 的第 v 个值是 `(tm·VM + v % VM, tk·VK + v // VM)`，值在内、线程在外，**值按列主序编号**。
  3. TV 映射的第 `i = t + T·v` 项是 tile 的列主序下标 `m + 32·k`：这就是 CuTe `get_layoutS_TV()` 的完整取值表。
  4. 向量宽度由“线程内部的连续性”决定：按 v 的顺序，w 个值地址连续且对齐；sector 数由“线程之间的连续性”决定。

**验收要点**：四种方案的 tile 都是 32×16 · 每个线程的 (m,k) 与真实 `partition_S` 逐项相同 · TV 取值表与真实 `get_layoutS_TV()` 相同 · 线程布局求逆 · 非双射报错 · 每个元素恰好被搬一次 · 文档例子：线程 5 搬 (1,4)..(1,7) · 值按 colex 编号 · 字节地址 · 四种方案的向量宽度 · 原点不对齐时向量变窄、sector 变多 · 两种连续性的统计 · 方案排名。

::: callout info 第一次出现的术语
- **TV layout**（Thread-Value layout）：把 (线程号, 线程内第几个值) 映射到 tile 坐标的 layout。Copy 的 TV layout 回答“谁搬哪个元素”。
- **ThrLayout / ValLayout**：线程在 tile 上怎么排、每个线程负责的小块形状。两者的乘积覆盖整个 tile。
- **sector**：global memory 访问的最小单位，32 字节。一个 warp 的一条访存指令碰到的 sector 越少，需要的内存事务越少。
- **向量化访问**：一个线程用一条指令读写 8 或 16 字节（例如 `LDG.64`、`LDG.128`），前提是这些字节连续且对齐。
:::

## 直觉与类比

::: callout analogy 搬家公司分工
32×16 的 tile 是 512 箱书，128 个搬运工每人搬 4 箱。“线程之间的连续性”像是一辆货车（一个 sector）能不能一次装下好几个人搬的箱子：大家搬同一排相邻的箱子，一辆车就够；大家各搬一排，就要很多辆车。“线程内部的连续性”像是一个人一次能抱几箱：4 箱挨着放，一次抱走（向量访问）；4 箱分散在不同排，只能跑 4 趟（4 条指令）。
:::

## 图解：从两个小 layout 到一张 TV 表

::: flow make_tiled_copy 的推导
cfg cw=230 rh=96 nw=180
n thr 0 0 "ThrLayout" "(32,4):(4,1)" k=acc
n inv 1 0 "线程号 → 坐标" "求逆：(t//4, t%4)" k=vio
n val 0 1 "ValLayout" "(1,4)" k=acc
n rk 1 1 "raked 组合" "值在内、线程在外" k=vio
n tv 2 0.5 "Copy TV 映射" "(t, v) → (m, k)" k=teal
n st 3 0.5 "两种连续性" "sector · 向量宽度" k=pass
e thr -> inv
e val -> rk
e inv -> tv
e rk -> tv
e tv -> st
cap: CuTe 里同一件事是 `layout_tv = right_inverse(raked_product(thr, val))`（[[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|make_tiled_copy]]）。Track B 会用代数重写这一步，这里先用整数推导。
:::

## 从小例子开始：先在草稿纸上手算

先别管 128 个线程、32×16 的 tile。拿一个小到能把每个格子都写出来的例子手算一遍：8 个线程、每个线程 2 个值、一个 4×4 的 tile。把“谁搬哪个格子”画清楚之后，再放大到测试的运行例子，再看会让结果变差的反例，最后写成一般公式。

① 小：ThrLayout (4,2):(2,1)，ValLayout (1,2)

8 个线程 × 2 个值 = 4×4 的 tile，16 个格子全部能写出来。

```
线程 5 → (tm,tk) = (2,1) → 搬 (2,2) (2,3)
tv_index_map = [0,8,1,9,2,10,3,11,
4,12,5,13,6,14,7,15]
vector_elems = 2，1 条指令，4 个 sector
```

② 运行例子：base (32,4):(4,1)，(1,4)

测试里的 128 个线程、32×16 tile。规则和 ① 完全一样，只是数字变大。

```
线程 5 → (1,1) → 搬 (1,4) (1,5) (1,6)
(1,7)
vector_elems = 4（8 字节），1 条指令
sectors_per_warp = [8,8,8,8]，效率 1.0
```

③ 反例：只改一个参数

线程 stride 改成 (1,32)、或者把 tile 原点挪到 k0 = 2。

```
colthr：每个 warp 32 个 sector，共 128，效
率 0.25
base, k0=2：w 掉到 2，2 条指令，共 96 个
sector
t16x8_v2x2：v0、v1 上下相邻 → w = 1，4 条指
令
```

④ 一般

把 ①–③ 里反复用到的规则写成公式。

```
(tm,tk) = ThrLayout⁻¹(t)
(m,k)   = (tm·VM + v%VM, tk·VK + v//VM)
TV[t + T·v] = m + TILE_M·k
addr    = ((m0+m)·ld + (k0+k))·2
```

**✎** 草稿纸：ThrLayout (4,2):(2,1)，ValLayout (1,2)，ld = 64，half（2 字节）

```text
1) 线程布局求逆   t = tm·2 + tk·1，把 8 个坐标全部代进去：
        (0,0)→0  (0,1)→1  (1,0)→2  (1,1)→3  (2,0)→4  (2,1)→5  (3,0)→6  (3,1)→7
   0..7 各出现一次 → 是双射。反过来读：thread_coord(t) = (t // 2, t % 2)，例如 t=5 → (2,1) 2) tile 形状 = (TM·VM, TK·VK) = (4·1, 2·2) = (4, 4)；每个线程的小块是 1×2（同一行两个相邻格子）
   线程 5 = (2,1)：v=0 → (2·1 + 0%1, 1·2 + 0//1) = (2,2)
                   v=1 → (2·1 + 1%1, 1·2 + 1//1) = (2,3)
   整张 tile（格子里写 t·v）：
              k=0    k=1    k=2    k=3
        m=0   t0·v0  t0·v1  t1·v0  t1·v1
        m=1   t2·v0  t2·v1  t3·v0  t3·v1
        m=2   t4·v0  t4·v1  t5·v0  t5·v1
        m=3   t6·v0  t6·v1  t7·v0  t7·v1        ← 16 格恰好各一次
3) TV 表：第 i = t + 8·v 项 = m + 4·k（t 变化最快）
   i = 0..7  (v=0)：t0..t7 的 v0 = (0,0)(0,2)(1,0)(1,2)(2,0)(2,2)(3,0)(3,2) → 0, 8, 1, 9, 2, 10, 3, 11
   i = 8..15 (v=1)：t0..t7 的 v1 = (0,1)(0,3)(1,1)(1,3)(2,1)(2,3)(3,1)(3,3) → 4, 12,
5, 13, 6, 14, 7, 15
4) 字节地址 (m·64 + k)·2：线程 5 的 v0 = (2·64 + 2)·2 = 260，v1 = (2·64 + 3)·2 = 262
   vector_elems：w=8、w=4 跳过（V = 2 不能整除）；w=2：260 → 262 相差 2 ✓，260 % 4 = 0 ✓
   8 个线程都满足 → w = 2，vec_bytes = 4，instr_per_thread = 2 // 2 = 1
   换成 k0 = 1：线程 0 的组首 = (0·64 + 1)·2 = 2，2 % 4 ≠ 0 → w=2 失败 → w = 1，2 条指令
5) sector（32 字节一格，sector 号 = byte // 32）：8 个线程都在 warp 0，只有 1 条指令
   第 m 行的 tile 占字节 [128m, 128m + 8) → 都在 sector 4m 里 → {0, 4, 8, 12} = 4 个
   efficiency = tile 字节数 / (sector 数·32) = 32 / 128 = 0.25   ← tile 一行只有 8 字节，每个 sector 只用到 1/4
   换成列主序线程 (4,2):(1,4)：线程 1 = (1,0) 搬 (1,0)(1,1)，格子换了主人，但同一个 warp 碰到的 sector 还是这 4 个
   → 小例子整张 tile 都在同一个 warp 里，看不出“线程之间的连续性”。这要到 ② 的规模才看得出来：
6) 放大到 base：warp 0 = 线程 0..31 = 第 0..7 行，每行 4 个线程 × 8 字节 = 32 字节 = 正好 1个 sector → 8 个
   colthr (1,32)：warp 0 = 线程 0..31 = 第 0..31 行的 k = 0..3，每个 lane 一行 → 32 个
sector，每个只用 8 字节
```

下面这张图可以随意改小参数（tile 每一维不超过 8）。左上是线程布局（格子里是线程号），左下是值布局（格子里是值号 v），右边是 tile：每个格子写着它归哪个线程 t、是这个线程的第几个值 v，粗框是一个线程的小块。底下一行是 `tv_index_map` 的取值表。下方面板打印你的函数对这组参数应该返回的结果。先用默认预设对照草稿纸，再切到“(1,1) 非双射”看

`thread_coord` 为什么要报错。

::: raw
<figure class="flow snap"><div class="flow-head">手算草稿 · 小尺寸 TV 映射</div><img src="../figs/a2/a2_new-01.png" alt="手算草稿 · 小尺寸 TV 映射" loading="lazy" style="width:100%;aspect-ratio:1461/1605"></figure>
:::

## 图解详解：TV 映射、sector 与向量宽度

::: callout info 关于这一节的图
这些图是交互版讲解页的静态快照，显示的是默认参数下的状态。文中提到的按钮、滑块和悬停提示在这里不能操作；图上的数字和下方的文字、表格、黑底面板说的是同一组值，可以直接对照你的函数输出。
:::

这一节把整道题拆成五张可交互的图，每张对应一两个要实现的函数。四种方案都能切换。下方黑底面板会把当前状态逐步代入公式。先看整道题在一个 kernel 里对应什么（伪代码，只保留坐标和地址部分）：

```
t = threadIdx.x                                   # 0..127
tm, tk = thread_coord(thr_shape, thr_stride, t)   # ①  线程布局求逆（图 1）
for g in range(V // w):                           # ③  每个线程发 V/w 条访存指令（w = vector_e
    vs = range(g*w, (g+1)*w)                      #    第 g 条指令搬值 v = g·w .. g·w+w-1
    m, k = tm*VM + vs[0] % VM, tk*VK + vs[0] // VM    # ②  raked 公式 → tile 坐标（图 1、图 2
    a = byte_addr(m, k, m0, k0, ld)               #    一个 warp 的 32 个 a 落在几个 32 字节
    reg[vs] = load_w_halves(A_bytes + a)          #    LDG.64 (w=4) / LDG.32 (w=2) / LDG.1
```

### 图 1 · 线程布局求逆 + raked：`thread_coord` / `copy_coords`

左边是 ThrLayout：格子 (tm, tk) 里写的是 `t = tm·s0 + tk·s1` 。搬运时线程只知道自己的 t，所以要反过来在左图里找到写着 t 的格子，得到 (tm, tk)。右边是 32×16 的 tile，每个格子写着搬它的线程号，颜色表示 warp（t // 32）。拖动 t 或者点播放，看高亮的 V 个格子（写着 v0..v3）怎样在 tile 上移动：

- base：t 每加 1，向右挪 4 列；每 4 个线程换一行。一个 warp（同一种颜色）是 8 整行。
- colthr：t 每加 1，向下挪 1 行；一个 warp 是第 0 列块的 32 行。每个线程搬的东西和 base 一样宽，只是邻居换了方向。
- t16x8\_v2x2：每个线程是 2×2 小块，v0、v1 上下相邻（值按列主序编号）。t8x16\_v4x1：每个线程是一列上的 4 个格子。

::: raw
<figure class="flow snap"><div class="flow-head">32×16 tile 上的 Copy TV 映射</div><img src="../figs/a2/a2_new-02.png" alt="32×16 tile 上的 Copy TV 映射" loading="lazy" style="width:100%;aspect-ratio:1461/1681"></figure>
:::

对照源码：`make_tiled_copy` 先做 `raked_product(thr_layout, val_layout)` （得到 (m,k) → (thread, value)），再 `right_inverse` 得到 (thread, value) → (m,k)，见 [[cutlass:include/cute/atom/copy_atom.hpp#L490-L516|make_tiled_copy]] ；`raked_product` 本身在 [[cutlass:include/cute/layout.hpp#L1784-L1800|layout.hpp]] ，`right_inverse` 在 [[cutlass:include/cute/layout.hpp#L1305|layout.hpp]] 。每个线程拿到自己那份的是 [[cutlass:include/cute/atom/copy_atom.hpp#L365-L373|partition_S]] 。CuTe 官方教程里一个完全同构的例子（`thr_layout (32,8)` 、`val_layout (4,1)` ）在 [[cutlass:examples/cute/tutorial/tiled_copy.cu#L182-L209|tiled_copy.cu]] 。

### 图 2 · TV 表的下标顺序：`tv_index_map`

`copy_coords` 按“线程在外、值在内”列出坐标（t-major），而 CuTe 的 `get_layoutS_TV()` 是一个形状为 (T, V) 的

layout，它的一维取值表按 colex 顺序排列：t 变化最快。所以第 i 项是 `t = i % 128, v = i // 128` 。下面把这张表画成 V 行 × 128 列（一行一个 v），一次看 16 个线程。格子里是 `m + 32·k` （tile 的列主序下标），小字是 (m,k)。注意 i 是沿一行增长的，读完 128 个线程的 v0 才轮到 v1。

::: raw
<figure class="flow snap"><div class="flow-head">tv_index_map = get_layoutS_TV() 的取值表</div><img src="../figs/a2/a2_new-03.png" alt="tv_index_map = get_layoutS_TV() 的取值表" loading="lazy" style="width:100%;aspect-ratio:1461/909"></figure>
:::

源码对照：`get_layoutS_TV()` 的定义在 [[cutlass:include/cute/atom/copy_atom.hpp#L318-L326|copy_atom.hpp]] ；layout 的一维下标按 colex 顺序拆成多维坐标的规则见 [[cutlass:media/docs/cpp/cute/01_layout.md|CuTe 文 档 01_layout.md]]。

### 图 3 · 线程之间的连续性：一个 warp 的一条指令碰几个 sector

这张图画的是 global memory 本身：A 是行主序、ld = 64 个 half，所以 A 的每一行是 128 字节 = 4 个 sector。图里每行只画前 64 字节（2 个 sector，中间的竖虚线是 sector 边界），足够装下 32 字节宽的 tile 行（虚线框，从第 k0 个元素开始）。选一个 warp 和它的第 g 条指令：彩色格子是这条指令里 32 个 lane 实际读的字节（颜色 = lane % 8），橙色框是被碰到的 sector。sector 号 = `byte // 32` ，第 m 行的两个 sector 号是 4m 和 4m+1。

- base：warp 0 只占 8 行，每行正好 1 个 sector 全用满 → 8 个。
- colthr：warp 0 占 32 行，每行只读 8 字节，却要碰 1 个 sector → 32 个，每个只用 1/4。
- t8x16\_v4x1：每条指令只读 2 字节，但 32 个 lane 拼成 2 整行 → 每条指令 2 个 sector，4 条共 8 个。
- k0 = 2：tile 行跨过 sector 边界，每行要碰 2 个 sector；而且 w 从 4 掉到 2，指令数翻倍。

::: raw
<figure class="flow snap"><div class="flow-head">一个 warp、一条指令的 sector 计数</div><img src="../figs/a2/a2_new-04.png" alt="一个 warp、一条指令的 sector 计数" loading="lazy" style="width:100%;aspect-ratio:1461/1703"></figure>
:::

为什么数 sector：一条 warp 级的 global 访存指令，硬件按 32 字节 sector 为单位发出内存事务，碰到的 sector 越少，浪费的带宽越少。见 [CUDA C++ Best Practices Guide · Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) 和 [CUDA C++ Programming Guide · Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)。本题的 `copy_stats` 只数“碰到了几个 sector”，是这个模型最简单的版本。

### 图 4 · 线程内部的连续性：`vector_elems`

向量宽度看的是一个线程按 v 的顺序排出的 V 个地址。从 w = 8 往下试：先排除 `w·2 > 16` 字节和 V 不能被 w 整除的；剩下的 w 要求每一组 `v ∈ [g·w, (g+1)·w)` 地址逐个相差 2，并且组首地址是 `2w` 的倍数。所有线程都满足才算通过（图里画的是选中的线程，面板会报告第一个失败的线程）。试试：base 的 k0 = 2 时为什么 w = 4 不行；t16x8\_v2x2 的 v0→v1 为什么相差 128。

::: raw
<figure class="flow snap"><div class="flow-head">一个线程的 V 个地址 → 能合并成多宽的访问</div><img src="../figs/a2/a2_new-05.png" alt="一个线程的 V 个地址 → 能合并成多宽的访问" loading="lazy" style="width:100%;aspect-ratio:1461/789"></figure>
:::

对照：真实 kernel 里 w 是由 Copy\_Atom 的类型决定的（例如`UniversalCopy<uint_byte_t<sizeof(Element)·size(val_layout)>>` ，见 [[cutlass:examples/cute/tutorial/tiled_copy.cu#L193-L196|tiled_copy.cu]] ；SM80 的 `cp.async` 128 位见 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu#L368-L373|sgemm_sm80.cu]] ）。本题反过来问“最多能用多宽”。对齐要求来自硬件：向量 load/store 的地址必须是访问宽度的倍数，见 [CUDA C++ Programming Guide · Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)。

### 图 5 · 两种连续性放在一起：`rank_variants`

排名键是 `(instr_per_thread, total_sectors, name)` ：先比指令数（线程内部的连续性），再比 sector 数（线程之间的连续性），最后按名字。在 k0 = 0 时四个方案正好落在四个象限。切换 k0 看排名怎样变化：k0 = 2 时 base 的指令数变 2，但仍然排第一；k0 = 1 时所有方案都只能 w = 1（4 条指令），只剩 sector 数决定名次，base 掉到第 3，t8x16\_v4x1 反而第一。

::: raw
<figure class="flow snap"><div class="flow-head">四种方案的指令数与 sector 数</div><img src="../figs/a2/a2_new-06.png" alt="四种方案的指令数与 sector 数" loading="lazy" style="width:100%;aspect-ratio:1461/801"></figure>
:::

### 速查：全部公式

| 函数 | 公式 / 规则 | base 的值 |
|---|---|---|
| `thread_coord(thr_shape, thr_stride, t)` | 在所有 (tm,tk) 里找 `tm·s0 + tk·s1 = t` ；线程号必须恰好是 0..T−1 各一次，否则 ValueError | t=5 → (1,1)；行主序即 (t//4, t%4) |
| `copy_coords(…)` | 线程 t 在外、v 在内：`(tm·VM + v%VM, tk·VK + v//VM)` ；tile = (TM·VM, TK·VK) | t=5 → (1,4) (1,5) (1,6) (1,7) |
| `tv_index_map(…)` | 第 i 项：`t = i % T, v = i // T` ，取 `m + TILE_M·k` （即 `cs[t·V + v]` ） | [0, 128, 256, 384, 1,…] |
| `byte_addr(m, k, m0, k0, ld, eb)` | `((m0+m)·ld + (k0+k))·eb` | (1,4) → 136 |
| `vector_elems(…)` | w ∈ {8,4,2,1}，`w·eb ≤ 16` 、`V % w == 0` ；所有线程每组地址差 eb 且组首 % (w·eb) == 0，取最大的 | 4（k0=2 时 2） |
| `copy_stats(…)` | 每 32 个线程一个 warp；每条指令 g 收集 `[a, a+vec_bytes)` 的 `byte // 32` 去重；efficiency = tile 字节 / (total·32) | [8,8,8,8]，32，1.0 |
| `rank_variants(variants)` | 按 `(instr_per_thread, total_sectors, name)` 升序 | base, colthr,t8x16\_v4x1,t16x8\_v2x2 |

记住两句话：线程之间看“同一条指令里 32 个 lane 的地址挤在几个 sector”（由 ThrLayout 决定邻居方向）；线程内部看“一个线程按 v 顺序的地址是否连续、对齐”（由 ValLayout 的形状和编号顺序、以及 tile 原点决定）。

## 逐步演示

::: steps 同样的 32×16 tile、128 个线程，四种分工
::: flow
cfg cw=210 rh=80 nw=180
n base 0 0 "base" "1 条指令 · 32 sector" k=pass
n col 1 0 "colthr" "1 条指令 · 128 sector" k=warn
n v41 2 0 "t8x16_v4x1" "4 条指令 · 32 sector" k=warn
n v22 3 0 "t16x8_v2x2" "4 条指令 · 64 sector" k=todo
:::
--- step 文档的方案 base {hl=base}
`ThrLayout (32,4):(4,1)`：线程 t 的坐标是 `(t // 4, t % 4)`；`ValLayout (1,4)`：4 个值在同一行。线程 5 搬 (1,4) (1,5) (1,6) (1,7)（真实 CuTe 的 `partition_S` 也是这四个）。每个 warp 32 个线程覆盖 8 整行 × 32 字节 = 8 个 sector；每个线程 8 字节连续 → 1 条指令。4 个 warp 共 32 个 sector，效率 1.0。
--- step 线程列主序 colthr {hl=col}
同样的形状，stride 改成 `(1,32)`：相邻 lane 沿 M 往下走。每个线程仍有 8 字节连续（1 条指令），但一个 warp 的 32 个 lane 落在 32 个不同的行上，每行碰一个 sector：每个 warp 32 个 sector，总共 128 个，效率只有 0.25。**线程内部连续，线程之间不连续。**
--- step 每线程一列 t8x16_v4x1 {hl=v41}
`ThrLayout (8,16):(16,1)`、`ValLayout (4,1)`：每个线程搬一列上的 4 个元素。线程内部不连续，只能 4 条 2 字节指令；但每条指令里整个 warp 覆盖 2 整行，总 sector 数仍是 32。**线程之间连续，线程内部不连续。**
--- step 2×2 小块 t16x8_v2x2 {hl=v22}
`ValLayout (2,2)` 按列主序编号，v0、v1 是上下相邻的两个元素，v0、v2 才在同一行。按值的顺序看没有连续的两个 → 4 条指令；每个 warp 4 条指令共碰 16 个 sector，总共 64 个。排名：`base < colthr < t8x16_v4x1 < t16x8_v2x2`（先比指令数，再比 sector）。
:::

再看一个容易忽略的点：同样是 base 方案，如果 tile 原点 `k0 = 2`（不是 4 的倍数），每组 4 个值的起点只有 4 字节对齐，向量宽度掉到 2，指令数变 2，每个 warp 碰 24 个 sector，总数 96。文档说“每线程 4 个连续 half 是 8 bytes 的连续数据，但最终能否发出某种特定宽度的向量 load/store，还要看地址对齐”，说的就是这种情况。

## 易错点

- **忘了求逆。** ThrLayout 是“坐标 → 线程号”，不是“线程号 → 坐标”。`(32,4):(4,1)` 的线程 5 在 (1,1)，不在 (5,0)。测试 `test_thread_coord_inverts_thread_layout`。
- **值按行主序编号。** CuTe 的值按 colex（列主序）编号。`test_values_are_numbered_colex` 用 t16x8_v2x2 抓这个：v1 是 (1,0)，不是 (0,1)。
- **TV 表的下标顺序。** 第 i 项是 `t = i % T, v = i // T`（线程维最快），不是 t-major。测试 `test_tv_index_map_matches_real_get_layoutS_TV`。
- **只看线程内部连续。** colthr 每个线程 8 字节连续，但 sector 数是 base 的 4 倍。测试 `test_stats_two_kinds_of_contiguity`。
- **忘了对齐。** 连续不等于能发向量指令。测试 `test_misaligned_origin_shrinks_vector`。

## 对照文档

> “这里必须同时检查两种连续性：线程之间的连续性：同一个 warp 的不同 lane 访问的地址是否集中、连续，能否减少 Global Memory transaction。线程内部的连续性：同一个线程负责的多个 value 是否连续、对齐，能否合并成向量化访问。这两个条件不是一回事。……于是我们需要一张表，把‘线程 t 的第 v 个 value’映射到 tile 的矩阵坐标：Copy TV Layout: (t,v) → (m,k)。”

本题的四种方案正好把“两个条件不是一回事”拆成了四个象限：两种都好（base）、只有线程内部好（colthr）、只有线程之间好（t8x16_v4x1）、都不好（t16x8_v2x2）。

## 上机验证

```bash
playground/run.sh playground/a2_copy_tv.cpp     # 真实 CuTe 的 make_tiled_copy
./check.sh oracle a2                            # 重新生成 tests/golden/a2_copy_tv.json
```

本机的真实输出：

```text
tiler     : (_32,_16)
TV layout : ((_4,_32),(_4,_1)):((_128,_1),(_32,_0))
thread   0 : (0,0) (0,1) (0,2) (0,3)
thread   1 : (0,4) (0,5) (0,6) (0,7)
thread   4 : (1,0) (1,1) (1,2) (1,3)
thread   5 : (1,4) (1,5) (1,6) (1,7)
thread 127 : (31,12) (31,13) (31,14) (31,15)
```

读 TV layout：线程号 t 拆成 `(t % 4, t // 4)`，对应列主序下标的 stride `(128, 1)`，即 k 组 ×128（4 列 × 32 行）和 m ×1；值 v 的 stride 是 32（下一列）。所以线程 t 的第 v 个值在下标 `128·(t%4) + t//4 + 32·v`，也就是 `m = t//4, k = 4·(t%4) + v`。golden 里用的是 1 个 half 的 atom，打印成 `((_4,_32),(_1,_4)):((_128,_1),(_0,_32))`：形状不同，取值表完全相同。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
thread_coord((32, 4), (4, 1), 5)               # (1, 1)
copy_coords((32, 4), (4, 1), (1, 4))[20:24]    # [(1, 4), (1, 5), (1, 6), (1, 7)]
copy_stats((32, 4), (1, 32), (1, 4))["total_sectors"]   # 128
copy_stats((32, 4), (4, 1), (1, 4), k0=2)["sectors_per_warp"]   # [24, 24, 24, 24]
rank_variants(VARIANTS)    # ['base', 'colthr', 't8x16_v4x1', 't16x8_v2x2']
```

#### 关键问题

1. ThrLayout 的逆为什么必须存在？什么样的 ThrLayout 不可逆？
2. 为什么一个 warp 的 sector 数是 base 方案的 8，而不是 32？
3. 线程内部有 2 个元素连续（像 t16x8_v2x2 的 v0、v2），为什么还是不能发 4 字节指令？
:::

::: hint 2 思路拆解与常见坑
- `thread_coord`：枚举所有 (tm, tk)，算线程号，存进反查表；检查线程号恰好是 0..T−1 各一次。
- `copy_coords`：两层循环，线程在外、值在内，用 raked 公式。
- `tv_index_map`：先拿到 t-major 的坐标表 `cs`，第 i 项取 `cs[(i % T)·V + i // T]`。
- `vector_elems`：从 w = 8 往下试，检查每个线程每组 w 个值的地址相差 2 字节且组首按 2w 对齐。
- `copy_stats`：每 32 个线程一组，每条指令收集 `[addr, addr + vec_bytes)` 覆盖的 sector 集合。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 搬运时每个线程都要知道自己的坐标，所以必须能从线程号唯一反推坐标。stride 让两个坐标算出同一个线程号（例如 `(4,4):(2,1)`），或者某些线程号没有坐标，都不可逆。
2. 一个 warp 的 32 个线程是 8 行 × 每行 4 个线程，每行 4 × 8 字节 = 32 字节，正好 1 个 sector，所以 8 个；colthr 每个线程独占一行，就是 32 个。
3. 向量访问取的是“按值顺序相邻”的几个值：CuTe 的 copy atom 一次拿 ValLayout 的前 w 个值，v0 和 v1 上下相邻、地址差 128 字节，所以不能合并。想让 v0、v1 同行，ValLayout 要写成 `(2,2):(2,1)`（行主序编号）。
:::
:::

## 参考解

::: solution solutions/a2_copy_tv.py
{{code: solutions/a2_copy_tv.py}}
:::

## 面试官追问 & 对照源码

- `make_tiled_copy` 返回的 TiledCopy 有一个 `Tiler_MN`。它和 ThrLayout、ValLayout 的 shape 是什么关系？（对照 [[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|make_tiled_copy]] 里 `product_each(shape(layout_mn))`。）
- SM80 教程为什么用 `(16,8):(8,1)` 个线程、每线程 8 个 half？这比本题的 (32,4) × (1,4) 好在哪里？（提示：16 字节 `cp.async`，见 [[cutlass:include/cute/arch/copy_sm80.hpp@SM80_CP_ASYNC_CACHEALWAYS|SM80_CP_ASYNC_CACHEALWAYS]]。）
- 如果 A 是 M-major（列主序）存的，ThrLayout 和 ValLayout 该怎样改，才能让两种连续性都成立？
- sector 数和 transaction 数是一回事吗？L1 的 128 字节 cache line 在这里扮演什么角色？
