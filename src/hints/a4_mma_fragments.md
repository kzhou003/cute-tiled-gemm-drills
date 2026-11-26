lede: 第四、第五个映射问题：Tensor Core 要的数据怎样进入线程寄存器、结果由谁持有并继续累加。本题只从 PTX 手册里的 groupID / threadID_in_group 两个公式出发，推出 m16n8k16 的 TV_A、TV_B、TV_C，再扩展成 2×2 个 warp 的 TiledMMA，最后让累加器在“寄存器”里跨 4 个 K 片累加，并和真实 CuTe 的 MMA_Traits、get_layoutX_TV、partition_A/B/C 逐个对拍。
# A4 · MMA fragment：从 PTX 规则推出 TV_A / TV_B / TV_C，扩展到 2×2 warp，在寄存器里累加

## 题面速览

- **背景知识**：[从零设计](../knowledge/from_scratch.html) 的第四、第五个映射问题；指令本身见 PTX ISA 的 [mma.sync m16n8k16 fragment 布局](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。
- **在体系里的位置**：CuTe 把这条指令的“谁持有谁”写成 [[cutlass:include/cute/atom/mma_traits_sm80.hpp@SM80_16x8x16_F32F16F16F32_TN|MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>]] 里的 `ALayout / BLayout / CLayout`；`make_tiled_mma` 按 AtomLayout (2,2,1) 和 `Tile<32,32,16>` 把它扩展到整个 CTA tile（[[cutlass:include/cute/atom/mma_atom.hpp@make_tiled_mma|make_tiled_mma]]）；kernel 里用的是 `partition_A / partition_B / partition_C`（[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu]] 的 `gemm(mma, ...)`）。
- **你要实现**：`group_id`、`thread_in_group`、`ptx_a/b/c`、`atom_tv`、`tiled_tv`、`registers_per_thread`、`fragment_shape`、`holders`、`warp_mma`、`cta_tile_gemm`、`valid_order`。IntTuple 打印函数、`MAPPINGS`、`DEPENDS` 已经给出。
- **关键规则**：
  1. `groupID = lane >> 2`，`threadID_in_group = lane % 4`。A、B、C 的行列公式各不相同，不能共用一张表。
  2. atom 内用列主序偏移：A 为 `m + 16k`，B 为 `n + 8k`，C 为 `m + 16n`；TV 的一维下标是 `t + 32·v`（线程最快）。
  3. TiledMMA：`lane = t % 32`，`wm = (t // 32) % 2`，`wn = t // 64`；B 和 C 在 N 方向每个 warp 重复一次：`v = fv + 4·r`，`n += 8·(wn + 2r)`。
  4. 累加器从第一个 K 片到最后一个 K 片一直留在 (128, 8) 的寄存器数组里，只在最后写回一次。

**验收要点**：groupID 公式与越界报错 · lane 5 的 PTX 坐标 · 三个 atom TV 与真实 `MMA_Traits` 逐项相同 · atom 覆盖整个 tile · 三个 tiled TV 与真实 `get_layoutX_TV` 相同 · 线程 0、5、37、127 的 fragment 与真实 `partition_A/B/C` 相同 · 寄存器数 4/4/8 与真实 fragment shape · A、B 每个元素有 2 个持有者，C 只有 1 个，`holders("C",5,7) == [(23,1)]` · 一条 `warp_mma` 等于 numpy 的 `C + A·Bᵀ` · 32×K 的 CTA tile GEMM 正确且 `mma`、`k_slices`、`acc_stores = 1024` 正确 · 形状错误报错 · 五个映射的依赖顺序。

::: callout info 第一次出现的术语
- **MMA**（Matrix Multiply-Accumulate）：Tensor Core 指令，`D = A·B + C`。**m16n8k16**：一条指令算 16×8 的 C，K 方向 16。
- **lane**：warp 里的线程号 0..31。**fragment**：一个 lane 在寄存器里持有的那几个元素。
- **accumulator**：保存部分和的 C fragment，跨 K 循环一直留在寄存器里。
- **TiledMMA**：多个 MMA atom 按 AtomLayout 排列、再按 Tile 重复，覆盖整个 CTA tile 的 TV 分布。
:::

## 直觉与类比

::: callout analogy 32 人合拼一张拼图
一条 `mma.sync` 是 32 个人（lane）同时拼一张拼图。规则写在说明书（PTX 手册）上：每 4 个人一组（groupID），组里第几个（threadID_in_group）决定你拿哪两列。A 是一张 16×16 的图，每人手里 8 块；B 是 8×16，每人 4 块；结果 C 是 16×8，每人保管 4 块。三张图的分法不同，所以有三张对照表（TV_A、TV_B、TV_C）。4 个小组（warp）排成 2×2 去拼 32×32 的大图时，每个组在 N 方向还要拼两次；做完一轮 K 不交卷，拿着手里的 C 块继续拼下一轮，四轮之后才交卷（写回全局内存）。
:::

## 图解：从 lane 到 tile 坐标

::: flow 一个 (t, v) 怎样变成 C tile 的坐标
cfg cw=240 rh=96 nw=190
n t 0 0.5 "线程 t" "0..127" k=dark
n lane 1 0 "lane = t % 32" "g = lane>>2, q = lane%4" k=acc
n w 1 1 "warp (wm, wn)" "(t//32)%2, t//64" k=vio
n atom 2 0 "atom 坐标" "PTX 行列公式" k=teal
n tile 3 0.5 "tile 坐标" "m+16wm, n+8(wn+2r)" k=pass
e t -> lane
e t -> w
e lane -> atom
e atom -> tile
e w -> tile
cap: 真实 CuTe 的 `thr_layout_vmnk` 是 `(_32,_2,_2,_1):(_1,_32,_64,_0)`：lane 最快，然后是 M 方向的 warp，再是 N 方向的 warp。`v = fv + 4r` 里的 r 是 `Tile<32,32,16>` 带来的 N 方向重复。
:::

## 从小例子开始：先在草稿纸上手算

先别管 128 个线程和 32×32 的 tile。拿几个 lane 手算一遍：只用 `g = lane >> 2` 、`q = lane % 4` 两个数，把它的 C、A、B 元素一个个写出来。规律清楚后，再把同样的计算铺满一个 atom（32 个 lane），最后扩展到 2×2 个 warp 的 tile。你自己推这道题时，草稿纸上也应该是这三步。

① 小：lane 0、1、4、5 的 C fragment 只用 g、q 两个数。C 是 16×8，每个 lane 持有 4 个 float。

```
lane (g,q)  c0    c1    c2    c3
 0   (0,0)  (0,0) (0,1) (8,0) (8,1)
 1   (0,1)  (0,2) (0,3) (8,2) (8,3)
 4   (1,0)  (1,0) (1,1) (9,0) (9,1)
 5   (1,1)  (1,2) (1,3) (9,2) (9,3)
```

② 一个 atom：lane 5 的 A / B / C 全部值

测试 `test_ptx_rules_known_entries` 检查的就是 lane 5。

```
A: (1,2) (1,3) (9,2) (9,3)
   (1,10)(1,11)(9,10)(9,11)
B: (1,2) (1,3) (1,10)(1,11)  (n,k)
C: (1,2) (1,3) (9,2) (9,3)
atom_tv("A")[5+32·2] = 9+16·2 = 41
```

③ 2×2 warp：线程 37

lane 5，wm = 1，wn = 0。atom 坐标平移，N 方向重复一次（r = 0, 1）。

```
r=0（行 +16）：
  (17,2) (17,3) (25,2) (25,3)
r=1（列再 +16）：
  (17,18)(17,19)(25,18)(25,19)
holders("C", 5, 7) = [(23, 1)]
```

④ 一般把 ①–③ 里反复出现的模式写成公式。

```
g = lane>>2   q = lane%4
C: (g + 8·(i>>1), 2q + (i&1))
tile: m += 16·wm
      n += 8·(wn + 2r), v = fv + 4r
TV 一维下标 = t + T·v
```

**✎** 草稿纸：从 lane 号到 fragment 坐标

```text
1) 先拆 lane：g = lane >> 2（第几组，0..7），q = lane % 4（组里第几个，0..3）；反过来 lane = 4g + q
       lane   0  1  2  3 | 4  5  6  7 | ... | 28 29 30 31
       g      0  0  0  0 | 1  1  1  1 | ... |  7  7  7  7
       q      0  1  2  3 | 0  1  2  3 | ... |  0  1  2  3
   → 同一组 4 个 lane 的 g 相同：组号决定“行”，组内编号决定“哪两列”
2) C（16×8，c0..c3）：row = g + 8·(i>>1)，col = 2q + (i&1)。lane 5：g = 1，q = 1
       i=0: row = 1 + 8·0 = 1   col = 2 + 0 = 2   → (1,2)
       i=1: row = 1 + 8·0 = 1   col = 2 + 1 = 3   → (1,3)
       i=2: row = 1 + 8·1 = 9   col = 2 + 0 = 2   → (9,2)
       i=3: row = 1 + 8·1 = 9   col = 2 + 1 = 3   → (9,3)
   → 每个 lane 拿两行（g 和 g+8）× 两列（2q 和 2q+1）的 2×2 小块；32 个 lane × 4 = 128 =
16×8 ✓
3) A（16×16，a0..a7）：row = g + 8·((i>>1)&1)，col = 2q + (i&1) + 8·(i>>2)。lane 5
       i     0      1      2      3      4       5       6       7
       row   1      1      9      9      1       1       9       9
       col   2      3      2      3      10      11      10      11
   → 和 C 一样的 2×2 小块，只是 K 有 16 列，所以 i = 4..7 再向右平移 8 列
   → 寄存器：{a0,a1} 是一个 32 位寄存器（同一行、相邻两个 k），共 4 个 32 位寄存器
4) B（N×K = 8×16，b0..b3，用 (n,k)）：n = g，k = 2q + (i&1) + 8·(i>>1)。lane 5
       i=0: (1,2)   i=1: (1,3)   i=2: (1,10)   i=3: (1,11)
   → B 每个 lane 只有一个 n（= g），沿 K 拿两对；2 个 32 位寄存器
5) atom 的列主序偏移（atom_tv 的值）：A = m + 16k，B = n + 8k，C = m + 16n
       A lane 5：(1,2)→33  (1,3)→49  (9,2)→41  (9,3)→57  (1,10)→161 (1,11)→177
(9,10)→169 (9,11)→185
       放在一维下标 t + 32·v：atom_tv("A")[5 + 32·2] = atom_tv("A")[69] = 41
       对照 CuTe：ALayout(5, 2) = 32·q + 1·g + 16·v0 + 8·v1 + 128·v2 = 32 + 1 + 0 + 8 + 0 = 41 ✓
6) 2×2 warp：线程 37 → lane = 37 % 32 = 5，wm = (37 // 32) % 2 = 1，wn = 37 // 64 = 0
       C：v = fv + 4r。fv=0..3 是 atom 的 c0..c3，r=0..1 是 N 方向的重复
       r=0： (1+16, 2+8·(0+0)) = (17,2)  (17,3)  (25,2)  (25,3)
       r=1： (1+16, 2+8·(0+2)) = (17,18) (17,19) (25,18) (25,19)
       tiled_tv("C")[37 + 128·v] = m + 32n = 81, 113, 89, 121, 593, 625, 601, 633
7) 反过来问：C[5][7] 由谁持有？
       m=5 < 16 → wm = 0，cm = 5；n=7 < 8 → wn + 2r = 0 → wn = 0, r = 0，cn = 7
       cn = 2q + (i&1) = 7 → q = 3, i&1 = 1；cm = g + 8·(i>>1) = 5 → g = 5, i>>1 = 0
       lane = 4·5 + 3 = 23，v = fv = 1，t = 23 + 32·wm + 64·wn = 23   → [(23, 1)]，只有一个
       A[9][2]：wm = 0，g = 1，q = 1，i = 2；但 wn 可以是 0 或 1 → t = 5 或 69 → [(5,2), (69,2)]
```

下面这张图可以任选一个 lane。三个矩阵里写着这个 lane 的寄存器名（a0..a7、b0..b3、c0..c3），浅青色是同一组（同一个 g）的另外 3 个 lane。下方面板把 g、q 代入每一个值的公式，打印出 `ptx_a / ptx_b / ptx_c` 和 `atom_tv` 在这个 lane 上应该返回的东西。先用默认的 lane 5 对照上面的草稿纸，再换成 0、4、31 自己验证。

::: raw
<figure class="flow snap"><div class="flow-head">手算草稿 · 一个 lane 的 A / B / C fragment</div><img src="../figs/a4/a4_new-01.png" alt="手算草稿 · 一个 lane 的 A / B / C fragment" loading="lazy" style="width:100%;aspect-ratio:1461/1164"></figure>
:::

## 图解详解：fragment 映射、2×2 warp 与 K 循环

::: callout info 关于这一节的图
这些图是交互版讲解页的静态快照，显示的是默认参数下的状态。文中提到的按钮、滑块和悬停提示在这里不能操作；图上的数字和下方的文字、表格、黑底面板说的是同一组值，可以直接对照你的函数输出。
:::

这一节把整道题画成五张可交互的图，每张对应一组要实现的函数。点按钮、拖滑块、把鼠标停在格子上，下方的面板会把当前状态逐步代入公式算给你看。

先看全局：整道题就是下面这段伪代码里的“谁持有谁”。测试 `test_cta_tile_gemm_exact_and_counts` 按同样的结构检查结果和计数。

```
g, q = lane >> 2, lane % 4                          # ①  group_id / thread_in_group（图 1）
a_i = ptx_a(lane, i); b_i = ptx_b(lane, i); c_i = ptx_c(lane, i)   # ②  三套不同的 PTX 公式
atom_tv(X)[t + 32·v] = 列主序偏移(ptx_x(t, v))        # ③ 摊平成 TV 列表（图 3）
t → (lane, wm, wn)，v → (fv, r)，平移到 32×32 tile      # ④ tiled_tv / holders（图 4）
acc = zeros(128, 8)                                   # ⑤  cta_tile_gemm：累加器常驻寄存器（图
for ks in range(K // 16):
    for w in range(4):                                # 4 个 warp（GPU 上并行）
        for r in range(2):                            # Tile<32,32,16> 带来的 N 方向重复
            acc[warp w, 4r:4r+4] = warp_mma(A 片段, B 片段, acc[...])
C_tile = scatter(acc, tiled_tv("C"))                  # 只在最后写回一次：1024 次
```

### 图 1 · lane → (groupID, threadID\_in\_group)

32 个 lane 按 `lane = 4g + q` 排成 8 行（g）× 4 列（q）。把鼠标停在一个 lane 上：右边的 C（16×8）里，同一行 g 的 4 个 lane（浅色）一起覆盖 C 的第 g 行和第 g+8 行，q 决定其中哪两列（2q、2q+1）。这就是“groupID 管行，threadID\_in\_group 管列对”的全部含义。

::: raw
<figure class="flow snap"><div class="flow-head">lane 的两个分量</div><img src="../figs/a4/a4_new-02.png" alt="lane 的两个分量" loading="lazy" style="width:100%;aspect-ratio:1461/1170"></figure>
:::

### 图 2 · 三张 fragment 地图：`ptx_a` / `ptx_b` / `ptx_c`

这是 PTX 手册里那三张图的可交互版本：A（16×16）、B（以 N×K 存，8×16）、C（16×8）的每一个格子都写着持有它的 lane（或寄存器名）。颜色默认按“值编号 i 的一对”：A、B 里同色的两格就是同一个 32 位寄存器装的两个 half。

- 悬停任一格子：这个 lane 在三张图里的所有元素同时高亮，其余变暗。对比 A 和 C：A 的左半边（k &lt; 8）的分布和 C 完全一样，右半边是它平移 8 列。
- 把着色切到 `g` ：A、C 里每种颜色占两行（g 和 g+8）；B 里每种颜色占一整行（n = g）。切到 `q` ：每种颜色占相邻两列。
- 用“只看值 i”逐个查看 a0..a7：a0/a1 在上半、a2/a3 在下半、a4..a7 是它们右移 8 列。B 的 b2/b3 是 b0/b1 右移 8 列，C 的 c2/c3 是 c0/c1 下移 8 行。

::: raw
<figure class="flow snap"><div class="flow-head">m16n8k16 的 A / B / C fragment</div><img src="../figs/a4/a4_new-03.png" alt="m16n8k16 的 A / B / C fragment" loading="lazy" style="width:100%;aspect-ratio:1461/1066"></figure>
:::

源码对照：PTX 的原始规则在 [PTX ISA · Matrix Fragments for mma.m16n8k16 with floating point type](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)（A、B、C 三张图和 `groupID` / `threadID_in_group` 公式）。CuTe 里这条指令的寄存器个数写在 [[cutlass:include/cute/arch/mma_sm80.hpp#L158-L163|SM80_16x8x16_F32F16F16F32_TN]] ：

`ARegisters = uint32_t[4]` 、`BRegisters = uint32_t[2]` 、`CRegisters = float[4]` ，正好对应 8 个 half、4 个 half、4 个 float。

### 图 3 · 摊平成 TV 列表：`atom_tv`

`atom_tv(which)` 是一张 32 列（t）× V 行（v）的表：格子 (t, v) 里写 列主序偏移`(ptx_x(t, v))` ，放在一维下标 `t +` `32·v` （线程变化最快）。读法：一行是“所有 lane 的同一个值”，沿一行往右走，偏移先按 q 跳（A、C 每次 +32，B 每次 +16），每 4 个 lane 回头再 +1（下一个 g）。悬停格子，右侧小图标出它在矩阵里的位置，面板给出 CuTe `ALayout / BLayout / CLayout` 的分解：

::: raw
<figure class="flow snap"><div class="flow-head">atom_tv(which)：(t, v) → 偏移</div><img src="../figs/a4/a4_new-04.png" alt="atom_tv(which)：(t, v) → 偏移" loading="lazy" style="width:100%;aspect-ratio:1461/1112"></figure>
:::

`(_32,_1)` 就是 `(q, g) → 32q + g` 。CuTe 官方文档 [[cutlass:media/docs/cpp/cute/0t_mma_atom.md#L143-L198|0t_mma_atom.md 的 Traits]]一[[cutlass:media/docs/cpp/cute/0t_mma_atom.md#L143-L198|节]] 讲了 `ThrID / ALayout /`源码对照：三个 layout 的定义在 [[cutlass:include/cute/atom/mma_traits_sm80.hpp#L43-L55|mma_traits_sm80.hpp 的 SM80_16x8_Row 等]] 和

[[cutlass:include/cute/atom/mma_traits_sm80.hpp#L76-L92|MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>]] （F32 累加版本 [[cutlass:include/cute/atom/mma_traits_sm80.hpp#L108-L116|直接继承它]]，只改了 ValType）。线程模式 `(_4,_8):` `BLayout / CLayout` 的含义。

### 图 4 · 2×2 个 warp 拼成 32×32：`tiled_tv` / `holders`

128 个线程：`lane = t % 32` ，`wm = (t // 32) % 2` ，`wn = t // 64` 。C 的 32×32 被切成 2（M）× 4（N）个 16×8 的 atom 块：warp (wm, wn) 拿第 wm 行、第 `wn` 和 `wn + 2` 列（r = 0, 1）的两块。拖动 t，看一个线程的 8 个值（格子里的数字是 v）分别落在哪里：

- A 只依赖 wm：wn 不同的两个 warp（例如 t = 5 和 t = 69）拿到完全相同的 A 片段。B 只依赖 wn：t = 5 和 t = 37 拿到同样的 B。
- C 每格只有一个持有者。悬停 C 的格子，看 `holders` 怎样把坐标反推回 (t, v)；悬停 A、B 的格子能看到两个持有者。

::: raw
<figure class="flow snap"><div class="flow-head">TiledMMA：AtomLayout (2,2,1) · Tile&lt;32,32,16></div><img src="../figs/a4/a4_new-05.png" alt="TiledMMA：AtomLayout (2,2,1) · Tile&lt;32,32,16>" loading="lazy" style="width:100%;aspect-ratio:1461/1424"></figure>
:::

kernel 里怎样用：[[cutlass:examples/cute/tutorial/sgemm_sm80.cu#L298|sgemm_sm80.cu 的 gemm(mma, ...)]] 。CuTe 文档的 [[cutlass:media/docs/cpp/cute/0t_mma_atom.md#L436-L510|TiledMMA]][[cutlass:include/cute/atom/mma_atom.hpp#L397-L434|一]][[cutlass:media/docs/cpp/cute/0t_mma_atom.md#L436-L510|节]] 有同样构造的图。源码对照：[[cutlass:include/cute/atom/mma_atom.hpp#L525-L540|make_tiled_mma]] 把 atom 和 `AtomLayout` 、`Tile` 组合起来；[[cutlass:include/cute/atom/mma_atom.hpp#L397-L434|get_layoutC_TV / get_layoutA_TV]] 是本题 `tiled_tv` 的真身：A 版本多了一步 `atile` ，用 stride 0 把 N 方向的 warp 坐标“压掉”，这就是 A 被复制两份的来源。

### 图 5 · 在寄存器里累加：`warp_mma` / `cta_tile_gemm`

测试的例子：A、B 都是 32×64（B 以 N×K 存），K 切成 4 片，每片 16。按参考解的循环顺序 `ks → warp w → r` 一条一条播放 `mma.sync` ：每条只碰 C 的一个 16×8 块（atom 大小），A 用第 wm 块 16 行，B 用第 `wn + 2r` 块 8 行。注意：

- C 的每个块格子上写着“已经累加了几次”，播放完 4 个 K 片后每块都是 4/4；整个过程中 `acc_stores` 一直是 0。
- 只有最后一步“写回”时，128 个线程各写 8 个值：`acc_stores = 1024 = 32×32` ，`mma = 4` 片 `× 4 warp × 2` `= 32` 。

::: raw
<figure class="flow snap"><div class="flow-head">C_tile = A · Bᵀ：32 条 mma.sync，累加器常驻寄存器</div><img src="../figs/a4/a4_new-06.png" alt="C_tile = A · Bᵀ：32 条 mma.sync，累加器常驻寄存器" loading="lazy" style="width:100%;aspect-ratio:1461/1276"></figure>
:::

### 速查：全部公式

| 量 | 公式 | 每 lane /线程的值数 | 32 位寄存器 | CuTe layout |
|---|---|---|---|---|
| `g, q` | `lane >> 2` ，`lane % 4` ；`lane = 4g + q` | — | 线程模式 `(_4,_8)` ：先 q 后 g |  |
| `ptx_a(lane, i)` | (g + 8·((i>>1)&1), 2q + (i&1) + 8·(i>>2)) | 8 half | 4 | `((_4,_8),(_2,_2,_2)): ((_32,_1), (_16,_8,_128))` |
| `ptx_b(lane, i)` (n, k) | (g, 2q + (i&1) + 8·(i>>1)) | 4 half | 2 | `((_4,_8),(_2,_2)): ((_16,_1),(_8,_64))` |
| `ptx_c(lane, i)` | (g + 8·(i>>1), 2q + (i&1)) | 4 float | 4 | `((_4,_8),(_2,_2)): ((_32,_1),(_16,_8))` |
| `atom_tv(X)[t + 32v]` | A: m + 16k　B: n + 8k　C: m + 16n |  |  |  |
| `tiled_tv` 的 t | lane = t % 32，wm = (t // 32) % 2，wn = t // 64（`thr_layout_vmnk = (_32,_2,_2,_1):(_1,_32,_64,_0)` ） |  |  |  |
| `tiled_tv("A" )` | (am + 16·wm, ak)，偏移 m + 32k | 8 | 4 | fragment `((2,2,2),1,1)` ，2 个持有者 |
| `tiled_tv("B" )` | v = fv + 4r：(bn + 8·(wn + 2r),bk)，偏移 n + 32k | 8 | 4 | fragment `((2,2),2,1)` ，2 个持有者 |
| `tiled_tv("C" )` | v = fv + 4r：(cm + 16·wm, cn + 8·(wn + 2r))，偏移 m + 32n | 8 | 8 | fragment `((2,2),1,2)` ，1 个持有者 |
| `cta_tile_gemm` | mma = (K/16)·4·2，k\_slices = K/16，acc\_stores = 128·8 = 1024（只在最后一次） |  |  |  |

抓住两件事就够了：atom 内只有 g、q 和值编号 i 的几个比特（每个比特是“+1 列 / +8 行 / +8 列”之一）；atom 外只是按 warp 坐标和重复次数 r 平移整块。

## 逐步演示

::: steps lane 5、线程 37、元素 C[5][7]
::: flow
cfg cw=185 rh=80 nw=160
n l5 0 0 "lane 5" "g = 1, q = 1" k=acc
n af 1 0 "A fragment" "8 个 half" k=vio
n t37 2 0 "线程 37 的 C" "wm = 1, wn = 0" k=teal
n hold 3 0 "谁持有 C[5][7]" "唯一：(23, 1)" k=pass
n kl 3 1 "K 循环" "4 片，累加在寄存器" k=warn
e l5 -> af
e af -> t37
e t37 -> hold
e hold -> kl
:::
--- step lane 5 {hl=l5}
`group_id(5) = 1`，`thread_in_group(5) = 1`。真实 CuTe 的 `ALayout = ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))`：线程模式 `(4,8):(32,1)` 正好是“q 乘 32（两列 = 32 个列主序偏移）、g 乘 1（行）”。
--- step A fragment {hl=l5,af}
`[ptx_a(5, i) for i in range(8)] = [(1,2), (1,3), (9,2), (9,3), (1,10), (1,11), (9,10), (9,11)]`。CuTe 的 `ALayout(5, 2) = 41`，`41 % 16 = 9`、`41 // 16 = 2`，即 (9, 2)，和第 2 个值一致。每个 lane 8 个 half = 4 个 32 位寄存器。
--- step 扩展到 2×2 warp {hl=af,t37}
线程 37：lane 5、`wm = 1`、`wn = 0`。真实 `partition_C` 给出它的 8 个累加器坐标：`(17,2) (17,3) (25,2) (25,3) (17,18) (17,19) (25,18) (25,19)`。前 4 个是 atom 坐标的行加 16，后 4 个再在 N 方向加 16（r = 1）。
--- step 持有者 {hl=t37,hold}
`holders("C", 5, 7) = [(23, 1)]`：C 的每个元素只有一个持有者。A 不一样：`holders("A", 9, 2) = [(5, 2), (69, 2)]`，N 方向的两个 warp 都要同一行 A；B 同理 `holders("B", 1, 2) = [(5, 0), (37, 0)]`。这就是文档说的“A/B fragment 可能在多个线程间复制”。
--- step 在寄存器里累加 {hl=hold,kl}
`cta_tile_gemm(A, B)`（A、B 都是 32×64）：每个 K 片 4 个 warp 各发 2 条 `mma.sync`，4 个 K 片一共 `mma = 32` 条；累加器 (128, 8) 从头到尾不离开“寄存器”，最后写回 `acc_stores = 1024` 次，恰好是 32×32 个元素各一次。
:::

## 易错点

- **A、B、C 共用一个行列公式。** A 的行是 `g + 8·((i>>1)&1)`，C 的行是 `g + 8·(i>>1)`：值的编号顺序不同。测试 `test_ptx_examples` 逐个比较 lane 5。
- **把 B 当成 K×N 存。** TN 的 B 以 N×K 存放，本题统一用 (n, k)，偏移 `n + 8k`。测试 `test_atom_tv_matches_mma_traits[B]`。
- **以为 C 也有复制。** A、B 有 2 个持有者，C 只有 1 个。测试 `test_replication_counts`、`test_holders_of_a_element_are_replicated_across_n_warps`。
- **TiledMMA 只按 warp 编号平移。** `Tile<32,32,16>` 让 N 方向每个 warp 重复一次，B 和 C 的每线程值数从 4 变成 8。测试 `test_tiled_tv_matches_get_layoutX_TV`、`test_registers_and_fragment_shapes`（C 的 fragment shape 是 `((_2,_2),_1,_2)`）。
- **每个 K 片写回一次 C。** 测试要求 `acc_stores == 1024`，即只在最后写一次。

## 对照文档

> “因此需要分别描述三套映射：TV_A: (t,v) → (m,k)，TV_B: (t,v) → (n,k)，TV_C: (t,v) → (m,n)。A、B、C 的分布不相同……不能随意把 A/B/C 的 TV Layout 当成同一张表。”
>
> “高性能 kernel 不应该每轮都把部分 C 写回 Global Memory……acc_next = MMA(A_tile, B_tile, acc_current)。”

本题把“三套映射”落成了三个可以和真实 CuTe 逐项对拍的函数，“分布不相同”体现在持有者个数（A/B 2 个，C 1 个）；“累加器留在寄存器”体现在 `acc_stores = 1024`。

## 上机验证

```bash
playground/run.sh playground/a4_mma_fragments.cpp   # 真实 CuTe，本机编译运行
./check.sh oracle a4                                # 重新生成 tests/golden/a4_mma_fragments.json
```

本机的真实输出：

```text
atom A TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
atom B TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
atom C TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
tiled A TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
A(9,2) is value 2 of thread 5
C(5,7) is value 1 of thread 23
A(9,2) is value 2 of thread 69
partition_fragment_A shape: ((_2,_2,_2),_1,_1)
```

读 CLayout 的值模式 `(_2,_2):(_16,_8)`：第一个 2 是“列 +1”（列主序偏移 +16），第二个 2 是“行 +8”，和 PTX 的 `col = 2q + (i&1)`、`row = g + 8(i>>1)` 一一对应。tiled A TV 里 N 方向 warp 的 stride 是 `_0`：这就是 A 被两个 warp 复制的代数表达。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
[ptx_c(5, i) for i in range(4)]      # [(1, 2), (1, 3), (9, 2), (9, 3)]
atom_tv("A")[:8]                     # [0, 32, 64, 96, 1, 33, 65, 97]
tiled_tv("C")[37::128]               # [81, 113, 89, 121, 593, 625, 601, 633]
holders("A", 9, 2)                   # [(5, 2), (69, 2)]
registers_per_thread()               # {'A': 4, 'B': 4, 'C': 8}
fragment_shape("B")                  # ((2, 2), 2, 1)
cta_tile_gemm(A, B)[1]               # {'mma': 32, 'k_slices': 4, 'acc_stores': 1024}
```

#### 关键问题

1. 为什么 ALayout 的线程模式是 `(4,8):(32,1)` 而不是 `(32):(1)`？
2. 为什么 A 的元素有两个持有者、C 只有一个？如果 AtomLayout 改成 (4,1,1)，A 和 B 的复制个数会怎样变？
3. `fragment_shape("C") = ((2,2),1,2)` 的三个模式分别是什么？
:::

::: hint 2 思路拆解与常见坑
- 先写 `ptx_a/b/c`：检查 `0 ≤ lane < 32`、`0 ≤ i < V`，直接套公式。
- `atom_tv(which)`：`out[t + 32·v] = 列主序偏移(ptx_x(t, v))`。
- `tiled_tv(which)`：对每个 `t < 128`、`v < 8`，先用 `lane, wm, wn` 和 `fv = v % 4, r = v // 4`（A 不拆 v）求 atom 坐标，再平移到 tile；偏移是 `m + 32·k`（A、B）或 `m + 32·n`（C）。
- `holders`：遍历 `tiled_tv`，收集偏移等于目标的 (t, v)。
- `warp_mma`：把 32 个 lane 的片段按 `ptx_x` 放回 16×16、8×16、16×8 的矩阵，算 `C + A·Bᵀ`，再按 `ptx_c` 拆回 (32, 4)。
- `cta_tile_gemm`：累加器 `acc = zeros((128, 8))`；每个 K 片、每个 warp、每个 r，从 A、B 取出该 warp 的片段，调一次 `warp_mma`，结果写回 acc 的对应 4 列；最后按 `tiled_tv("C")` 写回 C。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 列主序偏移 `m + 16k` 里，lane 的 q 管“两列”（偏移 +32），g 管“行”（偏移 +1）。线程号 `lane = q + 4g` 按 colex 拆成 (q, g)，所以是 `(4,8):(32,1)`：CuTe 不需要分段公式，一个两层的 layout 就把 PTX 规则写完了。
2. AtomLayout (2,2,1) 有两个 N 方向的 warp，它们算的是 C 的不同列，但需要同一行 A，所以 A 被复制 2 份；同理 B 被 M 方向的两个 warp 复制。C 的每个元素只由一个 warp 计算。改成 (4,1,1) 后 A 不再复制、B 被 4 个 warp 复制（当然 tile 形状也要跟着变）。
3. (2,2) 是一条 atom 的 4 个值（列 +1、行 +8），第二个模式 1 是 M 方向的重复次数，第三个模式 2 是 N 方向的重复（`Tile<32,32,16>` 带来的 r）。
:::
:::

## 参考解

::: solution solutions/a4_mma_fragments.py
{{code: solutions/a4_mma_fragments.py}}
:::

## 面试官追问 & 对照源码

- SM80 的路径里 SMEM → 寄存器通常用 `ldmatrix`（`SM75_U32x4_LDSM_N`）。它的 TV layout 和 MMA 的 A fragment 是什么关系？对照 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu]] 的 s2r copy 和 [[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy_A|make_tiled_copy_A]]。
- `get_layoutC_TV` 是怎样由 `thrfrg_C` 和线程编号重排构造的？读 [[cutlass:include/cute/atom/mma_atom.hpp@get_layoutC_TV|mma_atom.hpp]]。
- 如果累加器换成 f16（`SM80_16x8x16_F16F16F16F16_TN`），每线程 C 的寄存器数怎样变？精度上要付出什么？
- 每个线程 8 个 f32 累加器只是 32×32 tile 的情况；tile 放大到 128×128、4 个 warp 时，每线程多少个累加器？寄存器压力在什么时候导致 spill？
