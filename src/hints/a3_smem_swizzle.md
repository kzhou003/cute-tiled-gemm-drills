lede: 设计 Tiled GEMM 的第三个问题：A tile 进了 shared memory 以后放在哪个地址。本题实现文档里的 Swizzle<2,2,2> ∘ (32,16):(16,1)，和真实 CuTe 逐个地址对拍；再用 bank 模型实测三种真实的访问模式。结论有点出乎意料：在这个 32 字节一行的 tile 上，Swizzle<2,2,2> 并没有消除任何冲突，真正有用的是 Swizzle<1,3,3>。
# A3 · Shared memory 怎样摆：行主序、Swizzle<2,2,2> 与 bank conflict 的实测

## 题面速览

- **背景知识**：先读 [从零设计](../knowledge/from_scratch.html) 的“③ SMEM 怎么摆”。
- **在体系里的位置**：真实 kernel 的 SMEM layout 是 `composition(Swizzle<B,M,S>{}, layout)`，例如 SM80 教程用 `Swizzle<3,3,3>` 配 8×64 的 half atom，再 `tile_to_shape` 铺满（[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_tn|sgemm_sm80.cu]]）。`Swizzle` 的定义在 [[cutlass:include/cute/swizzle.hpp@Swizzle|swizzle.hpp]]，它和 layout 的复合在 [[cutlass:include/cute/swizzle_layout.hpp|swizzle_layout.hpp]]。
- **你要实现**：`swizzle`、`smem_offset`、`smem_map`、`is_bijection`、`chunk_contiguous`、`wavefronts`、`copy_store_addrs`、`ldmatrix_addrs`、`lds32_addrs`、`conflict_report`、`find_min_swizzle`。
- **关键规则**：
  1. `swizzle(x) = x ^ ((x & yyy) >> S)`，`yyy = ((1<<B)−1) << (M + max(0,S))`；`|S| < B` 非法。
  2. 先算普通 layout `16·m + k`，再套 swizzle：Swizzle ∘ Layout。
  3. bank 模型：32 个 4 字节 bank，`128 / access_bytes` 个线程一个 phase，同一个 word 是广播。
  4. 一个访问需要的 w 个元素在 swizzle 后不再按顺序连续时，这个 swizzle 对该访问非法（ValueError）。

**验收要点**：普通和 7 种 swizzle 的完整地址表与真实 CuTe 的 `composition(Swizzle, layout)` 相同 · swizzle 的单点值、负 S 与非法参数 · 自逆 · 全部是双射 · M = 2 保住 4 元素块、M = 3 保住 8 元素块 · bank 模型（无冲突、32 路、广播、不对齐报错）· 三种访问的地址 · Sw<2,2,2> 拆散 16 字节行 · 普通与 Sw<2,2,2> 的冲突报告 · Sw<1,3,3> 修好 ldmatrix 与 LDS.32 · 每种访问模式的最小 swizzle · 搜索空间太小时返回 None。

::: callout info 第一次出现的术语
- **SMEM**（shared memory）：SM 上的片上存储，CTA 内共享。**bank**：SMEM 分成 32 个 bank，每个 bank 每周期送出一个 4 字节 word；同一 phase 里两个线程要同一 bank 的不同 word，就要排队（bank conflict）。
- **wavefront**：完成一次访问实际需要的轮数，无冲突时每个 phase 一轮。
- **Swizzle<B,M,S>**：把地址第 [M+S, M+S+B) 位异或到第 [M, M+B) 位上。M 决定“保持连续的最小块”是 2^M 个元素。
- **ldmatrix**：warp 级指令，8 个线程各给一行 16 字节的地址，把 8×8 的 16-bit 矩阵分发到寄存器。**LDS.32**：每个线程读 4 字节的 shared memory 指令。
:::

## 直觉与类比

::: callout analogy 四排储物柜
32 个 bank 像一面 32 列的储物柜，每列一次只能开一个柜门。这个 tile 每行只有 32 字节（8 个柜子），所以 4 行才排满一层；第 0 行和第 4 行正好上下叠在同一列。如果一次要取第 0..7 行同一位置的东西，第 0 行和第 4 行抢同一列，要排两次。`Swizzle<2,2,2>` 只在“同一层的 4 行之间”调换柜子，它们本来就不在同一列，调了也没用；`Swizzle<1,3,3>` 把第 4..7 行整体错开半层，才真的把冲突拆开。
:::

## 图解：地址映射与三种访问

::: flow 从逻辑坐标到 bank
cfg cw=210 rh=96 nw=170
n lc 0 0.5 "逻辑坐标 (m, k)" "32 × 16 half" k=dark
n pl 1 0.5 "普通 layout" "16·m + k" k=acc
n sw 2 0.5 "Swizzle<B,M,S>" "只改地址的位" k=vio
n st 3 0 "拷贝写入" "8 字节 / 线程" k=teal
n ld 3 1 "ldmatrix / LDS.32" "16 字节行 · 4 字节" k=teal
n bk 4 0.5 "bank 模型" "wavefront 数" k=warn
e lc -> pl
e pl -> sw
e sw -> st
e sw -> ld
e st -> bk
e ld -> bk
cap: 元素的逻辑身份不变，只是换了地址；每种访问要求的“连续块”大小不同（8 字节、16 字节、4 字节），所以同一个 swizzle 对不同访问可能合法也可能非法。
:::

## 逐步演示

::: steps 32 字节一行的 tile 上的三个 swizzle
::: flow
cfg cw=205 rh=80 nw=176
n p 0 0 "普通行主序" "ldmatrix 2 路冲突" k=warn
n s222 1 0 "Sw<2,2,2>" "行内 4 块轮换" k=todo
n s133 2 0 "Sw<1,3,3>" "第 4..7 行错开" k=pass
n s224 3 0 "Sw<2,2,4>" "列向写入无冲突" k=teal
:::
--- step 普通行主序 {hl=p}
每行 32 字节，第 r 行和第 r + 4 行相差 128 字节，落在同一组 bank。ldmatrix 读一个 8×8 块的 8 行：2 个 wavefront；MMA A 操作数的 LDS.32（lane l 读第 l//4 行）同样 2 个。A2 的 base 拷贝写入每个 phase 覆盖 4 整行 = 128 字节，没有冲突。
--- step Sw<2,2,2> {hl=s222}
`yyy` 是地址第 4、5 位（行号的低两位），异或到第 2、3 位（4 元素块号）。第 1 行的 4 个块变成 `20..23, 16..19, 28..31, 24..27`（真实 CuTe 的 `sA(1,0) = 20`）。但第 0..3 行本来就在不同 bank，第 r 和 r + 4 行的行号低两位相同、换法也相同，所以 2 路冲突原样保留；而且 M = 2 只保住 4 元素（8 字节）块，ldmatrix 需要的 16 字节行被拆开了，`ldmatrix_addrs` 直接报错。
--- step Sw<1,3,3> {hl=s133}
`yyy` 是地址第 6 位（第 4..7 行），异或到第 3 位：第 4..7 行整体右移 8 个元素（16 字节）。真实 CuTe 的第 0 列：`0 16 32 48 72 88 104 120`。ldmatrix、LDS.32 都变成 1 个 wavefront；M = 3 保住 16 字节行。`find_min_swizzle("ldmatrix") == (1, 3, 3)`。
--- step 换一种拷贝方式 {hl=s224}
如果拷贝改用 A2 的 colthr（lane 沿 M 走），每个 phase 的 16 个线程写同一列的 16 行：普通布局每个 phase 要 4 个 wavefront（总数 32），`Sw<2,2,4>` 把它降到理想的 8。但在本题的搜索范围里，没有一个 swizzle 能同时让 colthr 写入、ldmatrix 和 LDS.32 三者都无冲突：swizzle 要和整条数据路径一起选。
:::

四种 layout 的冲突报告（wavefront 总数，理想值都是 8；None 表示这个 swizzle 拆散了该访问需要的向量）：

| layout | base 写入 | colthr 写入 | ldmatrix | LDS.32 |
|---|---|---|---|---|
| 普通行主序 | 8 | 32 | 16 | 16 |
| Sw<2,2,2>（文档） | 8 | 32 | None | 16 |
| Sw<1,3,3> | 8 | 16 | 8 | 8 |
| Sw<2,2,4> | 8 | 8 | None | 16 |

## 易错点

- **swizzle 放在 layout 前面。** 顺序是 `swizzle(16·m + k)`，先 layout 后 swizzle。`test_swizzled_map_matches_cute_composition` 逐个地址和真实 CuTe 比较。
- **以为 swizzle 改变了元素。** 它只改地址；`test_every_swizzle_is_a_bijection` 确认每个地址恰好用一次。
- **忽略向量粒度。** M = 2 只保证 4 个元素连续；16 字节的 ldmatrix 行需要 M ≥ 3。测试 `test_breaking_a_vector_raises`、`test_chunk_contiguity_depends_on_M`。
- **把 128 字节当成一行。** 这个 tile 一行只有 32 字节，冲突发生在相隔 4 行的地方，不是相隔 1 行。测试 `test_conflict_report_plain`。
- **广播算成冲突。** 同一个 word 被多个线程读只算一次。测试 `test_wavefront_model_basics`。

## 对照文档

> “为什么不直接按普通二维数组摆放？因为 Shared Memory 被划分成 banks；若同一条 warp 指令中的多个 lane 访问发生 bank conflict，访问可能被拆成多个阶段。通过合适的布局或 swizzle，可以改善 bank 访问模式，也能配合后续的矩阵读取路径。SmemLayout = Swizzle ∘ ordinary_layout。示例使用的形式为 Swizzle<2,2,2> ∘ Layout((32,16):(16,1))……Swizzle 改变物理地址排列，不改变矩阵元素在数学上的坐标身份。”

文档说“合适的” swizzle 可以改善访问；本题把“合适”量化了：`Swizzle<2,2,2>` 在这个 tile 上是一个合法的双射，但对三种真实访问都没有改善，ldmatrix 路径需要 `Swizzle<1,3,3>`。swizzle 的参数必须由“行有多宽、访问有多宽、哪些行会撞 bank”推出来，而不是照抄。

## 上机验证

```bash
playground/run.sh playground/a3_smem_swizzle.cpp   # 真实 CuTe 的 composition(Swizzle, layout)
./check.sh oracle a3                               # 重新生成 tests/golden/a3_smem_swizzle.json
```

本机的真实输出（节选）：

```text
Sw<2,2,2> = Sw<2,2,2> o _0 o (_32,_16):(_16,_1)
  row 0:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  row 1:  20  21  22  23  16  17  18  19  28  29  30  31  24  25  26  27
  row 2:  40  41  42  43  44  45  46  47  32  33  34  35  36  37  38  39
  row 3:  60  61  62  63  56  57  58  59  52  53  54  55  48  49  50  51
  bijection: yes
```

第 4 行和第 0 行的排列完全一样（行号低两位都是 0）：这就是 `Sw<2,2,2>` 拆不开第 0、4 行冲突的原因。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
smem_offset(1, 0, (2, 2, 2))                 # 20
smem_offset(5, 4, (2, 2, 2))                 # 80
wavefronts(ldmatrix_addrs(None, 0, 0), 16)   # 2
conflict_report((1, 3, 3))                   # {'store_base': 8, 'store_colthr': 16, 'ldmatrix': 8, 'lds32': 8}
find_min_swizzle("store_colthr")             # (2, 2, 4)
```

#### 关键问题

1. 这个 tile 一行 32 字节。哪两行会落在同一组 bank？
2. `Swizzle<2,2,2>` 用行号的哪几位去打乱地址？这几位对应的行本来是否冲突？
3. 为什么 M 决定了“一个向量访问能不能被保住”？
:::

::: hint 2 思路拆解与常见坑
- `swizzle`：先校验参数，算 `yyy`，`x ^ ((x & yyy) >> S)`（S < 0 时左移）。
- `wavefronts`：按 `128 // access_bytes` 个地址一组；每个地址覆盖 `access_bytes // 4` 个 word；每个 bank 收集不同 word 的集合，取最大集合大小，累加。
- 三个地址函数都先检查需要的 w 个元素在 swizzle 后仍然连续，再返回 `2 × 首元素偏移`。
- `conflict_report`：每个模式包一层 try/except，非法时记 None。
- `find_min_swizzle`：按 B、M、S 三层循环，跳过 `M + S + B > 9` 的候选。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 第 r 行和第 r + 4 行：相差 4 × 32 = 128 字节，正好绕 32 个 bank 一圈。
2. 用地址第 4、5 位，也就是行号的低两位（第 0..3 行之间的区别）。这 4 行在 bank 上本来就错开了 0、8、16、24 个 word，不冲突；冲突发生在第 6 位（相隔 4 行），`Sw<2,2,2>` 没有碰它。
3. swizzle 只改第 M 位及以上的位，低 M 位不动，所以每 2^M 个连续元素在 swizzle 后仍然连续、顺序不变。ldmatrix 一行 8 个 half 需要 2^M ≥ 8，即 M ≥ 3。
:::
:::

## 参考解

::: solution solutions/a3_smem_swizzle.py
{{code: solutions/a3_smem_swizzle.py}}
:::

## 面试官追问 & 对照源码

- SM80 教程用 `Swizzle<3,3,3>` 配 8×64 的 half（每行 128 字节）：按本题的方法推一推，为什么那里是 B = 3、S = 3（[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_tn|sgemm_sm80.cu]]）？
- 如果把 BK 从 16 改成 64（一行 128 字节），`Swizzle<2,2,2>` 会不会变得有用？
- 本题的冲突报告里没有一个 swizzle 能同时让 colthr 写入和 ldmatrix 读都无冲突。真实 kernel 怎么处理“写入和读出偏好不同”的情况？（提示：换拷贝方式，或用 TMA 让硬件按 swizzle 写入，见 Track D。）
- `Swizzle` 是一个自逆的双射。这对“写进去再读出来”有什么好处？
