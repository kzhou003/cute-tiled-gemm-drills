lede: 先暂时忘掉 CuTe 的 API。假设我们自己要写一个尽可能快、又不容易出错的 GEMM kernel，只从 GPU 硬件和数据流出发，逐步推导需要哪些映射；最后再回头看，CuTe 为什么会有这些 layout 和代数操作。这一页对应 Track A。
# 从零设计一个 Tiled GEMM：为什么我们会自然地发明这些 Layout

## 设计目标与六个硬件问题

目标：数据尽量少搬；搬运尽量连续、对齐；线程协作符合硬件执行方式；结果留在合适的位置继续累加；最后高效写回。为此要回答六个问题：

| # | 问题 | 类型 |
|---|---|---|
| 1 | Global Memory 怎么读？访问是否连续、对齐，能否合并成少量 transaction？ | 空间 |
| 2 | CTA 怎么切矩阵？哪个 CTA 负责哪块输出？K 方向怎么循环？边界怎么处理？ | 空间 |
| 3 | Shared Memory 怎么摆？有没有 bank conflict？适不适合 TMA 和后续矩阵指令读取？ | 空间 |
| 4 | Register fragment 怎么分？每个线程的寄存器里存 A/B/C 的哪些元素？ | 空间 |
| 5 | Warp / Tensor Core 怎么算？哪个 warp 执行哪个 MMA atom？ | 空间 |
| 6 | 计算和搬运怎样重叠？生产者和消费者怎样同步？后继 kernel 能否提前启动？ | 时间 |

前五个形成空间映射，第六个引入时间、依赖和调度。PDL 属于第六类，不是一种 layout。

## 五个空间映射

| 硬件问题 | 要设计的映射 | CuTe 里的概念 | 练习 |
|---|---|---|---|
| CTA 负责哪一块？ | 全局坐标 → 当前 CTA 的局部 tile | `local_tile` | A1 |
| 哪个线程从 GMEM 搬哪些元素？ | (thread, value) → tile 坐标 | Copy TV layout、TiledCopy | A2 |
| 元素在 SMEM 放哪里？ | 逻辑坐标 → SMEM 地址 | SMEM layout、Swizzle | A3 |
| 哪个线程的哪个寄存器拿 MMA 的哪个输入？ | (thread, value) → A/B fragment 坐标 | MMA 的 A/B TV layout | A4 |
| 结果由谁持有并继续累加？ | (thread, value) → C 坐标 | MMA 的 C TV layout | A4、A5 |

这五个映射有依赖顺序：先确定 CTA 算哪块，才知道要搬什么；先知道 SMEM 里有什么，才能讨论怎样喂给 MMA；最后才知道 accumulator 怎样更新和写回。

::: steps 沿着数据路径，一层一层推出五个映射
::: flow
cfg cw=170 rh=80 nw=140
n cta 0 0 "① CTA tile" "local_tile" k=acc
n cp 1 0 "② Copy TV" "(t,v) → (m,k)" k=vio
n sm 2 0 "③ SMEM" "(m,k) → 地址" k=teal
n mma 3 0 "④ MMA TV" "(t,v) → (m,k)" k=warn
n acc 4 0 "⑤ 累加器" "(t,v) → (m,n)" k=pass
e cta -> cp
e cp -> sm
e sm -> mma
e mma -> acc
:::
--- step ① CTA 拿哪一块 {hl=cta}
64×64 的 C 按 32×32 切成 4 个 CTA：CTA (0,0) 负责 `C[0:32, 0:32]`，(0,1) 负责 `C[0:32, 32:64]`，依此类推。CTA (0,0) 需要 A 的前 32 行和 B 的前 32 行，但 K = 64 不必一次搬完：每轮 16 个，`A_tile = A[0:32, k:k+16]`，`B_tile = B[0:32, k:k+16]`，k = 0, 16, 32, 48。`local_tile` 只决定当前 CTA 观察哪一块，它不搬数据，也不决定每个线程搬什么。
--- step ② 线程怎么搬 {hl=cta,cp}
一个 32×16 的 half tile 有 512 个元素，128 个线程平均每人 4 个。`ThrLayout = (32,4):(4,1)`、`ValLayout = (1,4)`：每个线程负责同一行上 4 个连续的 K 元素。要同时检查两种连续性：线程之间（一个 warp 的地址是否集中，影响 transaction 数）和线程内部（一个线程的 4 个值是否连续对齐，能否合成向量访问，这里是 8 字节）。
--- step ③ SMEM 怎么摆 {hl=cp,sm}
进入 SMEM 后元素的逻辑身份不变，但物理地址可以重排。直接按二维数组摆，MMA 读取时多个 lane 会落到同一个 bank。`SmemLayout = Swizzle<2,2,2> ∘ (32,16):(16,1)`：先有普通的 shape/stride 地址映射，再复合一个地址变换。swizzle 改的是物理地址，不是矩阵元素的坐标身份，也不是转置。A3 会实测一个值得注意的结论：在这个 32 字节一行的小 tile 上，`Swizzle<2,2,2>` 是合法的一一重排，但并不能消除 bank conflict，还会把 `ldmatrix` 要读的 16 字节行拆开；真正能修好 `ldmatrix` 读取的是 `Swizzle<1,3,3>`。swizzle 的参数必须和行宽、访问宽度一起推导，不能照抄。
--- step ④ Tensor Core 要什么 {hl=sm,mma}
`mma.sync.m16n8k16` 由一个 warp 的 32 个线程协作执行，每个线程在寄存器里持有 A、B、C 的一部分。A、B、C 各有一张 TV layout，三张表不相同。SMEM 到寄存器常用 `ldmatrix`（LDSM）。Copy TV layout 和 MMA TV layout 都长得像 (t,v) → 坐标，但一个描述谁搬数据，一个描述谁持有 MMA 操作数，不能混为一谈。
--- step ⑤ 结果留在寄存器里累加 {hl=mma,acc}
`C = A₀B₀ᵀ + A₁B₁ᵀ + A₂B₂ᵀ + A₃B₃ᵀ`。不应该每轮把部分和写回 GMEM 再读回来；accumulator 留在寄存器里：`acc = MMA(A_tile, B_tile, acc)`，K 全部做完再写回。2×2 个 MMA atom 覆盖 32×32 的 C tile，由 TiledMMA 的 atom 排列决定。
:::

## 回头看：为什么需要 Layout Algebra

到这里我们有五张映射表：Global tensor layout、CTA tile 映射、Copy TV layout、SMEM layout、MMA TV layout。它们不是孤立的：GMEM tile 由 Copy TV 分给线程；copy 的目标坐标映到 SMEM 地址；SMEM 里的数据要转成 MMA 需要的寄存器 fragment；MMA 输出最后还要映回 GMEM 的 C。

如果每一步都手写下标公式，代码脆弱、重复，也很难证明没有漏元素或冲突。CuTe 的做法是把每种分布都表示成 layout，再用代数操作组合、切分、复制、变换这些映射，这就是 [Layout Algebra 与编译](algebra_compile.html) 一页的内容。

| CuTe 操作 | 从 GEMM 角度看它解决什么 |
|---|---|
| `local_tile` | 从全局矩阵选出当前 CTA 的 tile（[[cutlass:include/cute/tensor_impl.hpp@local_tile|local_tile]]） |
| `zipped_divide` / `logical_divide` | 把大 layout 拆成 tile 内坐标和 tile 编号 |
| `logical_product` / `tiled_product` | 把小布局按模式重复铺开 |
| `raked_product` | 交错组合 block 与 tile 维度，常用于线程和值的分布 |
| `composition` | 把地址变换（例如 swizzle）接到另一个 layout 上 |
| `right_inverse` | 从元素坐标反查对应的线程 / value |
| `partition_S/D` | 按 TiledCopy 切出 copy 的源 / 目标分片 |
| `partition_A/B/C` | 按 TiledMMA 切出 MMA 操作数和累加器分片 |

::: callout analogy 搬家公司的五张单子
把一次 GEMM 想成一次大搬家。① 是“这辆车负责哪几个房间”；② 是“每个工人搬哪几箱”（要连续、要顺手）；③ 是“新家仓库里每箱放哪个货架”（避免大家挤到同一个货架前）；④ 是“开箱时每个工人手里拿哪几件”（Tensor Core 规定了拿法）；⑤ 是“组装好的家具先放在谁手里，最后统一摆进房间”。Layout Algebra 就是把这五张单子用同一种表格语言写，并且能互相换算。
:::

## 和练习题的对应关系

| 题目 | 复刻了什么 | 源码 |
|---|---|---|
| A1 | CTA tile、K 循环、边界谓词 | [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@local_tile|sgemm_sm80.cu 的 local_tile]] |
| A2 | Copy TV layout、合并与向量化 | [[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|make_tiled_copy]] |
| A3 | SMEM swizzle 与 bank conflict | [[cutlass:include/cute/swizzle.hpp@Swizzle|Swizzle]] |
| A4 | MMA atom 与 TiledMMA 的 TV layout | [[cutlass:include/cute/atom/mma_traits_sm80.hpp@SM80_16x8x16_F32F16F16F32_TN|MMA_Traits SM80_16x8x16]] |
| A5 | 五个映射串成完整 GEMM | [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu 主循环]] |
