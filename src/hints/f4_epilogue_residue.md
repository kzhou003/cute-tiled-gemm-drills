lede: 真实的 GEMM 形状很少正好是 128 的倍数，输出也很少就是 FP32 的累加器。F4 在 F3 的流水线kernel 上加两件事：M、N 不整除时，用 identity tensor 算出每条 cp.async 和每个累加器的全局坐标，越界的加载不发、越界的输出不写；epilogue 做 D = bf16(α·acc + β·C)。主机部分用 NumPy 复刻边界逻辑和 BF16 舍入，先在没有 GPU 的机器上把“哪些关掉、写出来是多少”算清楚。
# F4 · 边界与 epilogue：谓词加载、α·acc + β·C、FP32 → BF16

## 题面速览

- 背景知识：A1 的边界谓词（`elem_less` ）；[执行模型](../knowledge/execution.html) 的“lane 什么时候不满”（边界 warp）；上机准备见 [F1](../hints/f1_simt_cute_gemm.html)。
- 在体系里的位置：谓词拷贝的写法对照 [[cutlass:examples/cute/tutorial/tiled_copy_if.cu|tiled_copy_if.cu]] ；`copy_if` 本身见 [[cutlass:include/cute/algorithm/copy.hpp#L65-L100|copy.hpp]] 。
- 你要写：

1. `mine/f4_epilogue_residue.cu` 的两个 `DRILL_TODO` ：① 加载谓词 `tApA` 、`tBpB` ，并把加载改成 `copy_if` ；② 带边界检查的 epilogue。identity tensor 和它们的 partition（`tAcA` 、`tBcB` 、`tCcC` ）已经给出。
2. `mine/f4_epilogue_residue.py` ：`cta_grid` 、`load_predicates` 、`store_mask` 、`tile_waste` 、`bf16_round` 、`epilogue` 、`max_rel_err` 。

- 关键规则：

1. grid 向上取整；线程 t 的第 i 条 A 加载要发，当且仅当全局行号 `128·bm + t // 8 + 32i < M` 。
2. 越界的 SMEM 不用清零：A 的第 m 行只影响 C 的第 m 行，B 的第 n 行只影响第 n 列。
3. K 仍然要求 64 的倍数：K 方向的越界会进入每一个输出。
4. `D = bf16_round(α·acc + β·C)` ，round-to-nearest-even；β == 0 时不读 C。

测试检查：grid 和浪费比例 · 加载谓词 · 所有 CTA 的 store mask 拼起来正好覆盖 M×N 一次 · BF16 舍入（含 tie-to-even 的边界例子、结果的低 16 位全为 0、NaN）· β == 0 时 C 里是 NaN 也不影响结果 · `.cu` 类型检查 · GPU 上四个形状（含 129×1×64 这种极端的）对拍。

## 直觉与类比

::: callout analogy 订单边上那几件零件
订单是 1000×760 个零件，车间按 128×128 一单来切，最后一行、一列的单子只有一部分是真的。小组照常开工，但搬料时不去仓库拿不存在的料（谓词加载），装箱时只装订单里有的那些（谓词写回）。料台上空着的那几格放的是上一单剩下的废料也没关系——它们只会加工出“不存在的零件”，最后不会被装箱。
:::

## 图解：坐标是怎么来的

::: flow identity tensor 给出每个元素的坐标
cfg cw=210 rh=90 nw=170
n id 0 0.5 "identity (M,K)" "值 = 自己的坐标" k=dark
n lt 1 0.5 "local_tile" "同一个 tiler" k=acc
n ps 2 0 "partition_S" "tAcA：加载坐标" k=teal
n pc 2 1 "partition_C" "tCcC：输出坐标" k=vio
n lp 3 0 "加载谓词" "行 < M" k=warn
n sp 3 1 "写回谓词" "elem_less" k=warn
e id -> lt
e lt -> ps
e lt -> pc
e ps -> lp
e pc -> sp
cap: identity tensor 不占内存（它的“值”是算出来的），所以给它做 local_tile 和 partition 几乎没有代价。
:::

## 逐步演示：一个最后一行的 CTA

::: steps M = 1000，bm = 7（第 8 个、也是最后一个行块）
::: flow
cfg cw=180 rh=80 nw=150
n rg 0 0 "行范围" "896–1023" k=dark
n ld 1 0 "加载" "前 104 行" k=acc
n ac 2 0 "累加" "128 行都算" k=vio
n wb 3 0 "写回" "只写 104 行" k=pass
e rg -> ld
e ld -> ac
e ac -> wb
:::
--- step 这个 CTA 覆盖的行 {hl=rg}
128·7 = 896 到 1023，其中 896–999 共 104 行在界内。
--- step 加载：线程 255 的 4 条 {hl=ld}
行号 896 + 255 // 8 + 32i = 927、959、991、1023 → `[True, True, True, False]`。第 4 条 cp.async 不发，sA 的那一行保持原来的内容。
--- step 累加：照常算 {hl=ac}
mma 不知道哪些行是越界的，128 行全部算。越界的 24 行用的是 sA 里的旧数据，结果是垃圾——但只影响 C 的那 24 行。
--- step 写回：只写 104 行 {hl=wb}
`elem_less(tCcC(i), shape(mC))` 为假的累加器跳过。β ≠ 0 时界内的才读 C。
:::

## 从小例子开始：先在草稿纸上手算

- `bf16_round(1 + 2⁻⁸)` ：1 附近 BF16 的间隔是 2⁻⁷，1 + 2⁻⁸ 正好在 1 和 1 + 2⁻⁷ 中间 → 取尾数为偶数的那个 →

1. 0。`1 + 3·2⁻⁹` 更靠近 1 + 2⁻⁷ → 1.0078125。

- 位运算版本：FP32 的位模式 u，加 `0x7FFF + ((u >> 16) & 1)` ，再清掉低 16 位。`(u >> 16) & 1` 就是“保留下来的最低位”，它决定平局时往哪边走。
- `tile_waste(1000, 760)` ：8 × 6 = 48 个 CTA × 16384 = 786432 个输出位置，真的只有 760000 个 → 浪费 3.4%。

## 上机验证

```
./check.sh f4 -s
python3 tools/cu_drill.py run mine/f4_epilogue_residue.cu 1000 760 1024 20 1.5 -0.5   # M
```

BF16 输出的容差是 1e-2（BF16 只有 8 位有效数字，相对精度约 2⁻⁸ ≈ 0.4%）。4096³ 的速度应该和 F3 基本一样——谓词只在 kernel 开头算一次，epilogue 只占很小一部分时间。

## 提示

::: hint 1 先看 identity tensor 长什么样
`if (thread0()) { print(tAcA(_, _, _, 0)); print("\n"); }` ：它的每个元素是一个 `(m, k)` 坐标。`get<0> (tAcA(0, i, 0, 0))` 就是这个线程第 i 条加载的全局行号。
:::

::: hint 2 思路拆解
① `make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), make_stride(Int<1>{}, Int<0>{}))` 建一个寄存器里的布尔张量，K 方向 stride 0（所有 k 共用同一个谓词）；循环填入 `get<0>(tAcA(0, i, 0, 0)) < M` ；四处 `copy(g2s, …)` 改成 `copy_if(g2s, tApA, …)` 。② `for i in size(tCrC)` ：`if (elem_less(tCcC(i), shape(mC)))` 时算 `alpha * tCrC(i)` ，β ≠ 0 时加 `beta * float(tCgC(i))` ，最后 `tCgC(i) = TD(v)` 。
:::

::: hint 3 关键问题的答案
为什么 K 方向不能用同样的办法？

M 方向越界的那一行只污染自己那一行的输出，可以不写；K 方向越界的那一列会被乘进每一个输出的求和里。所以最后一个 k-tile 的越界部分必须是 0——要么在 SMEM 里清零，要么用 ZFILL 版本的 cp.async（`src-size =` `0` 时硬件写 0）。CUTLASS 的 SM80 mainloop 用的是后一种。
:::

## 参考解

::: solution solutions/f4_epilogue_residue.py
{{code: solutions/f4_epilogue_residue.py}}
:::

::: solution solutions/f4_epilogue_residue.cu
{{code: solutions/f4_epilogue_residue.cu}}
:::

## 面试官追问 & 对照源码

- 现在每个线程的 epilogue 写的是零散的 2 个 BF16（4 B）。改成“累加器 → SMEM（stmatrix）→ 每线程 16 B 的整行写出”能快多少？这就是 CUTLASS 的 SM80 epilogue 和 [端到端例子](../knowledge/e2e.html) 第 5 节说的那一步。
- 用 `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` 把 K 方向的边界也处理掉，需要改哪几行？
- 把 bias（每列一个数）和 ReLU 融合进 epilogue，需要多读多少字节？这就是 CUTLASS 的 EVT（epilogue visitor tree）在做的事。
