lede: 静态 shape 和 stride 的第二个好处是：很多配置错误在编译期就被 static_assert 拦下来，而不是在 GPU 上悄悄算出错误结果。这道题先复刻几条真实的 CuTe static_assert（composition 的两条整除条件、Swizzle 参数、TiledCopy 线程/值数、tile_to_shape），和真实编译器对拍 63 个程序；再给运行例子的整套 GEMM 配置做一次体检，分清哪些错误 CuTe 会拦、哪些要靠 CUTLASS 层面的约束、哪些能编译但需要运行时谓词。
# C2 · 编译期合法性检查：复刻 CuTe 的 static_assert，再给 GEMM 配置做体检

## 题面速览

- **背景知识**：[Layout Algebra 与编译](../knowledge/algebra_compile.html) 的“编译期合法性检查”。
- **在体系里的位置**：composition 的两条整除条件在 [[cutlass:include/cute/layout.hpp@composition_impl|composition_impl]]；Swizzle 的三条参数检查在 [[cutlass:include/cute/swizzle.hpp@Swizzle|swizzle.hpp]]；“too few thrs / vals” 在 [[cutlass:include/cute/atom/copy_atom.hpp|copy_atom.hpp]]；`tile_to_shape` 的两条在 [[cutlass:include/cute/layout.hpp@tile_to_shape|layout.hpp]]。
- **你要实现**：`composition_check`、`swizzle_check`、`tiled_copy_check`、`tile_to_shape_check`、`predict_compiles`、`check_config`。`coalesce_flat`、`Violation`、`GemmConfig`、`SNIPPETS`、`RUNNING_EXAMPLE` 已经给出。
- **关键规则**：
  1. 动态值用 `None` 表示，涉及它的检查一律跳过（CuTe 叫 weak divisibility condition：只在能静态验证时验证）。
  2. composition 先把 lhs `coalesce_flat`；rhs 是 tuple 时每个模式单独检查（右分配律），消息去重。
  3. 每个 rhs 模式：`rest_stride % curr_shape == 0 or rest_stride < curr_shape`，否则 Stride 条件失败；`new_shape = min(ceil(curr_shape/rest_stride), rest_shape)` 必须整除 `rest_shape`，否则 Shape 条件失败。
  4. `check_config` 按题面表格顺序报告 13 种 code，layer 是 `cute` / `cutlass` / `runtime` 之一。

**验收要点**：15 个手写 CuTe 程序和 48 个随机 composition 程序，能否编译、第一条 `static_assert` 消息都和真实 clang + CuTe 头文件一致 · 两条整除条件、coalesce 先行、动态跳过、stride 0 · rhs 分配与去重 · Swizzle、TiledCopy、tile_to_shape 的消息 · 运行例子体检为空 · ldmatrix、线程数、tile 整除、向量宽度、对齐、swizzle 跨度、smem 容量（24 级刚好放下，25 级超）、非整除问题尺寸各自报出正确的 code 和 layer。

::: callout info 第一次出现的术语
- **static_assert**：C++ 的编译期断言，条件为假时编译失败并打印消息。
- **weak divisibility condition**（弱整除条件）：CuTe 的说法：两边都是静态整数时才检查，有一边是运行时值就放行。
- **谓词**（predication）：给每次访存加一个“是否越界”的条件，处理问题尺寸不是 tile 整数倍的最后一块。
- **ldmatrix**：SM75+ 的 warp 级指令，一次从 shared memory 读若干个 8×8 的 16-bit 矩阵到寄存器，每行要求 16 字节连续。
:::

## 直觉与类比

::: callout analogy 建筑图纸审查
盖楼之前图纸要过审：结构规范（CuTe 的 `static_assert`）是硬性规定，梁的跨度不能被柱距整除，图纸直接打回；设计院的内部规范（CUTLASS collective 的约束，如“copy 的向量宽度必须正好是一个 atom”“ldmatrix 需要 16 字节一行”）不是物理定律，但违反了要么造不出来、要么很难用；现场情况（运行时）比如地块不是整数个开间，图纸能过审，但施工时要在最后一跨特殊处理（谓词）。好的审查要把三类问题分开说清楚，而不是只说“不行”。
:::

::: callout key 能静态知道的，就静态拦下
CuTe 的检查都写成 `static_assert`，条件里只用 `Int<N>` 这样的编译期常量；一旦某个 shape 或 stride 是运行时 `int`，检查就退化为“放行”（`composition_impl` 里的 `if constexpr (is_static<...>::value)`）。所以同一个 composition，静态时会被拒绝，动态时能编译、运行时给出没有意义的结果：`comp_dynamic_ok` 就是这样。这也是 CUTLASS 把 tile 形状做成模板参数的另一个理由。
:::

## 图解：一个配置要过的三道关

::: flow 检查的三层
cfg cw=220 rh=100 nw=180
n cfg 0 0.5 "GemmConfig" "tile、copy、mma、swizzle" k=dark
n cute 1 0.5 "CuTe static_assert" "swizzle · copy_vals" k=todo
n cl 2 0.5 "CUTLASS 约束" "线程·tile·向量·smem" k=warn
n rt 3 0 "运行时" "ragged_tile → 谓词" k=teal
n ok 3 1 "可以生成 kernel" "check_config == []" k=pass
e cfg -> cute
e cute -> cl
e cl -> rt
e cl -> ok
cap: 只有第一层是“编译器会拦”的真实 `static_assert`（本题和真实 CuTe 对拍）；第二层是 CUTLASS 的 collective / 硬件指令的要求，本题按 CUTLASS 的写法建模，不是对拍出来的；第三层能编译，但最后一块 tile 要加谓词。
:::

## 逐步演示

::: steps composition 的 Stride 与 Shape 条件：lhs = (4,6):(1,5)，它不能 coalesce
::: flow
cfg cw=200 rh=80 nw=170
n l 0 0 "lhs (4,6):(1,5)" "coalesce 后不变" k=dark
n r8 1 0 "rhs 3:8" "8 % 4 == 0 → 通过" k=pass
n r6 2 0 "rhs 3:6" "6 % 4 ≠ 0 且 6 ≥ 4" k=todo
n r2 3 0 "rhs 3:2" "2 不整除 3" k=todo
n lc 1 1 "lhs (4,6):(1,4)" "= 24:1，3:6 通过" k=pass
:::
--- step 为什么要检查 {hl=l}
`composition(A, B)(i) = A(B(i))`。B 的 stride 告诉你“在 A 的一维索引空间里每次跨多远”。A 是 `(4,6):(1,5)`：一维索引 0..3 走第一个模式，4..23 每 4 个进一行。只有当 B 的步长“和 A 的模式边界对齐”时，结果才是一个合法的 layout（每个模式还是 shape:stride 的等差形式）。
--- step 3:8 通过 {hl=l,r8}
`rest_stride = 8`，`curr_shape = 4`：`8 % 4 == 0`，正好跨两行。`next_shape = ceil(4/8) = 1`，只更新 `rest_stride = 2`。结果 `_3:_10`（每步跨 A 的两行，即 2·5 = 10）。
--- step 3:6 失败：Stride Divisibility {hl=l,r6}
`6 % 4 = 2` 且 `6 ≥ 4`：每一步落在“一行半”的位置，第 0、1、2 个元素在 A 里分别是 0、(2,1)→7、(0,3)→15，不是等差数列，没法写成一个 shape:stride。真实 CuTe：`Stride Divisibility Condition`。
--- step 3:2 失败：Shape Divisibility {hl=l,r2}
`2 < 4` 通过 stride 条件；`next_shape = ceil(4/2) = 2`，`new_shape = min(2, 3) = 2`，但 `3 % 2 ≠ 0`：3 个元素里前 2 个在 A 的第一个模式里，第 3 个要跨到下一行，不能整齐地切开。真实 CuTe：`Shape Divisibility Condition`。
--- step 先 coalesce {hl=lc}
`(4,6):(1,4)` 里 `4·1 == 4`，coalesce 成 `24:1`，只剩一个模式，任何 rhs 都合法：`3:6` 得到 `_3:_6`。同一个 rhs，lhs 换一个“等价但形状不同”的写法，结果就不一样：检查是在规范化之后做的。
:::

## 易错点

- **忘了先 coalesce lhs。** `(4,6):(1,4)` 和 `(4,6):(1,5)` 对 `3:6` 的结论相反；测试 `test_composition_rules` 和 `comp_coalesced_ok`。
- **动态值也去检查。** `None` 参与的任何判断都要跳过，算术结果也变成 `None`；`comp_dynamic_ok` 在真实 CuTe 里能编译。
- **stride 0 也检查。** `rd == 0` 是广播，永远合法。
- **rhs 是 tuple 时只查第一个模式，或者不去重。** 测试 `test_composition_messages_are_deduplicated`。
- **Swizzle 的消息顺序。** 先 MBase，再 BBits，最后 SShift；`swizzle_check(-1, -1, 0)` 有两条。
- **以为 TiledMMA 遇到不整除的 tile 会编译失败。** 真实 CuTe 里 `tiled_mma_bad_tile` 能编译（partition 允许不整除，最后一块要靠谓词）；所以 `mma_tile` 是 CUTLASS 层面的约束，不是 CuTe 的 `static_assert`。
- **smem 只算一个操作数。** 每级要放 A 和 B 两个 tile：`stages·(BM+BN)·BK·2 B`，运行例子每级 2048 B，24 级 = 49152 B 刚好等于 48 KB。测试 `test_smem_capacity_boundary`。

## 上机验证

`oracle/c2_snippets/` 里有 15 个手写程序，`python3 oracle/c2_sweep.py` 再生成 48 个随机 composition（lhs 5 种 × rhs 随机抽样，`random.Random(7)`）。`oracle/c2_try_compile.sh` 用真实 CuTe 头文件对每个程序跑 `clang++ -std=c++17 -fsyntax-only`，记录能否编译和第一条 `static_assert` 消息，写进 `tests/golden/c2_compile_checks.json`（63 个程序，本机约两分半钟；48 个随机程序里 9 个编译失败：5 个 Stride、4 个 Shape 条件）。

```text
$ oracle/c2_try_compile.sh
comp_coalesced_ok      ok
comp_dynamic_ok        ok
comp_ok                ok
comp_shape_fail        FAIL Shape Divisibility Condition
comp_stride_fail       FAIL Stride Divisibility Condition
local_tile_ok          ok
local_tile_ragged      ok
swizzle_ok             ok
swizzle_shift_fail     FAIL abs(SShift) must be more than BBits.
tile_to_shape_fail     FAIL tile_to_shape: block shape does not divide the target shape.
tile_to_shape_ok       ok
tiled_copy_ok          ok
tiled_copy_vec_fail    FAIL TiledCopy uses too few vals for selected CopyAtom
tiled_mma_bad_tile     ok
tiled_mma_ok           ok
sweep_00 ...
```

`playground/c2_compile_checks.cpp` 里有几行注释掉的非法写法，取消注释再运行就能看到对应的 `static_assert`。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
composition_check(((4, 6), (1, 5)), (3, 8))        # []
composition_check(((4, 6), (1, 5)), (3, 6))        # ['Stride Divisibility Condition']
composition_check(((4, 6), (1, 5)), (3, None))     # []   动态 stride：跳过
predict_compiles(SNIPPETS["swizzle_shift_fail"])   # (False, 'abs(SShift) must be more than BBits.')
[v.code for v in check_config(dataclasses.replace(RUNNING_EXAMPLE, threads=256))]
                                                   # ['copy_threads', 'mma_threads']
```

#### 关键问题

1. 运行例子的 copy atom 是 64 bit、元素 16 bit，每线程 4 个值。如果换成 128 bit 的 atom，哪条 CuTe 断言会失败？要让它通过，`copy_val` 应该改成什么，`copy_thr` 又要跟着怎么改？
2. `Swizzle<2,2,2>` 的 M=2 表示“4 个元素一组不被打乱”。为什么它必须 ≥ 每线程连续的 K 元素数？
3. 为什么开启 ldmatrix 时 `Swizzle<2,2,2>` 不再合格，换成 `Swizzle<1,3,3>` 就可以？
4. 问题尺寸 M=100 时，CuTe 能编译，但为什么体检要报 `ragged_tile`？它属于哪一层？
:::

::: hint 2 思路拆解与常见坑
- `_composition_mode(lhs, rs, rd)`：完全照着题面的伪代码写；每个比较前判断是否都已知，算术遇到 `None` 结果就是 `None`。
- `composition_check`：rhs shape 是 tuple 就 `zip(rs, rd)`，否则当成一个模式；对每个模式的结果按顺序追加、跳过已有的。
- 三个小检查各自返回一个消息列表，顺序和题面一致。
- `predict_compiles`：按 `desc[0]` 分派，返回 `(not msgs, msgs[0] if msgs else "")`。
- `check_config`：严格按表格顺序一条条写 `if`，每条最多 append 一次；`copy_vals` 调 `tiled_copy_check(copy_atom_bits, 8*elem_bytes, 1, prod(copy_thr), prod(copy_val))`。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 128 bit / 16 bit = 8 个值，每线程 4 个值不是 8 的倍数：`TiledCopy uses too few vals for selected CopyAtom`。`copy_val` 改成 (1,8)，BK=16 时 K 方向只需要 2 个线程，`copy_thr` 改成 (64,2)（仍是 128 线程，且要求 BM ≥ 64）或保持 (32,4) 但把 BK 改成 32。这就是 B2（向量宽度）讨论的取舍。
2. Swizzle 只对“块号”做异或，块内低 M 位保持不变。一个线程一次向量写入的 4 个元素必须落在同一个不被打乱的块里，否则 4 个元素在 smem 里不再连续，向量指令就不成立。
3. ldmatrix 每行读 16 字节 = 8 个 half，要求这 8 个元素在 smem 里连续：`2^M · 2 B ≥ 16 B`，即 M ≥ 3。`Swizzle<1,3,3>` 的块是 8 个元素；`Swizzle<2,2,2>` 的块只有 4 个，会被拆开。
4. `local_tile` 和 partition 允许最后一块不完整（`local_tile_ragged` 能编译），但访存会越界，需要用 `make_identity_tensor` 加谓词，或者在 host 端 pad。它是 runtime 层：编译器帮不了你，只能靠 kernel 逻辑。
:::
:::

## 参考解

::: solution solutions/c2_compile_checks.py
{{code: solutions/c2_compile_checks.py}}
:::

## 面试官追问 & 对照源码

- composition 的断言写成 `CUTE_STATIC_ASSERT_V(cond, msg)`，`cond` 本身是一个 `C<bool>` 类型，并且包在 `if constexpr (is_static<...> and is_static<...>)` 里（[[cutlass:include/cute/layout.hpp@composition_impl|composition_impl]]）。如果去掉这层 `if constexpr`、改成普通 `static_assert`，动态 layout 会怎样？
- 本题的 `mma_tile`、`vector_bytes`、`alignment`、`swizzle_span`、`swizzle_vector`、`ldmatrix`、`smem_capacity` 是按 CUTLASS 的写法建模的，不是真实 `static_assert`。去 `include/cutlass/gemm/collective/` 里找一找，SM90 的 collective 实际用 `static_assert` 检查了哪些？
- CUTLASS 3.x 的 `can_implement()` 在 host 端检查对齐和问题尺寸。哪些检查只能放在那里，不能放进 `static_assert`？
- 如果你要写一个 autotuner，本题的 `check_config` 应该放在搜索的哪一步？它能把搜索空间缩小多少（E2 会给出答案）？
- CuTe DSL 里同样的错误什么时候报？是 Python tracing 时、MLIR verifier 时，还是 ptxas 时（[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst|dsl_code_generation.rst]]）？
