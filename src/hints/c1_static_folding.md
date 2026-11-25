lede: CuTe 的 layout 代数不会在 GPU 上运行：shape 和 stride 是静态整数时，C++ 编译器在实例化模板的那一刻就把整套“TV layout → tile 坐标 → gmem offset”的计算折叠成几条移位、按位与和加法。这道题把运行例子里一个线程的 copy 地址写成表达式树，亲手做常量折叠、强度削减、coalesce 和循环不变量外提，再和真实 clang 编出来的汇编对照。
# C1 · 编译期化简：常量折叠、强度削减、coalesce 与循环外提

## 题面速览

- **背景知识**：先读 [Layout Algebra 与编译](../knowledge/algebra_compile.html) 的“三层要分开”。运行例子见 [从零设计](../knowledge/from_scratch.html)：A[64,64] 行主序，CTA tile 32×16，128 个线程，`ThrLayout (32,4):(4,1)`、`ValLayout (1,4)`。
- **在体系里的位置**：三层里的第一层“编译期 Layout 处理”：常量折叠、映射复合、规范化（[[cutlass:include/cute/layout.hpp@coalesce|coalesce]]）、推导访问属性。真实 CuTe 算出的 TV layout 是 `((_4,_32),(_4,_1)):((_128,_1),(_32,_0))`（由 `make_tiled_copy` 生成），offset 表在 `tests/golden/c1_static_folding.json`。
- **你要实现**：`layout_expr`、`idx2crd_expr`、`copy_offset_expr`、`fold`、`strength_reduce`、`simplify`、`count_ops`、`coalesce_flat`、`layout_index_expr`、`hoist`。`C`、`V`、`evaluate`、`pretty` 已给出。
- **关键规则**：
  1. 表达式是元组：`('c', n)`、`('v', name)`、`(op, a, b)`；变量都是非负整数。
  2. `idx2crd_expr` 用 colex，**最后一个模式不取模**（和 CuTe 一样）。
  3. `fold` 做到不动点：常量求值、`x*1`、`x+0`、`(x*c1)*c2`、`(x//c1)//c2`、常量放右边。
  4. `strength_reduce` 只处理 2 的幂：`*2^n → <<n`、`//2^n → >>n`、`%2^n → &(2^n−1)`。
  5. `count_ops` 对结构相同的子表达式只算一次（模拟 CSE），常量子树不算。

**验收要点**：`layout_expr` / `idx2crd_expr` 的结构 · 4 轮 k 的静态和动态 offset 与真实 CuTe 逐个相等（各 512 个，化简前后都要相等）· 静态化简后没有 `*`、`//`、`%`，正好是 `{+:4, <<:4, &:3, >>:2}` · 动态保留 4 个 `*`、1 个 `//`、1 个 `%` · 折叠与强度削减的每条规则 · CSE 计数 · coalesce 能消掉表达式化简看不出来的恒等函数 · k 循环外提出步长 16（动态时是 `bk`）· 非线性时报错 · 300 个随机表达式化简前后值相同 · 真实汇编里静态版本没有乘除、动态版本有。

::: callout info 第一次出现的术语
- **常量折叠**（constant folding）：编译期就能算出的子表达式直接换成结果。
- **强度削减**（strength reduction）：把贵的运算换成便宜的等价运算，例如乘以 8 换成左移 3 位。
- **CSE**（Common Subexpression Elimination）：相同的计算只做一次。
- **循环不变量外提**（loop-invariant code motion / hoisting）：循环里每轮都一样的部分挪到循环外，循环内只做增量。
- **SASS**：NVIDIA GPU 的真实机器指令；**PTX**：NVIDIA 的虚拟指令集，由 ptxas 编译成 SASS。
:::

## 直觉与类比

::: callout analogy 提前印好的快递单
每个线程每一轮都要“算出自己要搬的 4 个元素在全局内存的地址”。如果仓库的货架规格（shape、stride）是写死在合同里的（静态整数），物流公司可以提前把每个快递员的取件单印好：“从 tid·16 + … 号货架开始，每轮往后挪 16 格”，快递员只需要做加法。如果货架规格要到开工那天才告诉你（动态整数），每个快递员每轮都得现场做乘法、除法、取模，才能算出货架号。CuTe 的静态 layout 就是“提前印好的取件单”：抽象本身在编译期被消化掉，这就是文档说的“零成本抽象”的准确含义。
:::

::: callout key 三层优化，各管各的
CuTe 的静态 layout 代数（本题：折叠、规范化、合法性检查）、CuTe C++ 模板实例化 / CuTe DSL 的 IR 与 lowering（结构化控制流、硬件语义）、CUDA 后端（PTX → SASS 的指令选择、寄存器分配、调度）。layout 代数提供的是“静态信息和低成本的基础”，最后发出什么指令仍然要看后端。CuTe DSL 也是同一个思路：用 AST rewrite 保留循环和分支，用 tracing 捕获算术和 tensor 操作，再 JIT 编译并缓存（[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst|dsl_code_generation.rst]]、[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.rst|dsl_jit_caching.rst]]）；静态和动态 layout 的区别见 [[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.rst|dsl_dynamic_layout.rst]]。
:::

## 图解：一个地址表达式经过的 pass

::: flow copy 地址的化简流水线
cfg cw=210 rh=100 nw=176
n src 0 0 "copy_offset_expr" "TV → tile → gmem" k=dark
n fd 1 0 "fold" "常量折叠与恒等" k=acc
n sr 2 0 "strength_reduce" "×÷% 2^n → 位运算" k=vio
n hs 3 0 "hoist(k)" "base + k·16" k=pass
n co 1 1 "coalesce_flat" "layout 层面合并" k=teal
n ct 2 1 "count_ops" "CSE 后的运行时运算" k=warn
e src -> fd
e fd -> sr
e sr -> hs
e co -> fd "先规范化" dash
e sr -> ct
cap: 前三个 pass 是“表达式层面”的化简，coalesce 是“layout 层面”的化简：它能看出 `(32,16):(1,32)` 就是恒等函数，而表达式化简只能得到 `(i & 31) + ((i >> 5) << 5)`。CuTe 在组合 layout 时总是先 coalesce（[[cutlass:include/cute/layout.hpp@composition_impl|composition_impl]] 之前的 `coalesce_x`），所以很多化简在生成表达式之前就完成了。
:::

## 逐步演示

::: steps 一个线程的 copy 地址：从 18 个运行时运算到 13 个廉价运算
::: flow
cfg cw=200 rh=80 nw=170
n r 0 0 "原始表达式" "+5 ×7 %3 ÷3" k=dark
n f 1 0 "fold 之后" "+4 ×4 %3 ÷2" k=acc
n s 2 0 "strength_reduce" "+4 <<4 &3 >>2" k=vio
n h 3 0 "hoist(k)" "base 一次，每轮 +16" k=pass
n d 2 1 "动态版本" "+4 ×4 %1 ÷1 …" k=todo
e r -> f
e f -> s
e s -> h
e f:b -> d:l "bm 未知" dash
:::
--- step 构造 {hl=r}
按 CuTe 的顺序写：`tc = idx2crd(tid, (4,32))`、`vc = idx2crd(v, (4,1))`，`idx = tc·(128,1) + vc·(32,0)`，`(m, kk) = idx2crd(idx, (32,16))`，`offset = m·64 + (k·16 + kk)·1`。没化简时 CSE 后有 `{+:5, *:7, %:3, //:3}`。
--- step 常量折叠 {hl=r,f}
`(tid//4)*1 → tid//4`、`(v//4)*0 → 0`、`(...)*1 → ...`：剩 `{+:4, *:4, %:3, //:2}`。结果是 `((((tid%4)*128 + tid//4 + (v%4)*32) % 32) * 64) + (k*16 + (... // 32))`。
--- step 强度削减 {hl=f,s}
所有乘数、除数、模数都是 2 的幂：`*128 → <<7`、`//4 → >>2`、`%32 → &31`、`*64 → <<6`。剩 `{+:4, <<:4, &:3, >>:2}`，没有一个乘法、除法或取模。用 golden 验证：`tid=0, v=0, k=1 → 16`，`tid=1 → 20`，`tid=4 → 80`，`tid=127, v=3 → 2015`。
--- step 循环外提 {hl=s,h}
`hoist(e, 'k')` 得到 `step = 16`：K 循环里每轮只需要把基地址加 16，base 只在循环外算一次（`{+:3, <<:3, &:3, >>:2}`）。这正是真实 kernel 里“每个 k-tile 指针前进 BK 个元素”。
--- step 动态版本 {hl=f,d}
同样的结构，但 stride 是 `4*bm`、`bm`、`ld`，shape 是 `(bm, bk)`：化简后还剩 `{*:4, //:1, %:1, &:2, <<:1, >>:1, +:4}`：编译器不知道 bm 是 32，就只能真的去乘、真的去除。外提仍然成立，只是步长变成 `bk`。
:::

## 易错点

- **colex 写成行主序。** CuTe 的第一个模式变化最快：`idx2crd(i, (4,8,2)) = (i%4, (i//4)%8, i//32)`。测试 `test_idx2crd_colex_last_mode_not_wrapped` 抓这个。
- **最后一个模式也取模。** CuTe 不对最后一个模式取模，越界的索引沿最后一个 stride 走；`idx2crd(70, (4,8,2))` 是 `(2,1,2)` 而不是 `(2,1,0)`。
- **fold 只做一遍。** `((x*2)*4)*1` 需要多轮；测试 `test_fold_identities_and_constant_chains`。
- **对非 2 的幂也做强度削减。** `x*6` 不能换成移位（可以拆成移位加法，但不在本题规则里）；测试 `test_strength_reduce_only_powers_of_two`。
- **count_ops 不去重。** idx 在 m 和 kk 里各出现一次，不去重会把静态版本数成 `>>:3`；测试要求 `{+:4, <<:4, &:3, >>:2}`。
- **以为“除以 2 的幂就是右移”对 C++ int 永远成立。** 本题假设非负；真实 C++ 用有符号 `int`，编译器要插入符号修正（见下面汇编里的 `csel`、`csneg`）。用 `uint32_t` 或让编译器知道值非负，才能得到纯移位。

## 上机验证

用真实 CuTe 算出 golden（`./check.sh oracle c1`），再看真实 clang 把这两种写法编译成什么（`oracle/c1_asm.sh`，编译 [[here:oracle/c1_asm/index_fns.cpp|oracle/c1_asm/index_fns.cpp]]，`-O2 -S`）。本机是 Apple silicon，所以下面是 ARM64 主机汇编，不是 GPU 的 SASS；但“静态 layout 被折叠成移位和按位与、动态 layout 留下乘除”这个结论是一样的。

```text
$ playground/run.sh playground/c1_static_folding.cpp
TV layout (tid,v) -> tile idx : ((_4,_32),(_4,_1)):((_128,_1),(_32,_0))
tid   0:    16    17    18    19
tid   1:    20    21    22    23
tid   4:    80    81    82    83
tid 127:  2012  2013  2014  2015

$ oracle/c1_asm.sh
offset_static 25 instructions, 0 mul/div []
offset_dynamic 22 instructions, 7 mul/div ['madd', 'msub', 'mul', 'sdiv']
```

`offset_static` 的汇编（节选）全是 `add`、`and`、`asr`、`lsl` 和有符号修正用的 `csel` / `csneg`，最后两条是 `add w9, w9, w2, lsl #4`（+ k·16）和 `add w0, w9, w8, lsl #6`（+ m·64）。`offset_dynamic` 里则有 `mul w9, w9, w3`（× bm）、`sdiv w9, w8, w3`（÷ bm）、`msub`（% bm）、`sdiv w10, w9, w4`（÷ bk）、`madd`（m × ld + …）。

两点值得注意：静态版本的指令总数（25）并不比动态版本（22）少，多出来的是有符号除法的修正指令；真正的差别是没有乘法和除法，而整数除法在 CPU 和 GPU 上都比移位贵一个数量级。在 GPU 上可以用 `nvcc -arch=sm_80 -cubin` + `cuobjdump -sass` 对一个 kernel 做同样的对比（本机没有 CUDA 编译器，未运行）。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
e = copy_offset_expr(True)
evaluate(e, {"tid": 127, "v": 3, "k": 1})   # 2015
count_ops(e)                                # {'+': 5, '*': 7, '%': 3, '//': 3}
count_ops(simplify(e))                      # {'+': 4, '<<': 4, '&': 3, '>>': 2}
hoist(e, "k")[1]                            # ('c', 16)
pretty(simplify(layout_index_expr(V("i"), (32, 16), (1, 32), coalesce=True)))   # 'i'
```

#### 关键问题

1. 为什么静态 TV layout 里 v 的第二个模式是 `(4,1):(32,0)` 中的 `1:0`？它在表达式里变成了什么，fold 之后去哪了？
2. `(tid % 4) * 128 + tid // 4` 这一步在算什么？为什么不是 `tid`？
3. 为什么 `simplify` 化简不掉 `(i & 31) + ((i >> 5) << 5)`，而 `coalesce_flat` 一眼就能看出它是 `i`？
4. 动态版本外提之后步长是 `bk`。如果 bk 也是静态的而 ld 是动态的，哪些乘法会消失？
:::

::: hint 2 思路拆解与常见坑
- `idx2crd_expr`：维护一个前缀积表达式 `prefix`（开始是“没有”）；第 i 个坐标是 `idx // prefix`（i=0 时就是 `idx`），不是最后一个就再 `% s_i`；然后 `prefix *= s_i`（两边都是常量时直接乘成常量）。
- `fold`：写一个“处理一个节点”的函数：先递归处理两个孩子，再按规则改写这个节点；外面套一个 `while`，直到不再变化。
- `strength_reduce`：同样递归；判断 2 的幂用 `n & (n-1) == 0`，指数用 `n.bit_length() - 1`。
- `count_ops`：一个 `seen` 集合记录已经计过的子树（元组可以直接放进 set），跳过常量子树（不含任何变量的子树）。
- `hoist`：先 `simplify`，把 `+` 树展平成项列表；含 k 的项必须是 `k`、`k*c` 或 `k<<n`，把它们的系数加起来就是 step。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 每个线程只有 4 个值，`ValLayout (1,4)` 在 TV layout 里变成 `(4,1)`：第一个模式 4 是“4 个连续的 K 元素”（stride 32，因为 tile 是 colex 的 32×16，K 方向每走一步跨 32 个索引），第二个模式尺寸 1、stride 0 是占位。表达式里它是 `(v // 4) * 0`，fold 第一轮就把它变成 0 并消掉。
2. 这是 `ThrLayout (32,4):(4,1)` 的逆映射：线程号 `tid = m_thr·4 + k_thr`，所以 `k_thr = tid % 4`、`m_thr = tid // 4`；再乘以 tile 内 colex 索引的 stride（`k_thr` 对应 4 个 K 元素 = 128 个索引，`m_thr` 对应 1 个索引）。CuTe 用 `right_inverse` 自动算出了这个逆。
3. 表达式化简只看局部结构，不知道 `i & 31` 和 `(i >> 5) << 5` 恰好是 i 的低位和高位、相加没有进位重叠；layout 层面的 coalesce 看的是“前一个模式走完正好接上后一个模式”（`32·1 == 32`），这是一个关于整个函数的全局事实。这就是为什么 CuTe 在代数层面先规范化，而不是指望后端去证明。
4. `k * bk` 变成 `k << 4`，`(...) // bm` 和 `% bm` 仍然在（bm 动态），`m * ld` 仍然是乘法。静态信息越多，后端越能把乘除换成移位；这也是 CUTLASS 把 tile 大小都做成模板参数、把问题尺寸（M、N、K、ld）留给运行时的原因。
:::
:::

## 参考解

::: solution solutions/c1_static_folding.py
{{code: solutions/c1_static_folding.py}}
:::

## 面试官追问 & 对照源码

- 本题的 `v % 4` 在 v ∈ [0,4) 时其实是恒等的，真实编译器会把 `for v in 0..3` 展开，v 变成常量。CuTe 里这类展开靠 `CUTE_UNROLL` 和静态的 `size(...)`；给 `simplify` 加一个“值域分析”，你能消掉哪些运算？
- CuTe 的 composition 先做 `coalesce_x` 再做分模除法（[[cutlass:include/cute/layout.hpp@composition_impl|composition_impl]]）。为什么它要在“组合之前”而不是“组合之后”规范化？
- 本题的化简能告诉你“这个配置的地址计算便宜”，但不能告诉你“这个配置最快”。后者需要什么（见 E2）？
- 在 CuTe DSL 里，`cutlass.Constexpr` 参数在编译期求值、不进入运行时参数（[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.rst|dsl_jit_caching.rst]]）。它和 C++ 的 `Int<N>` 在“让编译器折叠”这件事上是同一种机制吗？JIT 缓存的 key 应该包含哪些东西？
- 有符号除法的修正指令（`csel`、`csneg`）在 GPU 上对应什么？为什么很多 CUDA kernel 把线程号和循环变量声明成 `unsigned` 或用 `__builtin_assume`？
