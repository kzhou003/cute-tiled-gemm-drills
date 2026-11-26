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

## 从小例子开始：先在草稿纸上手算

运行例子的表达式树有 59 个节点，直接盯着它看很难看出规律。先拿一个小到每个节点、每个格子都能手写出来的例子：一个 4×2 的 tile 放在每行 8 个元素（ld = 8）的行主序矩阵里，第 k 轮 tile 往右挪 2 列。它和运行例子的 `offset = m·64 +` `(k·16 + kk)` 是同一个形状，只是数字小。推完之后再放大到测试里的尺寸，最后看一般规则。

① 小：tile (4,2)，ld = 8，静态

15 个节点的树，四个 pass 每一步都能手写。

```
offset = (i%4)·8 + (k·2 + i//4)·1
原始  {+:2, *:3, %:1, //:1}  7 个
fold  {+:2, *:2, %:1, //:1}  6 个
SR    {+:2, <<:2, &:1, >>:1}  0 个乘除模
hoist step = 2
```

② 同一个小例子，动态

shape、stride 换成运行时的 bm、bk、ld，结构完全一样。

```
fold 只能删掉 ·1
SR    什么也做不了：
      {+:2, *:2, %:1, //:1}
hoist step = bk
```

③ 运行例子：copy\_offset\_expr(True)

TV layout ((4,32),(4,1)):((128,1),(32,0))，tile (32,16)，ld = 64。

```
原始  59 节点 {+:5, *:7, %:3, //:3}
fold  41 节点 {+:4, *:4, %:3, //:2}
SR    41 节点 {+:4, <<:4, &:3, >>:2}
hoist step = 16，base 11 个运算
```

④ 一般

每个 pass 只看局部模式；能不能消掉乘除，只取决于常数是不是 2 的幂、是不是编译期已知。

```
fold：c⊕c → 值；x·1, x+0, x//1 → x
      x·0, x%1 → 0；常量链合并
SR：  ·2ⁿ → <<n；//2ⁿ → >>n；%2ⁿ → &
(2ⁿ−1)
hoist：e = base + k·step
```

**✎** 草稿纸 ①：tile (4,2) 在 ld = 8 的矩阵里，第 k 轮

```text
1) 坐标   idx2crd_expr(i, (4, 2))：前缀积 1 → 4
          m  = i % 4                 ← 第 0 个模式取模
          kk = i // 4                ← 最后一个模式只除不模（CuTe 约定）
2) 地址   layout_expr((8, 1), (m, k·2 + kk))
          = ((i % 4) * 8) + (((k * 2) + (i // 4)) * 1)
          树：      +
                 /     \
               *         *
              / \       /  \
             %   8     +    1        ← 15 个节点
            / \       /  \
           i   4     *    //
                    / \   / \
                   k   2 i   4
          count_ops = {+:2, *:3, %:1, //:1}       ← 7 个运行时运算
3) fold   自底向上：(… ) * 1 → …（x*1 → x），其他节点没有可用的规则
          = ((i % 4) * 8) + ((k * 2) + (i // 4))   {+:2, *:2, %:1, //:1}
4) SR     8 = 2³ → <<3；  2 = 2¹ → <<1；  %4 → &3；  //4 → >>2
          = ((i & 3) << 3) + ((k << 1) + (i >> 2))   {+:2, <<:2, &:1, >>:1}
5) 验算   i = 5, k = 1：5 & 3 = 1 → 1 << 3 = 8；  1 << 1 = 2；  5 >> 2 = 1
          offset = 8 + (2 + 1) = 11            （第 1 行第 3 列：1·8 + 3 = 11 ✓）
          k = 0：i = 0..7 → [0, 8, 16, 24, 1, 9, 17, 25]
          k = 1：i = 0..7 → [2, 10, 18, 26, 3, 11, 19, 27]   ← 每个都 +2
6) hoist  把 + 树摊平：[(i&3)<<3,  k<<1,  i>>2]
          含 k 的项只有 k<<1 → step = 1<<1 = 2
          base = ((i & 3) << 3) + (i >> 2)            {+:1, <<:1, &:1, >>:1}
          base(5) = 8 + 1 = 9，9 + 1·2 = 11 ✓
7) 动态   同样的树，叶子 4、2、8 换成 bm、bk、ld：
          fold → ((i % bm) * ld) + ((k * bk) + (i // bm))
          SR   → 不变（编译器不知道 bm 是不是 2 的幂）
          hoist → step = bk，base 里还留着 *、%、//
```

**✎** 草稿纸 ③：把运行例子代入一个线程（tid = 127, v = 3, k = 1）

```text
tc = idx2crd(127, (4, 32)) = (127 % 4, 127 // 4) = (3, 31)
vc = idx2crd(3,   (4, 1))  = (3 % 4,   3 // 4)    = (3, 0)
idx = 3·128 + 31·1 + 3·32 + 0·0 = 384 + 31 + 96 + 0 = 511
(m, kk) = idx2crd(511, (32, 16)) = (511 % 32, 511 // 32) = (31, 15)
offset = 31·64 + (1·16 + 15)·1 = 1984 + 31 = 2015          ← golden
off_static_k1[127·4+3] = 2015 ✓
化简后：(127&3)<<7 = 384，127>>2 = 31，(3&3)<<5 = 96 → idx = 511；511&31 = 31，511>>5 =
15
        (31<<6) + ((1<<4) + 15) = 1984 + 31 = 2015 ✓
```

下面这个小游乐场可以改 tile 的 shape、ld 和模式。上面是当前阶段的表达式树，绿色是这个 pass 新造出来的节点（fold 在这个例子里只删不造，所以 ② 没有绿色；③ 里每个移位、按位与都是新的），红色虚线框是下一个 pass 会改写或删掉的节点。右边是矩阵本身，格子里写的是 tile 内索引 i（当前 k 那一块）。下方面板打印出每个函数应该返回的东西。试试把 shape 改成 (3,2)：3 不是 2 的幂，`% 3` 和 `// 3` 会一直留到最后。

::: raw
<figure class="flow snap"><div class="flow-head">小游乐场 · 一个 tile 的地址表达式</div><img src="../figs/c1/c1_new-01.png" alt="小游乐场 · 一个 tile 的地址表达式" loading="lazy" style="width:100%;aspect-ratio:1461/1506"></figure>
:::

## 图解详解：一棵地址表达式树怎样被一步步削减

::: callout info 关于这一节的图
这些图是交互版讲解页的静态快照，显示的是默认参数下的状态。文中提到的按钮、滑块和悬停提示在这里不能操作；图上的数字和下方的文字、表格、黑底面板说的是同一组值，可以直接对照你的函数输出。
:::

这一节有七张可交互的图，每张对应一个要实现的函数。点按钮、拖滑块、把鼠标停在节点或格子上，下方面板会把当前状态逐步代入公式。所有图里的化简都由页面里的一份 JS 实现计算，规则和参考解逐条相同，并且在 2000 个随机表达式上和 Python 参考解对拍过，结果一致。

整道题的数据流就是这几行：

```
e = copy_offset_expr(True)          # 图 1：TV layout → tile 坐标 → gmem offset，59 个节点
f = fold(e)                         # 图 2：常量折叠 + 恒等式，自底向上到不动点
s = strength_reduce(f)              # 图 3：×、÷、% 2 的幂 → 移位 / 按位与
count_ops(s)                        # 图 4：结构相同的子树只算一次（CSE）
layout_index_expr(i, sh, st, True)  # 图 5：先 coalesce，表达式化简看不出来的恒等函数直接消失
base, step = hoist(e, "k")          # 图 6：K 循环里只剩“指针 += step”
# 图 7：真实 clang 把同一个 CuTe 地址编成什么指令
```

### 图 1 · 运行例子的表达式树：每个 pass 改了哪些节点

`idx` （TV layout 算出的 tile 内索引）在 `m` 和 `kk` 里各用一次，所以这里把它画成左边单独一棵树，右边那棵树里紫色的 `idx` 叶子就指向它（这正是 CSE 之后的样子）。切换阶段看：fold 删掉 `·1` 、`·0` 和整个 `(v//4)·0` 分支（节点 59 → 41）；strength\_reduce 节点数不变，只是把每个 `* // %` 换成 `<< >> &` ；hoist 把 `k<<4` 整个拿走。切到“动态”再看：同样的位置留下了 `* bm` 、`% bm` 、`// bm` 、`* ld` 。

::: raw
<figure class="flow snap"><div class="flow-head">copy_offset_expr 的两棵树</div><img src="../figs/c1/c1_new-02.png" alt="copy_offset_expr 的两棵树" loading="lazy" style="width:100%;aspect-ratio:1461/1342"></figure>
:::

源码对照：`tv(tid, v)` 对每个顶层模式调用整数坐标版的 `crd2idx` ，它在最后一个模式上不取模（[[cutlass:include/cute/stride.hpp#L76-L95|crd2idx_itt ： Avoid recursion and mod on single/last iter]] ），这就是题面规则 2 的出处。

::: callout warn 一个和真实 CuTe 的细微差别
`tile.get_hier_coord(idx)` 调用的是带 stride 的 [[cutlass:include/cute/stride.hpp#L169-L215|idx2crd(i, s, d) = (i / d) % s]] ，它对每个模式都取模，包括最后一个。所以 oracle 里真实的 `kk` 是 `(idx / 32) % 16` ，而本题的模型（`idx2crd_expr` ）把它写成 `idx // 32` 。对 `idx < 512` 两者相等，所以 golden 一致；但真实汇编里多出来的 4 条指令（图 7 紫色）正是这个 `% 16` 。
:::

### 图 2 · `fold` ：自底向上，一次访问一个节点

`fold` 的核心是“先化简两个孩子，再看这个节点能用哪条规则”。下面按后序（左孩子、右孩子、自己）一个个节点播放。灰色是已经访问过但没变的节点，绿色是规则改写过的节点，黄色是当前节点，上方小框是它化简后的结果。注意 `((x*2)*4)*1` ：内层先变成 `x*8` ，外层看到的已经是 `x*8` 和 `1` ，所以一轮自底向上就做完了，第二轮只是确认没有变化。

::: raw
<figure class="flow snap"><div class="flow-head">fold 的访问顺序与触发的规则</div><img src="../figs/c1/c1_new-03.png" alt="fold 的访问顺序与触发的规则" loading="lazy" style="width:100%;aspect-ratio:1461/1079"></figure>
:::

### 图 3 · `strength_reduce` ：乘、除、模 2ⁿ 在二进制里是什么

为什么只有 2 的幂能换：在二进制里，`x·2ⁿ` 就是整体左移 n 位，`x // 2ⁿ` 就是右移 n 位（低 n 位掉出去），`x % 2ⁿ` 就是只留低 n 位（和 `2ⁿ−1` 按位与）。每一位的颜色跟着它自己走，你能看到移位只是在“搬格子”。换成 6、3、5 这样的数就没有这种对应关系，本题的规则保持原样。

::: raw
<figure class="flow snap"><div class="flow-head">x 的二进制与三种改写</div><img src="../figs/c1/c1_new-04.png" alt="x 的二进制与三种改写" loading="lazy" style="width:100%;aspect-ratio:1461/747"></figure>
:::

### 图 4 · `count_ops` ：同一个子树只算一次

这次把 `idx` 展开两份，画出完整的树。`count_ops` 从根开始先序遍历，第一次见到的子树计数（绿色），以后再遇到结构相同的子树就整棵跳过（半透明）。把鼠标停在任何一个节点上，所有和它结构相同的节点都会被黄框圈出来。切换到“每个节点都数”就是没有 CSE 的计数：静态化简后的 `>>` 会被数成 3 个，而测试要求 2 个。

::: raw
<figure class="flow snap"><div class="flow-head">CSE：展开的树与 count_ops 的访问</div><img src="../figs/c1/c1_new-05.png" alt="CSE：展开的树与 count_ops 的访问" loading="lazy" style="width:100%;aspect-ratio:1461/1199"></figure>
:::

### 图 5 · `coalesce_flat` ：表达式化简看不见的全局事实

列主序的 4×2 layout `(4,2):(1,4)` 把 i 映射到 i 自己，但它的表达式化简完是 `(i & 3) + ((i >> 2) << 2)` ：局部看，`&3` 和 `>>2 <<2` 都没法再化简。coalesce 看的是模式之间的关系：前一个模式 `4:1` 走完一圈正好到 `4·1 = 4` ，等于下一个模式的 stride 4，所以两个模式能合成一个 `8:1` 。逐步播放看每个模式被丢掉、合并还是保留。最下面一排是 L(i)：全是绿色就说明它是恒等函数。

::: raw
<figure class="flow snap"><div class="flow-head">coalesce 一个模式一个模式地合并</div><img src="../figs/c1/c1_new-06.png" alt="coalesce 一个模式一个模式地合并" loading="lazy" style="width:100%;aspect-ratio:1461/1023"></figure>
:::

源码对照：CuTe 的 [[cutlass:include/cute/layout.hpp#L773-L815|bw_coalesce]] 从右往左做同样的事（注释里那张“continue / replace\_front / prepend”表），合并条件是 `is_constant<true, s*d == d'>` ：只有静态整数才能合并，动态 layout 的模式永远不会被 coalesce。`composition` 在动手之前先 `coalesce_x` 左边的 layout（[[cutlass:include/cute/layout.hpp#L1134-L1141|composition]] ）。概念讲解见 [[cutlass:media/docs/cpp/cute/02_layout_algebra.md|02_layout_algebra.md]] 的 Coalesce 一节。

### 图 6 · `hoist` ：K 循环里只剩“指针 += 16”

这是 CTA (0,0) 在 A[64,64] 上读的 32 行 × 64 列，四种颜色是 k = 0..3 四轮的 tile。选一个线程：它每一轮读同一行里连续的 4 个元素（v = 0..3），四轮之间正好往右挪 16 列，也就是 offset 加 16。`hoist` 证明了这一点：`offset = base(tid, v)` `+ k·16` ，base 只在循环外算一次。悬停任意格子可以反查它属于哪个 (tid, v, k)。

::: raw
<figure class="flow snap"><div class="flow-head">一个线程四轮的地址 = base + k·step</div><img src="../figs/c1/c1_new-07.png" alt="一个线程四轮的地址 = base + k·step" loading="lazy" style="width:100%;aspect-ratio:1461/1562"></figure>
:::

### 图 7 · 和真实编译器对照：clang -O2 的 ARM64 汇编

图里的数据来自写这页时的一次编译（Apple clang 21，`-O2`，ARM64）：用真实 CuTe 头文件编译 `oracle/c1_asm/index_fns.cpp`，再按助记符分类计数。本仓库的 `oracle/c1_asm.sh` 做同样的编译，把本机的结果写进 `tests/golden/c1_asm.json`（每个函数的指令数、乘除指令和完整的指令列表），编译器版本不同时数字会略有出入。第三行“静态 + unsigned”是额外编译的：把 `off_static` 的参数和 `idx` 改成 `unsigned` ，其余不变（Apple clang 21，同样的 `-O2` ），并用 2048 个 offset 和 golden 对拍过。下面的指令表按“作用”上色，悬停一行看它在算表达式里的哪一部分。

::: raw
<figure class="flow snap"><div class="flow-head">off_static vs off_dynamic：指令数与每条指令的作用</div><img src="../figs/c1/c1_new-08.png" alt="off_static vs off_dynamic：指令数与每条指令的作用" loading="lazy" style="width:100%;aspect-ratio:1461/2363"></figure>
:::

GPU 上整数除法没有单条硬件指令，编译器会展开成一串乘法和修正（CUDA C++ Programming Guide 的 [Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions) 一节专门提醒整数除法和取模很贵，除数是 2 的幂时会被换成移位和按位与），所以“静态 layout 没有乘除”在 GPU 上比在 CPU 上更值钱。

### 速查

| 函数 | 做什么 | 小例子 (4,2),ld=8 | 运行例子（静态） |
|---|---|---|---|
| `idx2crd_expr(idx, shape)` | colex：c₀ = idx % s₀，cᵢ = (idx //Πs&lt;ᵢ) % sᵢ，最后一个只除不模 | `(i % 4, i // 4)` | `idx2crd(tid,(4,32)) = (tid%4, tid//4)` |
| `layout_expr(stride, coord)` | Σ cᵢ·dᵢ，从左到右累加 | `(i%4)·8 + (k·2 + i//4)·1` | 59 节点，`{+:5, *:7, %:3, //:3}` |
| `fold` | c⊕c → 值；·1、+0、//1 去掉；·0、%1 → 0；(x·c₁)·c₂、(x//c₁)//c₂、(x+c₁)+c₂ 合并；常量放右边 | `{+:2, *:2, %:1, //:1}` | 41 节点，`{+:4, *:4, %:3, //:2}` |
| `strength_reduce` | 右边是 2ⁿ（n ≥ 1）：·→&lt;&lt;n，// →>>n，%→&(2ⁿ−1) | `((i&3)<<3) + ((k<<1) + (i>>2))` | `{+:4, <<:4, &:3, >>:2}` ；动态 `{+:4, *:4, %:1, //:1, &:2, <<:1, >>:1}` |
| `count_ops` | 先序遍历，结构相同的子树只计一次；常量子树不计 | — | 不去重会数成 `{+:6, <<:6, &:5, >>:3}` |
| `coalesce_flat` | 丢尺寸 1；sᵢ·dᵢ == dᵢ₊₁ 就合并 | `(4,2):(1,4) → (8,):(1,)` | `(32,16):(1,32) → (512,):(1,)` |
| `hoist(e, 'k')` | simplify 后摊平 +，含 k 的项必须是 k、k·c、k&lt;&lt;n | step = 2 | step = 16（动态 `bk` ），base `{+:3, <<:3, &:3, >>:2}` |
| 真实 clang（ARM64） | 静态 24 条，0 乘 0 除；动态 22 条，5 个乘类 + 2 个 sdiv；静态 + unsigned 只要 9 条 |  |  |

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
- **fold 不先递归化简孩子。** 规则必须作用在“孩子已经化简过”的节点上：自顶向下只看一次的话，`((x*2)*4)*1` 只会去掉最外层的 `*1`，剩下 `(x*2)*4`。先递归孩子再改写自己（后序），一轮就能得到 `x*8`（图 2）；外面的 while 循环是保证到达不动点的安全网（用这套规则在大量随机表达式上验证过，自底向上一轮之后第二轮不会再变）。测试 `test_fold_identities_and_constant_chains`。
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
