lede: Track A 手写的五张映射表，用 Layout Algebra 可以写成几行组合；这些组合在编译期被化简、检查，最后由 CUDA 工具链生成机器码。这一页讲清楚三层各自做什么、“零成本抽象”到底是什么意思，对应 Track B 和 C。
# Layout Algebra 与编译：映射怎样组合，编译器怎样化简

## Layout 是一个带定义域的映射

Shape 规定合法坐标的范围，stride 规定每个坐标维度对偏移的贡献：

```text
L(i₀, i₁, …) = i₀·s₀ + i₁·s₁ + …
```

`(4,8):(8,1)` 是 4×8 行主序。shape 和 stride 可以是嵌套的 IntTuple，所以 layout 能保留层级结构。CuTe 的多维坐标用 **colex** 顺序线性化（第一个 mode 变化最快）：这是一条统一的枚举规则，让嵌套、分块和组合共享同一套代数定义；它不是硬件规定，物理上哪一维连续由 stride 决定（[[cutlass:media/docs/cpp/cute/01_layout.md|01_layout.md]]）。

三种经常出现的映射对象：

| 对象 | 回答的问题 | 例子 |
|---|---|---|
| Tensor layout | 逻辑坐标对应哪个内存位置？ | GMEM A、SMEM A、输出 C |
| TV layout | 哪个线程持有哪个 value？ | TiledCopy、MMA 操作数 / 累加器 |
| Atom / Tiled layout | 硬件操作单元怎样复制、排列到更大的 tile？ | Copy Atom、MMA Atom、TiledMMA |

## 代数操作，以及它们在 GEMM 里的用处

| 操作 | 定义 / 直觉 | 在本例中 |
|---|---|---|
| `composition` | `(A ∘ B)(i) = A(B(i))`，用 shape/stride 的分模除法构造结果，并检查整除条件 | `SMEM = Swizzle ∘ (32,16):(16,1)`；`local_tile` 背后也是 composition |
| `coalesce` | 删掉 size-1 的 mode，合并“走完正好接上”的相邻 mode：`(2,(1,6)):(1,(6,2)) → 12:1` | 化简索引，便于后续分析 |
| `complement` | 给定已覆盖的模式，构造补齐整个整数空间的另一组坐标 | divide 和 product 的定义里要用 |
| `right_inverse` / `left_inverse` | 从位置反查输入坐标；非双射时左右逆不同 | 由 TV 布局反推“这个元素在哪个线程”（B1） |
| `logical_divide` / `zipped_divide` | 拆成 (tile 内, tile 编号) | CTA tile、warp tile、每线程分块 |
| `logical` / `blocked` / `raked_product` | 按模式把小布局复制铺开 | `raked_product(thr, val)` 交错线程与连续 value（Copy TV） |
| `max_common_vector` | 两个 layout 共同的连续宽度 | GMEM 和 SMEM 之间一次能搬几个元素（B2） |

这些操作的定义都在 [[cutlass:include/cute/layout.hpp|layout.hpp]]（[[cutlass:include/cute/layout.hpp@composition|composition]] · [[cutlass:include/cute/layout.hpp@coalesce|coalesce]] · [[cutlass:include/cute/layout.hpp@right_inverse|right_inverse]] · [[cutlass:include/cute/layout.hpp@zipped_divide|zipped_divide]] · [[cutlass:include/cute/layout.hpp@raked_product|raked_product]] · [[cutlass:include/cute/layout.hpp@max_common_vector|max_common_vector]]），文档见 [[cutlass:media/docs/cpp/cute/02_layout_algebra.md|02_layout_algebra.md]]。

最后一项要特别小心：layout 说“连续”，不等于最终一定发出某条 128 位指令。还要看地址对齐、指令支持、边界和数据类型（B2）。

::: flow Track A 的映射怎样用代数重写（B1）
cfg cw=230 rh=100 nw=190
n lt 0 0 "local_tile" "zipped_divide + 切片" k=acc
n cp 1 0 "Copy TV" "raked_product + inverse" k=vio
n sm 2 0 "SMEM" "composition(Swizzle, L)" k=teal
n t 1 1 "和手写映射逐点相等" "测试检查" k=pass
e lt -> t
e cp -> t
e sm -> t
cap: B1 让你用代数构造写出同样的三张表，并逐点证明它们和 Track A 手推的结果一致；golden 来自真实 CuTe 打印的 layout。
:::

## 编译器怎样处理这些东西：三层要分开

::: flow 从 CuTe 源码到机器码
cfg cw=250 rh=90 nw=170
n src 0 0 "CuTe 源码" "Layout · Tensor · Atom" k=acc
n st 1 0 "静态类型与代数" "Int<N> · IntTuple" k=vio
n ir 2 0 "IR / lowering" "控制流 · 硬件语义" k=teal
n cc 3 0 "CUDA 工具链" "PTX → SASS" k=dark
e src -> st "模板实例化"
e st -> ir
e ir -> cc
cap: 三层协作，但不是同一个优化器。C1 模拟第一层；第三层的真实效果要看 PTX / SASS 和资源报告。
:::

### A. 编译期 layout 处理：让索引变简单（C1）

- **常量折叠**：静态的 shape、stride、tile 大小、线程布局在编译期算完。
- **映射复合**：多个 layout 组合后直接得到最终映射，不在运行时保留中间坐标。
- **规范化**：coalesce 删冗余 mode、合并维度。
- **强度削减**：乘以 / 除以 / 模 2 的幂变成移位和与运算。
- **循环不变量外提**：线程相关的基地址只算一次，K 循环里只加一个增量。

C1 的上机部分用 `clang++ -O2 -S` 编译两个函数：一个用静态 `Int<N>` 的 layout 计算地址，一个用动态整数；对比生成的汇编。注意这是主机（ARM64）的汇编，GPU 的 SASS 不同，但“静态信息让乘法和取模消失”这一点是一样的。

### B. 编译期合法性检查（C2）

CuTe 和 CUTLASS 在编译期用 `static_assert` 检查：composition 的整除条件、线程布局大小和线程数是否匹配、tile 能否被 MMA atom 和 copy tile 整除、swizzle 和行宽是否匹配、SMEM 是否放得下、向量访问是否对齐。非法组合编译失败，而不是运行时算错。C2 的上机部分真的拿一组 C++ 片段去编译，记录哪些通过、哪些报出什么 `static_assert`。

### C. CuTe DSL：Python 不是在 GPU 上运行的解释器

CuTe DSL 是一个 hybrid DSL：用 AST rewrite 保留 `for` / `while`、`if` 这类结构化控制流，用 tracing 捕获算术和 tensor 操作，然后进入编译器 IR 和 GPU 后端；`cute.compile` 生成的 JIT executor 可以缓存复用，`cutlass.Constexpr` 参数在编译期求值（[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst|dsl_code_generation.rst]]、[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.rst|dsl_jit_caching.rst]]、[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.rst|dsl_dynamic_layout.rst]]）。CuTe C++ 则依赖模板、类型系统和编译器实例化表达静态结构。

### D. “零成本抽象”的准确含义

目标是：Layout、Tensor 这些抽象主要在编译期决定，最终代码不因为抽象包装本身付出额外运行时成本。但它不代表每种写法都生成最优指令，也不代表任何动态信息都能被静态消除。

::: callout warn 不要误解
CuTe 不会自动搜索所有 CTA tile、warp 数、pipeline stage、swizzle 和调度参数并保证找到最优 kernel。它提供的是可组合的表达、静态推导、合法性检查和代码生成路径；性能搜索仍然需要人工设计搜索空间、autotuning 和实机 profiling（E2）。也不要以为 coalesce 一定让机器码更快：它首先简化映射结构，最终效果要看 lowering。
:::

::: callout analogy 提前印好的快递单
静态 layout 像写在合同里的货架规格：物流公司可以提前把每个快递员的取件单印好（“从第 tid·16 号货架开始，每轮往后挪 16 格”），快递员只做加法。动态 layout 像开工当天才告诉你货架规格：每个快递员每轮都得现场乘、除、取模。CuTe 的“零成本”指的是前一种情况下抽象被编译期消化掉。
:::

## 和练习题的对应关系

| 题目 | 内容 | 源码 |
|---|---|---|
| B1 | 用 divide / product / composition / inverse 重写 A1–A3 | [[cutlass:include/cute/layout.hpp@zipped_divide|zipped_divide]] · [[cutlass:include/cute/layout.hpp@raked_product|raked_product]] · [[cutlass:include/cute/layout.hpp@right_inverse|right_inverse]] |
| B2 | `max_common_vector` 与可用的 copy atom | [[cutlass:include/cute/layout.hpp@max_common_vector|max_common_vector]] · [[cutlass:include/cute/swizzle_layout.hpp@max_common_vector|swizzle 重载]] |
| C1 | 静态化简与上机汇编对比 | [[cutlass:include/cute/layout.hpp@coalesce|coalesce]] |
| C2 | 编译期合法性检查与上机 `static_assert` | [[cutlass:include/cute/layout.hpp@composition_impl|composition_impl]] |
