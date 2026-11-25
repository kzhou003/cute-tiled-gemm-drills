lede: 设计 Tiled GEMM 的第一个问题：每个 CTA 负责 C 的哪一块、每轮 K 循环要拿 A 和 B 的哪一段。本题先忘掉 CuTe 的 API，只用整数运算把四个 CTA、四轮 K 循环、越界的边界块全部推出来，再和真实 CuTe 的 local_tile 逐个坐标对拍。
# A1 · CTA 切分：C tile、K 循环、local_tile 视图与边界谓词

## 题面速览

- **背景知识**：先读 [从零设计](../knowledge/from_scratch.html) 的“五个空间映射”。
- **在体系里的位置**：真实 kernel 里这一步就是 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu]] 的三行 `local_tile(mA, cta_tiler, cta_coord, Step<...>)`。`local_tile` 的定义在 [[cutlass:include/cute/tensor_impl.hpp@local_tile|tensor_impl.hpp]]，带 `Step` 投影的版本紧跟其后（[[cutlass:include/cute/tensor_impl.hpp#L1054-L1068|带投影的 local_tile]]）；越界判断用 `elem_less`。
- **你要实现**：`num_tiles`、`cta_grid`、`c_tile_coords`、`a_tile_coords`、`b_tile_coords`、`predicate`、`view_offset`、`verify_cover`。IntTuple 的打印 / 解析函数已经给出。
- **关键规则**：
  1. tile 总是完整的 BM×BN（BM×BK），最后一块越界的部分用谓词屏蔽，**不缩小 tile**。
  2. 视图里的元素按 **colex** 顺序（第一维最快）排列，这和 CuTe 对 identity tensor 的逐元素顺序一致。
  3. A 的视图只依赖 `pid_m` 和 `k`，B 的视图只依赖 `pid_n` 和 `k`：B 按 N×K 存储。
  4. `local_tile` 只是视图：局部坐标平移到全局坐标再用原 tensor 的 layout 求偏移，没有拷贝。

**验收要点**：tile 数向上取整 · 非正参数报错 · 两种规模的 CTA 网格与 K tile 数和 CuTe 相同 · 每个 CTA 的 C 视图坐标与真实 `local_tile` 逐个相同（含顺序）· 每轮的 A、B 视图坐标与真实 CuTe 相同 · A 视图与 `pid_n` 无关 · 谓词与 `elem_less` 相同 · ragged CTA (2,1) 仍有 1024 个坐标、108 个有效 · 视图偏移是纯平移（与真实 CuTe 指针差、numpy 切片都相同）· 各种 M、N 下覆盖恰好一次 · 只用这些坐标拼出的 70×50×40 GEMM 等于 `A @ B.T`。

::: callout info 第一次出现的术语
- **CTA**（Cooperative Thread Array）：一个 thread block。一个 CTA 负责输出 C 的一个 tile。
- **tile**：矩阵里一个连续的矩形子块。BM、BN、BK：CTA tile 在 M、N、K 方向的大小。
- **K 循环**（K-loop / mainloop）：一个 CTA 沿 K 方向每次处理 BK 个元素，累加到同一个 C tile。
- **identity tensor**：每个元素的“值”就是它自己的坐标。拿它做 `local_tile`，就能直接看到视图覆盖了哪些坐标。
- **谓词**（predicate）：一个布尔掩码，标记哪些坐标在矩阵内，越界的读写被跳过（或读 0）。
:::

## 直觉与类比

::: callout analogy 拼一面瓷砖墙
C 是一面 64×64 的墙，每块瓷砖 32×32，四个工人（CTA）各贴一块。每块瓷砖要用的“原料”是 A 的一条横带和 B 的一条横带，原料一次只运来 16 列宽（BK），运四次。墙的尺寸不是瓷砖的整数倍时（70×50），最后一排瓷砖照样按整块来裁，但只把落在墙内的部分贴上去：这就是谓词。`local_tile` 相当于给工人一张“只框出他那块墙面的取景框”，墙本身一点没动。
:::

## 图解：一个 CTA 看到的三块视图

::: flow CTA (pid_m, pid_n) 的视图
cfg cw=230 rh=90 nw=180
n grid 0 0.5 "CTA 网格" "ceil(M/BM) × ceil(N/BN)" k=dark
n c 1 0.5 "C 视图" "起点 (pm·BM, pn·BN)" k=pass
n a 2 0 "A 视图 × k" "(BM,BK)：行由 pm 定" k=acc
n b 2 1 "B 视图 × k" "(BN,BK)：行由 pn 定" k=acc
n p 3 0.5 "谓词" "coord < (M,N,K)" k=warn
e grid -> c
e c -> a
e c -> b
e a -> p
e b -> p
cap: 三个视图都只是坐标平移。A 视图不依赖 pn：同一行的两个 CTA 读同一块 A，这块数据的复用发生在 L2 cache 里，CTA 自己看不到。
:::

## 逐步演示

::: steps 64×64×64，CTA tile 32×32×16
::: flow
cfg cw=185 rh=80 nw=150
n nt 0 0 "num_tiles" "64/32 = 2" k=dark
n ct 1 0 "C tile" "CTA (1,0)：行 32..63" k=pass
n kl 2 0 "K 循环" "64/16 = 4 轮" k=vio
n av 3 0 "A 视图" "k=1：列 16..31" k=acc
n rg 3 1 "ragged" "70×50：108/1024" k=warn
e nt -> ct
e ct -> kl
e kl -> av
e ct:b -> rg:l "边界" dash
:::
--- step 网格 {hl=nt}
`cta_grid(64, 64, 32, 32) = (2, 2)`：四个 CTA。CuTe 对 `local_tile(A, (32,32,16), (pm,pn,_), Step<_1,X,_1>)` 给出的视图形状是 `(_32,_16,_4)`：32 行、16 列、4 个 K tile。
--- step CTA (1,0) 的 C tile {hl=nt,ct}
起点 (32, 0)。真实 CuTe 的 `gC(0,0) = (32,0)`、`gC(31,31) = (63,31)`。视图第 i 个元素是 `(32 + i % 32, i // 32)`。
--- step K 循环 {hl=ct,kl}
每轮 BK = 16，四轮覆盖 K 的 0..15、16..31、32..47、48..63。数学上 `C_tile = Σ_k A[32:64, 16k:16k+16] · B[0:32, 16k:16k+16]ᵀ`。
--- step 第 1 轮的 A 视图 {hl=kl,av}
`a_tile_coords(1, 1, 32, 16)` 的前几项是 `(32,16), (33,16), (34,16) …`，和 oracle 记录的一致。真实 CuTe 的 `gA(0,0,1) = (32,16)`、`gA(31,15,3) = (63,63)`。
--- step 边界块 {hl=ct,rg}
70×50×40：网格 `(3, 2)`，K tile 数 3。CTA (2,1) 的 C 视图仍是完整的 32×32，起点 (64, 32)，但只有行 64..69（6 行）× 列 32..49（18 列）= **108** 个坐标在矩阵内，其余 916 个被谓词屏蔽。{{cells: ok:6行 bad:26行 _:× ok:18列 bad:14列}}
:::

## 易错点

- **最后一块缩小 tile。** CuTe 的视图形状永远是静态的 tile 形状，越界靠谓词。测试 `test_ragged_tile_keeps_full_static_shape` 要求返回完整的 1024 个坐标。
- **把 B 当成 K×N。** 运行例子里 B 按 N×K 存，B 视图的第一维是 n。测试 `test_a_and_b_tile_coords_match_local_tile`。
- **按行主序枚举视图。** 视图的元素顺序是 colex（m 最快），和 C 的内存是否行主序无关。测试逐个坐标比较顺序。
- **谓词只看一维。** 必须每一维都在范围内（`elem_less`）。`test_predicate_matches_elem_less` 的 ragged A 视图里 m 和 k 同时可能越界。
- **以为 local_tile 拷贝了数据。** `view_offset` 只是平移；`test_view_offset_matches_numpy_slicing` 用 numpy 切片对照，`test_view_offset_is_pure_translation` 用真实 CuTe 的指针差对照。

## 对照文档

> “以 CTA (0,0) 为例，它需要 A 的前 32 行和 B 的前 32 行，但 K=64 不必一次搬完。每轮处理 K 的 16 个元素……因此自然出现第一个需求：从全局 tensor 中，按 CTA 坐标取出当前 tile。CuTe 的 local_tile 负责构造这个局部视图……注意：local_tile 只确定当前 CTA 观察哪一块数据；它本身不决定每个线程搬什么，也不等于已经执行了内存 copy。”

本题把这段话里的每一个数都算了出来，并补上文档没有展开的边界块：tile 不变，谓词屏蔽。

## 上机验证

```bash
playground/run.sh playground/a1_cta_tiling.cpp      # 真实 CuTe，本机编译运行
./check.sh oracle a1                                # 重新生成 tests/golden/a1_cta_tiling.json
```

本机的真实输出（节选）：

```text
CTA (2,1): live  108 / 1024
   gC(0) = (64,32)  gC(last) = (95,63)
gA(pid_m=1, all k): (_32,_16,3):(_1@0,_1@1,_16@1)
view of CTA (0,1) on row-major 64x64: offset(0,0)=32 offset(1,0)=96
```

`(_32,_16,3):(_1@0,_1@1,_16@1)` 读作：第一维 stride 是“坐标第 0 维加 1”，第二维是“坐标第 1 维加 1”，第三维（K tile）是“坐标第 1 维加 16”。这正是“第 k 轮从第 16k 列开始”。`3` 前面没有下划线，因为 K = 40 是运行时值，K tile 数只能在运行时算出。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
num_tiles(70, 32)                   # 3
cta_grid(70, 50, 32, 32)            # (3, 2)
c_tile_coords(1, 0, 32, 32)[:3]     # [(32, 0), (33, 0), (34, 0)]
a_tile_coords(1, 1, 32, 16)[:2]     # [(32, 16), (33, 16)]
sum(predicate(c_tile_coords(2, 1, 32, 32), (70, 50)))   # 108
view_offset((1, 0), (0, 32), 64)    # 96
```

#### 关键问题

1. CTA (pm, pn) 的 C tile 起点是什么？A 视图和 B 视图分别依赖哪些坐标？
2. 为什么 CuTe 的边界块不缩小 tile，而是加谓词？对编译期优化有什么好处？
3. 视图的第 i 个元素对应哪个局部坐标？为什么是 colex？
:::

::: hint 2 思路拆解与常见坑
- `num_tiles`：先检查正数，再 `(extent + tile − 1) // tile`。
- 写一个辅助函数：给定起点 `(r0, c0)` 和 tile 形状 `(R, S)`，返回 `[(r0 + i % R, c0 + i // R) for i in range(R·S)]`。三个视图函数都调用它。
- `predicate`：逐维比较 `coord[d] < shape[d]`，全部成立才为真。
- `view_offset`：`(origin_r + r)·ld + origin_c + c`，不要先拷贝一块再求偏移。
- `verify_cover`：遍历所有 CTA，把有效坐标放进计数字典，最后检查个数和每个计数都等于 1。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 起点 `(pm·BM, pn·BN)`。A 视图依赖 pm 和 k（从 `(pm·BM, k·BK)` 起），B 视图依赖 pn 和 k（从 `(pn·BN, k·BK)` 起），C 视图依赖 pm 和 pn。
2. tile 形状是编译期常量，所有循环上界、寄存器数、指令序列都能在编译期确定；越界只影响少数边界 CTA，用谓词处理比让每个 CTA 带一个动态形状便宜得多。
3. 第 i 个元素是 `(i % R, i // R)`。CuTe 所有多维坐标都按 colex 线性化（最左边的模式变化最快），这是代数定义的约定，不代表内存连续方向；内存连续方向由 stride 决定。
:::
:::

## 参考解

::: solution solutions/a1_cta_tiling.py
{{code: solutions/a1_cta_tiling.py}}
:::

## 面试官追问 & 对照源码

- `local_tile(A, tiler, (pm, pn, _), Step<_1,X,_1>{})` 里的 `X` 起什么作用？去掉 `Step` 会得到什么形状？（对照 [[cutlass:include/cute/tensor_impl.hpp#L1050-L1068|带投影的 local_tile]] 里的注释例子。）
- 真实 kernel 怎样处理越界？对照 [[cutlass:examples/cute/tutorial/tiled_copy_if.cu|tiled_copy_if.cu]]：它用 identity tensor 的 partition 做谓词，和本题的做法相同。
- 同一行的两个 CTA 读同一块 A：如果把 CTA 的执行顺序换成“先列后行”（swizzled rasterization），L2 命中率会怎样变化？
- 70×50 的例子里有多少比例的线程在做无用功？如果 M 远大于 BM，这个比例怎么变？
