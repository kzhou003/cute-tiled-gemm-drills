lede: 把 TMA 加进 GEMM 的第一步：生产者不再让 128 个线程各算地址，而是由一个线程发一条带“张量描述符”的指令，硬件把整块 box 搬进 shared memory。本题复刻 CuTe 建描述符时检查的硬件限制，算出 mbarrier 要等的字节数，模拟越界填零，并和真实 CuTe 的 GMMA swizzle atom 在真实 smem 指针上逐个地址对拍。
# D1 · TMA box：描述符合法性、expect_tx、越界填零与 swizzle 后的 SMEM 位置

## 题面速览

- **背景知识**：先读 [时间维度](../knowledge/time_dimension.html) 的 TMA 一节；swizzle 的位运算在 A3 已经做过。
- **在体系里的位置**：CuTe 的 `make_tma_copy` 最终调用 [[cutlass:include/cute/atom/copy_traits_sm90_tma.hpp@make_tma_copy_desc|make_tma_copy_desc]]：从 gmem tensor 的 layout 推出维度、字节 stride、box，用一串 `assert` 检查硬件限制，再调用驱动的 `cuTensorMapEncodeTiled`。swizzle 模式由 smem atom 决定（[[cutlass:include/cute/atom/copy_traits_sm90_tma_swizzle.hpp@get_tma_swizzle_bits|get_tma_swizzle_bits]]），atom 本身定义在 [[cutlass:include/cute/atom/mma_traits_sm90_gmma.hpp@Layout_K_SW128_Atom|mma_traits_sm90_gmma.hpp]]。
- **你要实现**：`validate`、`elements_per_dim`、`expect_tx_bytes`、`swizzle_byte`、`smem_byte_offset`、`box_load`、`unswizzle`、`tile_element_map`。`SWIZZLE_B`、`SWIZZLE_SPAN`、`TmaDesc` 已经给出。
- **关键规则**：
  1. 所有元组**最内层在前**；`gstrides` 是维度 1 起的**字节** stride，维度 0 的 stride 隐含为一个元素。
  2. 十条合法性规则按固定顺序检查、全部报告；维数对不上时只报 `E_RANK`。
  3. `expect_tx_bytes = Π ceil(box[i] / elem_strides[i]) × elem_bytes`，越界填零的部分也算。
  4. swizzle 作用在 SMEM **字节地址**上：`Swizzle<B,4,3>`，B = 1/2/3 对应 32B/64B/128B。

**验收要点**：合法描述符 · 每条规则各自触发 · swizzle 跨度规则 · 维数错误只报一条 · 多条错误同时报告 · expect_tx（1024、8192、越界仍 8192、带 element stride 的 352 和 4096）· 三种字节 swizzle 的单点值、自逆、16 字节块不拆、不出 128 字节行 · 三种 box 的元素位置与真实 CuTe 的 GMMA atom（`tile_to_shape` 后放在 smem 指针上）逐个相同 · box 往返、越界填零、负起点、element stride、非法描述符报错。

::: callout info 第一次出现的术语
- **TMA**（Tensor Memory Accelerator）：Hopper（SM90）起的张量搬运单元。一个线程发 `cp.async.bulk.tensor`，硬件按描述符搬一整块。
- **描述符**（`CUtensorMap`）：128 字节的对象，记录全局 tensor 的基址、各维长度、字节 stride、box 大小、element stride、swizzle 模式。由主机端的驱动 API 编码，kernel 里只读。
- **box**：一次 TMA 搬运的矩形子块。**element stride**：box 内每隔几个元素取一个。
- **mbarrier**：shared memory 里的异步屏障。TMA 搬完后在它上面“交付”字节数；消费者等它。D2 专门讲。**expect_tx**：生产者预先告诉 mbarrier“这一轮要来多少字节”。
- **GMMA / WGMMA**（warp-group 级矩阵乘指令）：直接从 SMEM 读操作数，所以 SMEM 的 swizzle 必须是它认得的几种。
:::

## 直觉与类比

::: callout analogy 仓库发货单
以前取货（线程协作 copy）是 128 个人每人拿一张小纸条去货架找自己那几件；TMA 是只写一张“发货单”（描述符 + 起点坐标）交给叉车。发货单有格式要求：货架编号要对齐、每层间距要是 16 的倍数、一次最多 256 件（这些就是 `validate` 的十条规则）。叉车按单子把整托货送到你的库位，并在门口的签收板（mbarrier）上记“到了 8192 字节”；单子上写的范围超出货架的部分，叉车放空箱（填零），但空箱也算进字节数。库位里货怎么摆（swizzle），是事先约定的固定花样，叉车和取货的人（WGMMA）都按同一个花样来。
:::

## 图解：一次 box load 的数据路径

::: flow 从描述符到 SMEM
cfg cw=220 rh=100 nw=180
n d 0 0.5 "TmaDesc" "gdims · gstrides · box" k=dark
n v 1 0 "validate" "十条硬件规则" k=todo
n c 1 1 "box 坐标 c" "g = start + c·estride" k=acc
n o 2 1 "越界？" "是 → 写 0" k=warn
n s 3 0.5 "swizzle_byte" "Sw<B,4,3> on bytes" k=vio
n m 4 0.5 "SMEM + mbarrier" "tx += expect_tx" k=pass
e d -> v
e d -> c
e c -> o
e o -> s
e v -> s
e s -> m
cap: 合法性只看描述符，与起点无关；越界判断只看起点和 gdims；SMEM 位置只看 box 内坐标和 swizzle。三件事互相独立，测试也分开检查。
:::

## 逐步演示

::: steps B_TILE = TmaDesc(2, (64, 256), (128,), (64, 64), swizzle="128B")
::: flow
cfg cw=200 rh=80 nw=170
n ok 0 0 "合法性" "inner = 128 B" k=pass
n tx 1 0 "expect_tx" "64·64·2 = 8192" k=acc
n row 2 0 "行 r 的第 0 块" "128r ^ 16(r%8)" k=vio
n cu 3 0 "真实 CuTe" "4096/4096 一致" k=teal
e ok -> tx
e tx -> row
e row -> cu
:::
--- step 合法性 {hl=ok}
inner box = 64 × 2 = 128 字节：是 16 的倍数（`E_INNER_16B` 通过），且不超过 128B swizzle 的跨度（`E_SWIZZLE_SPAN` 通过）。行 pitch 128 字节是 16 的倍数。`validate(B_TILE) == []`。同一个 box 换成 `"64B"` 就只报一条：`E_SWIZZLE_SPAN: inner box 128 bytes > 64B swizzle span`。
--- step 字节数 {hl=ok,tx}
`expect_tx_bytes(B_TILE) = 8192`。A 的 16×32 half box 是 1024。如果 M 方向 element stride 取 3，box 32 行只取 `ceil(32/3) = 11` 行：`elements_per_dim = (16, 11)`，352 字节。生产者在 `arrive_and_expect_tx` 里报的就是这个数，报错了 mbarrier 永远等不到（D2）。
--- step 每一行搬到哪 {hl=tx,row}
box 坐标 (k=0, m=r) 的线性字节偏移是 128r，swizzle 后：`[0, 144, 288, 432, 576, 720, 864, 1008]`。也就是第 r 行的第 0 个 16 字节块被放到第 r % 8 块上。换成元素偏移就是 `[0, 72, 144, 216, …]`，正是元素域的 `Swizzle<3,3,3>`。{{cells: on:0 _:1 _:2 _:3 _:4 _:5 _:6 _:7}} {{cells: _:0 on:1 _:2 _:3 _:4 _:5 _:6 _:7}} {{cells: _:0 _:1 on:2 _:3 _:4 _:5 _:6 _:7}}
--- step 和真实 CuTe 对拍 {hl=row,cu}
把真实的 `GMMA::Layout_K_SW128_Atom<half_t>` 用 `tile_to_shape` 铺成 64×64，放到一个 smem 指针上，读 `&t(m,k) − base`：4096 个元素全部等于 `tile_element_map(64, 64, 2, "128B")`。32B、64B 两种 atom 也一样（`test_tile_element_map_matches_gmma_swizzle_atoms`）。
:::

## 易错点

- **把 atom 打印出来的 `Sw<3,4,3>` 当成元素域的 swizzle。** `Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)` 里的 swizzle 作用在指针的字节地址上（[[cutlass:include/cute/pointer_swizzle.hpp@apply_swizzle|apply_swizzle]] 对 `uintptr_t` 做异或）。直接对 layout 求整数值会得到第 1 行不动（64），真实地址是 72。测试 `test_tile_element_map_matches_gmma_swizzle_atoms`。
- **gstrides 写成元素数。** TMA 的 stride 是字节；40 个 half 一行要写 80。测试 `test_box_load_fills_out_of_bounds_with_zero` 用的就是 80 字节的 pitch。
- **越界的元素不算字节。** 硬件照样把填零的字节交付给 mbarrier；少报会让 tx 永远不归零。测试 `test_expect_tx_counts_every_box_element`。
- **负起点当作错误。** box 起点可以是负数（卷积的 halo 常见），越出的部分填零。测试里 `start = (-8, -4)`。
- **swizzle 跨度只看 box 总大小。** 规则看的是最内层一行的字节数（`box[0] × elem_bytes`），不是整个 box。
- **维数错时继续检查。** 长度对不上时后面的逐维检查会越界；和 CuTe 一样先报 `E_RANK` 就返回。

## 对照文档

> “TMA（Tensor Memory Accelerator）允许由指定线程/warp 发起描述符驱动的多维数据搬运，由硬件执行大块 tensor transfer。Layout 仍然重要：描述 tensor 的形状、stride、shared-memory 排布以及 swizzle 等……所以 TMA 并没有让 Layout 失去意义；它改变的是‘执行搬运的机制’，而不是‘数据坐标和物理排布’这件事。”

本题就是这句话的具体内容：描述符里的每个字段都来自 gmem layout，SMEM 排布来自 smem atom；`box_load` 之后用 `unswizzle` 还原出的逻辑 tile 和 numpy 切片完全相同，变的只是“谁来搬”。

## 真实代码

CuTe 建描述符时检查的硬件限制（简化自 [[cutlass:include/cute/atom/copy_traits_sm90_tma.hpp@make_tma_copy_desc|make_tma_copy_desc]]）：

```cpp
assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);   // Address must be 16B-aligned
assert(gmem_prob_shape[i] >= 1 && gmem_prob_shape[i] <= (uint64_t(1) << 32));
assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");
assert((gmem_prob_stride[i]) < (uint64_t(1) << 40));               // Stride must be max 2^40
assert((gmem_prob_stride[i] & 0b1111) == 0);                         // Stride must be multiple of 16B
assert(smem_box_shape[i] >= 1 && smem_box_shape[i] <= (uint32_t(1) << 8));   // max 256
assert(smem_box_stride[i] >= 1 && smem_box_stride[i] <= 8);
```

`E_INNER_16B` 和 `E_SWIZZLE_SPAN` 不在这段 assert 里：它们是 CUDA 驱动文档对 `cuTensorMapEncodeTiled` 的要求，违反时由驱动返回错误。atom 到 TMA 模式的映射在 `get_tma_swizzle_bits`：M == 4（16 字节块）时 B = 0/1/2/3 分别是 DISABLE / B32 / B64 / B128。

## 上机验证

```bash
playground/run.sh playground/d1_tma_box.cpp            # 真实 CuTe 的 GMMA atom 放在真实 smem 指针上
./check.sh oracle d1                                   # 重新生成 tests/golden/d1_tma_box.json
```

本机的真实输出：

```text
K_SW128 half atom : Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)
physical element offset of (m, k) for k = 0, 8, ..., 56:
 m=0:     0     8    16    24    32    40    48    56
 m=1:    72    64    88    80   104    96   120   112
 m=2:   144   152   128   136   176   184   160   168
 m=7:   504   496   488   480   472   464   456   448
element-domain Swizzle<3,3,3> agrees on 4096 of 4096 elements
```

每一行的 8 个 16 字节块被异或 m % 8 重新排列：8 行里同一列 k 的块各落在不同位置，WGMMA 和 TMA 都没有 bank conflict。TMA 描述符本身要驱动编码，CPU 上验证不了；有 Hopper（SM90）或更新的 GPU 时，可以用 CUTLASS 的 TMA 例子把 gdims 改成不对齐的值，看 `cuTensorMapEncodeTiled` 的返回码（本机没有 GPU，未核对）。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
A = TmaDesc(2, (64, 64), (128,), (16, 32), swizzle="32B")
validate(A)                                   # []
validate(TmaDesc(2, (64, 64), (100,), (16, 512), swizzle="32B", base_addr=4))
# ['E_ALIGN: ...', 'E_GSTRIDE_ALIGN: ...', 'E_BOX: box[1]=512 not in [1, 256]']
expect_tx_bytes(A)                            # 1024
elements_per_dim(TmaDesc(2, (64, 64), (128,), (16, 32), elem_strides=(1, 3)))   # (16, 11)
[swizzle_byte(o, "128B") for o in (0, 16, 128, 144, 256, 1152)]   # [0, 16, 144, 128, 288, 1168]
```

#### 关键问题

1. `Swizzle<3,4,3>` 在字节地址上读哪几位、写哪几位？为什么它永远不会把一个 16 字节块拆开，也不会把数据移出 128 字节行？
2. 越界填零的字节为什么也要算进 expect_tx？
3. 元素域里 half / bf16 的 128B swizzle 为什么是 `Swizzle<3,3,3>`，而 fp32 是什么？
:::

::: hint 2 思路拆解与常见坑
- `validate`：先比较四个元组的长度和 rank 范围，不对就直接返回 `[E_RANK]`；然后按规则表顺序逐条 append。
- `swizzle_byte`：`yyy = ((1<<B)−1) << 7`，返回 `off ^ ((off & yyy) >> 3)`；B = 0 原样返回。
- `smem_byte_offset`：box 坐标按 colex（最内层最快）线性化，乘 `elem_bytes`，再 swizzle。
- `box_load`：先 `validate`，非法抛 ValueError；`np.zeros(expect_tx_bytes)`；枚举每个 box 坐标，算全局坐标，越界跳过（已经是 0），否则按字节地址 `Σ g[i]·stride_b[i]`（`stride_b[0] = elem_bytes`）拷 `elem_bytes` 个字节。
- `unswizzle`：swizzle 是自逆的，所以“从 SMEM 读回逻辑元素 i”也用 `swizzle_byte(i·eb)`。
- `tile_element_map`：`out[m + rows·k] = swizzle_byte((m·inner + k)·eb) // eb`。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 读第 7、8、9 位（第几个 128 字节行的低 3 位），异或到第 4、5、6 位（行内第几个 16 字节块）。低 4 位（块内字节）和第 7 位以上都不变，所以块不拆、行不换。
2. 硬件对整块 box 写 SMEM，越界部分写的是 0，同样是一次写入，同样计入 transaction。mbarrier 的 phase 要等 tx 归零才完成；少报会提前完成（读到没写完的数据），多报会永远等不到。
3. 元素偏移 e 的字节地址是 2e，`Sw<3,4,3>(2e) / 2` 等于把位 [6,9) 异或进位 [3,6)，即 `Swizzle<3,3,3>`。fp32 是 4 字节，同理得 `Swizzle<3,2,3>`。
:::
:::

## 参考解

::: solution solutions/d1_tma_box.py
{{code: solutions/d1_tma_box.py}}
:::

## 面试官追问 & 对照源码

- `make_tma_copy` 怎样从一个任意的 gmem layout 和 smem layout 找出 TMA 的“基”（哪几维连续、box 怎么切）？对照同一文件里的 [[cutlass:include/cute/atom/copy_traits_sm90_tma.hpp@construct_tma_gbasis|construct_tma_gbasis]]。
- multicast：一个 cluster 里多个 CTA 要同一块 A，怎样只读一次 gmem？box 为什么要能被 multicast 数整除？
- 为什么 TMA 要求 stride 是 16 字节的倍数，而线程协作 copy 没有这个要求？如果矩阵的行长度是 100 个 half，该怎么办？
- Ping-pong kernel 的 epilogue 用 TMA store 写回 C：store 方向的 swizzle 和越界规则有什么不同？
