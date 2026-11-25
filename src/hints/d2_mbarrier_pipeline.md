lede: TMA 是异步的：指令发出去就返回，数据几百个周期后才落地。要让搬运和计算重叠，就得有 S 个 SMEM stage 轮流用，并且用屏障保证“没到的不读、没用完的不覆盖”。本题复刻 Hopper 的 mbarrier 语义和 CUTLASS 的 PipelineState / PipelineTmaAsync 协议（PipelineState 与真实 CUTLASS 逐步对拍），再用离散事件模拟主循环：算出 ceil(L/T)+1 这个 stage 数从哪来，并亲手造出死锁和数据竞争。
# D2 · mbarrier 多级流水线：phase 奇偶、PipelineState、producer / consumer 协议与三种同步 bug

## 题面速览

- **背景知识**：先读 [时间维度](../knowledge/time_dimension.html) 的 pipeline 一节，再做完 D1（expect_tx 字节数从那里来）。
- **在体系里的位置**：Hopper GEMM 的主循环由 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineTmaAsync|PipelineTmaAsync]] 驱动；循环变量是 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineState|PipelineState]]，生产者的起点由 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@make_producer_start_state|make_producer_start_state]] 给出。屏障本身是 [[cutlass:include/cutlass/arch/barrier.h@ClusterTransactionBarrier|ClusterTransactionBarrier]]，底层是 PTX 的 `mbarrier.*` 指令。
- **你要实现**：`MBarrier`（`arrive`、`expect_tx`、`arrive_and_expect_tx`、`complete_tx`、`try_wait`）、`PipelineState`（`increment`、`advance`、`as_tuple`）、`producer_start_state`、`simulate`、`predicted_interval`、`min_stages`。三个异常类和 `SimResult` 已经给出。
- **关键规则**：
  1. phase 完成 = 待到达数为 0 且 tx 字节为 0；`try_wait(p) == (completed % 2 != p)`。
  2. `PipelineState` 的 index 到 S 回 0 时翻转 phase；生产者从 phase 1 起步，消费者从 phase 0 起步。
  3. 生产者：等 `empty[i]` → `full[i].arrive_and_expect_tx` → 发 TMA；消费者：等 `full[i]` → MMA → `empty[i].arrive`。
  4. 稳态间隔 `max(T, (L+T)/S)`，最小 stage 数 `ceil(L/T) + 1`。

**验收要点**：新 barrier 的奇偶 · arrive 到了但字节没到不算完成 · 先登记字节再 arrive · arrive 过量报错 · increment 序列（含 S = 1）· advance 等于重复 increment · **PipelineState 的 4 条序列和 7 个 advance 与真实 CUTLASS 相同** · S = 4、K = 8 的完整时间线 · S = 1..6 的稳态间隔等于公式 · `min_stages` 恰好藏住延迟、多了没用 · 短循环没有稳态 · 四种 bug 分别死锁或数据竞争。

::: callout info 第一次出现的术语
- **mbarrier**（memory barrier）：放在 shared memory 里的 64 位异步屏障。它有一个 phase（阶段）计数：每当预期的 arrive 次数和预期的字节都到齐，当前 phase 完成，进入下一个。
- **parity**（奇偶）：等待时只告诉它“我等的是第奇数个还是第偶数个 phase”，而不是完整编号。只用 1 位，所以 S 个 stage 轮转时必须自己记住 phase 位。
- **transaction 字节**（tx）：TMA 落地时，硬件直接在 mbarrier 上扣掉对应字节数（`complete_tx`），不需要任何线程 arrive。
- **full / empty barrier**：每个 stage 两个 mbarrier。full：数据到了；empty：消费者用完了，可以覆盖。
- **WG**（warp group，4 个 warp = 128 线程）：Hopper 上 WGMMA 的执行单位。主循环的消费者是一个或两个 WG（D3）。
:::

## 直觉与类比

::: callout analogy 回转寿司的 S 个盘子
厨师（生产者）和食客（消费者）之间只有 S 个盘子在转。每个盘子上有两盏灯：绿灯（full）亮表示“菜做好了”，红灯（empty）亮表示“吃完了，可以再装”。灯只显示“这是第奇数轮还是偶数轮”，所以厨师和食客各自记一个“我现在等第几号盘子的第几轮”（PipelineState）。厨师一开始等的是“上一轮的红灯”，新盘子默认已经是空的，所以直接通过（phase 1 起步）。厨师记错了轮次，就会等一盏永远不会再亮的灯（死锁）；食客记错了，就会把空盘子当成菜（数据竞争）；食客吃完不按红灯，厨师转完一圈就再也拿不到盘子（饿死）；厨师不看红灯直接装菜，就会把还没吃完的菜倒掉（覆盖）。
:::

## 图解：一个 stage 的生命周期

::: flow stage i 的两盏灯
cfg cw=230 rh=100 nw=190
n acq 0 0 "producer_acquire" "wait empty[i] (phase p)" k=acc
n tx 1 0 "arrive_and_expect_tx" "full[i]: tx += bytes" k=vio
n land 2 0 "TMA 落地" "complete_tx → 完成" k=teal
n wait 2 1 "consumer_wait" "wait full[i] (phase c)" k=warn
n mma 1 1 "MMA" "T 个时间单位" k=pass
n rel 0 1 "consumer_release" "empty[i].arrive" k=todo
e acq -> tx
e tx -> land "L" anim
e land -> wait
e wait -> mma
e mma -> rel
e rel -> acq "下一轮" dash
cap: 一个 stage 从发出 TMA 到可以再用，至少要 L + T。S 个 stage 轮转，所以每 (L + T) / S 才能开始一块新的；同时 MMA 本身每块要 T。两者取大就是稳态间隔。
:::

## 逐步演示

::: steps S = 4，L = 700，T = 200，K = 8：四个 stage 藏不住 700 的延迟
::: flow
cfg cw=200 rh=80 nw=170
n fill 0 0 "填满" "t=0 发 4 块" k=acc
n go 1 0 "开算" "t=700..1500" k=pass
n stuck 2 0 "卡住" "t=900 才放出 stage 0" k=warn
n st 3 0 "稳态" "225 / 块" k=teal
e fill -> go
e go -> stuck
e stuck -> st
:::
--- step t = 0：生产者一口气发 4 块 {hl=fill}
生产者起点 `(0, 1, 0)`：在新的 empty barrier 上 `try_wait(1)` 立即为真。四个 stage 都通过，`load_issue = [0, 0, 0, 0, …]`。第 5 块要等 stage 0 的第 1 轮（`producer_start_state(4)` 增加 4 次后是 `(0, 0, 4)`）。
--- step t = 700：数据落地，消费者开算 {hl=fill,go}
四块同时在 700 落地。消费者按 0、1、2、3 依次计算：`mma_start = [700, 900, 1100, 1300]`。
--- step t = 900：stage 0 释放，第 5 块才发出 {hl=go,stuck}
第 0 块在 900 算完并 release，生产者马上发第 5 块，700 之后（1600）才落地。消费者在 1500 算完第 3 块后空等 100。`load_issue[4:] = [900, 1100, 1300, 1500]`，`load_done[4:] = [1600, 1800, 2000, 2200]`。
--- step 稳态 {hl=stuck,st}
`total = 2400`，最后 4 块的平均间隔 `225 = (700 + 200) / 4 > 200`。S = 1..6 的稳态间隔是 900、450、300、225、200、200：要完全藏住延迟需要 `min_stages(700, 200) = ceil(3.5) + 1 = 5` 个 stage。
:::

## 易错点

- **生产者从 phase 0 起步。** 新 barrier 的 phase 0 还没完成，生产者等 `empty[0]` 的 phase 0，消费者等 `full[0]`，谁也不动：t = 0 死锁（`producer waits empty[0] parity 0; consumer waits full[0] parity 0`）。测试 `test_bug_producer_phase0_deadlocks`。
- **消费者从 phase 1 起步。** 新 barrier 上 `try_wait(1)` 立即为真，消费者读到还没写的 stage：`consumer reads stage 0 for tile 0 but it holds None`。测试 `test_bug_consumer_phase1_reads_dirty_data`。
- **忘了 release。** 前 S 块正常，第 S+1 块的生产者永远等不到 empty：t = 1500 死锁。K ≤ S 时这个 bug 不会暴露（`total = 1500`），所以小测试抓不到它。测试 `test_bug_no_release_deadlocks_after_filling_the_ring`。
- **跳过 acquire。** S = 2 时第 3 块直接覆盖 stage 0：`TMA for tile 2 overwrites stage 0 still holding tile 0`。测试 `test_bug_skip_acquire_overwrites_a_busy_stage`。
- **先 arrive 再 expect_tx。** arrive 的那一刻 tx 还是 0，phase 立刻完成，消费者会提前读。`arrive_and_expect_tx` 必须先加字节再 arrive。测试 `test_expect_tx_before_arrive`。
- **S = 1 时 index 不动。** CUTLASS 对 `Stages == 1` 单独处理：每次 ++ 都翻转 phase。测试 `test_single_stage_flips_every_step` 和真实 CUTLASS 的 `consumer_1` 序列。

## 对照文档

> “多 stage pipeline 会用多个 shared-memory buffer：计算当前 stage 时，producer 预取后续 stage；consumer 在确认数据 ready 后再消费；stage 被用完后才能释放并复用……同步协议保证不会读到未完成的数据，也不会覆盖仍在使用的 buffer。”

文档里的 `issue_load / wait_until_ready / release_stage` 三个伪代码调用，在 CUTLASS 里分别是 `producer_acquire`（加上 TMA 发射）、`consumer_wait`、`consumer_release`。本题补上了文档没有展开的部分：ready / empty 用什么表示（mbarrier 的 phase 奇偶），以及“保证不会读到 / 不会覆盖”在协议写错时具体怎样失效。

## 真实代码

生产者和消费者的四个调用（简化自 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineTmaAsync|PipelineTmaAsync]]）：

```cpp
void producer_acquire(uint32_t stage, uint32_t phase) {
  empty_barrier_ptr_[stage].wait(phase);
  if (params_.is_leader)
    full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
}
void producer_commit(uint32_t stage, uint32_t bytes) { }   // NOP for TMA：硬件自己 complete_tx
void consumer_wait(uint32_t stage, uint32_t phase) { full_barrier_ptr_[stage].wait(phase); }
void consumer_release(uint32_t stage) { empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread); }
```

`wait` 就是一个 `try_wait.parity` 的自旋循环（简化自 [[cutlass:include/cutlass/arch/barrier.h@ClusterBarrier|ClusterBarrier::wait]]）：

```text
LAB_WAIT:
  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;
  @P1 bra DONE;
  bra     LAB_WAIT;
```

生产者的起点（[[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@make_producer_start_state|make_producer_start_state]]）：

```cpp
// Producer starts with an opposite phase as the buffers are initially empty
return {/*index*/ 0, /*phase*/ 1, /*count*/ 0};
```

## 上机验证

```bash
playground/run.sh playground/d2_mbarrier_pipeline.cpp   # 真实 cutlass::PipelineState，在主机上编译
./check.sh oracle d2                                    # 重新生成 tests/golden/d2_mbarrier_pipeline.json
```

本机的真实输出（节选）：

```text
  k | producer (index,phase) | consumer (index,phase)
  0 |        (0,1)           |        (0,0)
  3 |        (3,1)           |        (3,0)
  4 |        (0,0)           |        (0,1)
  8 |        (0,1)           |        (0,0)
PipelineState<4>(2,1,0).advance(13) = (3,0,13)
```

生产者和消费者在同一个 index 上的 phase 永远相反：生产者等“上一轮被吃完”，消费者等“这一轮被装好”。`PipelineState` 只是整数记账，所以 oracle 只需把头文件引用的 `threadIdx`、`__syncthreads` 在主机上补一个空定义就能编译。mbarrier 本身是 PTX 指令，CPU 上跑不了，本题按 PTX ISA 的语义建模；真正的时序要在 Hopper 或更新的 GPU 上测（本机没有，未核对）。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
b = MBarrier(1); b.try_wait(1), b.try_wait(0)         # (True, False)
b.arrive_and_expect_tx(1024); b.try_wait(0)           # False：字节还没到
b.complete_tx(1024); b.try_wait(0), b.completed       # (True, 1)
s = producer_start_state(4)
seq = []
for _ in range(6):
    seq.append(s.as_tuple()); s.increment()
seq        # [(0,1,0), (1,1,1), (2,1,2), (3,1,3), (0,0,4), (1,0,5)]
PipelineState(4, 2, 1).advance(13).as_tuple()          # (3, 0, 13)
r = simulate(4, 8, 700.0, 200.0)
r.mma_start, r.total, r.steady_interval                # [700, 900, 1100, 1300, 1600, 1800, 2000, 2200], 2400, 225
min_stages(700, 200)                                   # 5
```

#### 关键问题

1. 为什么 parity 只要 1 位就够？什么情况下 1 位会不够（提示：生产者领先消费者超过一整圈）？
2. 生产者为什么从 phase 1 起步，而不是先把 empty barrier 都 arrive 一遍？
3. 推导 `min_stages = ceil(L/T) + 1`：为什么是 +1 而不是 `ceil(L/T)`？
:::

::: hint 2 思路拆解与常见坑
- `MBarrier`：三个整数 `pending`、`tx`、`completed`；每次 arrive 或 complete_tx 后检查“两者都为 0”就完成一个 phase，并把 `pending` 重置。
- `PipelineState.advance(n)`：`wraps = (index + n) // S`，奇数次就翻转 phase，然后 `index = (index + n) % S`。
- `simulate`：把生产者和消费者写成生成器，`yield ("wait", 条件, 说明)` 或 `yield ("sleep", 时长, None)`。主循环：同一时刻反复推进所有能推进的参与者（生产者在前）；都推进不了时，若没有待发生事件就是死锁，否则把时间推进到下一个事件并执行它（TMA 落地、MMA 结束）。
- 生产者每块：等 empty（除非 `skip_acquire`）→ 检查 stage 是否仍被占用（是就 RaceError）→ `arrive_and_expect_tx` → 安排 L 之后的落地事件（写 stage 内容并 `complete_tx`）→ increment。
- 消费者每块：等 full → 检查 stage 里是不是第 k 块（不是就 RaceError）→ sleep T → release（除非 `no_release`）→ increment。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 协议保证同一个 stage 上生产者最多领先消费者一轮：生产者要 acquire 第 r+1 轮，必须先等消费者 release 第 r 轮。所以等待方看到的 phase 只可能是“当前”或“上一个”，1 位足以区分。如果绕过 acquire（`skip_acquire`），领先就可能超过一轮，奇偶就会混淆。
2. 两种写法等价，但 phase 1 起步不需要任何线程在初始化后再做一次 arrive，也就不需要额外的同步：新 barrier 的“上一个 phase”天然算完成。CUTLASS 的注释原话是 “Producer starts with an opposite phase as the buffers are initially empty”。
3. 一个 stage 的周期是 L + T（落地 L，再被算 T 才释放），S 个 stage 轮转，稳态间隔 `(L + T) / S`。要它 ≤ T，需要 `S ≥ L/T + 1`，取整得 `ceil(L/T) + 1`。多出来的那 1 个是“正在被 MMA 占用”的 stage：它在算的时候不能同时在搬。
:::
:::

## 参考解

::: solution solutions/d2_mbarrier_pipeline.py
{{code: solutions/d2_mbarrier_pipeline.py}}
:::

## 面试官追问 & 对照源码

- cluster 里 multicast 时，一个 stage 的 empty barrier 要等几个 consumer arrive？对照 `PipelineTmaAsync` 构造函数里的 `multicast_consumer_arrival_count` 和 `dst_blockid_`。
- 消费者的 `consumer_release` 放在 MMA 发射之后还是 MMA 完成之后？WGMMA 是异步的：CUTLASS 主循环为什么把 release 推迟一拍，配合 `warpgroup_wait<K>`？对照 [[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@mma|collective mma]] 循环。
- SMEM 一共 228 KB，一个 128×128×64 的 bf16 stage 要多大？最多能放几个 stage，还剩多少给 epilogue？
- 如果 L 不是常数，而是在 500 到 900 之间抖动，`min_stages` 该按什么取？
