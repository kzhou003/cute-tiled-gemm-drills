# cute-tiled-gemm-drills

20 one-hour drills that design a Tiled GEMM from first principles and then rebuild it with CuTe:
the five spatial mappings (CTA tile, Copy TV, SMEM swizzle, MMA fragments, accumulator), CuTe's
Layout Algebra and what the compiler folds and checks at compile time, and the time dimension of a
Hopper GEMM (TMA, mbarrier pipelines, warp specialization, PDL). Every layout answer is checked
against real CuTe compiled on the host (CUTLASS pinned at `af95104`), the pipeline state against
the real `cutlass::PipelineState`; the rest are protocol and timeline simulators written from the
CUTLASS source. Track F then writes the GEMM as real CuTe C++ kernels on an RTX 5090 (SIMT → Tensor
Core → multi-stage cp.async + ldmatrix pipeline → ragged edges and a BF16 epilogue → a config sweep
with split-K against cuBLAS). Six knowledge pages build the hardware picture and the sizing rules
(RTX 5090 numbers, execution model, BM/BN/BK, a 9-step config procedure with split-K / Stream-K, an
end-to-end example). Tests A–E run on CPU only; Track F's host parts too, its GPU parts skip without
a GPU. Quick start: `./check.sh a1`. The rest of this README is in Chinese.

---

20 道一小时左右的练习，沿着一条主线：先只从硬件需求出发，推导一个 64×64 Tiled GEMM 需要的五种映射，把它们拼成一个可验证的 GEMM；再用 Layout Algebra 重新表达这些映射，看 CuTe 在编译期怎样化简和检查它们；然后加入 TMA、mbarrier 流水线、warp specialization 和 PDL，得到一个结构完整的现代 GEMM，并用代价模型选配置；最后（Track F）在 RTX 5090 上用真实 CuTe 把它写出来：SIMT → Tensor Core → 多级流水线 + ldmatrix → 边界与 epilogue → 配置扫描、split-K 与 cuBLAS 对比。

开头 6 页知识页先建立硬件画面和数字直觉：硬件心智模型（工厂类比 + 5090 的真实数字）、执行模型、怎样定 CTA / tile / warp 的数量、BM / BN / BK 各管什么、9 步配置决策（含 split-K / Stream-K）、一个 1024×1024 · 1024×256 的端到端例子。

- 在线浏览：<https://kzhou003.github.io/cute-tiled-gemm-drills/> · 本地打开 `index.html` · PDF：[`cute-tiled-gemm-drills.pdf`](cute-tiled-gemm-drills.pdf)
- `mine/` 接口骨架 · `tests/` 验收测试 · `solutions/` 带注释的参考解
- 每题的讲解与提示：`hints/<stem>.html`；领域知识：`knowledge/`
- `oracle/` 用真实 CuTe 生成 `tests/golden/` 的 C++ 程序 · `playground/` 可以直接编译运行的 CuTe 小实验
- Track F：`mine/f*.cu` + `mine/f*.py` · `gpu/` 共用的头文件（计时、对拍、五个映射的 `Config`、模板化主循环）· `tools/cu_drill.py` 编译 / 运行 / 类型检查

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy
./check.sh a1                # 测你写的 mine/a1_*.py
./check.sh a1 --solutions    # 测参考解
./check.sh all
```

名字以 `test_given_` 或 `test_golden_` 开头的测试检查的是已经给出的代码和 golden 本身，从一开始就会通过；其余测试在你实现之前都会失败。

## 上机（可选，不需要 GPU）

`oracle/` 和 `playground/` 里的程序只在主机上编译真实的 CuTe / CUTLASS 头文件：

```bash
tools/fetch_headers.sh                            # CUTLASS @ af95104、CCCL、CUDA 公共头文件 → third_party/
./check.sh oracle a1                              # 用真实 CuTe 重新生成 tests/golden/a1_cta_tiling.json
./check.sh oracle all                             # 全部 golden（C2 要编译 63 个程序，约 3 分钟）
playground/run.sh playground/a1_cta_tiling.cpp    # 编译运行一个实验
```

需要 `clang++`（macOS 自带的即可）或 `g++`，用 `CXX=...` 指定。

## Track F：在 GPU 上跑

```bash
tools/fetch_headers.sh                                     # 固定版本的 CUTLASS
./check.sh f1                                              # 主机部分 + .cu 类型检查；没有 GPU 时 GPU 测试 skipped
./check.sh f3 -s                                           # 在 5090 上：再加 nvcc 编译、对拍、计时（-s 打印 JSON）
python3 tools/cu_drill.py run mine/f3_pipelined_gemm.cu 4096 4096 4096
CUTE_DRILL_ARCH=sm_86 ./check.sh f3 -s                     # 别的卡（F1–F5 只用 sm_80 指令）
python3 tools/gen_cu_stubs.py --check                      # mine/*.cu 的 DRILL_TODO 骨架
```

每个程序最后打印一行 JSON：`ok`（误差 ≤ 容差）、`max_rel_err`、`ms`、`tflops`。参考结果来自 `gpu/common.cuh` 里“一个线程算一个输出”的朴素 kernel。`mine/*.cu` 里没写的地方是 `DRILL_TODO("…")`：一个必然失败的 `static_assert`，所以没写完时编译器会把每个 TODO 的说明列出来。

参考解在 RTX 5090（CUDA 13.0）上的实测（10 次平均，量级参考）：

| 题 | 1024³ | 4096³ | 8192³ |
|---|---|---|---|
| F1 SIMT 128×128×8 | — | ≈ 32 | — |
| F2 Tensor Core，单级 SMEM（每 SM 2 个 CTA） | ≈ 47 | ≈ 106 | ≈ 111 |
| F3 3 级流水线 + ldmatrix（每 SM 1 个 CTA） | ≈ 70 | ≈ 107 | ≈ 111 |
| F4 F3 + 谓词 + BF16 epilogue | — | ≈ 106 | — |
| cuBLAS（`cublasGemmEx`，BF16 → FP32） | — | ≈ 218 | — |

单位 TFLOPS。F5 的扫描里最好的配置在 4096³、1024×1024×16384、2048×512×8192 上达到 cuBLAS 的 0.53–0.60。F2 和 F3 在大问题上打平，原因见 F3 讲解页的“上机验证”。

## 需要什么硬件

- **测试**：Track A–E 和 Track F 的主机部分只要 CPU（Python + numpy），几秒跑完。Track F 的 `.cu` 类型检查要 `nvcc` 或带 NVPTX 后端的 `clang++`（macOS：`brew install llvm`），GPU 部分要 NVIDIA GPU + CUDA ≥ 12.8，默认 RTX 5090（`sm_120`），其他 Ampere 以后的卡设 `CUTE_DRILL_ARCH`。
- **oracle / playground**：只要主机编译器，不需要 GPU。
- **真正运行这些机制**：TMA、mbarrier、`setmaxnreg`、WGMMA、PDL 要在 Hopper（SM90）或更新的 GPU 上才能跑。讲解页里凡是“在 GPU 上可以这样验证”的说法都标明了没有在本机核对。RTX 5090（SM120）有 TMA、mbarrier 和 PDL，但没有 WGMMA，所以 Track F 用的是 SM80 风格的 cp.async + ldmatrix + mma.sync；在 5090 上真机写 GEMM / FlashAttention 的更多练习见 [cuda-kernels-dev-drills](https://github.com/kzhou003/cuda-kernels-dev-drills)。

## 怎么做一道题

1. 读讲解页（5–10 分钟），包括“对照文档”和“上机验证”。
2. 在 `mine/` 里实现，限时为卡片上的分钟数。
3. 跑测试；卡住 10–15 分钟再按 L1 → L2 → L3 看提示。
4. 对照参考解、真实 CUTLASS 代码和面试官追问。

建议顺序：A1 → A5（主线）→ B1、B2 → C1、C2 → D1 → D4 → E1、E2 → F1 → F5（A 组做完后就可以开始 F1、F2，D2 之后做 F3）。

## 进度表

| ✓ | 编号 | 题目 | 分钟 | 难度 |
|---|----|------|-----|-----|
| ☐ | A1 | [CTA 切分：C tile、K 循环、local_tile 视图与边界谓词](hints/a1_cta_tiling.html) | 45 | 易 |
| ☐ | A2 | [线程怎样搬数据：从 ThrLayout / ValLayout 推出 Copy TV 映射，数 sector 与向量宽度](hints/a2_copy_tv.html) | 55 | 中 |
| ☐ | A3 | [Shared memory 怎样摆：行主序、Swizzle<2,2,2> 与 bank conflict 的实测](hints/a3_smem_swizzle.html) | 55 | 中 |
| ☐ | A4 | [MMA fragment：从 PTX 的 groupID 规则推出 m16n8k16 的 TV_A / TV_B / TV_C，再扩展到 2×2 warp 并在寄存器里累加](hints/a4_mma_fragments.html) | 55 | 难 |
| ☐ | A5 | [把五个映射拼成完整的 Tiled GEMM：逐 CTA、逐 K 片模拟，统计 sector、wavefront 与 mma，并追踪 C[5][7]](hints/a5_assemble_gemm.html) | 65 | 难 |
| ☐ | B1 | [用 Layout Algebra 重写 Track A：local_tile = zipped_divide + 切片，Copy TV = right_inverse(raked_product)，SMEM = Swizzle ∘ Layout](hints/b1_algebra_rewrite.html) | 60 | 难 |
| ☐ | B2 | [一次能搬多宽：max_common_vector 推出向量宽度，swizzle 把它压到 2^M，padding 和对齐让它失效，再选 copy atom](hints/b2_vector_width.html) | 50 | 中 |
| ☐ | C1 | [编译期化简：把 copy 地址写成表达式，做常量折叠、强度削减、coalesce 与循环外提，再和真实编译器的汇编对照](hints/c1_static_folding.html) | 60 | 难 |
| ☐ | C2 | [编译期合法性检查：复刻 CuTe 的 static_assert，并给一个 GEMM 配置做完整体检](hints/c2_compile_checks.html) | 55 | 中 |
| ☐ | D1 | [TMA box：描述符合法性、expect_tx、越界填零与 swizzle 后的 SMEM 位置](hints/d1_tma_box.html) | 55 | 中 |
| ☐ | D2 | [mbarrier 多级流水线：phase 奇偶、PipelineState、producer / consumer 协议与三种同步 bug](hints/d2_mbarrier_pipeline.html) | 60 | 难 |
| ☐ | D3 | [Warp specialization：setmaxnreg 寄存器预算与 Cooperative / Ping-pong 调度](hints/d3_warp_specialization.html) | 55 | 中 |
| ☐ | D4 | [PDL：两个串联 GEMM 的提前启动、等待位置与正确性检查](hints/d4_pdl_chain.html) | 55 | 中 |
| ☐ | E1 | [完整成熟的 Tiled GEMM：TMA → swizzled SMEM 环形缓冲 → mbarrier 流水线 → warp specialization → epilogue → PDL](hints/e1_full_gemm.html) | 75 | 难 |
| ☐ | E2 | [选配置：CTA tile、stage 数、cooperative / pingpong 与 swizzle，在硬件约束下用代价模型排序](hints/e2_config_search.html) | 55 | 中 |
| ☐ | F1 | [第一个真实 CuTe kernel：local_tile + local_partition + SMEM，用 CUDA core（FFMA）算 GEMM](hints/f1_simt_cute_gemm.html) | 60 | 中 |
| ☐ | F2 | [第一个 Tensor Core GEMM：TiledCopy（cp.async 16 B）+ swizzle 过的 SMEM + TiledMMA（mma.sync）](hints/f2_tensorcore_gemm.html) | 75 | 难 |
| ☐ | F3 | [多级 cp.async 流水线 + ldmatrix + 寄存器双缓冲：写出 sgemm_sm80 的完整主循环](hints/f3_pipelined_gemm.html) | 90 | 难 |
| ☐ | F4 | [边界与 epilogue：M、N 不整除时的谓词加载，α·acc + β·C，FP32 → BF16 写回](hints/f4_epilogue_residue.html) | 70 | 中 |
| ☐ | F5 | [在 5090 上扫配置：tile、stage、warp 排列、split-K，用代价模型预测并和 cuBLAS 对比](hints/f5_config_sweep_splitk.html) | 90 | 难 |

## 目录结构

```text
drills.json            配置：题目、track、固定的 CUTLASS commit
src/                   页面源文件（index.md、knowledge/*.md、hints/*.md）
index.html knowledge/ hints/ cute-tiled-gemm-drills.pdf   构建产物
mine/ tests/ solutions/ tests/golden/
oracle/                *.cpp → tests/golden/*.json（run.sh、c1_asm.sh、c2_try_compile.sh、c2_sweep.py）
playground/            CuTe 小实验（run.sh）
tools/                 fetch_headers.sh、cute_given.py、cu_drill.py（Track F 编译 / 运行 / 类型检查）、gen_cu_stubs.py
gpu/                   Track F 共用：common.cuh（对拍、计时、JSON）、tiled_gemm_config.hpp、pipelined_mainloop.cuh
figs/                  讲解页里的静态图（交互图的快照）
```

## 重新生成页面

页面由 practice-drills 工具链生成：

```bash
bash <toolkit>/run.sh build_drills.py drills.json --strict --shots --links --pdf
bash <toolkit>/run.sh gen_stubs.py drills.json --check
```

## License

原创内容（题目、讲解、测试、参考解）MIT。链接到的 CUTLASS / CuTe 源码遵循 NVIDIA 的 BSD-3-Clause 许可证；`tools/fetch_headers.sh` 下载的头文件不随本仓库分发。
