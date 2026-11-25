# cute-tiled-gemm-drills

15 one-hour drills that design a Tiled GEMM from first principles and then rebuild it with CuTe:
the five spatial mappings (CTA tile, Copy TV, SMEM swizzle, MMA fragments, accumulator), CuTe's
Layout Algebra and what the compiler folds and checks at compile time, and the time dimension of a
Hopper GEMM (TMA, mbarrier pipelines, warp specialization, PDL). Every layout answer is checked
against real CuTe compiled on the host (CUTLASS pinned at `af95104`), the pipeline state against
the real `cutlass::PipelineState`; the rest are protocol and timeline simulators written from the
CUTLASS source. Tests run on CPU only. Quick start: `./check.sh a1`. The rest of this README is in
Chinese.

---

15 道一小时左右的练习，沿着一条主线：先只从硬件需求出发，推导一个 64×64 Tiled GEMM 需要的五种映射，把它们拼成一个可验证的 GEMM；再用 Layout Algebra 重新表达这些映射，看 CuTe 在编译期怎样化简和检查它们；最后加入 TMA、mbarrier 流水线、warp specialization 和 PDL，得到一个结构完整的现代 GEMM，并用代价模型选配置。

- 在线浏览：<https://kzhou003.github.io/cute-tiled-gemm-drills/> · 本地打开 `index.html` · PDF：[`cute-tiled-gemm-drills.pdf`](cute-tiled-gemm-drills.pdf)
- `mine/` 接口骨架 · `tests/` 验收测试 · `solutions/` 带注释的参考解
- 每题的讲解与提示：`hints/<stem>.html`；领域知识：`knowledge/`
- `oracle/` 用真实 CuTe 生成 `tests/golden/` 的 C++ 程序 · `playground/` 可以直接编译运行的 CuTe 小实验

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

## 需要什么硬件

- **测试**：只要 CPU（Python + numpy），整套几秒跑完。
- **oracle / playground**：只要主机编译器，不需要 GPU。
- **真正运行这些机制**：TMA、mbarrier、`setmaxnreg`、WGMMA、PDL 要在 Hopper（SM90）或更新的 GPU 上才能跑。讲解页里凡是“在 GPU 上可以这样验证”的说法都标明了没有在本机核对。RTX 5090（SM120）有 TMA、mbarrier 和 PDL，但没有 WGMMA；在 5090 上真机写 GEMM / FlashAttention 的练习见 [cuda-kernels-dev-drills](https://github.com/kzhou003/cuda-kernels-dev-drills)。

## 怎么做一道题

1. 读讲解页（5–10 分钟），包括“对照文档”和“上机验证”。
2. 在 `mine/` 里实现，限时为卡片上的分钟数。
3. 跑测试；卡住 10–15 分钟再按 L1 → L2 → L3 看提示。
4. 对照参考解、真实 CUTLASS 代码和面试官追问。

建议顺序：A1 → A5（主线）→ B1、B2 → C1、C2 → D1 → D4 → E1、E2。

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

## 目录结构

```text
drills.json            配置：题目、track、固定的 CUTLASS commit
src/                   页面源文件（index.md、knowledge/*.md、hints/*.md）
index.html knowledge/ hints/ cute-tiled-gemm-drills.pdf   构建产物
mine/ tests/ solutions/ tests/golden/
oracle/                *.cpp → tests/golden/*.json（run.sh、c1_asm.sh、c2_try_compile.sh、c2_sweep.py）
playground/            CuTe 小实验（run.sh）
tools/                 fetch_headers.sh、cute_given.py
```

## 重新生成页面

页面由 practice-drills 工具链生成：

```bash
bash <toolkit>/run.sh build_drills.py drills.json --strict --shots --links --pdf
bash <toolkit>/run.sh gen_stubs.py drills.json --check
```

## License

原创内容（题目、讲解、测试、参考解）MIT。链接到的 CUTLASS / CuTe 源码遵循 NVIDIA 的 BSD-3-Clause 许可证；`tools/fetch_headers.sh` 下载的头文件不随本仓库分发。
