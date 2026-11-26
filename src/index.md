## 怎么用这套练习

20 道题沿着一条主线：**从零设计 Tiled GEMM → 用 Layout Algebra 组合映射 → 理解 CuTe 的编译期优化 → 加入 TMA、Pipeline、Warp Specialization、PDL → 拼成完整成熟的 GEMM → 在 RTX 5090 上用真实 CuTe 写出来。** Layout 相关题目的期望答案来自本机编译运行的真实 CuTe（`oracle/<stem>.cpp → tests/golden/`）；异步执行相关题目是照着 CUTLASS 源码写的协议与时间线模拟器。

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy
./check.sh a1                 # 测你的 mine/a1_*.py
./check.sh a1 --solutions     # 测参考解
./check.sh all

# 上机（可选，不需要 GPU）：用真实 CuTe 重新生成 golden、自己做实验
tools/fetch_headers.sh        # 下载 CUTLASS、CCCL、CUDA 公共头文件到 third_party/
./check.sh oracle a1
playground/run.sh playground/a1_cta_tiling.cpp

# Track F：没有 GPU 时跑主机部分和类型检查；在 RTX 5090 上再加编译、对拍、计时
./check.sh f3 -s
python3 tools/cu_drill.py run mine/f3_pipelined_gemm.cu 4096 4096 4096
```

1. **先读知识页**（按顺序）：总览 → 硬件心智模型 → 执行模型 → 怎样定 CTA / tile / warp 的数量 → BM、BN、BK 各管什么 → 配置决策：一步一步定配置（含 split-K）→ 端到端例子：1024×1024 · 1024×256 → 从零设计 Tiled GEMM（Track A）→ Layout Algebra 与编译（Track B、C）→ 时间维度：TMA、Pipeline、WS、PDL（Track D）→ 真实代码导读（Track E 前必读）→ 参考资料。
2. **每道题**：读讲解页（含“对照文档”与“上机验证”）→ 在 `mine/` 里实现 → `./check.sh <id>` → 卡住再看 L1 → L2 → L3 → 对照参考解和真实 CUTLASS 代码。

名字以 `test_given_` 或 `test_golden_` 开头的测试检查的是已经给出的代码和 golden 本身，从一开始就会通过；其余测试在你实现之前都会失败。

## 建议顺序与依赖

- **A1 → A2 → A3 → A4 → A5**：五个空间映射，最后串成一个可验证的 64×64 GEMM。这是全套的主线。
- **B1、B2**：用代数重写 A1–A3、推导向量宽度（需要 A 组的结论）。
- **C1、C2**：编译期化简与合法性检查（可以在 B 之后或之前做）。
- **D1 → D2 → D3 → D4**：TMA、mbarrier 流水线、warp specialization、PDL。
- **E1 → E2**：完整的 Hopper 风格 GEMM（用到 A、D 的全部部件），最后在约束下搜索配置。
- **F1 → F2 → F3 → F4 → F5**：上机。把前面的映射和流水线写成真正在 GPU 上跑的 CuTe C++ kernel：SIMT → Tensor Core → 多级流水线 + ldmatrix → 边界与 epilogue → 配置扫描、split-K 与 cuBLAS 对比。建议 A 组做完后开始 F1、F2，D2 之后做 F3。

## 需要什么硬件

Track A–E 的测试只用 CPU（Python + numpy），几秒跑完。`oracle/` 和 `playground/` 里的 C++ 程序只在主机上编译真实的 CuTe / CUTLASS 头文件，也不需要 GPU。TMA、mbarrier、`setmaxnreg`、PDL 这些机制必须在 Hopper（SM90）或更新的 GPU 上才能真正运行；讲解页里凡是“在 GPU 上验证”的说法都标明了没有在本机核对。

Track F 的每道题分两部分：

| 部分 | 需要什么 | 不满足时 |
|---|---|---|
| 主机部分（`mine/f*.py`，和真实 CuTe 的 golden 对拍） | 只要 Python | — |
| `.cu` 类型检查 | `nvcc`，或带 NVPTX 后端的 `clang++`（macOS：`brew install llvm`；也可以用 `CUTE_DRILL_CLANG` 指定） | 测试显示 skipped |
| 编译、对拍、计时 | NVIDIA GPU + CUDA ≥ 12.8 + `tools/fetch_headers.sh` 取的 CUTLASS | 测试显示 skipped |

默认目标是 RTX 5090（`-arch=sm_120`）。F1–F5 只用 `sm_80` 的指令（cp.async、ldmatrix、mma.sync），所以 Ampere 以后的卡都能跑：设 `CUTE_DRILL_ARCH=sm_86`（或 `sm_89`、`sm_90`）即可。SMEM 上限不同的卡（例如 H100 每 CTA 227 KB）上，F5 的合法配置会不一样。F5 的 cuBLAS 基准会自动找 CUDA 自带的 cuBLAS 或 pip 包 `nvidia-cublas`（PyTorch 会装），找不到就跳过基准。
