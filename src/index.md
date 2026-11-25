## 怎么用这套练习

15 道题沿着一条主线：**从零设计 Tiled GEMM → 用 Layout Algebra 组合映射 → 理解 CuTe 的编译期优化 → 加入 TMA、Pipeline、Warp Specialization、PDL → 拼成完整成熟的 GEMM。** Layout 相关题目的期望答案来自本机编译运行的真实 CuTe（`oracle/<stem>.cpp → tests/golden/`）；异步执行相关题目是照着 CUTLASS 源码写的协议与时间线模拟器。

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy
./check.sh a1                 # 测你的 mine/a1_*.py
./check.sh a1 --solutions     # 测参考解
./check.sh all

# 上机（可选，不需要 GPU）：用真实 CuTe 重新生成 golden、自己做实验
tools/fetch_headers.sh        # 下载 CUTLASS、CCCL、CUDA 公共头文件到 third_party/
./check.sh oracle a1
playground/run.sh playground/a1_cta_tiling.cpp
```

1. **先读知识页**（按顺序）：总览 → 从零设计 Tiled GEMM（Track A）→ Layout Algebra 与编译（Track B、C）→ 时间维度：TMA、Pipeline、WS、PDL（Track D）→ 真实代码导读（Track E 前必读）→ 参考资料。
2. **每道题**：读讲解页（含“对照文档”与“上机验证”）→ 在 `mine/` 里实现 → `./check.sh <id>` → 卡住再看 L1 → L2 → L3 → 对照参考解和真实 CUTLASS 代码。

名字以 `test_given_` 或 `test_golden_` 开头的测试检查的是已经给出的代码和 golden 本身，从一开始就会通过；其余测试在你实现之前都会失败。

## 建议顺序与依赖

- **A1 → A2 → A3 → A4 → A5**：五个空间映射，最后串成一个可验证的 64×64 GEMM。这是全套的主线。
- **B1、B2**：用代数重写 A1–A3、推导向量宽度（需要 A 组的结论）。
- **C1、C2**：编译期化简与合法性检查（可以在 B 之后或之前做）。
- **D1 → D2 → D3 → D4**：TMA、mbarrier 流水线、warp specialization、PDL。
- **E1 → E2**：完整的 Hopper 风格 GEMM（用到 A、D 的全部部件），最后在约束下搜索配置。

## 需要什么硬件

全部测试只用 CPU（Python + numpy），几秒跑完。`oracle/` 和 `playground/` 里的 C++ 程序只在主机上编译真实的 CuTe / CUTLASS 头文件，也不需要 GPU。TMA、mbarrier、`setmaxnreg`、PDL 这些机制必须在 Hopper（SM90）或更新的 GPU 上才能真正运行；讲解页里凡是“在 GPU 上验证”的说法都标明了没有在本机核对。
