"""C2 · 编译期合法性检查：CuTe 的 static_assert 与一个 GEMM 配置的完整体检 — your implementation (run ./check.sh c2).

CuTe 的“零成本抽象”有两面：静态的 shape / stride 让编译器把索引折叠成几条指令（C1），
也让很多错误在**编译期**就被 static_assert 拦下来，而不是在 GPU 上算出错的结果。这道题分两层：
1. 复刻几条真实的 CuTe static_assert（结果和真实编译器对拍，tests/golden/c2_compile_checks.json
   由 oracle/c2_try_compile.sh 用真实 CuTe 头文件编译 oracle/c2_snippets/*.cpp 得到）：
   - composition 的 “Stride Divisibility Condition” 与 “Shape Divisibility Condition”
     （cute/layout.hpp 的 detail::composition_impl；只在两边都是静态整数时检查，动态值跳过）；
   - Swizzle<B,M,S> 的三条参数检查（cute/swizzle.hpp）；
   - TiledCopy 的 “uses too few thrs / vals for selected CopyAtom”（cute/atom/copy_atom.hpp）；
   - tile_to_shape 的整除检查（cute/layout.hpp）。
2. 对一个完整 GEMM 配置做体检：把 CuTe 会拦下的、CUTLASS 的 collective 会拦下的、以及
   “能编译但需要运行时谓词或性能会差”的问题分别列出来。

约定
- 静态整数用 int 表示；动态（运行时才知道）的值用 None 表示，涉及它的检查一律跳过（CuTe 的
  “weak divisibility condition”：只在能静态验证时验证）。
- 扁平 layout 用 (shape_tuple, stride_tuple) 或单个 (int, int) 表示 rank-1。

API
    coalesce_flat(shape, stride) -> (shape, stride)      given：静态扁平 coalesce（去掉尺寸 1，s_i*d_i == d_{i+1} 合并）
    Violation(code, layer, message)                       given：layer ∈ {"cute", "cutlass", "runtime"}
    GemmConfig(...)                                       given：见下面的字段说明
    SNIPPETS                                              given：oracle/c2_snippets里每个程序的结构化描述
    RUNNING_EXAMPLE                                       given：仓库的运行例子配置

    composition_check(lhs, rhs) -> list[str]
        lhs = (shape, stride) 扁平；rhs = (shape, stride)，shape/stride 是 int（rank1）或扁平 tuple。
        rhs 是 tuple 时对每个模式分别检查（右分配律），消息按出现顺序合并、去重。
        对每个 rank-1 的 rhs 模式 (rs, rd)：
          1. lhs 先 coalesce_flat；若 rd == 0，或 coalesce 后 lhs 只有一个模式 → 合法。
          2. rest_shape, rest_stride = rs, rd；依次处理 lhs 除最后一个之外的每个模式 curr_shape：
             - 若 rest_stride 和 curr_shape 都已知：要求 rest_stride % curr_shape == 0 或
               rest_stride < curr_shape，否则记 "Stride Divisibility Condition"（并停止检查这个模式）；
             - next_shape = ceil(curr_shape / |rest_stride|)，next_stride = ceil(|rest_stride| / curr_shape)·sign；
             - 若 next_shape == 1 或 rest_shape == 1：只更新 rest_stride = next_stride；
             - 否则 new_shape = min(next_shape, rest_shape)；若都已知且 rest_shape % new_shape != 0，
               记 "Shape Divisibility Condition"（并停止）；rest_shape //= new_shape，rest_stride = next_stride。
             未知值参与的算术结果也是未知（None）。
    swizzle_check(B, M, S) -> list[str]
        M < 0 → "MBase must be positive."；B < 0 → "BBits must be positive."；
        |S| < B → "abs(SShift) must be more than BBits."（按这个顺序）。
    tiled_copy_check(atom_bits, elem_bits, atom_threads, thr_size, val_size) -> list[str]
        AtomNumVal = atom_bits // elem_bits（每个线程一次搬的元素数）；
        thr_size % atom_threads != 0 → "TiledCopy uses too few thrs for selected CopyAtom"；
        val_size % AtomNumVal != 0 → "TiledCopy uses too few vals for selected CopyAtom"。
    tile_to_shape_check(block_shape, target_shape) -> list[str]
        len(block) > len(target) → "Rank of layout must be <= rank of target shape."；
        否则 block 末尾补 1，若 target 全部已知且某一维 target % block != 0 →
        "tile_to_shape: block shape does not divide the target shape."
    predict_compiles(desc) -> (bool, str)
        desc 是 SNIPPETS 的一项：("composition", lhs, rhs) / ("swizzle", B, M, S) /
        ("tiled_copy", atom_bits, elem_bits, atom_threads, thr_size, val_size) /
        ("tile_to_shape", block_shape, target_shape) / ("always_ok", reason)。
        返回 (是否能编译, 第一条错误消息或 "")。
    check_config(cfg) -> list[Violation]
        按下面的顺序检查，返回所有违反项（code 唯一）：
          code              layer      条件（违反时报告）
          swizzle_params    cute       cfg.swizzle 不为 None 且 swizzle_check 非空
          copy_vals         cute       tiled_copy_check(copy_atom_bits, 8*elem_bytes, 1, size(copy_thr), size(copy_val)) 非空
          copy_threads      cutlass    size(copy_thr) != threads
          mma_threads       cutlass    32 * prod(mma_warps) != threads
          copy_tile         cutlass    BM % (thr_m*val_m) 或 BK % (thr_k*val_k) 不为0（copy_thr/val 是 (M,K) 两维）
          mma_tile          cutlass    BM % (atom_m*warps_m)、BN % (atom_n*warps_n)、BK % (atom_k*warps_k) 有不为 0 的
          vector_bytes      cutlass    val_k * elem_bytes * 8 != copy_atom_bits（每线程连续 K 元素必须正好是一个 atom）
          alignment         cutlass    K*elem_bytes 或 BK*elem_bytes 不是 copy_atom_bits//8 的倍数
          swizzle_span      cutlass    swizzle 存在且 (BM*BK) % 2**(B+M+S) != 0
          swizzle_vector    cutlass    swizzle 存在且 2**M < val_k（swizzle 会把一个向量拆开）
          ldmatrix          cutlass    use_ldmatrix 且（elem_bytes != 2，或 BK*elem_bytes % 16 != 0，
                                       或 swizzle 存在且 2**M * elem_bytes < 16）
          smem_capacity     cutlass    stages * (BM + BN) * BK * elem_bytes > smem_limit
          ragged_tile       runtime    M % BM、N % BN、K % BK 有不为 0 的（能编译，但需要谓词）
        message 是一句话的说明（内容不测试，code 与 layer 被测试）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod


def coalesce_flat(shape, stride):
    """静态扁平 coalesce；rank-1 输入可以是 int。"""
    if isinstance(shape, int):
        shape, stride = (shape,), (stride,)
    modes = [(s, d) for s, d in zip(shape, stride) if s != 1]
    if not modes:
        return (1,), (0,)
    out = [modes[0]]
    for s, d in modes[1:]:
        ps, pd = out[-1]
        if ps is not None and pd is not None and d is not None and ps * pd == d:
            out[-1] = (ps * s if s is not None else None, pd)
        else:
            out.append((s, d))
    return tuple(s for s, _ in out), tuple(d for _, d in out)


@dataclass(frozen=True)
class Violation:
    code: str
    layer: str          # "cute" | "cutlass" | "runtime"
    message: str


@dataclass(frozen=True)
class GemmConfig:
    """一个 SM80 风格 GEMM 的静态配置。"""
    M: int = 64
    N: int = 64
    K: int = 64
    BM: int = 32
    BN: int = 32
    BK: int = 16
    threads: int = 128
    elem_bytes: int = 2                       # half
    copy_atom_bits: int = 64                  # UniversalCopy<uint64_t>：4 个 half
    copy_thr: tuple = (32, 4)                 # (M, K) 方向的线程数
    copy_val: tuple = (1, 4)                  # 每线程 (M, K) 方向的值数
    swizzle: tuple | None = (2, 2, 2)         # (B, M, S) 或 None
    mma_atom: tuple = (16, 8, 16)             # m16n8k16
    mma_warps: tuple = (2, 2, 1)
    use_ldmatrix: bool = False
    stages: int = 1
    smem_limit: int = 48 * 1024


RUNNING_EXAMPLE = GemmConfig()

SNIPPETS = {
    "comp_ok": ("composition", ((4, 8), (8, 1)), ((2, 4), (1, 2))),
    "comp_stride_fail": ("composition", ((4, 6), (1, 5)), (3, 6)),
    "comp_shape_fail": ("composition", ((4, 6), (1, 5)), (3, 2)),
    "comp_coalesced_ok": ("composition", ((4, 6), (1, 4)), (3, 6)),
    "comp_dynamic_ok": ("composition", ((4, 6), (1, 5)), (3, None)),
    "swizzle_ok": ("swizzle", 2, 2, 2),
    "swizzle_shift_fail": ("swizzle", 3, 3, 2),
    "tiled_copy_ok": ("tiled_copy", 64, 16, 1, 128, 4),
    "tiled_copy_vec_fail": ("tiled_copy", 128, 16, 1, 128, 4),
    "tile_to_shape_ok": ("tile_to_shape", (8, 64), (128, 64)),
    "tile_to_shape_fail": ("tile_to_shape", (8, 64), (12, 64)),
    "tiled_mma_ok": ("always_ok", "partition_C of a divisible 32x32 tile"),
    "tiled_mma_bad_tile": ("always_ok", "partition of a ragged tile is a runtime concern"),
    "local_tile_ok": ("always_ok", "local_tile of a divisible tile"),
    "local_tile_ragged": ("always_ok", "local_tile allows a ragged last tile"),
}


def composition_check(lhs, rhs) -> list:
    """CuTe composition 的两条静态整除条件。"""
    raise NotImplementedError("composition_check")


def swizzle_check(B, M, S) -> list:
    raise NotImplementedError("swizzle_check")


def tiled_copy_check(atom_bits, elem_bits, atom_threads, thr_size, val_size) -> list:
    raise NotImplementedError("tiled_copy_check")


def tile_to_shape_check(block_shape, target_shape) -> list:
    raise NotImplementedError("tile_to_shape_check")


def predict_compiles(desc):
    raise NotImplementedError("predict_compiles")


def check_config(cfg: GemmConfig) -> list:
    raise NotImplementedError("check_config")
