"""C2 · 编译期合法性检查：CuTe 的 static_assert 与一个 GEMM 配置的完整体检 — reference solution.

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


def _cdiv(a, b):
    return -(-a // b)


def _known(*xs):
    return all(x is not None for x in xs)


def _composition_mode(lhs, rs, rd):
    ls, ld = coalesce_flat(*lhs)
    if rd == 0 or len(ls) == 1:
        return []
    rest_shape, rest_stride = rs, rd
    for curr_shape in ls[:-1]:
        if _known(rest_stride, curr_shape):
            if not (rest_stride % curr_shape == 0 or rest_stride < curr_shape):
                return ["Stride Divisibility Condition"]
        if _known(rest_stride, curr_shape):
            next_shape = _cdiv(curr_shape, abs(rest_stride))
            next_stride = _cdiv(abs(rest_stride), curr_shape) * (1 if rest_stride >= 0 else -1)
        else:
            next_shape = next_stride = None
        if next_shape == 1 or rest_shape == 1:
            rest_stride = next_stride
            continue
        new_shape = min(next_shape, rest_shape) if _known(next_shape, rest_shape) else None
        if _known(new_shape, rest_shape) and rest_shape % new_shape != 0:
            return ["Shape Divisibility Condition"]
        rest_shape = rest_shape // new_shape if _known(new_shape, rest_shape) else None
        rest_stride = next_stride
    return []


def composition_check(lhs, rhs) -> list:
    """CuTe composition 的两条静态整除条件。"""
    rs, rd = rhs
    pairs = list(zip(rs, rd)) if isinstance(rs, tuple) else [(rs, rd)]
    out = []
    for s, d in pairs:          # 右分配律：rhs 的每个模式单独和 lhs 复合
        for msg in _composition_mode(lhs, s, d):
            if msg not in out:
                out.append(msg)
    return out


def swizzle_check(B, M, S) -> list:
    out = []
    if M < 0:
        out.append("MBase must be positive.")
    if B < 0:
        out.append("BBits must be positive.")
    if abs(S) < B:
        out.append("abs(SShift) must be more than BBits.")
    return out


def tiled_copy_check(atom_bits, elem_bits, atom_threads, thr_size, val_size) -> list:
    out = []
    atom_vals = atom_bits // elem_bits
    if thr_size % atom_threads != 0:
        out.append("TiledCopy uses too few thrs for selected CopyAtom")
    if val_size % atom_vals != 0:
        out.append("TiledCopy uses too few vals for selected CopyAtom")
    return out


def tile_to_shape_check(block_shape, target_shape) -> list:
    if len(block_shape) > len(target_shape):
        return ["Rank of layout must be <= rank of target shape."]
    block = tuple(block_shape) + (1,) * (len(target_shape) - len(block_shape))
    if all(t is not None for t in target_shape):
        if any(t % b != 0 for t, b in zip(target_shape, block)):
            return ["tile_to_shape: block shape does not divide the target shape."]
    return []


def predict_compiles(desc):
    kind = desc[0]
    if kind == "composition":
        msgs = composition_check(desc[1], desc[2])
    elif kind == "swizzle":
        msgs = swizzle_check(*desc[1:])
    elif kind == "tiled_copy":
        msgs = tiled_copy_check(*desc[1:])
    elif kind == "tile_to_shape":
        msgs = tile_to_shape_check(desc[1], desc[2])
    elif kind == "always_ok":
        msgs = []
    else:
        raise ValueError(f"unknown snippet kind {kind!r}")
    return (not msgs, msgs[0] if msgs else "")


def check_config(cfg: GemmConfig) -> list:
    v = []
    add = lambda code, layer, msg: v.append(Violation(code, layer, msg))
    thr_m, thr_k = cfg.copy_thr
    val_m, val_k = cfg.copy_val
    elem_bits = 8 * cfg.elem_bytes
    if cfg.swizzle is not None and swizzle_check(*cfg.swizzle):
        add("swizzle_params", "cute", swizzle_check(*cfg.swizzle)[0])
    tc = tiled_copy_check(cfg.copy_atom_bits, elem_bits, 1, prod(cfg.copy_thr), prod(cfg.copy_val))
    if tc:
        add("copy_vals", "cute", tc[0])
    if prod(cfg.copy_thr) != cfg.threads:
        add("copy_threads", "cutlass", f"copy thread layout has {prod(cfg.copy_thr)} threads, CTA has {cfg.threads}")
    if 32 * prod(cfg.mma_warps) != cfg.threads:
        add("mma_threads", "cutlass", f"TiledMMA needs {32 * prod(cfg.mma_warps)} threads, CTA has {cfg.threads}")
    if cfg.BM % (thr_m * val_m) or cfg.BK % (thr_k * val_k):
        add("copy_tile", "cutlass", f"copy tile {(thr_m * val_m, thr_k * val_k)} does not divide CTA tile {(cfg.BM, cfg.BK)}")
    am, an, ak = cfg.mma_atom
    wm, wn, wk = cfg.mma_warps
    if cfg.BM % (am * wm) or cfg.BN % (an * wn) or cfg.BK % (ak * wk):
        add("mma_tile", "cutlass", f"MMA tile {(am * wm, an * wn, ak * wk)} does not divide {(cfg.BM, cfg.BN, cfg.BK)}")
    if val_k * cfg.elem_bytes * 8 != cfg.copy_atom_bits:
        add("vector_bytes", "cutlass", f"{val_k} contiguous elements are {val_k * cfg.elem_bytes} B, atom moves {cfg.copy_atom_bits // 8} B")
    vb = cfg.copy_atom_bits // 8
    if (cfg.K * cfg.elem_bytes) % vb or (cfg.BK * cfg.elem_bytes) % vb:
        add("alignment", "cutlass", f"row bytes are not multiples of the {vb} B vector")
    if cfg.swizzle is not None:
        B, Mb, S = cfg.swizzle
        if (cfg.BM * cfg.BK) % (2 ** (B + Mb + S)):
            add("swizzle_span", "cutlass", f"swizzle span 2^{B + Mb + S} elements does not divide the {cfg.BM}x{cfg.BK} tile")
        if 2 ** Mb < val_k:
            add("swizzle_vector", "cutlass", f"Swizzle keeps runs of {2 ** Mb} elements, copy vector has {val_k}")
    if cfg.use_ldmatrix:
        bad = cfg.elem_bytes != 2 or (cfg.BK * cfg.elem_bytes) % 16
        if cfg.swizzle is not None and 2 ** cfg.swizzle[1] * cfg.elem_bytes < 16:
            bad = True
        if bad:
            add("ldmatrix", "cutlass", "ldmatrix needs 16-byte contiguous rows of 16-bit elements")
    smem = cfg.stages * (cfg.BM + cfg.BN) * cfg.BK * cfg.elem_bytes
    if smem > cfg.smem_limit:
        add("smem_capacity", "cutlass", f"{smem} B of shared memory > {cfg.smem_limit} B")
    if cfg.M % cfg.BM or cfg.N % cfg.BN or cfg.K % cfg.BK:
        add("ragged_tile", "runtime", "problem size is not a multiple of the tile: needs predication")
    return v
