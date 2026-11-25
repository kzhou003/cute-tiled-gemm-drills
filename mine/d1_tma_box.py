"""D1 · TMA box：描述符的合法性、expect_tx 字节数、越界填零与 swizzle 后的 SMEM 位置 — your implementation (run ./check.sh d1).

TMA（Tensor Memory Accelerator，Hopper 起的张量搬运硬件）不再让每个线程算地址：一个线程发一条
`cp.async.bulk.tensor` 指令，带上一个“张量描述符”（CUtensorMap）和 box 的起点坐标，硬件就把整块 box
搬进 shared memory，并按描述符里的 swizzle 模式打散地址。CuTe 在
include/cute/atom/copy_traits_sm90_tma.hpp 的 make_tma_copy_desc 里从 Tensor 的 layout 推出描述符的
维度、字节 stride、box 和 swizzle，并用一串 assert 检查硬件限制；本题复刻这套规则和它的数据效果。

记号
- 所有元组都是“最内层维度在前”（和 CUtensorMap、CuTe 的 TMA 基一致）：gdims=(K, M) 表示 K 连续。
- gdims：每一维的元素个数；gstrides：维度 1.. 的**字节** stride（维度 0 的 stride 隐含为 elem_bytes）。
- box：每一维搬多少个元素；elem_strides：每一维每隔几个元素取一个（默认全 1）。
- swizzle："none" / "32B" / "64B" / "128B"。在字节地址上就是 CuTe 的 Swizzle<B,4,3>，B= 0/1/2/3：
  把“第几个 128 字节行”的低 B 位异或进“行内第几个 16 字节块”。

规则（validate 按这个顺序检查，返回所有违反项，每条以错误码开头）
  E_RANK          1 <= rank <= 5，且 gdims / gstrides / box / elem_strides 的长度一致
  E_ALIGN         base_addr % 16 == 0
  E_GDIM          1 <= gdims[i] <= 2**32
  E_GSTRIDE_ALIGN gstrides[j] % 16 == 0
  E_GSTRIDE_MAX   gstrides[j] < 2**40
  E_BOX           1 <= box[i] <= 256
  E_ESTRIDE       1 <= elem_strides[i] <= 8
  E_INNER_16B     box[0] * elem_bytes % 16 == 0（非 interleave 模式的要求）
  E_SWIZZLE_MODE  swizzle 必须是四种之一
  E_SWIZZLE_SPAN  swizzle != "none" 时 box[0] * elem_bytes <= 32 / 64 / 128

数据效果
- elements_per_dim[i] = ceil(box[i] / elem_strides[i])；SMEM 里是一块稠密 box，最内层连续。
- expect_tx_bytes = prod(elements_per_dim) * elem_bytes：越界部分也算（硬件填 0，照样计入 transaction 字节）。
  这正是生产者在 mbarrier 上 arrive_and_expect_tx 时要报的字节数（D2）。
- box 元素 c（最内层在前）对应全局坐标 g[i] = start[i] + c[i] * elem_strides[i]；任一维 g[i] < 0 或 >= gdims[i]
  就是越界，写 0。全局字节地址 = g[0]*elem_bytes + Σ_{i>=1} g[i]*gstrides[i-1]。
- SMEM 字节位置 = swizzle_byte(线性字节偏移, swizzle)，线性字节偏移 = elem_bytes * colex(c, elements_per_dim)。

API
    SWIZZLE_B / SWIZZLE_SPAN / TmaDesc                given
    validate(desc) -> list[str]
    elements_per_dim(desc) -> tuple
    expect_tx_bytes(desc) -> int
    swizzle_byte(off, mode) -> int                     Swizzle<B,4,3> on byte offsets
    smem_byte_offset(desc, box_coord) -> int
    box_load(gmem: np.ndarray[uint8], desc, start) -> np.ndarray[uint8]   长度 expect_tx_bytes
    unswizzle(desc, smem, dtype) -> np.ndarray         把 SMEM 还原成逻辑 box，shape= 反过来的 elements_per_dim
    tile_element_map(rows, inner, elem_bytes, mode) -> list[int]
                                                        (rows, inner) 的二维 box（inner 连续）按 CuTe colex 顺序
                                                        (m + rows*k) 列出元素偏移，用来和 CuTe 的 tile_to_shape 对拍
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SWIZZLE_B = {"none": 0, "32B": 1, "64B": 2, "128B": 3}
SWIZZLE_SPAN = {"none": 0, "32B": 32, "64B": 64, "128B": 128}


@dataclass(frozen=True)
class TmaDesc:
    """一个简化的 CUtensorMap：最内层维度在前。elem_strides 缺省为全 1。"""
    elem_bytes: int
    gdims: tuple
    gstrides: tuple          # bytes, for dims 1..rank-1
    box: tuple
    elem_strides: tuple | None = None
    swizzle: str = "none"
    base_addr: int = 0

    def estrides(self) -> tuple:
        return self.elem_strides if self.elem_strides is not None else (1,) * len(self.gdims)


def validate(desc: TmaDesc) -> list[str]:
    """返回所有违反的规则（错误码开头的字符串），合法时返回空列表。"""
    raise NotImplementedError("validate")


def elements_per_dim(desc: TmaDesc) -> tuple:
    """每一维实际搬运的元素个数：ceil(box / elem_stride)。"""
    raise NotImplementedError("elements_per_dim")


def expect_tx_bytes(desc: TmaDesc) -> int:
    """一次 box load 交付到 SMEM 的字节数（越界填 0 的部分也算）。"""
    raise NotImplementedError("expect_tx_bytes")


def swizzle_byte(off: int, mode: str) -> int:
    """CuTe Swizzle<B,4,3> 作用在字节偏移上：bits[7, 7+B) 异或进 bits[4, 4+B)。"""
    raise NotImplementedError("swizzle_byte")


def smem_byte_offset(desc: TmaDesc, box_coord) -> int:
    """box 内坐标（最内层在前）对应的 SMEM 字节位置（已 swizzle）。"""
    raise NotImplementedError("smem_byte_offset")


def box_load(gmem: np.ndarray, desc: TmaDesc, start) -> np.ndarray:
    """模拟一次 TMA box load：返回 expect_tx_bytes 长的 SMEM 字节数组。gmem 是 uint8 字节缓冲区。"""
    raise NotImplementedError("box_load")


def unswizzle(desc: TmaDesc, smem: np.ndarray, dtype) -> np.ndarray:
    """把 SMEM 字节按 swizzle 还原成逻辑 box（numpy shape 为最外层在前）。"""
    raise NotImplementedError("unswizzle")


def tile_element_map(rows: int, inner: int, elem_bytes: int, mode: str) -> list[int]:
    """(rows, inner) 的 box，inner 连续：按 CuTe colex 顺序 (m + rows*k) 列出每个元素在SMEM 的元素偏移。"""
    raise NotImplementedError("tile_element_map")
