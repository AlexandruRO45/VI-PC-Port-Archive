"""Semantic tests for RDNA2 ds_swizzle_b32 (LDS cross-lane swizzle).

RDNA2 LDS architecture contract:
- Opcode: ds_swizzle_b32 vdst, vsrc offset:swizzle(...) | offset:N
- Operation: Subgroup intra-wave cross-lane shuffle within each 32-lane group (lanes 0..31 in wave32;
  lanes 0..31 and 32..63 independently in wave64).
- 16-bit offset encoding:
  * Bit 15 = 1 (QUAD_PERM): src = (lane & ~3) | ((offset >> (2 * (lane & 3))) & 3)
  * Bit 15 = 0 (BITMASK_PERM): src = (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m),
    where and_m = offset[4:0], or_m = offset[9:5], xor_m = offset[14:10].
  * Shorthands:
    - SWAP(stride): lane ^ stride (and=31, or=0, xor=stride)
    - REVERSE(width): lane ^ (width - 1)
    - BROADCAST(group_size, lane_idx): (lane & ~(group_size - 1)) | lane_idx
- Inactive lanes:
  * Inactive destination lanes (EXEC=0) preserve their destination VGPR.
  * Inactive or absent source lanes return 0 to the destination.
- Condition codes: VALU/LDS operations do not modify SCC, VCC, or EXEC.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si
import support

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def encode_quad_perm(a: int, b: int, c: int, d: int) -> int:
    for v in (a, b, c, d):
        assert 0 <= v <= 3
    return 0x8000 | a | (b << 2) | (c << 4) | (d << 6)


def encode_bitmask_perm(pattern: str) -> int:
    assert len(pattern) == 5 and all(ch in "01pi" for ch in pattern)
    and_m = or_m = xor_m = 0
    for k, ch in enumerate(pattern):
        bit = 1 << (4 - k)
        if ch in "pi":
            and_m |= bit
        if ch == "1":
            or_m |= bit
        if ch == "i":
            xor_m |= bit
    return and_m | (or_m << 5) | (xor_m << 10)


def encode_swap(stride: int) -> int:
    assert stride in (1, 2, 4, 8, 16)
    return 0x1F | (stride << 10)


def encode_reverse(width: int) -> int:
    assert width in (2, 4, 8, 16, 32)
    return 0x1F | ((width - 1) << 10)


def encode_broadcast(group_size: int, lane_idx: int) -> int:
    assert group_size in (2, 4, 8, 16, 32) and 0 <= lane_idx < group_size
    and_m = (~(group_size - 1)) & 0x1F
    return and_m | (lane_idx << 5)


def ref_swizzle_src_lane(lane: int, offset16: int) -> int:
    """RDNA2 ISA source lane within a 32-lane group computed from the 16-bit offset."""
    if offset16 & 0x8000:
        sel = (offset16 >> (2 * (lane & 3))) & 3
        return (lane & ~3) | sel
    and_m = offset16 & 0x1F
    or_m = (offset16 >> 5) & 0x1F
    xor_m = (offset16 >> 10) & 0x1F
    return (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m)


def ref_swizzle_wave(
    src_vals: list[int],
    dst_old: list[int],
    offset16: int,
    exec_mask: int,
    zero_inactive_src: bool = True,
) -> list[int]:
    """Pure architectural oracle for ds_swizzle_b32 across all lanes."""
    lanes = len(src_vals)
    assert len(dst_old) == lanes and lanes in (32, 64)
    out = []
    for lane in range(lanes):
        if not ((exec_mask >> lane) & 1):
            out.append(dst_old[lane] & M32)
            continue
        src = ref_swizzle_src_lane(lane, offset16)
        if zero_inactive_src and not ((exec_mask >> src) & 1):
            out.append(0)
        else:
            out.append(src_vals[src] & M32)
    return out


class TestDsSwizzleB32Oracle(unittest.TestCase):
    def test_swap_stride_1(self) -> None:
        off = encode_swap(1)
        src = [100 + i for i in range(32)]
        dst_old = [0] * 32
        res = ref_swizzle_wave(src, dst_old, off, 0xFFFFFFFF)
        self.assertEqual(res[0], 101)
        self.assertEqual(res[1], 100)
        self.assertEqual(res[2], 103)
        self.assertEqual(res[3], 102)

    def test_swap_stride_16(self) -> None:
        off = encode_swap(16)
        src = [i for i in range(32)]
        dst_old = [0] * 32
        res = ref_swizzle_wave(src, dst_old, off, 0xFFFFFFFF)
        for i in range(16):
            self.assertEqual(res[i], i + 16)
            self.assertEqual(res[i + 16], i)

    def test_reverse_width_4(self) -> None:
        off = encode_reverse(4)
        src = [i for i in range(32)]
        dst_old = [0] * 32
        res = ref_swizzle_wave(src, dst_old, off, 0xFFFFFFFF)
        self.assertEqual(res[0:4], [3, 2, 1, 0])
        self.assertEqual(res[4:8], [7, 6, 5, 4])

    def test_broadcast_quad(self) -> None:
        off = encode_broadcast(4, 2)
        src = [i for i in range(32)]
        dst_old = [0] * 32
        res = ref_swizzle_wave(src, dst_old, off, 0xFFFFFFFF)
        self.assertEqual(res[0:4], [2, 2, 2, 2])
        self.assertEqual(res[4:8], [6, 6, 6, 6])

    def test_quad_perm_custom(self) -> None:
        off = encode_quad_perm(1, 2, 3, 0)
        src = [i for i in range(32)]
        dst_old = [0] * 32
        res = ref_swizzle_wave(src, dst_old, off, 0xFFFFFFFF)
        self.assertEqual(res[0:4], [1, 2, 3, 0])
        self.assertEqual(res[4:8], [5, 6, 7, 4])

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------
class SwizzleKind(Enum):
    QUAD_PERM = auto()
    BITMASK_PERM = auto()
    SWAP = auto()
    REVERSE = auto()
    BROADCAST = auto()
    NUMERIC = auto()


@dataclass(frozen=True)
class DsSwizzleB32AST:
    vdst_idx: int
    vsrc_idx: int
    kind: SwizzleKind
    offset16: int
    args: tuple[int, ...] = ()
    pattern: str = ""

    def to_assembly(self) -> str:
        if self.kind == SwizzleKind.SWAP:
            return f"ds_swizzle_b32 v{self.vdst_idx}, v{self.vsrc_idx} offset:swizzle(SWAP,{self.args[0]})"
        elif self.kind == SwizzleKind.REVERSE:
            return f"ds_swizzle_b32 v{self.vdst_idx}, v{self.vsrc_idx} offset:swizzle(REVERSE,{self.args[0]})"
        elif self.kind == SwizzleKind.BROADCAST:
            return f"ds_swizzle_b32 v{self.vdst_idx}, v{self.vsrc_idx} offset:swizzle(BROADCAST,{self.args[0]},{self.args[1]})"
        elif self.kind == SwizzleKind.QUAD_PERM:
            return f"ds_swizzle_b32 v{self.vdst_idx}, v{self.vsrc_idx} offset:swizzle(QUAD_PERM,{','.join(str(x) for x in self.args)})"
        elif self.kind == SwizzleKind.BITMASK_PERM:
            return f'ds_swizzle_b32 v{self.vdst_idx}, v{self.vsrc_idx} offset:swizzle(BITMASK_PERM,"{self.pattern}")'
        else:
            return f"ds_swizzle_b32 v{self.vdst_idx}, v{self.vsrc_idx} offset:{self.offset16:#x}"


class TestDsSwizzleB32AST(unittest.TestCase):
    def test_ast_swap_assembly(self) -> None:
        ast = DsSwizzleB32AST(
            vdst_idx=0,
            vsrc_idx=1,
            kind=SwizzleKind.SWAP,
            offset16=encode_swap(1),
            args=(1,),
        )
        self.assertEqual(ast.to_assembly(), "ds_swizzle_b32 v0, v1 offset:swizzle(SWAP,1)")

    def test_ast_quad_perm_assembly(self) -> None:
        ast = DsSwizzleB32AST(
            vdst_idx=2,
            vsrc_idx=3,
            kind=SwizzleKind.QUAD_PERM,
            offset16=encode_quad_perm(3, 2, 1, 0),
            args=(3, 2, 1, 0),
        )
        self.assertEqual(ast.to_assembly(), "ds_swizzle_b32 v2, v3 offset:swizzle(QUAD_PERM,3,2,1,0)")

    def test_ast_numeric_offset(self) -> None:
        ast = DsSwizzleB32AST(
            vdst_idx=5,
            vsrc_idx=6,
            kind=SwizzleKind.NUMERIC,
            offset16=0x801F,
        )
        self.assertEqual(ast.to_assembly(), "ds_swizzle_b32 v5, v6 offset:0x801f")

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------
@dataclass
class SwizzleSimdState:
    wave_size: int
    exec_mask: int
    vgprs: dict[int, list[int]]

    @classmethod
    def create(cls, wave_size: int = 32, exec_mask: int = 0xFFFFFFFF) -> SwizzleSimdState:
        return cls(wave_size=wave_size, exec_mask=exec_mask, vgprs={})

    def ensure_vgpr(self, idx: int, init_val: int = 0) -> None:
        if idx not in self.vgprs:
            self.vgprs[idx] = [init_val] * self.wave_size


def step_swizzle_instruction(ast: DsSwizzleB32AST, state: SwizzleSimdState) -> None:
    state.ensure_vgpr(ast.vsrc_idx, 0)
    state.ensure_vgpr(ast.vdst_idx, 0xBAADF00D)
    src_vals = state.vgprs[ast.vsrc_idx]
    dst_old = state.vgprs[ast.vdst_idx]
    out = ref_swizzle_wave(src_vals, dst_old, ast.offset16, state.exec_mask)
    state.vgprs[ast.vdst_idx] = out


class TestDsSwizzleB32ScalarInterpreter(unittest.TestCase):
    def test_interpreter_swap_execution(self) -> None:
        state = SwizzleSimdState.create(32, 0xFFFFFFFF)
        state.vgprs[1] = [1000 + i for i in range(32)]
        ast = DsSwizzleB32AST(0, 1, SwizzleKind.SWAP, encode_swap(2), (2,))
        step_swizzle_instruction(ast, state)

        # Stride 2 swap: 0 <-> 2, 1 <-> 3
        self.assertEqual(state.vgprs[0][0], 1002)
        self.assertEqual(state.vgprs[0][1], 1003)
        self.assertEqual(state.vgprs[0][2], 1000)
        self.assertEqual(state.vgprs[0][3], 1001)

    def test_interpreter_determinism(self) -> None:
        state1 = SwizzleSimdState.create(32, 0xFFFFFFFF)
        state2 = SwizzleSimdState.create(32, 0xFFFFFFFF)
        src = [(i * 37 + 11) & M32 for i in range(32)]
        state1.vgprs[1] = list(src)
        state2.vgprs[1] = list(src)
        ast = DsSwizzleB32AST(0, 1, SwizzleKind.QUAD_PERM, encode_quad_perm(2, 3, 0, 1), (2, 3, 0, 1))

        step_swizzle_instruction(ast, state1)
        step_swizzle_instruction(ast, state2)
        self.assertEqual(state1.vgprs[0], state2.vgprs[0])

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# -----------------------------------------------------------------------------
class TestDsSwizzleB32SIMD(unittest.TestCase):
    def test_inactive_destination_preservation(self) -> None:
        # Inactive destination lanes (exec_mask bit = 0) must preserve old vdst
        exec_mask = 0x55555555  # Even lanes active, odd lanes inactive
        state = SwizzleSimdState.create(32, exec_mask)
        state.vgprs[1] = [100 + i for i in range(32)]
        state.vgprs[0] = [0xDEADBEEF] * 32

        ast = DsSwizzleB32AST(0, 1, SwizzleKind.SWAP, encode_swap(1), (1,))
        step_swizzle_instruction(ast, state)

        for i in range(32):
            if i % 2 == 0:
                # Even lane: active dest, but source is odd (which is inactive in EXEC)!
                # RDNA2 rule: inactive source lane returns 0
                self.assertEqual(state.vgprs[0][i], 0)
            else:
                # Odd lane: inactive destination, preserved
                self.assertEqual(state.vgprs[0][i], 0xDEADBEEF)

    def test_inactive_source_returns_zero(self) -> None:
        # Only lane 0 active. When other lanes read lane 0, they would read lane 0, but they're inactive.
        # When lane 0 reads lane 1 (which is inactive), lane 0 receives 0!
        exec_mask = 0x00000001
        state = SwizzleSimdState.create(32, exec_mask)
        state.vgprs[1] = [999] * 32
        state.vgprs[0] = [0xFEEDC0DE] * 32

        ast = DsSwizzleB32AST(0, 1, SwizzleKind.SWAP, encode_swap(1), (1,))
        step_swizzle_instruction(ast, state)

        # Lane 0 attempted to read lane 1, which is inactive -> returns 0
        self.assertEqual(state.vgprs[0][0], 0)
        # Other lanes preserved
        for i in range(1, 32):
            self.assertEqual(state.vgprs[0][i], 0xFEEDC0DE)

    def test_wave64_independent_halves(self) -> None:
        # In wave64, swizzle operates within each 32-lane group independently (0..31 and 32..63)
        state = SwizzleSimdState.create(64, 0xFFFFFFFFFFFFFFFF)
        state.vgprs[1] = [i for i in range(64)]
        ast = DsSwizzleB32AST(0, 1, SwizzleKind.BROADCAST, encode_broadcast(32, 0), (32, 0))
        step_swizzle_instruction(ast, state)

        # Lanes 0..31 broadcast lane 0
        for i in range(32):
            self.assertEqual(state.vgprs[0][i], 0)
        # Lanes 32..63 broadcast lane 32 (not lane 0!)
        for i in range(32, 64):
            self.assertEqual(state.vgprs[0][i], 32)
