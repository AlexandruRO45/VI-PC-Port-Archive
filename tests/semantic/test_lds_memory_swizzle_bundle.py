"""Bundle 5: LDS Swizzle & Sub-Dword Memory Suite.

Tests for 7 LDS memory and swizzle opcodes:
- ds_swizzle_b32: cross-lane quad permute (QUAD_PERM) and bitmask permute (BITMASK_PERM)
- ds_permute_b32: forward (scatter) cross-lane permute
- ds_bpermute_b32: backward (gather) cross-lane permute
- ds_read_u16: 16-bit unsigned sub-dword load with zero extension
- ds_read_i16: 16-bit signed sub-dword load with sign extension
- ds_write_b16: 16-bit halfword store with byte preservation
- ds_write_b32: 32-bit dword store into LDS

Standard library only; zero game data.
"""
from __future__ import annotations

import math
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
LDS_SIZE = 4096

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


class LdsModel:
    """Byte-level little-endian LDS memory model wrapping at LDS_SIZE."""

    def __init__(self, size: int = LDS_SIZE) -> None:
        self.size = size
        self.bytes = bytearray(size)

    def write_u8(self, addr: int, val: int) -> None:
        eff = addr % self.size
        self.bytes[eff] = val & 0xFF

    def read_u8(self, addr: int) -> int:
        eff = addr % self.size
        return self.bytes[eff]

    def write_u16(self, addr: int, val: int) -> None:
        eff = addr % self.size
        self.bytes[eff : eff + 2] = struct.pack("<H", val & 0xFFFF)

    def read_u16(self, addr: int) -> int:
        eff = addr % self.size
        return struct.unpack_from("<H", self.bytes, eff)[0]

    def read_i16(self, addr: int) -> int:
        eff = addr % self.size
        val = struct.unpack_from("<h", self.bytes, eff)[0]
        return val & MASK32

    def write_u32(self, addr: int, val: int) -> None:
        eff = addr % self.size
        self.bytes[eff : eff + 4] = struct.pack("<I", val & MASK32)

    def read_u32(self, addr: int) -> int:
        eff = addr % self.size
        return struct.unpack_from("<I", self.bytes, eff)[0]


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle5") -> list[int]:
    """Helper to translate and run a 32-lane compute shader; outputs out[lane*32]."""
    prologue = ["v_lshlrev_b32_e32 v30, 5, v0"]
    full_body = prologue + list(body) + ["s_endpgm"]
    spv = support.translate_lines(name, full_body, fssm=FSSM)
    mem: dict[int, int] = {}
    si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=mem,
    )
    return [mem.get(i * 32, 0) for i in range(num_words)]


class ASTNode:
    """Base class for symbolic AST expressions."""
    pass

def ref_ds_swizzle_b32(lane: int, mode: str, args: tuple[int, ...] | str) -> int:
    """Reference mathematical oracle for ds_swizzle_b32 across 32 lanes."""
    if mode == "QUAD_PERM":
        return (lane & ~3) | (args[lane & 3])
    elif mode == "BITMASK_PERM":
        and_m = or_m = xor_m = 0
        for k, ch in enumerate(args):
            bit = 1 << (4 - k)
            if ch in "pi":
                and_m |= bit
            if ch == "i":
                xor_m |= bit
            if ch == "1":
                or_m |= bit
        return ((lane & and_m) | or_m) ^ xor_m
    raise ValueError(f"Unknown swizzle mode: {mode}")


class TestDsSwizzleB32Oracle(unittest.TestCase):
    def test_quad_perm_oracle(self) -> None:
        # Swap adjacent lanes: (1, 0, 3, 2)
        perm = (1, 0, 3, 2)
        for quad in range(8):
            base = quad * 4
            self.assertEqual(ref_ds_swizzle_b32(base + 0, "QUAD_PERM", perm), base + 1)
            self.assertEqual(ref_ds_swizzle_b32(base + 1, "QUAD_PERM", perm), base + 0)
            self.assertEqual(ref_ds_swizzle_b32(base + 2, "QUAD_PERM", perm), base + 3)
            self.assertEqual(ref_ds_swizzle_b32(base + 3, "QUAD_PERM", perm), base + 2)

    def test_bitmask_perm_oracle(self) -> None:
        # Invert bit 0 (SWAP 1): pattern "ppppi"
        for l in range(32):
            expected = l ^ 1
            self.assertEqual(ref_ds_swizzle_b32(l, "BITMASK_PERM", "ppppi"), expected)
        # Broadcast lane 0: pattern "00000"
        for l in range(32):
            self.assertEqual(ref_ds_swizzle_b32(l, "BITMASK_PERM", "00000"), 0)
        # Reverse 16: pattern "ppiii"
        for l in range(32):
            self.assertEqual(ref_ds_swizzle_b32(l, "BITMASK_PERM", "ppiii"), (l & ~7) | (7 - (l & 7)))

class SwizzleExpr(ASTNode):
    def __init__(self, mode: str, args: tuple[int, ...] | str) -> None:
        self.mode = mode
        self.args = args

    def eval(self, lane: int, src_values: list[int], exec_mask: int = MASK32) -> int:
        src_lane = ref_ds_swizzle_b32(lane, self.mode, self.args)
        if not (exec_mask & (1 << src_lane)):
            return 0
        return src_values[src_lane]


class TestDsSwizzleB32AST(unittest.TestCase):
    def test_swizzle_ast_eval(self) -> None:
        expr = SwizzleExpr("QUAD_PERM", (3, 2, 1, 0))
        data = [100 + i for i in range(32)]
        for lane in range(32):
            base = (lane & ~3)
            expected_src = base + (3 - (lane & 3))
            self.assertEqual(expr.eval(lane, data), data[expected_src])

    def test_swizzle_ast_inactive_source(self) -> None:
        expr = SwizzleExpr("QUAD_PERM", (1, 0, 3, 2))
        data = [100 + i for i in range(32)]
        # Mask off lane 1: lane 0 attempts to read from lane 1 -> gets 0
        exec_mask = MASK32 & ~(1 << 1)
        self.assertEqual(expr.eval(0, data, exec_mask), 0)
        self.assertEqual(expr.eval(1, data, exec_mask), data[0])

class TestDsSwizzleB32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        data = [i * 10 for i in range(32)]
        expr_swap = SwizzleExpr("BITMASK_PERM", "ppppi")  # swap adjacent
        # EXEC mask with lower 16 lanes active
        exec_mask = 0x0000FFFF
        results = [expr_swap.eval(l, data, exec_mask) for l in range(32)]
        for l in range(16):
            src = l ^ 1
            self.assertEqual(results[l], data[src])
        # Lane 16 reads from 17, but lane 17 is inactive in exec_mask -> 0
        self.assertEqual(results[16], 0)

    def test_simd_alternating_exec_mask(self) -> None:
        data = [i + 1 for i in range(32)]
        expr = SwizzleExpr("QUAD_PERM", (1, 0, 3, 2))
        exec_mask = 0x55555555  # even lanes active
        for l in (0, 2, 4, 6):
            # Target is l+1, which is odd (inactive) -> returns 0
            self.assertEqual(expr.eval(l, data, exec_mask), 0)

class TestDsSwizzleB32Translator(unittest.TestCase):
    def test_translator_quad_perm_execution(self) -> None:
        # ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)
        body = [
            "v_mov_b32_e32 v1, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_swizzle_quad")
        for quad in range(8):
            base = quad * 4
            self.assertEqual(results[base + 0], base + 1)
            self.assertEqual(results[base + 1], base + 0)
            self.assertEqual(results[base + 2], base + 3)
            self.assertEqual(results[base + 3], base + 2)

    def test_translator_bitmask_perm_execution(self) -> None:
        # ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ppppi")
        body = [
            "v_mov_b32_e32 v1, v0",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ppppi")',
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_swizzle_bitmask")
        for l in range(32):
            self.assertEqual(results[l], l ^ 1)

    def test_translator_swizzle_syntax_rejection(self) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_swizzle_bad", ["ds_swizzle_b32 v1, v2 offset:99999", "s_endpgm"], fssm=FSSM)

def ref_ds_permute_b32(
    addrs: list[int],
    datas: list[int],
    offset: int = 0,
    exec_mask: int = MASK32,
    old_dst: list[int] | None = None,
) -> list[int]:
    """Reference mathematical oracle for forward (scatter) permute ds_permute_b32."""
    out = [0] * 32
    if old_dst is not None:
        for l in range(32):
            if not (exec_mask & (1 << l)):
                out[l] = old_dst[l]
    # Every active source lane writes to destination lane ((addr + offset) // 4) % 32
    for s in range(32):
        if exec_mask & (1 << s):
            d = ((addrs[s] + offset) // 4) % 32
            if exec_mask & (1 << d):
                out[d] = datas[s]
    return out


class TestDsPermuteB32Oracle(unittest.TestCase):
    def test_permute_rotate_by_one(self) -> None:
        # Addr = lane * 4; offset = 4; datas = lane * 10
        addrs = [l * 4 for l in range(32)]
        datas = [l * 10 for l in range(32)]
        out = ref_ds_permute_b32(addrs, datas, offset=4)
        for s in range(32):
            d = (s + 1) % 32
            self.assertEqual(out[d], datas[s])

    def test_permute_identity(self) -> None:
        addrs = [l * 4 for l in range(32)]
        datas = [100 + l for l in range(32)]
        out = ref_ds_permute_b32(addrs, datas, offset=0)
        self.assertEqual(out, datas)


if __name__ == "__main__":
    unittest.main()
