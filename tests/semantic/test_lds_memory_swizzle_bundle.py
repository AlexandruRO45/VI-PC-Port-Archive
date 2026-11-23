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

class PermuteExpr(ASTNode):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def eval(
        self,
        addrs: list[int],
        datas: list[int],
        exec_mask: int = MASK32,
        old_dst: list[int] | None = None,
    ) -> list[int]:
        return ref_ds_permute_b32(addrs, datas, self.offset, exec_mask, old_dst)


class TestDsPermuteB32AST(unittest.TestCase):
    def test_permute_ast_eval(self) -> None:
        expr = PermuteExpr(offset=8)  # shift by +2 lanes
        addrs = [l * 4 for l in range(32)]
        datas = [500 + l for l in range(32)]
        out = expr.eval(addrs, datas)
        for s in range(32):
            d = (s + 2) % 32
            self.assertEqual(out[d], datas[s])

class TestDsPermuteB32SIMD(unittest.TestCase):
    def test_simd_unwritten_destinations_are_zero(self) -> None:
        # Only even lanes are active in EXEC
        exec_mask = 0x55555555
        addrs = [l * 4 for l in range(32)]
        datas = [1000 + l for l in range(32)]
        out = ref_ds_permute_b32(addrs, datas, offset=0, exec_mask=exec_mask)
        for l in range(32):
            if l % 2 == 0:
                self.assertEqual(out[l], datas[l])
            else:
                self.assertEqual(out[l], 0)

    def test_simd_inactive_destination_preserves_old(self) -> None:
        # Upper 16 lanes inactive in EXEC
        exec_mask = 0x0000FFFF
        addrs = [l * 4 for l in range(32)]
        datas = [1000 + l for l in range(32)]
        old_dst = [0xDEAD0000 + l for l in range(32)]
        out = ref_ds_permute_b32(addrs, datas, offset=0, exec_mask=exec_mask, old_dst=old_dst)
        for l in range(16):
            self.assertEqual(out[l], datas[l])
        for l in range(16, 32):
            self.assertEqual(out[l], old_dst[l])

class TestDsPermuteB32Translator(unittest.TestCase):
    def test_translator_permute_execution(self) -> None:
        # ds_permute_b32 v3, v1, v2 offset:4
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # addr = lane * 4
            "v_add_nc_u32_e32 v2, 200, v0",  # data = 200 + lane
            "ds_permute_b32 v3, v1, v2 offset:4",  # writes to ((4*l + 4)/4)%32 = (l+1)%32
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_permute_exec")
        for s in range(32):
            d = (s + 1) % 32
            self.assertEqual(results[d], 200 + s)

    def test_translator_permute_zero_offset(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 300, v0",
            "ds_permute_b32 v3, v1, v2",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_permute_zero_off")
        for l in range(32):
            self.assertEqual(results[l], 300 + l)

def ref_ds_bpermute_b32(
    addrs: list[int],
    datas: list[int],
    offset: int = 0,
    exec_mask: int = MASK32,
    old_dst: list[int] | None = None,
) -> list[int]:
    """Reference mathematical oracle for backward (gather) permute ds_bpermute_b32."""
    out = [0] * 32
    for d in range(32):
        if not (exec_mask & (1 << d)):
            out[d] = old_dst[d] if old_dst is not None else 0
        else:
            s = ((addrs[d] + offset) // 4) % 32
            if exec_mask & (1 << s):
                out[d] = datas[s]
            else:
                out[d] = 0
    return out


class TestDsBPermuteB32Oracle(unittest.TestCase):
    def test_bpermute_rotate_oracle(self) -> None:
        addrs = [l * 4 for l in range(32)]
        datas = [l * 15 for l in range(32)]
        # offset = 4 means destination d reads from s = (d + 1) % 32
        out = ref_ds_bpermute_b32(addrs, datas, offset=4)
        for d in range(32):
            self.assertEqual(out[d], datas[(d + 1) % 32])

    def test_bpermute_broadcast_oracle(self) -> None:
        # All destinations point to lane 7's address (28)
        addrs = [28] * 32
        datas = [99 + l for l in range(32)]
        out = ref_ds_bpermute_b32(addrs, datas, offset=0)
        for d in range(32):
            self.assertEqual(out[d], datas[7])

class BPermuteExpr(ASTNode):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def eval(
        self,
        addrs: list[int],
        datas: list[int],
        exec_mask: int = MASK32,
        old_dst: list[int] | None = None,
    ) -> list[int]:
        return ref_ds_bpermute_b32(addrs, datas, self.offset, exec_mask, old_dst)


class TestDsBPermuteB32AST(unittest.TestCase):
    def test_bpermute_ast_eval(self) -> None:
        expr = BPermuteExpr(offset=12)  # d reads from (d + 3) % 32
        addrs = [l * 4 for l in range(32)]
        datas = [200 + l for l in range(32)]
        out = expr.eval(addrs, datas)
        for d in range(32):
            self.assertEqual(out[d], datas[(d + 3) % 32])

class TestDsBPermuteB32SIMD(unittest.TestCase):
    def test_simd_inactive_source_returns_zero(self) -> None:
        # Source lane 5 is inactive
        exec_mask = MASK32 & ~(1 << 5)
        addrs = [l * 4 for l in range(32)]
        datas = [300 + l for l in range(32)]
        out = ref_ds_bpermute_b32(addrs, datas, offset=4, exec_mask=exec_mask)
        # Destination 4 reads from source 5 -> gets 0
        self.assertEqual(out[4], 0)
        # Destination 0 reads from source 1 -> gets datas[1]
        self.assertEqual(out[0], datas[1])

    def test_simd_inactive_destination_preserves_old(self) -> None:
        exec_mask = 0x000000FF  # only lower 8 lanes active
        addrs = [l * 4 for l in range(32)]
        datas = [700 + l for l in range(32)]
        old_dst = [0xCAFE0000 + l for l in range(32)]
        out = ref_ds_bpermute_b32(addrs, datas, offset=0, exec_mask=exec_mask, old_dst=old_dst)
        for l in range(8):
            self.assertEqual(out[l], datas[l])
        for l in range(8, 32):
            self.assertEqual(out[l], old_dst[l])

class TestDsBPermuteB32Translator(unittest.TestCase):
    def test_translator_bpermute_execution(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # addr = lane * 4
            "v_add_nc_u32_e32 v2, 100, v0",  # data = 100 + lane
            "ds_bpermute_b32 v3, v1, v2 offset:4",  # reads from (l+1)%32
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_bpermute_exec")
        for d in range(32):
            self.assertEqual(results[d], 100 + ((d + 1) % 32))

def ref_ds_read_u16(lds: LdsModel, addr: int, offset: int = 0) -> int:
    """Reference mathematical oracle for ds_read_u16."""
    return lds.read_u16(addr + offset)


class TestDsReadU16Oracle(unittest.TestCase):
    def test_read_u16_zero_extension(self) -> None:
        lds = LdsModel()
        lds.write_u32(0x100, 0x8ABC1234)
        # Read low halfword at 0x100 -> 0x1234
        self.assertEqual(ref_ds_read_u16(lds, 0x100, 0), 0x1234)
        # Read high halfword at 0x100 + 2 -> 0x8ABC (zero-extended)
        self.assertEqual(ref_ds_read_u16(lds, 0x100, 2), 0x8ABC)

    def test_read_u16_wrapping(self) -> None:
        lds = LdsModel()
        lds.write_u16(0, 0xBEEF)
        self.assertEqual(ref_ds_read_u16(lds, LDS_SIZE, 0), 0xBEEF)

class LdsReadU16Expr(ASTNode):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def eval(self, lds: LdsModel, addr: int) -> int:
        return ref_ds_read_u16(lds, addr, self.offset)


class TestDsReadU16AST(unittest.TestCase):
    def test_read_u16_ast_eval(self) -> None:
        lds = LdsModel()
        lds.write_u16(0x40, 0x55AA)
        expr = LdsReadU16Expr(offset=0x40)
        self.assertEqual(expr.eval(lds, 0), 0x55AA)

class TestDsReadU16SIMD(unittest.TestCase):
    def test_simd_read_u16_divergent_lanes(self) -> None:
        lds = LdsModel()
        for l in range(32):
            lds.write_u16(l * 4, 0x1000 + l)
        expr = LdsReadU16Expr(offset=0)
        for l in range(32):
            self.assertEqual(expr.eval(lds, l * 4), 0x1000 + l)

class TestDsReadU16Translator(unittest.TestCase):
    def test_translator_read_u16_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x80",
            "v_mov_b32_e32 v2, 0x98765432",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_read_u16_exec")
        for l in range(32):
            self.assertEqual(results[l], 0x9876)

def ref_ds_read_i16(lds: LdsModel, addr: int, offset: int = 0) -> int:
    """Reference mathematical oracle for ds_read_i16 (sign-extended to 32 bits)."""
    return lds.read_i16(addr + offset)


class TestDsReadI16Oracle(unittest.TestCase):
    def test_read_i16_sign_extension(self) -> None:
        lds = LdsModel()
        # 0x8001 has bit 15 set -> negative: sign-extends to 0xFFFF8001
        lds.write_u16(0x200, 0x8001)
        self.assertEqual(ref_ds_read_i16(lds, 0x200), 0xFFFF8001)
        # 0x7FFF is positive -> 0x00007FFF
        lds.write_u16(0x202, 0x7FFF)
        self.assertEqual(ref_ds_read_i16(lds, 0x200, offset=2), 0x7FFF)

class LdsReadI16Expr(ASTNode):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def eval(self, lds: LdsModel, addr: int) -> int:
        return ref_ds_read_i16(lds, addr, self.offset)


class TestDsReadI16AST(unittest.TestCase):
    def test_read_i16_ast_eval(self) -> None:
        lds = LdsModel()
        lds.write_u16(0x30, 0xF000)
        expr = LdsReadI16Expr(offset=0)
        self.assertEqual(expr.eval(lds, 0x30), 0xFFFFF000)

class TestDsReadI16SIMD(unittest.TestCase):
    def test_simd_read_i16_divergent_signs(self) -> None:
        lds = LdsModel()
        for l in range(32):
            val = 0x8000 + l if l % 2 == 1 else 0x1000 + l
            lds.write_u16(l * 4, val)
        expr = LdsReadI16Expr(offset=0)
        for l in range(32):
            got = expr.eval(lds, l * 4)
            if l % 2 == 1:
                self.assertEqual(got, (0xFFFF8000 + l) & MASK32)
            else:
                self.assertEqual(got, 0x1000 + l)

class TestDsReadI16Translator(unittest.TestCase):
    def test_translator_read_i16_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x60",
            "v_mov_b32_e32 v2, 0xff801234",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_read_i16_exec")
        for l in range(32):
            self.assertEqual(results[l], 0xFFFFFF80)

def ref_ds_write_b16(lds: LdsModel, addr: int, val: int, offset: int = 0) -> None:
    """Reference mathematical oracle for ds_write_b16."""
    lds.write_u16(addr + offset, val)


class TestDsWriteB16Oracle(unittest.TestCase):
    def test_write_b16_preserves_neighbor_bytes(self) -> None:
        lds = LdsModel()
        lds.write_u32(0x50, 0x11223344)
        ref_ds_write_b16(lds, 0x50, 0xAAAA, offset=2)
        # Bytes 0 and 1 (0x3344) preserved, bytes 2 and 3 replaced by 0xAAAA
        self.assertEqual(lds.read_u32(0x50), 0xAAAA3344)

class LdsWriteB16Expr(ASTNode):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def eval(self, lds: LdsModel, addr: int, val: int) -> None:
        ref_ds_write_b16(lds, addr, val, self.offset)


class TestDsWriteB16AST(unittest.TestCase):
    def test_write_b16_ast_eval(self) -> None:
        lds = LdsModel()
        expr = LdsWriteB16Expr(offset=4)
        expr.eval(lds, 0x10, 0x4321)
        self.assertEqual(lds.read_u16(0x14), 0x4321)

class TestDsWriteB16SIMD(unittest.TestCase):
    def test_simd_write_b16_multi_lane(self) -> None:
        lds = LdsModel()
        expr = LdsWriteB16Expr(offset=0)
        for l in range(32):
            expr.eval(lds, l * 4, 0xA000 + l)
        for l in range(32):
            self.assertEqual(lds.read_u16(l * 4), 0xA000 + l)

class TestDsWriteB16Translator(unittest.TestCase):
    def test_translator_write_b16_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0x0000abcd",
            "ds_write_b16 v1, v3 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v4, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b16_exec")
        for l in range(32):
            self.assertEqual(results[l], 0xABCD5678)

def ref_ds_write_b32(lds: LdsModel, addr: int, val: int, offset: int = 0) -> None:
    """Reference mathematical oracle for ds_write_b32."""
    lds.write_u32(addr + offset, val)


class TestDsWriteB32Oracle(unittest.TestCase):
    def test_write_b32_oracle(self) -> None:
        lds = LdsModel()
        ref_ds_write_b32(lds, 0x100, 0xDEADBEEF, offset=0x20)
        self.assertEqual(lds.read_u32(0x120), 0xDEADBEEF)

    def test_write_b32_wrapping(self) -> None:
        lds = LdsModel()
        ref_ds_write_b32(lds, LDS_SIZE + 8, 0x12345678)
        self.assertEqual(lds.read_u32(8), 0x12345678)

class LdsWriteB32Expr(ASTNode):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def eval(self, lds: LdsModel, addr: int, val: int) -> None:
        ref_ds_write_b32(lds, addr, val, self.offset)


class TestDsWriteB32AST(unittest.TestCase):
    def test_write_b32_ast_eval(self) -> None:
        lds = LdsModel()
        expr = LdsWriteB32Expr(offset=16)
        expr.eval(lds, 0x20, 0xCAFEBABE)
        self.assertEqual(lds.read_u32(0x30), 0xCAFEBABE)

class TestDsWriteB32SIMD(unittest.TestCase):
    def test_simd_write_b32_divergent_lanes(self) -> None:
        lds = LdsModel()
        expr = LdsWriteB32Expr(offset=0)
        for l in range(32):
            expr.eval(lds, l * 4, 0x70000000 + l)
        for l in range(32):
            self.assertEqual(lds.read_u32(l * 4), 0x70000000 + l)


if __name__ == "__main__":
    unittest.main()
