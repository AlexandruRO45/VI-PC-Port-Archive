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
All tests directly compile assembly listings through support.translate_lines(...) and execute via run_cs.
"""
from __future__ import annotations

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
        body = [
            "v_mov_b32_e32 v1, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_swizzle_quad_oracle")
        for quad in range(8):
            base = quad * 4
            self.assertEqual(results[base + 0], ref_ds_swizzle_b32(base + 0, "QUAD_PERM", (1, 0, 3, 2)))
            self.assertEqual(results[base + 1], ref_ds_swizzle_b32(base + 1, "QUAD_PERM", (1, 0, 3, 2)))
            self.assertEqual(results[base + 2], ref_ds_swizzle_b32(base + 2, "QUAD_PERM", (1, 0, 3, 2)))
            self.assertEqual(results[base + 3], ref_ds_swizzle_b32(base + 3, "QUAD_PERM", (1, 0, 3, 2)))

    def test_bitmask_perm_oracle(self) -> None:
        # Swap adjacent lanes: "ppppi"
        body = [
            "v_mov_b32_e32 v1, v0",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ppppi")',
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_swizzle_bitmask_oracle")
        for l in range(32):
            self.assertEqual(results[l], ref_ds_swizzle_b32(l, "BITMASK_PERM", "ppppi"))

        # Broadcast lane 0: "00000"
        body_bcast = [
            "v_mov_b32_e32 v1, v0",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"00000")',
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results_bcast = run_cs(body_bcast, name="cs_swizzle_bcast_oracle")
        for l in range(32):
            self.assertEqual(results_bcast[l], ref_ds_swizzle_b32(l, "BITMASK_PERM", "00000"))


class TestDsSwizzleB32AST(unittest.TestCase):
    def test_swizzle_ast_eval(self) -> None:
        body = [
            "v_add_nc_u32_e32 v1, 100, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,3,2,1,0)",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_swizzle_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpGroupNonUniformShuffle", spv)
        results = run_cs(body, name="cs_swizzle_ast_eval")
        for lane in range(32):
            base = (lane & ~3)
            expected_src = base + (3 - (lane & 3))
            self.assertEqual(results[lane], 100 + expected_src)

    def test_swizzle_ast_inactive_source(self) -> None:
        body = [
            "v_add_nc_u32_e32 v1, 200, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_swizzle_ast_inactive", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpGroupNonUniformShuffle", spv)
        results = run_cs(body, name="cs_swizzle_ast_inactive")
        for quad in range(8):
            base = quad * 4
            self.assertEqual(results[base + 0], 200 + base + 1)
            self.assertEqual(results[base + 1], 200 + base + 0)


class TestDsSwizzleB32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 3, v0",  # lane * 8
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ppppi")',
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_swizzle_simd_div")
        for l in range(32):
            src = l ^ 1
            self.assertEqual(results[l], src * 8)

    def test_simd_alternating_exec_mask(self) -> None:
        body = [
            "v_add_nc_u32_e32 v1, 1, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_swizzle_simd_alt")
        for quad in range(8):
            base = quad * 4
            self.assertEqual(results[base + 0], (base + 1) + 1)
            self.assertEqual(results[base + 1], (base + 0) + 1)


class TestDsSwizzleB32Translator(unittest.TestCase):
    def test_translator_quad_perm_execution(self) -> None:
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


class TestDsPermuteB32Oracle(unittest.TestCase):
    def test_permute_rotate_by_one(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # addr = lane * 4
            "v_lshlrev_b32_e32 v2, 3, v0",  # data = lane * 8
            "ds_permute_b32 v3, v1, v2 offset:4",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_permute_rot_oracle")
        for s in range(32):
            d = (s + 1) % 32
            self.assertEqual(results[d], s * 8)

    def test_permute_identity(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 100, v0",
            "ds_permute_b32 v3, v1, v2 offset:0",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_permute_ident_oracle")
        for l in range(32):
            self.assertEqual(results[l], 100 + l)


class TestDsPermuteB32AST(unittest.TestCase):
    def test_permute_ast_eval(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 500, v0",
            "ds_permute_b32 v3, v1, v2 offset:8",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_permute_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpBitwiseAnd", spv)
        results = run_cs(body, name="cs_permute_ast_eval")
        for s in range(32):
            d = (s + 2) % 32
            self.assertEqual(results[d], 500 + s)


class TestDsPermuteB32SIMD(unittest.TestCase):
    def test_simd_unwritten_destinations_are_zero(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 1000, v0",
            "ds_permute_b32 v3, v1, v2 offset:0",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_permute_simd_unwritten")
        for l in range(32):
            self.assertEqual(results[l], 1000 + l)

    def test_simd_inactive_destination_preserves_old(self) -> None:
        body = [
            "v_mov_b32_e32 v3, 0",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 1000, v0",
            "ds_permute_b32 v3, v1, v2 offset:0",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_permute_simd_preserve")
        for l in range(32):
            self.assertEqual(results[l], 1000 + l)


class TestDsPermuteB32Translator(unittest.TestCase):
    def test_translator_permute_execution(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 200, v0",
            "ds_permute_b32 v3, v1, v2 offset:4",
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


class TestDsBPermuteB32Oracle(unittest.TestCase):
    def test_bpermute_rotate_oracle(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # addr = lane * 4
            "v_lshlrev_b32_e32 v2, 4, v0",  # data = lane * 16
            "ds_bpermute_b32 v3, v1, v2 offset:4",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_bpermute_rot_oracle")
        for d in range(32):
            self.assertEqual(results[d], ((d + 1) % 32) * 16)

    def test_bpermute_broadcast_oracle(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 28",  # lane 7 * 4
            "v_add_nc_u32_e32 v2, 99, v0",
            "ds_bpermute_b32 v3, v1, v2 offset:0",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_bpermute_bcast_oracle")
        for d in range(32):
            self.assertEqual(results[d], 99 + 7)


class TestDsBPermuteB32AST(unittest.TestCase):
    def test_bpermute_ast_eval(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 200, v0",
            "ds_bpermute_b32 v3, v1, v2 offset:12",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_bpermute_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpGroupNonUniformShuffle", spv)
        results = run_cs(body, name="cs_bpermute_ast_eval")
        for d in range(32):
            self.assertEqual(results[d], 200 + ((d + 3) % 32))


class TestDsBPermuteB32SIMD(unittest.TestCase):
    def test_simd_inactive_source_returns_zero(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 300, v0",
            "ds_bpermute_b32 v3, v1, v2 offset:4",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_bpermute_simd_src")
        for d in range(32):
            self.assertEqual(results[d], 300 + ((d + 1) % 32))

    def test_simd_inactive_destination_preserves_old(self) -> None:
        body = [
            "v_mov_b32_e32 v3, 0",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 700, v0",
            "ds_bpermute_b32 v3, v1, v2 offset:0",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_bpermute_simd_dst")
        for l in range(32):
            self.assertEqual(results[l], 700 + l)


class TestDsBPermuteB32Translator(unittest.TestCase):
    def test_translator_bpermute_execution(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 100, v0",
            "ds_bpermute_b32 v3, v1, v2 offset:4",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_bpermute_exec")
        for d in range(32):
            self.assertEqual(results[d], 100 + ((d + 1) % 32))


class TestDsReadU16Oracle(unittest.TestCase):
    def test_read_u16_zero_extension(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x100",
            "v_mov_b32_e32 v2, 0x8ABC1234",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:0",
            "ds_read_u16 v4, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "v_cmp_ne_u32_e64 s[4:5], v0, 0",
            "v_cndmask_b32_e64 v5, v3, v4, s[4:5]",
            "buffer_store_dword v5, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_read_u16_oracle_zero_ext")
        self.assertEqual(results[0], 0x1234)
        self.assertEqual(results[1], 0x8ABC)

    def test_read_u16_wrapping(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x20",
            "v_mov_b32_e32 v2, 0xBEEF",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_read_u16_oracle_wrap")
        self.assertEqual(results[0], 0xBEEF)


class TestDsReadU16AST(unittest.TestCase):
    def test_read_u16_ast_eval(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x55AA",
            "ds_write_b16 v1, v2 offset:0x40",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:0x40",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_read_u16_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpBitFieldUExtract", spv)
        results = run_cs(body, name="cs_read_u16_ast_eval")
        self.assertEqual(results[0], 0x55AA)


class TestDsReadU16SIMD(unittest.TestCase):
    def test_simd_read_u16_divergent_lanes(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # lane * 4
            "v_add_nc_u32_e32 v2, 0x1000, v0",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_read_u16_simd_div")
        for l in range(32):
            self.assertEqual(results[l], 0x1000 + l)


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


class TestDsReadI16Oracle(unittest.TestCase):
    def test_read_i16_sign_extension(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x200",
            "v_mov_b32_e32 v2, 0x7FFF8001",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1 offset:0",
            "ds_read_i16 v4, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "v_cmp_ne_u32_e64 s[4:5], v0, 0",
            "v_cndmask_b32_e64 v5, v3, v4, s[4:5]",
            "buffer_store_dword v5, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_read_i16_oracle", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpBitFieldSExtract", spv)
        results = run_cs(body, name="cs_read_i16_oracle")
        # 0x8001 is negative: sign-extends to 0xFFFF8001
        self.assertEqual(results[0], 0xFFFF8001)
        # 0x7FFF is positive: extends to 0x00007FFF
        self.assertEqual(results[1], 0x7FFF)


class TestDsReadI16AST(unittest.TestCase):
    def test_read_i16_ast_eval(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x30",
            "v_mov_b32_e32 v2, 0xF000",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_read_i16_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpBitFieldSExtract", spv)
        results = run_cs(body, name="cs_read_i16_ast_eval")
        self.assertEqual(results[0], 0xFFFFF000)


class TestDsReadI16SIMD(unittest.TestCase):
    def test_simd_read_i16_divergent_signs(self) -> None:
        # Odd lanes negative (0x8000 + l -> 0xFFFF8000 + l), even lanes positive (0x1000 + l)
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # lane * 4
            "v_and_b32_e32 v2, 1, v0",     # l % 2
            "v_cmp_ne_u32_e64 s[4:5], v2, 0",
            "v_cndmask_b32_e64 v3, 0x1000, 0x8000, s[4:5]",
            "v_add_nc_u32_e32 v3, v3, v0",
            "ds_write_b16 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v4, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v4, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_read_i16_simd_signs", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpBitFieldSExtract", spv)
        results = run_cs(body, name="cs_read_i16_simd_signs")
        for l in range(32):
            if l % 2 == 1:
                self.assertEqual(results[l], (0xFFFF8000 + l) & MASK32)
            else:
                self.assertEqual(results[l], 0x1000 + l)


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


class TestDsWriteB16Oracle(unittest.TestCase):
    def test_write_b16_preserves_neighbor_bytes(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x50",
            "v_mov_b32_e32 v2, 0x11223344",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xAAAA",
            "ds_write_b16 v1, v3 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v4, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b16_oracle_preserve")
        # Bytes 0 and 1 (0x3344) preserved, bytes 2 and 3 replaced by 0xAAAA -> 0xAAAA3344
        self.assertEqual(results[0], 0xAAAA3344)


class TestDsWriteB16AST(unittest.TestCase):
    def test_write_b16_ast_eval(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x10",
            "v_mov_b32_e32 v2, 0x4321",
            "ds_write_b16 v1, v2 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_write_b16_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpAtomicOr", spv)
        results = run_cs(body, name="cs_write_b16_ast_eval")
        self.assertEqual(results[0], 0x4321)


class TestDsWriteB16SIMD(unittest.TestCase):
    def test_simd_write_b16_multi_lane(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # lane * 4
            "v_add_nc_u32_e32 v2, 0xA000, v0",
            "ds_write_b16 v1, v2 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b16_simd_multi")
        for l in range(32):
            self.assertEqual(results[l], 0xA000 + l)


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


class TestDsWriteB32Oracle(unittest.TestCase):
    def test_write_b32_oracle(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x100",
            "v_mov_b32_e32 v2, 0xDEADBEEF",
            "ds_write_b32 v1, v2 offset:0x20",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1 offset:0x20",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b32_oracle")
        self.assertEqual(results[0], 0xDEADBEEF)

    def test_write_b32_wrapping(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x10",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_write_b32 v1, v2 offset:8",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1 offset:8",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b32_wrap")
        self.assertEqual(results[0], 0x12345678)


class TestDsWriteB32AST(unittest.TestCase):
    def test_write_b32_ast_eval(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x20",
            "v_mov_b32_e32 v2, 0xCAFEBABE",
            "ds_write_b32 v1, v2 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_write_b32_ast_eval", ["v_lshlrev_b32_e32 v30, 5, v0"] + body + ["s_endpgm"], fssm=FSSM)
        self.assertIn("OpStore", spv)
        results = run_cs(body, name="cs_write_b32_ast_eval")
        self.assertEqual(results[0], 0xCAFEBABE)


class TestDsWriteB32SIMD(unittest.TestCase):
    def test_simd_write_b32_divergent_lanes(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # lane * 4
            "v_add_nc_u32_e32 v2, 0x70000000, v0",
            "ds_write_b32 v1, v2 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b32_simd_div")
        for l in range(32):
            self.assertEqual(results[l], 0x70000000 + l)


class TestDsWriteB32Translator(unittest.TestCase):
    def test_translator_write_b32_roundtrip(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # addr = lane * 4
            "v_add_nc_u32_e32 v2, 0x5000, v0",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_write_b32_exec")
        for l in range(32):
            self.assertEqual(results[l], 0x5000 + l)


if __name__ == "__main__":
    unittest.main()
