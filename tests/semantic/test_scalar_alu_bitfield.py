"""Semantic interpreter tests for RDNA2 scalar bitfield operations.

Instructions tested:
  - s_bfe_u32 D, S0, S1: Unsigned scalar bitfield extract.
  - s_bfe_i32 D, S0, S1: Signed scalar bitfield extract (sign-extended).
  - s_bfm_b32 D, S0, S1: Scalar bitfield mask generation.

RDNA2 ISA rules:
  - s_bfe_u32 / s_bfe_i32:
    * S0 is the source dword (SGPR, immediate, or constant).
    * S1 is the packed control word:
        offset = S1[4:0] (bits [4:0], 0..31).
        width  = S1[22:16] (bits [22:16], 0..127).
        Other bits of S1 are ignored.
    * Width limit / clamping: SPIR-V specifies that offset + count <= 32. In RDNA2
      hardware and LLVM constantFoldBFE, an extract running past bit 31 is clamped
      such that the effective width is min(width, 32 - offset).
    * If width == 0 or offset >= 32: result is 0.
    * s_bfe_u32 zero-extends the extracted field into D.
    * s_bfe_i32 sign-extends the extracted field from bit (effective_width - 1).
    * Condition Code (SCC): SCC = 1 if D != 0 else 0 (both signed and unsigned).
  - s_bfm_b32:
    * S0 is the width operand (only low 5 bits S0[4:0] are used).
    * S1 is the offset operand (only low 5 bits S1[4:0] are used).
    * D = ((1 << (S0 & 31)) - 1) << (S1 & 31).
    * If (S0 & 31) == 0, result is 0.
    * s_bfm_b32 does NOT alter the SCC flag.
  - Scalar bitfield insert composition:
    * Scalar bitfield insert is composed via s_bfm_b32 and bitwise selection:
      mask = s_bfm_b32(width, offset)
      result = (base & ~mask) | ((insert << offset) & mask)
    * Matches SPIR-V OpBitFieldInsert semantics when offset + count <= 32.

Verification approach:
  1. Pure Python reference models verifying exact RDNA2 bitfield math.
  2. Deterministic ScalarAluInterpreter tracking SGPR state and SCC.
  3. rdna2spirv translation emission checks via rdna2spirv.Gen.
  4. spirv_mini_interp execution cross-checking lowered SPIR-V modules.
  5. Negative controls ensuring test sensitivity to subtle bitfield errors.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
PUSH_VSHARP = [0x1000, 0, 0x1000, 0x31016FAC] + [0] * 28


def pack_bfe_ctl(offset: int, width: int) -> int:
    """Pack bitfield offset [4:0] and width [22:16] into an RDNA scalar control word."""
    return ((width & 0x7F) << 16) | (offset & 0x1F)


def unpack_bfe_ctl(ctl: int) -> tuple[int, int]:
    """Unpack offset [4:0] and width [22:16] from an RDNA scalar control word."""
    offset = ctl & 0x1F
    width = (ctl >> 16) & 0x7F
    return offset, width


def ref_s_bfe_u32(src: int, ctl: int) -> tuple[int, int]:
    """Reference implementation of RDNA2 s_bfe_u32.

    Returns:
        (result, scc): 32-bit unsigned extracted field and SCC flag (1 if res != 0 else 0).
    """
    src &= MASK32
    offset, width = unpack_bfe_ctl(ctl)
    if width == 0 or offset >= 32:
        return 0, 0
    eff_width = min(width, 32 - offset)
    mask = (1 << eff_width) - 1
    res = (src >> offset) & mask
    scc = 1 if res != 0 else 0
    return res & MASK32, scc


def ref_s_bfe_i32(src: int, ctl: int) -> tuple[int, int]:
    """Reference implementation of RDNA2 s_bfe_i32.

    Returns:
        (result, scc): 32-bit sign-extended extracted field and SCC flag (1 if res != 0 else 0).
    """
    src &= MASK32
    offset, width = unpack_bfe_ctl(ctl)
    if width == 0 or offset >= 32:
        return 0, 0
    eff_width = min(width, 32 - offset)
    mask = (1 << eff_width) - 1
    val = (src >> offset) & mask
    sign_bit = 1 << (eff_width - 1)
    if val & sign_bit:
        val -= (1 << eff_width)
    res = val & MASK32
    scc = 1 if res != 0 else 0
    return res, scc


def ref_s_bfm_b32(width: int, offset: int) -> int:
    """Reference implementation of RDNA2 s_bfm_b32.

    Mask generator: ((1 << (width & 31)) - 1) << (offset & 31).
    Note: Does NOT alter SCC.
    """
    w = width & 31
    o = offset & 31
    if w == 0:
        return 0
    mask = ((1 << w) - 1) & MASK32
    return (mask << o) & MASK32


def ref_bitfield_insert(base: int, insert: int, offset: int, count: int) -> int:
    """Reference implementation of bitfield insert matching SPIR-V OpBitFieldInsert."""
    base &= MASK32
    insert &= MASK32
    if count == 0:
        return base
    mk = (((1 << count) - 1) << offset) & MASK32
    return ((base & ~mk) | ((insert << offset) & mk)) & MASK32


class ScalarAluInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar ALU bitfield instructions.

    Models SGPR file (106 registers) and 1-bit Scalar Condition Code (SCC).
    """

    def __init__(self, num_sgprs: int = 106) -> None:
        self.num_sgprs = num_sgprs
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self) -> None:
        self.sgpr = [0] * self.num_sgprs
        self.scc = 0

    def get_sgpr(self, r: int) -> int:
        return self.sgpr[r] & MASK32

    def set_sgpr(self, r: int, val: int) -> None:
        self.sgpr[r] = val & MASK32

    def resolve_src(self, opnd: int | dict | str) -> int:
        """Resolve scalar operand to 32-bit unsigned value."""
        if isinstance(opnd, int):
            return opnd & MASK32
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "imm":
                return opnd["v"] & MASK32
            if k == "sgpr":
                return self.get_sgpr(opnd["r"])
            raise ValueError(f"Unknown operand dict: {opnd}")
        if isinstance(opnd, str):
            if opnd.startswith("s"):
                return self.get_sgpr(int(opnd[1:]))
            return int(opnd, 0) & MASK32
        raise TypeError(f"Unsupported operand type: {type(opnd)}")

    def resolve_dst(self, dst: int | dict | str) -> int:
        """Resolve scalar destination to SGPR register index."""
        if isinstance(dst, int):
            return dst
        if isinstance(dst, dict) and dst.get("k") == "sgpr":
            return dst["r"]
        if isinstance(dst, str) and dst.startswith("s"):
            return int(dst[1:])
        raise ValueError(f"Unsupported destination operand: {dst}")

    def step(self, op: str, dst: int | dict | str, s0: int | dict | str, s1: int | dict | str) -> int:
        """Execute a single scalar ALU instruction."""
        v0 = self.resolve_src(s0)
        v1 = self.resolve_src(s1)
        d_idx = self.resolve_dst(dst)

        if op == "s_bfe_u32":
            res, scc = ref_s_bfe_u32(v0, v1)
            self.set_sgpr(d_idx, res)
            self.scc = scc
            return res
        if op == "s_bfe_i32":
            res, scc = ref_s_bfe_i32(v0, v1)
            self.set_sgpr(d_idx, res)
            self.scc = scc
            return res
        if op == "s_bfm_b32":
            res = ref_s_bfm_b32(v0, v1)
            self.set_sgpr(d_idx, res)
            # RDNA2 s_bfm_b32 does not update SCC
            return res
        if op == "s_mov_b32":
            self.set_sgpr(d_idx, v0)
            return v0
        if op == "s_and_b32":
            res = (v0 & v1) & MASK32
            self.set_sgpr(d_idx, res)
            self.scc = 1 if res != 0 else 0
            return res
        if op == "s_or_b32":
            res = (v0 | v1) & MASK32
            self.set_sgpr(d_idx, res)
            self.scc = 1 if res != 0 else 0
            return res
        if op == "s_lshl_b32":
            sh = v1 & 31
            res = (v0 << sh) & MASK32
            self.set_sgpr(d_idx, res)
            self.scc = 1 if res != 0 else 0
            return res
        raise NotImplementedError(f"Instruction {op} not implemented in interpreter")

    def run_sequence(self, instructions: list[tuple[str, int | dict | str, int | dict | str, int | dict | str]]) -> int:
        """Run a sequence of instructions (op, dst, s0, s1)."""
        last_res = 0
        for op, dst, s0, s1 in instructions:
            last_res = self.step(op, dst, s0, s1)
        return last_res


class ReferenceModelTests(unittest.TestCase):
    """Validation of pure-Python bitfield reference models."""

    def test_pack_unpack_ctl(self) -> None:
        ctl = pack_bfe_ctl(5, 12)
        off, wid = unpack_bfe_ctl(ctl)
        self.assertEqual(off, 5)
        self.assertEqual(wid, 12)
        self.assertEqual(ctl, (12 << 16) | 5)

        # Upper bits [15:5] and [31:23] ignored on unpack
        dirty = ctl | 0xFF80FFE0
        off_dirty, wid_dirty = unpack_bfe_ctl(dirty)
        self.assertEqual(off_dirty, 5)
        self.assertEqual(wid_dirty, 12)

    def test_ref_s_bfe_u32_basic(self) -> None:
        # Extract 8 bits from offset 8
        ctl = pack_bfe_ctl(8, 8)
        res, scc = ref_s_bfe_u32(0x12345678, ctl)
        self.assertEqual(res, 0x56)
        self.assertEqual(scc, 1)

    def test_ref_s_bfe_u32_zero_width(self) -> None:
        ctl = pack_bfe_ctl(8, 0)
        res, scc = ref_s_bfe_u32(0xFFFFFFFF, ctl)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_ref_s_bfe_u32_full_width(self) -> None:
        ctl = pack_bfe_ctl(0, 32)
        res, scc = ref_s_bfe_u32(0xDEADBEEF, ctl)
        self.assertEqual(res, 0xDEADBEEF)
        self.assertEqual(scc, 1)

    def test_ref_s_bfe_u32_clamping_past_bit_31(self) -> None:
        # Offset 28, width 8 runs to bit 36: effective width is min(8, 4) = 4 bits
        ctl = pack_bfe_ctl(28, 8)
        res, scc = ref_s_bfe_u32(0xF0000000, ctl)
        self.assertEqual(res, 0xF)
        self.assertEqual(scc, 1)

        # Width up to 127
        ctl127 = pack_bfe_ctl(16, 127)
        res127, scc127 = ref_s_bfe_u32(0xABCD0000, ctl127)
        self.assertEqual(res127, 0xABCD)
        self.assertEqual(scc127, 1)

    def test_ref_s_bfe_i32_positive(self) -> None:
        ctl = pack_bfe_ctl(4, 4)
        # 0x70 >> 4 is 0x7 (sign bit 3 is 0 -> positive 7)
        res, scc = ref_s_bfe_i32(0x70, ctl)
        self.assertEqual(res, 7)
        self.assertEqual(scc, 1)

    def test_ref_s_bfe_i32_negative(self) -> None:
        ctl = pack_bfe_ctl(4, 4)
        # 0x80 >> 4 is 0x8 (sign bit 3 is 1 -> negative, -8 = 0xFFFFFFF8)
        res, scc = ref_s_bfe_i32(0x80, ctl)
        self.assertEqual(res, 0xFFFFFFF8)
        self.assertEqual(scc, 1)

        # 0xF0 >> 4 is 0xF (sign bit 3 is 1 -> -1 = 0xFFFFFFFF)
        res, scc = ref_s_bfe_i32(0xF0, ctl)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_ref_s_bfe_i32_zero_width(self) -> None:
        ctl = pack_bfe_ctl(10, 0)
        res, scc = ref_s_bfe_i32(0xFFFFFFFF, ctl)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_ref_s_bfe_i32_clamping_past_bit_31(self) -> None:
        # Offset 28, width 8: effective width is 4. Value 0x80000000 >> 28 is 8 (negative in 4-bit)
        ctl = pack_bfe_ctl(28, 8)
        res, scc = ref_s_bfe_i32(0x80000000, ctl)
        self.assertEqual(res, 0xFFFFFFF8)
        self.assertEqual(scc, 1)

    def test_ref_s_bfm_b32_basic(self) -> None:
        # width = 8, offset = 8 -> 0x0000FF00
        mask = ref_s_bfm_b32(8, 8)
        self.assertEqual(mask, 0x0000FF00)

        # width = 4, offset = 28 -> 0xF0000000
        mask = ref_s_bfm_b32(4, 28)
        self.assertEqual(mask, 0xF0000000)

        # width = 0 -> 0
        mask = ref_s_bfm_b32(0, 16)
        self.assertEqual(mask, 0)

        # width = 31, offset = 0 -> 0x7FFFFFFF
        mask = ref_s_bfm_b32(31, 0)
        self.assertEqual(mask, 0x7FFFFFFF)

        # width = 32 (32 & 31 = 0) -> 0
        mask = ref_s_bfm_b32(32, 0)
        self.assertEqual(mask, 0)

        # width modulo 32 and offset modulo 32
        self.assertEqual(ref_s_bfm_b32(8 + 32, 8 + 64), 0x0000FF00)

    def test_ref_bitfield_insert(self) -> None:
        base = 0x12345678
        insert = 0xAB
        # insert 8 bits at offset 8
        res = ref_bitfield_insert(base, insert, 8, 8)
        self.assertEqual(res, 0x1234AB78)

        # count = 0 returns base unchanged
        self.assertEqual(ref_bitfield_insert(base, insert, 8, 0), base)

        # full 32-bit replace
        self.assertEqual(ref_bitfield_insert(base, 0xCAFEBABE, 0, 32), 0xCAFEBABE)


class ScalarAluInterpreterTests(unittest.TestCase):
    """Execution tests for the ScalarAluInterpreter modeling RDNA2 scalar instructions."""

    def setUp(self) -> None:
        self.interp = ScalarAluInterpreter()

    def test_s_bfe_u32_all_single_bit_positions(self) -> None:
        # Verify single bit extraction across all 32 positions (0..31)
        for pos in range(32):
            pattern = 1 << pos
            ctl = pack_bfe_ctl(pos, 1)
            self.interp.set_sgpr(0, pattern)
            self.interp.set_sgpr(1, ctl)

            res = self.interp.step("s_bfe_u32", 2, {"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 1})
            self.assertEqual(res, 1)
            self.assertEqual(self.interp.get_sgpr(2), 1)
            self.assertEqual(self.interp.scc, 1)

            # Inverted bit test (bit is 0)
            self.interp.set_sgpr(0, pattern ^ (1 << pos))
            res_zero = self.interp.step("s_bfe_u32", 2, {"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 1})
            self.assertEqual(res_zero, 0)
            self.assertEqual(self.interp.get_sgpr(2), 0)
            self.assertEqual(self.interp.scc, 0)

    def test_s_bfe_u32_scc_transitions(self) -> None:
        # Start with scc = 0
        self.interp.scc = 0
        self.interp.set_sgpr(0, 0xFF)
        self.interp.set_sgpr(1, pack_bfe_ctl(0, 4))
        self.interp.step("s_bfe_u32", "s2", "s0", "s1")
        self.assertEqual(self.interp.scc, 1)

        # Non-zero to zero: SCC must clear
        self.interp.set_sgpr(0, 0x00)
        self.interp.step("s_bfe_u32", "s2", "s0", "s1")
        self.assertEqual(self.interp.scc, 0)

    def test_s_bfe_i32_sign_extension_matrix(self) -> None:
        cases = [
            # (val, offset, width, expected_result, expected_scc)
            (0x00000007, 0, 4, 7, 1),
            (0x00000008, 0, 4, 0xFFFFFFF8, 1),
            (0x0000000F, 0, 4, 0xFFFFFFFF, 1),
            (0x00000000, 0, 4, 0, 0),
            (0x00008000, 8, 8, 0xFFFFFF80, 1),
            (0x00007F00, 8, 8, 0x0000007F, 1),
            (0x80000000, 16, 16, 0xFFFF8000, 1),
            (0x7FFF0000, 16, 16, 0x00007FFF, 1),
            (0x00000001, 0, 1, 0xFFFFFFFF, 1),
            (0x00000000, 0, 1, 0x00000000, 0),
        ]
        for val, off, wid, want_res, want_scc in cases:
            with self.subTest(val=hex(val), off=off, wid=wid):
                ctl = pack_bfe_ctl(off, wid)
                self.interp.set_sgpr(0, val)
                self.interp.set_sgpr(1, ctl)
                res = self.interp.step("s_bfe_i32", "s2", "s0", "s1")
                self.assertEqual(res, want_res)
                self.assertEqual(self.interp.get_sgpr(2), want_res)
                self.assertEqual(self.interp.scc, want_scc)

    def test_s_bfm_b32_mask_generation(self) -> None:
        test_vectors = [
            # (width, offset, expected_mask)
            (1, 0, 0x00000001),
            (1, 31, 0x80000000),
            (8, 0, 0x000000FF),
            (8, 8, 0x0000FF00),
            (8, 16, 0x00FF0000),
            (8, 24, 0xFF000000),
            (16, 0, 0x0000FFFF),
            (16, 16, 0xFFFF0000),
            (31, 0, 0x7FFFFFFF),
            (0, 0, 0x00000000),
            (0, 16, 0x00000000),
            (32, 0, 0x00000000),  # 32 & 31 == 0
        ]
        for w, o, want in test_vectors:
            with self.subTest(w=w, o=o):
                self.interp.set_sgpr(0, w)
                self.interp.set_sgpr(1, o)
                res = self.interp.step("s_bfm_b32", "s2", "s0", "s1")
                self.assertEqual(res, want)
                self.assertEqual(self.interp.get_sgpr(2), want)

    def test_s_bfm_b32_preserves_scc(self) -> None:
        # s_bfm_b32 must not overwrite SCC whether SCC is 0 or 1
        for initial_scc in (0, 1):
            self.interp.scc = initial_scc
            self.interp.set_sgpr(0, 8)
            self.interp.set_sgpr(1, 4)
            self.interp.step("s_bfm_b32", "s2", "s0", "s1")
            self.assertEqual(self.interp.scc, initial_scc)

    def test_in_place_destination_aliasing(self) -> None:
        # Destination aliases source operand
        ctl = pack_bfe_ctl(8, 8)
        self.interp.set_sgpr(0, 0x12345678)
        self.interp.set_sgpr(1, ctl)
        res = self.interp.step("s_bfe_u32", "s0", "s0", "s1")
        self.assertEqual(res, 0x56)
        self.assertEqual(self.interp.get_sgpr(0), 0x56)

        # Dest aliases control word
        self.interp.set_sgpr(0, 0x12345678)
        self.interp.set_sgpr(1, ctl)
        res = self.interp.step("s_bfe_u32", "s1", "s0", "s1")
        self.assertEqual(res, 0x56)
        self.assertEqual(self.interp.get_sgpr(1), 0x56)

    def test_adjacent_registers_untouched(self) -> None:
        self.interp.set_sgpr(0, 0x11111111)
        self.interp.set_sgpr(1, 0x22222222)
        self.interp.set_sgpr(2, 0x33333333)
        self.interp.set_sgpr(3, 0x44444444)

        ctl = pack_bfe_ctl(4, 4)
        self.interp.step("s_bfe_u32", "s1", "s2", ctl)
        self.assertEqual(self.interp.get_sgpr(0), 0x11111111)
        self.assertEqual(self.interp.get_sgpr(2), 0x33333333)
        self.assertEqual(self.interp.get_sgpr(3), 0x44444444)

    def test_scalar_bitfield_insert_composition_pipeline(self) -> None:
        # Simulate scalar bitfield insert:
        # s_bfm_b32  s_mask, count, offset
        # s_lshl_b32 s_ins_shifted, s_ins, offset
        # s_and_b32  s_ins_masked, s_ins_shifted, s_mask
        # s_and_b32  s_base_cleared, s_base, ~s_mask
        # s_or_b32   s_res, s_base_cleared, s_ins_masked
        base_val = 0xAA55AA55
        ins_val = 0x3F
        offset = 12
        count = 6

        mask = ref_s_bfm_b32(count, offset)
        expected = ref_bitfield_insert(base_val, ins_val, offset, count)

        self.interp.set_sgpr(0, base_val)
        self.interp.set_sgpr(1, ins_val)
        self.interp.set_sgpr(2, count)
        self.interp.set_sgpr(3, offset)

        seq = [
            ("s_bfm_b32", "s4", "s2", "s3"),                       # s4 = mask
            ("s_lshl_b32", "s5", "s1", "s3"),                      # s5 = ins << offset
            ("s_and_b32", "s5", "s5", "s4"),                       # s5 = (ins << offset) & mask
            ("s_and_b32", "s6", "s0", {"k": "imm", "v": ~mask}),  # s6 = base & ~mask
            ("s_or_b32", "s7", "s6", "s5"),                        # s7 = s6 | s5
        ]
        result = self.interp.run_sequence(seq)
        self.assertEqual(result, expected)
        self.assertEqual(self.interp.get_sgpr(7), expected)

        # Roundtrip: extract the inserted field back out with s_bfe_u32
        ctl = pack_bfe_ctl(offset, count)
        extracted = self.interp.step("s_bfe_u32", "s8", "s7", ctl)
        self.assertEqual(extracted, ins_val)


class TranslatorEmissionTests(unittest.TestCase):
    """Verify that rdna2spirv.Gen correctly emits SPIR-V for s_bfe and s_bfm instructions."""

    def test_translate_s_bfe_u32_emission(self) -> None:
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bfe_u32",
            "args": [
                {"k": "sgpr", "r": 2},
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        # Unpack offset [4:0] and width [22:16]
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpBitFieldUExtract %uint", emitted)
        # Clamping via bfe_width
        self.assertIn("OpISub %uint", emitted)
        self.assertIn("OpULessThan %bool", emitted)
        # Store result and SCC
        self.assertIn("OpStore %scc", emitted)

    def test_translate_s_bfe_i32_emission(self) -> None:
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bfe_i32",
            "args": [
                {"k": "sgpr", "r": 3},
                {"k": "sgpr", "r": 4},
                {"k": "sgpr", "r": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertIn("OpBitFieldSExtract %int", emitted)
        self.assertIn("OpBitcast %int", emitted)
        self.assertIn("OpBitcast %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_translate_s_bfm_b32_emission(self) -> None:
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bfm_b32",
            "args": [
                {"k": "sgpr", "r": 6},
                {"k": "sgpr", "r": 7},
                {"k": "sgpr", "r": 8},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertIn("OpShiftLeftLogical %uint", emitted)
        self.assertIn("OpISub %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        # s_bfm_b32 must NOT emit SCC store
        self.assertNotIn("OpStore %scc", emitted)

    def test_translation_determinism(self) -> None:
        def emit_sequence() -> str:
            g = rdna2spirv.Gen("ps")
            for op in ("s_bfe_u32", "s_bfe_i32", "s_bfm_b32"):
                g.insn({
                    "op": op,
                    "args": [
                        {"k": "sgpr", "r": 0},
                        {"k": "sgpr", "r": 1},
                        {"k": "sgpr", "r": 2},
                    ],
                    "mods": [],
                })
            return "\n".join(g.body)

        self.assertEqual(emit_sequence(), emit_sequence())


SPIRV_HEADER = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
{constants}
%main = OpFunction %void None %fn
%entry = OpLabel
{body}
OpStore %out %res
OpReturn
OpFunctionEnd
"""


def run_spv(body: str, const_defs: dict[str, int]) -> int:
    """Assemble and run a SPIR-V fragment module defining %res."""
    consts_text = "\n".join(f"%{name} = OpConstant %uint {val & MASK32}" for name, val in const_defs.items())
    mod_text = SPIRV_HEADER.format(constants=consts_text, body=body)
    return si.run_fragment(mod_text)["%out"][0]


class SpirvMiniInterpTests(unittest.TestCase):
    """Cross-validation between interpreter models and direct spirv_mini_interp execution."""

    def test_direct_uextract_in_spirv(self) -> None:
        consts = {"x": 0xDEADBEEF, "off": 8, "wid": 8}
        body = "%res = OpBitFieldUExtract %uint %x %off %wid\n"
        self.assertEqual(run_spv(body, consts), 0xBE)

    def test_direct_sextract_in_spirv(self) -> None:
        # Negative 4-bit field: 0xF0 >> 4 is 0xF -> sign extended to 0xFFFFFFFF
        consts = {"x": 0xF0, "off": 4, "wid": 4}
        body = """\
%xi = OpBitcast %int %x
%ri = OpBitFieldSExtract %int %xi %off %wid
%res = OpBitcast %uint %ri
"""
        self.assertEqual(run_spv(body, consts), 0xFFFFFFFF)

    def test_direct_insert_in_spirv(self) -> None:
        consts = {"base": 0xFFFFFFFF, "ins": 0x12, "off": 8, "cnt": 8}
        body = "%res = OpBitFieldInsert %uint %base %ins %off %cnt\n"
        self.assertEqual(run_spv(body, consts), 0xFFFF12FF)

    def test_lowered_s_bfe_u32_with_large_width_avoids_rejection(self) -> None:
        # SPIR-V rejects offset + count > 32 if passed directly to OpBitFieldUExtract.
        # But rdna2spirv lowers width via bfe_width:
        # room = 32 - off; fits = wid < room; clamped_wid = fits ? wid : room
        # Here offset = 28, raw width = 16 (28 + 16 = 44 > 32).
        # Clamped width becomes 4, extracting the top 4 bits (0xA).
        ctl = pack_bfe_ctl(28, 16)
        consts = {"x": 0xA0000000, "ctl": ctl, "c31": 31, "c16": 16, "c7": 7, "c32": 32}
        body = """\
%off = OpBitwiseAnd %uint %ctl %c31
%raw_wid = OpBitFieldUExtract %uint %ctl %c16 %c7
%room = OpISub %uint %c32 %off
%fits = OpULessThan %bool %raw_wid %room
%clamped_wid = OpSelect %uint %fits %raw_wid %room
%res = OpBitFieldUExtract %uint %x %off %clamped_wid
"""
        got = run_spv(body, consts)
        self.assertEqual(got, 0xA)
        expected, _ = ref_s_bfe_u32(0xA0000000, ctl)
        self.assertEqual(got, expected)

    def test_direct_out_of_bounds_raises_unsupported(self) -> None:
        # Direct raw OpBitFieldUExtract with off + wid > 32 must be rejected by spirv_mini_interp
        consts = {"x": 0xFF, "off": 30, "wid": 4}
        body = "%res = OpBitFieldUExtract %uint %x %off %wid\n"
        with self.assertRaises(si.UnsupportedConstruct):
            run_spv(body, consts)

        consts_ins = {"base": 0, "ins": 1, "off": 25, "cnt": 10}
        body_ins = "%res = OpBitFieldInsert %uint %base %ins %off %cnt\n"
        with self.assertRaises(si.UnsupportedConstruct):
            run_spv(body_ins, consts_ins)


def translate_ps_bitfield(lines: list[str]) -> str:
    """Translate a hand-written pixel shader listing exporting v0..v3 to mrt0."""
    return translate_lines(
        "ps_salu_bitfield",
        [*lines, "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
    )


class EndToEndTranslatorTests(unittest.TestCase):
    """End-to-end rdna2spirv.translate + spirv_mini_interp execution for s_bfe_u32, s_bfe_i32, and s_bfm_b32."""

    def test_runtime_s_bfe_u32_and_i32_dynamic_ctl_and_scc_transitions(self) -> None:
        # Both S0 and S1 are loaded from buffer memory at runtime (neither is in sgpr_imm),
        # exercising the dynamic control-word unpack + bfe_width clamp path and chained
        # 0->1, 1->0, 1->1, and 0->0 SCC transitions without re-seeding SCC between steps.
        for op, ref_fn in (("s_bfe_u32", ref_s_bfe_u32), ("s_bfe_i32", ref_s_bfe_i32)):
            spv = translate_ps_bitfield([
                "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
                "s_waitcnt lgkmcnt(0)",
                "s_cmp_eq_u32 0, 1",          # initial SCC = 0
                f"{op} s8, s4, s5",           # step 1: transition from SCC = 0
                "s_cselect_b32 s9, 1, 0",
                f"{op} s10, s6, s7",          # step 2: transition directly from step 1's SCC
                "s_cselect_b32 s11, 1, 0",
                "v_mov_b32 v0, s8",
                "v_mov_b32 v1, s9",
                "v_mov_b32 v2, s10",
                "v_mov_b32 v3, s11",
            ])

            cases = [
                # 0 -> 1 (8-bit field with dirty ignored bits in S1), then 1 -> 0 (width == 0 on all-ones source)
                (0x12345678, pack_bfe_ctl(8, 8) | 0xFF80FFE0, 0xFFFFFFFF, pack_bfe_ctl(12, 0) | 0x80001FE0),
                # 0 -> 0 (zero field), then 0 -> 1 (clamped past bit 31: offset=28, width=16 -> effective width 4)
                (0xFFFF00FF, pack_bfe_ctl(8, 8), 0x80000000, pack_bfe_ctl(28, 16)),
                # 0 -> 1 (1-bit negative/nonzero field), then 1 -> 1 (max 7-bit width=127 at offset=16 -> top 16 bits)
                (0x00000001, pack_bfe_ctl(0, 1), 0xABCD1234, pack_bfe_ctl(16, 127)),
                # 0 -> 1 (full 32-bit extract: offset=0, width=32), then 1 -> 0 (1-bit zero field at bit 31)
                (0xDEADBEEF, pack_bfe_ctl(0, 32), 0x7FFFFFFF, pack_bfe_ctl(31, 5)),
                # 0 -> 0 (width=0 at offset=31), then 0 -> 0 (extracted field is all zeros)
                (0x80000000, pack_bfe_ctl(31, 0), 0xF0F00000, pack_bfe_ctl(4, 12)),
                # Sign-bit boundary transitions for 4-bit (0x7 vs 0x8) and 8-bit (0x7F vs 0x80) fields
                (0x00000070, pack_bfe_ctl(4, 4), 0x00000080, pack_bfe_ctl(4, 4)),
                (0x00007F00, pack_bfe_ctl(8, 8), 0x00008000, pack_bfe_ctl(8, 8)),
            ]

            for s0_a, ctl_a, s0_b, ctl_b in cases:
                want_da, want_scca = ref_fn(s0_a, ctl_a)
                want_db, want_sccb = ref_fn(s0_b, ctl_b)
                mem = {
                    0x1000: s0_a,
                    0x1004: ctl_a,
                    0x1008: s0_b,
                    0x100C: ctl_b,
                }
                with self.subTest(op=op, s0_a=hex(s0_a), ctl_a=hex(ctl_a), s0_b=hex(s0_b), ctl_b=hex(ctl_b)):
                    got = tuple(mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem))
                    self.assertEqual(got, (want_da, want_scca, want_db, want_sccb))

    def test_runtime_src_with_folded_ctl_path(self) -> None:
        # S0 is loaded at runtime from memory (x_val is None), while S1 is an immediate or
        # sgpr_imm constant (ctl_val is not None), exercising rdna2spirv's constant-control-word
        # lowering branch (including compile-time width clamping past bit 31 and width == 0).
        ctl_clamped = pack_bfe_ctl(28, 12) | 0x008003E0  # dirty bits + width 12 clamped to 4 at offset 28
        ctl_byte1 = pack_bfe_ctl(8, 8)
        ctl_zero_wid = pack_bfe_ctl(4, 0)
        spv = translate_ps_bitfield([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            f"s_mov_b32 s6, {ctl_clamped:#x}",
            "s_bfe_u32 s8, s4, s6",                      # sgpr_imm control word, clamped width
            f"s_bfe_i32 s9, s4, {ctl_clamped:#x}",       # immediate control word, signed clamped width
            "s_cmp_eq_u32 1, 1",                         # pre-seed SCC = 1
            f"s_bfe_u32 s10, s4, {ctl_zero_wid:#x}",     # width == 0 -> D = 0, SCC transitions 1 -> 0
            "s_cselect_b32 s10, 1, 0",
            f"s_bfe_i32 s11, s5, {ctl_byte1:#x}",        # signed byte extract, SCC transitions 0 -> 1
            "s_cselect_b32 s12, 1, 0",
            "s_lshl_b32 s10, s10, 1",
            "s_or_b32 s10, s10, s12",                    # pack (scc_after_zero_wid << 1) | scc_after_byte1
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s11",
            "v_mov_b32 v3, s10",
        ])

        for s4_val, s5_val in (
            (0x80000000, 0x00008000),  # nibble = 8 (-8 signed), byte1 = 0x80 (-128 signed)
            (0x70000000, 0x00007F00),  # nibble = 7 (+7 signed), byte1 = 0x7F (+127 signed)
            (0xF0000000, 0xFFFF00FF),  # nibble = 15 (-1 signed), byte1 = 0x00 (0, SCC = 0)
        ):
            want_u, _ = ref_s_bfe_u32(s4_val, ctl_clamped)
            want_i, _ = ref_s_bfe_i32(s4_val, ctl_clamped)
            want_b1, want_scc_b1 = ref_s_bfe_i32(s5_val, ctl_byte1)
            mem = {0x1000: s4_val, 0x1004: s5_val}
            with self.subTest(s4=hex(s4_val), s5=hex(s5_val)):
                got = tuple(mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem))
                self.assertEqual(got, (want_u, want_i, want_b1, want_scc_b1))

    def test_runtime_bfe_aliasing_and_special_operands(self) -> None:
        # Verify dst==S0, dst==S1, dst==S0==S1, m0 as control word, and null destination on runtime values.
        spv = translate_ps_bitfield([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 m0, s5",
            "s_bfe_u32 s4, s4, m0",       # dst == S0, control word in m0
            "s_bfe_i32 s5, s6, s5",       # dst == S1 (control word SGPR)
            "s_bfe_u32 s7, s7, s7",       # dst == S0 == S1
            "s_cmp_eq_u32 1, 1",          # pre-seed SCC = 1
            "s_bfe_i32 null, s6, 0x40000",  # bits [3:0] of s6 (0) -> discarded, SCC transitions 1 -> 0
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s7",
            "v_mov_b32 v3, s8",
        ])
        ctl = pack_bfe_ctl(8, 8)          # 0x00080008
        self_val = pack_bfe_ctl(16, 8)    # 0x00080010: offset=16, width=8 -> extracts bits [23:16] = 0x08
        mem = {
            0x1000: 0x1234AB78,           # s4 -> bits [15:8] = 0xAB
            0x1004: ctl,                  # s5 -> control word (offset=8, width=8)
            0x1008: 0x00008500,           # s6 -> bits [15:8] = 0x85 -> sign-extended 0xFFFFFF85; bits [3:0] = 0
            0x100C: self_val,             # s7 -> dst == S0 == S1 -> 0x08
        }
        got = tuple(mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem))
        self.assertEqual(
            got,
            (
                ref_s_bfe_u32(0x1234AB78, ctl)[0],
                ref_s_bfe_i32(0x00008500, ctl)[0],
                ref_s_bfe_u32(self_val, self_val)[0],
                0,
            ),
        )

    def test_runtime_s_bfm_b32_modulo32_scc_preservation_and_insert_pipeline(self) -> None:
        # 1. Runtime s_bfm_b32 with dirty upper bits [31:5] in S0 (width) and S1 (offset),
        #    destination aliasing, and SCC preservation across SCC = 0 and SCC = 1.
        spv_bfm = translate_ps_bitfield([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # SCC = 0
            "s_bfm_b32 s8, s4, s5",       # must preserve SCC = 0
            "s_cselect_b32 s9, 1, 0",
            "s_cmp_eq_u32 1, 1",          # SCC = 1
            "s_bfm_b32 s10, s6, s7",      # must preserve SCC = 1
            "s_cselect_b32 s11, 1, 0",
            "s_lshl_b32 s9, s9, 1",
            "s_or_b32 s9, s9, s11",       # (scc0 << 1) | scc1 == 0b01 == 1
            "s_bfm_b32 s4, s4, s5",       # dst == S0 aliasing
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s10",
            "v_mov_b32 v2, s4",
            "v_mov_b32 v3, s9",
        ])

        bfm_pairs = [
            # Standard and modulo-32 dirty upper bits
            (8, 8, 8 | 0xFFFFFFE0, 8 | 0x80000040),
            # width & 31 == 0 (32 -> 0) and width = 31 at offset = 0 (0x7FFFFFFF)
            (32, 12, 31 | 0x100, 0 | 0x40),
            # width + offset > 32 truncates to 32 bits (width=16, offset=24 -> 0xFF000000)
            (16, 24, 1 | 0x20, 31 | 0x60),
            (0, 0, 4, 28),
        ]
        for w_a, o_a, w_b, o_b in bfm_pairs:
            want_a = ref_s_bfm_b32(w_a, o_a)
            want_b = ref_s_bfm_b32(w_b, o_b)
            mem = {0x1000: w_a, 0x1004: o_a, 0x1008: w_b, 0x100C: o_b}
            with self.subTest(w_a=hex(w_a), o_a=hex(o_a), w_b=hex(w_b), o_b=hex(o_b)):
                got = tuple(mrt0_words(spv_bfm, push_constants=PUSH_VSHARP, memory=mem))
                self.assertEqual(got, (want_a, want_b, want_a, 1))

        # 2. End-to-end translated scalar bitfield-insert composition + roundtrip s_bfe_u32 extract.
        spv_insert = translate_ps_bitfield([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",  # s4=base, s5=insert, s6=count, s7=offset
            "s_waitcnt lgkmcnt(0)",
            "s_bfm_b32 s8, s6, s7",                       # s8 = mask = ((1 << count) - 1) << offset
            "s_lshl_b32 s9, s5, s7",                      # s9 = insert << offset
            "s_and_b32 s9, s9, s8",                       # s9 = (insert << offset) & mask
            "s_andn2_b32 s10, s4, s8",                    # s10 = base & ~mask
            "s_or_b32 s11, s10, s9",                      # s11 = inserted result
            "s_lshl_b32 s12, s6, 16",
            "s_or_b32 s12, s12, s7",                      # s12 = pack_bfe_ctl(offset, count)
            "s_bfe_u32 s13, s11, s12",                    # roundtrip extract
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s11",
            "v_mov_b32 v2, s13",
            "v_mov_b32 v3, 0",
        ])
        for base_val, ins_val, count, offset in (
            (0xAA55AA55, 0x3F, 6, 12),
            (0x12345678, 0xAB, 8, 8),
            (0xFFFFFFFF, 0x00, 16, 8),
            (0x80000001, 0x7FFF, 15, 16),
        ):
            want_mask = ref_s_bfm_b32(count, offset)
            want_inserted = ref_bitfield_insert(base_val, ins_val, offset, count)
            want_roundtrip, _ = ref_s_bfe_u32(want_inserted, pack_bfe_ctl(offset, count))
            mem = {0x1000: base_val, 0x1004: ins_val, 0x1008: count, 0x100C: offset}
            with self.subTest(base=hex(base_val), ins=hex(ins_val), count=count, offset=offset):
                got = tuple(mrt0_words(spv_insert, push_constants=PUSH_VSHARP, memory=mem)[:3])
                self.assertEqual(got, (want_mask, want_inserted, want_roundtrip))


class NegativeControlTests(unittest.TestCase):
    """Sensitivity tests ensuring mutations in bitfield logic and translated SPIR-V are detected."""

    def test_signed_vs_unsigned_mutation_caught(self) -> None:
        # Extracting 0x00008000 with offset 8, width 8 gives 0x80.
        # Signed gives 0xFFFFFF80, unsigned gives 0x00000080.
        ctl = pack_bfe_ctl(8, 8)
        signed_res, _ = ref_s_bfe_i32(0x00008000, ctl)
        unsigned_res, _ = ref_s_bfe_u32(0x00008000, ctl)
        self.assertEqual(signed_res, 0xFFFFFF80)
        self.assertEqual(unsigned_res, 0x00000080)
        self.assertNotEqual(signed_res, unsigned_res)

    def test_dropped_offset_mutation_caught(self) -> None:
        ctl_good = pack_bfe_ctl(4, 8)
        ctl_mutated = pack_bfe_ctl(0, 8)
        val = 0x12345678
        res_good, _ = ref_s_bfe_u32(val, ctl_good)
        res_mutated, _ = ref_s_bfe_u32(val, ctl_mutated)
        self.assertEqual(res_good, 0x67)
        self.assertEqual(res_mutated, 0x78)
        self.assertNotEqual(res_good, res_mutated)

    def test_scc_zero_flag_mutation_caught(self) -> None:
        # An implementation that fails to clear SCC when the result is 0
        ctl = pack_bfe_ctl(0, 4)
        res_nonzero, scc_nonzero = ref_s_bfe_u32(0x5, ctl)
        res_zero, scc_zero = ref_s_bfe_u32(0x0, ctl)
        self.assertEqual(scc_nonzero, 1)
        self.assertEqual(scc_zero, 0)
        self.assertNotEqual(scc_nonzero, scc_zero)

    def test_translated_spirv_mutations_caught(self) -> None:
        # 1. s_bfe_i32: replacing OpBitFieldSExtract with OpBitFieldUExtract breaks sign extension,
        #    and dropping the SCC store breaks SCC transitions.
        spv_bfe = translate_ps_bitfield([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",
            "s_bfe_i32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        mem_bfe = {0x1000: 0x00008000, 0x1004: pack_bfe_ctl(8, 8)}
        self.assertEqual(
            tuple(mrt0_words(spv_bfe, push_constants=PUSH_VSHARP, memory=mem_bfe)[:2]),
            (0xFFFFFF80, 1),
        )
        bad_sextract = spv_bfe.replace("OpBitFieldSExtract %int", "OpBitFieldUExtract %int", 1)
        self.assertNotEqual(bad_sextract, spv_bfe)
        self.assertNotEqual(
            mrt0_words(bad_sextract, push_constants=PUSH_VSHARP, memory=mem_bfe)[0],
            0xFFFFFF80,
        )
        bad_scc = spv_bfe.replace("OpStore %scc", "OpStore %vcc_hi")
        self.assertNotEqual(bad_scc, spv_bfe)
        self.assertNotEqual(
            mrt0_words(bad_scc, push_constants=PUSH_VSHARP, memory=mem_bfe)[1],
            1,
        )

        # 2. s_bfm_b32: sabotaging OpISub (1 << w) - 1 breaks the mask.
        spv_bfm = translate_ps_bitfield([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_bfm_b32 s6, s4, s5",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        mem_bfm = {0x1000: 8, 0x1004: 8}
        self.assertEqual(mrt0_words(spv_bfm, push_constants=PUSH_VSHARP, memory=mem_bfm)[0], 0x0000FF00)
        bad_bfm = spv_bfm.replace("OpISub %uint", "OpIAdd %uint", 1)
        self.assertNotEqual(bad_bfm, spv_bfm)
        self.assertNotEqual(mrt0_words(bad_bfm, push_constants=PUSH_VSHARP, memory=mem_bfm)[0], 0x0000FF00)


if __name__ == "__main__":
    unittest.main()
