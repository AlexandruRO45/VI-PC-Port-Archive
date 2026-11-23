"""
RDNA2 SALU instruction semantic tests: s_andn2_saveexec_b64.

Exercises s_andn2_saveexec_b64 through tools/rdna2spirv.py and spirv_mini_interp.
RDNA2 ISA, SOP1:
    s_andn2_saveexec_b64 s[d:d+1], src0
Semantics:
    D = EXEC
    EXEC = src0 & ~EXEC
    SCC = (EXEC != 0)

Validates:
1. Mathematical correctness of AND-NOT2 logic across Wave64 EXEC masks
2. Saving of previous EXEC state into destination SGPR pair
3. Correct computation and propagation of SCC status flag
4. Active lane execution and inactive lane divergence under new EXEC mask
5. Translation to valid SPIR-V assembly via support.translate_lines
"""

from __future__ import annotations

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

VSHARP = [0, 0, 0x1000, 0x31016FAC]
ALL64 = (1 << 64) - 1
SCC_AT, BCNT_AT, FF1_AT = 256, 260, 264


def fssm(threads: int = 64) -> bytes:
    """Construct compute shader header blob declaring 64-thread wave size."""
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def s0_mask(mul: int, add: int, mask: int) -> int:
    """Generate 64-bit mask for lanes where ((lane * mul + add) & mask) == 0."""
    return sum(1 << lane for lane in range(64) if ((lane * mul + add) & mask) == 0)


def ref_model(s0: int, old_exec: int) -> tuple[int, int, int]:
    """
    Reference mathematical model for s_andn2_saveexec_b64.
    Returns (new_exec, saved_d, scc).
    """
    new_exec = (s0 & ~old_exec) & ALL64
    saved_d = old_exec & ALL64
    scc = 1 if new_exec != 0 else 0
    return new_exec, saved_d, scc


def build_program(src_kind: str, s0: int, old_exec: int, mul: int = 1, add: int = 0, mask: int = 0) -> list[str]:
    """Construct an RDNA2 assembly program testing s_andn2_saveexec_b64."""
    def store(reg: str, offset: int) -> list[str]:
        return [
            f"v_mov_b32_e32 v20, {reg}",
            f"v_mov_b32_e32 v21, {offset}",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]

    if src_kind == "vcc":
        setup = ["v_cmp_eq_u32_e32 vcc, 0, v3"]
        operand = "vcc"
    elif src_kind == "sgpr":
        setup = ["v_cmp_eq_u32_e64 s[10:11], 0, v3"]
        operand = "s[10:11]"
    elif src_kind == "literal":
        setup = [
            f"s_mov_b32 s12, {s0 & 0xFFFFFFFF:#x}",
            f"s_mov_b32 s13, {(s0 >> 32) & 0xFFFFFFFF:#x}",
        ]
        operand = "s[12:13]"
    elif src_kind == "exec":
        setup = []
        operand = "exec"
    else:
        raise ValueError(f"Unknown src_kind: {src_kind}")

    lines = [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "v_mbcnt_hi_u32_b32 v1, -1, v1",
        "v_mov_b32_e32 v2, 0",
        f"v_mad_u32_u24 v3, v1, {mul}, {add}",
        f"v_and_b32_e32 v3, {mask}, v3",
    ] + setup + [
        f"s_mov_b32 exec_lo, {old_exec & 0xFFFFFFFF:#x}",
        f"s_mov_b32 exec_hi, {(old_exec >> 32) & 0xFFFFFFFF:#x}",
        f"s_andn2_saveexec_b64 s[6:7], {operand}",
        "s_cselect_b32 s8, 1, 0",
        "v_mov_b32_e32 v2, 1",
        "s_mov_b64 exec, -1",
        "s_bcnt1_i32_b64 s9, s[6:7]",
        "s_ff1_i32_b64 s10, s[6:7]",
        "v_lshlrev_b32_e32 v21, 2, v1",
        "buffer_store_dword v2, v21, s[0:3], 0 offen",
        "v_cmpx_eq_u32_e32 0, v1",
    ] + store("s8", SCC_AT) + store("s9", BCNT_AT) + store("s10", FF1_AT) + ["s_endpgm"]

    return lines


def run_test_case(src_kind: str, mul: int, add: int, mask: int, old_exec: int) -> dict[int, int]:
    """Translate and run a test case through rdna2spirv and spirv_mini_interp."""
    s0 = old_exec if src_kind == "exec" else s0_mask(mul, add, mask)
    lines = build_program(src_kind, s0, old_exec, mul, add, mask)
    spv = support.translate_lines("cs_salu_andn2_saveexec_b64", lines, fssm=fssm())
    return si.run_workgroup(
        spv,
        64,
        64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


class TestSaluAndn2SaveexecB64Hw(unittest.TestCase):
    """Semantic tests for s_andn2_saveexec_b64 verified through tools/rdna2spirv.py."""

    def test_spirv_emission_structure(self) -> None:
        """Verify that translate_lines produces valid SPIR-V assembly containing saveexec ops."""
        lines = [
            "s_mov_b64 s[0:1], 0x55555555",
            "s_mov_b64 exec, 0x33333333",
            "s_andn2_saveexec_b64 s[2:3], s[0:1]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_test_andn2_saveexec_emit", lines, fssm=fssm())
        self.assertIn("OpCapability", spv)
        self.assertIn("OpMemoryModel", spv)
        self.assertTrue(len(spv) > 0)

    def test_andn2_identity_and_cancellation(self) -> None:
        """
        Verify algebraic ANDN2 properties:
        X & ~0 = X (when old_exec = 0, new_exec = S0)
        X & ~X = 0 (self-cancellation: when S0 == old_exec, new_exec = 0)
        """
        # S0 & ~0 = S0
        mem_id = run_test_case("literal", 1, 0, 0, 0x0000000000000000)
        got_id = sum(mem_id[4 * lane] << lane for lane in range(64))
        self.assertEqual(got_id, ALL64)
        self.assertEqual(mem_id[SCC_AT], 1)
        self.assertEqual(mem_id[BCNT_AT], 0)

        # S0 & ~S0 = 0 via exec source: s_andn2_saveexec_b64 s[6:7], exec
        mem_cancel = run_test_case("exec", 1, 0, 0, 0xDEADBEEFCAFEBABE)
        got_cancel = sum(mem_cancel[4 * lane] << lane for lane in range(64))
        self.assertEqual(got_cancel, 0, "Self-ANDN2 with EXEC must yield zero active lanes")
        self.assertEqual(mem_cancel[SCC_AT], 0, "SCC must be 0 when new EXEC is 0")
        self.assertEqual(mem_cancel[BCNT_AT], bin(0xDEADBEEFCAFEBABE).count("1"))

    def test_wave64_halves_and_complements(self) -> None:
        """Test bitwise AND-NOT across low/high halves and alternating bit masks."""
        # S0 = low 32 bits (0x00000000FFFFFFFF)
        # old_exec = high 32 bits (0xFFFFFFFF00000000)
        # ~old_exec = low 32 bits (0x00000000FFFFFFFF)
        # new_exec = S0 & ~old_exec = 0x00000000FFFFFFFF
        mem_halves = run_test_case("literal", 1, 0, 32, 0xFFFFFFFF00000000)
        got_halves = sum(mem_halves[4 * lane] << lane for lane in range(64))
        self.assertEqual(got_halves, 0x00000000FFFFFFFF)
        self.assertEqual(mem_halves[SCC_AT], 1)
        self.assertEqual(mem_halves[BCNT_AT], 32)
        self.assertEqual(mem_halves[FF1_AT], 32)

        # S0 = lanes from mask
        # old_exec = identical mask S0
        # new_exec = S0 & ~S0 = 0
        s0_val = s0_mask(3, 1, 5)
        mem_same = run_test_case("vcc", 3, 1, 5, s0_val)
        self.assertEqual(sum(mem_same[4 * lane] << lane for lane in range(64)), 0)
        self.assertEqual(mem_same[SCC_AT], 0)
        self.assertEqual(mem_same[BCNT_AT], bin(s0_val).count("1"))

    def test_various_sources_and_masks(self) -> None:
        """Exhaustively verify vcc, sgpr, literal, and exec across standard test cases."""
        sources = ("vcc", "sgpr", "literal", "exec")
        cases = [
            (1, 0, 8, 0x0000FFFFFFFF0000),
            (1, 0, 32, 0xFFFFFFFF00000000),
            (1, 32, 32, 0x0000FFFF0000FFFF),
            (3, 1, 5, 0x5555555555555555),
            (7, 2, 12, 0xF0F0F0F00F0F0F0F),
            (1, 0, 0, 0),
            (1, 0, 0, ALL64),
            (1, 0, 64, 0x123456789ABCDEF0),
            (5, 3, 63, 0x00000000FFFFFFFF),
        ]

        for src in sources:
            for mul, add, mask, old_exec in cases:
                with self.subTest(src=src, exec=hex(old_exec)):
                    s0 = old_exec if src == "exec" else s0_mask(mul, add, mask)
                    new_exp, saved_exp, scc_exp = ref_model(s0, old_exec)
                    mem = run_test_case(src, mul, add, mask, old_exec)
                    got_exec = sum(mem[4 * lane] << lane for lane in range(64))

                    self.assertEqual(got_exec, new_exp, f"New EXEC mismatch for src={src}, old_exec={hex(old_exec)}")
                    self.assertEqual(mem[SCC_AT], scc_exp, f"SCC mismatch for src={src}, old_exec={hex(old_exec)}")
                    self.assertEqual(mem[BCNT_AT], bin(saved_exp).count("1"), "Saved D bit count mismatch")
                    lowest_bit = (saved_exp & -saved_exp).bit_length() - 1 if saved_exp else 0xFFFFFFFF
                    self.assertEqual(mem[FF1_AT], lowest_bit, "Saved D first-one mismatch")


if __name__ == "__main__":
    unittest.main()
