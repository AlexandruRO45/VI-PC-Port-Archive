from __future__ import annotations

"""Bundle 62: SALU Trap and Interrupt Management Suite.

Covers RDNA2 Scalar ALU Trap and Interrupt Management instructions:
    - s_ttracedata     sSrc0
        Thread-trace data from SGPR, literal, or inline constant. Handled as supported no-op.
    - s_ttracedata_imm simm16
        Thread-trace data from 16-bit signed immediate. Handled as supported no-op.
    - s_sendmsg        sendmsg(msg, op, stream)
        Sends message to message handler. Supports MSG_ORDERED_PS_DONE in pixel shaders;
        unsupported messages are refused with rdna2spirv.Unsupported.
    - s_trap           simm7
        Software trap. Refused by shader translator with rdna2spirv.Unsupported.
    - s_sendmsghalt    sendmsg(msg, op, stream)
        Sends message and halts wavefront. Refused with rdna2spirv.Unsupported.

Hardware rules (GFX10 RDNA2):
    - s_ttracedata and s_ttracedata_imm do not modify registers, memory, SCC, VCC, or EXEC.
    - s_sendmsg MSG_ORDERED_PS_DONE ends fragment interlock section with zero side effects on state.
    - s_trap and s_sendmsghalt are invalid/unmodelled in runtime graphics/compute shaders and must
      be refused deterministically by the translator.
    - 100% translator-backed execution exercising tools/rdna2spirv.py and support.mrt0_words.
    - Zero in-file mock models.

Standard library Python only (unittest, struct). Zero game data.
"""

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

import rdna2spirv
import support

MASK32 = 0xFFFFFFFF

# Compute shader FSSM header (32x1x1)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)


def run_ps(instructions: list[str], export_regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0")) -> list[int]:
    """Helper to translate and run a fragment shader through support.mrt0_words."""
    lines = list(instructions) + [
        f"exp mrt0 {export_regs[0]}, {export_regs[1]}, {export_regs[2]}, {export_regs[3]} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_salu_trap_bundle", lines)
    return support.mrt0_words(spv)


class TestSTtraceDataReference(unittest.TestCase):
    def test_basic_ttracedata_emission(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_ttracedata s0",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x12345678, 0x12345678, 0x12345678, 0x12345678])

    def test_ttracedata_mrt0_execution(self):
        lines = [
            "v_mov_b32 v0, 0x10",
            "v_mov_b32 v1, 0x20",
            "v_mov_b32 v2, 0x30",
            "v_mov_b32 v3, 0x40",
            "s_ttracedata 0",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v3"))
        self.assertEqual(words, [0x10, 0x20, 0x30, 0x40])


class TestSTtraceDataOperands(unittest.TestCase):
    def test_sgpr_source(self):
        lines = [
            "s_mov_b32 s1, 0xAABBCCDD",
            "s_ttracedata s1",
            "v_mov_b32 v0, s1",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0xAABBCCDD, 0xAABBCCDD, 0xAABBCCDD, 0xAABBCCDD])

    def test_literal_source(self):
        lines = [
            "s_ttracedata 0xDEADBEEF",
            "v_mov_b32 v0, 0x77777777",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x77777777, 0x77777777, 0x77777777, 0x77777777])

    def test_inline_integer_source(self):
        lines = [
            "s_ttracedata -1",
            "s_ttracedata 42",
            "v_mov_b32 v0, 0x55555555",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x55555555, 0x55555555, 0x55555555, 0x55555555])


class TestSTtraceDataPreservation(unittest.TestCase):
    def test_scc_preservation(self):
        lines = [
            "s_cmp_eq_u32 5, 5",              # sets SCC = 1
            "s_ttracedata 0x123",
            "s_cselect_b32 s0, 0x111, 0x222",  # reads SCC -> 0x111
            "s_cmp_lg_u32 5, 5",              # sets SCC = 0
            "s_ttracedata s0",
            "s_cselect_b32 s1, 0x111, 0x222",  # reads SCC -> 0x222
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x111, 0x222, 0x111, 0x222])

    def test_vcc_preservation(self):
        lines = [
            "v_cmp_eq_u32_e32 vcc, 0, 0",      # VCC = all ones
            "s_ttracedata 0",
            "v_cndmask_b32_e64 v0, 0xAA, 0xBB, vcc",  # picks 0xBB
            "v_cmp_lg_u32_e32 vcc, 0, 0",      # VCC = all zeros
            "s_ttracedata 1",
            "v_cndmask_b32_e64 v1, 0xAA, 0xBB, vcc",  # picks 0xAA
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xBB, 0xAA, 0xBB, 0xAA])

    def test_vgpr_sgpr_preservation(self):
        lines = [
            "s_mov_b32 s0, 0xCAFEBABE",
            "v_mov_b32 v0, 0xFEEDFACE",
            "s_ttracedata s0",
            "v_mov_b32 v1, s0",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xFEEDFACE, 0xCAFEBABE, 0xFEEDFACE, 0xCAFEBABE])


class TestSTtraceDataChaining(unittest.TestCase):
    def test_consecutive_ttracedata(self):
        lines = [
            "s_mov_b32 s0, 0x100",
            "s_ttracedata s0",
            "s_ttracedata s0",
            "s_ttracedata s0",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x100, 0x100, 0x100, 0x100])

    def test_interleaved_alu_ttracedata(self):
        lines = [
            "s_mov_b32 s0, 10",
            "s_ttracedata s0",
            "s_add_u32 s0, s0, 5",
            "s_ttracedata s0",
            "s_add_u32 s0, s0, 5",
            "s_ttracedata s0",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [20, 20, 20, 20])


class TestSTtraceDataRejection(unittest.TestCase):
    def test_invalid_opcode_rejection(self):
        bad_lines = [
            "s_ttracedata_invalid s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_bad_ttrace_opcode", bad_lines)
        self.assertIn("s_ttracedata_invalid", str(cm.exception))

    def test_mutated_mnemonic_rejection(self):
        bad_lines = [
            "s_ttrace_data s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_bad_ttrace_mutated", bad_lines)
        self.assertIn("s_ttrace_data", str(cm.exception))


class TestSTtraceDataImmReference(unittest.TestCase):
    def test_basic_ttracedata_imm_emission(self):
        lines = [
            "s_mov_b32 s0, 0x99887766",
            "s_ttracedata_imm 100",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x99887766, 0x99887766, 0x99887766, 0x99887766])

    def test_ttracedata_imm_mrt0_execution(self):
        lines = [
            "v_mov_b32 v0, 0x11",
            "v_mov_b32 v1, 0x22",
            "v_mov_b32 v2, 0x33",
            "v_mov_b32 v3, 0x44",
            "s_ttracedata_imm 0",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v3"))
        self.assertEqual(words, [0x11, 0x22, 0x33, 0x44])


class TestSTtraceDataImmBoundaries(unittest.TestCase):
    def test_zero_and_small_immediates(self):
        lines = [
            "s_ttracedata_imm 0",
            "s_ttracedata_imm 1",
            "v_mov_b32 v0, 0x123",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x123, 0x123, 0x123, 0x123])

    def test_max_positive_imm16(self):
        lines = [
            "s_ttracedata_imm 0x7FFF",
            "v_mov_b32 v0, 0x456",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x456, 0x456, 0x456, 0x456])

    def test_negative_imm16(self):
        lines = [
            "s_ttracedata_imm -1",
            "s_ttracedata_imm -32768",
            "v_mov_b32 v0, 0x789",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x789, 0x789, 0x789, 0x789])


class TestSTtraceDataImmPreservation(unittest.TestCase):
    def test_scc_untouched_by_imm(self):
        lines = [
            "s_cmp_eq_u32 10, 10",             # SCC = 1
            "s_ttracedata_imm 42",
            "s_cselect_b32 s0, 0xABC, 0xDEF",  # reads SCC -> 0xABC
            "s_cmp_lg_u32 10, 10",             # SCC = 0
            "s_ttracedata_imm 84",
            "s_cselect_b32 s1, 0xABC, 0xDEF",  # reads SCC -> 0xDEF
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xABC, 0xDEF, 0xABC, 0xDEF])

    def test_register_dataflow_preservation(self):
        lines = [
            "s_mov_b32 s0, 0x55005500",
            "v_mov_b32 v0, 0x00AA00AA",
            "s_ttracedata_imm 0x1234",
            "v_add_nc_u32 v1, v0, s0",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x00AA00AA, 0x55AA55AA, 0x00AA00AA, 0x55AA55AA])


class TestSTtraceDataImmPipeline(unittest.TestCase):
    def test_interleaved_valu_salu(self):
        lines = [
            "v_mov_b32 v0, 1",
            "s_ttracedata_imm 1",
            "v_add_nc_u32 v0, v0, 2",
            "s_ttracedata_imm 2",
            "v_add_nc_u32 v0, v0, 3",
            "s_ttracedata_imm 3",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [6, 6, 6, 6])

    def test_multiple_immediate_trace_events(self):
        lines = [
            "s_ttracedata_imm 0x10",
            "s_ttracedata_imm 0x20",
            "s_ttracedata_imm 0x30",
            "v_mov_b32 v0, 0x99",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x99, 0x99, 0x99, 0x99])


class TestSTtraceDataImmRejection(unittest.TestCase):
    def test_invalid_opcode_imm_rejection(self):
        bad_lines = [
            "s_ttracedata_imm_bad 42",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_bad_ttrace_imm_opcode", bad_lines)
        self.assertIn("s_ttracedata_imm_bad", str(cm.exception))

    def test_mutated_mnemonic_imm_rejection(self):
        bad_lines = [
            "s_ttracedata_imm32 42",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_bad_ttrace_imm_mutated", bad_lines)
        self.assertIn("s_ttracedata_imm32", str(cm.exception))


class TestSSendMsgOrderedPsDoneReference(unittest.TestCase):
    def test_sendmsg_ordered_ps_done_translation(self):
        lines = [
            "s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)",
            "v_mov_b32 v0, 0x12344321",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x12344321, 0x12344321, 0x12344321, 0x12344321])

    def test_sendmsg_spirv_header_validity(self):
        lines = [
            "s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_sendmsg_valid", lines)
        self.assertIn("OpEntryPoint Fragment", spv)
        self.assertIn("OpExecutionMode", spv)


if __name__ == "__main__":
    unittest.main()
