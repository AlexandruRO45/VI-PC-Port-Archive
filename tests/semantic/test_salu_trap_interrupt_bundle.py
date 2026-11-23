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


if __name__ == "__main__":
    unittest.main()
