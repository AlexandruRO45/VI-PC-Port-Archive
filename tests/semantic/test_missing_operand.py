"""An instruction line with fewer operands than its handler reads is Unsupported, not a translator crash.

tools/rdna2spirv.py hands each handler the printed operand list and the handler indexes it directly, so before this
check `v_add_f32 v0, v1` stopped with a bare `IndexError: list index out of range` and `v_fma_f32 v0, v1, v2` with
`ValueError: not enough values to unpack`. main() counted both as "translator error IndexError/ValueError", the
bucket for translator bugs. Gen._insn now reads the operands through rdna2spirv.Operands, which raises
MissingOperand (an Unsupported and an IndexError) naming the mnemonic and the operand index. #8177 rejects an
empty VALU operand list before Gen runs; these cases have some operands, or are scalar.

Hand-written listings only; standard library only; no GPU and no game files.
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, rdna2spirv, translate_lines  # noqa: E402

# (line, mnemonic in the message, index of the first operand the handler reads that is not there)
SHORT_LINES = (
    ("v_add_f32 v0, v1", "v_add_f32", 2),               # VOP2, src1 missing
    ("v_cndmask_b32 v0, v1", "v_cndmask_b32", 2),       # VOP2 with an implicit VCC
    ("v_perm_b32 v0, v1, v2", "v_perm_b32", 3),         # VOP3, selector missing
    ("v_mov_b32_sdwa", "v_mov_b32_sdwa", 0),            # the SDWA path copies the operands first
    ("s_mov_b32", "s_mov_b32", 0),                      # SALU, nothing at all
    ("s_add_u32 s0, s1", "s_add_u32", 2),               # SOP2, src1 missing
)

# Handlers that unpack a slice of sources: `x, y, z = (... for t in a[1:4])` fails with a ValueError.
SHORT_UNPACKS = (
    ("v_fma_f32 v0, v1, v2", "v_fma_f32", 3),
    ("v_mad_u32_u24 v0, v1, v2", "v_mad_u32_u24", 3),
)

# The same instructions with every operand: they must still translate.
COMPLETE_LINES = (
    "v_add_f32 v0, v1, v2",
    "v_cndmask_b32 v0, v1, v2, vcc_lo",
    "v_perm_b32 v0, v1, v2, v3",
    "s_mov_b32 s0, s1",
    "s_add_u32 s0, s1, s2",
    "v_fma_f32 v0, v1, v2, v3",
    "v_mad_u32_u24 v0, v1, v2, v3",
)


class MissingOperandTests(unittest.TestCase):
    def assert_missing(self, line, mnemonic, index):
        with self.assertRaises(rdna2spirv.MissingOperand) as ctx:
            translate_lines("ps_missing_operand", [line])
        self.assertEqual(str(ctx.exception), f"{mnemonic}: missing operand {index}")
        self.assertIsInstance(ctx.exception, rdna2spirv.Unsupported)     # counted as a reason by main()
        self.assertIsInstance(ctx.exception, IndexError)                 # old `except IndexError` still applies

    def test_short_operand_list_is_unsupported(self):
        for line, mnemonic, index in SHORT_LINES:
            with self.subTest(line=line):
                self.assert_missing(line, mnemonic, index)

    def test_short_source_unpack_is_unsupported(self):
        for line, mnemonic, index in SHORT_UNPACKS:
            with self.subTest(line=line):
                self.assert_missing(line, mnemonic, index)

    def test_complete_lines_still_translate(self):
        for line in COMPLETE_LINES:
            with self.subTest(line=line):
                self.assertIn("OpEntryPoint", translate_lines("ps_complete_operands", [line]))

    def test_operandless_instruction_still_translates(self):
        self.assertIn("OpEntryPoint", translate_lines("ps_nop", ["v_nop", "s_nop 0"]))

    def test_unrelated_value_error_is_not_renamed(self):
        # Only an unpacking error after a slice ran past the operands is a missing operand. Any other ValueError
        # from a handler is a translator bug and must keep its type.
        for message in ("not enough values to unpack (expected 3, got 2)", "synthetic handler bug"):
            with self.subTest(message=message), \
                    mock.patch.object(rdna2spirv.Gen, "_insn_op", side_effect=ValueError(message)):
                with self.assertRaises(ValueError) as ctx:
                    translate_lines("ps_value_error", ["v_add_f32 v0, v1, v2"])
                self.assertNotIsInstance(ctx.exception, rdna2spirv.Unsupported)

    def test_optional_slice_past_the_end_is_a_short_list(self):
        # Handlers slice optional operands on purpose (`a[3:4]`): that must stay an empty list, not an error.
        ops = rdna2spirv.Operands("v_add_f32", [{"k": "vgpr", "r": 0}])
        self.assertEqual(ops[1:4], [])
        self.assertEqual(ops.short, 4)
        self.assertEqual(ops[:1], [{"k": "vgpr", "r": 0}])
        with self.assertRaises(rdna2spirv.MissingOperand):
            ops[-2]

    def test_main_reports_the_reason_not_a_translator_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm, out = os.path.join(tmp, "disasm"), os.path.join(tmp, "out")
            os.mkdir(disasm)
            with open(os.path.join(disasm, "ps_short.s"), "w", encoding="utf-8", newline="\n") as f:
                f.write("; ps_short  hand-written test program\n\tv_add_f32 v0, v1\n\ts_endpgm\n")
            with open(os.path.join(tmp, "ps_short.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            output = io.StringIO()
            argv = ["rdna2spirv.py", disasm, "--out", out, "--assembler", "python"]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
                rdna2spirv.main()
        text = output.getvalue()
        self.assertIn("v_add_f32: missing operand 2", text)
        self.assertNotIn("translator error", text)


if __name__ == "__main__":
    unittest.main()
