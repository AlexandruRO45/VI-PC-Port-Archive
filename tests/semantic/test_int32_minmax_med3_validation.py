"""Validation and positive-control tests for 32-bit integer min3/max3/med3 VOP3 ops.

Covers:
  - v_min3_i32, v_min3_u32
  - v_max3_i32, v_max3_u32
  - v_med3_i32, v_med3_u32

Verifies that unsupported modifiers (clamp, mul:2, div:2, op_sel, neg_lo, unknown),
DPP/SDWA encodings, operand modifiers (neg, abs, sext on dst or src), invalid
destination kinds, multi-register destinations or sources, and wrong operand counts
all raise rdna2spirv.Unsupported.

Positive controls verify valid plain, _e64, plain negative integer immediate, and
mixed SGPR/VGPR forms through spirv-as, spirv-val --target-env vulkan1.2, and
spirv_mini_interp.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OPS = (
    "v_min3_i32",
    "v_min3_u32",
    "v_max3_i32",
    "v_max3_u32",
    "v_med3_i32",
    "v_med3_u32",
)


def _s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def _ref(op: str, a: int, b: int, c: int) -> int:
    key = _s32 if op.endswith("_i32") else (lambda v: v & MASK32)
    ordered = sorted((a & MASK32, b & MASK32, c & MASK32), key=key)
    if op.startswith("v_min3_"):
        return ordered[0]
    if op.startswith("v_max3_"):
        return ordered[2]
    return ordered[1]


def _validate_spirv(spvasm: str) -> None:
    sa = shutil.which("spirv-as")
    sv = shutil.which("spirv-val")
    if not (sa and sv):
        if os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS"):
            raise AssertionError("spirv-as / spirv-val required but not found on PATH")
        return
    with tempfile.TemporaryDirectory() as td:
        asm_p = os.path.join(td, "mod.spvasm")
        spv_p = os.path.join(td, "mod.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([sa, "--target-env", "vulkan1.2", asm_p, "-o", spv_p], check=True)
        subprocess.run([sv, "--target-env", "vulkan1.2", spv_p], check=True)


class Int32MinMaxMed3ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._td = tempfile.TemporaryDirectory()
        disasm_dir = os.path.join(cls._td.name, "disasm")
        os.mkdir(disasm_dir)
        cls._src_path = os.path.join(disasm_dir, "ps_int32_minmax_med3_val.s")
        with open(os.path.join(cls._td.name, "ps_int32_minmax_med3_val.dxbc"), "wb") as f:
            f.write(STUB_DXBC)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._td.cleanup()

    def _translate_lines(self, lines: list[str]) -> str:
        asm = (
            "; ps_int32_minmax_med3_val\n"
            + "\n".join(f"\t{ln}" for ln in lines)
            + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        )
        with open(self._src_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(self._src_path)

    def _assert_unsupported(self, lines: list[str]) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            self._translate_lines(lines)

    def test_reject_modifiers(self) -> None:
        bad_mods = (
            "clamp",
            "mul:2",
            "div:2",
            "op_sel:[0,0,0]",
            "op_sel:[1,0,0,1]",
            "neg_lo:[1,0,0]",
            "unknown_mod:1",
        )
        for op in OPS:
            for mod in bad_mods:
                with self.subTest(op=op, mod=mod):
                    self._assert_unsupported([
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        f"{op} v0, v1, v2, v3 {mod}",
                    ])

    def test_reject_injected_unknown_mod_in_ir(self) -> None:
        for op in OPS:
            with self.subTest(op=op):
                gen = rdna2spirv.Gen("ps")
                ins = {
                    "op": op,
                    "args": [
                        {"k": "vgpr", "r": 0, "n": 1},
                        {"k": "vgpr", "r": 1, "n": 1},
                        {"k": "vgpr", "r": 2, "n": 1},
                        {"k": "vgpr", "r": 3, "n": 1},
                    ],
                    "mods": ["bogus_mod"],
                    "enc": None,
                }
                with self.assertRaises(rdna2spirv.Unsupported):
                    gen.insn(ins)

    def test_reject_dpp_and_sdwa(self) -> None:
        forms = (
            ("_dpp", "v0, v1, v2, v3 quad_perm:[1,0,3,2]"),
            ("_dpp", "v0, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]"),
            ("_sdwa", "v0, v1, v2, v3"),
            ("_sdwa", "v0, v1, v2, v3 src0_sel:WORD_0"),
        )
        for op in OPS:
            for suffix, operands in forms:
                with self.subTest(op=op + suffix, operands=operands):
                    self._assert_unsupported([
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        f"{op}{suffix} {operands}",
                    ])

    def test_reject_operand_modifiers_neg_abs_sext(self) -> None:
        operand_cases = (
            "-v0, v1, v2, v3",
            "|v0|, v1, v2, v3",
            "sext(v0), v1, v2, v3",
            "v0, -v1, v2, v3",
            "v0, v1, neg(v2), v3",
            "v0, v1, v2, neg(1)",
            "v0, |v1|, v2, v3",
            "v0, v1, abs(v2), v3",
            "v0, v1, v2, |1|",
            "v0, sext(v1), v2, v3",
            "v0, v1, sext(v2), v3",
            "v0, v1, v2, sext(s4)",
        )
        for op in OPS:
            for operands in operand_cases:
                with self.subTest(op=op, operands=operands):
                    self._assert_unsupported([
                        "s_mov_b32 s4, 4",
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        f"{op} {operands}",
                    ])

    def test_reject_bad_destination_and_multi_register_operands(self) -> None:
        operand_cases = (
            "s0, v1, v2, v3",
            "vcc_lo, v1, v2, v3",
            "null, v1, v2, v3",
            "v[0:1], v1, v2, v3",
            "v0, v[1:2], v2, v3",
            "v0, v1, v[2:3], v3",
            "v0, v1, v2, v[3:4]",
            "v0, s[4:5], v2, v3",
        )
        for op in OPS:
            for operands in operand_cases:
                with self.subTest(op=op, operands=operands):
                    self._assert_unsupported([
                        "s_mov_b32 s4, 4",
                        "s_mov_b32 s5, 5",
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        "v_mov_b32 v4, 4",
                        f"{op} {operands}",
                    ])

    def test_reject_wrong_operand_counts(self) -> None:
        counts = (
            "",
            "v0, v1",
            "v0, v1, v2",
            "v0, v1, v2, v3, v4",
        )
        for op in OPS:
            for operands in counts:
                stmt = f"{op} {operands}".rstrip()
                with self.subTest(stmt=stmt):
                    self._assert_unsupported([
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        "v_mov_b32 v4, 4",
                        stmt,
                    ])

    def test_positive_controls_validate_and_execute(self) -> None:
        # Test plain, _e64, plain negative integer immediates, and SGPR/VGPR sources across all 6 ops.
        cases = (
            ("plain_vgpr", "", ("v1", "v2", "v3"), (0x80000000, 5, 0xFFFFFFFF)),
            ("e64_vgpr", "_e64", ("v1", "v2", "v3"), (0x7FFFFFFF, 0x80000000, 0)),
            ("neg_imm", "", ("v1", "-16", "10"), (0x7FFFFFFF, -16 & MASK32, 10)),
            ("e64_sgpr_neg_imm", "_e64", ("s4", "v2", "-1"), (0x00000007, 0x80000000, -1 & MASK32)),
        )
        for op in OPS:
            for label, suffix, ops, vals in cases:
                with self.subTest(op=op, case=label):
                    a, b, c = vals
                    lines = [
                        f"s_mov_b32 s4, {a:#x}",
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        f"{op}{suffix} v0, {ops[0]}, {ops[1]}, {ops[2]}",
                    ]
                    spvasm = self._translate_lines(lines)
                    _validate_spirv(spvasm)
                    got = si.run_fragment(spvasm)["%out_mrt0"][0]
                    want = _ref(op, a, b, c)
                    self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
