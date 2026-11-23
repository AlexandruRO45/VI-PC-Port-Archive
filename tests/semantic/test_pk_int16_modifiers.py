"""Regression tests for VOP3P modifier parsing/validation on packed 16-bit integer ops.

Covers all 11 packed 16-bit integer VOP3P ops:
  2-source: v_pk_add_u16, v_pk_add_i16, v_pk_sub_u16, v_pk_sub_i16, v_pk_mul_lo_u16,
            v_pk_max_u16, v_pk_max_i16, v_pk_min_u16, v_pk_min_i16
  3-source: v_pk_mad_u16, v_pk_mad_i16

Verifies:
- Valid 3-entry op_sel:[a,b,0]/op_sel_hi:[a,b,1] (and zero neg_lo/neg_hi) on 2-source ops
  translate cleanly, validate with spirv-val (--target-env vulkan1.2), and match an independent
  ISA reference model under spirv_mini_interp.
- Rejection with rdna2spirv.Unsupported (never IndexError/ValueError) of non-binary op_sel/op_sel_hi
  entries, malformed bracket lists, wrong list lengths, duplicate modifiers, non-zero neg_lo/neg_hi,
  source neg/abs, unknown modifiers, and _dpp/_sdwa forms.

Standard library only; no GPU, no game files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

TWO_SRC_OPS = (
    "v_pk_add_u16",
    "v_pk_add_i16",
    "v_pk_sub_u16",
    "v_pk_sub_i16",
    "v_pk_mul_lo_u16",
    "v_pk_max_u16",
    "v_pk_max_i16",
    "v_pk_min_u16",
    "v_pk_min_i16",
)
THREE_SRC_OPS = (
    "v_pk_mad_u16",
    "v_pk_mad_i16",
)
ALL_OPS = TWO_SRC_OPS + THREE_SRC_OPS


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def ref_pk_int16(op, srcs, sel=None, selhi=None, clamp=False):
    ns = len(srcs)
    sel = list(sel[:ns]) if sel is not None else [0] * ns
    selhi = list(selhi[:ns]) if selhi is not None else [1] * ns
    signed = op.endswith("i16")
    kind = op[len("v_pk_"):].rsplit("_", 1)[0]
    out = 0
    for h, sl in enumerate((sel, selhi)):
        vals = [(srcs[i] >> (16 * sl[i])) & 0xFFFF for i in range(ns)]
        if signed:
            vals = [s16(v) for v in vals]
        if kind == "add":
            r = vals[0] + vals[1]
        elif kind == "sub":
            r = vals[0] - vals[1]
        elif kind == "mul_lo":
            r = vals[0] * vals[1]
        elif kind == "mad":
            r = vals[0] * vals[1] + vals[2]
        elif kind == "max":
            r = max(vals[0], vals[1])
        elif kind == "min":
            r = min(vals[0], vals[1])
        else:
            raise ValueError(f"unexpected op {op}")
        if clamp and kind in ("add", "sub", "mad"):
            r = max(-32768, min(32767, r)) if signed else max(0, min(0xFFFF, r))
        out |= (r & 0xFFFF) << (16 * h)
    return out


def translate_prog(lines):
    return support.translate_lines("ps_pk_int16_mods", lines)


def run_op(line, push):
    prog = [
        "v_mov_b32 v1, s0",
        "v_mov_b32 v2, s1",
        "v_mov_b32 v3, s2",
        "v_mov_b32 v0, 0xdeadbeef",
        line,
        "v_mov_b32 v4, 0",
        "exp mrt0 v0, v4, v4, v4 done vm",
        "s_endpgm",
    ]
    spv = translate_prog(prog)
    return support.mrt0_words(spv, push_constants=push)[0], spv


def find_spirv_tool(name):
    found = shutil.which(name)
    if found:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = os.path.join(sdk, "Bin", name + (".exe" if os.name == "nt" else ""))
        if os.path.isfile(cand):
            return cand
    return None


class PkInt16ModifierTests(unittest.TestCase):
    def test_valid_three_entry_opsel_on_two_source_ops(self):
        # llvm-mc / RDNA2 disassemblers frequently emit 3-entry op_sel / op_sel_hi on 2-source VOP3P ops.
        push = [0x80037FFE, 0xFFF20015, 0]
        combos = (
            ([1, 0, 0], [0, 1, 1], ""),
            ([0, 1, 0], [1, 0, 0], ""),
            ([1, 1, 1], [0, 0, 1], " neg_lo:[0,0,0] neg_hi:[0,0,0]"),
            ([1, 0, 0], [0, 1, 1], " clamp"),
        )
        for op in TWO_SRC_OPS:
            for sel, selhi, extra in combos:
                sel_s = ",".join(map(str, sel))
                selhi_s = ",".join(map(str, selhi))
                line3 = f"{op} v0, v1, v2 op_sel:[{sel_s}] op_sel_hi:[{selhi_s}]{extra}"
                line2 = (
                    f"{op} v0, v1, v2 op_sel:[{sel[0]},{sel[1]}] op_sel_hi:[{selhi[0]},{selhi[1]}]"
                    + (" clamp" if "clamp" in extra else "")
                )
                with self.subTest(op=op, line=line3):
                    got3, spv3 = run_op(line3, push)
                    got2, spv2 = run_op(line2, push)
                    want = ref_pk_int16(op, push[:2], sel=sel, selhi=selhi, clamp=("clamp" in extra))
                    self.assertEqual(got3, want)
                    self.assertEqual(got3, got2)
                    self.assertEqual(spv3, spv2)

    def test_valid_three_source_mad_modifiers(self):
        push = [0xFFFC0005, 0x0007FFFA, 0x7FF08010]
        for op in THREE_SRC_OPS:
            for sel, selhi, extra in (
                ([1, 0, 1], [0, 1, 0], ""),
                ([0, 1, 1], [1, 0, 0], " neg_lo:[0,0,0] neg_hi:[0,0,0] clamp"),
            ):
                sel_s = ",".join(map(str, sel))
                selhi_s = ",".join(map(str, selhi))
                line = f"{op} v0, v1, v2, v3 op_sel:[{sel_s}] op_sel_hi:[{selhi_s}]{extra}"
                with self.subTest(op=op, line=line):
                    got, _ = run_op(line, push)
                    want = ref_pk_int16(op, push, sel=sel, selhi=selhi, clamp=("clamp" in extra))
                    self.assertEqual(got, want)

    def test_reject_non_binary_opsel_entries(self):
        bad_mods_2 = (
            "op_sel:[2,0]",
            "op_sel:[0,2]",
            "op_sel:[-1,0]",
            "op_sel:[0,1,2]",
            "op_sel:[0,1,-1]",
            "op_sel_hi:[2,1]",
            "op_sel_hi:[1,-1]",
            "op_sel_hi:[1,0,2]",
            "op_sel_hi:[1,0,-1]",
        )
        for op in TWO_SRC_OPS:
            for mod in bad_mods_2:
                line = f"{op} v0, v1, v2 {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog(["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

        bad_mods_3 = (
            "op_sel:[0,1,2]",
            "op_sel:[-1,0,1]",
            "op_sel_hi:[1,2,0]",
            "op_sel_hi:[1,0,-1]",
        )
        for op in THREE_SRC_OPS:
            for mod in bad_mods_3:
                line = f"{op} v0, v1, v2, v3 {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog([
                            "v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 3",
                            line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm",
                        ])

    def test_reject_malformed_bracket_lists_and_lengths(self):
        malformed = (
            "op_sel:bad",
            "op_sel:",
            "op_sel:[0,1",
            "op_sel:[0,1]]",
            "op_sel:[]",
            "op_sel:[0,x]",
            "op_sel:foo[0,1]",
            "op_sel:[0]",
            "op_sel:[0,0,0,0]",
            "op_sel_hi:bad",
            "op_sel_hi:[1,1",
            "op_sel_hi:[1]",
            "op_sel_hi:[1,1,1,1]",
            "neg_lo:bad",
            "neg_lo:[0,x]",
            "neg_lo:[0]",
            "neg_lo:[0,0,0,0]",
            "neg_hi:bad",
            "neg_hi:[0,0",
            "neg_hi:[0]",
            "neg_hi:[0,0,0,0]",
            "op_sel:[0,0] op_sel:[1,1]",
            "op_sel_hi:[1,1] op_sel_hi:[0,0]",
            "neg_lo:[0,0] neg_lo:[0,0]",
        )
        for op in TWO_SRC_OPS:
            for mod in malformed:
                line = f"{op} v0, v1, v2 {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog(["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

        # 3-source ops also reject 2-entry lists and malformed lists.
        for op in THREE_SRC_OPS:
            for mod in ("op_sel:[0,1]", "op_sel_hi:[1,0]", "neg_lo:[0,0]", "neg_hi:[0,0]",
                        "op_sel:bad", "op_sel:[0,1,2", "op_sel:[0,0,0,0]", "op_sel:[0,0,0] op_sel:[1,1,1]"):
                line = f"{op} v0, v1, v2, v3 {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog([
                            "v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 3",
                            line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm",
                        ])

    def test_reject_nonzero_neg_lo_neg_hi_and_unknown_mods(self):
        bad_2 = (
            "neg_lo:[1,0]",
            "neg_lo:[0,1]",
            "neg_lo:[0,0,1]",
            "neg_lo:[-1,0]",
            "neg_hi:[1,0]",
            "neg_hi:[0,1]",
            "neg_hi:[0,0,1]",
            "neg_hi:[0,-1]",
            "mul:2",
            "div:2",
        )
        for op in TWO_SRC_OPS:
            for mod in bad_2:
                line = f"{op} v0, v1, v2 {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog(["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

        bad_3 = ("neg_lo:[1,0,0]", "neg_lo:[0,0,1]", "neg_hi:[0,1,0]", "neg_hi:[0,0,-1]", "mul:2", "div:2")
        for op in THREE_SRC_OPS:
            for mod in bad_3:
                line = f"{op} v0, v1, v2, v3 {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog([
                            "v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 3",
                            line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm",
                        ])

    def test_reject_source_neg_abs_and_dpp_sdwa(self):
        for op in TWO_SRC_OPS:
            cases = (
                f"{op} v0, -v1, v2",
                f"{op} v0, v1, -v2",
                f"{op} v0, |v1|, v2",
                f"{op} v0, v1, |v2|",
                f"{op} v0, -s0, v2",
                f"{op} v0, |s0|, v2",
                f"{op}_dpp v0, v1, v2",
                f"{op}_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_sdwa v0, v1, v2",
                f"{op}_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            )
            for line in cases:
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog(["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

        for op in THREE_SRC_OPS:
            cases = (
                f"{op} v0, -v1, v2, v3",
                f"{op} v0, v1, |v2|, v3",
                f"{op} v0, v1, v2, -v3",
                f"{op} v0, v1, v2, |v3|",
                f"{op}_dpp v0, v1, v2, v3",
                f"{op}_sdwa v0, v1, v2, v3",
            )
            for line in cases:
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_prog([
                            "v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 3",
                            line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm",
                        ])

    def test_spirv_val_vulkan12_three_entry_opsel(self):
        spirv_as = find_spirv_tool("spirv-as")
        spirv_val = find_spirv_tool("spirv-val")
        if not spirv_as or not spirv_val:
            self.skipTest("spirv-as / spirv-val not found")

        prog = [
            "v_mov_b32 v1, s0",
            "v_mov_b32 v2, s1",
            "v_mov_b32 v3, s2",
            "v_pk_add_u16 v4, v1, v2 op_sel:[1,0,0] op_sel_hi:[0,1,1] clamp",
            "v_pk_add_i16 v5, v4, v2 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
            "v_pk_sub_u16 v4, v5, v1 op_sel:[1,1,0] op_sel_hi:[0,0,1] clamp",
            "v_pk_sub_i16 v5, v4, v2 op_sel:[0,0,0] op_sel_hi:[1,1,0]",
            "v_pk_mul_lo_u16 v4, v5, v1 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[0,0,0] neg_hi:[0,0,0]",
            "v_pk_max_u16 v5, v4, v2 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
            "v_pk_max_i16 v4, v5, v1 op_sel:[1,0,0] op_sel_hi:[0,1,1]",
            "v_pk_min_u16 v5, v4, v2 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
            "v_pk_min_i16 v4, v5, v1 op_sel:[1,0,0] op_sel_hi:[0,1,1]",
            "v_pk_mad_u16 v5, v4, v2, v3 op_sel:[1,0,1] op_sel_hi:[0,1,0] clamp",
            "v_pk_mad_i16 v0, v5, v1, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1] neg_lo:[0,0,0] neg_hi:[0,0,0]",
            "v_mov_b32 v6, 0",
            "exp mrt0 v0, v4, v5, v6 done vm",
            "s_endpgm",
        ]
        spv = translate_prog(prog)
        with tempfile.TemporaryDirectory() as td:
            spvasm = os.path.join(td, "shader.spvasm")
            spvbin = os.path.join(td, "shader.spv")
            with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin], check=True)
            subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin], check=True)


if __name__ == "__main__":
    unittest.main()
