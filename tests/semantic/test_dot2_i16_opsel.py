"""Semantic SPIR-V and interpreter tests for VOP3P op_sel/op_sel_hi on v_dot2_i32_i16 and v_dot2_u32_u16.

Translates synthetic RDNA2 pixel shaders via tools/rdna2spirv.py, runs them in tests/semantic/spirv_mini_interp.py
against the gfx1036-measured model in tools/dot2_i16_opsel_probe.py, pins observed gfx1036 hardware words,
checks byte-identical default/no-op SPIR-V emission, inline/literal/SGPR sources, inactive EXEC preservation,
Unsupported rejection of invalid/unmodelled modifiers, and spirv-as/spirv-val validation. Standard library only.
"""
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import dot2_i16_opsel_probe as probe  # noqa: E402
from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF

# Exact words measured natively on RDNA2 gfx1036 (AMD Radeon(TM) Graphics, tools/hw_oracle, wave32 & wave64).
MEASURED_OPSEL_GFX1036 = (
    ("v_dot2_i32_i16", 0x80007FFF, 0xFFFF0002, 0x00000123, "op_sel:[1,0,0] op_sel_hi:[0,1,1]", 0xFFFE8124),
    ("v_dot2_i32_i16", 0x80007FFF, 0xFFFF0002, 0x00000123, "op_sel:[0,1,0] op_sel_hi:[1,0,1]", 0xFFFE8124),
    ("v_dot2_i32_i16", 0x00018000, 0x80000001, 0xFFFFFFFF, "op_sel:[0,1,0] op_sel_hi:[0,1,1] clamp", 0x7FFFFFFF),
    ("v_dot2_i32_i16", 0x00018000, 0x80000001, 0x00000000, "op_sel:[0,1,0] op_sel_hi:[0,1,1] clamp", 0x7FFFFFFF),
    ("v_dot2_i32_i16", 0x00018000, 0x80000001, 0x00000000, "op_sel:[0,1,0] op_sel_hi:[0,1,1]", 0x80000000),
    ("v_dot2_i32_i16", 0x80007FFF, 0x7FFF8000, 0x80000000, "op_sel:[0,1,0] op_sel_hi:[1,0,1] clamp", 0xFFFF0001),
    ("v_dot2_i32_i16", 0x1234ABCD, 0x5678EF01, 0x9ABCDEF0, "op_sel:[1,1,0] op_sel_hi:[0,0,1]", 0xA679EE1D),
    ("v_dot2_u32_u16", 0x80007FFF, 0xFFFF0002, 0x00000123, "op_sel:[1,0,0] op_sel_hi:[0,1,1]", 0x7FFF8124),
    ("v_dot2_u32_u16", 0x80007FFF, 0xFFFF0002, 0x00000123, "op_sel:[0,1,0] op_sel_hi:[1,0,1]", 0x7FFF8124),
    ("v_dot2_u32_u16", 0xFFFF0001, 0x0001FFFF, 0x80000000, "op_sel:[1,0,0] op_sel_hi:[0,1,1] clamp", 0xFFFFFFFF),
    ("v_dot2_u32_u16", 0xFFFF0001, 0x0001FFFF, 0x80000000, "op_sel:[1,0,0] op_sel_hi:[0,1,1]", 0x7FFE0002),
    ("v_dot2_u32_u16", 0xFFFF0001, 0x0001FFFF, 0x0001FFFE, "op_sel:[1,0,0] op_sel_hi:[0,1,1] clamp", 0xFFFFFFFF),
    ("v_dot2_u32_u16", 0x1234ABCD, 0x5678EF01, 0x9ABCDEF0, "op_sel:[1,1,0] op_sel_hi:[0,0,1]", 0x4147EE1D),
)


def _run_batch4(cases, op, mod):
    """Execute up to 4 (s0, s1, s2) cases in one shader and return mrt0 words."""
    lines = []
    suffix = f" {mod}" if mod else ""
    for k, (s0, s1, s2) in enumerate(cases):
        lines.extend([
            f"v_mov_b32 v{4 + k}, {s0 & MASK32:#x}",
            f"v_mov_b32 v{8 + k}, {s1 & MASK32:#x}",
            f"v_mov_b32 v{12 + k}, {s2 & MASK32:#x}",
            f"{op} v{k}, v{4 + k}, v{8 + k}, v{12 + k}{suffix}",
        ])
    lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    return mrt0_words(translate_lines("ps_dot2_opsel_batch", lines))[:len(cases)]


def _run_chunks(cases, op, mod):
    out = []
    for i in range(0, len(cases), 4):
        chunk = cases[i:i + 4]
        padded = chunk + [(0, 0, 0)] * (4 - len(chunk))
        out.extend(_run_batch4(padded, op, mod)[:len(chunk)])
    return out


class Dot2I16OpselSemanticTests(unittest.TestCase):
    def test_all_16_opsel_and_clamp_edge_vectors(self):
        for op in probe.OPS:
            for s0_lo, s1_lo, s0_hi, s1_hi in itertools.product((0, 1), repeat=4):
                sel = (s0_lo, s1_lo, 0)
                selhi = (s0_hi, s1_hi, 1)
                for clamp in (False, True):
                    mod = f"op_sel:[{s0_lo},{s1_lo},0] op_sel_hi:[{s0_hi},{s1_hi},1]" + (" clamp" if clamp else "")
                    got = _run_chunks(list(probe.EDGE_VECS), op, mod)
                    want = [
                        probe.ref_dot2(op, s0, s1, s2, sel=sel, selhi=selhi, clamp=clamp)
                        for s0, s1, s2 in probe.EDGE_VECS
                    ]
                    with self.subTest(op=op, mod=mod):
                        self.assertEqual(got, want)

    def test_measured_gfx1036_words(self):
        for i in range(0, len(MEASURED_OPSEL_GFX1036), 4):
            chunk = MEASURED_OPSEL_GFX1036[i:i + 4]
            lines = []
            wants = []
            for k, (op, s0, s1, s2, mod, want) in enumerate(chunk):
                lines.extend([
                    f"v_mov_b32 v{4 + k}, {s0:#x}",
                    f"v_mov_b32 v{8 + k}, {s1:#x}",
                    f"v_mov_b32 v{12 + k}, {s2:#x}",
                    f"{op} v{k}, v{4 + k}, v{8 + k}, v{12 + k} {mod}",
                ])
                wants.append(want)
            for k in range(len(chunk), 4):
                lines.append(f"v_mov_b32 v{k}, 0")
            lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
            got = mrt0_words(translate_lines("ps_dot2_opsel_hw", lines))[:len(chunk)]
            with self.subTest(chunk=i):
                self.assertEqual(got, wants)

    def test_sources_omitted_defaults_and_zero_neg_noops(self):
        for op in probe.OPS:
            for sc in probe.SRC_CASES:
                lines = list(sc.setup)
                if sc.s0_tok == "v1":
                    lines.append(f"v_mov_b32 v1, {sc.s0_val:#x}")
                if sc.s1_tok == "v2":
                    lines.append(f"v_mov_b32 v2, {sc.s1_val:#x}")
                if sc.s2_tok == "v3":
                    lines.append(f"v_mov_b32 v3, {sc.s2_val:#x}")
                lines.extend([
                    f"{op} v0, {sc.s0_tok}, {sc.s1_tok}, {sc.s2_tok} {sc.mod}",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ])
                want = probe.ref_dot2(op, sc.s0_val, sc.s1_val, sc.s2_val, sel=sc.sel, selhi=sc.selhi)
                with self.subTest(op=op, case=sc.name):
                    got = mrt0_words(translate_lines("ps_dot2_opsel_src", lines))[0]
                    self.assertEqual(got, want)

    def test_default_and_zero_neg_emit_byte_identical_spirv(self):
        preamble = ["v_mov_b32 v1, 0x00030002", "v_mov_b32 v2, 0x00070005", "v_mov_b32 v3, 0x0000000a"]
        tail = ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        for op in probe.OPS:
            for clamp_suf in ("", " clamp"):
                spv_plain = translate_lines("ps_dot2_ident", preamble + [f"{op} v0, v1, v2, v3{clamp_suf}"] + tail)
                for eq_mod in (
                    "op_sel:[0,0,0]",
                    "op_sel_hi:[1,1,1]",
                    "op_sel:[0,0,0] op_sel_hi:[1,1,1]",
                    "neg_lo:[0,0,0] neg_hi:[0,0,0]",
                    "op_sel:[0,0,0] op_sel_hi:[1,1,1] neg_lo:[0,0,0] neg_hi:[0,0,0]",
                ):
                    with self.subTest(op=op, clamp=bool(clamp_suf), mod=eq_mod):
                        spv_mod = translate_lines(
                            "ps_dot2_ident",
                            preamble + [f"{op} v0, v1, v2, v3 {eq_mod}{clamp_suf}"] + tail,
                        )
                        self.assertEqual(spv_mod, spv_plain)

    def test_inactive_exec_preserves_dst(self):
        sentinel = 0xA5A55A5A
        for op in probe.OPS:
            for mod in ("op_sel:[1,0,0] op_sel_hi:[0,1,1]", "op_sel:[0,1,0] op_sel_hi:[0,1,1] clamp"):
                with self.subTest(op=op, mod=mod):
                    got = mrt0_words(translate_lines("ps_dot2_opsel_exec", [
                        f"v_mov_b32 v0, {sentinel:#x}",
                        "v_mov_b32 v1, 0x00030002",
                        "v_mov_b32 v2, 0x00070005",
                        "v_mov_b32 v3, 10",
                        "s_mov_b32 s10, exec_lo",
                        "s_mov_b32 exec_lo, 0",
                        f"{op} v0, v1, v2, v3 {mod}",
                        "s_mov_b32 exec_lo, s10",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ]))[0]
                    self.assertEqual(got, sentinel)

    def test_rejected_modifiers_and_forms(self):
        bad_lines = [
            # Non-default S2 half-select on v_dot2_i32_i16 / v_dot2_u32_u16
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0,1]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[1,1,0]",
            "v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,0,1]",
            "v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[1,1,0]",
            # Malformed / short / out-of-range / duplicate modifiers
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[2,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0,0] op_sel:[0,0,0]",
            "v_dot2_u32_u16 v0, v1, v2, v3 neg_lo:[0,0]",
            "v_dot2_u32_u16 v0, v1, v2, v3 op_sel:bad",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[x,1,1]",
            # Non-zero neg_lo / neg_hi bits
            "v_dot2_i32_i16 v0, v1, v2, v3 neg_lo:[1,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 neg_hi:[0,1,0]",
            "v_dot2_u32_u16 v0, v1, v2, v3 neg_lo:[0,0,1]",
            "v_dot2_u32_u16 v0, v1, v2, v3 neg_hi:[0,0,1]",
            # Float source/output modifiers and DPP/SDWA
            "v_dot2_i32_i16 v0, -v1, v2, v3 op_sel:[1,0,0]",
            "v_dot2_u32_u16 v0, |v1|, v2, v3 op_sel:[1,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 mul:2",
            "v_dot2_u32_u16 v0, v1, v2, v3 div:2",
            "v_dot2_i32_i16_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2]",
            "v_dot2_u32_u16_sdwa v0, v1, v2, v3 src0_sel:DWORD",
            # Other dot ops keep strict rejection
            "v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0,0]",
            "v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,1,1]",
            "v_dot8_u32_u4 v0, v1, v2, v3 neg_lo:[0,0,0]",
            "v_dot2c_i32_i16 v0, v1, v2 op_sel:[1,0,0]",
            "v_dot2c_i32_i16 v0, v1, v2 neg_lo:[0,0,0]",
        ]
        for line in bad_lines:
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines("ps_dot2_opsel_bad", [
                    "v_mov_b32 v1, 1",
                    "v_mov_b32 v2, 2",
                    "v_mov_b32 v3, 3",
                    line,
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ])

    def test_spirv_validates_with_spirv_val(self):
        if not shutil.which("spirv-as") or not shutil.which("spirv-val"):
            if os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS") == "1":
                self.fail("spirv-as / spirv-val missing while LEONIDA_REQUIRE_SPIRV_TOOLS=1")
            self.skipTest("spirv-as or spirv-val not on PATH")
        with tempfile.TemporaryDirectory() as d:
            for op in probe.OPS:
                for mod in (
                    "op_sel:[1,0,0] op_sel_hi:[0,1,1]",
                    "op_sel:[0,1,0] op_sel_hi:[0,1,1] clamp",
                    "op_sel:[1,1,0] op_sel_hi:[0,0,1] neg_lo:[0,0,0] neg_hi:[0,0,0] clamp",
                ):
                    spv_text = translate_lines("ps_dot2_opsel_val", [
                        "v_mov_b32 v1, 0x00018000",
                        "v_mov_b32 v2, 0x80000001",
                        "v_mov_b32 v3, 0xffffffff",
                        f"{op} v0, v1, v2, v3 {mod}",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ])
                    sa_path = os.path.join(d, "test.spvasm")
                    spv_path = os.path.join(d, "test.spv")
                    with open(sa_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spv_text)
                    rdna2spirv.assemble_spirv(sa_path, spv_path)
                    res = subprocess.run(
                        ["spirv-val", "--target-env", "vulkan1.3", spv_path],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(res.returncode, 0, f"{op} {mod}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
