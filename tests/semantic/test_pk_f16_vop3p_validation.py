"""Validation and positive-control tests for packed f16 VOP3P ops (v_pk_{add,mul,max,min,fma}_f16).

Verifies that tools/rdna2spirv.py rejects malformed, non-binary, over-long, duplicate, or illegal VOP3P modifiers
(including mul:2/mul:4/div:2), _dpp/_sdwa encodings, per-operand neg/abs on VOP3P sources, invalid destinations, and
wrong arities with Unsupported, while preserving every valid LLVM/RDNA2 syntax form (1..3-entry op_sel/op_sel_hi/
neg_lo/neg_hi lists on 2- and 3-source ops, plus clamp) with llvm-mc gfx1030 decode round-trip, spirv-val vulkan1.2,
and spirv_mini_interp execution checks. Standard library only; synthetic inputs only.
"""
import os
import shutil
import struct
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

import mc_command  # noqa: E402
import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

TWO_SRC_OPS = ("v_pk_add_f16", "v_pk_mul_f16", "v_pk_max_f16", "v_pk_min_f16")
ALL_OPS = TWO_SRC_OPS + ("v_pk_fma_f16",)

# Synthetic f16 inputs (exact binary16 values, no rounding ambiguity):
#   v1 = (1.0, 2.0)   = 0x40003c00
#   v2 = (3.0, -4.0)  = 0xc4004200
#   v3 = (0.5, 0.25)  = 0x34003800
V1_HEX = "0x40003c00"
V2_HEX = "0xc4004200"
V3_HEX = "0x34003800"
V1_VALS = (1.0, 2.0)
V2_VALS = (3.0, -4.0)
V3_VALS = (0.5, 0.25)


def f2h(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def pack2(lo, hi):
    return f2h(lo) | (f2h(hi) << 16)


def eval_pk(op, sel, selhi, nlo, nhi, clamp=False):
    srcs = (V1_VALS, V2_VALS, V3_VALS) if op == "v_pk_fma_f16" else (V1_VALS, V2_VALS)
    halves = []
    for s_bits, n_bits in ((sel, nlo), (selhi, nhi)):
        vals = [(-srcs[i][s_bits[i]] if n_bits[i] else srcs[i][s_bits[i]]) for i in range(len(srcs))]
        if op == "v_pk_add_f16":
            r = vals[0] + vals[1]
        elif op == "v_pk_mul_f16":
            r = vals[0] * vals[1]
        elif op == "v_pk_max_f16":
            r = max(vals[0], vals[1])
        elif op == "v_pk_min_f16":
            r = min(vals[0], vals[1])
        else:
            r = vals[0] * vals[1] + vals[2]
        if clamp:
            r = min(max(r, 0.0), 1.0)
        halves.append(r)
    return pack2(halves[0], halves[1])


def _run_ps(op_line):
    lines = [
        f"v_mov_b32 v1, {V1_HEX}",
        f"v_mov_b32 v2, {V2_HEX}",
        f"v_mov_b32 v3, {V3_HEX}",
        op_line,
        "v_mov_b32 v4, 0",
        "exp mrt0 v0, v4, v4, v4 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_pk_f16_val", lines)
    return support.mrt0_words(spv)[0], spv


def _translate_reject(op_line):
    lines = [
        f"v_mov_b32 v1, {V1_HEX}",
        f"v_mov_b32 v2, {V2_HEX}",
        f"v_mov_b32 v3, {V3_HEX}",
        op_line,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    return support.translate_lines("ps_pk_f16_bad", lines)


def _spirv_tools():
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as and spirv_val:
        return spirv_as, spirv_val
    return None, None


def _spirv_validate(spv):
    spirv_as, spirv_val = _spirv_tools()
    if not spirv_as or not spirv_val:
        return False
    with tempfile.TemporaryDirectory() as td:
        asm_path = os.path.join(td, "shader.spvasm")
        bin_path = os.path.join(td, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", bin_path], check=True)
        subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_path], check=True)
    return True


def _llvm_mc():
    cmd = mc_command.mc_command()
    try:
        r = subprocess.run(
            cmd + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
            input="s_endpgm\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and "encoding" in r.stdout else None


MC = _llvm_mc()


def _assemble_gfx1030(lines):
    r = subprocess.run(
        MC + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{r.stderr}")
    blob = bytearray()
    for ln in r.stdout.splitlines():
        if "encoding: [" in ln:
            body = ln.split("encoding: [", 1)[1].split("]", 1)[0]
            blob.extend(int(b, 16) for b in body.split(","))
    return bytes(blob)


class PkF16Vop3pRejectionTests(unittest.TestCase):
    def _srcs(self, op):
        return "v1, v2, v3" if op == "v_pk_fma_f16" else "v1, v2"

    def test_malformed_and_out_of_range_modifiers(self):
        bad_mods = (
            "op_sel:[]",
            "op_sel:[1,",
            "op_sel:1,0]",
            "op_sel:0",
            "op_sel:[1,]",
            "op_sel:[0,1]bad",
            "op_sel:[[0,1]]",
            "op_sel:[2,0]",
            "op_sel:[-1,0]",
            "op_sel:[x,0]",
            "op_sel:[01,0]",
            "op_sel:[+1,0]",
            "op_sel:[0,0,0,0]",
            "op_sel_hi:[]",
            "op_sel_hi:[0,2]",
            "op_sel_hi:[1,1,1,1]",
            "neg_lo:[1",
            "neg_lo:[0,-1]",
            "neg_lo:[0,0,0,0]",
            "neg_hi:0,1]",
            "neg_hi:[1,x]",
            "neg_hi:[0,0,0,0]",
        )
        for op in ALL_OPS:
            srcs = self._srcs(op)
            for mod in bad_mods:
                line = f"{op} v0, {srcs} {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_reject(line)

    def test_duplicate_and_illegal_modifiers(self):
        bad_mods = (
            "op_sel:[0,1] op_sel:[1,0]",
            "op_sel_hi:[0,1] op_sel_hi:[1,0]",
            "neg_lo:[1,0] neg_lo:[0,1]",
            "neg_hi:[1,0] neg_hi:[0,1]",
            "clamp clamp",
            "mul:2",
            "mul:4",
            "div:2",
            "abs:[1,0]",
            "foo",
        )
        for op in ALL_OPS:
            srcs = self._srcs(op)
            for mod in bad_mods:
                line = f"{op} v0, {srcs} {mod}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_reject(line)

    def test_dpp_and_sdwa_rejected(self):
        for op in ALL_OPS:
            srcs = self._srcs(op)
            for line in (
                f"{op}_dpp v0, {srcs} row_mirror",
                f"{op}_dpp v0, {srcs} quad_perm:[1,0,3,2]",
                f"{op}_sdwa v0, {srcs} src0_sel:DWORD",
            ):
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_reject(line)

    def test_per_operand_neg_abs_rejected(self):
        for op in TWO_SRC_OPS:
            for line in (
                f"{op} v0, -v1, v2",
                f"{op} v0, |v1|, v2",
                f"{op} v0, v1, -v2",
                f"{op} v0, v1, |v2|",
                f"{op} v0, -|v1|, v2",
            ):
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_reject(line)
        for line in (
            "v_pk_fma_f16 v0, -v1, v2, v3",
            "v_pk_fma_f16 v0, |v1|, v2, v3",
            "v_pk_fma_f16 v0, v1, -v2, v3",
            "v_pk_fma_f16 v0, v1, v2, |v3|",
            "v_pk_fma_f16 v0, v1, v2, -|v3|",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate_reject(line)

    def test_invalid_destination_and_arity_rejected(self):
        for op in ALL_OPS:
            srcs = self._srcs(op)
            for dst in ("s0", "vcc_lo", "v[0:1]", "-v0", "|v0|"):
                line = f"{op} {dst}, {srcs}"
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_reject(line)
        for op in TWO_SRC_OPS:
            for line in (f"{op} v0, v1", f"{op} v0, v1, v2, v3"):
                with self.subTest(line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_reject(line)
        for line in ("v_pk_fma_f16 v0, v1, v2", "v_pk_fma_f16 v0, v1, v2, v3, v4"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate_reject(line)


class PkF16Vop3pPositiveControlTests(unittest.TestCase):
    def test_valid_modifier_forms_interpreter_and_spirv_val(self):
        cases = []
        for op in TWO_SRC_OPS:
            cases.extend([
                (op, f"{op} v0, v1, v2", (0, 0), (1, 1), (0, 0), (0, 0), False),
                (op, f"{op} v0, v1, v2 clamp", (0, 0), (1, 1), (0, 0), (0, 0), True),
                (op, f"{op} v0, v1, v2 op_sel:[1] op_sel_hi:[0] neg_lo:[1] neg_hi:[1]",
                 (1, 0), (0, 0), (1, 0), (1, 0), False),
                (op, f"{op} v0, v1, v2 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[0,1,0] neg_hi:[1,0,0] clamp",
                 (1, 0), (0, 1), (0, 1), (1, 0), True),
            ])
        cases.extend([
            ("v_pk_fma_f16", "v_pk_fma_f16 v0, v1, v2, v3", (0, 0, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0), False),
            ("v_pk_fma_f16", "v_pk_fma_f16 v0, v1, v2, v3 clamp", (0, 0, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0), True),
            ("v_pk_fma_f16", "v_pk_fma_f16 v0, v1, v2, v3 op_sel:[1] op_sel_hi:[0] neg_lo:[1] neg_hi:[1]",
             (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), False),
            ("v_pk_fma_f16", "v_pk_fma_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1]",
             (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), False),
            ("v_pk_fma_f16",
             "v_pk_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,1] neg_hi:[0,1,0] clamp",
             (1, 0, 1), (0, 1, 0), (1, 0, 1), (0, 1, 0), True),
        ])
        for op, line, sel, selhi, nlo, nhi, clamp in cases:
            with self.subTest(line=line):
                got, spv = _run_ps(line)
                want = eval_pk(op, sel, selhi, nlo, nhi, clamp=clamp)
                self.assertEqual(got, want, f"{line}: got {got:#010x}, want {want:#010x}")
                _spirv_validate(spv)

    @unittest.skipUnless(MC, "needs llvm-mc with AMDGPU gfx1030 target (set LLVM_MC)")
    def test_llvm_mc_gfx1030_roundtrip_and_semantics(self):
        cases = [
            ("v_pk_add_f16", "v_pk_add_f16 v0, v1, v2 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[1,0,0] clamp",
             (1, 0), (0, 1), (1, 0), (0, 0), True),
            ("v_pk_mul_f16", "v_pk_mul_f16 v0, v1, v2 op_sel:[1] op_sel_hi:[0] neg_hi:[1]",
             (1, 0), (0, 0), (0, 0), (1, 0), False),
            ("v_pk_max_f16", "v_pk_max_f16 v0, v1, v2 op_sel:[0,1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[1,0]",
             (0, 1), (1, 0), (0, 1), (1, 0), False),
            ("v_pk_min_f16", "v_pk_min_f16 v0, v1, v2 op_sel:[1,1] op_sel_hi:[0,0] clamp",
             (1, 1), (0, 0), (0, 0), (0, 0), True),
            ("v_pk_fma_f16", "v_pk_fma_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] clamp",
             (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0), True),
        ]
        for op, ins_line, sel, selhi, nlo, nhi, clamp in cases:
            with self.subTest(op=op):
                asm_lines = [
                    f"v_mov_b32 v1, {V1_HEX}",
                    f"v_mov_b32 v2, {V2_HEX}",
                    f"v_mov_b32 v3, {V3_HEX}",
                    ins_line,
                    "v_mov_b32 v4, 0",
                    "exp mrt0 v0, v4, v4, v4 done vm",
                    "s_endpgm",
                ]
                blob = _assemble_gfx1030(asm_lines)
                decoded, unknown = rdna2_decode.disasm_bytes(blob)
                self.assertEqual(unknown, 0, decoded)
                self.assertIn(op, decoded)
                decoded_lines = [ln for ln in decoded.splitlines() if ln.strip() and not ln.strip().startswith(";")]
                spv = support.translate_lines(f"ps_rt_{op}", decoded_lines)
                got = support.mrt0_words(spv)[0]
                want = eval_pk(op, sel, selhi, nlo, nhi, clamp=clamp)
                self.assertEqual(got, want, f"{op} roundtrip: got {got:#010x}, want {want:#010x}")
                _spirv_validate(spv)


if __name__ == "__main__":
    unittest.main()
