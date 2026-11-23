"""Semantic, binary-decode, and spirv-val tests for VOP3P neg_lo / neg_hi on v_dot4_i32_i8.

RDNA2 hardware behavior (measured on gfx1036, docs/design.md "Signed-byte dot products"):
  - S0 and S1: neg_lo[k] flips bit 15 (0x00008000) and neg_hi[k] flips bit 31 (0x80000000).
  - S2 (addend): neg_hi[2] clears bit 31 (& 0x7FFFFFFF), then neg_lo[2] flips bit 31 (^ 0x80000000).
  - clamp: saturates the 32-bit signed sum (modified_acc + dot) to [INT32_MIN, INT32_MAX].
  - v_dot4c_i32_i8 (VOP2): refuses neg_lo / neg_hi.

    python -I -m unittest discover -s tests/semantic -p test_dot4_i8_neg.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2_decode  # noqa: E402
import support  # noqa: E402
from mc_command import mc_command  # noqa: E402

rdna2spirv = support.rdna2spirv

INT32_MIN = 0x80000000
INT32_MAX = 0x7FFFFFFF

# Hardware-measured on gfx1036 (same table as tests/semantic/test_alu_dot4_i8.py)
MEASURED_NEG = [
    ((0, 0, 0), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0x000000AA),
    ((1, 0, 0), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFDAA),
    ((0, 0, 0), (1, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFCAA),
    ((0, 1, 0), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFFAA),
    ((0, 0, 0), (0, 1, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFEAA),
    ((0, 0, 1), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0x800000AA),
    ((0, 0, 0), (0, 0, 1), 0x04030201, 0x08070605, 0x00000064, 0x000000AA),
    ((0, 0, 1), (0, 0, 1), 0x04030201, 0x08070605, 0x00000064, 0x800000AA),
    ((1, 1, 1), (1, 1, 1), 0x04030201, 0x08070605, 0x00000064, 0x800076AA),
]


def _s8(x):
    return x - 256 if x & 0x80 else x


def _s32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def ref_dot4_i8_neg(neg_lo, neg_hi, a, b, addend, clamp=False):
    a ^= (0x8000 if neg_lo[0] else 0) ^ (0x80000000 if neg_hi[0] else 0)
    b ^= (0x8000 if neg_lo[1] else 0) ^ (0x80000000 if neg_hi[1] else 0)
    if neg_hi[2]:
        addend &= 0x7FFFFFFF
    if neg_lo[2]:
        addend ^= 0x80000000
    dot = sum(_s8((a >> (8 * i)) & 0xFF) * _s8((b >> (8 * i)) & 0xFF) for i in range(4))
    total = _s32(addend) + dot
    if clamp:
        if total > 0x7FFFFFFF:
            return INT32_MAX
        if total < -0x80000000:
            return INT32_MIN
    return total & 0xFFFFFFFF


def _fmt_mods(neg_lo, neg_hi, clamp=False):
    parts = []
    if any(neg_lo):
        parts.append(f"neg_lo:[{','.join(str(b) for b in neg_lo)}]")
    if any(neg_hi):
        parts.append(f"neg_hi:[{','.join(str(b) for b in neg_hi)}]")
    if clamp:
        parts.append("clamp")
    return (" " + " ".join(parts)) if parts else ""


def _has_llvm_mc_amdgpu():
    try:
        cmd = mc_command() + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"]
        res = subprocess.run(cmd, input=b"s_endpgm\n", capture_output=True, timeout=15)
        return res.returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _validate_spvasm(spvasm):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_path],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")
    return True


HAS_LLVM_MC = _has_llvm_mc_amdgpu()


class Dot4I8NegSemanticTests(unittest.TestCase):
    def test_measured_hardware_table(self):
        for neg_lo, neg_hi, a, b, addend, measured in MEASURED_NEG:
            mod = f"neg_lo:[{','.join(map(str, neg_lo))}] neg_hi:[{','.join(map(str, neg_hi))}]"
            with self.subTest(neg_lo=neg_lo, neg_hi=neg_hi):
                self.assertEqual(ref_dot4_i8_neg(neg_lo, neg_hi, a, b, addend), measured)
                lines = [
                    f"v_mov_b32_e32 v1, 0x{a:08x}",
                    f"v_mov_b32_e32 v2, 0x{b:08x}",
                    f"v_mov_b32_e32 v3, 0x{addend:08x}",
                    f"v_dot4_i32_i8 v8, v1, v2, v3 {mod}",
                    "exp mrt0 v8, v8, v8, v8 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines("ps_dot4_neg_hw", lines)
                self.assertEqual(support.mrt0_words(spv), [measured] * 4)

    def test_all_64_neg_masks_positive_and_negative_addend(self):
        # Test every combination of neg_lo[0..2] x neg_hi[0..2] (64 masks) with both a positive and negative addend.
        a = 0x7F8001FF
        b = 0x807FFF01
        for addend in (0x00001234, 0x80001234):
            masks = [
                (((m >> 0) & 1, (m >> 1) & 1, (m >> 2) & 1),
                 ((m >> 3) & 1, (m >> 4) & 1, (m >> 5) & 1))
                for m in range(64)
            ]
            for base in range(0, 64, 4):
                batch = masks[base:base + 4]
                lines = [
                    f"v_mov_b32_e32 v1, 0x{a:08x}",
                    f"v_mov_b32_e32 v2, 0x{b:08x}",
                    f"v_mov_b32_e32 v3, 0x{addend:08x}",
                ]
                expected = []
                for i, (nlo, nhi) in enumerate(batch):
                    lines.append(f"v_dot4_i32_i8 v{8 + i}, v1, v2, v3{_fmt_mods(nlo, nhi)}")
                    expected.append(ref_dot4_i8_neg(nlo, nhi, a, b, addend))
                lines.extend(["exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm"])
                with self.subTest(addend=f"{addend:#010x}", base=base):
                    spv = support.translate_lines(f"ps_dot4_neg_64_{base}", lines)
                    self.assertEqual(support.mrt0_words(spv), expected)

    def test_edge_values_bytes_and_addend(self):
        # Edge bytes: 0x80 (-128) <-> 0x00 (0), 0x7F (+127) <-> 0xFF (-1); edge addends: INT32_MIN, INT32_MAX, 0, -1.
        cases = [
            # Flipping 0x80 -> 0x00 in bytes 1 and 3 eliminates those products
            ((1, 0, 0), (1, 0, 0), 0x80008000, 0x80808080, 0x00000000),
            # Flipping 0x00 -> 0x80 in bytes 1 and 3 produces (-128)*(-128)*2 = 32768
            ((1, 0, 0), (1, 0, 0), 0x00000000, 0x80808080, 0x00000000),
            # Flipping 0x7F -> 0xFF (-1) on both sources
            ((1, 1, 0), (1, 1, 0), 0x7F7F7F7F, 0x7F7F7F7F, 0x00000000),
            # S2 = INT32_MIN (0x80000000): neg_hi clears bit 31 -> 0; neg_lo flips bit 31 -> 0; both -> INT32_MIN
            ((0, 0, 0), (0, 0, 1), 0x01010101, 0x01010101, INT32_MIN),
            ((0, 0, 1), (0, 0, 0), 0x01010101, 0x01010101, INT32_MIN),
            ((0, 0, 1), (0, 0, 1), 0x01010101, 0x01010101, INT32_MIN),
            # S2 = INT32_MAX (0x7FFFFFFF): neg_hi keeps INT32_MAX; neg_lo -> 0xFFFFFFFF (-1); both -> 0xFFFFFFFF (-1)
            ((0, 0, 0), (0, 0, 1), 0x01010101, 0x01010101, INT32_MAX),
            ((0, 0, 1), (0, 0, 0), 0x01010101, 0x01010101, INT32_MAX),
            ((0, 0, 1), (0, 0, 1), 0x01010101, 0x01010101, INT32_MAX),
            # S2 = 0xFFFFFFFF (-1): neg_hi -> INT32_MAX; neg_lo -> INT32_MAX; both -> 0xFFFFFFFF (-1)
            ((0, 0, 0), (0, 0, 1), 0x01010101, 0x01010101, 0xFFFFFFFF),
            ((0, 0, 1), (0, 0, 0), 0x01010101, 0x01010101, 0xFFFFFFFF),
            ((0, 0, 1), (0, 0, 1), 0x01010101, 0x01010101, 0xFFFFFFFF),
        ]
        for nlo, nhi, a, b, addend in cases:
            expected = ref_dot4_i8_neg(nlo, nhi, a, b, addend)
            with self.subTest(nlo=nlo, nhi=nhi, a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                lines = [
                    f"v_mov_b32_e32 v1, 0x{a:08x}",
                    f"v_mov_b32_e32 v2, 0x{b:08x}",
                    f"v_mov_b32_e32 v3, 0x{addend:08x}",
                    f"v_dot4_i32_i8 v8, v1, v2, v3{_fmt_mods(nlo, nhi)}",
                    "exp mrt0 v8, v8, v8, v8 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines("ps_dot4_neg_edge", lines)
                self.assertEqual(support.mrt0_words(spv), [expected] * 4)

    def test_neg_with_clamp(self):
        cases = [
            # neg_hi[2] turns 0xFFFFFFFF into INT32_MAX; positive dot overflows -> clamp saturates to INT32_MAX
            ((0, 0, 0), (0, 0, 1), 0x7F7F7F7F, 0x7F7F7F7F, 0xFFFFFFFF, True, INT32_MAX),
            ((0, 0, 0), (0, 0, 1), 0x7F7F7F7F, 0x7F7F7F7F, 0xFFFFFFFF, False, (INT32_MAX + 4 * 127 * 127) & 0xFFFFFFFF),
            # neg_lo[2] turns 0x00000000 into INT32_MIN; negative dot underflows -> clamp saturates to INT32_MIN
            ((0, 0, 1), (0, 0, 0), 0x80808080, 0x7F7F7F7F, 0x00000000, True, INT32_MIN),
            ((0, 0, 1), (0, 0, 0), 0x80808080, 0x7F7F7F7F, 0x00000000, False, (INT32_MIN - 4 * 128 * 127) & 0xFFFFFFFF),
            # neg_lo[2]=1, neg_hi[2]=1 turns 0x00000001 into 0x80000001; negative dot underflows -> INT32_MIN
            ((0, 0, 1), (0, 0, 1), 0x80808080, 0x7F7F7F7F, 0x00000001, True, INT32_MIN),
            # neg_lo[0]/neg_hi[0] flips 0x00000000 -> 0x80008000, giving negative dot with 0x7F7F7F7F -> INT32_MIN
            ((1, 0, 0), (1, 0, 0), 0x00000000, 0x7F7F7F7F, INT32_MIN, True, INT32_MIN),
            # neg_lo[0]/neg_hi[0] flips 0x00000000 -> 0x80008000, giving positive dot with 0x80808080 -> INT32_MAX
            ((1, 0, 0), (1, 0, 0), 0x00000000, 0x80808080, INT32_MAX, True, INT32_MAX),
            # Opposite signs after neg_lo[2] -> no overflow even with clamp
            ((0, 0, 1), (0, 0, 0), 0x7F7F7F7F, 0x7F7F7F7F, 0x00000000, True, (INT32_MIN + 4 * 127 * 127) & 0xFFFFFFFF),
        ]
        for nlo, nhi, a, b, addend, clamp, expected in cases:
            self.assertEqual(ref_dot4_i8_neg(nlo, nhi, a, b, addend, clamp=clamp), expected)
            with self.subTest(nlo=nlo, nhi=nhi, clamp=clamp, addend=f"{addend:#010x}"):
                lines = [
                    f"v_mov_b32_e32 v1, 0x{a:08x}",
                    f"v_mov_b32_e32 v2, 0x{b:08x}",
                    f"v_mov_b32_e32 v3, 0x{addend:08x}",
                    f"v_dot4_i32_i8 v8, v1, v2, v3{_fmt_mods(nlo, nhi, clamp=clamp)}",
                    "exp mrt0 v8, v8, v8, v8 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines("ps_dot4_neg_clamp", lines)
                self.assertEqual(support.mrt0_words(spv), [expected] * 4)

    def test_sgpr_and_literal_operands(self):
        a, b, addend = 0x04030201, 0x08070605, 0x00000064
        nlo, nhi = (1, 0, 1), (0, 1, 1)
        expected = ref_dot4_i8_neg(nlo, nhi, a, b, addend)
        lines = [
            f"v_mov_b32_e32 v1, 0x{a:08x}",
            f"s_mov_b32 s4, 0x{addend:08x}",
            f"v_dot4_i32_i8 v8, v1, 0x{b:08x}, s4{_fmt_mods(nlo, nhi)}",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_dot4_neg_lit_sgpr", lines)
        self.assertEqual(support.mrt0_words(spv), [expected] * 4)

    def test_refused_forms(self):
        for name, inst, err in (
            ("ps_dot4c_neg_lo", "v_dot4c_i32_i8 v8, v1, v2 neg_lo:[1,0,0]", "v_dot4c_i32_i8 neg"),
            ("ps_dot4c_neg_hi", "v_dot4c_i32_i8 v8, v1, v2 neg_hi:[0,1,0]", "v_dot4c_i32_i8 neg"),
            ("ps_dot4_neg_bad_len", "v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[1,0]", "v_dot4_i32_i8 neg"),
            ("ps_dot4_neg_bad_val", "v_dot4_i32_i8 v8, v1, v2, v3 neg_hi:[0,2,0]", "v_dot4_i32_i8 neg"),
        ):
            with self.subTest(name=name):
                lines = [
                    "v_mov_b32_e32 v1, 1",
                    "v_mov_b32_e32 v2, 1",
                    "v_mov_b32_e32 v3, 0",
                    inst,
                    "exp mrt0 v8, v8, v8, v8 done vm",
                    "s_endpgm",
                ]
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    support.translate_lines(name, lines)
                self.assertEqual(str(ctx.exception), err)

    def test_spirv_val_vulkan12(self):
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            self.skipTest("spirv-as / spirv-val not on PATH")
        lines = [
            "v_mov_b32_e32 v1, 0x04030201",
            "v_mov_b32_e32 v2, 0x08070605",
            "v_mov_b32_e32 v3, 0x80000064",
            "s_mov_b32 s4, 0xffffffff",
            "v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[1,0,0] neg_hi:[0,1,0]",
            "v_dot4_i32_i8 v9, v1, v2, v3 neg_lo:[0,0,1]",
            "v_dot4_i32_i8 v10, v1, v2, v3 neg_hi:[0,0,1] clamp",
            "v_dot4_i32_i8 v11, v1, 0x7f7f7f7f, s4 neg_lo:[1,1,1] neg_hi:[1,1,1] clamp",
            "exp mrt0 v8, v9, v10, v11 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_dot4_neg_val", lines)
        self.assertTrue(_validate_spvasm(spv))

    @unittest.skipUnless(HAS_LLVM_MC, "needs llvm-mc with AMDGPU target (set LLVM_MC)")
    def test_llvm_mc_gfx1030_binary_decode_round_trip(self):
        cases = [
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[1,0,0]", ["neg_lo:[1,0,0]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_hi:[1,0,0]", ["neg_hi:[1,0,0]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[0,1,0]", ["neg_lo:[0,1,0]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_hi:[0,1,0]", ["neg_hi:[0,1,0]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[0,0,1]", ["neg_lo:[0,0,1]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_hi:[0,0,1]", ["neg_hi:[0,0,1]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[0,0,1] neg_hi:[0,0,1]",
             ["neg_lo:[0,0,1]", "neg_hi:[0,0,1]"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[1,1,1] neg_hi:[1,1,1] clamp",
             ["neg_lo:[1,1,1]", "neg_hi:[1,1,1]", "clamp"], 8),
            ("v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[0,0,0] neg_hi:[0,0,0]", [], 8),
            ("v_dot4_i32_i8 v8, v1, 0x08070605, s4 neg_lo:[1,0,1] neg_hi:[0,1,1] clamp",
             ["neg_lo:[1,0,1]", "neg_hi:[0,1,1]", "clamp"], 12),
        ]
        asm = "\n".join(c[0] for c in cases) + "\n"
        res = subprocess.run(
            mc_command() + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
            input=asm.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        encoded = []
        for line in res.stdout.decode("utf-8").splitlines():
            if "encoding:" not in line:
                continue
            raw_hex = line.split("encoding:", 1)[1].strip().strip("[]")
            encoded.append(bytes(int(tok, 16) for tok in raw_hex.split(",")))
        self.assertEqual(len(encoded), len(cases))
        for (src_line, expected_mods, expected_size), raw in zip(cases, encoded):
            with self.subTest(inst=src_line):
                self.assertEqual(len(raw), expected_size)
                ins, nbytes = rdna2_decode._decode_one(raw, 0, wave32=True, mcpu="gfx1030")
                self.assertIsNotNone(ins)
                self.assertEqual(nbytes, expected_size)
                self.assertEqual(ins["op"], "v_dot4_i32_i8")
                self.assertEqual(ins["mods"], expected_mods)


if __name__ == "__main__":
    unittest.main()
