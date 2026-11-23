"""cs_w32_frexp_ldexp_f64: the GPU fixture for v_frexp_mant_f64, v_frexp_exp_i32_f64 and v_ldexp_f64, run on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_frexp_ldexp_f64.asm (derivations in its header) runs:
- v_frexp_mant_f64 and v_frexp_exp_i32_f64 on a negative binary64 subnormal crossing the 32-bit dword boundary
  (|-0x00000001_00000001| -> mantissa 0x3FE00000_00100000, exponent -1041 = 0xFFFFFBEF under denorm_mode_16_64=3);
- v_frexp_mant_f64 with VOP3 -v and mul:2 on normal 3.5 (-0.875 = 0xBFEC0000_20000000, mul:2 ignored under
  denorm_mode_16_64=3) and v_frexp_exp_i32_f64 on 3.5 (exponent 2);
- v_frexp_mant_f64 with VOP3 -v and v_frexp_exp_i32_f64 on a signalling NaN (0x7FF00000_12345678 -> payload
  preserved as 0xFFF00000_12345678 under IEEE=0, exponent 0);
- v_ldexp_f64 normalizing a binary64 subnormal input (0x00080000_00000001, S1 = 1024 -> 0x40000000_00000002);
- v_ldexp_f64 subnormal output round-to-nearest-even at an exact half-ULP tie (0x3FF00001_80000000, S1 = -1042 ->
  sh = 20 rounds odd quotient 0x00000001_000017FF to even 0x00000001_00001800) and rounding carry out of bit 51 into
  -min_norm (0xBFFFFFFF_FFFFFFFF, S1 = -1023 -> 0x80100000_00000000);
- v_ldexp_f64 with VOP3 -v, |v|, DX10 clamp ([0.0, 1.0]), and a 32-bit RDNA2 literal in bits [63:32] (0x40080000, 1
  -> 6.0 = 0x40180000_00000000).

This test runs the fixture as one wave32 workgroup through spirv_mini_interp, checks the words against a Python
reference model, verifies that the integer-only lowering emits neither OpCapability Float64 nor OpCapability Int64,
checks a negative control where binary64 subnormals are flushed instead of normalized, and checks that check.py and
scripts/run_gpu_tests.py expect the same words.
"""
import math
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_alu_frexp_f64 import ref_exp, ref_mant  # noqa: E402
from test_alu_ldexp_f64 import ref_ldexp_f64  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)  # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE
LANES = 32


def lo32(u):
    return u & 0xFFFFFFFF


def hi32(u):
    return (u >> 32) & 0xFFFFFFFF


def reference_words():
    # 1. Subnormal |v| and normal -v mul:2 (mul:2 ignored under denorm_mode_16_64=3)
    sub_in = 0x8000000100000001
    sub_abs = sub_in & 0x7FFFFFFFFFFFFFFF
    m_sub = ref_mant(sub_abs)
    e_sub = ref_exp(sub_abs)

    norm_in = 0x400C000020000000
    m_norm = ref_mant(norm_in ^ 0x8000000000000000)
    e_norm = ref_exp(norm_in)

    out0 = lo32(m_sub) ^ lo32(m_norm)
    out1 = hi32(m_sub) ^ hi32(m_norm)
    out2 = ((e_norm << 16) | e_sub) & 0xFFFFFFFF

    # 2. Signalling NaN with -v preserves payload under IEEE=0 and returns exponent 0
    snan_in = 0x7FF0000012345678
    m_snan = ref_mant(snan_in ^ 0x8000000000000000)
    e_snan = ref_exp(snan_in)
    out3 = (lo32(m_snan) ^ hi32(m_snan)) | e_snan

    # 3. v_ldexp_f64 normalizing a binary64 subnormal input
    ld_sub = ref_ldexp_f64(0x0008000000000001, 1024)
    out4 = lo32(ld_sub)
    out5 = hi32(ld_sub)

    # 4. v_ldexp_f64 subnormal output round-to-nearest-even tie + carry into -min_norm + abs/neg/clamp + 32-bit literal
    ld_tie = ref_ldexp_f64(0x3FF0000180000000, -1042)
    ld_carry = ref_ldexp_f64(0xBFFFFFFFFFFFFFFF, -1023)
    out6 = lo32(ld_tie) | lo32(ld_carry)

    three = 0x4008000000000000
    ld_neg_cl = ref_ldexp_f64(three, 1, mod="neg", clamp=True)
    ld_abs_cl = ref_ldexp_f64(three, 1, mod="abs", clamp=True)
    ld_lit = ref_ldexp_f64(three, 1)
    clamp_mix = (
        lo32(ld_neg_cl)
        | hi32(ld_neg_cl)
        | lo32(ld_abs_cl)
        | lo32(ld_lit)
        | (hi32(ld_abs_cl) ^ hi32(ld_lit))
    )
    out7 = (hi32(ld_tie) ^ hi32(ld_carry)) | clamp_mix
    return [out0, out1, out2, out3, out4, out5, out6, out7]


EXPECTED = reference_words()


class FrexpLdexpF64Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(
            EXPECTED,
            [0x20100000, 0x800C0000, 0xFFFFFBEF, 0xEDC45678, 0x00000002, 0x40000000, 0x00001800, 0xFFF80001],
        )
        self.assertFalse(math.isnan(struct.unpack("<d", struct.pack("<Q", 0x400C000020000000))[0]))

    def test_translated_shader_and_capabilities(self):
        spv = support.translate_fixture("cs_w32_frexp_ldexp_f64", fssm=FSSM)
        self.assertNotIn("OpCapability Float64", spv)
        self.assertNotIn("%double", spv)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(
            spv,
            LANES,
            LANES,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_flushing_subnormals_corrupts_frexp_and_ldexp_words(self):
        # Negative control: if the subnormal branches (OpLogicalAnd %bool %tN %tM where %tN is OpIEqual e == 0)
        # evaluate to %false (as under denorm_mode_16_64=0), out[0], out[1], out[2], out[4], and out[5] all diverge.
        spv = support.translate_fixture("cs_w32_frexp_ldexp_f64", fssm=FSSM)
        mutated = re.sub(r"OpExtInst %int %glsl FindUMsb %\w+", "OpBitcast %int %c2", spv)
        self.assertNotEqual(mutated, spv)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(
            mutated,
            LANES,
            LANES,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        for k in (0, 2, 4, 5):
            with self.subTest(word=k):
                self.assertNotEqual(mem[4 * k], EXPECTED[k])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_frexp_ldexp_f64"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
