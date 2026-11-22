"""--native-f16: f16 add/sub/mul (fma with --fma khr), scalar and packed, as 16-bit float ops.

The mini interpreter has no 16-bit floats, so this checks what is emitted; the values were compared with the integer
path on a GPU over every operand pair (docs/design.md, "Native f16 arithmetic"). The default translation must not
change at all.
"""
import os
import re
import unittest

import support
import rdna2spirv

H = bytes(0x138) + (0x8090).to_bytes(4, "little") + (64).to_bytes(4, "little") + (1).to_bytes(4, "little") * 2
EXIT = ["buffer_store_dword v1, v0, s[0:3], 0 offen", "s_endpgm"]


def translate(lines, native=True, fma_khr=False, float_controls=True):
    old = rdna2spirv.NATIVE_F16, rdna2spirv.FMA_KHR, rdna2spirv.FLOAT_CONTROLS
    rdna2spirv.NATIVE_F16, rdna2spirv.FMA_KHR, rdna2spirv.FLOAT_CONTROLS = native, fma_khr, float_controls
    try:
        spv = support.translate_lines("cs_native_f16", lines + EXIT, fssm=H)
        return spv, rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.NATIVE_F16, rdna2spirv.FMA_KHR, rdna2spirv.FLOAT_CONTROLS = old


def half_ops(spv):
    return re.findall(r"= (OpF\w+|OpFmaKHR) %(?:v2)?half ", spv)


class NativeF16Tests(unittest.TestCase):
    def test_off_by_default_and_unchanged(self):
        lines = ["v_add_f16_e32 v1, v2, v3", "v_pk_mul_f16 v1, v1, v2"]
        self.assertFalse(rdna2spirv.NATIVE_F16)
        default, _ = translate(lines, native=False)
        self.assertNotIn("%half", default)
        self.assertEqual(default, support.translate_lines("cs_native_f16", lines + EXIT, fssm=H))

    def test_scalar_and_packed_ops(self):
        spv, meta = translate(["v_add_f16_e32 v1, v2, v3", "v_sub_f16_e32 v1, v1, v2", "v_subrev_f16_e32 v1, v1, v2",
                               "v_mul_f16_e64 v1, -v1, |v2|", "v_pk_add_f16 v1, v1, v2 op_sel:[1,0] neg_hi:[0,1]",
                               "v_pk_mul_f16 v1, v1, v2"])
        self.assertEqual(half_ops(spv), ["OpFAdd", "OpFSub", "OpFSub", "OpFMul", "OpFAdd", "OpFMul"])
        for line in ("OpCapability Float16", "OpCapability RoundingModeRTE", "OpCapability DenormPreserve",
                     "OpExecutionMode %main RoundingModeRTE 16", "OpExecutionMode %main DenormPreserve 16",
                     "OpExecutionMode %main SignedZeroInfNanPreserve 16"):
            self.assertIn(line, spv)
        self.assertNotIn("OpFNegate", spv)         # neg/abs are applied to the bits
        self.assertIn("shaderFloat16", meta["features"])
        for prop in ("shaderSignedZeroInfNanPreserveFloat16", "shaderRoundingModeRTEFloat16", "shaderDenormPreserveFloat16"):
            self.assertIn(prop, meta["float_controls"])

    def test_fma_needs_fma_khr(self):
        lines = ["v_fma_f16 v1, v2, v3, v4", "v_fmac_f16_e32 v1, v2, v3", "v_pk_fma_f16 v1, v1, v2, v3"]
        spv, _ = translate(lines)
        self.assertEqual(half_ops(spv), [])
        spv, meta = translate(lines, fma_khr=True)
        self.assertEqual(half_ops(spv), ["OpFmaKHR", "OpFmaKHR", "OpFmaKHR"])
        self.assertIn("shaderFmaFloat16", meta["features"])

    def test_integer_path_where_native_is_not_exact(self):
        for line in ("v_add_f16_e64 v1, v2, v3 clamp", "v_mul_f16_e64 v1, v2, v3 mul:2", "v_max_f16_e32 v1, v2, v3"):
            spv, _ = translate([line])
            self.assertEqual(half_ops(spv), [], line)
        # a shader that writes MODE (f16 rounding toward zero) keeps the integer narrowing for every op
        spv, _ = translate(["s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3", "v_add_f16_e32 v1, v2, v3"])
        self.assertEqual(half_ops(spv), [])

    def test_no_float_controls_keeps_the_integer_path(self):
        spv, _ = translate(["v_add_f16_e32 v1, v2, v3"], float_controls=False)
        self.assertEqual(half_ops(spv), [])

    def test_spirv_tools_accept_it(self):
        import shutil, subprocess, tempfile
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            self.skipTest("SPIRV-Tools not installed")
        spv, _ = translate(["v_add_f16_e32 v1, v2, v3", "v_pk_fma_f16 v1, v1, v2, v3", "v_fmac_f16_e32 v1, v2, v3"], fma_khr=True)
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
            with open(a, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            r = subprocess.run(["spirv-as", a, "-o", b, "--target-env", "vulkan1.2"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", b], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
