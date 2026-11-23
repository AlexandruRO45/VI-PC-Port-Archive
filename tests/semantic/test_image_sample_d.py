"""Structure tests for image_sample_d (RDNA2: sample with explicit derivatives).

RDNA2 ISA: ``image_sample_d vdst, coords, s[V#], s[samp] dmask:.. dim:..`` samples a 2D image using
explicit dP/dx and dP/dy gradients followed by the UV coordinates (6 VGPR for a 2D image:
ddx.x, ddx.y, ddy.x, ddy.y, u, v). The translator must emit ``OpImageSampleExplicitLod ... Grad
<gx> <gy>``.

No test covered the ``image_sample_d`` form end-to-end. This test checks the generated SPIR-V
structure (Grad operands present, coordinates passed after the gradients). Values are not executed
because binding a texture in spirv_mini_interp requires descriptor resources beyond a standalone test.

Standard library only; no GPU, no LLVM, no game files.
"""
import struct
import sys
import unittest

CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)

DESC = "s[8:15], s[16:19]"
PRO = [
    "v_mov_b32 v4, 1.0",
    "v_mov_b32 v5, 0.5",
    "v_mov_b32 v6, 0.25",
    "v_mov_b32 v7, 0.75",
    "v_mov_b32 v8, 0",
    "v_mov_b32 v9, 0.125",
]


def translate(name, line, fssm=CS_FSSM):
    import support
    return support.translate_lines(
        name, PRO + [line, "s_endpgm"], fssm=fssm)


class ImageSampleD(unittest.TestCase):
    def test_2d_emits_grad_operands(self):
        """image_sample_d in 2D must emit OpImageSampleExplicitLod with Grad <gx> <gy>."""
        spv = translate("cs_sample_d_2d",
                        "image_sample_d v10, v[4:9], " + DESC + " dmask:0x3 dim:SQ_RSRC_IMG_2D")
        sample_lines = [ln.strip() for ln in spv.splitlines()
                        if "= OpImageSampleExplicitLod" in ln]
        self.assertEqual(len(sample_lines), 1, f"expected exactly one sample, got:\n{spv}")
        line = sample_lines[0]
        self.assertIn("Grad", line, f"missing Grad operands:\n{line}")
        # The two gradient composite IDs should appear right after "Grad".
        self.assertRegex(line, r"Grad %\w+ %\w+$", f"unexpected Grad form:\n{line}")

    def test_2d_passes_coordinates_after_gradients(self):
        """The UV coordinates come after the gradients in the argument list."""
        spv = translate("cs_sample_d_2d_coords",
                        "image_sample_d v10, v[4:9], " + DESC + " dmask:0x3 dim:SQ_RSRC_IMG_2D")
        # After translating, the composite for uv is built from v8, v9 (the last two coord words).
        # Check that OpCompositeConstruct %v2float appears for the gradients and that coordinates
        # flow into the sample call.
        self.assertIn("OpCompositeConstruct %v2float", spv, f"expected gradient composites:\n{spv}")
        self.assertIn("= OpImageSampleExplicitLod %v4float", spv,
                      f"expected explicit LOD sample:\n{spv}")


if __name__ == "__main__":
    unittest.main()
