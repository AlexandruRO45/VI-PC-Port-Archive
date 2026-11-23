"""The pixel shader's SV_Position registers (docs/design.md, "PS system inputs").

POS_X/Y/Z_FLOAT are the pixel centre and z, as Vulkan's FragCoord has them. POS_W_FLOAT is not FragCoord.w: it holds
the clip-space w, so its reciprocal is SV_Position.w. Measured on an AMD Radeon iGPU (gfx1036, Windows driver
26.10.1) with tools/ps_sysvals_probe.py: a vertex w of 2 reads 2.0 where FragCoord.w is 0.5, and across a triangle with
vertex w 1, 2 and 4 the register is 1 / (the interpolated 1/w) to 6e-8, not the interpolated w. Until the change that
added this file the translator wrote FragCoord.w into the register.

The fixture is runtime/test/shaders/ps_pos_w.asm (its header derives the words; the GPU run draws it with a
vertex w of 2). The interpreter runs the translated module with FragCoord given by hand.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402

F32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]    # noqa: E731


def isg1_dxbc(elements):
    """Synthetic DXBC with one ISG1 chunk; elements = (register, mask, rw_mask, system value, component type)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, mask, rw_mask, sysv, ctype in elements:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, 0, sysv, ctype, reg)
        elem[24], elem[25] = mask, rw_mask
        payload += elem
    part = b"ISG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


POSITION_XYZW = isg1_dxbc([(0, 0xF, 0xF, 1, 3)])


def position_w_register(frag_w):
    """What the hardware keeps in POS_W_FLOAT for a FragCoord.w: the clip w, the reciprocal."""
    return 1.0 / frag_w


class PositionRegisters(unittest.TestCase):
    def run_fixture(self, frag_coord):
        spv = support.translate_fixture("ps_pos_w", dxbc=POSITION_XYZW)
        return support.mrt0_words(spv, inputs={"%frag_coord": frag_coord})

    def test_gpu_draw(self):
        # The vs_pos_w draw: pixel (5, 2) has FragCoord (5.5, 2.5, 0.25, 0.5) for a vertex w of 2.
        words = self.run_fixture((5.5, 2.5, 0.25, 0.5))
        self.assertEqual(words, [0x3F000000, 0x3E800000, 0x40000000, 0x3F800000])

    def test_w_register_is_the_reciprocal_of_frag_coord_w(self):
        # out.z is v5 itself. FragCoord.w 0.25 -> w 4; 0.5 -> 2; 1 -> 1; 8 -> 0.125 (exact powers of two).
        for frag_w, reg in ((0.25, 4.0), (0.5, 2.0), (1.0, 1.0), (8.0, 0.125)):
            with self.subTest(frag_w=frag_w):
                self.assertEqual(position_w_register(frag_w), reg)
                words = self.run_fixture((0.5, 0.5, 0.25, frag_w))
                self.assertEqual(words[2], F32(reg))
                self.assertEqual(words[0], F32(frag_w))      # v_rcp_f32 of the register is SV_Position.w again

    def test_perspective_values_measured_on_gfx1036(self):
        # Vertex w 1, 2, 4 and screen weights 0.5, 0.25, 0.25 (hand arithmetic): the interpolated 1/w is
        # 0.5 + 0.125 + 0.0625 = 0.6875, FragCoord.w = 0.6875, and the register is 1 / 0.6875 = 1.4545454...
        # (not the interpolated w, 0.5 + 0.5 + 1.0 = 2.0).
        words = self.run_fixture((0.5, 0.5, 0.25, 0.6875))
        self.assertEqual(words[2], F32(1.0 / 0.6875))
        self.assertNotEqual(words[2], F32(2.0))

    def test_x_y_z_are_frag_coord(self):
        # v2 and v3 pass through (fract of x + y centres), v4 is z.
        words = self.run_fixture((9.5, 63.5, 0.125, 0.5))
        self.assertEqual(words[1], F32(0.125))
        self.assertEqual(words[3], F32(1.0))


if __name__ == "__main__":
    unittest.main()
