"""image_store with d16 (RDNA2 MIMG, D16): the store data is 16 bits per component.

RDNA2 ISA, MIMG: with D16 set the data VGPRs hold 16-bit components, two per VGPR, the enabled components (dmask)
in order from the low half of vDATA. The float view (UNORM/SNORM/FLOAT formats, which convert from the shader's
float) reads each as an f16, widened exactly to f32; the integer views take the 16 bits (zero-extended for UINT,
sign-extended for SINT). Also with a16 (16-bit coordinates, two per VGPR), the form that was refused as
"image_store 2D a16 d16".

The generated SPIR-V's data path (VGPR loads, bit-field extracts, UnpackHalf2x16, bitcasts) is evaluated here with
a small interpreter and the three texels the instruction writes (float, uint and sint view) are compared with a
plain-Python reference. The mini SPIR-V interpreter has no image support.
Run with: python -I -m unittest discover -s tests/semantic -p test_image_store_d16.py
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools")
sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402

M = 0xFFFFFFFF


def f16_to_f32_bits(h: int) -> int:
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", h))[0]))[0]


def sext16(x: int) -> int:
    x &= 0xFFFF
    return (x - 0x10000 if x & 0x8000 else x) & M


def run_store(line: str, vgprs: dict[int, int]) -> dict[str, list[int]]:
    ins = parse_line(line)
    g = rdna2spirv.Gen("cs")
    dim = next(m for m in ins["mods"] if m.startswith("dim:"))
    g.image_store(ins["args"], ins["mods"], dim)
    val = {name: v for v, name in g.consts.items()}       # const id -> value
    reg_of: dict[str, int] = {}
    out: dict[str, list[int]] = {}

    def step(res, op, ty, args):
        if op == "OpAccessChain" and args[0] == "%vgpr":
            reg_of[res] = val[args[1]]
        elif op == "OpLoad" and args[0] in reg_of:
            val[res] = vgprs.get(reg_of[args[0]], 0)
        elif op == "OpBitcast" and args[0] in val:
            val[res] = val[args[0]] & M
        elif op in ("OpBitFieldUExtract", "OpBitFieldSExtract"):
            x, off, cnt = val[args[0]], val[args[1]], val[args[2]]
            f = (x >> off) & ((1 << cnt) - 1)
            val[res] = (sext16(f) if op == "OpBitFieldSExtract" else f) & M
        elif op == "OpExtInst" and args[1] == "UnpackHalf2x16":
            x = val[args[2]]
            val[res] = (f16_to_f32_bits(x & 0xFFFF), f16_to_f32_bits(x >> 16 & 0xFFFF))
        elif op == "OpCompositeExtract" and args[0] in val and isinstance(val[args[0]], tuple):
            val[res] = val[args[0]][int(args[1])]
        elif op == "OpCompositeConstruct" and ty in ("%v4float", "%v4uint_st", "%v4int_st"):
            out[ty] = [val[x] & M for x in args]
    for ln in g.body:
        w = ln.split()
        if len(w) < 4 or w[1] != "=":
            continue
        res, op, ty, args = w[0], w[2], w[3], w[4:]
        try:
            step(res, op, ty, args)
        except KeyError:        # descriptor lookup, coordinates and control flow: not part of the data path
            pass

    return out


class ImageStoreD16(unittest.TestCase):
    def check(self, line, vgprs, f16s):
        """f16s: the expected component f16 bit patterns (None = not enabled: 0)."""
        got = run_store(line, vgprs)
        self.assertEqual(got["%v4float"], [0 if h is None else f16_to_f32_bits(h) for h in f16s])
        self.assertEqual(got["%v4uint_st"], [0 if h is None else h for h in f16s])
        self.assertEqual(got["%v4int_st"], [0 if h is None else sext16(h) for h in f16s])

    def test_single_component_low_half(self):
        # a16 d16, dmask 1, the form seen in a real program: R is vD[15:0]; the high half is ignored.
        self.check("image_store v10, v5, s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 d16",
                   {10: 0xBEEF3C00, 5: 0x00030004}, [0x3C00, None, None, None])

    def test_two_components_share_one_vgpr(self):
        self.check("image_store v4, v[0:1], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16",
                   {4: 0xC0003800, 0: 1, 1: 2}, [0x3800, 0xC000, None, None])

    def test_four_components_use_two_vgprs(self):
        self.check("image_store v[4:5], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16",
                   {4: 0x40003C00, 5: 0x8001FC00, 0: 1, 1: 2}, [0x3C00, 0x4000, 0xFC00, 0x8001])

    def test_sparse_dmask_packs_enabled_components(self):
        # dmask 0x5 (R, B): B is the second enabled component, so the high half of vD.
        self.check("image_store v4, v[0:1], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm d16",
                   {4: 0x7BFF0001, 0: 1, 1: 2}, [0x0001, None, 0x7BFF, None])

    def test_three_components(self):
        self.check("image_store v[4:5], v[0:1], s[8:15] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16",
                   {4: 0x40003C00, 5: 0xDEAD3555, 0: 1, 1: 2}, [0x3C00, 0x4000, 0x3555, None])

    def test_without_d16_the_words_are_whole(self):
        got = run_store("image_store v[4:5], v[0:1], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
                        {4: 0x11111111, 5: 0x22222222, 0: 1, 1: 2})
        self.assertEqual(got["%v4uint_st"], [0x11111111, 0x22222222, 0, 0])


if __name__ == "__main__":
    unittest.main()
