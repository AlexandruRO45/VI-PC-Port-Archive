"""buffer_store_format_* through the V# FORMAT stays small and keeps its values (SHADER-774 compile time).

A typed store whose format is only known at run time used to encode and store the element in a branch for each of
the 41 formats in Gen.FORMATS (about 2,000 SPIR-V instructions, a dozen of those branches with atomics). The NVIDIA
driver inlines everything, so a shader with 22 of them took 30 s to compile; the module is now one function per
(components, d16) that encodes every format with plain arithmetic, selects the words of the matching format and
stores once. These tests keep the shape (counts) and the stored words (synthetic V#s, no game data).
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_buffer_store_rdna2 as base  # noqa: E402

probe, si = base.probe, base.si
BASE = base.BASE
FILL = [0xDEAD0000 + i for i in range(5)]
SEL = probe.SEL_XYZW


def _lines(inst, code, vals, sel=SEL):
    d16 = "_d16_" in inst
    n = len(inst.rsplit("_", 1)[1])
    n_regs = (n + 1) // 2 if d16 else n
    vreg = "v4" if n_regs == 1 else f"v[4:{3 + n_regs}]"
    out = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
           f"s_mov_b32 s3, {probe.vsharp_d3(code, sel):#x}", "v_mov_b32 v1, 0"]
    out += [f"v_mov_b32 v{4 + i}, {vals[i]:#x}" for i in range(n_regs)]
    out.append(f"{inst} {vreg}, v1, s[0:3], 0 offen")
    return out


def _store(inst, code, vals, sel=SEL):
    """The five dwords at BASE after one store; they start as FILL."""
    spv = base._translate(_lines(inst, code, vals, sel) + ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"])
    mem = {BASE + 4 * i: v for i, v in enumerate(FILL)}
    si.run_fragment(spv, memory=mem)
    return [mem[BASE + 4 * i] for i in range(5)]


def f32(x):
    import struct
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Shape(unittest.TestCase):
    def test_one_function_per_form_and_one_atomic_pair(self):
        lines = ["s_mov_b32 s0, 0x1000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100", "s_mov_b32 s3, 0x31001000",
                 "v_mov_b32 v1, 0", "v_mov_b32 v4, 0", "v_mov_b32 v5, 0", "v_mov_b32 v6, 0", "v_mov_b32 v7, 0"]
        lines += ["buffer_store_format_xyzw v[4:7], v1, s[0:3], 0 offen"] * 6
        lines += ["buffer_store_format_d16_xyzw v[4:5], v1, s[0:3], 0 offen"] * 3
        lines += ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
        spv = base._translate(lines)
        self.assertEqual(spv.count("= OpFunction %void None %fn_fmt_store_"), 2, "one function per (components, d16) form")
        self.assertEqual(spv.count("= OpFunctionCall %void %fmt_store_fn_"), 9)
        self.assertEqual(spv.count("OpAtomicOr"), 2)         # the 1- and 2-byte formats merge in one place each
        self.assertEqual(spv.count("OpAtomicAnd"), 2)
        self.assertEqual(spv.count("Aligned 4"), 8)         # four dword stores in each function

    def test_forms_with_one_signature_share_its_type(self):
        # (3 components) and (2 components, d16) both take ten parameters: a second OpTypeFunction of the same
        # signature fails spirv-val
        lines = ["s_mov_b32 s0, 0x1000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100", "s_mov_b32 s3, 0x31001000",
                 "v_mov_b32 v1, 0", "v_mov_b32 v4, 0", "v_mov_b32 v5, 0", "v_mov_b32 v6, 0",
                 "buffer_store_format_xyz v[4:6], v1, s[0:3], 0 offen",
                 "buffer_store_format_d16_xy v[4:5], v1, s[0:3], 0 offen",
                 "exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
        spv = base._translate(lines)
        self.assertEqual(spv.count("= OpFunction %void None %fn_fmt_store_"), 2)
        types = [ln.split(" = ", 1)[1] for ln in spv.splitlines() if " = OpTypeFunction " in ln]
        self.assertEqual(len(types), len(set(types)), "duplicate OpTypeFunction")

    def test_each_further_store_is_a_call(self):
        def size(count):
            lines = ["s_mov_b32 s0, 0x1000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100", "s_mov_b32 s3, 0x31001000",
                     "v_mov_b32 v1, 0", "v_mov_b32 v4, 0"]
            lines += ["buffer_store_format_x v4, v1, s[0:3], 0 offen"] * count
            lines += ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
            return base._translate(lines).count("\n")
        self.assertLess(size(9) - size(1), 8 * 60)           # was about 2,000 lines per store


class Words(unittest.TestCase):
    """The words each kind of format leaves in memory; whatever a format does not cover keeps its value."""

    def test_whole_dwords(self):
        a, b, c, d = f32(1.0), f32(-2.5), 0x12345678, 0
        self.assertEqual(_store("buffer_store_format_xyzw", 77, (a, b, c, d)), [a, b, c, d, FILL[4]])
        self.assertEqual(_store("buffer_store_format_xyz", 74, (a, b, c)), [a, b, c, FILL[3], FILL[4]])
        self.assertEqual(_store("buffer_store_format_xy", 62, (a, b), probe.SEL_XY01), [a, b] + FILL[2:])
        self.assertEqual(_store("buffer_store_format_x", 22, (a,)), [a] + FILL[1:])

    def test_packed_dwords(self):
        one, zero = f32(1.0), 0
        self.assertEqual(_store("buffer_store_format_xyzw", 56, (one, zero, one, zero))[0], 0x00FF00FF)
        self.assertEqual(_store("buffer_store_format_xy", 29, (f32(1.0), f32(-2.0)), probe.SEL_XY01)[0], 0xC0003C00)
        self.assertEqual(_store("buffer_store_format_xyzw", 71, (f32(1.0), f32(-2.0), f32(0.5), f32(4.0)))[:3],
                         [0xC0003C00, 0x44003800, FILL[2]])

    def test_sub_dword_elements_keep_their_neighbours(self):
        self.assertEqual(_store("buffer_store_format_x", 1, (f32(1.0),))[0], 0xDEAD00FF)           # 8_UNORM
        self.assertEqual(_store("buffer_store_format_x", 7, (f32(1.0),))[0], 0xDEADFFFF)           # 16_UNORM
        self.assertEqual(_store("buffer_store_format_x", 6, (0xFFFFFFF0,))[0], 0xDEAD00F0)         # 8_SINT -16
        self.assertEqual(_store("buffer_store_format_x", 5, (0x1FF,))[0], 0xDEAD00FF)              # 8_UINT saturates
        self.assertEqual(_store("buffer_store_format_x", 12, (0x8000,))[0], 0xDEAD7FFF)            # 16_SINT clamps

    def test_d16_sources(self):
        # two f16 values per register, widened for the f32 formats, kept as 16-bit integers for the integer ones
        self.assertEqual(_store("buffer_store_format_d16_xy", 64, (0xC0003C00,), probe.SEL_XY01)[:2],
                         [f32(1.0), f32(-2.0)])
        self.assertEqual(_store("buffer_store_format_d16_xy", 62, (0xFFFF0003,), probe.SEL_XY01)[:2], [3, 0xFFFF])
        self.assertEqual(_store("buffer_store_format_d16_xy", 63, (0xFFFF0003,), probe.SEL_XY01)[:2],
                         [3, 0xFFFFFFFF])

    def test_formats_it_does_not_decode_store_nothing(self):
        for code in (0, 100):
            self.assertEqual(_store("buffer_store_format_xyzw", code, (1, 2, 3, 4)), FILL)


if __name__ == "__main__":
    unittest.main()
