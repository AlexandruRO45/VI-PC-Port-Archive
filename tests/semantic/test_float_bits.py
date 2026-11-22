"""Interpreter tests for exact float bit preservation (OpBitcast and sNaN, #509).

A Python float cannot hold a signalling NaN (converting to/from double quiets it,
setting bit 22), so the interpreter keeps float values as their 32-bit pattern
and only widens to a Python float inside arithmetic. These tests check that
OpBitcast and the float outputs round-trip sNaN words bit-exactly.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402

# Signalling-NaN patterns (exponent all ones, fraction nonzero, bit 22 clear).
SNAN_WORDS = [0x7F800001, 0x7FBFFFFF, 0xFF800001, 0xFFBFFFFF, 0xFFA3A095, 0x7F80ABCD]

# A fragment shader that round-trips a uint constant through float and back and
# writes it to mrt0. `{bits}` is the decimal form of the 32-bit pattern.
_BITCAST_SHADER = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
OpDecorate %out_mrt0 Location 0
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v4float = OpTypeVector %float 4
%ptr = OpTypePointer Output %v4float
%out_mrt0 = OpVariable %ptr Output
%u = OpConstant %uint {bits}
%main = OpFunction %void None %fnty
%entry = OpLabel
%f = OpBitcast %float %u
%u2 = OpBitcast %uint %f
%f2 = OpBitcast %float %u2
%c = OpCompositeConstruct %v4float %f2 %f2 %f2 %f2
OpStore %out_mrt0 %c
OpReturn
OpFunctionEnd
"""


class FloatBitsTests(unittest.TestCase):
    def test_bits_f32_round_trip_preserves_snan(self):
        for b in SNAN_WORDS:
            x = si.bits_f32(b)
            self.assertEqual(x.bits, b & 0xFFFFFFFF)
            self.assertEqual(si.f32_bits(x), b & 0xFFFFFFFF)

    def test_opbitcast_round_trip_preserves_snan(self):
        for b in SNAN_WORDS:
            out = si.run_fragment(_BITCAST_SHADER.format(bits=b & 0xFFFFFFFF))
            self.assertEqual(list(out["%out_mrt0"]), [b & 0xFFFFFFFF] * 4)


if __name__ == "__main__":
    unittest.main()
