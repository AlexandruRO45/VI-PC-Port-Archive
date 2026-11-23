"""Regression tests for the vector-as-tuple invariant in tests/semantic/spirv_mini_interp.py.

Checks that:
- Input vector variables load as tuples (default zero, tuple input, or list input) and work with
  vector float arithmetic and Output passthrough in run_fragment / run_workgroup.
- OpConstantComposite of a vector type produces a tuple (while constant arrays remain lists).
- OpAccessChain + OpStore into tuple-backed vector variables (Output, Function, and arrays of
  vectors) updates the backing vector element cleanly instead of raising TypeError.

Standard library only; no GPU or game files.
"""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import spirv_mini_interp as si  # noqa: E402


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class SpirvMiniInterpVectorTests(unittest.TestCase):
    def test_input_vec4_passthrough_default_tuple_and_list(self):
        asm = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_v %out_v
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
     %ptr_in = OpTypePointer Input %v4float
    %ptr_out = OpTypePointer Output %v4float
       %in_v = OpVariable %ptr_in Input
      %out_v = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %val = OpLoad %v4float %in_v
               OpStore %out_v %val
               OpReturn
               OpFunctionEnd
"""
        self.assertEqual(si.run_fragment(asm)["%out_v"], (0, 0, 0, 0))
        expected = tuple(f32_bits(x) for x in (1.0, -2.5, 3.25, 0.5))
        self.assertEqual(si.run_fragment(asm, inputs={"%in_v": (1.0, -2.5, 3.25, 0.5)})["%out_v"], expected)
        self.assertEqual(si.run_fragment(asm, inputs={"%in_v": [1.0, -2.5, 3.25, 0.5]})["%out_v"], expected)

    def test_input_vec4_float_add_and_mul(self):
        asm = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_v %out_v
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
     %ptr_in = OpTypePointer Input %v4float
    %ptr_out = OpTypePointer Output %v4float
       %in_v = OpVariable %ptr_in Input
      %out_v = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %val = OpLoad %v4float %in_v
        %sum = OpFAdd %v4float %val %val
       %prod = OpFMul %v4float %sum %val
               OpStore %out_v %prod
               OpReturn
               OpFunctionEnd
"""
        got = si.run_fragment(asm, inputs={"%in_v": (1.0, -2.0, 0.5, 4.0)})["%out_v"]
        self.assertEqual(got, tuple(f32_bits(x) for x in (2.0, 8.0, 0.5, 32.0)))

    def test_constant_composite_vec4_store_and_arithmetic(self):
        asm = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out_v
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %ptr_out = OpTypePointer Output %v4float
       %f1_0 = OpConstant %float 1.0
       %f2_0 = OpConstant %float 2.0
       %f3_0 = OpConstant %float 3.0
       %f4_0 = OpConstant %float 4.0
        %c_v = OpConstantComposite %v4float %f1_0 %f2_0 %f3_0 %f4_0
      %out_v = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %dbl = OpFAdd %v4float %c_v %c_v
               OpStore %out_v %dbl
               OpReturn
               OpFunctionEnd
"""
        got = si.run_fragment(asm)["%out_v"]
        self.assertEqual(got, tuple(f32_bits(x) for x in (2.0, 4.0, 6.0, 8.0)))

    def test_access_chain_store_into_output_and_array_of_vectors(self):
        asm = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out_v
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %u2_c = OpConstant %uint 2
    %arr_vec = OpTypeArray %v4float %u2_c
    %ptr_out = OpTypePointer Output %v4float
   %ptr_of32 = OpTypePointer Output %float
    %ptr_arr = OpTypePointer Function %arr_vec
   %ptr_ff32 = OpTypePointer Function %float
         %u0 = OpConstant %uint 0
         %u1 = OpConstant %uint 1
         %u2 = OpConstant %uint 2
         %u3 = OpConstant %uint 3
       %f7_5 = OpConstant %float 7.5
      %f11_0 = OpConstant %float 11.0
      %out_v = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %arr = OpVariable %ptr_arr Function
       %p_a1 = OpAccessChain %ptr_ff32 %arr %u1 %u2
               OpStore %p_a1 %f7_5
         %v1 = OpLoad %float %p_a1
       %p_o0 = OpAccessChain %ptr_of32 %out_v %u0
       %p_o3 = OpAccessChain %ptr_of32 %out_v %u3
               OpStore %p_o0 %v1
               OpStore %p_o3 %f11_0
               OpReturn
               OpFunctionEnd
"""
        got = si.run_fragment(asm)["%out_v"]
        self.assertEqual(got, (f32_bits(7.5), 0, 0, f32_bits(11.0)))

    def test_workgroup_input_vec_passthrough(self):
        asm = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %in_u %out_u
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
     %ptr_in = OpTypePointer Input %v3uint
    %ptr_out = OpTypePointer Output %v3uint
       %in_u = OpVariable %ptr_in Input
      %out_u = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %val = OpLoad %v3uint %in_u
               OpStore %out_u %val
               OpReturn
               OpFunctionEnd
"""
        _, res = si.run_workgroup(asm, 2, 32, inputs_for=lambda i: {"%in_u": (i, i + 10, i + 20)},
                                  return_outputs=True)
        self.assertEqual(res[0]["%out_u"], (0, 10, 20))
        self.assertEqual(res[1]["%out_u"], (1, 11, 21))


if __name__ == "__main__":
    unittest.main()
