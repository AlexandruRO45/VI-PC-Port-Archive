"""Hand-written SPIR-V checks for the loop/barrier workgroup interpreter's call frames."""
import unittest

from test_loop_ballot_barrier import NonUniformBarrierError, run_compute_workgroup


def module(main, helpers=""):
    return """
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%fn_uint = OpTypePointer Function %uint
%out_uint = OpTypePointer PhysicalStorageBuffer %uint
%in_uint = OpTypePointer Input %uint
%main_type = OpTypeFunction %void
%outer_type = OpTypeFunction %uint %fn_uint
%increment_type = OpTypeFunction %void %fn_uint
%identity_type = OpTypeFunction %uint %uint
%barrier_type = OpTypeFunction %void
%local_index = OpVariable %in_uint Input
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c7 = OpConstant %uint 7
%main = OpFunction %void None %main_type
%start = OpLabel
""" + main + "\nOpReturn\nOpFunctionEnd\n" + helpers


class FunctionCallTests(unittest.TestCase):
    def test_main_without_calls(self):
        self.assertEqual(run_compute_workgroup(module(""), 1, 1), {})

    def test_nested_calls_preserve_locals_and_return_values(self):
        asm = module("""
%local = OpVariable %fn_uint Function %c7
%first = OpFunctionCall %uint %outer %local
%second = OpFunctionCall %uint %outer %local
%sum = OpIAdd %uint %first %second
%p = OpConvertUToPtr %out_uint %c0
OpStore %p %sum
""", """
%outer = OpFunction %uint None %outer_type
%ptr = OpFunctionParameter %fn_uint
%outer_entry = OpLabel
%scratch = OpVariable %fn_uint Function %c1
%ignored = OpFunctionCall %void %increment %ptr
%value = OpLoad %uint %ptr
%initial = OpLoad %uint %scratch
%result = OpIAdd %uint %value %initial
OpStore %scratch %result
OpReturnValue %result
OpFunctionEnd
%increment = OpFunction %void None %increment_type
%dest = OpFunctionParameter %fn_uint
%increment_entry = OpLabel
%old = OpLoad %uint %dest
%next = OpIAdd %uint %old %c1
OpStore %dest %next
OpReturn
OpFunctionEnd
""")
        self.assertEqual(run_compute_workgroup(asm, 1, 1), {0: 19})

    def test_barrier_at_callsite(self):
        asm = module("""
%value = OpFunctionCall %uint %identity %c7
OpControlBarrier %c2 %c2 %c0
%p = OpConvertUToPtr %out_uint %c0
OpStore %p %value
""", self.identity())
        self.assertEqual(run_compute_workgroup(asm, 4, 2), {0: 7})

    def test_wrong_argument_count(self):
        asm = module("%value = OpFunctionCall %uint %identity", self.identity())
        with self.assertRaisesRegex(ValueError, "Wrong argument count for %identity"):
            run_compute_workgroup(asm, 1, 1)

    def test_unknown_function(self):
        with self.assertRaises(KeyError):
            run_compute_workgroup(module("%value = OpFunctionCall %uint %missing"), 1, 1)

    def test_barrier_in_helper_distinguishes_callers(self):
        asm = module("""
%lane = OpLoad %uint %local_index
%is_zero = OpIEqual %bool %lane %c0
OpSelectionMerge %merge None
OpBranchConditional %is_zero %left %right
%left = OpLabel
%a = OpFunctionCall %void %barrier
OpBranch %merge
%right = OpLabel
%b = OpFunctionCall %void %barrier
OpBranch %merge
%merge = OpLabel
""", """
%barrier = OpFunction %void None %barrier_type
%barrier_entry = OpLabel
OpControlBarrier %c2 %c2 %c0
OpReturn
OpFunctionEnd
""")
        with self.assertRaises(NonUniformBarrierError):
            run_compute_workgroup(asm, 2, 2)

    @staticmethod
    def identity():
        return """
%identity = OpFunction %uint None %identity_type
%arg = OpFunctionParameter %uint
%identity_entry = OpLabel
OpReturnValue %arg
OpFunctionEnd
"""


if __name__ == "__main__":
    unittest.main()
