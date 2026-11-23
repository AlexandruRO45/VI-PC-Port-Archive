"""Repeated bit-mask spellings consume one parameter list per numeric flag.

The Python assembler is compared with canonical complete modules and, when
available, SPIRV-Tools. Current translator emission of repeated flags is not
established; these are synthetic assembly compatibility regressions.
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_spirv_asm import SPIRV_AS, _validate, spirv_asm  # noqa: E402


def memory_module(body):
    return f"""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%ptr = OpTypePointer Function %uint
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%main = OpFunction %void None %fn
%entry = OpLabel
%src = OpVariable %ptr Function %one
%dst = OpVariable %ptr Function %zero
{body}
OpReturn
OpFunctionEnd
"""


def visible_module(load_flags, store_flags):
    return f"""OpCapability Shader
OpCapability VulkanMemoryModel
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical Vulkan
OpEntryPoint GLCompute %main "main" %buffer
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %buffer DescriptorSet 0
OpDecorate %buffer Binding 0
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%block = OpTypeStruct %uint
%ptr_block = OpTypePointer StorageBuffer %block
%ptr_uint = OpTypePointer StorageBuffer %uint
%zero = OpConstant %uint 0
%scope = OpConstant %uint 2
%buffer = OpVariable %ptr_block StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%p = OpAccessChain %ptr_uint %buffer %zero
%r = OpLoad %uint %p {load_flags} 4 %scope
OpStore %p %r {store_flags} 4 %scope
OpReturn
OpFunctionEnd
"""


def loop_module(flags):
    return f"""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%fn = OpTypeFunction %void
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%four = OpConstant %uint 4
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %loop
%loop = OpLabel
%i = OpPhi %uint %zero %entry %next %cont
%test = OpULessThan %bool %i %four
OpLoopMerge %merge %cont {flags} 4 4 2
OpBranchConditional %test %body %merge
%body = OpLabel
OpBranch %cont
%cont = OpLabel
%next = OpIAdd %uint %i %one
OpBranch %loop
%merge = OpLabel
OpReturn
OpFunctionEnd
"""


class RepeatedFlagTests(unittest.TestCase):
    def reference(self, asm):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "module.spv")
            result = subprocess.run(
                [SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", path],
                input=asm, encoding="utf-8", errors="replace", capture_output=True,
            )
            data = None
            if result.returncode == 0:
                with open(path, "rb") as source:
                    data = source.read()
            return result, data

    def assert_matches(self, asm, canonical):
        expected = spirv_asm.assemble(canonical)
        _validate(expected)
        if SPIRV_AS:
            result, reference = self.reference(asm)
            self.assertEqual(result.returncode, 0, result.stderr)
            _validate(reference)
            # The generator word is the only intentional assembler difference.
            self.assertEqual(reference[:8] + reference[12:], expected[:8] + expected[12:])
        actual = spirv_asm.assemble(asm)
        self.assertEqual(actual, expected)
        _validate(actual)

    def test_repeated_alignment_on_load_store_and_copy(self):
        for body in (
            "%r = OpLoad %uint %src Aligned|Aligned 4",
            "OpStore %dst %one Aligned|Aligned 4",
            "OpCopyMemory %dst %src Aligned|Aligned 4 Aligned|Aligned 4",
            "%r = OpLoad %uint %src Nontemporal|Aligned|Volatile|Aligned 4",
        ):
            with self.subTest(body=body):
                canonical = body.replace("Aligned|Aligned", "Aligned").replace(
                    "Nontemporal|Aligned|Volatile|Aligned", "Volatile|Aligned|Nontemporal")
                self.assert_matches(memory_module(body), memory_module(canonical))

    def test_memory_flag_aliases_share_parameters(self):
        canonical = visible_module(
            "Aligned|MakePointerVisible|NonPrivatePointer",
            "Aligned|MakePointerAvailable|NonPrivatePointer",
        )
        for visible, available in (
            ("MakePointerVisibleKHR|Aligned|MakePointerVisible",
             "MakePointerAvailable|MakePointerAvailableKHR|Aligned"),
            ("MakePointerVisible|MakePointerVisibleKHR|Aligned|MakePointerVisibleKHR",
             "Aligned|MakePointerAvailableKHR|MakePointerAvailable|MakePointerAvailableKHR"),
        ):
            with self.subTest(visible=visible, available=available):
                self.assert_matches(visible_module(
                    "NonPrivatePointerKHR|" + visible,
                    "NonPrivatePointer|NonPrivatePointerKHR|" + available,
                ), canonical)

    def test_loop_parameters_keep_numeric_flag_order(self):
        canonical = loop_module("MinIterations|MaxIterations|IterationMultiple")
        for flags in (
            "IterationMultiple|MinIterations|MaxIterations|MinIterations|IterationMultiple",
            "MaxIterations|IterationMultiple|MaxIterations|MinIterations",
        ):
            with self.subTest(flags=flags):
                self.assert_matches(loop_module(flags), canonical)

    def test_parameterless_repetitions_and_single_flags(self):
        for flags, canonical in (
            ("Aligned 4", "Aligned 4"),
            ("Volatile|Volatile", "Volatile"),
            ("Nontemporal|Aligned|Volatile|Volatile 4", "Volatile|Aligned|Nontemporal 4"),
            ("None|Aligned|None 4", "Aligned 4"),
        ):
            with self.subTest(flags=flags):
                self.assert_matches(
                    memory_module("%r = OpLoad %uint %src " + flags),
                    memory_module("%r = OpLoad %uint %src " + canonical),
                )

    def test_missing_and_excess_parameters_are_rejected(self):
        for body in (
            "%r = OpLoad %uint %src Aligned|Aligned",
            "%r = OpLoad %uint %src Aligned|Aligned 4 4",
            "OpStore %dst %one Aligned|Aligned 4 4",
            "OpCopyMemory %dst %src Aligned 4 Aligned|Aligned 4 4",
        ):
            with self.subTest(body=body):
                asm = memory_module(body)
                if SPIRV_AS:
                    result, _ = self.reference(asm)
                    self.assertNotEqual(result.returncode, 0)
                with self.assertRaises(spirv_asm.SpirvAsmError):
                    spirv_asm.assemble(asm)


if __name__ == "__main__":
    unittest.main()
