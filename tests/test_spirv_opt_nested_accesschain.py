"""Regression tests for out-of-order nested OpAccessChain tracking in tools/spirv_opt.py.

In SPIR-V, only the entry block must come first in a function; a dominated block
containing `%p2 = OpAccessChain ... %p1 ...` may appear earlier in module text
order than the dominating block defining `%p1 = OpAccessChain ... %var ...`.

These tests verify that tools/spirv_opt.py resolves nested OpAccessChain chains
to a fixed point (regardless of block text order and chain depth) so:
  - stores through `%p2` invalidate aliased slots instead of forwarding stale values,
  - loads through `%p2` keep live stores to aliased slots from being deleted,
  - dynamic indices on `%p2` mark the root variable dynamically indexed,
  - mixed-depth accesses through `%p2` mark the root variable escaped, and
  - readonly global (PushConstant / Input) nested OpAccessChains still participate in CSE.

Standard library only; no GPU or game data.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import spirv_asm  # noqa: E402
import spirv_opt  # noqa: E402


def _assemble(spv: str) -> bytes:
    return spirv_asm.assemble(spv, target_env="vulkan1.2")


def _extract_store_to_outp(spv: str) -> list[str]:
    vals = []
    for raw in spv.splitlines():
        toks = raw.strip().split()
        if len(toks) >= 3 and toks[0] == "OpStore" and toks[1] == "%outp":
            vals.append(toks[2])
    return vals


class OutOfOrderNestedAccessChainTests(unittest.TestCase):
    def test_store_through_earlier_nested_chain_invalidates_direct_slot(self):
        """%p2 = OpAccessChain %p1 %c0 (in earlier block %use) overwrites %arr[0][0] with 42;
        %v = OpLoad %p_direct must NOT forward the stale 99 from %setup."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%c42 = OpConstant %uint 42
%c99 = OpConstant %uint 99
%arr1 = OpTypeArray %uint %c2
%arr2 = OpTypeArray %arr1 %c2
%null2 = OpConstantNull %arr2
%ptr_arr2 = OpTypePointer Function %arr2
%ptr_arr1 = OpTypePointer Function %arr1
%ptr_uint = OpTypePointer Function %uint
%buf = OpTypeStruct %uint
%ptr_buf = OpTypePointer StorageBuffer %buf
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%out = OpVariable %ptr_buf StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%arr = OpVariable %ptr_arr2 Function %null2
OpBranch %setup
%use = OpLabel
%p2 = OpAccessChain %ptr_uint %p1 %c0
OpStore %p2 %c42
%v = OpLoad %uint %p_direct
%outp = OpAccessChain %ptr_sb_uint %out %c0
OpStore %outp %v
OpReturn
%setup = OpLabel
%p1 = OpAccessChain %ptr_arr1 %arr %c0
%p_direct = OpAccessChain %ptr_uint %arr %c0 %c0
OpStore %p_direct %c99
OpBranch %use
OpFunctionEnd
"""
        _assemble(spv)
        opt = spirv_opt.optimize(spv)
        _assemble(opt)
        stored = _extract_store_to_outp(opt)
        self.assertEqual(stored, ["%c42"], f"stale value forwarded instead of %c42:\n{opt}")

    def test_load_through_earlier_nested_chain_keeps_live_store(self):
        """%v = OpLoad %p2 (where %p2 is in earlier block %use) reads %arr[0][0] written in %setup;
        whole-function dead-store elimination must NOT delete OpStore %p_direct %c99."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%c99 = OpConstant %uint 99
%arr1 = OpTypeArray %uint %c2
%arr2 = OpTypeArray %arr1 %c2
%null2 = OpConstantNull %arr2
%ptr_arr2 = OpTypePointer Function %arr2
%ptr_arr1 = OpTypePointer Function %arr1
%ptr_uint = OpTypePointer Function %uint
%buf = OpTypeStruct %uint
%ptr_buf = OpTypePointer StorageBuffer %buf
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%out = OpVariable %ptr_buf StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%arr = OpVariable %ptr_arr2 Function %null2
OpBranch %setup
%use = OpLabel
%p2 = OpAccessChain %ptr_uint %p1 %c0
%v = OpLoad %uint %p2
%outp = OpAccessChain %ptr_sb_uint %out %c0
OpStore %outp %v
OpReturn
%setup = OpLabel
%p1 = OpAccessChain %ptr_arr1 %arr %c0
%p_direct = OpAccessChain %ptr_uint %arr %c0 %c0
OpStore %p_direct %c99
OpBranch %use
OpFunctionEnd
"""
        _assemble(spv)
        opt = spirv_opt.optimize(spv)
        _assemble(opt)
        # Either forwarded cleanly to %c99, or the OpStore %p_direct %c99 + OpLoad survive intact.
        stored = _extract_store_to_outp(opt)
        self.assertEqual(len(stored), 1)
        if stored[0] != "%c99":
            self.assertIn("OpStore %p_direct %c99", opt, f"live store was dropped:\n{opt}")

    def test_dynamic_store_through_earlier_nested_chain_prevents_stale_forwarding(self):
        """A dynamic store through %p_dyn = OpAccessChain %p1 %idx (earlier in text than %p1)
        must invalidate cached slots of %arr and prevent folding OpLoad %p_direct to %c99."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %in_buf %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %in_buf DescriptorSet 0
OpDecorate %in_buf Binding 0
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 1
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%c42 = OpConstant %uint 42
%c99 = OpConstant %uint 99
%arr1 = OpTypeArray %uint %c2
%arr2 = OpTypeArray %arr1 %c2
%null2 = OpConstantNull %arr2
%ptr_arr2 = OpTypePointer Function %arr2
%ptr_arr1 = OpTypePointer Function %arr1
%ptr_uint = OpTypePointer Function %uint
%buf = OpTypeStruct %uint
%ptr_buf = OpTypePointer StorageBuffer %buf
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%in_buf = OpVariable %ptr_buf StorageBuffer
%out = OpVariable %ptr_buf StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%arr = OpVariable %ptr_arr2 Function %null2
%inp = OpAccessChain %ptr_sb_uint %in_buf %c0
%idx = OpLoad %uint %inp
OpBranch %setup
%use = OpLabel
%p_dyn = OpAccessChain %ptr_uint %p1 %idx
OpStore %p_dyn %c42
%v = OpLoad %uint %p_direct
%outp = OpAccessChain %ptr_sb_uint %out %c0
OpStore %outp %v
OpReturn
%setup = OpLabel
%p1 = OpAccessChain %ptr_arr1 %arr %c0
%p_direct = OpAccessChain %ptr_uint %arr %c0 %c0
OpStore %p_direct %c99
OpBranch %use
OpFunctionEnd
"""
        _assemble(spv)
        opt = spirv_opt.optimize(spv)
        _assemble(opt)
        stored = _extract_store_to_outp(opt)
        self.assertEqual(len(stored), 1)
        self.assertNotEqual(stored[0], "%c99", f"dynamic nested store failed to invalidate slot:\n{opt}")
        self.assertIn("OpLoad %uint %p_direct", opt, f"load must survive dynamic store:\n{opt}")
        self.assertIn("OpStore %p_direct %c99", opt, f"initial store must survive dynamic load/store:\n{opt}")

    def test_three_level_reverse_order_nested_chain(self):
        """Three nested OpAccessChain steps in reverse text order must resolve to a fixed point."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%c7 = OpConstant %uint 7
%c88 = OpConstant %uint 88
%arr1 = OpTypeArray %uint %c2
%arr2 = OpTypeArray %arr1 %c2
%arr3 = OpTypeArray %arr2 %c2
%ptr_arr3 = OpTypePointer Function %arr3
%ptr_arr2 = OpTypePointer Function %arr2
%ptr_arr1 = OpTypePointer Function %arr1
%ptr_uint = OpTypePointer Function %uint
%buf = OpTypeStruct %uint
%ptr_buf = OpTypePointer StorageBuffer %buf
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%out = OpVariable %ptr_buf StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%arr = OpVariable %ptr_arr3 Function
OpBranch %b1
%b3 = OpLabel
%p3 = OpAccessChain %ptr_uint %p2 %c0
OpStore %p3 %c7
%v = OpLoad %uint %p_direct
%outp = OpAccessChain %ptr_sb_uint %out %c0
OpStore %outp %v
OpReturn
%b2 = OpLabel
%p2 = OpAccessChain %ptr_arr1 %p1 %c0
OpBranch %b3
%b1 = OpLabel
%p1 = OpAccessChain %ptr_arr2 %arr %c0
%p_direct = OpAccessChain %ptr_uint %arr %c0 %c0 %c0
OpStore %p_direct %c88
OpBranch %b2
OpFunctionEnd
"""
        _assemble(spv)
        opt = spirv_opt.optimize(spv)
        _assemble(opt)
        stored = _extract_store_to_outp(opt)
        self.assertEqual(stored, ["%c7"], f"3-level reverse-order chain forwarded stale value:\n{opt}")

    def test_mixed_depth_through_earlier_nested_chain_escapes_variable(self):
        """A sub-array OpStore via an earlier nested OpAccessChain (depth 2) mixed with an element
        OpLoad (depth 3) must mark the root variable escaped rather than forwarding a stale element."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%c55 = OpConstant %uint 55
%c99 = OpConstant %uint 99
%arr1 = OpTypeArray %uint %c2
%arr2 = OpTypeArray %arr1 %c2
%arr3 = OpTypeArray %arr2 %c2
%row55 = OpConstantComposite %arr1 %c55 %c55
%ptr_arr3 = OpTypePointer Function %arr3
%ptr_arr2 = OpTypePointer Function %arr2
%ptr_arr1 = OpTypePointer Function %arr1
%ptr_uint = OpTypePointer Function %uint
%buf = OpTypeStruct %uint
%ptr_buf = OpTypePointer StorageBuffer %buf
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%out = OpVariable %ptr_buf StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%arr = OpVariable %ptr_arr3 Function
OpBranch %setup
%use = OpLabel
%p_row = OpAccessChain %ptr_arr1 %p1 %c0
OpStore %p_row %row55
%v = OpLoad %uint %p_elem
%outp = OpAccessChain %ptr_sb_uint %out %c0
OpStore %outp %v
OpReturn
%setup = OpLabel
%p1 = OpAccessChain %ptr_arr2 %arr %c0
%p_elem = OpAccessChain %ptr_uint %arr %c0 %c0 %c0
OpStore %p_elem %c99
OpBranch %use
OpFunctionEnd
"""
        _assemble(spv)
        opt = spirv_opt.optimize(spv)
        _assemble(opt)
        stored = _extract_store_to_outp(opt)
        self.assertEqual(len(stored), 1)
        self.assertNotEqual(stored[0], "%c99", f"mixed-depth nested store forwarded stale %c99:\n{opt}")
        self.assertIn("OpStore %p_row %row55", opt)
        self.assertIn("OpLoad %uint %p_elem", opt)

    def test_readonly_global_earlier_nested_chain_cse(self):
        """Duplicate OpLoads through an earlier-in-text nested PushConstant OpAccessChain still CSE."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %pc %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %arr_pc ArrayStride 4
OpDecorate %PC Block
OpMemberDecorate %PC 0 Offset 0
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%arr_pc = OpTypeArray %uint %c2
%PC = OpTypeStruct %arr_pc
%ptr_PC = OpTypePointer PushConstant %PC
%ptr_arr_pc = OpTypePointer PushConstant %arr_pc
%ptr_pc_u = OpTypePointer PushConstant %uint
%buf = OpTypeStruct %uint
%ptr_buf = OpTypePointer StorageBuffer %buf
%ptr_sb_u = OpTypePointer StorageBuffer %uint
%pc = OpVariable %ptr_PC PushConstant
%out = OpVariable %ptr_buf StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %setup
%use = OpLabel
%p2 = OpAccessChain %ptr_pc_u %p1 %c0
%a = OpLoad %uint %p2
%b = OpLoad %uint %p2
%sum = OpIAdd %uint %a %b
%outp = OpAccessChain %ptr_sb_u %out %c0
OpStore %outp %sum
OpReturn
%setup = OpLabel
%p1 = OpAccessChain %ptr_arr_pc %pc %c0
OpBranch %use
OpFunctionEnd
"""
        _assemble(spv)
        opt = spirv_opt.optimize(spv)
        _assemble(opt)
        self.assertEqual(opt.count("OpLoad %uint %p2"), 1, f"expected PushConstant load CSE:\n{opt}")


if __name__ == "__main__":
    unittest.main()
