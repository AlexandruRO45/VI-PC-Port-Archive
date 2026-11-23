"""Tests for helper function specialization in tools/spirv_metal_set0.py.

Standard library only; uses synthetic SPIR-V disassembly, no game data and no external SPIRV-Tools binaries.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import spirv_metal_set0 as m  # noqa: E402


def _assert_valid_calls_and_types(testcase, text):
    """Validate declaration order, OpFunctionParameter types, and OpFunctionCall arg types."""
    lines = text.splitlines()
    defs = {}
    def_line = {}
    fntys = {}
    val_ty = {}

    for idx, ln in enumerate(lines):
        m_def = re.match(r"^\s*(%\S+) = (.+)$", ln)
        if not m_def:
            continue
        rid, rhs = m_def.group(1), m_def.group(2)
        testcase.assertNotIn(rid, defs, f"duplicate definition {rid}")
        defs[rid] = rhs
        def_line[rid] = idx

        m_ft = re.match(r"^OpTypeFunction (%\S+)(.*)$", rhs)
        if m_ft:
            fntys[rid] = (m_ft.group(1), m_ft.group(2).split())
            testcase.assertIn(m_ft.group(1), def_line, f"{rid} uses undeclared return type {m_ft.group(1)}")
            for pt in m_ft.group(2).split():
                testcase.assertIn(pt, def_line, f"{rid} uses undeclared param type {pt}")

        m_var = re.match(r"^OpVariable (%\S+) \S+$", rhs)
        if m_var:
            val_ty[rid] = m_var.group(1)
        m_fn = re.match(r"^OpFunction (%\S+) \S+ (%\S+)$", rhs)
        if m_fn:
            testcase.assertIn(m_fn.group(2), fntys, f"{rid} references missing OpTypeFunction {m_fn.group(2)}")
            val_ty[rid] = m_fn.group(2)
        m_p = re.match(r"^OpFunctionParameter (%\S+)$", rhs)
        if m_p:
            val_ty[rid] = m_p.group(1)

    # Propagate OpCopyObject types
    changed = True
    while changed:
        changed = False
        for rid, rhs in defs.items():
            m_cp = re.match(r"^OpCopyObject (%\S+) (%\S+)$", rhs)
            if m_cp and rid not in val_ty:
                val_ty[rid] = m_cp.group(1)
                changed = True
            if m_cp and m_cp.group(2) in val_ty:
                testcase.assertEqual(m_cp.group(1), val_ty[m_cp.group(2)],
                                     f"OpCopyObject {rid} type {m_cp.group(1)} != src type {val_ty[m_cp.group(2)]}")

    # Check OpFunction parameters match OpTypeFunction
    i = 0
    while i < len(lines):
        m_fn = re.match(r"^\s*(%\S+) = OpFunction (%\S+) \S+ (%\S+)$", lines[i])
        if not m_fn:
            i += 1
            continue
        fn_id, ft_id = m_fn.group(1), m_fn.group(3)
        _, expected_pts = fntys[ft_id]
        actual_pts = []
        i += 1
        while i < len(lines) and lines[i].strip() != "OpFunctionEnd":
            m_p = re.match(r"^\s*(%\S+) = OpFunctionParameter (%\S+)$", lines[i])
            if m_p:
                actual_pts.append(m_p.group(2))
            m_call = re.match(r"^\s*%\S+ = OpFunctionCall (%\S+) (%\S+)((?:\s+%\S+)*)$", lines[i])
            if m_call:
                ret_ty, callee, args = m_call.group(1), m_call.group(2), m_call.group(3).split()
                testcase.assertIn(callee, val_ty, f"call to unknown callee {callee}")
                c_ret, c_pts = fntys[val_ty[callee]]
                testcase.assertEqual(ret_ty, c_ret, f"call return type mismatch for {callee}")
                testcase.assertEqual(len(args), len(c_pts), f"call arg count mismatch for {callee}")
                for arg, expected_pty in zip(args, c_pts):
                    if arg in val_ty:
                        testcase.assertEqual(val_ty[arg], expected_pty,
                                             f"call to {callee}: arg {arg} has type {val_ty[arg]}, param expects {expected_pty}")
            i += 1
        testcase.assertEqual(actual_pts, expected_pts, f"{fn_id} OpFunctionParameter types != {ft_id}")
        i += 1


HEADER = """\
OpCapability Shader
OpCapability RuntimeDescriptorArray
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %images0 %images3 %samplers %foreign_images
OpExecutionMode %main OriginUpperLeft
OpDecorate %images0 DescriptorSet 0
OpDecorate %images0 Binding 0
OpDecorate %samplers DescriptorSet 0
OpDecorate %samplers Binding 1
OpDecorate %images3 DescriptorSet 0
OpDecorate %images3 Binding 3
OpDecorate %foreign_images DescriptorSet 1
OpDecorate %foreign_images Binding 0
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
%smp = OpTypeSampler
%rt_img = OpTypeRuntimeArray %img2d
%rt_smp = OpTypeRuntimeArray %smp
%ptr_img = OpTypePointer UniformConstant %rt_img
%ptr_smp = OpTypePointer UniformConstant %rt_smp
%elem_img_ptr = OpTypePointer UniformConstant %img2d
%elem_smp_ptr = OpTypePointer UniformConstant %smp
%zero = OpConstant %uint 0
%fn_void = OpTypeFunction %void
%fn_smp = OpTypeFunction %void %ptr_smp
%fn_img = OpTypeFunction %void %ptr_img
%images0 = OpVariable %ptr_img UniformConstant
%samplers = OpVariable %ptr_smp UniformConstant
%images3 = OpVariable %ptr_img UniformConstant
%foreign_images = OpVariable %ptr_img UniformConstant
"""


class SpirvMetalSet0HelpersTest(unittest.TestCase):
    def setUp(self):
        self.counts = m.capacities()

    def test_single_helper_called_with_resized_array(self):
        spv = HEADER + """\
%helper = OpFunction %void None %fn_smp
%p_smp = OpFunctionParameter %ptr_smp
%h_entry = OpLabel
%sp = OpAccessChain %elem_smp_ptr %p_smp %zero
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
%c1 = OpFunctionCall %void %helper %samplers
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        self.assertIn("%helper_msl = OpFunction", out)
        self.assertIn("%c1 = OpFunctionCall %void %helper_msl %samplers", out)

    def test_polymorphic_helper_two_set0_lengths_and_untouched_set1(self):
        spv = HEADER + """\
%helper = OpFunction %void None %fn_img
%p_img = OpFunctionParameter %ptr_img
%h_entry = OpLabel
%ip = OpAccessChain %elem_img_ptr %p_img %zero
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
%c0 = OpFunctionCall %void %helper %images0
%c3 = OpFunctionCall %void %helper %images3
%cf = OpFunctionCall %void %helper %foreign_images
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        m0 = re.search(r"%c0 = OpFunctionCall %void (%\S+) %images0", out)
        m3 = re.search(r"%c3 = OpFunctionCall %void (%\S+) %images3", out)
        mf = re.search(r"%cf = OpFunctionCall %void (%\S+) %foreign_images", out)
        self.assertNotEqual(m0.group(1), m3.group(1))
        self.assertEqual(mf.group(1), "%helper")

    def test_dont_inline_helper_preserved(self):
        spv = HEADER + """\
%helper = OpFunction %void DontInline %fn_smp
%p_smp = OpFunctionParameter %ptr_smp
%h_entry = OpLabel
%sp = OpAccessChain %elem_smp_ptr %p_smp %zero
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
%c1 = OpFunctionCall %void %helper %samplers
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        self.assertRegex(out, r"%helper_msl = OpFunction %void DontInline %\S+")

    def test_unused_helper_calling_helper_with_resized_global(self):
        spv = HEADER + """\
%leaf = OpFunction %void None %fn_smp
%p_smp = OpFunctionParameter %ptr_smp
%l_entry = OpLabel
%sp = OpAccessChain %elem_smp_ptr %p_smp %zero
OpReturn
OpFunctionEnd
%unused = OpFunction %void None %fn_void
%u_entry = OpLabel
%cu = OpFunctionCall %void %leaf %samplers
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        self.assertIn("%cu = OpFunctionCall %void %leaf_msl %samplers", out)

    def test_chained_helpers_and_copy_object_inside_helper(self):
        spv = HEADER + """\
%leaf = OpFunction %void None %fn_smp
%p_leaf = OpFunctionParameter %ptr_smp
%l_entry = OpLabel
%sp = OpAccessChain %elem_smp_ptr %p_leaf %zero
OpReturn
OpFunctionEnd
%mid = OpFunction %void None %fn_smp
%p_mid = OpFunctionParameter %ptr_smp
%mid_entry = OpLabel
%cp = OpCopyObject %ptr_smp %p_mid
%cm = OpFunctionCall %void %leaf %cp
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
%c1 = OpFunctionCall %void %mid %samplers
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        self.assertIn("%mid_msl = OpFunction", out)
        self.assertIn("%leaf_msl = OpFunction", out)

    def test_duplicate_call_sites_reuse_single_clone(self):
        spv = HEADER + """\
%helper = OpFunction %void None %fn_smp
%p_smp = OpFunctionParameter %ptr_smp
%h_entry = OpLabel
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
%c1 = OpFunctionCall %void %helper %samplers
%c2 = OpFunctionCall %void %helper %samplers
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        self.assertEqual(len(re.findall(r"%helper_msl\S* = OpFunction ", out)), 1)
        self.assertIn("%c1 = OpFunctionCall %void %helper_msl %samplers", out)
        self.assertIn("%c2 = OpFunctionCall %void %helper_msl %samplers", out)

    def test_recursive_helper_with_resized_array(self):
        spv = HEADER + """\
%rec = OpFunction %void None %fn_smp
%p_smp = OpFunctionParameter %ptr_smp
%r_entry = OpLabel
%cr = OpFunctionCall %void %rec %p_smp
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn_void
%m_entry = OpLabel
%c1 = OpFunctionCall %void %rec %samplers
OpReturn
OpFunctionEnd
"""
        out = m.rewrite(spv, self.counts)
        _assert_valid_calls_and_types(self, out)
        self.assertIn("%rec_msl = OpFunction", out)
        self.assertIn("= OpFunctionCall %void %rec_msl %p_smp_msl", out)


if __name__ == "__main__":
    unittest.main()
