"""The optimizer must keep result types when folding comparisons and integer identities.

These complete synthetic fragment modules exercise the standalone SPIR-V pass.
They do not establish that these operand/result type combinations occur in current RDNA2 translations.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spirv_interp  # noqa: E402
from test_spirv_opt import _validate, spirv_opt  # noqa: E402


def comparison_module(op, components, *, same=True, constant_scalar=False):
    logical = op in ("OpLogicalAnd", "OpLogicalOr")
    declarations = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        'OpEntryPoint Fragment %main "main" %out_color',
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %out_color Location 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%uint = OpTypeInt 32 0",
        "%v4uint = OpTypeVector %uint 4",
        "%out_ptr = OpTypePointer Output %v4uint",
        "%reg_ptr = OpTypePointer Function %uint",
        "%zero = OpConstant %uint 0",
        "%one = OpConstant %uint 1",
        "%true = OpConstantTrue %bool",
        "%false = OpConstantFalse %bool",
    ]
    if components > 1:
        declarations.append(f"%vbool = OpTypeVector %bool {components}")
        if not logical and components != 4:
            declarations.append(f"%vuint = OpTypeVector %uint {components}")
    body = [
        "%out_color = OpVariable %out_ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        # The existing interpreter also returns these named register variables.
        # Volatile loads keep them present in both the raw and optimized modules.
        "%vgpr = OpVariable %reg_ptr Function %zero",
        "%sgpr = OpVariable %reg_ptr Function %zero",
        "%keep_v = OpLoad %uint %vgpr Volatile",
        "%keep_s = OpLoad %uint %sgpr Volatile",
    ]
    if components > 1:
        values = ["%true" if i % 2 == 0 else "%false" for i in range(components)] if logical else [
            "%one" if i % 2 == 0 else "%zero" for i in range(components)
        ]
        source_type = "%vbool" if logical else "%v4uint" if components == 4 else "%vuint"
        body.append(f"%lhs = OpCompositeConstruct {source_type} {' '.join(values)}")
        if not same:
            body.append(f"%rhs = OpCompositeConstruct {source_type} " + " ".join(["%zero"] * components))
        lhs, rhs, result_type = "%lhs", "%lhs" if same else "%rhs", "%vbool"
    else:
        lhs = "%one" if constant_scalar else "%keep_v"
        rhs, result_type = lhs, "%bool"
    body.append(f"%cmp = {op} {result_type} {lhs} {rhs}")
    channels = []
    for i in range(components):
        condition = "%cmp"
        if components > 1:
            condition = f"%bit{i}"
            body.append(f"{condition} = OpCompositeExtract %bool %cmp {i}")
        channels.append(f"%value{i}")
        body.append(f"%value{i} = OpSelect %uint {condition} %one %zero")
    channels += ["%zero"] * (4 - components)
    body += [
        f"%rgba = OpCompositeConstruct %v4uint {' '.join(channels)}",
        "OpStore %out_color %rgba",
        "OpReturn",
        "OpFunctionEnd",
    ]
    return "\n".join(declarations + body) + "\n"


def integer_identity_module(op, source_type, result_type):
    rhs = "%value" if op == "OpBitwiseAnd" else "%zero"
    return f'''OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_value
OpExecutionMode %main OriginUpperLeft
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%out_ptr = OpTypePointer Output {result_type}
%reg_ptr = OpTypePointer Function {source_type}
%zero = OpConstant {source_type} 0
%seed = OpConstant {source_type} 305419896
%out_value = OpVariable %out_ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
%vgpr = OpVariable %reg_ptr Function %seed
%sgpr = OpVariable %reg_ptr Function %zero
%value = OpLoad {source_type} %vgpr Volatile
%keep_s = OpLoad {source_type} %sgpr Volatile
%result = {op} {result_type} %value {rhs}
OpStore %out_value %result
OpReturn
OpFunctionEnd
'''


class ResultTypeTests(unittest.TestCase):
    def check_module(self, source, expected, output="out_color"):
        expected = tuple(expected) if isinstance(expected, list) else expected
        _validate(source)
        self.assertEqual(spirv_interp.run(source).outputs[output], expected)
        optimized = spirv_opt.optimize(source)
        _validate(optimized)
        self.assertEqual(spirv_interp.run(optimized).outputs[output], expected)
        return optimized

    def test_vector_comparison_identities_preserve_boolean_vectors(self):
        for op, value in (("OpIEqual", 1), ("OpINotEqual", 0)):
            for components in (2, 3, 4):
                with self.subTest(op=op, components=components):
                    self.check_module(comparison_module(op, components),
                                      [value] * components + [0] * (4 - components))

    def test_vector_comparisons_preserve_mixed_component_results(self):
        for op in ("OpIEqual", "OpINotEqual"):
            for components in (2, 3, 4):
                expected = [int((i % 2 == 1) == (op == "OpIEqual")) for i in range(components)]
                with self.subTest(op=op, components=components):
                    self.check_module(comparison_module(op, components, same=False),
                                      expected + [0] * (4 - components))

    def test_vector_logical_identities_keep_boolean_vectors(self):
        for op in ("OpLogicalAnd", "OpLogicalOr"):
            for components in (2, 3, 4):
                with self.subTest(op=op, components=components):
                    self.check_module(comparison_module(op, components),
                                      [int(i % 2 == 0) for i in range(components)] + [0] * (4 - components))

    def test_scalar_comparison_identities_still_fold(self):
        for op, value in (("OpIEqual", 1), ("OpINotEqual", 0)):
            with self.subTest(op=op):
                optimized = self.check_module(comparison_module(op, 1), [value, 0, 0, 0])
                self.assertNotIn(f"= {op} %bool", optimized)

    def test_scalar_constant_comparisons_still_fold(self):
        for op, value in (("OpIEqual", 1), ("OpINotEqual", 0)):
            with self.subTest(op=op):
                optimized = self.check_module(comparison_module(op, 1, constant_scalar=True), [value, 0, 0, 0])
                self.assertNotIn(f"= {op} %bool", optimized)

    def test_integer_identities_preserve_result_signedness(self):
        for op in ("OpIAdd", "OpISub", "OpBitwiseOr", "OpBitwiseAnd"):
            for source, result in (("%int", "%uint"), ("%uint", "%int")):
                with self.subTest(op=op, source=source, result=result):
                    self.check_module(integer_identity_module(op, source, result), 305419896, "out_value")

    def test_same_type_integer_identities_still_fold(self):
        for op in ("OpIAdd", "OpISub", "OpBitwiseOr", "OpBitwiseAnd"):
            for same_type in ("%int", "%uint"):
                with self.subTest(op=op, type=same_type):
                    optimized = self.check_module(integer_identity_module(op, same_type, same_type),
                                                  305419896, "out_value")
                    self.assertNotIn(f"= {op} ", optimized)


if __name__ == "__main__":
    unittest.main()
