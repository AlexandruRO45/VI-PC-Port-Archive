"""Guards for the tools/spirv_opt.py hot-path gates and fast paths.

The optimizer skips work by opcode (which instructions can a rule touch?), so a new rule that is not added to the
matching opcode set would silently stop firing. These tests pin the sets to the rules, and pin the output of a small
hand-written module (not game data), so a speed-up cannot change the text that comes out.
"""
import inspect
import os
import re
import sys
import unittest
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import spirv_opt  # noqa: E402

# Every opcode a rule in process_block / _fold_int_op / the DCE can name appears quoted in its source; each must be in
# the set that gates the rule.
_OP_RE = re.compile(r'"(Op[A-Za-z0-9]+)"')


def _quoted_ops(source: str) -> set[str]:
    return set(_OP_RE.findall(source))


def _slice(source: str, start: str, end: str) -> str:
    i = source.index(start)
    return source[i:source.index(end, i)]


_MODULE = (
    '               OpCapability Shader\n'
    '          %1 = OpExtInstImport "GLSL.std.450"\n'
    '               OpMemoryModel Logical GLSL450\n'
    '               OpEntryPoint GLCompute %main "main"\n'
    '               OpExecutionMode %main LocalSize 1 1 1\n'
    '       %void = OpTypeVoid\n'
    '       %uint = OpTypeInt 32 0\n'
    '       %bool = OpTypeBool\n'
    '     %uint_0 = OpConstant %uint 0\n'
    '     %uint_1 = OpConstant %uint 1\n'
    '     %uint_2 = OpConstant %uint 2\n'
    '     %uint_3 = OpConstant %uint 3\n'
    '       %true = OpConstantTrue %bool\n'
    '      %false = OpConstantFalse %bool\n'
    '%_ptr_Function_uint = OpTypePointer Function %uint\n'
    '%_arr_uint_2 = OpTypeArray %uint %uint_2\n'
    '%_ptr_Function_arr = OpTypePointer Function %_arr_uint_2\n'
    '       %null = OpConstantNull %_arr_uint_2\n'
    '         %fn = OpTypeFunction %void\n'
    '       %fnp = OpTypeFunction %void %_ptr_Function_uint\n'
    '       %sink = OpFunction %void None %fnp\n'
    '       %param = OpFunctionParameter %_ptr_Function_uint\n'
    ' %sink_entry = OpLabel\n'
    '               OpReturn\n'
    '               OpFunctionEnd\n'
    '       %main = OpFunction %void None %fn\n'
    '      %entry = OpLabel\n'
    '       %regs = OpVariable %_ptr_Function_arr Function %null\n'
    '    %escaped = OpVariable %_ptr_Function_uint Function\n'
    '       %slot = OpAccessChain %_ptr_Function_uint %regs %uint_0\n'
    '      %slot2 = OpAccessChain %_ptr_Function_uint %regs %uint_0\n'
    '               OpStore %slot %uint_3\n'
    '         %a = OpLoad %uint %slot2\n'
    '         %b = OpIAdd %uint %a %uint_0\n'
    '         %c = OpIAdd %uint %uint_1 %uint_2\n'
    '         %d = OpIAdd %uint %b %c\n'
    '         %e = OpIAdd %uint %c %b\n'
    '         %f = OpIMul %uint %d %e\n'
    '         %g = OpCopyObject %uint %f\n'
    '         %h = OpISub %uint %g %g\n'
    '         %i = OpIEqual %bool %h %uint_0\n'
    '         %j = OpSelect %uint %i %uint_1 %uint_2\n'
    '         %dead = OpIMul %uint %j %j\n'
    '               OpStore %escaped %j\n'
    '               OpBranch %next\n'
    '       %next = OpLabel\n'
    '         %k = OpLoad %uint %slot\n'
    '         %l = OpIAdd %uint %k %j\n'
    '               OpStore %escaped %l\n'
    '         %m = OpFunctionCall %void %sink %escaped\n'
    '               OpReturn\n'
    '               OpFunctionEnd\n'
)

_EXPECTED = (
    '               OpCapability Shader\n'
    '          %1 = OpExtInstImport "GLSL.std.450"\n'
    '               OpMemoryModel Logical GLSL450\n'
    '               OpEntryPoint GLCompute %main "main"\n'
    '               OpExecutionMode %main LocalSize 1 1 1\n'
    '       %void = OpTypeVoid\n'
    '       %uint = OpTypeInt 32 0\n'
    '       %bool = OpTypeBool\n'
    '     %uint_0 = OpConstant %uint 0\n'
    '     %uint_1 = OpConstant %uint 1\n'
    '     %uint_2 = OpConstant %uint 2\n'
    '     %uint_3 = OpConstant %uint 3\n'
    '       %true = OpConstantTrue %bool\n'
    '      %false = OpConstantFalse %bool\n'
    '%_ptr_Function_uint = OpTypePointer Function %uint\n'
    '%_arr_uint_2 = OpTypeArray %uint %uint_2\n'
    '%_ptr_Function_arr = OpTypePointer Function %_arr_uint_2\n'
    '       %null = OpConstantNull %_arr_uint_2\n'
    '         %fn = OpTypeFunction %void\n'
    '       %fnp = OpTypeFunction %void %_ptr_Function_uint\n'
    '       %sink = OpFunction %void None %fnp\n'
    '         %param = OpFunctionParameter %_ptr_Function_uint\n'
    ' %sink_entry = OpLabel\n'
    '         OpReturn\n'
    '               OpFunctionEnd\n'
    '       %main = OpFunction %void None %fn\n'
    '      %entry = OpLabel\n'
    '    %escaped = OpVariable %_ptr_Function_uint Function\n'
    '         OpStore %escaped %uint_1\n'
    '         OpBranch %next\n'
    '       %next = OpLabel\n'
    '         %l = OpIAdd %uint %uint_3 %uint_1\n'
    '         OpStore %escaped %l\n'
    '         %m = OpFunctionCall %void %sink %escaped\n'
    '         OpReturn\n'
    '               OpFunctionEnd\n'
)


class SpirvOptGateTests(unittest.TestCase):
    def setUp(self):
        self.fn_src = inspect.getsource(spirv_opt._optimize_function)
        self.block_src = _slice(self.fn_src, "def process_block", "        if blocks:\n")

    def test_process_block_opcodes_are_gated_in(self):
        # OpLabel is skipped before the gate on purpose; everything else a rule names must pass the gate.
        named = _quoted_ops(self.block_src) - {"OpLabel"}
        self.assertEqual(named - spirv_opt._BLOCK_OPS, set(),
                         "a rule in process_block names an opcode that _BLOCK_OPS filters out before it runs")

    def test_simplify_block_opcodes_are_gated_in(self):
        simplify = _slice(self.block_src, "if plain and op in _SIMPLIFY_OPS:", "if plain and op in _FOLD_OPS:")
        named = _quoted_ops(simplify)
        self.assertEqual(named - spirv_opt._SIMPLIFY_OPS, set(),
                         "a rule in the copy/select/logical block names an opcode _SIMPLIFY_OPS does not list")

    def test_fold_opcodes_are_gated_in(self):
        fold_src = inspect.getsource(spirv_opt._fold_int_op)
        self.assertEqual(_quoted_ops(fold_src) - spirv_opt._FOLD_OPS, set(),
                         "_fold_int_op folds an opcode that _FOLD_OPS keeps it from being asked about")

    def test_dce_candidate_opcodes_are_gated_in(self):
        dce = _slice(self.fn_src, "def is_pure_dce_candidate", "dce_dead: set[int]")
        self.assertEqual(_quoted_ops(dce) - spirv_opt._DCE_OPS, set())
        self.assertTrue(spirv_opt._PURE_OPS <= spirv_opt._DCE_OPS)

    def test_escape_scan_opcodes(self):
        # Loads, stores and access chains are scanned by hand for escapes, every other instruction generically.
        self.assertTrue(spirv_opt._MEMORY_OPS <= spirv_opt._NOT_GENERIC_OPS)
        self.assertEqual(spirv_opt._NOT_GENERIC_OPS - spirv_opt._MEMORY_OPS, {"OpVariable"})

    def test_fold_gate_matches_fold_function(self):
        # No opcode outside _FOLD_OPS may ever fold, whatever its operands are.
        types = {"%u": (32, 0)}
        consts = {"%c0": ("%u", 0, 32, 0), "%c1": ("%u", 1, 32, 0), "%c3": ("%u", 3, 32, 0),
                  "%cmax": ("%u", 0xFFFFFFFF, 32, 0)}
        by_val = {("%u", v): f"%v{v}" for v in range(0, 70)}
        by_val[("%u", 0xFFFFFFFF)] = "%vmax"
        ops = sorted(spirv_opt._PURE_OPS | spirv_opt._FOLD_OPS | {"OpLoad", "OpPhi", "OpCopyObject"})
        operands = ["%c0", "%c1", "%c3", "%cmax", "%x", "%y"]
        folded = set()
        for op in ops:
            for arity in (2, 3, 4):
                for combo in product(operands, repeat=arity - 1):
                    args = ["%u"] + list(combo)
                    r = spirv_opt._fold_int_op(op, "%u", args, consts, by_val, types, "%true", "%false")
                    if r is not None:
                        folded.add(op)
        self.assertTrue(folded <= spirv_opt._FOLD_OPS, folded - spirv_opt._FOLD_OPS)
        # and the gate is not vacuous: most listed opcodes do fold for some operands
        self.assertGreaterEqual(len(folded), 18)

    def test_fold_fast_reject_keeps_every_rule(self):
        types = {"%u": (32, 0)}
        consts = {"%c0": ("%u", 0, 32, 0), "%c1": ("%u", 1, 32, 0)}
        by_val = {("%u", 0): "%c0", ("%u", 1): "%c1"}
        fold = spirv_opt._fold_int_op
        # two unrelated non-constants: nothing to fold
        for op in sorted(spirv_opt._FOLD_OPS):
            self.assertIsNone(fold(op, "%u", ["%u", "%x", "%y"], consts, by_val, types, "%true", "%false"), op)
        # identical operands and one constant operand still fold
        self.assertEqual(fold("OpISub", "%u", ["%u", "%x", "%x"], consts, by_val, types, "%true", "%false"), "%c0")
        self.assertEqual(fold("OpIEqual", "%u", ["%bool", "%x", "%x"], consts, by_val, types, "%true", "%false"),
                         "%true")
        self.assertEqual(fold("OpIAdd", "%u", ["%u", "%x", "%c0"], consts, by_val, types, "%true", "%false"), "%x")
        self.assertEqual(fold("OpIMul", "%u", ["%u", "%c1", "%y"], consts, by_val, types, "%true", "%false"), "%y")

    def test_drop_indices_keeps_order(self):
        items = list("abcdef")
        self.assertEqual(spirv_opt._drop_indices(items, {0, 3, 5}), ["b", "c", "e"])
        self.assertEqual(spirv_opt._drop_indices(items, set()), items)
        self.assertEqual(items, list("abcdef"))


class SpirvOptGoldenTests(unittest.TestCase):
    def test_hand_written_module_output_is_pinned(self):
        # Store forwarding across a block edge, an escaped variable, constant folding, CSE, copy removal and DCE.
        self.assertEqual(spirv_opt.optimize(_MODULE), _EXPECTED)

    def test_pass_limit_is_respected(self):
        once = spirv_opt.optimize(_MODULE, max_passes=1)
        self.assertNotEqual(once, _MODULE)
        self.assertEqual(spirv_opt.optimize(_MODULE, max_passes=8), _EXPECTED)

    def test_escaped_variable_stores_survive(self):
        out = spirv_opt.optimize(_MODULE)
        self.assertIn("OpFunctionCall %void %sink %escaped", out)
        self.assertEqual(out.count("OpStore %escaped"), 2)


if __name__ == "__main__":
    unittest.main()
