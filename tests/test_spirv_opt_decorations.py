"""Keep direct and grouped annotations valid while optimizing complete SPIR-V modules.

Group tests cover the standalone optimizer and an equivalent grouping of a real
translation's decorations. They do not claim that the translator currently emits groups.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_spirv_opt import _validate, spirv_opt, support  # noqa: E402


GROUP = "%group = OpDecorationGroup\nOpDecorate %group RelaxedPrecision\n"


def module(annotations=""):
    return f'''OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
{annotations}OpDecorate %out_mrt0 Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%S = OpTypeStruct %uint %uint
%out_ptr = OpTypePointer Output %v4uint
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%out_mrt0 = OpVariable %out_ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
%dead = OpIAdd %uint %zero %zero
%live = OpIAdd %uint %one %zero
%rgba = OpCompositeConstruct %v4uint %live %live %live %live
OpStore %out_mrt0 %rgba
OpReturn
OpFunctionEnd
'''


def group_targets(text, group="%group"):
    applications = []
    for line in text.splitlines():
        tokens = line.partition(";")[0].split()
        if len(tokens) >= 2 and tokens[:2] == ["OpGroupDecorate", group]:
            applications.append(tokens[2:])
    return applications


class DecorationTests(unittest.TestCase):
    def check_module(self, source, expected=(1, 1, 1, 1)):
        _validate(source)
        self.assertEqual(support.mrt0_words(source), list(expected))
        optimized = spirv_opt.optimize(source)
        _validate(optimized)
        self.assertEqual(support.mrt0_words(optimized), list(expected))
        return optimized

    def test_grouped_no_contraction_on_translated_module(self):
        raw = support.translate_lines("ps_opt_group", [
            "v_mov_b32_e32 v1, 1.0",
            "v_add_f32_e32 v2, v1, 2.0",
            "v_mul_f32_e32 v2, v1, 3.0",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        _validate(raw)
        targets, lines = [], []
        for line in raw.splitlines():
            tokens = line.split()
            if len(tokens) == 3 and tokens[0] == "OpDecorate" and tokens[2] == "NoContraction":
                targets.append(tokens[1])
            else:
                lines.append(line)
        self.assertGreater(len(targets), 1)
        self.assertNotIn("%group", raw)
        at = next(i for i, line in enumerate(lines) if line.strip().startswith("OpDecorate "))
        lines[at:at] = ["%group = OpDecorationGroup", "OpDecorate %group NoContraction",
                       "OpGroupDecorate %group " + " ".join(targets)]
        optimized = self.check_module("\n".join(lines) + "\n", [0x40400000] * 4)
        self.assertNotIn("OpFAdd", optimized)
        self.assertIn("OpDecorate %group NoContraction", optimized)
        kept = [target for application in group_targets(optimized) for target in application]
        defined = {tokens[0] for line in optimized.splitlines()
                   if len(tokens := line.split()) >= 3 and tokens[1] == "="}
        self.assertTrue(kept)
        self.assertLess(len(kept), len(targets))
        self.assertTrue(set(kept) <= defined)

    def test_group_decorations_protect_live_targets(self):
        for separator in (" ", "\t"):
            application = f"OpGroupDecorate{separator}%group %live"
            with self.subTest(separator=repr(separator)):
                optimized = self.check_module(module(GROUP + application + "\n"))
                self.assertIn(application, optimized)
                self.assertIn("%live = OpIAdd %uint %one %zero", optimized)

    def test_deleted_group_targets_are_pruned(self):
        for applications, expected in (
            ("OpGroupDecorate %group %dead", []),
            ("OpGroupDecorate %group %dead %live", [["%live"]]),
            ("OpGroupDecorate %group %live %dead", [["%live"]]),
            ("OpGroupDecorate %group %dead\nOpGroupDecorate %group %live", [["%live"]]),
            ("OpGroupDecorate\t%group %dead %live ; preserve this comment", [["%live"]]),
        ):
            with self.subTest(applications=applications):
                optimized = self.check_module(module(GROUP + applications + "\n"))
                self.assertEqual(group_targets(optimized), expected)
                self.assertNotIn("%dead", optimized)
                if ";" in applications:
                    self.assertIn("; preserve this comment", optimized)

    def test_unrelated_group_and_member_annotations_are_preserved(self):
        member_name = r'OpMemberName %S 0 "member; \"name\""'
        member_application = "OpGroupMemberDecorate\t%group %S 0"
        direct_member = "OpMemberDecorate\t%S 1 RelaxedPrecision"
        unrelated_application = "OpGroupDecorate\t%unrelated %out_mrt0"
        annotations = "\n".join([
            member_name, GROUP.rstrip(), "OpGroupDecorate %group %dead", member_application, direct_member,
            "%unrelated = OpDecorationGroup", "OpDecorate %unrelated RelaxedPrecision",
            unrelated_application,
        ]) + "\n"
        optimized = self.check_module(module(annotations))
        self.assertEqual(group_targets(optimized), [])
        self.assertIn(member_name, optimized)
        self.assertIn(member_application, optimized)
        self.assertIn(direct_member, optimized)
        self.assertIn(unrelated_application, optimized)
        self.assertIn("OpDecorate %group RelaxedPrecision", optimized)

    def test_direct_decorations_and_names_accept_tabs(self):
        for separator in (" ", "\t"):
            for kind in ("OpDecorate", "OpName"):
                live_name = "OpName" + separator + r'%live "live; \"quoted\"\\path"'
                dead_name = f'OpName{separator}%dead "dead"'
                live_decoration = f"OpDecorate{separator}%live RelaxedPrecision"
                dead_decoration = f"OpDecorate{separator}%dead RelaxedPrecision"
                names = [live_name] + ([dead_name] if kind == "OpName" else [])
                annotations = names + [live_decoration] + ([dead_decoration] if kind == "OpDecorate" else [])
                with self.subTest(kind=kind, separator=repr(separator)):
                    optimized = self.check_module(module("\n".join(annotations) + "\n"))
                    self.assertNotIn("%dead", optimized)
                    self.assertIn(live_name, optimized)
                    self.assertIn(live_decoration, optimized)

    def test_undecorated_values_still_fold(self):
        optimized = self.check_module(module())
        self.assertNotIn("%dead", optimized)
        self.assertNotIn("%live", optimized)


if __name__ == "__main__":
    unittest.main()
