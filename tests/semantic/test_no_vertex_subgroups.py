"""rdna2spirv.py --no-vertex-subgroups: a VS computes its subgroup lane instead of reading SubgroupLocalInvocationId.

A Metal vertex function has no SIMD-group lane id, so MoltenVK, whose subgroupSupportedStages has no vertex bit, can't
create a pipeline for a VS that declares SubgroupLocalInvocationId ("Subgroup builtins are not available in this type
of function"; vs_m15 on an Apple M5, MoltenVK 1.4.2). Ballots and scans do compile there. With the flag a VS that
reads its lane (Gen.wave_lane: the ADD_TID_ENABLE term of a run-time V#, a lane mask held in an SGPR, its own wave
operations) computes it at entry into a Private %sg_lane (vertex_lane_lines): the invocation's rank among the active
ones (an exclusive add of 1) selects that set bit of the active lanes' ballot, which is its own bit. Here:
* the computation, run by the subgroup interpreter on 32- and 64-lane subgroups under many active sets, gives every
  active invocation its own lane (spirv_wave_interp reads SubgroupLocalInvocationId as the lane number);
* a VS with a run-time V# fetch and an SGPR lane mask declares no SubgroupLocalInvocationId under the flag, and every
  lane writes the same words as with the builtin, words that differ from lane to lane;
* a VS that never reads its lane declares no lane at all, and CS and PS translate byte for byte as without the flag.
"""
import random
import re
import unittest

import support
import rdna2spirv
import test_buffer_swizzle as bs         # module only: importing its classes would collect them a second time
from spirv_wave_interp import run_wave

STRIDE = 4
RECORDS = 0x100
ODD_LANES = 0xAAAAAAAA


def translate(name, lines, flag, fssm=None):
    old = rdna2spirv.NO_VERTEX_SUBGROUPS
    rdna2spirv.NO_VERTEX_SUBGROUPS = flag
    try:
        return support.translate_lines(name, lines, fssm=fssm)
    finally:
        rdna2spirv.NO_VERTEX_SUBGROUPS = old


def lane_module(lo, hi):
    """A VS in which the invocations whose bit is set in the 64-bit mask hi:lo (bit = VertexIndex) compute their lane
    with vertex_lane_lines and output (computed lane, SubgroupLocalInvocationId); the others output 0xFFFFFFFF."""
    consts = {}

    def c(n):
        consts.setdefault(n, f"%c{n}")
        return consts[n]
    calc = rdna2spirv.vertex_lane_lines(c)
    body = ["%vi = OpLoad %uint %vertex_index", f"%hiw = OpULessThan %bool %vi {c(32)}",
            f"%word = OpSelect %uint %hiw {c(lo)} {c(hi)}", f"%sh = OpBitwiseAnd %uint %vi {c(31)}",
            "%bits = OpShiftRightLogical %uint %word %sh", f"%bit = OpBitwiseAnd %uint %bits {c(1)}",
            f"%on = OpINotEqual %bool %bit {c(0)}", "OpStore %out %none",
            "OpSelectionMerge %merge None", "OpBranchConditional %on %in %merge", "%in = OpLabel",
            *calc, "%got = OpLoad %uint %sg_lane", "%want = OpLoad %uint %builtin_lane",
            "%res = OpCompositeConstruct %v4uint %got %want %got %want", "OpStore %out %res",
            "OpBranch %merge", "%merge = OpLabel", "OpReturn", "OpFunctionEnd"]
    head = ["OpCapability Shader", "OpCapability GroupNonUniform", "OpCapability GroupNonUniformBallot",
            "OpCapability GroupNonUniformArithmetic", "OpMemoryModel Logical GLSL450",
            'OpEntryPoint Vertex %main "main" %vertex_index %builtin_lane %out %sg_lane',
            "OpDecorate %vertex_index BuiltIn VertexIndex", "OpDecorate %builtin_lane BuiltIn SubgroupLocalInvocationId",
            "OpDecorate %out Location 0",
            "%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%bool = OpTypeBool", "%true = OpConstantTrue %bool",
            "%uint = OpTypeInt 32 0", "%v4uint = OpTypeVector %uint 4", "%v4uint_bal = OpTypeVector %uint 4",
            "%_ptr_In_uint = OpTypePointer Input %uint", "%_ptr_Out_v4u = OpTypePointer Output %v4uint",
            "%_ptr_Private_lane = OpTypePointer Private %uint"]
    head += [f"{name} = OpConstant %uint {v}" for v, name in sorted(consts.items())]
    head += ["%cmax = OpConstant %uint 4294967295", "%none = OpConstantComposite %v4uint %cmax %cmax %cmax %cmax",
             "%vertex_index = OpVariable %_ptr_In_uint Input", "%builtin_lane = OpVariable %_ptr_In_uint Input",
             "%out = OpVariable %_ptr_Out_v4u Output", "%sg_lane = OpVariable %_ptr_Private_lane Private",
             "%main = OpFunction %void None %fn", "%entry = OpLabel"]
    return "\n".join(head + body) + "\n"


class LaneComputation(unittest.TestCase):
    def check(self, wave, mask):
        lo, hi = mask & 0xFFFFFFFF, mask >> 32
        outs = run_wave(lane_module(lo, hi), wave=wave, builtins={"VertexIndex": lambda lane: lane})["outputs"]
        for lane, out in enumerate(outs):
            words = list(out["%out"])
            if mask >> lane & 1:
                self.assertEqual(words[:2], [lane, lane], f"wave {wave}, mask {mask:#x}, lane {lane}")
            else:
                self.assertEqual(words[0], 0xFFFFFFFF)

    def test_every_active_lane_finds_itself(self):
        rng = random.Random(302)
        for wave in (32, 64):
            full = (1 << wave) - 1
            masks = [full, 1, 1 << (wave - 1), 0x55555555_55555555 & full, 0xAAAAAAAA_AAAAAAAA & full,
                     0xF0F0F0F0_0F0F0F0F & full, 0x80000001, 0xFFFF0000]
            if wave == 64:
                masks += [0xFFFFFFFF_00000000, 1 << 32, 0x80000000_00000001, 0x00000001_80000000]
            masks += [rng.getrandbits(wave) | 1 << rng.randrange(wave) for _ in range(24)]
            for mask in masks:
                with self.subTest(wave=wave, mask=hex(mask)):
                    self.check(wave, mask)


def fetch_vs():
    """A VS that reads record `lane` through the V# in user SGPRs s[4:7] (no idxen, ADD_TID_ENABLE set at run time:
    the index is the lane) and keeps it only in the lanes of the mask in user SGPRs s[8:9] (an SGPR the translator can't
    know is a lane mask, so each lane takes its bit), then exports both."""
    return ["buffer_load_dword v3, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
            "v_cndmask_b32_e64 v4, 0, v3, s[8:9]", "v_mov_b32 v5, 1.0",
            "exp pos0 v5, v5, v5, v5 done", "exp param0 v3, v4, v3, v4", "s_endpgm"]


def run_vs(spv, wave=32):
    w3 = bs.W3 | 22 << 12 | 1 << 23                 # FORMAT 32, ADD_TID_ENABLE, OOB_SELECT 0
    user = [0, 0, 0, 0, bs.BASE, STRIDE << 16, RECORDS, w3, ODD_LANES, ODD_LANES] + [0] * 22
    outs = run_wave(spv, wave=wave, push_constants=user, memory=bs.PatternMemory(),
                    builtins={"VertexIndex": lambda lane: 3 + lane, "InstanceIndex": lambda lane: 0})["outputs"]
    return [list(o["%out_param0"]) for o in outs]


class VertexShader(unittest.TestCase):
    def test_flag_declares_no_lane_builtin(self):
        default = translate("vs_fetch_lane", fetch_vs(), flag=False)
        flagged = translate("vs_fetch_lane", fetch_vs(), flag=True)
        self.assertIn("BuiltIn SubgroupLocalInvocationId", default)
        self.assertNotIn("SubgroupLocalInvocationId", flagged)
        self.assertIn("%sg_lane = OpVariable %_ptr_Private_lane Private", flagged)
        self.assertRegex(flagged, r"OpGroupNonUniformIAdd %uint %c3 ExclusiveScan")
        for cap in ("GroupNonUniformBallot", "GroupNonUniformArithmetic"):
            self.assertIn(f"OpCapability {cap}", flagged)

    def test_same_words_as_with_the_builtin(self):
        default = run_vs(translate("vs_fetch_lane", fetch_vs(), flag=False))
        flagged = run_vs(translate("vs_fetch_lane", fetch_vs(), flag=True))
        self.assertEqual(flagged, default)
        for lane, words in enumerate(flagged):       # the lane matters: record `lane`, kept on odd lanes
            self.assertEqual(words[0], bs.tv._pattern_word(lane * STRIDE))
            self.assertEqual(words[1], words[0] if lane & 1 else 0)

    def test_vs_without_a_lane_read_declares_no_lane(self):
        plain = ["v_mov_b32 v5, 1.0", "exp pos0 v5, v5, v5, v5 done", "exp param0 v0, v0, v0, v0", "s_endpgm"]
        self.assertEqual(translate("vs_plain", plain, flag=True), translate("vs_plain", plain, flag=False))
        self.assertNotIn("%sg_lane", translate("vs_plain", plain, flag=True))


class OtherStages(unittest.TestCase):
    def test_compute_and_pixel_shaders_are_unchanged(self):
        access = ["buffer_load_dword v3, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
                  "v_cndmask_b32_e64 v3, 0, v3, s[8:9]"]
        cs = [*access, "buffer_store_dword v3, off, s[0:3], 0", "s_endpgm"]
        ps = [*access, "exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
        for name, lines, fssm in (("cs_fetch_lane", cs, bs.tv.CS_BOUNDS_FSSM), ("ps_fetch_lane", ps, None)):
            with self.subTest(stage=name[:2]):
                flagged = translate(name, lines, flag=True, fssm=fssm)
                self.assertEqual(flagged, translate(name, lines, flag=False, fssm=fssm))
                if name.startswith("ps"):           # a Metal fragment function has the lane id
                    self.assertIn("BuiltIn SubgroupLocalInvocationId", flagged)


if __name__ == "__main__":
    unittest.main()
