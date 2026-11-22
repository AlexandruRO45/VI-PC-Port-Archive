"""Static check (TEST-638): no subgroup operation can be forwarded into a short-circuiting && / || / ?:.

SPIRV-Cross (MoltenVK) forwards a value read once into the expression that reads it, and writes OpLogicalAnd /
OpLogicalOr / OpSelect as the short-circuiting && / || / ?:. A single-use OpGroupNonUniform* result (or a single-use
value computed from one) that is the right side of OpLogicalAnd / OpLogicalOr, or an arm of OpSelect, is then
evaluated only by the invocations that take that side, so a ballot or shuffle misses the others (SHADER-529: the GPU
test cs_dpp_ops read 100 instead of 10 in lane 1 on MoltenVK 1.4.2).

The translator's fix is --subgroup-temps (rdna2spirv.SUBGROUP_TEMPS, #3086), which MoltenVK gets: a dead
OpCopyObject after each subgroup result, a second read that makes SPIRV-Cross bind the result to a temporary where it
is computed. This test translates every lowering that emits subgroup operations with that flag and fails if any
module still has the hazard, so a new lowering (or a change to the pass) can't bring the bug back unnoticed.

The detector is the rule of @SweatierKey's check in #3021 (tests/semantic/test_subgroup_no_shortcircuit.py there),
unchanged: per basic block, a value is "inlinable" when it has exactly one use and is an OpGroupNonUniform* result or
is computed (not loaded, not a phi) from an inlinable value; it is a hazard as the third operand of OpLogicalAnd /
OpLogicalOr or an arm of OpSelect whose condition isn't an OpGroupNonUniformAny / All result (the same in every
invocation, so it cannot diverge). The self-tests below show it flags hand-written hazards and nothing else.

Inputs: every fixture in fixtures/disasm (DPP, ds_permute / ds_bpermute, ballots and wave masks, wave64 and its
emulation among them), in the default, --wave64-emulate and --dispatch-loop modes, plus short hand-written pixel
shaders for the lowerings no fixture has (ds_swizzle, v_permlane16 / v_permlanex16, v_readfirstlane, v_readlane,
v_mbcnt, s_cmp_eq / lg_u64, s_wqm, s_bcnt1, s_ff1, s_bitcmp, SCC from a lane mask). No GPU, no game files.
"""
import collections
import contextlib
import os
import struct
import unittest

import support
import rdna2spirv

SUBGROUP = "OpGroupNonUniform"
SHORT_CIRCUIT_BINARY = ("OpLogicalAnd", "OpLogicalOr")
UNIFORM_VOTES = ("OpGroupNonUniformAny", "OpGroupNonUniformAll")
NOT_FORWARDED = ("OpLoad", "OpPhi", "OpVariable", "OpFunctionParameter")


def find_short_circuit_subgroup_hazards(spvasm):
    """[(opcode, result id, operands)] for every OpLogicalAnd / OpLogicalOr whose right side, or OpSelect whose arm,
    SPIRV-Cross can replace by a single-use subgroup expression from the same block (#3021's rule)."""
    insts = []
    uses = collections.Counter()
    for line in spvasm.splitlines():
        toks = line.split(";", 1)[0].split()
        if not toks:
            continue
        if len(toks) >= 3 and toks[1] == "=":
            res, op, args = toks[0], toks[2], toks[3:]
        else:
            res, op, args = None, toks[0], toks[1:]
        insts.append((res, op, args))
        uses.update(a for a in args if a.startswith("%"))

    inlinable, uniform, hazards = set(), set(), []
    for res, op, args in insts:
        if op == "OpLabel":
            inlinable.clear()
            uniform.clear()
            continue
        if op in SHORT_CIRCUIT_BINARY and len(args) == 3:
            if args[2] in inlinable:
                hazards.append((op, res, args))
        elif op == "OpSelect" and len(args) == 4:
            if args[1] not in uniform and (args[2] in inlinable or args[3] in inlinable):
                hazards.append((op, res, args))
        if res is not None and op in UNIFORM_VOTES:
            uniform.add(res)
        if res is not None and uses[res] == 1:
            if op.startswith(SUBGROUP):
                inlinable.add(res)
            elif op not in NOT_FORWARDED and any(a in inlinable for a in args[1:]):
                inlinable.add(res)
    return hazards


@contextlib.contextmanager
def translator_modes(**modes):
    """Set rdna2spirv's module-level switches for the duration, then put them back."""
    old = {name: getattr(rdna2spirv, name, False) for name in modes}
    try:
        for name, value in modes.items():
            setattr(rdna2spirv, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(rdna2spirv, name, value)


# The compute-shader header for cs_* fixtures: 4 user SGPRs, workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)

# How MoltenVK's modules are translated (make_game.py / run_gpu_tests.py pick --subgroup-temps for driver id 14),
# in each mode that changes the subgroup lowerings.
MOLTENVK_MODES = {
    "default": dict(SUBGROUP_TEMPS=True),
    "wave64-emulate": dict(SUBGROUP_TEMPS=True, WAVE64_EMULATE=True),
    "dispatch-loop": dict(SUBGROUP_TEMPS=True, STRUCTURED=False),
}

LANE_MASK = ["v_cmp_lt_u32_e32 vcc_lo, v0, 16"]
# (body, wave64): pixel shaders for the lowerings that emit subgroup operations but have no fixture.
LISTINGS = {
    "ps_hz_dpp": ([
        "v_mov_b32_e32 v6, 0x11111111",
        "v_mov_b32_e32 v7, v0",
        "v_add_nc_u32_dpp v6, v7, v0 row_shr:1 row_mask:0xf bank_mask:0xf",
        "v_mov_b32_dpp v8, v7 row_shl:1 row_mask:0xa bank_mask:0x5 bound_ctrl:1",
        "v_mov_b32_dpp v9, v7 dpp8:[1,2,3,4,5,6,7,0]",
    ], False),
    "ps_hz_swizzle": (["ds_swizzle_b32 v6, v0 offset:swizzle(SWAP,1)"], False),
    "ps_hz_permlane16": (["v_permlane16_b32 v7, v0, 0x76543210, 0xfedcba98"], False),
    "ps_hz_permlanex16": (["v_permlanex16_b32 v8, v0, 0x76543210, 0xfedcba98"], False),
    "ps_hz_bpermute": (["ds_bpermute_b32 v9, v0, v0"], False),
    "ps_hz_permute": (["v_lshlrev_b32_e32 v2, 2, v0", "ds_permute_b32 v6, v2, v0"], False),
    "ps_hz_readfirstlane": (["v_readfirstlane_b32 s4, v0", "v_mov_b32_e32 v8, s4"], False),
    "ps_hz_readlane": (["v_readlane_b32 s4, v0, 5", "v_mov_b32_e32 v8, s4"], False),
    "ps_hz_mbcnt": (LANE_MASK + ["v_mbcnt_lo_u32_b32 v6, vcc_lo, 0", "v_mbcnt_hi_u32_b32 v7, vcc_hi, v6"], True),
    "ps_hz_cmp_eq_u64": (LANE_MASK + ["s_cmp_eq_u64 vcc, 0x12345678", "s_cselect_b32 s5, 1, 0",
                                      "v_mov_b32_e32 v9, s5"], True),
    "ps_hz_cmp_lg_u64": (LANE_MASK + ["s_cmp_lg_u64 vcc, 0", "s_cselect_b32 s5, 1, 0", "v_mov_b32_e32 v9, s5"], True),
    "ps_hz_wqm_b32": (LANE_MASK + ["s_wqm_b32 s6, vcc_lo", "v_mov_b32_e32 v9, s6"], True),
    "ps_hz_wqm_b64": (LANE_MASK + ["s_wqm_b64 s[6:7], vcc", "v_mov_b32_e32 v9, s6"], True),
    "ps_hz_bcnt1": (LANE_MASK + ["s_bcnt1_i32_b32 s4, vcc_lo", "v_mov_b32_e32 v9, s4"], False),
    "ps_hz_ff1": (LANE_MASK + ["s_ff1_i32_b32 s4, vcc_lo", "v_mov_b32_e32 v9, s4"], False),
    "ps_hz_bitcmp": (LANE_MASK + ["s_bitcmp1_b32 vcc_lo, 3", "s_cselect_b32 s5, 1, 0", "v_mov_b32_e32 v9, s5"], False),
    "ps_hz_scc_from_mask": (LANE_MASK + ["s_and_b32 s4, vcc_lo, exec_lo", "s_cselect_b32 s5, 1, 0",
                                         "v_mov_b32_e32 v9, s5"], False),
}


def translate_listing(name, body, wave64):
    prefix = ["s_mov_b64 s[10:11], exec"] if wave64 else []     # a 64-bit EXEC read makes it a wave64 shader
    return support.translate_lines(name, prefix + body + ["exp mrt0 v6, v7, v8, v9 done vm", "s_endpgm"])


def fixture_names():
    return sorted(f[:-2] for f in os.listdir(support.FIXTURE_DIR) if f.endswith(".s"))


def translate_fixture(name):
    return support.translate_fixture(name, fssm=FSSM if name.startswith("cs_") else None)


class DetectorSelfTest(unittest.TestCase):
    """The detector flags hand-written hazards (so a clean result below means something) and nothing else."""

    def test_ballot_bit_right_of_logical_and(self):
        # The cs_dpp_ops shape: in_row && ballotBitExtract(ballot(...), src).
        spv = ("%entry = OpLabel\n"
               "%in_row = OpULessThan %bool %l %c16\n"
               "%bal = OpGroupNonUniformBallot %v4uint_bal %c3 %act\n"
               "%bit = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %src\n"
               "%both = OpLogicalAnd %bool %in_row %bit\n")
        self.assertEqual(find_short_circuit_subgroup_hazards(spv),
                         [("OpLogicalAnd", "%both", ["%bool", "%in_row", "%bit"])])

    def test_shuffle_in_select_arm(self):
        spv = ("%entry = OpLabel\n"
               "%sh = OpGroupNonUniformShuffle %uint %c3 %v %src\n"
               "%out = OpSelect %uint %ok %c0 %sh\n")
        self.assertEqual(len(find_short_circuit_subgroup_hazards(spv)), 1)

    def test_value_derived_from_subgroup_result_right_of_logical_or(self):
        spv = ("%entry = OpLabel\n"
               "%bit = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %src\n"
               "%nbit = OpLogicalNot %bool %bit\n"
               "%either = OpLogicalOr %bool %ok %nbit\n")
        self.assertEqual(len(find_short_circuit_subgroup_hazards(spv)), 1)

    def test_subgroup_temps_pass_clears_the_hazard(self):
        # The translator's own pass on a hazardous module: the copy is a second use, so nothing is forwarded.
        spv = ("%entry = OpLabel\n"
               "         %bit = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %src\n"
               "         %both = OpLogicalAnd %bool %in_row %bit\n"
               "         %sh = OpGroupNonUniformShuffle %uint %c3 %v %src\n"
               "         %out = OpSelect %uint %ok %sh %c0\n")
        self.assertEqual(len(find_short_circuit_subgroup_hazards(spv)), 2)
        self.assertEqual(find_short_circuit_subgroup_hazards(rdna2spirv.subgroup_temps(spv)), [])

    def test_legitimate_uses_are_not_flagged(self):
        spv = ("%entry = OpLabel\n"
               # left side of && is evaluated by every invocation
               "%bit = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %src\n"
               "%both = OpLogicalAnd %bool %bit %in_row\n"
               # the condition of ?: is evaluated by every invocation
               "%c = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %src2\n"
               "%sel = OpSelect %uint %c %a %b\n"
               # an arm guarded by a subgroup-uniform vote cannot diverge (v_readfirstlane)
               "%any = OpGroupNonUniformAny %bool %c3 %act\n"
               "%lsb = OpGroupNonUniformBallotFindLSB %uint %c3 %bal\n"
               "%first = OpSelect %uint %any %lsb %c0\n"
               # a result read twice is bound to a temporary where it is computed
               "%sh = OpGroupNonUniformShuffle %uint %c3 %v %src\n"
               "%keep = OpIAdd %uint %sh %sh\n"
               "%twice = OpSelect %uint %ok %sh %c0\n"
               # a subgroup result from another block is already computed
               "%sh2 = OpGroupNonUniformShuffle %uint %c3 %v %src\n"
               "OpBranch %next\n"
               "%next = OpLabel\n"
               "%later = OpSelect %uint %ok %sh2 %c0\n")
        self.assertEqual(find_short_circuit_subgroup_hazards(spv), [])


class NoSubgroupShortCircuitHazard(unittest.TestCase):
    """Every subgroup-emitting lowering, translated as for MoltenVK, has no forwardable subgroup operand."""

    def assert_no_hazards(self, name, spv):
        hazards = find_short_circuit_subgroup_hazards(spv)
        self.assertEqual(hazards, [], f"{name}: a subgroup result SPIRV-Cross can forward into a short-circuiting "
                                      f"&& / || / ?: ({len(hazards)}, first {hazards[:3]})")

    def test_fixtures(self):
        names = fixture_names()
        for mode, switches in MOLTENVK_MODES.items():
            with translator_modes(**switches):
                subgroup_ops = 0
                for name in names:
                    with self.subTest(mode=mode, fixture=name):
                        spv = translate_fixture(name)
                        subgroup_ops += spv.count(f"= {SUBGROUP}")
                        self.assert_no_hazards(f"{name} ({mode})", spv)
            # Not vacuous: the corpus has the DPP, permute, ballot and wave-mask fixtures, hundreds of operations.
            self.assertGreater(subgroup_ops, 300, mode)

    def test_hand_written_lowerings(self):
        with translator_modes(**MOLTENVK_MODES["default"]):
            for name, (body, wave64) in LISTINGS.items():
                with self.subTest(listing=name):
                    spv = translate_listing(name, body, wave64)
                    self.assertIn(f"= {SUBGROUP}", spv, f"{name} no longer emits a subgroup operation")
                    self.assert_no_hazards(name, spv)


if __name__ == "__main__":
    unittest.main()
