"""Tests for tools/cf_diff.py and tools/cf_ref.py, the FUZZ-8 control-flow fuzzer (no game data, no GPU, no
llvm-mc).

    python -m pytest tests/test_cf_diff.py -v
"""
import os
import random
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "cf_diff.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cf_diff  # noqa: E402
import cf_ref  # noqa: E402
import rdna2spirv  # noqa: E402

# Condition true for lanes 4 and 20 only (v3 = lane * 5 & 15), so a lane's own bit and the wave's mask disagree.
FEW_LANES = ("eq", "u32", 4, 3)

# Wave32 shapes whose SCC comes from a 32-bit logic op on lane masks. Until 2026-11-22 the translator set that SCC
# from the lane's own bit instead of the whole mask, and every one of these mismatched.
SCC_MASK_CASES = {
    "s_and_b32 + s_cbranch_scc0": [("sccmask", FEW_LANES, "and", "s_cbranch_scc0", [("probe", 11)])],
    "s_xor_b32 + s_cbranch_scc1": [("sccmask", FEW_LANES, "xor", "s_cbranch_scc1", [("probe", 11)])],
    "s_xor_b32 + s_cselect_b32": [("csel", FEW_LANES, "xor", 11)],
    "s_not_b32 + s_cbranch_scc0": [("sccmask", FEW_LANES, "not", "s_cbranch_scc0", [("probe", 11)])],
    "s_not_b32 + s_cselect_b32": [("csel", FEW_LANES, "not", 11)],
    "loop closed by s_andn2_b32 exec + s_cbranch_scc1": [
        ("vloop", (0, 3, True), None, [("probe", 9), ("op", "s_add_u32 s10, s10, 1")])],
}

# Wave64 32-bit ops on one half of EXEC / VCC (FUZZ-8). Until 2026-11-22 a register written to exec_lo changed all
# 64 lanes (exec_hi was rejected), and the SCC of such an op took the lane's own bit instead of a vote of its half.
HALF = ("lt", "u32", 40, 0)       # lanes 41-63 only: the low half's result is 0, the high half's is not
HALF_CASES = {
    "s_and_b32 exec_lo, exec_lo, vcc_lo": [("halfexec", HALF, "lo", [("probe", 7)])],
    "s_and_b32 exec_hi, exec_hi, vcc_hi": [("halfexec", HALF, "hi", [("probe", 7)])],
    "s_mov_b32 exec_lo, <register>": [("op", "s_mov_b32 s60, 0"), ("op", "s_mov_b32 s61, exec_hi"),
                                       ("op", "s_mov_b64 s[62:63], exec"), ("op", "s_mov_b32 exec_lo, s60"),
                                       ("probe", 7), ("op", "s_mov_b64 exec, s[62:63]")],
    "SCC of s_and_b32 on vcc_hi/exec_hi": [("halfscc", HALF, "and", "hi", "s_cbranch_scc0", [("probe", 7)])],
    "SCC of s_xor_b32 on vcc_lo/exec_lo": [("halfscc", ("lt", "u32", 3, 0), "xor", "lo", "s_cbranch_scc1",
                                            [("probe", 7)])],
}

# Scalar reads of a whole lane mask: the count or first lane of EXEC / VCC / an SGPR compare mask, and the
# v_readfirstlane waterfall loop. Each value is the same for every lane of the wave.
MASK_READ_CASES = {
    "s_bcnt1 of exec -> s_cmp -> branch": [("if", FEW_LANES, False, [
        ("mread", FEW_LANES, "bcnt1", "exec", "b32", "lo", ("gt", 1, "s_cbranch_scc0"), [("probe", 5)])])],
    "s_ff1 of vcc": [("mread", HALF, "ff1", "vcc", "b32", "lo", None, [])],
    "s_ff1 of an SGPR mask -> branch": [("mread", HALF, "ff1", "sgpr", "b32", "lo", ("lt", 64, "s_cbranch_scc1"),
                                         [("probe", 5)])],
    "waterfall, exit on EXECNZ": [("waterfall", 1, False, [("probe", 5)])],
    "waterfall, exit on SCC of s_xor exec": [("waterfall", 2, True, [("probe", 5)])],
    "nested waterfalls": [("waterfall", 3, False, [("probe", 3), ("waterfall", 1, True, [("probe", 5)])])],
}
MASK_READ_W64 = {
    "s_bcnt1_i32_b64 of vcc": [("mread", HALF, "bcnt1", "vcc", "b64", "lo", ("gt", 20, "s_cbranch_scc0"),
                                [("probe", 5)])],
    "s_bcnt1_i32_b32 of vcc_hi": [("mread", HALF, "bcnt1", "vcc", "b32", "hi", None, [])],
    "s_ff1_i32_b32 of exec_hi inside an if": [("if", HALF, False, [
        ("mread", HALF, "ff1", "exec", "b32", "hi", None, [])])],
    "s_ff1_i32_b32 of an SGPR mask's high word": [("mread", HALF, "ff1", "sgpr", "b32", "hi", None, [])],
}

# Whole-quad mode: s_wqm turns on every lane of a quad that has an active one, also lanes an enclosing if turned off.
WQM_CASES = {
    "s_wqm of exec after a compare": [("wqm", ("lt", "u32", 5, 0), "exec", "b32", "lo", None, [("probe", 5)])],
    "s_wqm of vcc into exec + execz": [("wqm", ("eq", "u32", 6, 0), "vcc", "b32", "lo", "s_cbranch_execz",
                                       [("probe", 5)])],
    "s_wqm into an SGPR + scc1": [("wqm", ("eq", "u32", 70, 0), "scc", "b32", "lo", "s_cbranch_scc1", [("probe", 5)])],
    "s_wqm inside an if": [("if", ("gt", "u32", 2, 1), False, [
        ("wqm", ("eq", "u32", 9, 0), "vcc", "b32", "lo", None, [("probe", 5)])])],
}

# v_readlane / v_writelane ignore EXEC: a lane an enclosing if turned off is still read and written.
OFF_LANE = ("gt", "u32", 4, 0)      # lanes 0..3 only
LANE_CASES = {
    "v_writelane of an inactive lane": [("if", OFF_LANE, False, [("lanes", "write", 20, 8, 11), ("probe", 3)])],
    "v_readlane of an inactive lane": [("if", OFF_LANE, False, [("lanes", "read", 20, 2, 10), ("probe", 3)])],
    # 100 is lane 4 in wave32 (& 31) and lane 36 in wave64 (& 63); both are off in OFF_LANE.
    "lane select past the wave wraps": [("if", OFF_LANE, True, [("lanes", "write", 100, 9, 11)])],
    "SGPR lane select in a loop": [("vloop", (0, 3, False), None, [("lanes", "read", "s11", 1, 10),
                                                                   ("lanes", "write", "s10", 9, 13), ("probe", 4)])],
}

# The far-branch idiom (fold_far_branches): s_getpc_b64; s_add|sub_u32 K; s_addc|subb_u32 0|-1; s_setpc_b64.
FAR_CASES = {
    "forward over a block that never runs": [("far", "fwd", 0, [("probe", 3), ("op", "v_add_nc_u32 v9, 1, v9")]),
                                             ("probe", 5)],
    "loop back edge, s_sub_u32 / s_subb_u32": [("far", "sub", 3, [("probe", 3)])],
    "loop back edge, s_add_u32 -K / s_addc_u32 -1": [("far", "addneg", 2, [("probe", 3),
                                                                           ("op", "s_add_u32 s10, s10, 1")])],
    "nested in a divergent if": [("if", ("gt", "u32", 4, 0), True, [
        ("far", "sub", 2, [("probe", 3), ("far", "fwd", 0, [("probe", 9)])])])],
}


@pytest.mark.parametrize("case", sorted(FAR_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_far_branches(case, wave, structured):
    status, detail = cf_diff.run_case("ps_far", FAR_CASES[case], wave, structured)
    assert status == "match", detail


def test_far_branch_cases_catch_a_lost_sign(monkeypatch):
    """With the s_sub_u32 form read as +K, the loop case jumps the wrong way."""
    orig = rdna2spirv.fold_far_branches

    def wrong(insns):
        flipped = [dict(x, op="s_add_u32") if rdna2spirv.base_op(x["op"]) == "s_sub_u32" else x for x in insns]
        flipped = [dict(x, op="s_addc_u32") if rdna2spirv.base_op(x["op"]) == "s_subb_u32" else x for x in flipped]
        return orig(flipped)
    monkeypatch.setattr(rdna2spirv, "fold_far_branches", wrong)
    assert cf_diff.run_case("ps_far", FAR_CASES["loop back edge, s_sub_u32 / s_subb_u32"], 32)[0] != "match"


@pytest.mark.parametrize("case", sorted(LANE_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_lane_select_ignores_exec(case, wave, structured):
    status, detail = cf_diff.run_case("ps_lanes", LANE_CASES[case], wave, structured)
    assert status == "match", detail


@pytest.mark.parametrize("case", sorted(WQM_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_whole_quad_mode(case, wave, structured):
    body = WQM_CASES[case]
    if wave == 64:      # the wave64 forms: s_wqm_b64 on the whole mask
        body = [tuple("b64" if x == "b32" else x for x in n) if n[0] == "wqm" else
                n[:-1] + ([tuple("b64" if x == "b32" else x for x in m) for m in n[-1]],) for n in body]
    status, detail = cf_diff.run_case("ps_wqm", body, wave, structured)
    assert status == "match", detail


@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_whole_quad_mode_on_one_half(structured):
    body = [("wqm", ("lt", "u32", 40, 0), "exec", "b32", "hi", "s_cbranch_scc0", [("probe", 5)])]
    status, detail = cf_diff.run_case("ps_wqm", body, 64, structured)
    assert status == "match", detail


def test_whole_quad_mode_cases_catch_an_identity_wqm(monkeypatch):
    """Before #156 SHADER-34 s_wqm was the identity; these cases see the difference."""
    monkeypatch.setattr(rdna2spirv.Gen, "wqm", lambda self, op, a: self.dst(a[0], self.src_u(a[1])))
    assert cf_diff.run_case("ps_wqm", WQM_CASES["s_wqm inside an if"], 32)[0] == "mismatch"


@pytest.mark.parametrize("case", sorted(MASK_READ_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_mask_reads(case, wave, structured):
    status, detail = cf_diff.run_case("ps_mask_read", MASK_READ_CASES[case], wave, structured)
    assert status == "match", detail


@pytest.mark.parametrize("case", sorted(MASK_READ_W64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_wave64_mask_reads(case, structured):
    status, detail = cf_diff.run_case("ps_mask_read", MASK_READ_W64[case], 64, structured)
    assert status == "match", detail


# s_andn1/andn2_wrexec (tools/cf_diff.py wrelse / wrloop) and the andn1/orn1 saveexec kinds. Each case runs a
# part of the wave on a side the other part skips, so swapping a complemented operand changes some lane's export.
WREXEC_CASES = {
    "andn2_wrexec else switch": [("wrelse", FEW_LANES, False, None,
                                  [("probe", 11)], [("probe", 12), ("op", "v_add_nc_u32 v9, 1, v9")])],
    "andn2_wrexec else switch, skips": [("wrelse", FEW_LANES, True, "s_cbranch_scc0",
                                         [("probe", 11)], [("probe", 12)])],
    "andn1_wrexec loop, execnz": [("wrloop", (0, 3), "s_cbranch_execnz", [("probe", 9)])],
    "andn1_wrexec loop, scc1": [("wrloop", (1, 3), "s_cbranch_scc1",
                                 [("probe", 9), ("op", "v_add_nc_u32 v8, 3, v8")])],
    "andn1_saveexec": [("saveexec", FEW_LANES, "andn1", "s_cbranch_execz", [("probe", 13)])],
    "orn1_saveexec": [("saveexec", FEW_LANES, "orn1", "s_cbranch_scc0", [("probe", 13)])],
}


@pytest.mark.parametrize("case", sorted(WREXEC_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_wrexec_and_n1_saveexec_match(case, wave, structured):
    status, detail = cf_diff.run_case("ps_wrexec", WREXEC_CASES[case], wave, structured)
    assert status == "match", detail


@pytest.mark.parametrize("case", sorted(WREXEC_CASES))
@pytest.mark.parametrize("wave", (32, 64))
def test_wrexec_cases_catch_a_swapped_complement(case, wave, monkeypatch):
    """The cases above are sensitive: with n1 and n2 swapped (S0 complemented instead of EXEC) they mismatch."""
    logic = dict(rdna2spirv.Gen.EXEC_LOGIC)
    for a, b in (("andn1", "andn2"), ("orn1", "orn2")):
        logic[a], logic[b] = logic[b], logic[a]
    monkeypatch.setattr(rdna2spirv.Gen, "EXEC_LOGIC", logic)
    status, _ = cf_diff.run_case("ps_wrexec", WREXEC_CASES[case], wave)
    assert status == "mismatch"


# s_bitcmp of a lane mask: FEW_LANES sets lanes 4 and 20 of VCC, so bit 4 / 20 is 1 and bit 5 is 0 in every lane,
# while each lane's own bit differs. EXEC inside a FEW_LANES `if` is lanes 4 and 20 only, so bit 5 of it is 0 although
# every running lane's own bit is 1.
BITCMP_CASES = {
    "bitcmp1 vcc bit 4": [("bitcmp", FEW_LANES, "vcc", "1", None, "4", "s_cbranch_scc0", [("probe", 11)])],
    "bitcmp0 vcc bit 5": [("bitcmp", FEW_LANES, "vcc", "0", None, "5", "s_cbranch_scc1", [("probe", 11)])],
    "bitcmp1 exec bit 5 in an if": [("if", FEW_LANES, False, [
        ("bitcmp", FEW_LANES, "exec", "1", None, "5", "s_cbranch_scc0", [("probe", 11)])])],
    "bitcmp1 vcc bit from s11": [("op", "s_mov_b32 s11, 20"),
                                 ("bitcmp", FEW_LANES, "vcc", "1", None, "s11", "s_cbranch_scc0", [("probe", 11)])],
}
BITCMP_W64_CASES = {
    # 36 >= lane: lanes 0-36 are set, so bit 4 of vcc_hi (lane 36) is 1.
    "bitcmp1 vcc_hi bit 4 (lane 36)": [("bitcmp", ("ge", "u32", 36, 0), "vcc", "1", "hi", "4", "s_cbranch_scc0",
                                        [("probe", 11)])],
    "bitcmp0 vcc bit 4 (b64)": [("bitcmp", FEW_LANES, "vcc", "0", None, "4", "s_cbranch_scc1", [("probe", 11)])],
    "bitcmp1 vcc bit 40 (b64)": [("bitcmp", ("le", "u32", 36, 0), "vcc", "1", None, "40", "s_cbranch_scc0",
                                  [("probe", 11)])],
}
SCMPK_CASES = {
    "cmpk lt i32 0xffff (-1)": [("op", "s_mov_b32 s11, -5"), ("scmpk", ("lt", "i32", 11, 0xFFFF), "s_cbranch_scc0",
                                                               [("probe", 11)])],
    "cmpk lt u32 0xffff": [("op", "s_mov_b32 s11, -5"), ("scmpk", ("lt", "u32", 11, 0xFFFF), "s_cbranch_scc1",
                                                          [("probe", 11)])],
    "cmpk eq u32 12": [("scmpk", ("eq", "u32", 13, 12), "s_cbranch_scc0", [("probe", 11)])],
}


@pytest.mark.parametrize("case", sorted(BITCMP_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_bitcmp_of_a_lane_mask_matches(case, wave, structured):
    status, detail = cf_diff.run_case("ps_bitcmp", BITCMP_CASES[case], wave, structured)
    assert status == "match", detail


@pytest.mark.parametrize("case", sorted(BITCMP_W64_CASES))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_bitcmp_of_a_wave64_mask_half_matches(case, structured):
    status, detail = cf_diff.run_case("ps_bitcmp", BITCMP_W64_CASES[case], 64, structured)
    assert status == "match", detail


def _per_lane_bitcmp(monkeypatch):
    """The slip this guards: s_bitcmp reading the lane's own mask bit (a lane mask taken as data)."""
    real = rdna2spirv.Gen.bitcmp

    def bitcmp(self, op, a):
        self.holds_lane_mask = lambda x: False
        try:
            return real(self, op, a)
        finally:
            del self.holds_lane_mask
    monkeypatch.setattr(rdna2spirv.Gen, "bitcmp", bitcmp)


@pytest.mark.parametrize("case", sorted(BITCMP_CASES))
def test_bitcmp_cases_catch_a_per_lane_bit(case, monkeypatch):
    _per_lane_bitcmp(monkeypatch)
    assert cf_diff.run_case("ps_bitcmp", BITCMP_CASES[case], 32)[0] == "mismatch"


@pytest.mark.parametrize("case", sorted(BITCMP_W64_CASES))
def test_bitcmp_w64_cases_catch_a_per_lane_bit(case, monkeypatch):
    _per_lane_bitcmp(monkeypatch)
    assert cf_diff.run_case("ps_bitcmp", BITCMP_W64_CASES[case], 64)[0] == "mismatch"


@pytest.mark.parametrize("case", sorted(SCMPK_CASES))
@pytest.mark.parametrize("wave", (32, 64))
def test_scmpk_matches(case, wave):
    status, detail = cf_diff.run_case("ps_scmpk", SCMPK_CASES[case], wave)
    assert status == "match", detail


def test_reference_cmpk_extends_the_immediate():
    # s11 = -5: as i32, -5 < sext(0xffff) = -1; as u32, 0xfffffffb < 0x0000ffff is false.
    for ty, want in (("i32", 1), ("u32", 0)):
        body = [("op", "s_mov_b32 s11, -5"), ("op", f"s_cmpk_lt_{ty} s11, 0xffff"), ("op", "s_cselect_b32 s12, 1, 0"),
                ("op", "v_mov_b32 v9, s12")]
        lines = cf_diff.build(body, 32)
        with tempfile.TemporaryDirectory() as t:
            os.mkdir(os.path.join(t, "disasm"))
            p = os.path.join(t, "disasm", "ps_k.s")
            with open(p, "w", encoding="utf-8") as f:
                f.write(cf_diff.listing_text(lines, "ps_k"))
            assert cf_ref.run(p, wave=32)["exports"][0][1] == want, ty


# Float compares into EXEC (values from Emitter.fvalue). v1 = lane & 7 gives normals and NaNs (bit 2); v3 =
# lane * 5 & 15 adds +inf and +0 (bits 2-3); v8 = 9 ^ lane reaches bit 4 too, so even in wave32 its lanes hold every
# value: +-normals, NaNs, +-inf and +-0.
FCMP_CASES = {
    f"{mn} {cc} {k} on v{v}": [(mn, (cc, k, v), skip, [("probe", 11), ("op", "v_add_nc_u32 v9, 1, v9")])]
    for mn, cc, k, v, skip in (
        ("fif", "lt", "1.0", 8, True), ("fif", "nge", "1.0", 8, False), ("fif", "u", "0", 8, True),
        ("fif", "o", "0x7fc00000", 8, False), ("fif", "eq", "0x80000000", 8, True),
        ("fif", "neq", "0x7f800000", 8, False), ("fif", "lg", "-4.0", 3, True),
        ("fcmpx", "nlg", "0", 8, True), ("fcmpx", "ge", "0xff800000", 8, False),
        ("fcmpx", "ngt", "-1.0", 8, True), ("fcmpx", "tru", "2.0", 1, False), ("fcmpx", "f", "0.5", 1, True),
        ("fif", "class", 0x003, 8, True), ("fcmpx", "class", 0x22C, 8, False), ("fif", "class", 0, 8, False),
        ("fcmpx", "class", 0x160, 8, True))
}


@pytest.mark.parametrize("case", sorted(FCMP_CASES))
@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_float_compares_into_exec_match(case, wave, structured):
    status, detail = cf_diff.run_case("ps_fcmp", FCMP_CASES[case], wave, structured)
    assert status == "match", detail


def _ord_unord_swapped(cc, x, y, real=cf_ref._fcmp):
    """The OpFOrd / OpFUnord slip: on a NaN the ordered forms answer true and the n forms false."""
    r = real(cc, x, y)
    nan = cf_ref._f32(x) is None or cf_ref._f32(y) is None
    return (not r) if nan and cc not in ("f", "tru") else r


@pytest.mark.parametrize("case", sorted(c for c in FCMP_CASES if "class" not in c and " f " not in c
                                        and "tru" not in c and " lg -4.0" not in c))
def test_float_cases_catch_an_ordered_unordered_swap(case, monkeypatch):
    """The compare cases are sensitive to NaN handling: with ordered and unordered swapped they mismatch."""
    monkeypatch.setattr(cf_ref, "_fcmp", _ord_unord_swapped)
    assert cf_diff.run_case("ps_fcmp", FCMP_CASES[case], 32)[0] == "mismatch"


@pytest.mark.parametrize("case", sorted(c for c in FCMP_CASES if "class" in c and "0x3 " not in c
                                        and " 3 on" not in c and " 0 on" not in c))
def test_class_cases_catch_a_sign_slip(case, monkeypatch):
    """The class cases are sensitive: with the sign ignored (every value classed as positive) they mismatch."""
    real = cf_ref._fclass
    monkeypatch.setattr(cf_ref, "_fclass", lambda x, mask: real(x & 0x7FFFFFFF, mask))
    assert cf_diff.run_case("ps_fcmp", FCMP_CASES[case], 32)[0] == "mismatch"


def test_fvalue_covers_every_class_in_one_wave32():
    """Emitter.fvalue of v8 = 9 ^ lane gives wave32 every kind of operand: +-normal, NaN, +-inf and +-0."""
    lines = cf_diff.build([("fif", ("class", 0, 8), False, [])], 32)     # fvalue(v8), then a compare nothing uses
    with tempfile.TemporaryDirectory() as t:
        os.mkdir(os.path.join(t, "disasm"))
        p = os.path.join(t, "disasm", "ps_v.s")
        with open(p, "w", encoding="utf-8") as f:
            f.write(cf_diff.listing_text(lines, "ps_v"))
        m = cf_ref.WaveMachine(cf_ref.parse_file(p), wave=32)
        m.run()
    seen = {next(b for b in range(10) if cf_ref._fclass(m._vgpr(12, lane), 1 << b)) for lane in range(32)}
    assert {1, 2, 3, 5, 6, 8, 9} <= seen, seen      # qNaN, -inf, -normal, -0, +0, +normal, +inf


@pytest.mark.parametrize("bits, cc, k, want", [
    (0x7FC00000, "lt", 0x3F800000, False), (0x7FC00000, "nge", 0x3F800000, True), (0x80000000, "eq", 0, True),
    (0x80000000, "lg", 0, False), (0x7F800000, "gt", 0x3F800000, True), (0x7FC00000, "u", 0, True),
    (0x3F800000, "o", 0, True), (0x7FC00000, "neq", 0x7FC00000, True), (0x7FC00000, "eq", 0x7FC00000, False)])
def test_reference_float_compare(bits, cc, k, want):
    assert cf_ref._fcmp(cc, bits, k) is want


@pytest.mark.parametrize("bits, cls", [(0x7F800001, 0), (0x7FC00000, 1), (0xFF800000, 2), (0xBF800000, 3),
                                       (0x80000001, 4), (0x80000000, 5), (0, 6), (1, 7), (0x3F800000, 8),
                                       (0x7F800000, 9)])
def test_reference_float_class(bits, cls):
    assert [cf_ref._fclass(bits, 1 << b) for b in range(10)] == [int(b == cls) for b in range(10)]


@pytest.mark.parametrize("count", ["0", "-1"])
def test_empty_program_batch_is_a_usage_error(count, capsys, monkeypatch):
    def unexpected_suite(*args, **kwargs):
        pytest.fail("started a suite with no programs")

    monkeypatch.setattr(cf_diff, "run_suite", unexpected_suite)
    with pytest.raises(SystemExit) as exc:
        cf_diff.main(["--programs", count])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--programs must be at least 1" in captured.err
    assert captured.out == ""


def test_a_run_where_every_program_was_skipped_is_not_a_success(capsys, monkeypatch):
    monkeypatch.setattr(cf_diff, "run_case", lambda *args, **kwargs: ("skip", "unsupported form"))
    assert cf_diff.main(["--programs", "2", "--seed", "1"]) == 2
    assert "no program ran: every one was skipped" in capsys.readouterr().out


@pytest.mark.parametrize("statuses, code", [
    (["skip", "skip"], 2),            # nothing was compared
    (["skip", "mismatch"], 1),        # a mismatch outranks "nothing else ran"
    (["match", "skip"], 0),           # one wave compared and matched
])
def test_a_single_program_run_exits_like_the_suite(statuses, code, capsys, monkeypatch):
    results = iter(statuses)
    monkeypatch.setattr(cf_diff, "run_case", lambda *args, **kwargs: (next(results), "detail"))
    assert cf_diff.main(["--program", "3", "--seed", "1", "--wave", "both"]) == code
    assert ("no program ran: every one was skipped" in capsys.readouterr().out) == (code == 2)


def test_generation_is_deterministic():
    assert cf_diff.program_for(5, 3, 32, 3) == cf_diff.program_for(5, 3, 32, 3)
    assert cf_diff.program_for(5, 3, 32, 3) != cf_diff.program_for(5, 4, 32, 3)


def test_branch_offsets_land_on_their_labels():
    e = cf_diff.Emitter(32)
    e.program([("if", FEW_LANES, True, [("probe", 3)]), ("sloop", 2, [("probe", 4)])])
    lines = e.assemble()
    pcs, pc = [], 0
    for _, size in lines:
        pcs.append(pc)
        pc += size
    for (text, size), at in zip(lines, pcs):
        if text.startswith(("s_branch", "s_cbranch")):
            target = at + 4 + 4 * int(text.split()[1])
            assert target in pcs, text


def test_reference_needs_encodings():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.s")
        with open(p, "w", encoding="utf-8") as f:
            f.write("; x  code 0x0..0x8\n\ts_branch 0\n\ts_endpgm\n")
        with pytest.raises(cf_ref.ProgramError):
            cf_ref.run(p)


def test_reference_wave32_saveexec_and_restore():
    body = [("if", FEW_LANES, True, [("op", "v_add_nc_u32 v9, 1, v9")])]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.s")
        with open(p, "w", encoding="utf-8") as f:
            f.write(cf_diff.listing_text(cf_diff.build(body, 32), "ps_x"))
        r = cf_ref.run(p, wave=32)
    assert sorted(r["exports"]) == list(range(32))
    assert [lane for lane in range(32) if r["exports"][lane][1] == 1] == [4, 20]
    assert r["exec"] == 0xFFFFFFFF


@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_small_fixed_seed_run_is_clean(wave, structured):
    counts, failures = cf_diff.run_suite(6, seed=20261122, waves=(wave,), structured=structured)
    assert counts["harness-error"] == 0, failures
    assert counts["mismatch"] == 0, failures
    assert counts["pass"] == 6, counts


@pytest.mark.parametrize("case", sorted(SCC_MASK_CASES))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_scc_of_wave32_mask_ops_is_wave_wide(case, structured):
    status, detail = cf_diff.run_case("ps_scc_mask", SCC_MASK_CASES[case], 32, structured)
    assert status == "match", detail


@pytest.mark.parametrize("case", sorted(SCC_MASK_CASES))
def test_scc_mask_cases_catch_a_per_lane_scc(case, monkeypatch):
    """The cases above are sensitive: with the old per-lane SCC they mismatch."""
    monkeypatch.setattr(rdna2spirv.Gen, "scc_logic", lambda self, a, r: self.scc_nonzero(r))
    status, _ = cf_diff.run_case("ps_scc_mask", SCC_MASK_CASES[case], 32)
    assert status == "mismatch"


@pytest.mark.parametrize("case", sorted(HALF_CASES))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_wave64_half_mask_ops(case, structured):
    status, detail = cf_diff.run_case("ps_half", HALF_CASES[case], 64, structured)
    assert status == "match", detail


def test_minimize_keeps_the_mismatch(monkeypatch):
    monkeypatch.setattr(rdna2spirv.Gen, "scc_logic", lambda self, a, r: self.scc_nonzero(r))
    body = [("probe", 1), ("op", "v_add_nc_u32 v8, 3, v8")] + SCC_MASK_CASES["s_and_b32 + s_cbranch_scc0"]
    small = cf_diff.minimize(body, 32)
    assert len(small) < len(body)
    assert cf_diff.run_case("ps_scc_mask", small, 32)[0] == "mismatch"


def test_self_test_passes():
    assert cf_diff.self_test() is True


def test_no_arguments_prints_usage():
    proc = subprocess.run([sys.executable, "-I", TOOL], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 2
    assert "usage:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_generator_covers_every_shape():
    kinds = set()

    def walk(nodes):
        for n in nodes:
            kinds.add(n[0])
            for x in n[1:]:
                if isinstance(x, list):
                    walk(x)
    for i in range(200):
        walk(cf_diff.generate(random.Random(i), 32))
    assert {"if", "ifelse", "cmpx", "saveexec", "vccz", "sif", "sloop", "vloop", "irr", "sccmask", "csel", "probe",
            "ifs", "cexec", "halfexec", "halfscc", "scmpmask", "cselmask", "vccloop", "cmpxloop", "mread", "waterfall", "wqm", "lanes", "far",
            "wrelse", "wrloop", "fif", "fcmpx", "bitcmp", "scmpk"} <= kinds


@pytest.mark.parametrize("op_name", ("andn1", "andn2", "orn1", "orn2", "xor", "nand", "nor", "xnor"))
@pytest.mark.parametrize("wave", (32, 64))
def test_saveexec_variants_match(op_name, wave):
    body = [("saveexec", FEW_LANES, op_name, "s_cbranch_scc0", [("probe", 13), ("op", "v_add_nc_u32 v9, 1, v9")])]
    status, detail = cf_diff.run_case(f"ps_saveexec_{op_name}", body, wave, structured=True)
    assert status == "match", detail
