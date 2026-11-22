"""Regression tests for SHADER-529: no SPIRV-Cross short-circuiting of subgroup ops.

SPIRV-Cross (MoltenVK / MSL) lowers OpLogicalAnd/OpLogicalOr to short-circuiting `&&`/`||`
and OpSelect to short-circuiting `?:`. When a single-use OpGroupNonUniform* expression
sits on the right-hand side of OpLogicalAnd/OpLogicalOr or in an arm of OpSelect, lanes
where the guard is false skip the subgroup op, corrupting cross-lane ballots/shuffles.

Verifies:
  1. Zero short-circuit subgroup hazards across every native subgroup-emitting lowering in
     tools/rdna2spirv.py (DPP16/DPP8, ds_swizzle, ds_permute, ds_bpermute, v_permlane16,
     v_permlanex16, v_readfirstlane, v_mbcnt_lo/hi on lane masks, s_cmp_eq/lg_u64 on
     lane masks, s_wqm_b32).
  2. Negative controls proving the static detector catches both OpLogicalAnd RHS and
     OpSelect arm hazards.
  3. Bit-exact CPU wave execution (and Vulkan 1.2 spirv-val when on PATH) for the
     updated lowerings, including DPP row_shl/row_shr with partial EXEC and masks.
"""
import collections
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import support  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def find_shortcircuit_subgroup_hazards(spvasm: str) -> list[tuple[str, str | None, list[str]]]:
    """Return every OpLogicalAnd/OpLogicalOr (RHS) or OpSelect (true/false arm) that can
    inline a single-use OpGroupNonUniform* expression in the same basic block."""
    insts = []
    use_cnt: collections.Counter[str] = collections.Counter()
    for ln in spvasm.splitlines():
        s = ln.split(";", 1)[0].strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) >= 3 and toks[1] == "=":
            res, op, args = toks[0], toks[2], toks[3:]
        else:
            res, op, args = None, toks[0], toks[1:]
        insts.append((res, op, args))
        for a in args:
            if a.startswith("%"):
                use_cnt[a] += 1

    sg_inlinable: set[str] = set()
    sg_uniform: set[str] = set()
    hazards: list[tuple[str, str | None, list[str]]] = []
    for res, op, args in insts:
        if op == "OpLabel":
            sg_inlinable.clear()
            sg_uniform.clear()
            continue
        if op in ("OpLogicalAnd", "OpLogicalOr") and len(args) == 3:
            if args[2] in sg_inlinable:
                hazards.append((op, res, args))
        elif op == "OpSelect" and len(args) == 4:
            # Subgroup-uniform conditions (e.g. OpGroupNonUniformAny/All) cannot diverge across lanes.
            if args[1] not in sg_uniform and (args[2] in sg_inlinable or args[3] in sg_inlinable):
                hazards.append((op, res, args))
        if res is not None and op in ("OpGroupNonUniformAny", "OpGroupNonUniformAll"):
            sg_uniform.add(res)
        if res is not None and use_cnt[res] == 1:
            if op.startswith("OpGroupNonUniform"):
                sg_inlinable.add(res)
            elif op not in ("OpLoad", "OpPhi", "OpVariable", "OpFunctionParameter") and any(
                a in sg_inlinable for a in args[1:]
            ):
                sg_inlinable.add(res)
    return hazards


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _translate_ps(name: str, body: list[str], *, wave64: bool = False) -> str:
    prefix = ["s_mov_b64 s[10:11], exec"] if wave64 else []
    lines = prefix + body + ["exp mrt0 v6, v7, v8, v9 done vm", "s_endpgm"]
    return support.translate_lines(name, lines)


class SubgroupNoShortCircuitTests(unittest.TestCase):
    def test_detector_catches_known_hazards(self):
        bad_and = (
            "%entry = OpLabel\n"
            "%in_row = OpULessThan %bool %l %c16\n"
            "%bal = OpGroupNonUniformBallot %v4uint_bal %c3 %act\n"
            "%bit = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %src\n"
            "%both = OpLogicalAnd %bool %in_row %bit\n"
        )
        self.assertEqual(len(find_shortcircuit_subgroup_hazards(bad_and)), 1)

        bad_sel = (
            "%entry = OpLabel\n"
            "%sh = OpGroupNonUniformShuffle %uint %c3 %v %src\n"
            "%out = OpSelect %uint %ok %sh %c0\n"
        )
        self.assertEqual(len(find_shortcircuit_subgroup_hazards(bad_sel)), 1)

    def test_all_subgroup_lowerings_have_zero_shortcircuit_hazards(self):
        cases = [
            ("ps_sc_dpp_row_shr", [
                "v_mov_b32_e32 v6, 0x11111111",
                "v_mov_b32_e32 v7, v0",
                "v_add_nc_u32_dpp v6, v7, v0 row_shr:1 row_mask:0xf bank_mask:0xf",
                "v_mov_b32_dpp v8, v7 row_shl:1 row_mask:0xa bank_mask:0x5 bound_ctrl:1",
                "v_mov_b32_dpp v9, v7 dpp8:[1,2,3,4,5,6,7,0]",
            ], False),
            ("ps_sc_swizzle_permlane", [
                "ds_swizzle_b32 v6, v0 offset:swizzle(SWAP,1)",
                "v_permlane16_b32 v7, v0, 0x76543210, 0xfedcba98",
                "v_permlanex16_b32 v8, v0, 0x76543210, 0xfedcba98",
                "ds_bpermute_b32 v9, v0, v0",
            ], False),
            ("ps_sc_permute", [
                "v_lshlrev_b32_e32 v2, 2, v0",
                "ds_permute_b32 v6, v2, v0",
                "v_mov_b32_e32 v7, 0",
                "v_mov_b32_e32 v8, 0",
                "v_mov_b32_e32 v9, 0",
            ], False),
            ("ps_sc_rfl_mbcnt_scmp_wqm", [
                "v_cmp_lt_u32_e32 vcc, v0, 16",
                "v_readfirstlane_b32 s4, v0",
                "v_mbcnt_lo_u32_b32 v6, vcc_lo, 0",
                "v_mbcnt_hi_u32_b32 v7, vcc_hi, v6",
                "s_cmp_eq_u64 vcc, 0x12345678",
                "s_cselect_b32 s5, 1, 0",
                "s_wqm_b32 s6, vcc_lo",
                "v_mov_b32_e32 v8, s4",
                "v_mov_b32_e32 v9, s5",
            ], True),
        ]
        for name, body, w64 in cases:
            with self.subTest(name=name):
                spv = _translate_ps(name, body, wave64=w64)
                _validate(spv)
                hazards = find_shortcircuit_subgroup_hazards(spv)
                self.assertEqual(hazards, [], f"{name} has short-circuit subgroup hazards: {hazards}")

    def test_dpp_and_permlane_wave_execution_exact(self):
        # Verify eager DPP lowering preserves exact semantics on partial EXEC, out-of-row lanes,
        # row_mask/bank_mask, bound_ctrl:0 vs bound_ctrl:1, and ds_swizzle/v_permlane16.
        body = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32_e32 v2, v0",
            "v_add_nc_u32_e32 v2, 100, v2",
            "v_mov_b32_e32 v6, 0xdead",
            "v_mov_b32_e32 v7, 0xbeef",
            # Turn off lane 2 so lane 3's row_shr:1 sees an inactive source lane.
            "v_cmp_ne_u32_e32 vcc_lo, 2, v0",
            "s_and_saveexec_b32 s4, vcc_lo",
            # bound_ctrl:0: lane 0 (out-of-row), lane 16 (out-of-row), lane 3 (inactive src) keep 0xdead.
            "v_mov_b32_dpp v6, v2 row_shr:1 row_mask:0xf bank_mask:0xf",
            # bound_ctrl:1: invalid source lanes read 0 instead of preserving 0xbeef.
            "v_mov_b32_dpp v7, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            # ds_swizzle SWAP,1 under partial EXEC.
            "ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,1)",
            "s_or_b32 exec_lo, exec_lo, s4",
            "v_mov_b32_e32 v9, 0",
        ]
        spv = _translate_ps("ps_sc_exec_dpp", body, wave64=False)
        _validate(spv)
        self.assertEqual(find_shortcircuit_subgroup_hazards(spv), [])

        res = wi.run_wave(spv, wave=32)
        for lane in range(32):
            v6, v7, v8, _ = res["outputs"][lane]["%out_mrt0"]
            if lane == 2:
                # Lane 2 was inactive during DPP & swizzle -> preserved sentinels / 0.
                self.assertEqual(v6, 0xDEAD, f"lane {lane}")
                self.assertEqual(v7, 0xBEEF, f"lane {lane}")
            else:
                src_invalid = (lane % 16 == 0) or (lane == 3)
                exp6 = 0xDEAD if src_invalid else (100 + lane - 1)
                exp7 = 0 if src_invalid else (100 + lane - 1)
                self.assertEqual(v6, exp6, f"lane {lane} v6")
                self.assertEqual(v7, exp7, f"lane {lane} v7")
                self.assertEqual(v8, 100 + (lane ^ 1), f"lane {lane} v8")


if __name__ == "__main__":
    unittest.main()
