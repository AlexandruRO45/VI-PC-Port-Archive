"""Translator-to-interpreter semantic tests for RDNA2 f16 transcendental special values:

  v_log_f16, v_rcp_f16, v_rsq_f16, v_sqrt_f16, v_sin_f16, v_cos_f16, v_exp_f16

Models the RDNA2 hardware-verified special results (FUZZ-29 on RX 6700 XT gfx1031) for inputs that SPIR-V /
GLSL.std.450 otherwise leaves undefined, sanitizing inputs before calling SPIR-V ops and selecting the RDNA2 f32
special result before VOP3 omod/clamp:
  - v_log_f16(+-0) = -inf (0xfc00); v_log_f16(x < 0, including -inf) = 0xfe00
  - v_rcp_f16(+0) = +inf (0x7c00); v_rcp_f16(-0) = -inf (0xfc00)
  - v_rsq_f16(+0) = +inf (0x7c00); v_rsq_f16(-0) = -inf (0xfc00); v_rsq_f16(x < 0, including -inf) = 0xfe00
  - v_sqrt_f16(-0) = -0 (0x8000); v_sqrt_f16(x < 0, including -inf) = 0xfe00
  - v_sin_f16(+-inf) = 0xfe00; v_cos_f16(+-inf) = 0xfe00
  - v_exp_f16 on +0, -0, +inf, -inf, NaN

Programs are assembled with real llvm-mc (-mcpu=gfx1013 -show-encoding), translated via tools/rdna2spirv.py,
validated with spirv-as + spirv-val (--target-env vulkan1.2), and executed in tests/semantic/spirv_mini_interp.py.
Standard library only; no game data.
"""
from collections import defaultdict, deque
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mc_command  # noqa: E402
from support import mrt0_words, translate_listing  # noqa: E402

POS_ZERO = 0x0000
NEG_ZERO = 0x8000
NEG_SUBNORM = 0x8001
NEG_HALF = 0xB800
NEG_QUARTER = 0xB400
NEG_ONE = 0xBC00
NEG_TWO = 0xC000
POS_QUARTER = 0x3400
POS_HALF = 0x3800
POS_ONE = 0x3C00
POS_TWO = 0x4000
POS_FOUR = 0x4400
POS_INF = 0x7C00
NEG_INF = 0xFC00
POS_QNAN = 0x7E00
NEG_QNAN = 0xFE00
POS_QNAN_PAY = 0x7E55
NEG_QNAN_PAY = 0xFEAA

OLD = 0xABCDBEEF
OLD_HI = OLD & 0xFFFF0000
OLD_LO = OLD & 0x0000FFFF


def _assemble_gfx1013(lines):
    cmd = mc_command.mc_command() + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
    src = "\n".join(lines) + "\n"
    try:
        res = subprocess.run(cmd, input=src, capture_output=True, text=True)
    except OSError as exc:
        raise unittest.SkipTest(f"llvm-mc unavailable: {exc}") from exc
    if res.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{res.stderr}\nsource:\n{src}")
    return res.stdout


def _validate_spirv(spv_text):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise unittest.SkipTest("spirv-as / spirv-val not on PATH")
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "test.spvasm")
        bin_p = os.path.join(d, "test.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}\n{spv_text}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}\n{spv_text}")


def run_ps_mc(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    disasm = f"; {name}\n" + _assemble_gfx1013(lines)
    spv = translate_listing(name, disasm.encode("utf-8"))
    _validate_spirv(spv)
    return mrt0_words(spv)[0]


def run_ps_listing(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    text = f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines)
    spv = translate_listing(name, text.encode("utf-8"))
    _validate_spirv(spv)
    return mrt0_words(spv)[0]


# Expected f16 bit patterns for every op across special and representative normal inputs.
SPECIAL_MATRIX = {
    "v_log_f16": [
        ("pos_zero", POS_ZERO, NEG_INF),
        ("neg_zero", NEG_ZERO, NEG_INF),
        ("neg_finite", NEG_ONE, NEG_QNAN),
        ("neg_subnorm", NEG_SUBNORM, NEG_QNAN),
        ("neg_inf", NEG_INF, NEG_QNAN),
        ("pos_inf", POS_INF, POS_INF),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_FOUR, POS_TWO),
    ],
    "v_rcp_f16": [
        ("pos_zero", POS_ZERO, POS_INF),
        ("neg_zero", NEG_ZERO, NEG_INF),
        ("neg_finite", NEG_TWO, NEG_HALF),
        ("neg_inf", NEG_INF, NEG_ZERO),
        ("pos_inf", POS_INF, POS_ZERO),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_FOUR, POS_QUARTER),
    ],
    "v_rsq_f16": [
        ("pos_zero", POS_ZERO, POS_INF),
        ("neg_zero", NEG_ZERO, NEG_INF),
        ("neg_finite", NEG_ONE, NEG_QNAN),
        ("neg_subnorm", NEG_SUBNORM, NEG_QNAN),
        ("neg_inf", NEG_INF, NEG_QNAN),
        ("pos_inf", POS_INF, POS_ZERO),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_FOUR, POS_HALF),
    ],
    "v_sqrt_f16": [
        ("pos_zero", POS_ZERO, POS_ZERO),
        ("neg_zero", NEG_ZERO, NEG_ZERO),
        ("neg_finite", NEG_ONE, NEG_QNAN),
        ("neg_subnorm", NEG_SUBNORM, NEG_QNAN),
        ("neg_inf", NEG_INF, NEG_QNAN),
        ("pos_inf", POS_INF, POS_INF),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_FOUR, POS_TWO),
    ],
    "v_sin_f16": [
        ("pos_zero", POS_ZERO, POS_ZERO),
        ("neg_zero", NEG_ZERO, NEG_ZERO),
        ("neg_finite", NEG_QUARTER, NEG_ONE),
        ("neg_inf", NEG_INF, NEG_QNAN),
        ("pos_inf", POS_INF, NEG_QNAN),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_QUARTER, POS_ONE),
    ],
    "v_cos_f16": [
        ("pos_zero", POS_ZERO, POS_ONE),
        ("neg_zero", NEG_ZERO, POS_ONE),
        ("neg_finite", NEG_HALF, NEG_ONE),
        ("neg_inf", NEG_INF, NEG_QNAN),
        ("pos_inf", POS_INF, NEG_QNAN),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_ONE, POS_ONE),
    ],
    "v_exp_f16": [
        ("pos_zero", POS_ZERO, POS_ONE),
        ("neg_zero", NEG_ZERO, POS_ONE),
        ("neg_finite", NEG_ONE, POS_HALF),
        ("neg_inf", NEG_INF, POS_ZERO),
        ("pos_inf", POS_INF, POS_INF),
        ("pos_qnan", POS_QNAN, POS_QNAN),
        ("neg_qnan", NEG_QNAN, NEG_QNAN),
        ("normal_pos", POS_TWO, POS_FOUR),
    ],
}


def _parse_spirv_ssa(spv_text):
    """Parse SPIR-V assembly into SSA definitions, resolved 32-bit constants, stores, and def-use edges."""
    defs = {}
    consts = {}
    access_chains = {}
    ssa_uses = defaultdict(list)
    slot_stores = defaultdict(list)
    slot_loads = defaultdict(list)
    output_vars = set()
    output_stores = []

    for raw in spv_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        m = re.match(r"^(%\S+)\s*=\s*(Op\S+)(.*)$", line)
        if m:
            dst, opcode, rest = m.group(1), m.group(2), m.group(3).split()
            defs[dst] = (opcode, rest)
            if opcode == "OpConstant" and len(rest) >= 2:
                if rest[0] == "%uint":
                    consts[dst] = int(rest[1], 0)
                elif rest[0] == "%float":
                    consts[dst] = struct.unpack("<I", struct.pack("<f", float(rest[1])))[0]
            elif opcode == "OpBitcast" and len(rest) >= 2 and rest[1] in consts:
                consts[dst] = consts[rest[1]]
            elif opcode == "OpVariable" and len(rest) >= 2 and rest[1] == "Output":
                output_vars.add(dst)
            elif opcode == "OpAccessChain" and len(rest) >= 3:
                access_chains[dst] = (rest[1], tuple(rest[2:]))
            elif opcode == "OpLoad" and len(rest) >= 2 and rest[1] in access_chains:
                slot_loads[access_chains[rest[1]]].append(dst)
            for tok in rest:
                if tok.startswith("%"):
                    ssa_uses[tok].append(dst)
            continue

        m_store = re.match(r"^OpStore\s+(%\S+)\s+(%\S+)$", line)
        if m_store:
            ptr, val = m_store.group(1), m_store.group(2)
            if ptr in access_chains:
                slot_stores[access_chains[ptr]].append(val)
            if ptr in output_vars:
                output_stores.append((ptr, val))

    val_to_slots = defaultdict(list)
    for slot, vals in slot_stores.items():
        for v in vals:
            val_to_slots[v].append(slot)

    def reachable(start, follow_memory=False):
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in ssa_uses.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
            if follow_memory:
                for slot in val_to_slots.get(cur, ()):
                    for nxt in slot_loads.get(slot, ()):
                        if nxt not in seen:
                            seen.add(nxt)
                            q.append(nxt)
        return seen

    return defs, consts, output_stores, reachable


def _is_f16_nan(h):
    return (h & 0x7C00) == 0x7C00 and (h & 0x3FF) != 0


class F16TranscendentalSpecialsTests(unittest.TestCase):
    def test_special_inputs_and_high_half_preservation_e32(self):
        for op, cases in SPECIAL_MATRIX.items():
            for label, src_h, want_h in cases:
                with self.subTest(op=op, case=label, src=hex(src_h)):
                    got = run_ps_mc(
                        f"ps_{op}_{label}",
                        [
                            f"v_mov_b32 v0, {OLD:#x}",
                            f"v_mov_b32 v1, {0xCAFE0000 | src_h:#x}",
                            f"{op} v0, v1",
                        ],
                    )
                    self.assertEqual(got, OLD_HI | want_h, f"{op}({label}={src_h:#06x}) -> {got:#010x}")

    def test_nan_payload_inputs_propagate_signed_nan(self):
        for op in SPECIAL_MATRIX:
            for src_h in (POS_QNAN_PAY, NEG_QNAN_PAY):
                with self.subTest(op=op, src=hex(src_h)):
                    got = run_ps_mc(
                        f"ps_{op}_nan_pay",
                        [
                            f"v_mov_b32 v0, {OLD:#x}",
                            f"v_mov_b32 v1, {src_h:#x}",
                            f"{op} v0, v1",
                        ],
                    )
                    self.assertEqual(got & 0xFFFF0000, OLD_HI)
                    self.assertTrue(_is_f16_nan(got & 0xFFFF), f"{op}({src_h:#06x}) -> {got:#010x}")

    def test_vop3_source_modifiers_on_specials(self):
        cases = [
            ("v_sqrt_f16_e64 v0, |v1|", NEG_INF, POS_INF),
            ("v_sqrt_f16_e64 v0, -v1", POS_INF, NEG_QNAN),
            ("v_sqrt_f16_e64 v0, -v1", POS_FOUR, NEG_QNAN),
            ("v_sqrt_f16_e64 v0, -v1", POS_ZERO, NEG_ZERO),
            ("v_sqrt_f16_e64 v0, |v1|", NEG_ZERO, POS_ZERO),
            ("v_rcp_f16_e64 v0, -v1", POS_ZERO, NEG_INF),
            ("v_rcp_f16_e64 v0, |v1|", NEG_ZERO, POS_INF),
            ("v_rsq_f16_e64 v0, -v1", POS_ZERO, NEG_INF),
            ("v_rsq_f16_e64 v0, |v1|", NEG_FOUR := 0xC400, POS_HALF),
            ("v_log_f16_e64 v0, |v1|", NEG_FOUR, POS_TWO),
            ("v_log_f16_e64 v0, -v1", POS_INF, NEG_QNAN),
            ("v_sin_f16_e64 v0, |v1|", NEG_INF, NEG_QNAN),
            ("v_cos_f16_e64 v0, -v1", POS_INF, NEG_QNAN),
            ("v_exp_f16_e64 v0, -v1", POS_INF, POS_ZERO),
        ]
        for insn, src_h, want_h in cases:
            with self.subTest(insn=insn, src=hex(src_h)):
                got = run_ps_mc(
                    "ps_f16_trans_src_mods",
                    [
                        f"v_mov_b32 v0, {OLD:#x}",
                        f"v_mov_b32 v1, {src_h:#x}",
                        insn,
                    ],
                )
                self.assertEqual(got, OLD_HI | want_h)

    def test_vop3_clamp_and_omod_interaction(self):
        # Special values are selected before clamp_mods so omod/clamp apply afterward as on hardware.
        cases = [
            ("v_log_f16_e64 v0, v1 clamp", POS_ZERO, POS_ZERO),       # -inf clamped to [0, 1] -> +0.0
            ("v_log_f16_e64 v0, v1 clamp", POS_FOUR, POS_ONE),        # 2.0 clamped to [0, 1] -> 1.0
            ("v_rcp_f16_e64 v0, v1 clamp", POS_ZERO, POS_ONE),        # +inf clamped to [0, 1] -> 1.0
            ("v_rcp_f16_e64 v0, v1 clamp", NEG_ZERO, POS_ZERO),       # -inf clamped to [0, 1] -> +0.0
            ("v_rsq_f16_e64 v0, v1 clamp", POS_ZERO, POS_ONE),        # +inf clamped to [0, 1] -> 1.0
            ("v_rsq_f16_e64 v0, v1 clamp", NEG_ZERO, POS_ZERO),       # -inf clamped to [0, 1] -> +0.0
            ("v_sqrt_f16_e64 v0, v1 clamp", POS_INF, POS_ONE),        # +inf clamped to [0, 1] -> 1.0
            ("v_sqrt_f16_e64 v0, v1 clamp", NEG_ZERO, POS_ZERO),      # FClamp(-0.0, 0.0, 1.0) -> +0.0
            ("v_sqrt_f16_e64 v0, v1 mul:2", NEG_ZERO, NEG_ZERO),      # -0.0 * 2.0 -> -0.0
            ("v_exp_f16_e64 v0, v1 clamp", POS_INF, POS_ONE),         # +inf clamped to [0, 1] -> 1.0
            ("v_exp_f16_e64 v0, v1 clamp", NEG_INF, POS_ZERO),        # +0.0 clamped to [0, 1] -> +0.0
            ("v_sin_f16_e64 v0, v1 clamp", NEG_QUARTER, POS_ZERO),    # -1.0 clamped to [0, 1] -> +0.0
            ("v_sin_f16_e64 v0, v1 clamp", POS_QUARTER, POS_ONE),     # 1.0 clamped to [0, 1] -> 1.0
            ("v_cos_f16_e64 v0, v1 clamp", NEG_HALF, POS_ZERO),       # -1.0 clamped to [0, 1] -> +0.0
            ("v_cos_f16_e64 v0, v1 clamp", POS_ZERO, POS_ONE),        # 1.0 clamped to [0, 1] -> 1.0
            ("v_rcp_f16_e64 v0, v1 mul:2", POS_FOUR, POS_HALF),       # 0.25 * 2.0 -> 0.5
        ]
        for insn, src_h, want_h in cases:
            with self.subTest(insn=insn, src=hex(src_h)):
                got = run_ps_mc(
                    "ps_f16_trans_clamp_omod",
                    [
                        f"v_mov_b32 v0, {OLD:#x}",
                        f"v_mov_b32 v1, {src_h:#x}",
                        insn,
                    ],
                )
                self.assertEqual(got, OLD_HI | want_h)

    def _assert_sin_cos_inf_guard_dataflow(self, spv, ext_op):
        defs, consts, output_stores, reachable = _parse_spirv_ssa(spv)
        is_inf_nodes = [
            (dst, args[1])
            for dst, (opc, args) in defs.items()
            if opc == "OpIsInf" and len(args) >= 2 and args[0] == "%bool"
        ]
        self.assertTrue(is_inf_nodes, f"missing OpIsInf %bool in SPIR-V for {ext_op}")

        trig_nodes = [
            dst
            for dst, (opc, args) in defs.items()
            if opc == "OpExtInst" and len(args) >= 3 and args[0] == "%float" and args[2] == ext_op
        ]
        self.assertTrue(trig_nodes, f"missing GLSL {ext_op} in SPIR-V")

        matched = False
        for is_inf_id, src_id in is_inf_nodes:
            # 1) Pre-trig sanitization: OpSelect %float %is_inf +0.0 %src -> RoundEven -> Sin/Cos
            pre_selects = [
                dst
                for dst, (opc, args) in defs.items()
                if opc == "OpSelect"
                and len(args) >= 4
                and args[0] == "%float"
                and args[1] == is_inf_id
                and consts.get(args[2]) == 0
                and args[3] == src_id
            ]
            live_pre = []
            for psel in pre_selects:
                reach_ssa = reachable(psel, follow_memory=False)
                has_round = any(
                    defs[nid][0] == "OpExtInst" and len(defs[nid][1]) >= 3 and defs[nid][1][2] == "RoundEven"
                    for nid in reach_ssa
                    if nid in defs
                )
                if has_round and any(tid in reach_ssa for tid in trig_nodes):
                    live_pre.append(psel)
            if not live_pre:
                continue

            # 2) Post-trig canonical NaN select: OpSelect %float %is_inf %qnan %trig -> pack/store
            trig_reach = set()
            for tid in trig_nodes:
                trig_reach |= reachable(tid, follow_memory=False)

            post_selects = [
                dst
                for dst, (opc, args) in defs.items()
                if opc == "OpSelect"
                and len(args) >= 4
                and args[0] == "%float"
                and args[1] == is_inf_id
                and consts.get(args[2]) in (0x7FC00000, 0xFFC00000)
                and args[3] in trig_reach
            ]
            for sel_id in post_selects:
                reach_ssa = reachable(sel_id, follow_memory=False)
                reaches_pack = any(
                    defs[nid][0] == "OpBitcast" and defs[nid][1][0] == "%uint"
                    or (defs[nid][0] == "OpExtInst" and len(defs[nid][1]) >= 3 and defs[nid][1][2] == "PackHalf2x16")
                    for nid in reach_ssa
                    if nid in defs
                )
                reach_mem = reachable(sel_id, follow_memory=True)
                reaches_out = any(val in reach_mem for _, val in output_stores)
                if reaches_pack and reaches_out:
                    matched = True
                    break
            if matched:
                break

        self.assertTrue(
            matched,
            f"expected live OpIsInf -> OpSelect(+0) -> RoundEven -> {ext_op} -> "
            f"OpSelect(qNaN) -> pack -> OpStore chain",
        )

    def test_spirv_guards_sanitize_before_undefined_ops(self):
        # Verify emitted SPIR-V guards +-inf before Sin/Cos/turn_angle (both pre- and post-OpSelects live)
        # and -0.0 before Sqrt.
        for op, ext_op in (("v_sin_f16", "Sin"), ("v_cos_f16", "Cos")):
            with self.subTest(op=op):
                disasm = f"; ps_{op}_guard\n" + _assemble_gfx1013([
                    "v_mov_b32 v0, 0",
                    "v_mov_b32 v1, 0x7c00",
                    f"{op} v0, v1",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ])
                spv = translate_listing(f"ps_{op}_guard", disasm.encode("utf-8"))
                _validate_spirv(spv)
                self._assert_sin_cos_inf_guard_dataflow(spv, ext_op)

        disasm_sqrt = "; ps_sqrt_guard\n" + _assemble_gfx1013([
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x8000",
            "v_sqrt_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        spv_sqrt = translate_listing("ps_sqrt_guard", disasm_sqrt.encode("utf-8"))
        _validate_spirv(spv_sqrt)
        self.assertIn("OpIEqual %bool", spv_sqrt)
        self.assertLess(spv_sqrt.index("OpIEqual %bool"), spv_sqrt.index("Sqrt"))

    def test_op_sel_on_five_ops_with_specials(self):
        # llvm-mc 21 rejects op_sel on VOP1 _e64, so test op_sel via hand-written listings + spirv-val.
        # v1 holds (hi, lo); verify both halves and preservation of the untouched destination half.
        op_cases = [
            ("v_log_f16", (NEG_INF << 16) | POS_ZERO, NEG_QNAN, NEG_INF),
            ("v_rcp_f16", (NEG_ZERO << 16) | POS_ZERO, NEG_INF, POS_INF),
            ("v_rsq_f16", (NEG_ONE << 16) | NEG_ZERO, NEG_QNAN, NEG_INF),
            ("v_sqrt_f16", (NEG_INF << 16) | NEG_ZERO, NEG_QNAN, NEG_ZERO),
            ("v_exp_f16", (NEG_INF << 16) | POS_INF, POS_ZERO, POS_INF),
        ]
        for op, v1_val, want_hi_src, want_lo_src in op_cases:
            for src_hi in (0, 1):
                want_h = want_hi_src if src_hi else want_lo_src
                for dst_hi in (0, 1):
                    want = ((want_h << 16) | OLD_LO) if dst_hi else (OLD_HI | want_h)
                    insn = f"{op}_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
                    with self.subTest(insn=insn):
                        got = run_ps_listing(
                            f"ps_{op}_opsel",
                            [
                                f"v_mov_b32 v0, {OLD:#x}",
                                f"v_mov_b32 v1, {v1_val:#x}",
                                insn,
                            ],
                        )
                        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
