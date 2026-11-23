"""Tests for tools/spirv_mem2reg.py (pruned SSA Mem2Reg, GVN-Hoist, branch narrowing, mark-and-sweep DCE, #884)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cf_diff  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mem2reg  # noqa: E402
import spirv_opt  # noqa: E402
import support  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _assert_valid_spirv(spvasm: str) -> None:
    """Validate binary assembly (Vulkan 1.2), SSA dominance, OpPhi placement/predecessors,
    structured merge adjacency, OpVariable entry placement, and decoration targets."""
    spirv_asm.assemble(spvasm, target_env="vulkan1.2")

    lines = spvasm.splitlines()
    global_defs: set[str] = set()
    func_ranges: list[tuple[int, int]] = []
    fn_start = None
    decor_targets: list[tuple[str, str]] = []

    for i, raw in enumerate(lines):
        p = spirv_opt._parse_line(raw)
        if fn_start is None:
            if p is None:
                s = raw.lstrip()
                if "OpExtInstImport" in s and "=" in s:
                    global_defs.add(s.split("=", 1)[0].strip())
                elif s.startswith(("OpName ", "OpDecorateString ", "OpDecorateStringGOOGLE ")):
                    toks = s.split(None, 2)
                    if len(toks) >= 2 and toks[1].startswith("%"):
                        decor_targets.append((toks[0], toks[1]))
                continue
            res, op, args = p
            if op == "OpFunction":
                fn_start = i
                if res:
                    global_defs.add(res)
                continue
            if res:
                if res in global_defs:
                    raise AssertionError(f"duplicate global SSA ID {res}")
                global_defs.add(res)
            if op in ("OpDecorate", "OpDecorateId") and args and args[0].startswith("%"):
                decor_targets.append((op, args[0]))
            elif op == "OpGroupDecorate" and len(args) >= 2:
                for t in args[1:]:
                    if t.startswith("%"):
                        decor_targets.append((op, t))
        else:
            if p is not None and p[1] == "OpFunctionEnd":
                func_ranges.append((fn_start, i + 1))
                fn_start = None

    all_defs = set(global_defs)
    for start, end in func_ranges:
        insts = []
        for raw in lines[start:end]:
            p = spirv_opt._parse_line(raw)
            if p is None:
                insts.append([None, None, None, raw])
            else:
                insts.append([p[0], p[1], p[2], raw])

        blocks, succs, _all_preds, rpo, idom, _dom_children = spirv_opt._build_cfg_and_dom(insts)
        depth = spirv_mem2reg._dom_depths(rpo, idom)
        blabels = [insts[b_insts[0]][0] for b_insts in blocks]
        l2b = {lbl: b for b, lbl in enumerate(blabels)}

        fn_params: set[str] = set()
        first_blk = blocks[0][0] if blocks else len(insts)
        for idx in range(first_blk):
            res, op, _args, _raw = insts[idx]
            if op == "OpFunctionParameter" and res:
                if res in all_defs:
                    raise AssertionError(f"duplicate parameter SSA ID {res}")
                fn_params.add(res)
                all_defs.add(res)

        def_pos: dict[str, tuple[int, int]] = {}
        for b, b_insts in enumerate(blocks):
            seen_non_var = False
            seen_non_phi = False
            for order, idx in enumerate(b_insts):
                res, op, args, raw = insts[idx]
                if op is None:
                    continue
                if res:
                    if res in all_defs:
                        raise AssertionError(f"duplicate SSA ID {res}: {raw}")
                    all_defs.add(res)
                    def_pos[res] = (b, order)
                if order == 0:
                    if op != "OpLabel":
                        raise AssertionError(f"block {b} does not start with OpLabel: {raw}")
                    continue
                if op == "OpVariable":
                    if b != 0 or seen_non_var:
                        raise AssertionError(f"OpVariable outside top of entry block: {raw}")
                else:
                    seen_non_var = True
                if op == "OpPhi":
                    if seen_non_phi:
                        raise AssertionError(f"OpPhi after non-OpPhi in {blabels[b]}: {raw}")
                else:
                    seen_non_phi = True
                if op in ("OpSelectionMerge", "OpLoopMerge"):
                    nxt_op = None
                    for k in range(order + 1, len(b_insts)):
                        if insts[b_insts[k]][1] is not None:
                            nxt_op = insts[b_insts[k]][1]
                            break
                    if nxt_op not in ("OpBranch", "OpBranchConditional", "OpSwitch"):
                        raise AssertionError(f"{op} not followed by branch/switch in {blabels[b]}: {nxt_op}")

        rpreds = [set() for _ in range(len(blocks))]
        for b in rpo:
            for s in succs[b]:
                rpreds[s].add(b)

        for b in rpo:
            # SPIR-V 2.16.1: every dominating block must appear earlier in block text order.
            if b != 0 and idom.get(b, 0) > b:
                raise AssertionError(
                    f"dominating block {blabels[idom[b]]} appears after dominated block {blabels[b]}"
                )
            for order, idx in enumerate(blocks[b]):
                res, op, args, raw = insts[idx]
                if op is None or not args or op == "OpLabel":
                    continue
                if op == "OpLoopMerge":
                    # SPIR-V 2.16.1: a loop header must have at most one back-edge from inside the loop.
                    back_edges = [p for p in rpreds[b] if spirv_mem2reg._dominates(b, p, idom, depth)]
                    if len(back_edges) > 1:
                        raise AssertionError(
                            f"loop header {blabels[b]} has {len(back_edges)} back-edges (must be <= 1)"
                        )
                if op == "OpPhi":
                    pairs = [(args[k], args[k + 1]) for k in range(1, len(args), 2)]
                    pred_lbls = [p for _, p in pairs]
                    expected_lbls = {blabels[p] for p in rpreds[b]}
                    if len(pred_lbls) != len(set(pred_lbls)) or set(pred_lbls) != expected_lbls:
                        raise AssertionError(
                            f"OpPhi predecessors mismatch in {blabels[b]}: got {pred_lbls}, expected {expected_lbls}"
                        )
                    for val_tok, pred_lbl in pairs:
                        if val_tok in global_defs or val_tok in fn_params:
                            continue
                        if val_tok not in def_pos:
                            raise AssertionError(f"undefined OpPhi value {val_tok}: {raw}")
                        vb, _ = def_pos[val_tok]
                        pb = l2b[pred_lbl]
                        if not spirv_mem2reg._dominates(vb, pb, idom, depth):
                            raise AssertionError(
                                f"OpPhi value {val_tok} ({blabels[vb]}) does not dominate pred {pred_lbl}: {raw}"
                            )
                elif op in ("OpBranch", "OpBranchConditional", "OpSwitch", "OpSelectionMerge", "OpLoopMerge"):
                    val_toks = [args[0]] if op in ("OpBranchConditional", "OpSwitch") else []
                    for tok in val_toks:
                        if tok.startswith("%") and tok not in global_defs and tok not in fn_params:
                            if tok not in def_pos:
                                raise AssertionError(f"undefined SSA use {tok}: {raw}")
                            db, dord = def_pos[tok]
                            if not ((db == b and dord < order)
                                    or (db != b and spirv_mem2reg._dominates(db, b, idom, depth))):
                                raise AssertionError(f"SSA dominance violated for {tok}: {raw}")
                else:
                    start_k = 1 if res is not None else 0
                    for tok in args[start_k:]:
                        if not tok.startswith("%") or tok in global_defs or tok in fn_params:
                            continue
                        if tok not in def_pos:
                            raise AssertionError(f"undefined SSA use {tok}: {raw}")
                        db, dord = def_pos[tok]
                        if not ((db == b and dord < order)
                                or (db != b and spirv_mem2reg._dominates(db, b, idom, depth))):
                            raise AssertionError(f"SSA dominance violated for {tok}: {raw}")

    for op, tgt in decor_targets:
        if tgt not in all_defs:
            raise AssertionError(f"dangling {op} target {tgt}")

    if SPIRV_AS and SPIRV_VAL:
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


def _count_ops(spvasm: str) -> tuple[Counter, list[str]]:
    ops: Counter = Counter()
    func_vars: list[str] = []
    for ln in spvasm.splitlines():
        p = spirv_opt._parse_line(ln)
        if p is None:
            continue
        res, op, args = p
        ops[op] += 1
        if op == "OpVariable" and len(args) >= 2 and args[1] == "Function" and res:
            func_vars.append(res)
    return ops, func_vars


class SpirvMem2RegTests(unittest.TestCase):
    def test_mem2reg_and_gvn_hoist_on_screenprobes_shaped_compute(self):
        lines = [
            "s_load_dwordx4 s[4:7], s[0:1], 0x0",
            "s_load_dwordx4 s[8:11], s[0:1], 0x10",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, v0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0x3f800000",
            "s_mov_b32 s12, 0",
        ]
        for step in range(12):
            lines += [
                f"v_cmp_lt_u32_e64 s[14:15], v1, {step + 10}",
                "s_and_saveexec_b64 s[16:17], s[14:15]",
                "s_cbranch_execz 10",
                "v_lshlrev_b32_e32 v10, 2, v1",
                f"v_add_nc_u32_e32 v11, v10, {step & 0x3f}",
                "buffer_load_dword v12, v11, s[4:7], 0 offen",
                "s_waitcnt vmcnt(0)",
                "v_add_f32_e32 v2, v2, v12",
                "v_mul_f32_e32 v13, v12, v3",
                "v_add_f32_e32 v3, v3, v13",
                "v_xor_b32_e32 v14, v10, v11",
                "v_add_nc_u32_e32 v14, v14, 1",
                "s_or_b64 exec, exec, s[16:17]",
            ]
        lines += [
            "s_add_u32 s12, s12, 1",
            "v_add_f32_e32 v2, v2, v3",
            "v_mul_f32_e32 v3, v3, 0.5",
            "s_cmp_lt_u32 s12, 8",
            "s_cbranch_scc1 -5",
            "v_lshlrev_b32_e32 v15, 2, v0",
            "buffer_store_dword v2, v15, s[8:11], 0 offen",
            "s_endpgm",
        ]

        def enc(ln: str) -> str:
            op = ln.split()[0]
            if op.startswith(("buffer_", "s_load", "v_cmp_lt_u32_e64")):
                return "[0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]"
            return "[0x00,0x00,0x00,0x00]"

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "disasm"))
            os.makedirs(os.path.join(d, "fspx"))
            sp = os.path.join(d, "disasm", "cs_m2r_sp.s")
            with open(sp, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_m2r_sp  code 0x0..0x2000, gfx1013, 0 invalid words\n")
                for ln in lines:
                    f.write(f"\t{ln} ; encoding: {enc(ln)}\n")
            with open(os.path.join(d, "fspx", "cs_m2r_sp.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": (16 << 1) | (16 << 15), "threads": [64, 1, 1]}, f)

            old_w64, old_opt = rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV
            try:
                for mode in (True, False):
                    rdna2spirv.WAVE64_EMULATE = mode
                    rdna2spirv.OPT_SPIRV = False
                    raw = rdna2spirv.translate(sp)
                    opt = spirv_opt.optimize(raw)
                    _assert_valid_spirv(opt)
                    raw_ops, _ = _count_ops(raw)
                    opt_ops, opt_fvars = _count_ops(opt)
                    # Every register array / scalar state Function variable must be fully promoted and removed.
                    self.assertEqual(opt_fvars, [], f"mode={mode} left Function vars: {opt_fvars}")
                    # Descriptor unpacking across the 12 guarded buffer_loads must collapse via GVN-Hoist.
                    self.assertLess(opt_ops["OpBitFieldUExtract"], raw_ops["OpBitFieldUExtract"] // 4)
                    # Register stores are completely gone; only the real PSB / WG init stores remain.
                    self.assertLessEqual(opt_ops["OpStore"], 2)
            finally:
                rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV = old_w64, old_opt

    def test_loop_swap_parallel_phi_and_dead_phi_cycle_dce(self):
        # Loop that swaps v1 and v2 each iteration (testing parallel OpPhi semantics on backedges)
        # while maintaining a dead self-updating loop counter in v14 (must be deleted by mark-and-sweep DCE).
        lines = [
            "v_mov_b32_e32 v1, 10 ; encoding: [0x8a,0x02,0x02,0x7e]",
            "v_mov_b32_e32 v2, 20 ; encoding: [0x94,0x02,0x04,0x7e]",
            "v_mov_b32_e32 v14, 7 ; encoding: [0x87,0x02,0x1c,0x7e]",
            "s_mov_b32 s4, 0 ; encoding: [0x80,0x03,0x84,0xbe]",
            # loop header (pc = 16): 7 single-dword instructions -> back-branch offset -7
            "v_mov_b32_e32 v3, v1 ; encoding: [0x01,0x03,0x06,0x7e]",
            "v_add_nc_u32_e32 v1, v2, 1 ; encoding: [0x81,0x04,0x02,0x4a]",
            "v_add_nc_u32_e32 v2, v3, 2 ; encoding: [0x82,0x06,0x04,0x4a]",
            "v_add_nc_u32_e32 v14, v14, 3 ; encoding: [0x83,0x1c,0x1c,0x4a]",
            "s_add_u32 s4, s4, 1 ; encoding: [0x04,0x81,0x04,0x80]",
            "s_cmp_lt_u32 s4, 5 ; encoding: [0x04,0x85,0x04,0xbf]",
            "s_cbranch_scc1 -7 ; encoding: [0xf9,0xff,0x85,0xbf]",
            "exp mrt0 v1, v2, v3, v3 done vm ; encoding: [0x0f,0x18,0x00,0xf8,0x01,0x02,0x03,0x03]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        raw = support.translate_lines("ps_m2r_swap", lines)
        opt = spirv_opt.optimize(raw)
        _assert_valid_spirv(opt)
        self.assertEqual(support.mrt0_words(opt), support.mrt0_words(raw))
        # After 5 iterations of (v1, v2) -> (v2+1, v1+2) from (10, 20):
        # 1:(21,12), 2:(13,23), 3:(24,15), 4:(16,26), 5:(27,18), v3=16
        self.assertEqual(support.mrt0_words(opt), [27, 18, 16, 16])
        opt_ops, opt_fvars = _count_ops(opt)
        self.assertEqual(opt_fvars, [])
        # Only 2 OpIAdd remain (v1+1, v2+2; s_add_u32 uses OpIAddCarry); the dead v14+3 cycle must be gone.
        self.assertEqual(opt_ops["OpIAdd"], 2)

    def test_dynamic_vgpr_stays_in_memory_while_sgpr_and_active_promote(self):
        lines = [
            "v_mov_b32_e32 v1, 0x11111111 ; encoding: [0xff,0x02,0x02,0x7e,0x11,0x11,0x11,0x11]",
            "v_mov_b32_e32 v2, 0x22222222 ; encoding: [0xff,0x02,0x04,0x7e,0x22,0x22,0x22,0x22]",
            "s_mov_b32 s4, 1 ; encoding: [0x81,0x03,0x84,0xbe]",
            "s_cmp_eq_u32 s4, 1 ; encoding: [0x04,0x81,0x06,0xbf]",
            "s_cbranch_scc0 1 ; encoding: [0x01,0x00,0x84,0xbf]",
            "s_mov_b32 s5, 7 ; encoding: [0x87,0x03,0x85,0xbe]",
            "s_mov_b32 m0, 1 ; encoding: [0x81,0x03,0xfc,0xbe]",
            "v_movrels_b32_e32 v8, v1 ; encoding: [0x01,0x87,0x10,0x7e]",
            "v_add_nc_u32_e32 v9, s5, v8 ; encoding: [0x05,0x10,0x12,0x4a]",
            "exp mrt0 v8, v9, v1, v2 done vm ; encoding: [0x0f,0x18,0x00,0xf8,0x08,0x09,0x01,0x02]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        raw = support.translate_lines("ps_m2r_dyn", lines)
        opt = spirv_opt.optimize(raw)
        _assert_valid_spirv(opt)
        self.assertEqual(support.mrt0_words(opt), support.mrt0_words(raw))
        self.assertEqual(support.mrt0_words(opt), [0x22222222, 0x22222229, 0x11111111, 0x22222222])
        _, opt_fvars = _count_ops(opt)
        # %vgpr must remain because of v_movrels_b32, while %sgpr / %active / %m0 / %scc / %pc promote away.
        self.assertEqual(opt_fvars, ["%vgpr"])

    def test_nested_access_chain_across_blocks_promotes_cleanly(self):
        # Valid SPIR-V (dominance-ordered blocks) with nested OpAccessChain across blocks + diamond merge.
        spvasm = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
         %c2 = OpConstant %uint 2
       %arr1 = OpTypeArray %uint %c2
       %arr2 = OpTypeArray %arr1 %c2
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
    %_ptr_F2 = OpTypePointer Function %arr2
    %_ptr_F1 = OpTypePointer Function %arr1
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
         %u0 = OpConstant %uint 0
         %u1 = OpConstant %uint 1
         %u7 = OpConstant %uint 7
         %u9 = OpConstant %uint 9
       %null = OpConstantNull %arr2
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpVariable %_ptr_F2 Function %null
         %iv = OpLoad %uint %in0
       %cond = OpINotEqual %bool %iv %u0
               OpBranch %b1
         %b1 = OpLabel
         %p1 = OpAccessChain %_ptr_F1 %a %u1
               OpSelectionMerge %merge None
               OpBranchConditional %cond %then %else
       %then = OpLabel
         %p2 = OpAccessChain %_ptr_Fu %p1 %u0
               OpStore %p2 %u7
               OpBranch %merge
       %else = OpLabel
         %p3 = OpAccessChain %_ptr_Fu %p1 %u0
               OpStore %p3 %u9
               OpBranch %merge
      %merge = OpLabel
         %p4 = OpAccessChain %_ptr_Fu %p1 %u0
          %r = OpLoad %uint %p4
        %vec = OpCompositeConstruct %v4uint %r %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _assert_valid_spirv(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _assert_valid_spirv(opt)
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 1})["%out0"]), [7, 0, 0, 0])
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 0})["%out0"]), [9, 0, 0, 0])
        self.assertNotIn("%a = OpVariable", opt)

    def test_uninitialized_live_in_slot_does_not_synthesize_undef(self):
        # %x has no initializer and is stored only on one branch; Mem2Reg must leave %x in memory.
        spvasm = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
         %u0 = OpConstant %uint 0
         %u5 = OpConstant %uint 5
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %x = OpVariable %_ptr_Fu Function
         %iv = OpLoad %uint %in0
       %cond = OpINotEqual %bool %iv %u0
               OpSelectionMerge %merge None
               OpBranchConditional %cond %then %merge
       %then = OpLabel
               OpStore %x %u5
               OpBranch %merge
      %merge = OpLabel
          %v = OpLoad %uint %x
        %sel = OpSelect %uint %cond %v %u0
        %vec = OpCompositeConstruct %v4uint %sel %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _assert_valid_spirv(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _assert_valid_spirv(opt)
        self.assertNotIn("OpUndef", opt)
        self.assertIn("%x = OpVariable", opt)
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 1})["%out0"]), [5, 0, 0, 0])
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 0})["%out0"]), [0, 0, 0, 0])

    def test_branch_narrowing_and_boolean_absorption(self):
        spvasm = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
         %u0 = OpConstant %uint 0
         %u1 = OpConstant %uint 1
         %u2 = OpConstant %uint 2
         %u9 = OpConstant %uint 9
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %res = OpVariable %_ptr_Fu Function %u0
         %iv = OpLoad %uint %in0
         %c1 = OpINotEqual %bool %iv %u0
         %c2 = OpUGreaterThan %bool %iv %u2
        %and = OpLogicalAnd %bool %c1 %c2
        %abs = OpLogicalOr %bool %and %c1
               OpSelectionMerge %merge None
               OpBranchConditional %abs %then %else
       %then = OpLabel
         %s1 = OpSelect %uint %abs %u1 %u9
               OpStore %res %s1
               OpBranch %merge
       %else = OpLabel
         %s2 = OpSelect %uint %abs %u9 %u2
               OpStore %res %s2
               OpBranch %merge
      %merge = OpLabel
         %rv = OpLoad %uint %res
        %vec = OpCompositeConstruct %v4uint %rv %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _assert_valid_spirv(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _assert_valid_spirv(opt)
        ops, fvars = _count_ops(opt)
        # (c1 && c2) || c1 absorbs to c1; OpSelect inside %then/%else folds to %u1/%u2; %res promotes to OpPhi.
        self.assertEqual(ops["OpLogicalOr"], 0)
        self.assertEqual(ops["OpLogicalAnd"], 0)
        self.assertEqual(ops["OpSelect"], 0)
        self.assertEqual(fvars, [])
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 5})["%out0"]), [1, 0, 0, 0])
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 0})["%out0"]), [2, 0, 0, 0])

    def test_wide_switch_phi_cap_promotes_readonly_and_bounds_mutable(self):
        # 40-case OpSwitch converging at %sw_merge (41 preds > max_phi_preds=32) inside a structured loop:
        # %ro is written only in %entry (0 phis -> must promote), %mut is modified in cases (kept in memory).
        n_cases = 40
        consts = "\n".join(f"        %k{i} = OpConstant %uint {i}" for i in range(n_cases))
        sw_pairs = " ".join(f"{i} %case{i}" for i in range(n_cases))
        cases = []
        for i in range(n_cases):
            cases.append(f"""      %case{i} = OpLabel
      %r{i} = OpLoad %uint %ro
      %m{i} = OpLoad %uint %mut
      %a{i} = OpIAdd %uint %r{i} %m{i}
              OpStore %mut %a{i}
              OpBranch %sw_merge""")
        spvasm = f"""               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
{consts}
        %u40 = OpConstant %uint 40
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %ro = OpVariable %_ptr_Fu Function %k0
        %mut = OpVariable %_ptr_Fu Function %k0
         %iv = OpLoad %uint %in0
               OpStore %ro %iv
               OpBranch %head
       %head = OpLabel
         %mv = OpLoad %uint %mut
       %done = OpUGreaterThanEqual %bool %mv %u40
               OpLoopMerge %exit %cont None
               OpBranchConditional %done %exit %body
       %body = OpLabel
               OpSelectionMerge %sw_merge None
               OpSwitch %mv %sw_merge {sw_pairs}
{chr(10).join(cases)}
   %sw_merge = OpLabel
               OpBranch %cont
       %cont = OpLabel
               OpBranch %head
       %exit = OpLabel
        %vec = OpCompositeConstruct %v4uint %mv %k0 %k0 %k0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _assert_valid_spirv(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _assert_valid_spirv(opt)
        # %ro must be promoted out (0 phis needed); %mut must stay in memory rather than emitting a 41-pred OpPhi.
        self.assertNotIn("%ro = OpVariable", opt)
        self.assertIn("%mut = OpVariable", opt)
        self.assertNotIn("OpPhi", opt)

    def test_gvn_hoist_deduplicates_pure_and_skips_trapping_ops(self):
        spvasm = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
         %u0 = OpConstant %uint 0
         %u4 = OpConstant %uint 4
         %u8 = OpConstant %uint 8
        %u10 = OpConstant %uint 10
         %f1 = OpConstant %float 1.0
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %out = OpVariable %_ptr_Fu Function %u0
         %iv = OpLoad %uint %in0
       %cond = OpINotEqual %bool %iv %u0
               OpSelectionMerge %merge None
               OpBranchConditional %cond %then %else
       %then = OpLabel
         %e1 = OpBitFieldUExtract %uint %iv %u4 %u8
         %s1 = OpIAdd %uint %e1 %u10
         %fs = OpConvertUToF %float %s1
         %fi = OpConvertUToF %float %iv
         %fd = OpFDiv %float %fs %fi
         %d1 = OpConvertFToU %uint %fd
               OpStore %out %d1
               OpBranch %merge
       %else = OpLabel
         %e2 = OpBitFieldUExtract %uint %iv %u4 %u8
         %s2 = OpIAdd %uint %e2 %u10
        %fs2 = OpConvertUToF %float %s2
        %fd2 = OpFDiv %float %fs2 %f1
         %d2 = OpConvertFToU %uint %fd2
               OpStore %out %d2
               OpBranch %merge
      %merge = OpLabel
         %rv = OpLoad %uint %out
        %vec = OpCompositeConstruct %v4uint %rv %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _assert_valid_spirv(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _assert_valid_spirv(opt)
        ops, fvars = _count_ops(opt)
        # Pure chain (OpBitFieldUExtract + OpIAdd) hoists into %entry as 1 copy; OpFDiv never hoists (div by 0).
        self.assertEqual(ops["OpBitFieldUExtract"], 1)
        self.assertEqual(ops["OpIAdd"], 1)
        self.assertEqual(ops["OpFDiv"], 2)
        self.assertEqual(fvars, [])
        # When %iv == 0, %else runs and %then's OpFDiv by 0 must NOT have been hoisted into %entry!
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 0})["%out0"]), [10, 0, 0, 0])
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 0x10})["%out0"]), [0, 0, 0, 0])
        self.assertEqual(list(run_fragment(opt, inputs={"%in0": 0x02})["%out0"]), [5, 0, 0, 0])

    def test_duplicate_branch_edge_and_group_decorate_stay_valid(self):
        # 1) OpBranchConditional %c %m %m (duplicate edge to %m) must not insert an invalid OpPhi in %m.
        # 2) OpGroupDecorate targeting a Function variable must prevent promotion and keep decorations intact.
        spvasm = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
               OpDecorate %dg Volatile
         %dg = OpDecorationGroup
               OpGroupDecorate %dg %gv
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
         %u0 = OpConstant %uint 0
         %u3 = OpConstant %uint 3
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %gv = OpVariable %_ptr_Fu Function %u0
         %dv = OpVariable %_ptr_Fu Function %u0
         %iv = OpLoad %uint %in0
               OpStore %gv %iv
       %cond = OpINotEqual %bool %iv %u0
               OpSelectionMerge %m None
               OpBranchConditional %cond %m %m
          %m = OpLabel
               OpStore %dv %u3
         %g1 = OpLoad %uint %gv
         %d1 = OpLoad %uint %dv
        %vec = OpCompositeConstruct %v4uint %g1 %d1 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _assert_valid_spirv(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _assert_valid_spirv(opt)
        self.assertIn("%gv = OpVariable", opt)
        self.assertIn("OpGroupDecorate %dg %gv", opt)
        self.assertIn("%g1 = OpLoad %uint %gv", opt)

        # Also execute a duplicate-branch-edge module without OpDecorationGroup via spirv_mini_interp.
        spv_dup = "\n".join(
            ln for ln in spvasm.splitlines()
            if "OpDecorationGroup" not in ln and "OpGroupDecorate" not in ln and "%dg" not in ln
        ) + "\n"
        _assert_valid_spirv(spv_dup)
        opt_dup = spirv_opt.optimize(spv_dup)
        _assert_valid_spirv(opt_dup)
        self.assertEqual(list(run_fragment(opt_dup, inputs={"%in0": 11})["%out0"]), [11, 3, 0, 0])

    def test_cf_diff_wave32_and_wave64_with_opt_bit_exact_and_valid(self):
        old_opt = rdna2spirv.OPT_SPIRV
        try:
            rdna2spirv.OPT_SPIRV = True
            self.assertEqual(cf_diff.main(["--programs", "25", "--wave", "both", "--seed", "20261123"]), 0)
            self.assertEqual(
                cf_diff.main(["--programs", "15", "--wave", "both", "--dispatch-loop", "--seed", "20261124"]), 0
            )
        finally:
            rdna2spirv.OPT_SPIRV = old_opt


if __name__ == "__main__":
    unittest.main()
