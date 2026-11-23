"""Tests for outlining image_bvh(_64)_intersect_ray and f32_{sub,mul}_rtz into OpFunction helpers (#884, #4315)."""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_opt  # noqa: E402
import support  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _validate(spvasm: str) -> bytes:
    binary = spirv_asm.assemble(spvasm, target_env="vulkan1.2")
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
    return binary


def _box32_node(children, boxes) -> list[int]:
    """Build a 128-byte RDNA2 BOX32 node (4 child ids + 4 * 6 f32 min/max bounds)."""
    words = [c & 0xFFFFFFFF for c in children]
    for b in boxes:
        words.extend(_f32_bits(x) for x in b)
    while len(words) < 32:
        words.append(0)
    return words


def _tri0_node(v0, v1, v2, tri_id_word: int = 0x09) -> list[int]:
    """Build a 64-byte RDNA2 TRI0 node (3 vertices * 3 floats + padding + triangle_id word at +60)."""
    words = []
    for v in (v0, v1, v2):
        words.extend(_f32_bits(x) for x in v)
    while len(words) < 15:
        words.append(0)
    words.append(tri_id_word & 0xFFFFFFFF)
    return words


class ImageBvhOutlineTests(unittest.TestCase):
    def test_multi_callsite_mixed_32_64_outlines_and_executes_exact(self):
        # Single PS invoking both image_bvh_intersect_ray (32-bit, box32) and image_bvh64_intersect_ray
        # (64-bit, tri0 + box32 + unknown node type 6) in one module.
        base_addr = 0x10000
        box_node_addr = base_addr + 0x100     # offset 0x100 -> node id = (0x100 >> 3) | 5 = 0x25 (BOX32)
        tri_node_addr = base_addr + 0x200     # offset 0x200 -> node id = (0x200 >> 3) | 0 = 0x40 (TRI0)
        oth_node_id = (0x300 >> 3) | 6        # node type 6 (instance/other) -> 4x 0xFFFFFFFF

        box_words = _box32_node(
            [0x11111111, 0x22222222, 0x33333333, 0x44444444],
            [
                (-1.0, -1.0, 2.0, 1.0, 1.0, 3.0),   # hit at t=2
                (-1.0, -1.0, 1.0, 1.0, 1.0, 1.5),   # closer hit at t=1 (sorted first when sort=1)
                (5.0, 5.0, 5.0, 6.0, 6.0, 6.0),     # miss
                (2.0, 2.0, 2.0, 1.0, 1.0, 1.0),     # empty box -> miss
            ],
        )
        tri_words = _tri0_node((-1.0, -1.0, 4.0), (1.0, -1.0, 4.0), (0.0, 1.0, 4.0))

        mem = {}
        for i, w in enumerate(box_words):
            mem[box_node_addr + 4 * i] = w
        for i, w in enumerate(tri_words):
            mem[tri_node_addr + 4 * i] = w

        # BVH descriptor in s[4:7]: base256 in bits [39:0], box_sort_enable = bit 63 (word1 bit 31).
        base256 = base_addr >> 8
        d0 = base256 & 0xFFFFFFFF
        d1 = ((base256 >> 32) & 0xFF) | 0x80000000

        lines = [
            f"s_mov_b32 s4, {d0:#x}",
            f"s_mov_b32 s5, {d1:#x}",
            "s_mov_b32 s6, 0",
            "s_mov_b32 s7, 0",
            # Ray: tmax=10.0, origin=(0,0,0), dir=(0,0,1), inv=(inf,inf,1) -> use large finite inv for x/y
            f"v_mov_b32_e32 v2, {_f32_bits(10.0):#x}",
            "v_mov_b32_e32 v3, 0",
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v5, 0",
            "v_mov_b32_e32 v6, 0",
            "v_mov_b32_e32 v7, 0",
            f"v_mov_b32_e32 v8, {_f32_bits(1.0):#x}",
            f"v_mov_b32_e32 v9, {_f32_bits(1e9):#x}",
            f"v_mov_b32_e32 v10, {_f32_bits(1e9):#x}",
            f"v_mov_b32_e32 v11, {_f32_bits(1.0):#x}",
            # Call 1 (32-bit): BOX32 node (sorted: child 1 then child 0, then 2 misses)
            f"v_mov_b32_e32 v1, {(0x100 >> 3) | 5:#x}",
            "image_bvh_intersect_ray v[20:23], v[1:11], s[4:7]",
            "s_waitcnt vmcnt(0)",
            # Call 2 (64-bit): TRI0 node (hits at t=4.0 -> det=4.0, T=16.0 -> t=T/det=4.0)
            f"v_mov_b32_e32 v12, {(0x200 >> 3) | 0:#x}",
            "v_mov_b32_e32 v13, 0",
            "image_bvh64_intersect_ray v[24:27], [v12, v13, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], s[4:7]",
            "s_waitcnt vmcnt(0)",
            # Call 3 (32-bit): unknown node type 6 -> 4x 0xFFFFFFFF
            f"v_mov_b32_e32 v1, {oth_node_id:#x}",
            "image_bvh_intersect_ray v[28:31], v[1:11], s[4:7]",
            "s_waitcnt vmcnt(0)",
            # Export: [box sorted 0, box sorted 1, tri T word, oth word 0]
            "exp mrt0 v20, v21, v24, v28 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bvh_outline_mix", lines)
        _validate(spv)

        # Must emit exactly one helper each for %image_bvh_fn, %f32_sub_rtz_fn, %f32_mul_rtz_fn.
        self.assertEqual(spv.count("%image_bvh_fn = OpFunction"), 1)
        self.assertEqual(spv.count("%f32_sub_rtz_fn = OpFunction"), 1)
        self.assertEqual(spv.count("%f32_mul_rtz_fn = OpFunction"), 1)
        # All 3 callsites in %main call the single shared %image_bvh_fn, with zero OpFMul/OpSelectionMerge in %main.
        main_body = spv[spv.index("%main = OpFunction"):]
        self.assertEqual(main_body.count("OpFunctionCall %v4uint_bvh %image_bvh_fn"), 3)
        self.assertNotIn("OpFMul", main_body)
        self.assertNotIn("OpSelectionMerge", main_body)
        # Every float arithmetic op across the helpers must carry NoContraction.
        float_ids = set(re.findall(r"^\s*(%\w+) = Op(?:FMul|FAdd|FSub|FDiv|FNegate|FRem|FMod) ", spv, re.M))
        nc_ids = set(re.findall(r"OpDecorate (%\w+) NoContraction", spv))
        self.assertTrue(float_ids.issubset(nc_ids), f"missing NoContraction: {float_ids - nc_ids}")

        # Export: [box sorted 0, box sorted 1, tri T word (-16.0 with det=-4.0 -> t=4.0), oth word 0]
        expected = [0x22222222, 0x11111111, _f32_bits(-16.0), 0xFFFFFFFF]
        out_raw = list(run_fragment(spv, memory=mem)["%out_mrt0"])
        self.assertEqual(out_raw, expected)

        # Also verify parity after --opt.
        opt = spirv_opt.optimize(spv)
        _validate(opt)
        out_opt = list(run_fragment(opt, memory=mem)["%out_mrt0"])
        self.assertEqual(out_opt, expected)

    def test_ten_live_calls_stay_compact_and_valid_with_emulate_and_ubo(self):
        lines = [
            "s_load_dwordx4 s[4:7], s[0:1], 0x0",
            "s_load_dwordx4 s[8:11], s[0:1], 0x10",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v20, 0",
        ]
        for i in range(10):
            op = "image_bvh64_intersect_ray v[4:7], v[8:19], s[4:7]" if (i % 2) else \
                "image_bvh_intersect_ray v[4:7], v[8:18], s[4:7]"
            lines += [
                f"v_add_nc_u32_e32 v8, v0, {i}",
                op,
                "s_waitcnt vmcnt(0)",
                "v_xor_b32_e32 v20, v20, v4",
                "v_xor_b32_e32 v20, v20, v5",
                "v_xor_b32_e32 v20, v20, v6",
                "v_xor_b32_e32 v20, v20, v7",
            ]
        lines += [
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v20, v21, s[8:11], 0 offen",
            "s_endpgm",
        ]

        def enc(ln: str) -> str:
            op = ln.split()[0]
            if op.startswith(("buffer_", "s_load", "image_")):
                return "[0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]"
            return "[0x00,0x00,0x00,0x00]"

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "disasm"))
            os.makedirs(os.path.join(d, "fspx"))
            sp = os.path.join(d, "disasm", "cs_bvh_10.s")
            with open(sp, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_bvh_10  code 0x0..0x2000, gfx1013, 0 invalid words\n")
                for ln in lines:
                    f.write(f"\t{ln} ; encoding: {enc(ln)}\n")
            with open(os.path.join(d, "fspx", "cs_bvh_10.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": (24 << 1) | (16 << 15), "threads": [64, 1, 1]}, f)

            old_w64, old_opt, old_ubo = rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV, rdna2spirv.PC_TAIL_UBO
            try:
                for ubo in (False, True):
                    rdna2spirv.WAVE64_EMULATE = True
                    rdna2spirv.OPT_SPIRV = False
                    rdna2spirv.PC_TAIL_UBO = ubo
                    raw = rdna2spirv.translate(sp)
                    raw_bin = _validate(raw)
                    # Unoutlined 10x image_bvh was ~628 KB / ~35,000 lines; outlined must stay under 60 KB / 2,500 lines.
                    self.assertLess(len(raw_bin), 60_000, f"ubo={ubo} binary too large: {len(raw_bin)}")
                    self.assertLess(len(raw.splitlines()), 2_500)
                    ops = Counter(p[1] for l in raw.splitlines() if (p := spirv_opt._parse_line(l)))
                    self.assertEqual(ops["OpFunction"], 4)  # %f32_sub_rtz_fn, %f32_mul_rtz_fn, %image_bvh_fn, %main
                    opt = spirv_opt.optimize(raw)
                    _validate(opt)
            finally:
                rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV, rdna2spirv.PC_TAIL_UBO = old_w64, old_opt, old_ubo

    def test_v4uint_st_name_preserved_with_typed_image_and_bvh(self):
        # SHADER-1016 regression guard: adding %v4uint_bvh must not steal %v4uint_st's name when typed images exist.
        lines = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 0",
            "s_mov_b32 s7, 0",
            "s_mov_b32 s8, 0",
            "s_mov_b32 s9, 0",
            "s_mov_b32 s10, 0",
            "s_mov_b32 s11, 0",
            "image_load v[20:23], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "image_bvh_intersect_ray v[24:27], v[1:11], s[4:7]",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v20, v24, v25, v26 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bvh_img_st", lines)
        _validate(spv)
        # dedupe_types keeps %v4uint_st as the canonical uvec4 type when typed image loads are present.
        self.assertIn("%v4uint_st = OpTypeVector %uint 4", spv)
        self.assertIn("%image_bvh_fn = OpFunction %v4uint_st None %fn_image_bvh", spv)


if __name__ == "__main__":
    unittest.main()
