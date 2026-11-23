"""Tests for RDNA2 image_msaa_load (MIMG 0x80) and MSAA image_load in tools/rdna2spirv.py.

Verifies:
- 2D_MSAA (x, y, sample_index) -> bindings 20/22/24 (float/uint/sint), OpTypeImage ... 2D 0 0 1 1 Unknown
- 2D_MSAA_ARRAY (x, y, slice, sample_index) -> bindings 21/23/25 (float/uint/sint), OpTypeImage ... 2D 0 1 1 1 Unknown
- Result semantics separation:
  - image_msaa_load (MIMG 0x80): fetches up to 4 consecutive samples (sample_idx + 0..3) of channel 0 (red)
    for enabled dmask bits 0..3, recovering distinct single-channel values across samples 0..3.
  - ordinary image_load on 2D_MSAA / 2D_MSAA_ARRAY: fetches 1 sample at sample_idx and extracts RGBA channels 0..3.
- dmask (0xf, 0x1, 0x5), unorm, d16, a16, NSA address lists, and binary decoding via rdna2_decode
- Unsupported forms (non-MSAA dim, glc, tfe/lwe, non-VGPR dest, short addr) raise Unsupported
- Every translated module passes spirv-as and spirv-val --target-env vulkan1.2.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import support
import rdna2_decode
import rdna2spirv

EPILOGUE = ["s_waitcnt vmcnt(0)", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
PROLOGUE = ["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 3", "v_mov_b32 v7, 0"]


def translate(lines, stage="ps", prologue=None):
    return support.translate_lines(f"{stage}_msaa_load", (PROLOGUE if prologue is None else prologue) + lines + EPILOGUE)


def find_spirv_tools():
    sdk_bin = r"C:\VulkanSDK\1.4.363.0\Bin"
    spirv_as = shutil.which("spirv-as") or (os.path.join(sdk_bin, "spirv-as.exe") if os.path.isfile(os.path.join(sdk_bin, "spirv-as.exe")) else None)
    spirv_val = shutil.which("spirv-val") or (os.path.join(sdk_bin, "spirv-val.exe") if os.path.isfile(os.path.join(sdk_bin, "spirv-val.exe")) else None)
    return spirv_as, spirv_val


class Module:
    def __init__(self, text):
        self.text = text
        self.defs = {}
        self.ops = []
        for line in text.splitlines():
            m = re.match(r"\s*(%\w+) = (Op\w+)\s*(.*)$", line)
            if m:
                self.defs[m.group(1)] = (m.group(2), m.group(3).split())
                self.ops.append((m.group(1), m.group(2), m.group(3).split()))
            else:
                w = line.split()
                if w:
                    self.ops.append((None, w[0], w[1:]))
        self.caps = {a[0] for _, op, a in self.ops if op == "OpCapability"}
        self.bindings = {
            a[0]: int(a[2])
            for _, op, a in self.ops
            if op == "OpDecorate" and len(a) >= 3 and a[1] == "Binding"
        }
        self.dsets = {
            a[0]: int(a[2])
            for _, op, a in self.ops
            if op == "OpDecorate" and len(a) >= 3 and a[1] == "DescriptorSet"
        }

    def variable(self, image_id):
        op, a = self.defs[image_id]
        assert op == "OpLoad", (image_id, op)
        op, a = self.defs[a[1]]
        assert op == "OpAccessChain", (image_id, op)
        return a[1]

    def fetches(self):
        """List of (result_id, result_type, var_name, coord_id, operands_list) for each OpImageFetch."""
        out = []
        for res_id, op, a in self.ops:
            if op == "OpImageFetch":
                out.append((res_id, a[0], self.variable(a[1]), a[2], a[3:]))
        return out

    def extracts_of(self, src_id):
        """List of (result_id, result_type, component_idx) for each OpCompositeExtract reading src_id."""
        out = []
        for res_id, op, a in self.ops:
            if op == "OpCompositeExtract" and a[1] == src_id:
                out.append((res_id, a[0], int(a[2])))
        return out

    def bitcasts_of(self, src_id):
        """List of (result_id, result_type) for each OpBitcast reading src_id."""
        return [(res_id, a[0]) for res_id, op, a in self.ops if op == "OpBitcast" and a[1] == src_id]

    def vgpr_of(self, val_id):
        """Trace an SSA value back through OpBitcast / OpLoad / OpAccessChain to its VGPR index, or None."""
        cur = val_id
        for _ in range(4):
            if cur not in self.defs:
                return None
            op, a = self.defs[cur]
            if op == "OpBitcast":
                cur = a[1]
            elif op == "OpLoad":
                cur = a[1]
            elif op == "OpAccessChain" and a[1] == "%vgpr":
                idx_const = a[2]
                cop, ca = self.defs[idx_const]
                assert cop == "OpConstant", (idx_const, cop)
                return int(ca[1])
            else:
                return None
        return None

    def sample_vgpr_offset(self, val_id):
        """Trace a Sample operand to (vgpr_index, constant_offset), supporting direct VGPR or OpIAdd %uint v + k."""
        v = self.vgpr_of(val_id)
        if v is not None:
            return (v, 0)
        if val_id in self.defs:
            op, a = self.defs[val_id]
            if op == "OpIAdd":
                base = self.vgpr_of(a[1])
                cop, ca = self.defs[a[2]]
                if base is not None and cop == "OpConstant":
                    return (base, int(ca[1]))
        return None


class ImageMsaaLoadTests(unittest.TestCase):
    def test_2d_msaa_bindings_types_and_sample_operand(self):
        m = Module(translate([
            "image_msaa_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
        ]))
        # Bindings 20 (float), 22 (uint), 24 (sint) in set 0
        self.assertEqual(m.bindings["%msv0_0"], 20)
        self.assertEqual(m.bindings["%msv1_0"], 22)
        self.assertEqual(m.bindings["%msv2_0"], 24)
        for cls in range(3):
            self.assertEqual(m.dsets[f"%msv{cls}_0"], 0)
        # OpTypeImage: 2D, Depth=0, Arrayed=0, MS=1, Sampled=1, Format=Unknown
        self.assertEqual(m.defs["%msimg0_0"], ("OpTypeImage", ["%float", "2D", "0", "0", "1", "1", "Unknown"]))
        self.assertEqual(m.defs["%msimg1_0"], ("OpTypeImage", ["%uint", "2D", "0", "0", "1", "1", "Unknown"]))
        self.assertEqual(m.defs["%msimg2_0"], ("OpTypeImage", ["%int", "2D", "0", "0", "1", "1", "Unknown"]))
        # No storage image capabilities
        self.assertNotIn("StorageImageReadWithoutFormat", m.caps)
        self.assertNotIn("StorageImageWriteWithoutFormat", m.caps)
        # 3 numeric classes x 4 consecutive samples (v6 + 0..3), each extracting channel 0
        fetches = m.fetches()
        self.assertEqual(len(fetches), 12)
        by_class = {}
        for res_id, ty, var, coord_id, operands in fetches:
            self.assertEqual(operands[0], "Sample")
            cop, ca = m.defs[coord_id]
            self.assertEqual(cop, "OpCompositeConstruct")
            self.assertEqual(ca[0], "%v2int")
            self.assertEqual([m.vgpr_of(c) for c in ca[1:]], [4, 5])
            by_class.setdefault((ty, var), []).append(m.sample_vgpr_offset(operands[1]))
            # Each fetch extracts only channel 0 (red)
            extracts = m.extracts_of(res_id)
            self.assertEqual([comp for _, _, comp in extracts], [0])
        self.assertEqual(
            by_class,
            {
                ("%v4float", "%msv0_0"): [(6, 0), (6, 1), (6, 2), (6, 3)],
                ("%v4uint_st", "%msv1_0"): [(6, 0), (6, 1), (6, 2), (6, 3)],
                ("%v4int_st", "%msv2_0"): [(6, 0), (6, 1), (6, 2), (6, 3)],
            },
        )

    def test_2d_msaa_array_bindings_types_and_sample_operand(self):
        m = Module(translate([
            "image_msaa_load v[0:3], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm",
        ]))
        # Bindings 21 (float), 23 (uint), 25 (sint) in set 0
        self.assertEqual(m.bindings["%msv0_1"], 21)
        self.assertEqual(m.bindings["%msv1_1"], 23)
        self.assertEqual(m.bindings["%msv2_1"], 25)
        for cls in range(3):
            self.assertEqual(m.dsets[f"%msv{cls}_1"], 0)
        # OpTypeImage: 2D, Depth=0, Arrayed=1, MS=1, Sampled=1, Format=Unknown
        self.assertEqual(m.defs["%msimg0_1"], ("OpTypeImage", ["%float", "2D", "0", "1", "1", "1", "Unknown"]))
        self.assertEqual(m.defs["%msimg1_1"], ("OpTypeImage", ["%uint", "2D", "0", "1", "1", "1", "Unknown"]))
        self.assertEqual(m.defs["%msimg2_1"], ("OpTypeImage", ["%int", "2D", "0", "1", "1", "1", "Unknown"]))
        # 3 numeric classes x 4 consecutive samples (v7 + 0..3) and %v3int coord (v4, v5, v6)
        fetches = m.fetches()
        self.assertEqual(len(fetches), 12)
        by_class = {}
        for res_id, ty, var, coord_id, operands in fetches:
            self.assertEqual(operands[0], "Sample")
            cop, ca = m.defs[coord_id]
            self.assertEqual(cop, "OpCompositeConstruct")
            self.assertEqual(ca[0], "%v3int")
            self.assertEqual([m.vgpr_of(c) for c in ca[1:]], [4, 5, 6])
            by_class.setdefault((ty, var), []).append(m.sample_vgpr_offset(operands[1]))
            self.assertEqual([comp for _, _, comp in m.extracts_of(res_id)], [0])
        self.assertEqual(
            by_class,
            {
                ("%v4float", "%msv0_1"): [(7, 0), (7, 1), (7, 2), (7, 3)],
                ("%v4uint_st", "%msv1_1"): [(7, 0), (7, 1), (7, 2), (7, 3)],
                ("%v4int_st", "%msv2_1"): [(7, 0), (7, 1), (7, 2), (7, 3)],
            },
        )

    def test_msaa_load_vs_image_load_single_channel_4x_msaa_semantics(self):
        """Distinguish image_msaa_load from ordinary image_load on a 4x MSAA R32_FLOAT texel (frag_id=0):
        - image_msaa_load fetches samples 0, 1, 2, 3 and extracts channel 0 (red) from each, recovering all
          four distinct red sample values.
        - image_load on 2D_MSAA fetches 1 sample at frag_id and extracts RGBA channels 0, 1, 2, 3."""
        frag0_prologue = ["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0"]

        # 1) image_msaa_load with dmask:0xf -> 4 fetches per numeric class (samples v6+0..3), channel 0 each
        m_msaa = Module(translate(
            ["image_msaa_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm"],
            prologue=frag0_prologue,
        ))
        float_fetches = [f for f in m_msaa.fetches() if f[1] == "%v4float" and f[2] == "%msv0_0"]
        self.assertEqual(len(float_fetches), 4)
        self.assertEqual(
            [m_msaa.sample_vgpr_offset(f[4][1]) for f in float_fetches],
            [(6, 0), (6, 1), (6, 2), (6, 3)],
        )
        extracted_reds = []
        for res_id, _, _, _, _ in float_fetches:
            exts = m_msaa.extracts_of(res_id)
            self.assertEqual(len(exts), 1)
            self.assertEqual(exts[0][1:], ("%float", 0))
            extracted_reds.append(exts[0][0])
        # The 4 extracted red channels are assembled into a %v4float vector in sample order 0..3
        constructs = [
            res_id for res_id, op, a in m_msaa.ops
            if op == "OpCompositeConstruct" and a[0] == "%v4float" and a[1:] == extracted_reds
        ]
        self.assertEqual(len(constructs), 1)

        # 2) image_msaa_load with sparse dmask:0x5 -> 2 fetches per numeric class (samples v6+0, v6+2), channel 0
        m_sparse = Module(translate(
            ["image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D_MSAA unorm"],
            prologue=frag0_prologue,
        ))
        sparse_float = [f for f in m_sparse.fetches() if f[1] == "%v4float" and f[2] == "%msv0_0"]
        self.assertEqual(len(sparse_float), 2)
        self.assertEqual(
            [m_sparse.sample_vgpr_offset(f[4][1]) for f in sparse_float],
            [(6, 0), (6, 2)],
        )
        for res_id, _, _, _, _ in sparse_float:
            self.assertEqual([comp for _, _, comp in m_sparse.extracts_of(res_id)], [0])

        # 3) ordinary image_load on 2D_MSAA with dmask:0xf -> 1 fetch per numeric class at sample v6, RGBA 0..3
        m_load = Module(translate(
            ["image_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm"],
            prologue=frag0_prologue,
        ))
        load_fetches = m_load.fetches()
        self.assertEqual(len(load_fetches), 3)
        load_float = [f for f in load_fetches if f[1] == "%v4float" and f[2] == "%msv0_0"]
        self.assertEqual(len(load_float), 1)
        self.assertEqual(m_load.sample_vgpr_offset(load_float[0][4][1]), (6, 0))
        bcs = m_load.bitcasts_of(load_float[0][0])
        self.assertEqual(len(bcs), 1)
        self.assertEqual([comp for _, _, comp in m_load.extracts_of(bcs[0][0])], [0, 1, 2, 3])

    def test_dmask_and_d16_and_a16_and_nsa(self):
        cases = [
            ("image_msaa_load v0, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm", 3),
            ("image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D_MSAA", 6),
            ("image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm d16", 12),
            ("image_msaa_load v0, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm d16", 3),
            ("image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm a16", 12),
            ("image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm a16", 12),
            ("image_msaa_load v[0:3], [v4, v5, v6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm", 12),
            ("image_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm", 3),
            ("image_load v[0:3], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm", 3),
        ]
        for line, expected_fetches in cases:
            with self.subTest(line=line):
                m = Module(translate([line]))
                fetches = m.fetches()
                self.assertEqual(len(fetches), expected_fetches)
                for _, _, _, _, operands in fetches:
                    self.assertEqual(operands[0], "Sample")

    def test_binary_decode_and_translate(self):
        # Encodings verified with llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding:
        #   image_msaa_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm
        b_2d = bytes([0x31, 0x1F, 0x00, 0xF0, 0x04, 0x00, 0x02, 0x00])
        #   image_msaa_load v[0:3], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm
        b_arr = bytes([0x39, 0x1F, 0x00, 0xF0, 0x04, 0x00, 0x02, 0x00])
        #   image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm d16
        b_d16 = bytes([0x31, 0x1F, 0x00, 0xF0, 0x04, 0x00, 0x02, 0x80])
        #   image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm a16
        b_a16 = bytes([0x31, 0x1F, 0x00, 0xF0, 0x04, 0x00, 0x02, 0x40])
        #   image_msaa_load v[0:3], [v4, v5, v6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm
        b_nsa = bytes([0x33, 0x1F, 0x00, 0xF0, 0x04, 0x00, 0x02, 0x00, 0x05, 0x06, 0x00, 0x00])

        expected = [
            (b_2d, "dim:SQ_RSRC_IMG_2D_MSAA", 4, {"k": "vgpr", "r": 4, "n": 3}),
            (b_arr, "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY", 4, {"k": "vgpr", "r": 4, "n": 4}),
            (b_d16, "dim:SQ_RSRC_IMG_2D_MSAA", 2, {"k": "vgpr", "r": 4, "n": 3}),
            (b_a16, "dim:SQ_RSRC_IMG_2D_MSAA", 4, {"k": "vgpr", "r": 4, "n": 2}),
            (b_nsa, "dim:SQ_RSRC_IMG_2D_MSAA", 4, {"k": "vlist", "r": [4, 5, 6]}),
        ]
        for raw, dim_mod, n_data, addr_op in expected:
            with self.subTest(dim_mod=dim_mod, addr=addr_op):
                dec = rdna2_decode.decode_bytes(raw)
                insns = dec["insns"]
                self.assertEqual(len(insns), 1)
                ins = insns[0]
                self.assertEqual(ins["op"], "image_msaa_load")
                self.assertIn(dim_mod, ins["mods"])
                self.assertEqual(ins["args"][0]["n"], n_data)
                self.assertEqual({k: ins["args"][1][k] for k in addr_op}, addr_op)
                line = rdna2_decode.format_insn(ins)
                m = Module(translate([line]))
                self.assertEqual(len(m.fetches()), 12)

    def test_unsupported_forms_are_refused(self):
        bad = [
            "image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "image_msaa_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm glc",
            "image_msaa_load v[0:4], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm tfe",
            "image_msaa_load v[0:4], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm lwe",
            "image_msaa_load off, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
        ]
        for line in bad:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate([line])


class ValidationTests(unittest.TestCase):
    def test_spirv_val_accepts_all_msaa_load_variants(self):
        spirv_as, spirv_val = find_spirv_tools()
        if not (spirv_as and spirv_val):
            self.skipTest("needs spirv-as and spirv-val")
        variants = {
            "2d_msaa_f": "image_msaa_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "2d_msaa_1": "image_msaa_load v0, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "2d_msaa_5": "image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D_MSAA",
            "2d_msaa_d16": "image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm d16",
            "2d_msaa_d16_1": "image_msaa_load v0, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm d16",
            "2d_msaa_a16": "image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm a16",
            "2d_msaa_nsa": "image_msaa_load v[0:3], [v4, v5, v6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "2d_msaa_arr": "image_msaa_load v[0:3], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm",
            "2d_msaa_arr_d16": "image_msaa_load v[0:1], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm d16",
            "2d_msaa_arr_a16": "image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm a16",
            "image_load_msaa": "image_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "image_load_msaa_arr": "image_load v[0:3], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, line in variants.items():
                with self.subTest(variant=name):
                    sa = os.path.join(tmp, f"{name}.spvasm")
                    sv = os.path.join(tmp, f"{name}.spv")
                    with open(sa, "w", encoding="utf-8", newline="\n") as f:
                        f.write(translate([line]))
                    r_as = subprocess.run([spirv_as, "--target-env", "vulkan1.2", sa, "-o", sv],
                                          capture_output=True, text=True)
                    self.assertEqual(r_as.returncode, 0, r_as.stderr + r_as.stdout)
                    r_val = subprocess.run([spirv_val, "--target-env", "vulkan1.2", sv],
                                           capture_output=True, text=True)
                    self.assertEqual(r_val.returncode, 0, r_val.stderr + r_val.stdout)


if __name__ == "__main__":
    unittest.main()
