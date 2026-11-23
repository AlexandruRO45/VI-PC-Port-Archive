"""Tests for RDNA2 image_msaa_load (MIMG 0x80) and MSAA image_load in tools/rdna2spirv.py.

Verifies:
- 2D_MSAA (x, y, sample_index) -> bindings 20/22/24 (float/uint/sint), OpTypeImage ... 2D 0 0 1 1 Unknown
- 2D_MSAA_ARRAY (x, y, slice, sample_index) -> bindings 21/23/25 (float/uint/sint), OpTypeImage ... 2D 0 1 1 1 Unknown
- OpImageFetch emits Sample with the exact sample_index VGPR (3rd coord for 2D_MSAA, 4th coord for 2D_MSAA_ARRAY)
- dmask (0xf, 0x1, 0x5), unorm, d16, a16, NSA address lists, and binary decoding via rdna2_decode
- Unsupported forms (non-MSAA dim, glc, tfe/lwe, non-VGPR dest, short addr) raise Unsupported
- Every translated module passes spirv-as and spirv-val.
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


def translate(lines, stage="ps"):
    return support.translate_lines(f"{stage}_msaa_load", PROLOGUE + lines + EPILOGUE)


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
        """List of (result_type, var_name, coord_id, operands_list) for each OpImageFetch."""
        out = []
        for _, op, a in self.ops:
            if op == "OpImageFetch":
                out.append((a[0], self.variable(a[1]), a[2], a[3:]))
        return out

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
        # Three typed fetches with Sample operand from v6 and %v2int coord from (v4, v5)
        fetches = m.fetches()
        self.assertEqual(len(fetches), 3)
        self.assertEqual(
            {(ty, var) for ty, var, _, _ in fetches},
            {("%v4float", "%msv0_0"), ("%v4uint_st", "%msv1_0"), ("%v4int_st", "%msv2_0")},
        )
        for _, _, coord_id, operands in fetches:
            self.assertEqual(operands[0], "Sample")
            self.assertEqual(m.vgpr_of(operands[1]), 6)
            cop, ca = m.defs[coord_id]
            self.assertEqual(cop, "OpCompositeConstruct")
            self.assertEqual(ca[0], "%v2int")
            self.assertEqual([m.vgpr_of(c) for c in ca[1:]], [4, 5])

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
        # Three typed fetches with Sample operand from v7 and %v3int coord from (v4, v5, v6)
        fetches = m.fetches()
        self.assertEqual(len(fetches), 3)
        self.assertEqual(
            {(ty, var) for ty, var, _, _ in fetches},
            {("%v4float", "%msv0_1"), ("%v4uint_st", "%msv1_1"), ("%v4int_st", "%msv2_1")},
        )
        for _, _, coord_id, operands in fetches:
            self.assertEqual(operands[0], "Sample")
            self.assertEqual(m.vgpr_of(operands[1]), 7)
            cop, ca = m.defs[coord_id]
            self.assertEqual(cop, "OpCompositeConstruct")
            self.assertEqual(ca[0], "%v3int")
            self.assertEqual([m.vgpr_of(c) for c in ca[1:]], [4, 5, 6])

    def test_dmask_and_d16_and_a16_and_nsa(self):
        cases = [
            "image_msaa_load v0, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D_MSAA",
            "image_msaa_load v[0:1], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm d16",
            "image_msaa_load v0, v[4:6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm d16",
            "image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm a16",
            "image_msaa_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm a16",
            "image_msaa_load v[0:3], [v4, v5, v6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "image_load v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm",
            "image_load v[0:3], v[4:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm",
        ]
        for line in cases:
            with self.subTest(line=line):
                m = Module(translate([line]))
                fetches = m.fetches()
                self.assertEqual(len(fetches), 3)
                for _, _, _, operands in fetches:
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
                self.assertEqual(len(m.fetches()), 3)

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
