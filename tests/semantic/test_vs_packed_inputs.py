"""Semantic translator test for vertex shader packed input register deduplication (SHADER-644).

When an ISG1 input signature packs multiple semantic elements into the same register (e.g. .xy and .zw),
rdna2spirv deduplicates vertex inputs per register so that each %vin{reg} OpVariable, decoration, and
entry-point interface reference is declared exactly once. Conflicting component types in the same register
raise Unsupported.

Standard library only; no GPU, no game data.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402


def make_isg1_dxbc(entries):
    """Build a synthetic DXBC container containing an ISG1 chunk with given (reg, mask, sysv, ctype) entries."""
    entry_offset = 8
    isg1_payload = struct.pack("<II", len(entries), entry_offset)
    for reg, mask, sysv, ctype in entries:
        isg1_payload += struct.pack("<6I", 0, 0, 0, sysv, ctype, reg) + bytes([mask, mask, 0, 0, 0, 0, 0, 0])
    chunk = b"ISG1" + struct.pack("<I", len(isg1_payload)) + isg1_payload
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 36 + len(chunk), 1) + struct.pack("<I", 36)
    return header + chunk


class VsPackedInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.disasm_dir = os.path.join(self.tmp.name, "disasm")
        os.mkdir(self.disasm_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def translate_vs(self, name, entries, lines):
        dxbc_path = os.path.join(self.tmp.name, f"{name}.dxbc")
        with open(dxbc_path, "wb") as f:
            f.write(make_isg1_dxbc(entries))
        s_path = os.path.join(self.disasm_dir, f"{name}.s")
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines))
        return rdna2spirv.translate(s_path)

    def test_packed_inputs_deduplication(self):
        # Register 0 has two entries: .xy (mask 3) and .zw (mask 12).
        # Register 1 has one entry: .xyz (mask 7).
        entries = [(0, 3, 0, 3), (0, 12, 0, 3), (1, 7, 0, 3)]
        lines = [
            "s_swappc_b64 s[2:3], s[4:5]",
            "v_mov_b32 v10, v4",       # reads r0.x
            "v_mov_b32 v11, v6",       # reads r0.z
            "v_mov_b32 v12, v8",       # reads r1.x
            "exp pos0 v10, v11, v12, v10 done vm",
            "s_endpgm",
        ]
        spv = self.translate_vs("vs_packed", entries, lines)

        # Check OpEntryPoint interface includes %vin0 and %vin1 exactly once
        ep_lines = [ln for ln in spv.splitlines() if "OpEntryPoint Vertex" in ln]
        self.assertEqual(len(ep_lines), 1)
        tokens = ep_lines[0].split()
        self.assertEqual(tokens.count("%vin0"), 1)
        self.assertEqual(tokens.count("%vin1"), 1)

        # Check decorations: exactly one OpDecorate per location
        dec_vin0 = [ln for ln in spv.splitlines() if "OpDecorate %vin0 Location 0" in ln]
        dec_vin1 = [ln for ln in spv.splitlines() if "OpDecorate %vin1 Location 1" in ln]
        self.assertEqual(len(dec_vin0), 1)
        self.assertEqual(len(dec_vin1), 1)

        # Check variable declarations: exactly one OpVariable per location
        var_vin0 = [ln for ln in spv.splitlines() if "%vin0 = OpVariable" in ln]
        var_vin1 = [ln for ln in spv.splitlines() if "%vin1 = OpVariable" in ln]
        self.assertEqual(len(var_vin0), 1)
        self.assertEqual(len(var_vin1), 1)

        # If spirv-as and spirv-val are present, verify module validity
        bin_dir = os.path.abspath(os.path.join(ROOT, ".venv", "VulkanSDK", "Bin"))
        spirv_as = os.path.join(bin_dir, "spirv-as.exe") if os.name == "nt" else shutil.which("spirv-as")
        spirv_val = os.path.join(bin_dir, "spirv-val.exe") if os.name == "nt" else shutil.which("spirv-val")
        if spirv_as and os.path.exists(spirv_as) and spirv_val and os.path.exists(spirv_val):
            spvasm_path = os.path.join(self.tmp.name, "vs_packed.spvasm")
            spv_bin_path = os.path.join(self.tmp.name, "vs_packed.spv")
            with open(spvasm_path, "w", encoding="utf-8") as f:
                f.write(spv)
            r_as = subprocess.run([spirv_as, spvasm_path, "-o", spv_bin_path], capture_output=True, text=True)
            self.assertEqual(r_as.returncode, 0, r_as.stderr)
            r_val = subprocess.run([spirv_val, spv_bin_path], capture_output=True, text=True)
            self.assertEqual(r_val.returncode, 0, r_val.stderr)

    def test_conflicting_component_types_rejected(self):
        # Register 0 has conflicting types: uint (1) for .xy and float (3) for .zw
        entries = [(0, 3, 0, 1), (0, 12, 0, 3)]
        lines = [
            "s_swappc_b64 s[2:3], s[4:5]",
            "v_mov_b32 v10, v4",
            "exp pos0 v10, v10, v10, v10 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            self.translate_vs("vs_conflict", entries, lines)
        self.assertIn("conflicting vertex input component types", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
