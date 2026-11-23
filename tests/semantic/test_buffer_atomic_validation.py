"""Synthetic validation tests for RDNA2 buffer_atomic* vdata register kind/width and lds/tfe modifiers.

Verifies that tools/rdna2spirv.py:
- Validates vdata register kind (must be VGPR) and width (2 VGPRs for cmpswap/fcmpswap and _x2,
  1 VGPR for other 32-bit ops and fmin/fmax) across buffer_atomic, buffer_atomic_float, and
  buffer_atomic_x2, raising rdna2spirv.Unsupported instead of KeyError or silent out-of-range VGPR
  reads.
- Rejects unsupported `lds` and `tfe` modifiers with rdna2spirv.Unsupported across all three
  buffer_atomic* handlers.
- Translates valid buffer_atomic* forms cleanly and passes spirv-val --target-env vulkan1.2.

Standard library only (spirv-as / spirv-val used when on PATH); no GPU, no LLVM, no game files.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)

SETUP_LINES = [
    "s_mov_b32 s8, 0",
    "s_mov_b32 s9, 0",
    "s_mov_b32 s10, 0x100",
    "s_mov_b32 s11, 0x31016fac",
    "v_mov_b32_e32 v1, 0x40",
    "v_mov_b32_e32 v2, 1",
    "v_mov_b32_e32 v3, 2",
    "v_mov_b32_e32 v4, 3",
    "v_mov_b32_e32 v5, 4",
]

SINGLE_DWORD_OPS = (
    "buffer_atomic_add",
    "buffer_atomic_sub",
    "buffer_atomic_umin",
    "buffer_atomic_umax",
    "buffer_atomic_smin",
    "buffer_atomic_smax",
    "buffer_atomic_and",
    "buffer_atomic_or",
    "buffer_atomic_xor",
    "buffer_atomic_swap",
    "buffer_atomic_inc",
    "buffer_atomic_dec",
    "buffer_atomic_csub",
)

CMPSWAP_OPS = (
    "buffer_atomic_cmpswap",
    "buffer_atomic_fcmpswap",
)

FLOAT_OPS = (
    "buffer_atomic_fmin",
    "buffer_atomic_fmax",
)

X2_OPS = (
    "buffer_atomic_and_x2",
    "buffer_atomic_or_x2",
    "buffer_atomic_xor_x2",
    "buffer_atomic_swap_x2",
)


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_baval.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_baval.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_baval.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; cs_baval\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def validate_spirv(spv_text):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_path],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")
    return True


class BufferAtomicValidationTest(unittest.TestCase):
    def test_non_vgpr_vdata_rejected(self):
        cases = (
            [(op, bad) for op in ("buffer_atomic_add", "buffer_atomic_sub", "buffer_atomic_swap",
                                  "buffer_atomic_inc", "buffer_atomic_dec", "buffer_atomic_csub")
             for bad in ("s0", "0", "1", "off")]
            + [(op, bad) for op in CMPSWAP_OPS for bad in ("s[0:1]", "s0", "0", "off")]
            + [(op, bad) for op in FLOAT_OPS for bad in ("s0", "0", "off")]
            + [(op, bad) for op in X2_OPS for bad in ("s[0:1]", "s0", "0", "off")]
        )
        for op, bad_vd in cases:
            line = f"{op} {bad_vd}, v1, s[8:11], 0 offen glc"
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(SETUP_LINES + [line])

    def test_wrong_vgpr_width_rejected(self):
        cases = (
            [(op, bad) for op in SINGLE_DWORD_OPS for bad in ("v[2:3]", "v[2:5]")]
            + [(op, bad) for op in CMPSWAP_OPS for bad in ("v2", "v[2:4]", "v[2:5]")]
            + [(op, bad) for op in FLOAT_OPS for bad in ("v[2:3]", "v[2:5]")]
            + [(op, bad) for op in X2_OPS for bad in ("v2", "v[2:4]", "v[2:5]")]
        )
        for op, bad_vd in cases:
            line = f"{op} {bad_vd}, v1, s[8:11], 0 offen glc"
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(SETUP_LINES + [line])

    def test_lds_and_tfe_modifiers_rejected(self):
        ops_with_valid_vdata = (
            [(op, "v2") for op in ("buffer_atomic_add", "buffer_atomic_sub", "buffer_atomic_swap",
                                   "buffer_atomic_inc", "buffer_atomic_dec", "buffer_atomic_csub")]
            + [(op, "v[2:3]") for op in CMPSWAP_OPS]
            + [(op, "v2") for op in FLOAT_OPS]
            + [(op, "v[2:3]") for op in X2_OPS]
        )
        for op, vd in ops_with_valid_vdata:
            for mod in ("lds", "tfe"):
                for extra in ("offen", "offen glc"):
                    line = f"{op} {vd}, v1, s[8:11], 0 {extra} {mod}"
                    with self.subTest(line=line):
                        with self.assertRaises(rdna2spirv.Unsupported):
                            translate(SETUP_LINES + [line])

    def test_valid_forms_translate_and_validate(self):
        valid_ops = (
            [(op, "v2") for op in ("buffer_atomic_add", "buffer_atomic_sub", "buffer_atomic_swap",
                                   "buffer_atomic_inc", "buffer_atomic_dec", "buffer_atomic_csub")]
            + [(op, "v[2:3]") for op in CMPSWAP_OPS]
            + [(op, "v2") for op in FLOAT_OPS]
            + [(op, "v[2:3]") for op in X2_OPS]
        )
        validated_any = False
        for op, vd in valid_ops:
            for suffix in ("offen", "offen glc"):
                line = f"{op} {vd}, v1, s[8:11], 0 {suffix}"
                with self.subTest(line=line):
                    spv = translate(SETUP_LINES + [line])
                    self.assertIn("OpEntryPoint GLCompute", spv)
                    if validate_spirv(spv):
                        validated_any = True
        if not validated_any:
            self.skipTest("spirv-as / spirv-val not on PATH")


if __name__ == "__main__":
    unittest.main()
