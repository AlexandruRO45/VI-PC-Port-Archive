"""Regression tests for SHADER-1265: SHADER-309 LDS bounds check on ds_add_f32, ds_cmpst_f32, and addtid_b32."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from support import translate_lines  # noqa: E402


def make_fssm(lds_units=1, threads=32):
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x88 | (lds_units << 15), threads, 1, 1)
    return bytes(fssm)


def validate_spirv(spv_text):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        spv = os.path.join(tmp, "out.spv")
        r = subprocess.run(["spirv-as", "--target-env", "vulkan1.3", "-o", spv, "-"],
                           input=spv_text, text=True, capture_output=True)
        if r.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r.stderr}")
        v = subprocess.run(["spirv-val", "--target-env", "vulkan1.3", spv],
                           text=True, capture_output=True)
        if v.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{v.stderr}")


class LdsBoundsAddtidAndFloatTests(unittest.TestCase):
    # 1 LDS_SIZE unit = 512 bytes (128 dwords); an access at offset:512 triggers SHADER-309 bounds checking.
    FSSM = make_fssm(lds_units=1)

    def assert_bounded_and_valid(self, name, lines):
        spv = translate_lines(name, lines, fssm=self.FSSM)
        self.assertIn("OpULessThan %bool", spv)
        validate_spirv(spv)
        return spv

    def test_ds_read_and_write_addtid_b32_are_bounds_checked(self):
        self.assert_bounded_and_valid("cs_ds_addtid_bounds", [
            "s_mov_b32 m0, 0",
            "v_mov_b32 v0, 42",
            "ds_write_addtid_b32 v0 offset:512",
            "ds_read_addtid_b32 v1 offset:512",
            "s_endpgm",
        ])

    def test_ds_add_f32_and_rtn_are_bounds_checked(self):
        self.assert_bounded_and_valid("cs_ds_add_f32_bounds", [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 1.0",
            "ds_add_f32 v0, v1 offset:512",
            "ds_add_rtn_f32 v2, v0, v1 offset:512",
            "s_endpgm",
        ])

    def test_ds_cmpst_f32_and_rtn_are_bounds_checked(self):
        self.assert_bounded_and_valid("cs_ds_cmpst_f32_bounds", [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 1.0",
            "v_mov_b32 v2, 2.0",
            "ds_cmpst_f32 v0, v1, v2 offset:512",
            "ds_cmpst_rtn_f32 v3, v0, v1, v2 offset:512",
            "s_endpgm",
        ])


if __name__ == "__main__":
    unittest.main()
