"""tools/gbuffer_export_census.py on hand-written listings and containers (no game files, no GPU)."""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gbuffer_export_census.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gbuffer_export_census as gec  # noqa: E402


def osg1_dxbc(masks, depth=False):
    """A DXBC container holding only an OSG1 part: one SV_Target per mask (register = index)."""
    elems = [(0, 0, i, 64, 3, i, mask) for i, mask in enumerate(masks)]
    if depth:
        elems.append((0, 0, 0, 65, 3, 0xFFFFFFFF, 1))
    body = struct.pack("<II", len(elems), 8)
    for stream, name, idx, sysv, ctype, reg, mask in elems:
        body += struct.pack("<6IBBHI", stream, name, idx, sysv, ctype, reg, mask, mask, 0, 0)
    part = b"OSG1" + struct.pack("<I", len(body)) + body
    head = b"DXBC" + bytes(16) + struct.pack("<III", 1, 36 + len(part), 1) + struct.pack("<I", 36)
    return head + part


def write_case(root, name, masks, lines):
    disasm = os.path.join(root, "disasm")
    os.makedirs(disasm, exist_ok=True)
    with open(os.path.join(root, name + ".dxbc"), "wb") as f:
        f.write(osg1_dxbc(masks))
    with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"\t{line:<44}; encoding: [0x00,0x00,0x00,0x00]\n" if not line.endswith(":") else line + "\n"
                        for line in lines))
    return disasm


def alpha_of(lines):
    ins = gec.parse(lines)
    ei = next(i for i, (op, ops) in enumerate(ins) if op == "exp" and ops[0] == "mrt0")
    return gec.mrt0_alpha(ins, ei)


EXPORT = ["v_cvt_pkrtz_f16_f32_e32 v1, v6, v5", "exp mrt0, v0, v0, v1, v1 compr"]


class Mrt0AlphaTest(unittest.TestCase):
    def test_byte_times_inverse_255(self):
        lines = ["v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, 0x3b808081, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("byte 0 x 1/255", False))

    def test_other_byte_and_legacy_multiply(self):
        lines = ["v_cvt_f32_ubyte2_e32 v4, v3", "v_mul_legacy_f32_e32 v5, v4, 0x3b808081"] + EXPORT
        self.assertEqual(alpha_of(lines), ("byte 2 x 1/255", False))

    def test_u32_and_a_scale_moved_into_a_register(self):
        lines = ["s_mov_b32 s9, 0x3b808081", "v_cvt_f32_u32_e32 v4, v3", "v_mul_f32_e32 v5, s9, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("u32 x 1/255", False))

    def test_lookalike_constant_is_a_float_product(self):
        # One ulp away from 1/255: not the scale, so not counted as a byte.
        lines = ["v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, 0x3b808082, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("float product", False))

    def test_a_later_overwrite_of_the_scale_register_wins(self):
        lines = ["s_mov_b32 s9, 0x3b808081", "s_mov_b32 s9, 0x3f800000", "v_cvt_f32_ubyte0_e32 v4, v3",
                 "v_mul_f32_e32 v5, s9, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("float product", False))

    def test_other_producer_is_named_and_a_label_marks_crossing(self):
        lines = ["v_cndmask_b32_e32 v5, v2, v3, vcc_lo", "BB0_1:"] + EXPORT
        self.assertEqual(alpha_of(lines), ("v_cndmask_b32", True))

    def test_alpha_slot_off_and_uncompressed(self):
        self.assertEqual(alpha_of(["exp mrt0, v0, v0, off, off compr"]), ("alpha off", False))
        self.assertEqual(alpha_of(["exp mrt0, v0, v1, v2, v3"]), ("not compr", False))

    def test_range_write_counts_as_writer(self):
        lines = ["s_buffer_load_dwordx2 s[8:9], s[0:3], 0x10", "v_mul_f32_e32 v5, s9, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("float product", False))


class CensusTest(unittest.TestCase):
    def test_counts_by_target_count_and_mrt4_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = ["v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, 0x3b808081, v4"] + EXPORT
            write_case(tmp, "PS_DeferredTextured_a", [15, 15, 15, 15, 3],
                       body + ["exp mrt4, v8, v8, off, off done compr vm"])
            write_case(tmp, ".PS_DeferredTextured_b", [15] * 6,
                       body + ["exp mrt4, v8, v8, v9, v9 compr"])
            disasm = write_case(tmp, "PS_Other_c", [15], body)
            shaders, alpha, mrt4 = gec.census(disasm, "PS_DeferredTextured")
        # The dot-named listing is counted (os.listdir, not glob); the non-matching one is not.
        self.assertEqual(dict(shaders), {5: 1, 6: 1})
        self.assertEqual(dict(alpha), {(5, "byte 0 x 1/255", "straight"): 1, (6, "byte 0 x 1/255", "straight"): 1})
        self.assertEqual(dict(mrt4), {(5, 3, "vv-- compr"): 1, (6, 15, "vvvv compr"): 1})

    def test_cli_prints_counts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = write_case(tmp, "PS_DeferredTextured_secretname", [15, 15, 15, 15, 3],
                                EXPORT + ["exp mrt4, v8, v8, off, off compr"])
            p = subprocess.run([sys.executable, "-I", TOOL, disasm], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("secretname", p.stdout)
        self.assertIn("listings 1: 5 targets:1", p.stdout)
        self.assertIn("mrt4, 5 targets, OSG1 mask 0x3: vv-- compr 1", p.stdout)

    def test_missing_directory_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing")
            p = subprocess.run([sys.executable, "-I", TOOL, missing], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn(missing, p.stderr)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
