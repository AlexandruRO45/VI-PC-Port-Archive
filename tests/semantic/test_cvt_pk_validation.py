"""Validation and positive-control semantic tests for the 7 packed-conversion VALU ops.

Covers `v_cvt_pk_u16_u32`, `v_cvt_pk_i16_i32`, `v_cvt_pknorm_i16_f32`, `v_cvt_pknorm_u16_f32`,
`v_cvt_pknorm_i16_f16`, `v_cvt_pknorm_u16_f16`, and `v_cvt_pkrtz_f16_f32`:
- Rejects bad arity, non-single-VGPR or modified destinations, multi-dword sources, invalid encodings
  (`_e32`/`_dpp`/`_sdwa` on VOP3-only ops; `_dpp`/`_sdwa` and `_e32` modifiers on `v_cvt_pkrtz_f16_f32`),
  integer source `neg`/`abs`/`sext`, float source `sext`, and unsupported/duplicate/malformed modifiers.
- Verifies valid positive forms (including `_e64`, plain negative integer immediates, VOP3 float `neg`/`abs`,
  valid `clamp` / `op_sel`, and `v_cvt_pkrtz_f16_f32_e32`) through `spirv_mini_interp` and
  `spirv-as` + `spirv-val --target-env vulkan1.2`.
Standard library only; synthetic inputs only.
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
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

INT_OPS = ("v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32")
PKNORM_F32_OPS = ("v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32")
PKNORM_F16_OPS = ("v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16")
PKRTZ_OPS = ("v_cvt_pkrtz_f16_f32",)
VOP3_ONLY_OPS = INT_OPS + PKNORM_F32_OPS + PKNORM_F16_OPS
FLOAT_OPS = PKNORM_F32_OPS + PKNORM_F16_OPS + PKRTZ_OPS
ALL_OPS = VOP3_ONLY_OPS + PKRTZ_OPS

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_pk_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_pk_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def _translate(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_cvt_pk_val\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def _run_v0(body_lines) -> tuple[int, str]:
    lines = list(body_lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
        "exp mrt0 v5, v6, v5, v5 done vm",
        "s_endpgm",
    ]
    spv = _translate(lines)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16), spv


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _f16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


class CvtPkValidationTests(unittest.TestCase):
    def _assert_rejected(self, line: str):
        with self.subTest(line=line):
            with self.assertRaises(rdna2spirv.Unsupported):
                _translate([
                    "s_mov_b32 s2, 0x3f000000",
                    "s_mov_b32 s3, 0xbf000000",
                    "v_mov_b32 v1, 0x3f000000",
                    "v_mov_b32 v2, 0xbf000000",
                    "v_mov_b32 v3, 0x3f800000",
                    line,
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ])

    def test_reject_bad_arity_dst_and_source_width(self):
        for op in ALL_OPS:
            for bad in (
                f"{op} v0, v1",
                f"{op} v0, v1, v2, v3",
                f"{op} s0, v1, v2",
                f"{op} v[0:1], v1, v2",
                f"{op} null, v1, v2",
                f"{op} vcc_lo, v1, v2",
                f"{op} exec_lo, v1, v2",
                f"{op} -v0, v1, v2",
                f"{op} |v0|, v1, v2",
                f"{op} sext(v0), v1, v2",
                f"{op} v0, v[1:2], v2",
                f"{op} v0, v1, v[2:3]",
                f"{op} v0, s[2:3], v2",
                f"{op} v0, v1, s[2:3]",
            ):
                self._assert_rejected(bad)

    def test_reject_bad_encodings(self):
        for op in VOP3_ONLY_OPS:
            for bad in (
                f"{op}_e32 v0, v1, v2",
                f"{op}_dpp v0, v1, v2 row_mirror",
                f"{op}_sdwa v0, v1, v2 dst_sel:DWORD src0_sel:DWORD src1_sel:DWORD",
            ):
                self._assert_rejected(bad)
        for bad in (
            "v_cvt_pkrtz_f16_f32_dpp v0, v1, v2 row_mirror",
            "v_cvt_pkrtz_f16_f32_sdwa v0, v1, v2 dst_sel:DWORD src0_sel:DWORD src1_sel:DWORD",
            "v_cvt_pkrtz_f16_f32_e32 v0, -v1, v2",
            "v_cvt_pkrtz_f16_f32_e32 v0, v1, |v2|",
            "v_cvt_pkrtz_f16_f32_e32 v0, sext(v1), v2",
            "v_cvt_pkrtz_f16_f32_e32 v0, v1, v2 clamp",
            "v_cvt_pkrtz_f16_f32_e32 v0, v1, v2 mul:2",
        ):
            self._assert_rejected(bad)

    def test_reject_bad_source_modifiers(self):
        for op in INT_OPS:
            for bad in (
                f"{op} v0, -v1, v2",
                f"{op} v0, v1, -v2",
                f"{op} v0, |v1|, v2",
                f"{op} v0, v1, |v2|",
                f"{op} v0, -|v1|, v2",
                f"{op} v0, neg(1), v2",
                f"{op} v0, v1, |1|",
                f"{op} v0, sext(v1), v2",
                f"{op} v0, v1, sext(v2)",
            ):
                self._assert_rejected(bad)
        for op in FLOAT_OPS:
            for bad in (
                f"{op} v0, sext(v1), v2",
                f"{op} v0, v1, sext(v2)",
            ):
                self._assert_rejected(bad)

    def test_reject_bad_and_duplicate_modifiers(self):
        common_bad = (
            "mul:2",
            "mul:4",
            "div:2",
            "op_sel_hi:[1,1]",
            "neg_lo:[1,0]",
            "neg_hi:[0,1]",
            "unknown_mod",
        )
        for op in ALL_OPS:
            for mod in common_bad:
                self._assert_rejected(f"{op} v0, v1, v2 {mod}")
        for op in INT_OPS + PKRTZ_OPS:
            self._assert_rejected(f"{op} v0, v1, v2 clamp")
        for op in PKNORM_F32_OPS + PKNORM_F16_OPS:
            self._assert_rejected(f"{op} v0, v1, v2 clamp clamp")
        for op in INT_OPS + PKNORM_F32_OPS + PKRTZ_OPS:
            self._assert_rejected(f"{op} v0, v1, v2 op_sel:[0,0,0]")
        for op in PKNORM_F16_OPS:
            for bad_sel in (
                "op_sel:[0,0,0] op_sel:[1,0,0]",
                "op_sel:[]",
                "op_sel:[2,0,0]",
                "op_sel:[0,0,0,1]",
                "op_sel:[+0,0,0]",
                "op_sel:0",
                "op_sel:[0,0,1] clamp clamp",
            ):
                self._assert_rejected(f"{op} v0, v1, v2 {bad_sel}")

    def test_positive_controls_interp_and_spirv_val(self):
        half_f32 = _f32_bits(0.5)
        neg_half_f32 = _f32_bits(-0.5)
        one_f16 = _f16_bits(1.0)
        half_f16 = _f16_bits(0.5)
        neg_half_f16 = _f16_bits(-0.5)
        qtr_f16 = _f16_bits(0.25)

        cases = [
            (
                "v_cvt_pk_u16_u32_e64_neg_imm",
                ["v_mov_b32 v1, 7", "v_cvt_pk_u16_u32_e64 v0, v1, -1"],
                7 | (0xFFFF << 16),
            ),
            (
                "v_cvt_pk_i16_i32_e64_neg_imm",
                ["v_mov_b32 v1, 0x12345", "v_cvt_pk_i16_i32_e64 v0, v1, -16"],
                0x7FFF | ((-16 & 0xFFFF) << 16),
            ),
            (
                "v_cvt_pknorm_i16_f32_e64_mods_clamp",
                [
                    f"v_mov_b32 v1, {half_f32:#x}",
                    f"v_mov_b32 v2, {neg_half_f32:#x}",
                    "v_cvt_pknorm_i16_f32_e64 v0, -v1, |v2| clamp",
                ],
                ((-16384) & 0xFFFF) | (16384 << 16),
            ),
            (
                "v_cvt_pknorm_u16_f32_e64_mods_clamp",
                [
                    f"v_mov_b32 v1, {neg_half_f32:#x}",
                    f"v_mov_b32 v2, {half_f32:#x}",
                    "v_cvt_pknorm_u16_f32_e64 v0, |v1|, -v2 clamp",
                ],
                32768 | (0 << 16),
            ),
            (
                "v_cvt_pknorm_i16_f16_e64_opsel_clamp",
                [
                    f"v_mov_b32 v1, {(one_f16 << 16) | qtr_f16:#x}",
                    f"v_mov_b32 v2, {(neg_half_f16 << 16) | half_f16:#x}",
                    "v_cvt_pknorm_i16_f16_e64 v0, -v1, |v2| op_sel:[1,1,0] clamp",
                ],
                ((-32767) & 0xFFFF) | (16384 << 16),
            ),
            (
                "v_cvt_pknorm_u16_f16_e64_opsel_dhi_clamp",
                [
                    f"v_mov_b32 v1, {(half_f16 << 16) | one_f16:#x}",
                    f"v_mov_b32 v2, {neg_half_f16:#x}",
                    "v_cvt_pknorm_u16_f16_e64 v0, v1, |v2| op_sel:[1,0,1] clamp",
                ],
                (32768 << 16),
            ),
            (
                "v_cvt_pkrtz_f16_f32_e32",
                [
                    f"v_mov_b32 v1, {half_f32:#x}",
                    "v_cvt_pkrtz_f16_f32_e32 v0, v1, -1.0",
                ],
                half_f16 | (_f16_bits(-1.0) << 16),
            ),
            (
                "v_cvt_pkrtz_f16_f32_e64_fmods",
                [
                    f"v_mov_b32 v1, {half_f32:#x}",
                    f"v_mov_b32 v2, {neg_half_f32:#x}",
                    "v_cvt_pkrtz_f16_f32_e64 v0, -v1, |v2|",
                ],
                neg_half_f16 | (half_f16 << 16),
            ),
        ]

        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        with tempfile.TemporaryDirectory() as td:
            for idx, (name, body, want) in enumerate(cases):
                with self.subTest(case=name):
                    got, spv = _run_v0(body)
                    self.assertEqual(got, want)
                    if spirv_as and spirv_val:
                        spvasm = os.path.join(td, f"{idx}_{name}.spvasm")
                        spvbin = os.path.join(td, f"{idx}_{name}.spv")
                        with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                            f.write(spv)
                        subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin], check=True)
                        subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin], check=True)
        if not spirv_as or not spirv_val:
            self.skipTest("spirv-as / spirv-val not on PATH")


if __name__ == "__main__":
    unittest.main()
