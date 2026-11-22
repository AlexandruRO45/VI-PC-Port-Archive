"""Semantic interpreter tests for RDNA2 v_cvt_pknorm_i16_f16 and v_cvt_pknorm_u16_f16.

D[15:0] = norm16(S0.f16), D[31:16] = norm16(S1.f16): each f16 is widened to f32,
clamped to [-1, 1] (i16) or [0, 1] (u16), scaled by 32767 or 65535, and rounded
to nearest even; NaN gives 0. The whole dword is written. Each case translates a
synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py, and compares against a plain-Python
reference. D leaves through exp as two 16-bit halves ORed into 0x40000000 so no
export channel is a NaN or denormal pattern.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f16(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def h16(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_round(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def norm16_f16(h: int, snorm: bool) -> int:
    x = f16(h)
    if x != x:
        return 0
    lo = -1.0 if snorm else 0.0
    x = min(max(x, lo), 1.0)
    # The product is rounded to f32 first, as the shader's OpFMul is.
    p = f32_round(x * (32767.0 if snorm else 65535.0))
    return round(p) & 0xFFFF


OPS = {"v_cvt_pknorm_i16_f16": True, "v_cvt_pknorm_u16_f16": False}
VALUES = [
    h16(0.0), h16(-0.0), h16(1.0), h16(-1.0), h16(0.5), h16(-0.5),
    h16(0.25), h16(-0.25), h16(0.75), h16(-0.75), h16(2.0), h16(-2.0),
    h16(65504.0), h16(-65504.0),
    0x7C00, 0xFC00,              # +inf, -inf
    0x7E00, 0xFE00,              # +qNaN, -qNaN
    0x7C01, 0xFC01,              # +sNaN, -sNaN
    0x0001, 0x8001,              # smallest +/- f16 subnormals
    0x03FF, 0x83FF,              # largest +/- f16 subnormals
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_pknorm_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_pknorm_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
    ]
    asm = ("; test_cvt_pknorm_f16\n"
           + "".join(f"\t{ln}\n" for ln in lines)
           + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def run_op(op: str, x_word: int, y_word: int, s0_mod: str = "", s1_mod: str = "", tail: str = "") -> int:
    suffix = f" {tail}" if tail else ""
    return run_lines([
        f"v_mov_b32 v1, {x_word & M32:#x}",
        f"v_mov_b32 v2, {y_word & M32:#x}",
        "v_mov_b32 v0, 0xdeadbeef",
        f"{op} v0, {s0_mod}v1, {s1_mod}v2{suffix}",
    ])


class CvtPknormF16Tests(unittest.TestCase):
    def test_reference_ties_and_extremes(self):
        self.assertEqual(norm16_f16(h16(1.0), True), 0x7FFF)
        self.assertEqual(norm16_f16(h16(-2.0), True), 0x8001)     # -32767, not -32768
        self.assertEqual(norm16_f16(h16(1.0), False), 0xFFFF)
        self.assertEqual(norm16_f16(h16(-1.0), False), 0)
        self.assertEqual(norm16_f16(h16(0.5), False), 32768)       # 32767.5 rounds to even
        self.assertEqual(norm16_f16(h16(0.5), True), 16384)        # 16383.5 rounds to even
        self.assertEqual(norm16_f16(h16(-0.5), True), (-16384) & 0xFFFF)
        self.assertEqual(norm16_f16(0x7E00, True), 0)
        self.assertEqual(norm16_f16(0xFE00, False), 0)

    def test_values_matrix(self):
        for op, snorm in OPS.items():
            for x in VALUES:
                for y in VALUES[::4]:
                    with self.subTest(op=op, x=hex(x), y=hex(y)):
                        self.assertEqual(run_op(op, x, y), norm16_f16(x, snorm) | (norm16_f16(y, snorm) << 16))

    def test_opsel_source_halves(self):
        v1 = (h16(1.0) << 16) | h16(0.5)
        v2 = (h16(-0.5) << 16) | h16(0.25)
        cases = [
            ("op_sel:[0,0,0]", 0.5, 0.25),
            ("op_sel:[1,0,0]", 1.0, 0.25),
            ("op_sel:[0,1,0]", 0.5, -0.5),
            ("op_sel:[1,1,0]", 1.0, -0.5),
        ]
        for op, snorm in OPS.items():
            for opsel, a, b in cases:
                with self.subTest(op=op, opsel=opsel):
                    expected = norm16_f16(h16(a), snorm) | (norm16_f16(h16(b), snorm) << 16)
                    self.assertEqual(run_op(op, v1, v2, tail=opsel), expected)

    def test_source_abs_and_neg_modifiers(self):
        half = h16(0.5)
        neg_half = h16(-0.5)
        # i16: -0.5 -> -16384, +0.5 -> +16384
        self.assertEqual(
            run_op("v_cvt_pknorm_i16_f16", half, half, s0_mod="-"),
            ((-16384) & 0xFFFF) | (16384 << 16),
        )
        # u16: |-0.5| -> 32768, -0.5 -> 0
        self.assertEqual(
            run_lines([
                f"v_mov_b32 v1, {neg_half:#x}",
                f"v_mov_b32 v2, {half:#x}",
                "v_cvt_pknorm_u16_f16 v0, |v1|, -v2",
            ]),
            32768,
        )

    def test_seeded_sweep(self):
        rng = random.Random(0xF16C0DE)
        for _ in range(40):
            op = rng.choice(list(OPS))
            snorm = OPS[op]
            x = h16(rng.uniform(-1.5, 1.5))
            y = h16(rng.uniform(-1.5, 1.5))
            with self.subTest(op=op, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y), norm16_f16(x, snorm) | (norm16_f16(y, snorm) << 16))

    def test_unsupported_modifiers_and_forms(self):
        bad_lines = [
            "v_cvt_pknorm_i16_f16 v0, v1, v2 clamp",
            "v_cvt_pknorm_u16_f16 v0, v1, v2 clamp",
            "v_cvt_pknorm_i16_f16 v0, v1, v2 op_sel:[0,0,1]",
            "v_cvt_pknorm_u16_f16 v0, v1, v2 op_sel:[0,0,1]",
            "v_cvt_pknorm_i16_f16_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_cvt_pknorm_u16_f16_sdwa v0, v1, v2 dst_sel:DWORD",
            "v_cvt_pknorm_i16_f16 v0, v1, v2 mul:2",
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    run_lines([line])


if __name__ == "__main__":
    unittest.main()
