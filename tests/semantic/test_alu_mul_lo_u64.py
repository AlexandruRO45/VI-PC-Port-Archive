"""Semantic interpreter tests for RDNA2 v_mul_lo_u64.

Tests the 64-bit integer product VALU instruction:
  v_mul_lo_u64: D = (S0 * S1) & 0xFFFFFFFFFFFFFFFF (64-bit product low 64 bits)

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported 64-bit pair with a plain-Python reference.
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

M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def u64(x: int) -> int:
    return x & M64


BOUNDARY_VALUES = [
    0,
    1,
    2,
    3,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    1 << 32,
    (1 << 32) + 1,
    (1 << 63) - 1,
    1 << 63,
    M64,
    0x0123456789ABCDEF,
    0xFEDCBA9876543210,
    0xDEADBEEFCAFEBABE,
    0x8000000000000001,
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mul_lo_64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mul_lo_64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


EXPORT_HALVES = [
    "v_and_b32 v5, 0xffff, v0",
    "v_lshrrev_b32 v6, 16, v0",
    "v_and_b32 v7, 0xffff, v1",
    "v_lshrrev_b32 v8, 16, v1",
    "v_or_b32 v5, 0x40000000, v5",
    "v_or_b32 v6, 0x40000000, v6",
    "v_or_b32 v7, 0x40000000, v7",
    "v_or_b32 v8, 0x40000000, v8",
]


def run_lines(lines) -> int:
    """Run the lines, export v[0:1] in 16-bit halves, return the pair as one 64-bit value."""
    if _TMP_DIR is None:
        setUpModule()
    body = list(lines) + EXPORT_HALVES
    asm = "; test_mul_lo_64\n" + "".join(f"\t{ln}\n" for ln in body) + "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all(x >> 16 == 0x4000 for x in w), [hex(x) for x in w]
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w))


class TestValuMulLoU64(unittest.TestCase):
    def test_zero_identity(self):
        cases = [
            (0, 0),
            (0, 1),
            (0, M64),
            (1, 0),
            (M64, 0),
        ]
        for a, b in cases:
            a_lo, a_hi = a & 0xFFFFFFFF, (a >> 32) & 0xFFFFFFFF
            b_lo, b_hi = b & 0xFFFFFFFF, (b >> 32) & 0xFFFFFFFF
            lines = [
                f"s_mov_b32 s2, {a_lo:#x}",
                f"s_mov_b32 s3, {a_hi:#x}",
                f"s_mov_b32 s4, {b_lo:#x}",
                f"s_mov_b32 s5, {b_hi:#x}",
                "v_mov_b32 v2, s2",
                "v_mov_b32 v3, s3",
                "v_mov_b32 v4, s4",
                "v_mov_b32 v5, s5",
                "v_mul_lo_u64 v[0:1], v[2:3], v[4:5]",
            ]
            self.assertEqual(run_lines(lines), 0)

    def test_one_identity(self):
        for val in BOUNDARY_VALUES:
            a_lo, a_hi = val & 0xFFFFFFFF, (val >> 32) & 0xFFFFFFFF
            lines = [
                f"s_mov_b32 s2, {a_lo:#x}",
                f"s_mov_b32 s3, {a_hi:#x}",
                "s_mov_b32 s4, 1",
                "s_mov_b32 s5, 0",
                "v_mov_b32 v2, s2",
                "v_mov_b32 v3, s3",
                "v_mov_b32 v4, s4",
                "v_mov_b32 v5, s5",
                "v_mul_lo_u64 v[0:1], v[2:3], v[4:5]",
            ]
            self.assertEqual(run_lines(lines), val)

    def test_boundary_pairs(self):
        pairs = [
            (2, 3),
            (0x10000, 0x10000),
            (0xFFFFFFFF, 2),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (1 << 32, 2),
            (1 << 32, 1 << 32),
            (M64, M64),
            (0x0123456789ABCDEF, 0xFEDCBA9876543210),
            (0xDEADBEEFCAFEBABE, 0x12345678),
        ]
        for a, b in pairs:
            expected = u64(a * b)
            a_lo, a_hi = a & 0xFFFFFFFF, (a >> 32) & 0xFFFFFFFF
            b_lo, b_hi = b & 0xFFFFFFFF, (b >> 32) & 0xFFFFFFFF
            lines = [
                f"s_mov_b32 s2, {a_lo:#x}",
                f"s_mov_b32 s3, {a_hi:#x}",
                f"s_mov_b32 s4, {b_lo:#x}",
                f"s_mov_b32 s5, {b_hi:#x}",
                "v_mov_b32 v2, s2",
                "v_mov_b32 v3, s3",
                "v_mov_b32 v4, s4",
                "v_mov_b32 v5, s5",
                "v_mul_lo_u64 v[0:1], v[2:3], v[4:5]",
            ]
            self.assertEqual(run_lines(lines), expected, f"v_mul_lo_u64(0x{a:016X}, 0x{b:016X})")

    def test_sgpr_sources(self):
        a = 0x123456789ABCDEF0
        b = 0x0FEDCBA987654321
        expected = u64(a * b)
        a_lo, a_hi = a & 0xFFFFFFFF, (a >> 32) & 0xFFFFFFFF
        b_lo, b_hi = b & 0xFFFFFFFF, (b >> 32) & 0xFFFFFFFF
        lines = [
            f"s_mov_b32 s2, {a_lo:#x}",
            f"s_mov_b32 s3, {a_hi:#x}",
            f"s_mov_b32 s4, {b_lo:#x}",
            f"s_mov_b32 s5, {b_hi:#x}",
            "v_mul_lo_u64 v[0:1], s[2:3], s[4:5]",
        ]
        self.assertEqual(run_lines(lines), expected)

    def test_imm_source(self):
        a = 0x123456789ABCDEF0
        b = 42
        expected = u64(a * b)
        a_lo, a_hi = a & 0xFFFFFFFF, (a >> 32) & 0xFFFFFFFF
        lines = [
            f"s_mov_b32 s2, {a_lo:#x}",
            f"s_mov_b32 s3, {a_hi:#x}",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
            "v_mul_lo_u64 v[0:1], 42, v[2:3]",
        ]
        self.assertEqual(run_lines(lines), expected)

    def test_random_samples(self):
        rng = random.Random(42)
        for _ in range(25):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            expected = u64(a * b)
            a_lo, a_hi = a & 0xFFFFFFFF, (a >> 32) & 0xFFFFFFFF
            b_lo, b_hi = b & 0xFFFFFFFF, (b >> 32) & 0xFFFFFFFF
            lines = [
                f"s_mov_b32 s2, {a_lo:#x}",
                f"s_mov_b32 s3, {a_hi:#x}",
                f"s_mov_b32 s4, {b_lo:#x}",
                f"s_mov_b32 s5, {b_hi:#x}",
                "v_mov_b32 v2, s2",
                "v_mov_b32 v3, s3",
                "v_mov_b32 v4, s4",
                "v_mov_b32 v5, s5",
                "v_mul_lo_u64 v[0:1], v[2:3], v[4:5]",
            ]
            self.assertEqual(run_lines(lines), expected, f"v_mul_lo_u64(0x{a:016X}, 0x{b:016X})")


if __name__ == "__main__":
    unittest.main()
