"""Semantic interpreter tests for RDNA2 v_min_u64, v_max_u64, v_min_i64 and v_max_i64.

Tests the 64-bit integer VALU min/max family:
  v_min_u64: D = min(S0, S1) (unsigned 64-bit)
  v_max_u64: D = max(S0, S1) (unsigned 64-bit)
  v_min_i64: D = min(S0, S1) (signed 64-bit)
  v_max_i64: D = max(S0, S1) (signed 64-bit)

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


def s64(x: int) -> int:
    x &= M64
    return x - (1 << 64) if (x >> 63) else x


def u64(x: int) -> int:
    return x & M64


OPS = {
    "v_min_u64": lambda a, b: min(u64(a), u64(b)),
    "v_max_u64": lambda a, b: max(u64(a), u64(b)),
    "v_min_i64": lambda a, b: u64(min(s64(a), s64(b))),
    "v_max_i64": lambda a, b: u64(max(s64(a), s64(b))),
}

BOUNDARY_VALUES = [
    0,
    1,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    1 << 32,
    (1 << 63) - 1,
    1 << 63,
    M64,
    0x0123456789ABCDEF,
    0xFEDCBA9876543210,
    0x8000000000000001,
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_minmax_64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_minmax_64.dxbc"), "wb") as f:
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
    asm = "; test_minmax_64\n" + "".join(f"\t{ln}\n" for ln in body) + "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all(x >> 16 == 0x4000 for x in w), [hex(x) for x in w]
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w))


def run_op(op: str, a: int, b: int) -> int:
    return run_lines([
        f"v_mov_b32 v2, {a & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v3, {(a >> 32) & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v4, {b & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v5, {(b >> 32) & 0xFFFFFFFF:#x}",
        f"{op} v[0:1], v[2:3], v[4:5]",
    ])


class MinMax64Tests(unittest.TestCase):
    def test_unsigned_min(self):
        for a in BOUNDARY_VALUES:
            for b in BOUNDARY_VALUES:
                with self.subTest(a=hex(a), b=hex(b)):
                    expected = OPS["v_min_u64"](a, b)
                    got = run_op("v_min_u64", a, b)
                    self.assertEqual(got, expected, f"v_min_u64({hex(a)}, {hex(b)})")

    def test_unsigned_max(self):
        for a in BOUNDARY_VALUES:
            for b in BOUNDARY_VALUES:
                with self.subTest(a=hex(a), b=hex(b)):
                    expected = OPS["v_max_u64"](a, b)
                    got = run_op("v_max_u64", a, b)
                    self.assertEqual(got, expected, f"v_max_u64({hex(a)}, {hex(b)})")

    def test_signed_min(self):
        for a in BOUNDARY_VALUES:
            for b in BOUNDARY_VALUES:
                with self.subTest(a=hex(a), b=hex(b)):
                    expected = OPS["v_min_i64"](a, b)
                    got = run_op("v_min_i64", a, b)
                    self.assertEqual(got, expected, f"v_min_i64({hex(a)}, {hex(b)})")

    def test_signed_max(self):
        for a in BOUNDARY_VALUES:
            for b in BOUNDARY_VALUES:
                with self.subTest(a=hex(a), b=hex(b)):
                    expected = OPS["v_max_i64"](a, b)
                    got = run_op("v_max_i64", a, b)
                    self.assertEqual(got, expected, f"v_max_i64({hex(a)}, {hex(b)})")

    def test_sgpr_and_immediate_operands(self):
        lines = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0x00000001",
            "v_mov_b32 v2, 0x87654321",
            "v_mov_b32 v3, 0x00000002",
            "v_min_u64 v[0:1], s[2:3], v[2:3]",
        ]
        self.assertEqual(run_lines(lines), 0x112345678)

        lines = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0x00000001",
            "v_mov_b32 v2, 0x87654321",
            "v_mov_b32 v3, 0x00000002",
            "v_max_u64 v[0:1], s[2:3], v[2:3]",
        ]
        self.assertEqual(run_lines(lines), 0x287654321)

    def test_signed_negative_vs_positive(self):
        self.assertEqual(run_op("v_min_i64", M64, 1), M64)
        self.assertEqual(run_op("v_min_u64", M64, 1), 1)

        self.assertEqual(run_op("v_max_i64", M64, 1), 1)
        self.assertEqual(run_op("v_max_u64", M64, 1), M64)

    def test_random_pairs(self):
        rng = random.Random(42)
        for _ in range(25):
            a = rng.randint(0, M64)
            b = rng.randint(0, M64)
            for op in ("v_min_u64", "v_max_u64", "v_min_i64", "v_max_i64"):
                expected = OPS[op](a, b)
                got = run_op(op, a, b)
                self.assertEqual(got, expected, f"{op}({hex(a)}, {hex(b)})")

    def test_invalid_operands_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_min_u64 s[0:1], v[2:3], v[4:5]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_min_u64 v0, v[2:3], v[4:5]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_max_i64 v[0:1], v2, v[4:5]"])


if __name__ == "__main__":
    unittest.main()
