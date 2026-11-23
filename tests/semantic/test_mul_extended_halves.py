"""Regression tests for 32x32 widening multiplies lowered through 16-bit half-products (Gen.mul_extended).

Intel's Windows Gen9.5 Vulkan driver (UHD Graphics 630, 31.0.101.2111) swaps the signedness of OpSMulExtended and
OpUMulExtended when constant-folding: OpSMulExtended folds as unsigned (cs_w32_int_ext out[4] = 0x3fffffff instead of
0xc0000000, out[5] = 0xff004000 instead of 0x4000) and OpUMulExtended folds as signed whenever an operand has bit 31
set (cs_w32_salu64 out[7] = 0 instead of 0xfffffffe). Lowering s_mul_hi_u32/i32, v_mul_hi_u32/i32,
v_mul_hi_u32_u24/i32_i24 and v_mad_u64_u32/i64_i32 through 16-bit half-products avoids both instructions while keeping
the module purely 32-bit.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TESTS_DIR)
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS_DIR, TESTS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_interp  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def s64(x: int) -> int:
    x &= M64
    return x - (1 << 64) if x & (1 << 63) else x


def u24(x: int) -> int:
    return x & 0xFFFFFF


def i24(x: int) -> int:
    v = x & 0xFFFFFF
    return v - (1 << 24) if v & 0x800000 else v


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mul_ext_halves.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mul_ext_halves.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_mul_extended_halves\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_ps(lines: list[str]) -> tuple[str, tuple[int, int, int, int]]:
    spv = translate_ps(lines)
    words_mini = tuple(si.run_fragment(spv)["%out_mrt0"][:4])
    res_interp = spirv_interp.run(spv)
    words_interp = tuple(spirv_interp.f2u(c) for c in res_interp.outputs["out_mrt0"][:4])
    if words_mini != words_interp:
        raise AssertionError(f"Interpreter mismatch: mini={words_mini} vs spirv_interp={words_interp}")
    return spv, words_mini


class MulExtendedHalvesTests(unittest.TestCase):
    def _assert_no_mul_extended(self, spv: str):
        self.assertNotIn("OpSMulExtended", spv)
        self.assertNotIn("OpUMulExtended", spv)
        blob = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertEqual(struct.unpack_from("<I", blob, 0)[0], 0x07230203)

    def test_intel_gen95_failing_constants_mul_hi(self):
        """Exact constants that Intel Windows 31.0.101.2111 mis-folded via OpSMulExtended / OpUMulExtended."""
        spv, got = run_ps([
            "s_mov_b32 s4, 0x80000000",
            "s_mov_b32 s5, 0x7fffffff",
            "s_mul_hi_i32 s6, s4, s5",
            "s_mul_hi_u32 s7, s4, s5",
            "v_mov_b32 v4, 0x800000",
            "v_mul_hi_i32_i24 v2, v4, v4",
            "v_mul_hi_u32_u24 v3, 0xffffff, 0xffffff",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
        ])
        self._assert_no_mul_extended(spv)
        want = (
            ((s32(0x80000000) * s32(0x7FFFFFFF)) >> 32) & M32,  # 0xC0000000 (Intel OpSMulExtended folded 0x3FFFFFFF)
            ((0x80000000 * 0x7FFFFFFF) >> 32) & M32,            # 0x3FFFFFFF
            ((i24(0x800000) * i24(0x800000)) >> 32) & M32,      # 0x00004000 (Intel OpSMulExtended folded 0xFF004000)
            ((u24(0xFFFFFF) * u24(0xFFFFFF)) >> 32) & M32,      # 0x0000FFFF
        )
        self.assertEqual(got, want)

    def test_v_mul_hi_i32_and_u32_edge_cases(self):
        cases = [
            (0x80000000, 0x80000000),
            (0x80000000, 0x7FFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFD, 0x00000005),
            (0x12345678, 0x9ABCDEF0),
            (0x7FFF8000, 0x80007FFF),
        ]
        for a, b in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                spv, got = run_ps([
                    f"s_mov_b32 s4, {a:#x}",
                    f"s_mov_b32 s5, {b:#x}",
                    "s_mul_hi_i32 s6, s4, s5",
                    "s_mul_hi_u32 s7, s4, s5",
                    f"v_mov_b32 v4, {a:#x}",
                    f"v_mov_b32 v5, {b:#x}",
                    "v_mul_hi_i32 v2, v4, v5",
                    "v_mul_hi_u32 v3, v4, v5",
                    "v_mov_b32 v0, s6",
                    "v_mov_b32 v1, s7",
                ])
                self._assert_no_mul_extended(spv)
                want_i = ((s32(a) * s32(b)) >> 32) & M32
                want_u = ((a * b) >> 32) & M32
                self.assertEqual(got, (want_i, want_u, want_i, want_u))

    def test_v_mad_u64_u32_and_i64_i32_intel_constants(self):
        """v_mad_u64_u32 / v_mad_i64_i32 on bit-31-set constants where Intel OpUMulExtended folded as signed."""
        u_cases = [
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x76543210),
            (0x80000000, 0x80000000, 0xFFFFFFFF, 0x00000001),
        ]
        for x, y, clo, chi in u_cases:
            with self.subTest(op="v_mad_u64_u32", x=hex(x), y=hex(y)):
                spv, got = run_ps([
                    f"v_mov_b32 v4, {x:#x}",
                    f"v_mov_b32 v5, {y:#x}",
                    f"v_mov_b32 v6, {clo:#x}",
                    f"v_mov_b32 v7, {chi:#x}",
                    "v_mad_u64_u32 v[0:1], s[10:11], v4, v5, v[6:7]",
                    "v_cndmask_b32_e64 v2, 0, 1, s[10:11]",
                    "v_mov_b32 v3, 0",
                ])
                self._assert_no_mul_extended(spv)
                tot = x * y + ((chi << 32) | clo)
                self.assertEqual(got, (tot & M32, (tot >> 32) & M32, int((tot >> 64) != 0), 0))

        i_cases = [
            (0x80000000, 0x7FFFFFFF, 0x00000001, 0x00000000),
            (0x80000000, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF),
            (0xFFFFFFFF, 0x00000005, 0x00000000, 0x00000000),
        ]
        for x, y, clo, chi in i_cases:
            with self.subTest(op="v_mad_i64_i32", x=hex(x), y=hex(y)):
                spv, got = run_ps([
                    f"v_mov_b32 v4, {x:#x}",
                    f"v_mov_b32 v5, {y:#x}",
                    f"v_mov_b32 v6, {clo:#x}",
                    f"v_mov_b32 v7, {chi:#x}",
                    "v_mad_i64_i32 v[0:1], s[10:11], v4, v5, v[6:7]",
                    "v_cndmask_b32_e64 v2, 0, 1, s[10:11]",
                    "v_mov_b32 v3, 0",
                ])
                self._assert_no_mul_extended(spv)
                exact = s32(x) * s32(y) + s64((chi << 32) | clo)
                u64 = exact & M64
                self.assertEqual(got, (u64 & M32, (u64 >> 32) & M32, int(exact < 0), 0))


if __name__ == "__main__":
    unittest.main()
