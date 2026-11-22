"""Semantic interpreter tests for RDNA2 v_mullit_f32 (VOP3 opcode 0x150).

v_mullit_f32 D, S0, S1, S2 computes:
  D = (S2 > 0.0 && S1 > -FLT_MAX) ? mul_legacy_f32(S0, S1) : -FLT_MAX
after applying VOP3 input abs/neg and flushing f32 denormals to signed zero (the default game MODE:
IEEE=0, DX10_CLAMP=1, denorm32=0), followed by VOP3 output scaling (with -0.0 -> +0.0 and post-div:2
denormal flush) and DX10 clamp to [0, 1] (NaN -> +0.0, -0.0 -> +0.0).

All edge cases and modifier interactions were measured natively on RDNA2 (AMD Radeon Graphics
gfx1036, Windows driver 32.0.21042.62, wave32 & wave64 via tools/hw_oracle: 1,530 cases / 3,060
executions, 0 mismatches).
"""
import math
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
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

P0 = 0x00000000
N0 = 0x80000000
PDEN = 0x00000001
NDEN = 0x80000001
PDEN_MAX = 0x007FFFFF
FLT_MIN = 0x00800000
NFLT_MIN = 0x80800000
P2M120 = 0x03800000
N2M120 = 0x83800000
HALF = 0x3F000000
ONE = 0x3F800000
NONE = 0xBF800000
TWO = 0x40000000
NTWO = 0xC0000000
THREE = 0x40400000
FOUR = 0x40800000
SIX = 0x40C00000
FLT_MAX = 0x7F7FFFFF
NFLT_MAX = 0xFF7FFFFF
NFLT_MAX_P1 = 0xFF7FFFFE
PINF = 0x7F800000
NINF = 0xFF800000
QNAN = 0x7FC00123
SNAN = 0x7F800456

S0_SENT = 0x11110001
S1_SENT = 0x22220002
S2_SENT = 0x33330003

_TMP_DIR = None
_SRC_PATH = None
_CS_PATH = None
_CS_FSSM = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mullit.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_mullit.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_mullit.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_mullit.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def f2u(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return PINF if x > 0 else NINF


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_nan_u32(u: int) -> bool:
    return ((u & 0x7F800000) == 0x7F800000) and ((u & 0x007FFFFF) != 0)


def flush_f32_u32(u: int) -> int:
    u &= 0xFFFFFFFF
    if (u & 0x7F800000) == 0:
        return u & 0x80000000
    return u


def ref_mullit_f32(s0: int, s1: int, s2: int, omod=None, clamp: bool = False) -> int:
    """Reference model matching native RDNA2 gfx1036 in default game MODE (IEEE=0, DX10_CLAMP=1, denorm32=0)."""
    x = flush_f32_u32(s0)
    y = flush_f32_u32(s1)
    z = flush_f32_u32(s2)

    z_pos = (not is_nan_u32(z)) and (u2f(z) > 0.0)
    y_gt = (not is_nan_u32(y)) and (u2f(y) > u2f(NFLT_MAX))
    if z_pos and y_gt:
        if (x & 0x7FFFFFFF) == 0 or (y & 0x7FFFFFFF) == 0:
            r = P0
        elif is_nan_u32(x):
            r = x
        else:
            r = flush_f32_u32(f2u(u2f(x) * u2f(y)))
    else:
        r = NFLT_MAX

    if omod is not None:
        if (r & 0x7FFFFFFF) == 0:
            r = P0
        if not is_nan_u32(r):
            k = {"mul:2": 2.0, "mul:4": 4.0, "div:2": 0.5}[omod]
            r = flush_f32_u32(f2u(u2f(r) * k))

    if clamp:
        if is_nan_u32(r) or u2f(r) <= 0.0:
            r = P0
        elif u2f(r) >= 1.0:
            r = ONE
    return r


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_mullit_f32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def translate_template(op_line: str) -> str:
    return translate_ps([
        f"v_mov_b32 v0, {S0_SENT:#x}",
        f"v_mov_b32 v1, {S1_SENT:#x}",
        f"v_mov_b32 v2, {S2_SENT:#x}",
        "v_mov_b32 v3, 0",
        op_line,
        "exp mrt0 v3, v3, v3, v3 done vm",
    ])


def eval_template(spv: str, s0: int, s1: int, s2: int) -> int:
    patched = (
        spv.replace(f"OpConstant %uint {S0_SENT}", f"OpConstant %uint {s0 & 0xFFFFFFFF}")
        .replace(f"OpConstant %uint {S1_SENT}", f"OpConstant %uint {s1 & 0xFFFFFFFF}")
        .replace(f"OpConstant %uint {S2_SENT}", f"OpConstant %uint {s2 & 0xFFFFFFFF}")
    )
    return si.run_fragment(patched)["%out_mrt0"][0]


class TestMullitF32(unittest.TestCase):
    def assertFloatWord(self, got: int, expected: int, msg: str = ""):
        if is_nan_u32(expected):
            self.assertTrue(is_nan_u32(got), f"{msg}: expected NaN, got {got:#010x}")
        else:
            self.assertEqual(got, expected, f"{msg}: got {got:#010x}, expected {expected:#010x}")

    def test_gfx1036_pinned_edges(self):
        """Pinned words measured natively on RDNA2 gfx1036 (wave32 and wave64, default game MODE)."""
        spv = translate_template("v_mullit_f32 v3, v0, v1, v2")
        # S2 gate: must be strictly > +0.0 after f32 denormal flush
        cases = [
            ("s2=+0", TWO, THREE, P0, NFLT_MAX),
            ("s2=-0", TWO, THREE, N0, NFLT_MAX),
            ("s2=+den", TWO, THREE, PDEN, NFLT_MAX),
            ("s2=+den_max", TWO, THREE, PDEN_MAX, NFLT_MAX),
            ("s2=-den", TWO, THREE, NDEN, NFLT_MAX),
            ("s2=+FLT_MIN", TWO, THREE, FLT_MIN, SIX),
            ("s2=-FLT_MIN", TWO, THREE, NFLT_MIN, NFLT_MAX),
            ("s2=-1", TWO, THREE, NONE, NFLT_MAX),
            ("s2=+inf", TWO, THREE, PINF, SIX),
            ("s2=-inf", TWO, THREE, NINF, NFLT_MAX),
            ("s2=qNaN", TWO, THREE, QNAN, NFLT_MAX),
            ("s2=sNaN", TWO, THREE, SNAN, NFLT_MAX),
            # S1 gate: must be strictly > -FLT_MAX (excludes -FLT_MAX, -inf, NaN)
            ("s1=-FLT_MAX+1ulp", ONE, NFLT_MAX_P1, ONE, NFLT_MAX_P1),
            ("s1=-FLT_MAX s0=+0.5", HALF, NFLT_MAX, ONE, NFLT_MAX),
            ("s1=-FLT_MAX s0=+0", P0, NFLT_MAX, ONE, NFLT_MAX),
            ("s1=-FLT_MAX s0=qNaN", QNAN, NFLT_MAX, ONE, NFLT_MAX),
            ("s1=-inf s0=+0", P0, NINF, ONE, NFLT_MAX),
            ("s1=-inf s0=-2", NTWO, NINF, ONE, NFLT_MAX),
            ("s1=qNaN s0=+0", P0, QNAN, ONE, NFLT_MAX),
            ("s1=sNaN s0=+0", P0, SNAN, ONE, NFLT_MAX),
            ("s1=qNaN s0=qNaN", QNAN, QNAN, ONE, NFLT_MAX),
            ("s1=+inf s0=+2", TWO, PINF, ONE, PINF),
            ("s1=+inf s0=-1", NONE, PINF, ONE, NINF),
            # Legacy multiply when both gates pass: 0 * anything = +0.0 (after denorm flush)
            ("s0=+0 s1=+inf", P0, PINF, ONE, P0),
            ("s0=-0 s1=+inf", N0, PINF, ONE, P0),
            ("s0=+den s1=+inf", PDEN, PINF, ONE, P0),
            ("s0=qNaN s1=+0", QNAN, P0, ONE, P0),
            ("s0=qNaN s1=-0", QNAN, N0, ONE, P0),
            ("s0=sNaN s1=+0", SNAN, P0, ONE, P0),
            ("s0=qNaN s1=+den", QNAN, PDEN, ONE, P0),
            ("s0=-0 s1=+1", N0, ONE, ONE, P0),
            ("s0=+0 s1=-1", P0, NONE, ONE, P0),
            ("s0=-0 s1=-1", N0, NONE, ONE, P0),
            # Underflow flush keeps product sign; non-zero S0 NaN propagates
            ("underflow +", P2M120, P2M120, ONE, P0),
            ("underflow -", N2M120, P2M120, ONE, N0),
            ("overflow +", FLT_MAX, TWO, ONE, PINF),
            ("overflow -", NFLT_MAX, TWO, ONE, NINF),
            ("s0=qNaN s1=+2", QNAN, TWO, ONE, QNAN),
            ("s0=sNaN s1=+2", SNAN, TWO, ONE, QNAN),
            # S2 <= 0 overrides S0 NaN
            ("s0=qNaN s2=-1", QNAN, ONE, NONE, NFLT_MAX),
        ]
        for label, s0, s1, s2, expected in cases:
            got = eval_template(spv, s0, s1, s2)
            self.assertFloatWord(got, expected, label)

    def test_output_modifiers_and_clamp(self):
        """VOP3 omod (with -0 -> +0 and post-div:2 flush) runs before DX10 clamp."""
        mod_specs = [
            ("v_mullit_f32 v3, v0, v1, v2 clamp", None, True, [
                ("s2=-1 clamp", TWO, THREE, NONE, P0),
                ("2*3 clamp", TWO, THREE, ONE, ONE),
                ("0.5*0.5 clamp", HALF, HALF, ONE, 0x3E800000),
                ("qNaN clamp", QNAN, TWO, ONE, P0),
                ("sNaN clamp", SNAN, TWO, ONE, P0),
                ("neg underflow clamp", N2M120, P2M120, ONE, P0),
            ]),
            ("v_mullit_f32 v3, v0, v1, v2 mul:2", "mul:2", False, [
                ("s2=-1 mul:2", TWO, THREE, NONE, NINF),
                ("0.5*0.5 mul:2", HALF, HALF, ONE, HALF),
                ("neg underflow mul:2 -> +0", N2M120, P2M120, ONE, P0),
                ("qNaN mul:2", QNAN, TWO, ONE, QNAN),
            ]),
            ("v_mullit_f32 v3, v0, v1, v2 mul:4", "mul:4", False, [
                ("s2=-1 mul:4", TWO, THREE, NONE, NINF),
                ("FLT_MAX*0.5 mul:4", FLT_MAX, HALF, ONE, PINF),
            ]),
            ("v_mullit_f32 v3, v0, v1, v2 div:2", "div:2", False, [
                ("s2=-1 div:2", TWO, THREE, NONE, 0xFEFFFFFF),
                ("+FLT_MIN*1 div:2 -> +0", FLT_MIN, ONE, ONE, P0),
                ("-FLT_MIN*1 div:2 -> -0", NFLT_MIN, ONE, ONE, N0),
                ("neg underflow div:2 -> +0", N2M120, P2M120, ONE, P0),
                ("FLT_MAX*2 div:2 -> +inf", FLT_MAX, TWO, ONE, PINF),
            ]),
            ("v_mullit_f32 v3, v0, v1, v2 clamp mul:2", "mul:2", True, [
                ("s2=-1 mul:2 clamp", TWO, THREE, NONE, P0),
                ("0.5*0.5 mul:2 clamp", HALF, HALF, ONE, HALF),
                ("0.5*1.0 mul:2 clamp", HALF, ONE, ONE, ONE),
            ]),
            ("v_mullit_f32 v3, v0, v1, v2 clamp mul:4", "mul:4", True, [
                ("s2=-1 mul:4 clamp", TWO, THREE, NONE, P0),
                ("0.5*1.0 mul:4 clamp", HALF, ONE, ONE, ONE),
            ]),
        ]
        for op_line, omod, clamp, cases in mod_specs:
            spv = translate_template(op_line)
            for label, s0, s1, s2, expected in cases:
                got = eval_template(spv, s0, s1, s2)
                self.assertFloatWord(got, expected, label)
                self.assertFloatWord(got, ref_mullit_f32(s0, s1, s2, omod=omod, clamp=clamp), f"ref {label}")

    def test_input_modifiers_before_range_checks(self):
        """Input abs/neg apply before S2 > 0.0 and S1 > -FLT_MAX checks."""
        spv_neg_abs_s1 = translate_template("v_mullit_f32 v3, v0, -|v1|, v2")
        # -|+FLT_MAX| == -FLT_MAX fails the S1 > -FLT_MAX gate
        self.assertEqual(eval_template(spv_neg_abs_s1, TWO, FLT_MAX, ONE), NFLT_MAX)
        # -|-3.0| == -3.0 passes the gate and multiplies with 2.0 -> -6.0
        self.assertEqual(eval_template(spv_neg_abs_s1, TWO, 0xC0400000, ONE), 0xC0C00000)

        spv_abs_s1 = translate_template("v_mullit_f32 v3, v0, |v1|, v2")
        # |-FLT_MAX| == +FLT_MAX passes the gate: 0.5 * +FLT_MAX = 0x7EFFFFFF
        self.assertEqual(eval_template(spv_abs_s1, HALF, NFLT_MAX, ONE), 0x7EFFFFFF)

        spv_neg_s2 = translate_template("v_mullit_f32 v3, v0, v1, -v2")
        # -(-1.0) == +1.0 > 0 passes the S2 gate
        self.assertEqual(eval_template(spv_neg_s2, TWO, THREE, NONE), SIX)
        # -(+1.0) == -1.0 fails the S2 gate
        self.assertEqual(eval_template(spv_neg_s2, TWO, THREE, ONE), NFLT_MAX)

    def test_sgpr_inline_literal_and_aliasing(self):
        """SGPR sources, inline float constants, 32-bit literals, and destination aliasing."""
        # SGPR + inline constant
        spv = translate_ps([
            f"s_mov_b32 s4, {TWO:#x}",
            f"v_mov_b32 v1, {THREE:#x}",
            "v_mullit_f32 v0, s4, v1, 1.0",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], SIX)

        # Literal source and destination aliasing each source operand
        for dst_reg in ("v0", "v1", "v2"):
            spv = translate_ps([
                f"v_mov_b32 v0, {TWO:#x}",
                f"v_mov_b32 v1, {THREE:#x}",
                f"v_mov_b32 v2, {ONE:#x}",
                f"v_mullit_f32 {dst_reg}, v0, v1, v2",
                f"exp mrt0 {dst_reg}, {dst_reg}, {dst_reg}, {dst_reg} done vm",
            ])
            self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], SIX, f"alias {dst_reg}")

        spv_lit = translate_ps([
            f"v_mov_b32 v1, {THREE:#x}",
            f"v_mov_b32 v2, {ONE:#x}",
            f"v_mullit_f32 v0, {TWO:#x}, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        self.assertEqual(si.run_fragment(spv_lit)["%out_mrt0"][0], SIX)

    def test_seeded_sweep(self):
        """100 seeded random triples against the gfx1036 reference model."""
        spv = translate_template("v_mullit_f32 v3, v0, v1, v2")
        pool = [P0, N0, PDEN, NDEN, FLT_MIN, NFLT_MIN, P2M120, N2M120, HALF, ONE, NONE, TWO, FLT_MAX, NFLT_MAX_P1, NFLT_MAX, PINF, NINF, QNAN, SNAN]
        rng = random.Random(0x4D554C4C)
        for i in range(100):
            s0, s1, s2 = (rng.choice(pool) if rng.random() < 0.4 else rng.getrandbits(32) for _ in range(3))
            got = eval_template(spv, s0, s1, s2)
            expected = ref_mullit_f32(s0, s1, s2)
            self.assertFloatWord(got, expected, f"seed[{i}] ({s0:#010x}, {s1:#010x}, {s2:#010x})")

    def test_inactive_exec_lanes_preserved(self):
        """Inactive EXEC lanes keep their previous destination register value."""
        spv_ps = translate_ps([
            f"v_mov_b32 v0, {TWO:#x}",
            f"v_mov_b32 v1, {THREE:#x}",
            f"v_mov_b32 v2, {ONE:#x}",
            "v_mov_b32 v3, 0xdeadbeef",
            "s_mov_b32 exec_lo, 0",
            "v_mullit_f32 v3, v0, v1, v2",
            "s_mov_b32 exec_lo, 1",
            "exp mrt0 v3, v3, v3, v3 done vm",
        ])
        self.assertEqual(si.run_fragment(spv_ps)["%out_mrt0"][0], 0xDEADBEEF)

        with open(_CS_FSSM, "wb") as f:
            f.write(_make_fssm(0x0088, (8, 1, 1)))
        cs_lines = [
            "v_lshlrev_b32 v4, 2, v0",
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v2, {THREE:#x}",
            f"v_mov_b32 v5, {ONE:#x}",
            "v_mov_b32 v3, 0xdeadbeef",
            "v_and_b32 v6, v0, 1",
            "v_cmpx_eq_u32 v6, 0",
            "v_mullit_f32 v3, v1, v2, v5",
            "s_mov_b32 exec_lo, -1",
            "buffer_store_dword v3, v4, s[0:3], 0 offen",
        ]
        asm = "; test_cs_mullit\n" + "".join(f"\t{ln}\n" for ln in cs_lines) + "\ts_endpgm\n"
        with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv_cs = rdna2spirv.translate(_CS_PATH)
        mem: dict[int, int] = {}
        si.run_workgroup(
            spv_cs,
            8,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        for lane in range(8):
            expected = 0xDEADBEEF if (lane & 1) else SIX
            self.assertEqual(mem[4 * lane], expected, f"lane {lane}")

    def test_assembles_without_float64_or_int64(self):
        """Translated SPIR-V assembles cleanly and needs no Float64 or Int64 capability."""
        spv = translate_template("v_mullit_f32 v3, v0, v1, v2 clamp mul:2")
        self.assertNotIn("OpCapability Float64", spv)
        self.assertNotIn("OpCapability Int64", spv)
        binary = spirv_asm.assemble(spv)
        self.assertGreater(len(binary), 20)


if __name__ == "__main__":
    unittest.main()
