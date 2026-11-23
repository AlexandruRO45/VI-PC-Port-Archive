"""Semantic interpreter + spirv-val tests for RDNA2 s_movreld_b32, s_movreld_b64, and s_movrelsd_2_b32.

RDNA2 ISA SOP1:
  - s_movreld_b32 (0x30):    SGPR[D + M0] = S0
  - s_movreld_b64 (0x31):    SGPR[D + M0] = S0.lo; SGPR[D + 1 + M0] = S0.hi
  - s_movrelsd_2_b32 (0x49): SGPR[D + M0[25:16]] = SGPR[S0 + M0[9:0]] (other M0 bits ignored)

SCC is not written. Standard library only (spirv-val used when on PATH); no GPU, no game files.
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
POISON_SCC = {0: "s_cmp_eq_u32 0, 0", 1: "s_cmp_eq_u32 0, 1"}  # SCC = 1 / SCC = 0 before the op
SPIRV_VAL = shutil.which("spirv-val")

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_movreld.s")
    with open(os.path.join(_TMP_DIR.name, "ps_movreld.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_s_movreld\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spv(spv_text: str):
    binary = spirv_asm.assemble(spv_text, target_env="vulkan1.2")
    if not SPIRV_VAL:
        return
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tf:
        tf.write(binary)
        spv_path = tf.name
    try:
        res = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{res.stderr}")
    finally:
        os.unlink(spv_path)


def run(lines, *, check_val: bool = False):
    """Run `lines`, which leave D.lo in s10, D.hi in s11 and SCC in s12; return (D.lo, D.hi, SCC)."""
    tail = ["s_cselect_b32 s12, 1, 0"]
    for i, s in enumerate((10, 11)):
        tail += [
            f"v_mov_b32 v{10 + i}, s{s}",
            f"v_and_b32 v{2 * i}, 0xffff, v{10 + i}",
            f"v_lshrrev_b32 v{2 * i + 1}, 16, v{10 + i}",
            f"v_or_b32 v{2 * i}, 0x40000000, v{2 * i}",
            f"v_or_b32 v{2 * i + 1}, 0x40000000, v{2 * i + 1}",
        ]
    tail += ["v_mov_b32 v4, s12", "v_or_b32 v4, 0x40000000, v4"]
    spv = translate(list(lines) + tail + [
        "exp mrt0 v0, v1, v2, v3 vm",
        "exp mrt1 v4, v4, v4, v4 done vm",
        "s_endpgm",
    ])
    if check_val:
        validate_spv(spv)
    out = si.run_fragment(spv)
    w = list(out["%out_mrt0"]) + [out["%out_mrt1"][0]]
    assert all(v >> 16 == 0x4000 for v in w), [hex(v) for v in w]
    return (w[0] & 0xFFFF) | (w[1] & 0xFFFF) << 16, (w[2] & 0xFFFF) | (w[3] & 0xFFFF) << 16, w[4] & 0xFFFF


SEED = [
    "s_mov_b32 s20, 0x11111111",
    "s_mov_b32 s21, 0x22222222",
    "s_mov_b32 s22, 0x33333333",
    "s_mov_b32 s23, 0x44444444",
    "s_mov_b32 s24, 0x55555555",
    "s_mov_b32 s25, 0x66666666",
]


class MovreldTests(unittest.TestCase):
    def test_b32_sgpr_source_and_offsets(self):
        for m0 in (0, 1, 2, 5):
            with self.subTest(m0=m0):
                target = 20 + m0
                neighbor = 20 + ((m0 + 1) % 6)
                got = run(SEED + [
                    "s_mov_b32 s30, 0xdeadbeef",
                    f"s_mov_b32 m0, {m0}",
                    "s_movreld_b32 s20, s30",
                    f"s_mov_b32 s10, s{target}",
                    f"s_mov_b32 s11, s{neighbor}",
                ], check_val=(m0 == 2))
                self.assertEqual(got[:2], (0xDEADBEEF, 0x11111111 * (((m0 + 1) % 6) + 1)))

    def test_b32_inline_imm_literal_null_and_m0(self):
        # Inline positive immediate
        got = run(SEED + [
            "s_mov_b32 m0, 2",
            "s_movreld_b32 s20, 42",
            "s_mov_b32 s10, s22",
            "s_mov_b32 s11, s21",
        ])
        self.assertEqual(got[:2], (42, 0x22222222))

        # Inline negative immediate (-1)
        got = run(SEED + [
            "s_mov_b32 m0, 3",
            "s_movreld_b32 s20, -1",
            "s_mov_b32 s10, s23",
            "s_mov_b32 s11, s22",
        ])
        self.assertEqual(got[:2], (0xFFFFFFFF, 0x33333333))

        # 32-bit literal
        got = run(SEED + [
            "s_mov_b32 m0, 1",
            "s_movreld_b32 s20, 0xcafebabe",
            "s_mov_b32 s10, s21",
            "s_mov_b32 s11, s20",
        ])
        self.assertEqual(got[:2], (0xCAFEBABE, 0x11111111))

        # null source -> 0, and m0 source -> value of m0
        got = run(SEED + [
            "s_mov_b32 m0, 4",
            "s_movreld_b32 s20, null",
            "s_mov_b32 m0, 5",
            "s_movreld_b32 s20, m0",
            "s_mov_b32 s10, s24",
            "s_mov_b32 s11, s25",
        ])
        self.assertEqual(got[:2], (0, 5))

    def test_b32_dynamic_m0_wrap_and_scc_kept(self):
        for poison in (0, 1):
            with self.subTest(poison=poison):
                # m0 = 130 wraps modulo 128 to +2 -> writes s22
                got = run(SEED + [
                    "s_mov_b32 s5, 130",
                    "s_mov_b32 m0, s5",
                    POISON_SCC[poison],
                    "s_movreld_b32 s20, 0xabcdef01",
                    "s_mov_b32 s10, s22",
                    "s_mov_b32 s11, s20",
                ])
                self.assertEqual(got, (0xABCDEF01, 0x11111111, 1 - poison))

    def test_b64_sgpr_pair_and_immediates(self):
        # SGPR pair source
        got = run(SEED + [
            "s_mov_b32 s30, 0x12345678",
            "s_mov_b32 s31, 0x9abcdef0",
            "s_mov_b32 m0, 2",
            "s_movreld_b64 s[20:21], s[30:31]",
            "s_mov_b32 s10, s22",
            "s_mov_b32 s11, s23",
        ], check_val=True)
        self.assertEqual(got[:2], (0x12345678, 0x9ABCDEF0))

        # Inline immediate -1 sign-extends to 64 bits
        got = run(SEED + [
            "s_mov_b32 m0, 1",
            "s_movreld_b64 s[20:21], -1",
            "s_mov_b32 s10, s21",
            "s_mov_b32 s11, s22",
        ])
        self.assertEqual(got[:2], (0xFFFFFFFF, 0xFFFFFFFF))

        # 32-bit literal zero-extends to 64 bits
        got = run(SEED + [
            "s_mov_b32 m0, 3",
            "s_movreld_b64 s[20:21], 0x87654321",
            "s_mov_b32 s10, s23",
            "s_mov_b32 s11, s24",
        ])
        self.assertEqual(got[:2], (0x87654321, 0))

    def test_b64_overlapping_src_and_dst(self):
        # s[21:22] = old s[20:21] (dst+m0 overlaps src at s21)
        got = run(SEED + [
            "s_mov_b32 m0, 1",
            "s_movreld_b64 s[20:21], s[20:21]",
            "s_mov_b32 s10, s21",
            "s_mov_b32 s11, s22",
        ])
        self.assertEqual(got[:2], (0x11111111, 0x22222222))

        # s[20:21] = old s[21:22] (m0 = 0, dst overlaps src at s21)
        got = run(SEED + [
            "s_mov_b32 m0, 0",
            "s_movreld_b64 s[20:21], s[21:22]",
            "s_mov_b32 s10, s20",
            "s_mov_b32 s11, s21",
        ])
        self.assertEqual(got[:2], (0x22222222, 0x33333333))

    def test_b64_scc_kept(self):
        for poison in (0, 1):
            with self.subTest(poison=poison):
                got = run(SEED + [
                    "s_mov_b32 m0, 128",
                    POISON_SCC[poison],
                    "s_movreld_b64 s[20:21], s[24:25]",
                    "s_mov_b32 s10, s20",
                    "s_mov_b32 s11, s21",
                ])
                self.assertEqual(got, (0x55555555, 0x66666666, 1 - poison))

    def test_movrelsd_2_b32_fields_and_ignored_bits(self):
        # m0 = 0xFC03FC02 -> dst offset 3 (s23), src offset 2 (s22); ignored bits set
        got = run(SEED + [
            "s_mov_b32 m0, 0xfc03fc02",
            "s_movrelsd_2_b32 s20, s20",
            "s_mov_b32 s10, s23",
            "s_mov_b32 s11, s22",
        ], check_val=True)
        self.assertEqual(got[:2], (0x33333333, 0x33333333))

        # Distinct base registers: s21 + dst_off(4) = s25 <- s20 + src_off(1) = s21; SCC kept
        for poison in (0, 1):
            with self.subTest(poison=poison):
                got = run(SEED + [
                    "s_mov_b32 m0, 0x00040001",
                    POISON_SCC[poison],
                    "s_movrelsd_2_b32 s21, s20",
                    "s_mov_b32 s10, s25",
                    "s_mov_b32 s11, s22",
                ])
                self.assertEqual(got, (0x22222222, 0x33333333, 1 - poison))

    def test_sgpr_imm_cache_invalidated(self):
        # s22 is seeded with a known immediate (0x11111111) tracked by sgpr_imm.
        # s_movreld_b32 overwrites s22 dynamically; a subsequent s_add_u32 / s_mov_b32
        # reading s22 MUST NOT fold using the stale cached immediate.
        got = run([
            "s_mov_b32 s22, 0x11111111",
            "s_mov_b32 m0, 2",
            "s_movreld_b32 s20, 0x22222222",
            "s_add_u32 s10, s22, 1",
            "s_mov_b32 s11, s22",
        ])
        self.assertEqual(got[:2], (0x22222223, 0x22222222))

        # Same check for s_movreld_b64
        got = run([
            "s_mov_b32 s22, 0x11111111",
            "s_mov_b32 s23, 0x11111111",
            "s_mov_b32 m0, 2",
            "s_movreld_b64 s[20:21], 0x33333333",
            "s_add_u32 s10, s22, 1",
            "s_add_u32 s11, s23, 5",
        ])
        self.assertEqual(got[:2], (0x33333334, 5))

        # Same check for s_movrelsd_2_b32
        got = run([
            "s_mov_b32 s20, 0x44444444",
            "s_mov_b32 s22, 0x11111111",
            "s_mov_b32 m0, 0x00020000",
            "s_movrelsd_2_b32 s20, s20",
            "s_add_u32 s10, s22, 2",
            "s_mov_b32 s11, s22",
        ])
        self.assertEqual(got[:2], (0x44444446, 0x44444444))

    def test_live_lane_mask_sgpr_rejected(self):
        for op_line in (
            "s_movreld_b32 s20, s21",
            "s_movreld_b64 s[20:21], s[22:23]",
            "s_movrelsd_2_b32 s20, s21",
        ):
            with self.subTest(op=op_line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate([
                        "v_cmp_eq_u32 s30, v0, 0",
                        "s_mov_b32 m0, 0",
                        op_line,
                        "s_and_saveexec_b32 s31, s30",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ])

    def test_rejected_operand_forms(self):
        for bad in (
            "s_movrelsd_2_b32 s20, 5",
            "s_movreld_b32 m0, s20",
            "s_movreld_b64 s20, s[21:22]",
            "s_movreld_b64 s[20:21], s22",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate([bad, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
