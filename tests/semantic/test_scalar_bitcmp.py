"""Semantic tests for RDNA2 scalar bit test and zero-counting instructions.

Covers:
  - s_bitcmp0_b32: SCC = ((S0 >> (S1 & 31)) & 1) == 0
  - s_bitcmp1_b32: SCC = ((S0 >> (S1 & 31)) & 1) == 1
  - s_bitcmp0_b64: SCC = ((S0_64 >> (S1 & 63)) & 1) == 0
  - s_bitcmp1_b64: SCC = ((S0_64 >> (S1 & 63)) & 1) == 1
  - s_bcnt0_i32_b32: D = count_zero_bits(S0), SCC = (D != 0)
  - s_bcnt0_i32_b64: D = count_zero_bits(S0_64), SCC = (D != 0)
  - s_ff0_i32_b32: D = find_first_zero_bit(S0), -1 if none
  - s_ff0_i32_b64: D = find_first_zero_bit(S0_64), -1 if none
  - s_ff1_i32_b32/b64: D = find_first_set_bit(S0), -1 if none
  - Non-mask `_b32` sources for s_ff0/s_ff1 are scalar values, including float immediates and m0
  - Lane mask awareness: testing bits on lane-mask registers (exec / vcc)
  - SPIR-V translation validation via rdna2spirv.Gen
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv, translate_lines  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
PUSH_VSHARP = [0x1000, 0, 0x1000, 0x31016FAC] + [0] * 28


def ref_bitcmp0_b32(s0: int, s1: int) -> int:
    bit = ((s0 & MASK32) >> (s1 & 31)) & 1
    return 1 if bit == 0 else 0


def ref_bitcmp1_b32(s0: int, s1: int) -> int:
    bit = ((s0 & MASK32) >> (s1 & 31)) & 1
    return 1 if bit == 1 else 0


def ref_bitcmp0_b64(s0_64: int, s1: int) -> int:
    bit = ((s0_64 & MASK64) >> (s1 & 63)) & 1
    return 1 if bit == 0 else 0


def ref_bitcmp1_b64(s0_64: int, s1: int) -> int:
    bit = ((s0_64 & MASK64) >> (s1 & 63)) & 1
    return 1 if bit == 1 else 0


def ref_bcnt0_b32(s0: int) -> tuple[int, int]:
    val = (s0 & MASK32) ^ MASK32
    count = bin(val).count("1")
    return count, 1 if count != 0 else 0


def ref_bcnt0_b64(s0_64: int) -> tuple[int, int]:
    val = (s0_64 & MASK64) ^ MASK64
    count = bin(val).count("1")
    return count, 1 if count != 0 else 0


def ref_ff0_b32(s0: int) -> int:
    val = (s0 & MASK32) ^ MASK32
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


def ref_ff0_b64(s0_64: int) -> int:
    val = (s0_64 & MASK64) ^ MASK64
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


class TestBitcmpAndBcnt0Semantics(unittest.TestCase):
    """Unit tests for reference models of bit test and zero-counting instructions."""

    def test_bitcmp_b32_all_bits(self):
        for bit_idx in range(32):
            pattern = 1 << bit_idx
            self.assertEqual(ref_bitcmp1_b32(pattern, bit_idx), 1)
            self.assertEqual(ref_bitcmp0_b32(pattern, bit_idx), 0)
            # Test inverted pattern
            self.assertEqual(ref_bitcmp1_b32(~pattern, bit_idx), 0)
            self.assertEqual(ref_bitcmp0_b32(~pattern, bit_idx), 1)

    def test_bitcmp_b64_all_bits(self):
        for bit_idx in range(64):
            pattern = 1 << bit_idx
            self.assertEqual(ref_bitcmp1_b64(pattern, bit_idx), 1)
            self.assertEqual(ref_bitcmp0_b64(pattern, bit_idx), 0)
            # Test inverted pattern
            self.assertEqual(ref_bitcmp1_b64(~pattern, bit_idx), 0)
            self.assertEqual(ref_bitcmp0_b64(~pattern, bit_idx), 1)

    def test_bitcmp_modulo_shift(self):
        # 32-bit: bits [31:5] are ignored (s1 & 31)
        self.assertEqual(ref_bitcmp1_b32(0x80000000, 31 + 64), 1)
        self.assertEqual(ref_bitcmp0_b32(0x80000000, 31 + 64), 0)
        # 64-bit: bits [31:6] are ignored (s1 & 63)
        self.assertEqual(ref_bitcmp1_b64(0x8000000000000000, 63 + 128), 1)
        self.assertEqual(ref_bitcmp0_b64(0x8000000000000000, 63 + 128), 0)

    def test_bcnt0_b32_boundaries(self):
        self.assertEqual(ref_bcnt0_b32(0), (32, 1))
        self.assertEqual(ref_bcnt0_b32(MASK32), (0, 0))
        self.assertEqual(ref_bcnt0_b32(0x55555555), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0x0000FFFF), (16, 1))

    def test_bcnt0_b64_boundaries(self):
        self.assertEqual(ref_bcnt0_b64(0), (64, 1))
        self.assertEqual(ref_bcnt0_b64(MASK64), (0, 0))
        self.assertEqual(ref_bcnt0_b64(0x5555555555555555), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0x00000000FFFFFFFF), (32, 1))

    def test_ff0_b32(self):
        self.assertEqual(ref_ff0_b32(MASK32), -1)
        self.assertEqual(ref_ff0_b32(0), 0)
        self.assertEqual(ref_ff0_b32(1), 1)
        self.assertEqual(ref_ff0_b32(0b111), 3)
        self.assertEqual(ref_ff0_b32(0x7FFFFFFF), 31)

    def test_ff0_b64(self):
        self.assertEqual(ref_ff0_b64(MASK64), -1)
        self.assertEqual(ref_ff0_b64(0), 0)
        self.assertEqual(ref_ff0_b64(1), 1)
        self.assertEqual(ref_ff0_b64(MASK32), 32)
        self.assertEqual(ref_ff0_b64(0x7FFFFFFFFFFFFFFF), 63)


class TestRdna2SpirvBitcmpEmission(unittest.TestCase):
    """Test SPIR-V code generation in rdna2spirv for bitcmp, bcnt0, ff0."""

    def test_s_bitcmp1_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 7},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpINotEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp0_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp0_b32",
            "args": [
                {"k": "sgpr", "r": 2},
                {"k": "sgpr", "r": 3},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpIEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp1_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b64",
            "args": [
                {"k": "sgpr", "r": 4, "n": 2},
                {"k": "imm", "v": 45},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpUGreaterThanEqual %bool", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpINotEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp0_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp0_b64",
            "args": [
                {"k": "sgpr", "r": 4, "n": 2},
                {"k": "sgpr", "r": 6},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpUGreaterThanEqual %bool", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpIEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp1_b64_exec_lane_mask(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b64",
            "args": [
                {"k": "special", "t": "exec"},
                {"k": "imm", "v": 16},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpGroupNonUniformBallot %v4uint_bal", emitted)
        self.assertIn("%c20", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_ff0_i32_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)

    def test_s_ff0_i32_b32_immediate_is_scalar(self):
        for value in (5, 61):
            with self.subTest(value=value):
                g = rdna2spirv.Gen("ps")
                ins = {
                    "op": "s_ff0_i32_b32",
                    "args": [
                        {"k": "sgpr", "r": 0},
                        {"k": "imm", "v": value},
                    ],
                    "mods": [],
                }
                g.insn(ins)
                emitted = "\n".join(g.body)
                self.assertIn(f"OpNot %uint %c{value:x}", emitted)
                self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff0_i32_b32_float_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "fimm", "v": 1.0},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint %c3f80000", emitted)
        self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff0_i32_b32_m0_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "special", "t": "m0"},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpLoad %uint %m0", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff1_i32_b32_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff1_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("FindILsb", emitted)
        self.assertIn("%c5", emitted)
        self.assertNotIn("%cffffffff", emitted)

    def test_s_ff0_i32_b64_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint %c5", emitted)
        self.assertIn("OpNot %uint %c0", emitted)
        self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff1_i32_b64_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff1_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("FindILsb", emitted)
        self.assertIn("OpBitcast %int %c5", emitted)
        self.assertNotIn("OpBitcast %int %cffffffff", emitted)

    def test_s_ff0_i32_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertIn("OpIAdd %uint", emitted)

    def test_s_ff0_s_ff1_b64_negative_inline_and_literal_immediates(self):
        """s_ff0_i32_b64 / s_ff1_i32_b64 sign-extend negative inline constants (-1..-16) and zero-extend literals."""
        import shutil
        import struct
        import subprocess
        import tempfile
        import warnings
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import spirv_mini_interp as si

        stub_dxbc = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
        vsharp = [0, 0, 0xFFFFFFFF, 0x31016FAC]
        cases = [
            # (instruction, expected u32 result)
            ("s_ff0_i32_b64 s10, -1", 0xFFFFFFFF),           # 0xFFFFFFFF_FFFFFFFF -> no 0 bit
            ("s_ff0_i32_b64 s10, -2", 0),                    # 0xFFFFFFFF_FFFFFFFE -> bit 0 is 0
            ("s_ff0_i32_b64 s10, -16", 0),                   # 0xFFFFFFFF_FFFFFFF0 -> bit 0 is 0
            ("s_ff0_i32_b64 s10, 0xffffffff", 32),           # 0x00000000_FFFFFFFF -> bit 32 is first 0
            ("s_ff0_i32_b64 s10, 0", 0),                     # 0x00000000_00000000 -> bit 0 is 0
            ("s_ff0_i32_b64 s10, 0x5", 1),                   # 0b101 -> bit 1 is 0
            ("s_ff1_i32_b64 s10, -1", 0),                    # 0xFFFFFFFF_FFFFFFFF -> bit 0 is 1
            ("s_ff1_i32_b64 s10, -16", 4),                   # 0xFFFFFFFF_FFFFFFF0 -> bit 4 is 1
            ("s_ff1_i32_b64 s10, 0", 0xFFFFFFFF),            # no 1 bit
            ("s_ff1_i32_b64 s10, 0x80000000", 31),           # literal bit 31
            ("s_ff0_i32_b64 s10, null", 0),                  # null -> 0
            ("s_ff1_i32_b64 s10, null", 0xFFFFFFFF),         # null -> 0
        ]
        asm_lines = []
        for idx, (insn, _) in enumerate(cases):
            asm_lines += [
                insn,
                "v_mov_b32_e32 v1, s10",
                f"v_mov_b32_e32 v2, {idx * 4}",
                "buffer_store_dword v1, v2, s[0:3], 0 offen",
            ]
        asm_lines.append("s_endpgm")
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            src = os.path.join(tmp, "disasm", "cs_ff_b64_imm.s")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_ff_b64_imm\n" + "".join(f"\t{ln}\n" for ln in asm_lines))
            with open(os.path.join(tmp, "cs_ff_b64_imm.dxbc"), "wb") as f:
                f.write(stub_dxbc)
            with open(os.path.join(tmp, "fssm", "cs_ff_b64_imm.fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
            spirv_as = shutil.which("spirv-as")
            spirv_val = shutil.which("spirv-val")
            if spirv_as and spirv_val:
                spvasm = os.path.join(tmp, "m.spvasm")
                spvbin = os.path.join(tmp, "m.spv")
                with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                    f.write(spv)
                r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin],
                                    capture_output=True, text=True)
                self.assertEqual(r1.returncode, 0, r1.stderr)
                r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin],
                                    capture_output=True, text=True)
                self.assertEqual(r2.returncode, 0, r2.stderr)
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=vsharp + [0] * 28, memory={},
        )
        for idx, (insn, want) in enumerate(cases):
            with self.subTest(insn=insn):
                self.assertEqual(mem[idx * 4], want)

    def test_s_ff1_i32_b64_high_word_immediate(self):
        """Synthetic 64-bit immediate with only high-word bits set uses hi via src64_data."""
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_ff1_i32_b64",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x0000000100000000}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpBitcast %int %c0", emitted)
        self.assertIn("OpBitcast %int %c1", emitted)


def translate_ps_bitcmp(lines: list[str]) -> str:
    """Translate a hand-written pixel shader listing exporting v0..v3 to mrt0."""
    return translate_lines(
        "ps_salu_bitcmp",
        [*lines, "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
    )


def translate_cs_bitcmp(lines: list[str], threads: int) -> str:
    """Translate a hand-written compute shader listing with 4 user SGPRs s[0:3] and v0 = local_id.x."""
    import struct

    hdr = bytearray(0x148)
    struct.pack_into("<4I", hdr, 0x138, 0x1008, threads, 1, 1)
    return translate_lines(
        "cs_salu_bitcmp",
        [*lines, "s_endpgm"],
        dxbc=STUB_DXBC,
        fssm=bytes(hdr),
    )


class EndToEndBitcmpTranslatorTests(unittest.TestCase):
    """End-to-end rdna2spirv.translate + spirv_mini_interp execution for s_bitcmp0/1_b32/b64."""

    def test_runtime_s_bitcmp0_and_s_bitcmp1_b32_walking_bits_and_scc_transitions(self) -> None:
        # Both operands (s4 = 32-bit value, s5 = bit index) are loaded from buffer memory at runtime
        # so neither is in sgpr_imm, exercising 0->1, 0->0, 1->0, and 1->1 SCC transitions.
        spv = translate_ps_bitcmp([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_bitcmp0_b32 s4, s5",
            "s_cselect_b32 s6, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when bit is 1)
            "s_bitcmp0_b32 s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_bitcmp1_b32 s4, s5",
            "s_cselect_b32 s8, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when bit is 0)
            "s_bitcmp1_b32 s4, s5",
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, s9",
        ])

        cases: list[tuple[int, int]] = []
        for bit_idx in range(32):
            one_hot = 1 << bit_idx
            zero_hot = (~one_hot) & MASK32
            cases.extend([
                (one_hot, bit_idx),
                (one_hot, (bit_idx + 1) & 31),
                (zero_hot, bit_idx),
                (zero_hot, (bit_idx + 1) & 31),
                # Modulo-32 index masking with dirty upper bits [31:5]
                (one_hot, bit_idx | 0x20),
                (one_hot, bit_idx | 0x80000040),
                (zero_hot, bit_idx | 0xFFFFFFE0),
            ])
        for pat in (0x00000000, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x55555555, 0xAAAAAAAA, 0x12345678):
            for idx in (0, 1, 7, 15, 16, 30, 31, 32, 33, 63, 0x80000000, 0xFFFFFFFF):
                cases.append((pat, idx))

        for s0, s1 in cases:
            want0 = ref_bitcmp0_b32(s0, s1)
            want1 = ref_bitcmp1_b32(s0, s1)
            self.assertEqual(want0 + want1, 1)
            mem = {0x1000: s0 & MASK32, 0x1004: s1 & MASK32}
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = tuple(mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem))
                self.assertEqual(got, (want0, want0, want1, want1))

    def test_runtime_s_bitcmp0_and_s_bitcmp1_b64_walking_bits_and_modulo64(self) -> None:
        # 64-bit SGPR pair s[4:5] and 32-bit shift count s6 loaded from buffer memory at runtime,
        # exercising both low dword (bits 0..31) and high dword (bits 32..63) plus modulo-64 masking.
        spv = translate_ps_bitcmp([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",            # seed SCC = 0
            "s_bitcmp0_b64 s[4:5], s6",
            "s_cselect_b32 s8, 1, 0",
            "s_cmp_eq_u32 1, 1",            # seed SCC = 1
            "s_bitcmp0_b64 s[4:5], s6",
            "s_cselect_b32 s9, 1, 0",
            "s_cmp_eq_u32 0, 1",            # seed SCC = 0
            "s_bitcmp1_b64 s[4:5], s6",
            "s_cselect_b32 s10, 1, 0",
            "s_cmp_eq_u32 1, 1",            # seed SCC = 1
            "s_bitcmp1_b64 s[4:5], s6",
            "s_cselect_b32 s11, 1, 0",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
        ])

        cases: list[tuple[int, int]] = []
        for bit_idx in range(64):
            one_hot = 1 << bit_idx
            zero_hot = (~one_hot) & MASK64
            # Also test the corresponding bit in the opposite 32-bit half ((bit_idx + 32) & 63)
            # so any confusion between low and high dwords fails immediately.
            opp_idx = (bit_idx + 32) & 63
            cases.extend([
                (one_hot, bit_idx),
                (one_hot, opp_idx),
                (zero_hot, bit_idx),
                (zero_hot, opp_idx),
                # Modulo-64 index masking with dirty upper bits [31:6]
                (one_hot, bit_idx | 0x40),
                (one_hot, bit_idx | 0x80000080),
                (zero_hot, bit_idx | 0xFFFFFFC0),
            ])
        for pat64 in (
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
            0x80000000_00000001,
            0x55555555_AAAAAAAA,
            0x12345678_9ABCDEF0,
        ):
            for idx in (0, 1, 31, 32, 33, 45, 62, 63, 64, 65, 95, 96, 127, 0x80000020, 0xFFFFFFFF):
                cases.append((pat64, idx))

        for s0_64, s1 in cases:
            want0 = ref_bitcmp0_b64(s0_64, s1)
            want1 = ref_bitcmp1_b64(s0_64, s1)
            self.assertEqual(want0 + want1, 1)
            mem = {
                0x1000: s0_64 & MASK32,
                0x1004: (s0_64 >> 32) & MASK32,
                0x1008: s1 & MASK32,
                0x100C: 0,
            }
            with self.subTest(s0_64=hex(s0_64), s1=hex(s1)):
                got = tuple(mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem))
                self.assertEqual(got, (want0, want0, want1, want1))

    def test_s_bitcmp_immediates_m0_and_null_operands(self) -> None:
        # 64-bit inline negative constants (-1..-16) sign-extend across all 64 bits, while
        # 32-bit hex literals zero-extend their high 32 bits; m0 and null work as S0 or S1.
        spv1 = translate_ps_bitcmp([
            "s_bitcmp1_b64 -1, 45",            # -1 -> 0xFFFFFFFF_FFFFFFFF: bit 45 is 1 -> SCC = 1
            "s_cselect_b32 s0, 1, 0",
            "s_bitcmp1_b64 0xffffffff, 31",    # 32-bit literal -> 0x00000000_FFFFFFFF: bit 31 is 1 -> SCC = 1
            "s_cselect_b32 s1, 1, 0",
            "s_bitcmp1_b64 0xffffffff, 32",    # 32-bit literal -> 0x00000000_FFFFFFFF: bit 32 is 0 -> SCC = 0
            "s_cselect_b32 s2, 1, 0",
            "s_bitcmp0_b64 0xffffffff, 45",    # high half of 32-bit literal is 0 -> s_bitcmp0 gives SCC = 1
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual(tuple(mrt0_words(spv1)), (1, 1, 0, 1))

        # Inline -16 (0xFFFFFFFF_FFFFFFF0): bits 0..3 are 0, bits 4..63 are 1; plus m0 and null operands.
        spv2 = translate_ps_bitcmp([
            "s_mov_b32 m0, 0x00080000",        # bit 19 set in m0
            "s_bitcmp0_b64 -16, 3",            # bit 3 of -16 is 0 -> SCC = 1
            "s_cselect_b32 s0, 1, 0",
            "s_bitcmp1_b64 -16, 36",           # bit 36 of -16 is 1 -> SCC = 1
            "s_cselect_b32 s1, 1, 0",
            "s_bitcmp1_b32 m0, 19",            # bit 19 of m0 is 1 -> SCC = 1
            "s_cselect_b32 s2, 1, 0",
            "s_bitcmp0_b64 null, m0",          # null is 0 at any bit index -> s_bitcmp0 is 1
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual(tuple(mrt0_words(spv2)), (1, 1, 1, 1))

    def test_s_bitcmp_wave64_and_wave32_lane_masks(self) -> None:
        # Wave64: lane 45 sets bit 45 of 64-bit vcc (which is bit 13 of vcc_hi, while vcc_lo is 0).
        spv64 = translate_cs_bitcmp([
            "v_cmp_eq_u32_e32 vcc, 45, v0",
            "s_bitcmp1_b64 vcc, 45",           # bit 45 of 64-bit vcc is 1 -> SCC = 1
            "s_cselect_b32 s4, 1, 0",
            "s_bitcmp1_b64 vcc, 13",           # bit 13 of 64-bit vcc (low half) is 0 -> SCC = 0
            "s_cselect_b32 s5, 1, 0",
            "s_bitcmp1_b32 vcc_hi, 13",        # bit 13 of vcc_hi (lane 45) is 1 -> SCC = 1
            "s_cselect_b32 s6, 1, 0",
            "s_bitcmp0_b32 vcc_lo, 13",        # bit 13 of vcc_lo is 0 -> s_bitcmp0 sets SCC = 1
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v1, s4",
            "v_mov_b32 v2, s5",
            "v_mov_b32 v3, s6",
            "v_mov_b32 v4, s7",
            "v_lshlrev_b32 v5, 4, v0",
            "buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen",
        ], threads=64)
        mem64: dict[int, int] = {}
        si.run_workgroup(
            spv64,
            64,
            64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
            memory=mem64,
        )
        self.assertEqual((mem64[0], mem64[4], mem64[8], mem64[12]), (1, 0, 1, 1))

        # Wave32: lane 19 sets bit 19 of lane-mask SGPR pair s[4:5]; high dword (bits 32..63) is 0.
        spv32 = translate_cs_bitcmp([
            "v_cmp_eq_u32_e64 s[4:5], 19, v0",
            "s_bitcmp1_b32 s4, 19",            # bit 19 of s4 is 1 -> SCC = 1
            "s_cselect_b32 s6, 1, 0",
            "s_bitcmp0_b32 s4, 19",            # bit 19 of s4 is 1 -> s_bitcmp0 sets SCC = 0
            "s_cselect_b32 s7, 1, 0",
            "s_bitcmp1_b64 s[4:5], 19",        # bit 19 of 64-bit mask s[4:5] is 1 -> SCC = 1
            "s_cselect_b32 s8, 1, 0",
            "s_bitcmp1_b64 s[4:5], 51",        # bit 51 (high word of 32-wide wave) is 0 -> SCC = 0
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v1, s6",
            "v_mov_b32 v2, s7",
            "v_mov_b32 v3, s8",
            "v_mov_b32 v4, s9",
            "v_lshlrev_b32 v5, 4, v0",
            "buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen",
        ], threads=32)
        mem32: dict[int, int] = {}
        si.run_workgroup(
            spv32,
            32,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
            memory=mem32,
        )
        self.assertEqual((mem32[0], mem32[4], mem32[8], mem32[12]), (1, 0, 1, 0))


class TranslatedBitcmpMutationTests(unittest.TestCase):
    """Negative controls verifying that sabotaging emitted s_bitcmp SPIR-V fails execution."""

    def test_translated_bitcmp_spirv_mutations_caught(self) -> None:
        # 1. s_bitcmp1_b32: flipping OpINotEqual -> OpIEqual, breaking the & 31 mask, or
        #    redirecting OpStore %scc must fail execution.
        spv32 = translate_ps_bitcmp([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",
            "s_bitcmp1_b32 s4, s5",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        mem32 = {0x1000: 1 << 19, 0x1004: 19}
        self.assertEqual(mrt0_words(spv32, push_constants=PUSH_VSHARP, memory=mem32)[0], 1)

        bad_cmp = spv32.replace("OpINotEqual %bool", "OpIEqual %bool", 1)
        self.assertNotEqual(bad_cmp, spv32)
        self.assertNotEqual(mrt0_words(bad_cmp, push_constants=PUSH_VSHARP, memory=mem32)[0], 1)

        bad_shift = spv32.replace("OpShiftRightLogical %uint", "OpShiftLeftLogical %uint", 1)
        self.assertNotEqual(bad_shift, spv32)
        self.assertNotEqual(mrt0_words(bad_shift, push_constants=PUSH_VSHARP, memory=mem32)[0], 1)

        bad_scc = spv32.replace("OpStore %scc", "OpStore %vcc_hi")
        self.assertNotEqual(bad_scc, spv32)
        self.assertNotEqual(mrt0_words(bad_scc, push_constants=PUSH_VSHARP, memory=mem32)[0], 1)

        # 2. s_bitcmp1_b64: replacing OpUGreaterThanEqual %bool (the >= 32 high-dword select)
        #    with OpULessThan %bool swaps the low and high dwords and must fail.
        spv64 = translate_ps_bitcmp([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",
            "s_bitcmp1_b64 s[4:5], s6",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        mem64 = {0x1000: 0, 0x1004: 1 << (45 - 32), 0x1008: 45, 0x100C: 0}
        self.assertEqual(mrt0_words(spv64, push_constants=PUSH_VSHARP, memory=mem64)[0], 1)

        bad_hi_sel = spv64.replace("OpUGreaterThanEqual %bool", "OpULessThan %bool", 1)
        self.assertNotEqual(bad_hi_sel, spv64)
        self.assertNotEqual(mrt0_words(bad_hi_sel, push_constants=PUSH_VSHARP, memory=mem64)[0], 1)


if __name__ == "__main__":
    unittest.main()
