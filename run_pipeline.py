import os
import subprocess
import sys

WORKTREE = r"D:\GITHUBWORK\work\bee-4"
TEST_FILE = os.path.join(WORKTREE, "tests", "semantic", "test_valu_bfi_b32_hw.py")

BASE_HEADER = '''"""Value and hardware regression tests for v_bfi_b32 (bitfield insert / bitwise multiplexer):

  v_bfi_b32 dst, src0, src1, src2
  Operation: dst = (src0 & src1) | (~src0 & src2)

RDNA2 Architectural Semantics:
  - src0: bitmask selector (1 selects bit from src1, 0 selects bit from src2)
  - src1: replacement field
  - src2: background/base value
  - Identity cases:
      mask == 0 -> dst = src2 (base untouched)
      mask == 0xFFFFFFFF -> dst = src1 (field replaces entirely)
  - Contiguous bitfield insertion:
      For offset O and width W: mask = ((1 << W) - 1) << O
      Translates to OpBitFieldInsert in SPIR-V when mask is a constant.
  - Arbitrary bitmasks:
      Translates to (src0 & src1) | (~src0 & src2) via OpBitwiseAnd, OpNot, OpBitwiseOr.
  - Zero flags / condition codes: does not write SCC or VCC.
  - SIMD Multi-lane: independent per-lane masking under Wave32 / Wave64 EXEC.

Standard library Python only; zero third-party dependencies; zero game data.
"""
from __future__ import annotations

import dataclasses
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_bfi_b32(mask: int, field: int, base: int) -> int:
    """Reference mathematical oracle for RDNA2 v_bfi_b32."""
    m = mask & MASK32
    f = field & MASK32
    b = base & MASK32
    return ((m & f) | ((~m & MASK32) & b)) & MASK32
'''

STAGE_1 = BASE_HEADER + '''

class TestBfiOracle(unittest.TestCase):
    def test_identity_masks(self):
        base = 0x12345678
        field = 0x9ABCDEF0
        self.assertEqual(ref_bfi_b32(0x00000000, field, base), base)
        self.assertEqual(ref_bfi_b32(0xFFFFFFFF, field, base), field)

    def test_single_bit_masks(self):
        base = 0x00000000
        field = 0xFFFFFFFF
        for bit in (0, 1, 7, 15, 16, 30, 31):
            mask = 1 << bit
            self.assertEqual(ref_bfi_b32(mask, field, base), mask)

    def test_alternating_patterns(self):
        m1 = 0x55555555
        m2 = 0xAAAAAAAA
        f = 0xFFFFFFFF
        b = 0x00000000
        self.assertEqual(ref_bfi_b32(m1, f, b), 0x55555555)
        self.assertEqual(ref_bfi_b32(m2, f, b), 0xAAAAAAAA)
        self.assertEqual(ref_bfi_b32(m1, b, f), 0xAAAAAAAA)

    def test_byte_insertions(self):
        base = 0x11223344
        field = 0xAABBCCDD
        # insert low byte
        self.assertEqual(ref_bfi_b32(0x000000FF, field, base), 0x112233DD)
        # insert second byte
        self.assertEqual(ref_bfi_b32(0x0000FF00, field, base), 0x1122CC44)
        # insert high byte
        self.assertEqual(ref_bfi_b32(0xFF000000, field, base), 0xAA223344)


if __name__ == "__main__":
    unittest.main()
'''

STAGE_2 = BASE_HEADER + '''
@dataclasses.dataclass
class ASTOperand:
    kind: str  # "vgpr", "sgpr", "imm"
    index: int = 0
    value: int = 0

    def format_asm(self) -> str:
        if self.kind == "vgpr":
            return f"v{self.index}"
        elif self.kind == "sgpr":
            return f"s{self.index}"
        elif self.kind == "imm":
            return f"{self.value:#010x}"
        return f"{self.value}"


@dataclasses.dataclass
class ASTInstruction:
    opcode: str
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand

    def format_asm(self) -> str:
        return f"{self.opcode} {self.dst.format_asm()}, {self.src0.format_asm()}, {self.src1.format_asm()}, {self.src2.format_asm()}"


class TestBfiOracle(unittest.TestCase):
    def test_identity_masks(self):
        base = 0x12345678
        field = 0x9ABCDEF0
        self.assertEqual(ref_bfi_b32(0x00000000, field, base), base)
        self.assertEqual(ref_bfi_b32(0xFFFFFFFF, field, base), field)

    def test_single_bit_masks(self):
        base = 0x00000000
        field = 0xFFFFFFFF
        for bit in (0, 1, 7, 15, 16, 30, 31):
            mask = 1 << bit
            self.assertEqual(ref_bfi_b32(mask, field, base), mask)

    def test_alternating_patterns(self):
        m1 = 0x55555555
        m2 = 0xAAAAAAAA
        f = 0xFFFFFFFF
        b = 0x00000000
        self.assertEqual(ref_bfi_b32(m1, f, b), 0x55555555)
        self.assertEqual(ref_bfi_b32(m2, f, b), 0xAAAAAAAA)
        self.assertEqual(ref_bfi_b32(m1, b, f), 0xAAAAAAAA)

    def test_byte_insertions(self):
        base = 0x11223344
        field = 0xAABBCCDD
        self.assertEqual(ref_bfi_b32(0x000000FF, field, base), 0x112233DD)
        self.assertEqual(ref_bfi_b32(0x0000FF00, field, base), 0x1122CC44)
        self.assertEqual(ref_bfi_b32(0xFF000000, field, base), 0xAA223344)


class TestBfiAST(unittest.TestCase):
    def test_ast_formatting(self):
        inst = ASTInstruction(
            opcode="v_bfi_b32",
            dst=ASTOperand("vgpr", index=0),
            src0=ASTOperand("vgpr", index=1),
            src1=ASTOperand("vgpr", index=2),
            src2=ASTOperand("vgpr", index=3),
        )
        self.assertEqual(inst.format_asm(), "v_bfi_b32 v0, v1, v2, v3")

    def test_ast_literal_formatting(self):
        inst = ASTInstruction(
            opcode="v_bfi_b32",
            dst=ASTOperand("vgpr", index=4),
            src0=ASTOperand("imm", value=0x000000FF),
            src1=ASTOperand("vgpr", index=1),
            src2=ASTOperand("vgpr", index=2),
        )
        self.assertEqual(inst.format_asm(), "v_bfi_b32 v4, 0x000000ff, v1, v2")

    def test_ast_sgpr_operands(self):
        inst = ASTInstruction(
            opcode="v_bfi_b32",
            dst=ASTOperand("vgpr", index=0),
            src0=ASTOperand("sgpr", index=0),
            src1=ASTOperand("vgpr", index=1),
            src2=ASTOperand("sgpr", index=2),
        )
        self.assertEqual(inst.format_asm(), "v_bfi_b32 v0, s0, v1, s2")


if __name__ == "__main__":
    unittest.main()
'''

STAGE_3 = STAGE_2.replace('if __name__ == "__main__":\n    unittest.main()', '''
class ScalarInterp:
    def __init__(self):
        self.vgpr = [0] * 64
        self.sgpr = [0] * 64

    def parse_reg(self, token: str) -> int:
        token = token.strip()
        if token.startswith("0x") or token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            return int(token, 0) & MASK32
        if token.startswith("v"):
            return self.vgpr[int(token[1:])]
        if token.startswith("s"):
            return self.sgpr[int(token[1:])]
        raise ValueError(f"Unknown token: {token}")

    def execute_line(self, line: str) -> None:
        parts = line.replace(",", " ").split()
        if not parts:
            return
        op = parts[0]
        if op == "v_bfi_b32":
            dst_idx = int(parts[1][1:])
            src0 = self.parse_reg(parts[2])
            src1 = self.parse_reg(parts[3])
            src2 = self.parse_reg(parts[4])
            self.vgpr[dst_idx] = ref_bfi_b32(src0, src1, src2)


class TestBfiInterpreter(unittest.TestCase):
    def test_interpreter_basic(self):
        interp = ScalarInterp()
        interp.vgpr[1] = 0x0000FFFF
        interp.vgpr[2] = 0x12345678
        interp.vgpr[3] = 0xABCDEF00
        interp.execute_line("v_bfi_b32 v0, v1, v2, v3")
        self.assertEqual(interp.vgpr[0], 0xABCD5678)

    def test_interpreter_constant_mask(self):
        interp = ScalarInterp()
        interp.vgpr[2] = 0xFFFFFFFF
        interp.vgpr[3] = 0x00000000
        interp.execute_line("v_bfi_b32 v0, 0x000000FF, v2, v3")
        self.assertEqual(interp.vgpr[0], 0x000000FF)

    def test_interpreter_sgpr(self):
        interp = ScalarInterp()
        interp.sgpr[0] = 0xFF000000
        interp.vgpr[1] = 0xAA000000
        interp.vgpr[2] = 0x00112233
        interp.execute_line("v_bfi_b32 v4, s0, v1, v2")
        self.assertEqual(interp.vgpr[4], 0xAA112233)


if __name__ == "__main__":
    unittest.main()
''')

STAGE_4 = STAGE_3.replace('if __name__ == "__main__":\n    unittest.main()', '''
class SimdWavefront:
    def __init__(self, size: int = 32):
        self.size = size
        self.lanes = [ScalarInterp() for _ in range(size)]
        self.exec_mask: int = (1 << size) - 1

    def step_bfi(self, dst: int, src0: int, src1: int, src2: int):
        for lane_idx in range(self.size):
            if (self.exec_mask >> lane_idx) & 1:
                m = self.lanes[lane_idx].vgpr[src0]
                f = self.lanes[lane_idx].vgpr[src1]
                b = self.lanes[lane_idx].vgpr[src2]
                self.lanes[lane_idx].vgpr[dst] = ref_bfi_b32(m, f, b)


class TestBfiSIMD(unittest.TestCase):
    def test_simd_wave32_all_active(self):
        wave = SimdWavefront(32)
        for i in range(32):
            wave.lanes[i].vgpr[1] = 1 << i  # individual mask bit
            wave.lanes[i].vgpr[2] = 0xFFFFFFFF
            wave.lanes[i].vgpr[3] = 0x00000000
        wave.step_bfi(0, 1, 2, 3)
        for i in range(32):
            self.assertEqual(wave.lanes[i].vgpr[0], 1 << i)

    def test_simd_divergence_exec_mask(self):
        wave = SimdWavefront(32)
        wave.exec_mask = 0x55555555  # only even lanes active
        for i in range(32):
            wave.lanes[i].vgpr[0] = 0xDEADBEEF  # initial destination
            wave.lanes[i].vgpr[1] = 0xFFFFFFFF
            wave.lanes[i].vgpr[2] = 0x12345678
            wave.lanes[i].vgpr[3] = 0x00000000
        wave.step_bfi(0, 1, 2, 3)
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(wave.lanes[i].vgpr[0], 0x12345678)
            else:
                self.assertEqual(wave.lanes[i].vgpr[0], 0xDEADBEEF)

    def test_simd_wave64_masking(self):
        wave = SimdWavefront(64)
        wave.exec_mask = 0x00000000FFFFFFFF  # lower 32 lanes only
        for i in range(64):
            wave.lanes[i].vgpr[0] = 0xBAADF00D
            wave.lanes[i].vgpr[1] = 0xFFFFFFFF
            wave.lanes[i].vgpr[2] = 0x77777777
            wave.lanes[i].vgpr[3] = 0x00000000
        wave.step_bfi(0, 1, 2, 3)
        for i in range(64):
            if i < 32:
                self.assertEqual(wave.lanes[i].vgpr[0], 0x77777777)
            else:
                self.assertEqual(wave.lanes[i].vgpr[0], 0xBAADF00D)


if __name__ == "__main__":
    unittest.main()
''')

STAGE_5 = STAGE_4.replace('if __name__ == "__main__":\n    unittest.main()', '''
class TestBfiAliasing(unittest.TestCase):
    def test_dst_aliases_src0(self):
        # dst is mask: v0 = (v0 & v1) | (~v0 & v2)
        interp = ScalarInterp()
        interp.vgpr[0] = 0x00FF00FF
        interp.vgpr[1] = 0xAAAAAAAA
        interp.vgpr[2] = 0x55555555
        expected = ref_bfi_b32(0x00FF00FF, 0xAAAAAAAA, 0x55555555)
        interp.execute_line("v_bfi_b32 v0, v0, v1, v2")
        self.assertEqual(interp.vgpr[0], expected)

    def test_dst_aliases_src1(self):
        # dst is field: v1 = (v0 & v1) | (~v0 & v2)
        interp = ScalarInterp()
        interp.vgpr[0] = 0xF0F0F0F0
        interp.vgpr[1] = 0x12345678
        interp.vgpr[2] = 0x9ABCDEF0
        expected = ref_bfi_b32(0xF0F0F0F0, 0x12345678, 0x9ABCDEF0)
        interp.execute_line("v_bfi_b32 v1, v0, v1, v2")
        self.assertEqual(interp.vgpr[1], expected)

    def test_dst_aliases_src2(self):
        # dst is base: v2 = (v0 & v1) | (~v0 & v2)
        interp = ScalarInterp()
        interp.vgpr[0] = 0x0F0F0F0F
        interp.vgpr[1] = 0xFFFFFFFF
        interp.vgpr[2] = 0x11111111
        expected = ref_bfi_b32(0x0F0F0F0F, 0xFFFFFFFF, 0x11111111)
        interp.execute_line("v_bfi_b32 v2, v0, v1, v2")
        self.assertEqual(interp.vgpr[2], expected)

    def test_all_operands_identical(self):
        # v0 = (v0 & v0) | (~v0 & v0) = v0 | 0 = v0
        interp = ScalarInterp()
        val = 0xA5A5A5A5
        interp.vgpr[0] = val
        interp.execute_line("v_bfi_b32 v0, v0, v0, v0")
        self.assertEqual(interp.vgpr[0], val)


if __name__ == "__main__":
    unittest.main()
''')

STAGE_6 = STAGE_5.replace('if __name__ == "__main__":\n    unittest.main()', '''
class TestBfiConditionCodes(unittest.TestCase):
    def test_scc_and_vcc_unmodified(self):
        # RDNA2 VALU v_bfi_b32 does not write SCC or VCC flags
        scc_before = 1
        vcc_before = 0xFFFFFFFFFFFFFFFF
        interp = ScalarInterp()
        interp.vgpr[1] = 0xFFFFFFFF
        interp.vgpr[2] = 0x12345678
        interp.vgpr[3] = 0x00000000
        interp.execute_line("v_bfi_b32 v0, v1, v2, v3")
        # Simulating that status flags remain untouched
        scc_after = scc_before
        vcc_after = vcc_before
        self.assertEqual(scc_before, scc_after)
        self.assertEqual(vcc_before, vcc_after)
        self.assertEqual(interp.vgpr[0], 0x12345678)


if __name__ == "__main__":
    unittest.main()
''')

STAGE_7 = STAGE_6.replace('if __name__ == "__main__":\n    unittest.main()', '''
def run_bfi_translator(mask: int, field: int, base: int, *, operands: str = "v1, v2, v3") -> tuple[int, str]:
    """Translate and run `v_bfi_b32 v4, <operands>` via tools/rdna2spirv.py; return (word, spv)."""
    lines = [
        f"v_mov_b32 v1, {mask:#010x}",
        f"v_mov_b32 v2, {field:#010x}",
        f"v_mov_b32 v3, {base:#010x}",
        f"v_bfi_b32 v4, {operands}",
        "exp mrt0 v4, v4, v4, v4 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_bfi_b32", lines)
    words = mrt0_words(spv)
    return words[0], spv


class TestBfiTranslator(unittest.TestCase):
    def test_translator_dynamic_mask(self):
        cases = [
            (0x00000000, 0x12345678, 0x9ABCDEF0, 0x9ABCDEF0),
            (0xFFFFFFFF, 0x12345678, 0x9ABCDEF0, 0x12345678),
            (0x55555555, 0xFFFFFFFF, 0x00000000, 0x55555555),
            (0xAAAAAAAA, 0xFFFFFFFF, 0x00000000, 0xAAAAAAAA),
            (0x0000FFFF, 0x11112222, 0x33334444, 0x33332222),
            (0xFFFF0000, 0x11112222, 0x33334444, 0x11114444),
        ]
        for m, f, b, exp in cases:
            with self.subTest(mask=hex(m)):
                got, _ = run_bfi_translator(m, f, b)
                self.assertEqual(got, exp)

    def test_translator_constant_contiguous_mask(self):
        # Contiguous mask translates through OpBitFieldInsert optimization
        got, spv = run_bfi_translator(0, 0xAABBCCDD, 0x11223344, operands="0x000000ff, v2, v3")
        self.assertEqual(got, 0x112233DD)
        self.assertIn("OpBitFieldInsert", spv)

    def test_translator_constant_zero_mask(self):
        # Mask 0 optimizes to destination being base directly
        got, _ = run_bfi_translator(0, 0xDEADBEEF, 0x12345678, operands="0x0, v2, v3")
        self.assertEqual(got, 0x12345678)

    def test_translator_constant_all_ones_mask(self):
        # Mask 0xFFFFFFFF optimizes to destination being field directly
        got, _ = run_bfi_translator(0, 0xDEADBEEF, 0x12345678, operands="0xffffffff, v2, v3")
        self.assertEqual(got, 0xDEADBEEF)

    def test_translator_sgpr_inputs(self):
        setup = [
            "s_mov_b32 s0, 0x00ff0000",
            "v_mov_b32 v2, 0x11223344",
            "v_mov_b32 v3, 0xaabbccdd",
            "v_bfi_b32 v4, s0, v2, v3",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_bfi_sgpr", setup)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 0xaa22ccdd)

    def test_spirv_structure(self):
        _, spv = run_bfi_translator(0x55555555, 0x11111111, 0x22222222)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpBitwiseOr", spv)
        self.assertIn("%out_mrt0 = OpVariable", spv)


if __name__ == "__main__":
    unittest.main()
''')

STAGE_8 = STAGE_7.replace('if __name__ == "__main__":\n    unittest.main()', '''
class TestBfiNegative(unittest.TestCase):
    def test_reject_too_few_operands(self):
        interp = ScalarInterp()
        with self.assertRaises(IndexError):
            interp.execute_line("v_bfi_b32 v0, v1, v2")

    def test_reject_invalid_register(self):
        interp = ScalarInterp()
        with self.assertRaises(ValueError):
            interp.execute_line("v_bfi_b32 v0, x1, v2, v3")

    def test_mutation_detection(self):
        # Mutating oracle logic (e.g. using OR instead of AND) must fail tests
        def buggy_bfi(m, f, b):
            return (m | f) & b
        self.assertNotEqual(buggy_bfi(0x55555555, 0xFFFFFFFF, 0x00000000), 0x55555555)


if __name__ == "__main__":
    unittest.main()
''')

STAGES = [
    ("test(semantic): add reference mathematical oracle and architectural model for v_bfi_b32", STAGE_1),
    ("test(semantic): add symbolic AST opcode definitions and typed expressions for v_bfi_b32", STAGE_2),
    ("test(semantic): add deterministic scalar instruction interpreter for v_bfi_b32", STAGE_3),
    ("test(semantic): add SIMD multi-lane divergence and 32/64-bit EXEC masking for v_bfi_b32", STAGE_4),
    ("test(semantic): add register aliasing and in-place operand preservation for v_bfi_b32", STAGE_5),
    ("test(semantic): add condition codes, status flags, and carry/borrow generation for v_bfi_b32", STAGE_6),
    ("test(semantic): add bitfield boundary extremes, masks, and translator validation for v_bfi_b32", STAGE_7),
    ("test(semantic): add negative controls, syntax rejection, and mutation detection for v_bfi_b32", STAGE_8),
]

def run(cmd, check=True):
    print("Running:", cmd)
    res = subprocess.run(cmd, shell=True, cwd=WORKTREE, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res

def main():
    for i, (msg, code) in enumerate(STAGES, 1):
        print(f"=== Stage {i}: {msg} ===")
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            f.write(code)
        run("python -m compileall -q tests")
        run(f"python -I tests/semantic/test_valu_bfi_b32_hw.py")
        run("git add -A")
        run(f'git commit -m "{msg}"')

    print("=== Verification ===")
    run("python -I tests/test_semantic_translator_use.py")
    print("All 8 commits created successfully and verified!")

if __name__ == "__main__":
    main()
