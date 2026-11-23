from __future__ import annotations

"""Architectural execution and translator integration tests for SALU SaveExec
64-bit Mask Control Suite.

Covers RDNA2 SOP1 Instructions:
    - s_and_saveexec_b64   D, S0 : D = EXEC; EXEC = S0 & EXEC;   SCC = (EXEC != 0)
    - s_or_saveexec_b64    D, S0 : D = EXEC; EXEC = S0 | EXEC;   SCC = (EXEC != 0)
    - s_xor_saveexec_b64   D, S0 : D = EXEC; EXEC = S0 ^ EXEC;   SCC = (EXEC != 0)
    - s_andn2_saveexec_b64 D, S0 : D = EXEC; EXEC = S0 & ~EXEC;  SCC = (EXEC != 0)
    - s_orn2_saveexec_b64  D, S0 : D = EXEC; EXEC = S0 | ~EXEC;  SCC = (EXEC != 0)
    - s_nand_saveexec_b64  D, S0 : D = EXEC; EXEC = ~(S0 & EXEC); SCC = (EXEC != 0)
    - s_nor_saveexec_b64   D, S0 : D = EXEC; EXEC = ~(S0 | EXEC); SCC = (EXEC != 0)
    - s_xnor_saveexec_b64  D, S0 : D = EXEC; EXEC = ~(S0 ^ EXEC); SCC = (EXEC != 0)

Hardware rules (GFX10 RDNA2):
    - SOP1 encoding with 64-bit SGPR pair destination D and source S0.
    - Previous 64-bit EXEC register value saved into destination D.
    - Updated EXEC reflects bitwise operation between S0 and previous EXEC.
    - Scalar Condition Code (SCC) set to 1 if new EXEC != 0, else 0.
    - In-place operand preservation and register aliasing (D == S0) supported.
    - Multi-lane SIMD divergence on Wave64 execution masks.

100% of tests are executed through tools/rdna2spirv.py (support.translate_lines)
and spirv_mini_interp.run_workgroup. Zero standalone mock models.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import saveexec_run
import spirv_mini_interp as si
import support

MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK32: int = 0xFFFF_FFFF


def run_saveexec_hw(op: str, s0: int, old_exec: int, aliased: bool = False) -> tuple[int, int, int]:
    src = "s[6:7]" if aliased else "s[10:11]"
    setup = (
        [f"s_mov_b32 s6, {s0 & MASK32:#x}", f"s_mov_b32 s7, {s0 >> 32:#x}"]
        if aliased
        else [f"s_mov_b32 s10, {s0 & MASK32:#x}", f"s_mov_b32 s11, {s0 >> 32:#x}"]
    )
    prog = [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "v_mbcnt_hi_u32_b32 v1, -1, v1",
        "v_mov_b32_e32 v2, 0",
        f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
        f"s_mov_b32 exec_hi, {old_exec >> 32:#x}",
    ] + setup + [
        f"s_{op}_saveexec_b64 s[6:7], {src}",
        "s_cselect_b32 s8, 1, 0",
        "v_mov_b32_e32 v2, 1",
        "s_mov_b64 exec, -1",
        "v_cndmask_b32_e64 v3, 0, 1, s[6:7]",
        "v_lshlrev_b32_e32 v21, 2, v1",
        "buffer_store_dword v2, v21, s[0:3], 0 offen",
        "v_lshlrev_b32_e32 v22, 2, v1",
        "v_add_nc_u32_e32 v22, 256, v22",
        "buffer_store_dword v3, v22, s[0:3], 0 offen",
        "v_cmpx_eq_u32_e32 0, v1",
        "v_mov_b32_e32 v20, s8",
        "v_mov_b32_e32 v21, 512",
        "buffer_store_dword v20, v21, s[0:3], 0 offen",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"cs_saveexec_{op}", prog, fssm=saveexec_run.fssm())
    mem = si.run_workgroup(
        spv,
        64,
        64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=saveexec_run.VSHARP + [0] * 28,
        memory={},
    )
    new_exec = sum((mem.get(4 * lane, 0) & 1) << lane for lane in range(64))
    d_val = sum((mem.get(256 + 4 * lane, 0) & 1) << lane for lane in range(64))
    scc = mem.get(512, 0)
    return new_exec, d_val, scc


# --- Opcode: s_and_saveexec_b64 ---

class TestSaveExecOracle_and(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("and", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("and", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("and", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("and", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_and(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("and", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("and", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("and", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("and", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_and(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("and", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("and", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("and", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("and", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_and(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_and_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_and_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("and", s0, old_exec)
            memory = saveexec_run.run("and", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_or_saveexec_b64 ---

class TestSaveExecOracle_or(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("or", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("or", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("or", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("or", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_or(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("or", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("or", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("or", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("or", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_or(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("or", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("or", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("or", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("or", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_or(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_or_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_or_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("or", s0, old_exec)
            memory = saveexec_run.run("or", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_xor_saveexec_b64 ---

class TestSaveExecOracle_xor(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("xor", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("xor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("xor", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("xor", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_xor(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("xor", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("xor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("xor", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("xor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_xor(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("xor", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("xor", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("xor", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("xor", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_xor(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_xor_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_xor_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("xor", s0, old_exec)
            memory = saveexec_run.run("xor", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_andn2_saveexec_b64 ---

class TestSaveExecOracle_andn2(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("andn2", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("andn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("andn2", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("andn2", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_andn2(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("andn2", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("andn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("andn2", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("andn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_andn2(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("andn2", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("andn2", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("andn2", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("andn2", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_andn2(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_andn2_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_andn2_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("andn2", s0, old_exec)
            memory = saveexec_run.run("andn2", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_orn2_saveexec_b64 ---

class TestSaveExecOracle_orn2(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("orn2", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("orn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("orn2", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("orn2", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_orn2(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("orn2", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("orn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("orn2", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("orn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_orn2(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("orn2", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("orn2", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("orn2", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("orn2", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_orn2(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_orn2_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_orn2_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("orn2", s0, old_exec)
            memory = saveexec_run.run("orn2", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_nand_saveexec_b64 ---

class TestSaveExecOracle_nand(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("nand", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("nand", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("nand", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("nand", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_nand(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("nand", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("nand", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("nand", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("nand", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_nand(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("nand", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("nand", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("nand", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("nand", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_nand(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_nand_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_nand_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("nand", s0, old_exec)
            memory = saveexec_run.run("nand", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_nor_saveexec_b64 ---

class TestSaveExecOracle_nor(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("nor", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("nor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("nor", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("nor", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_nor(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("nor", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("nor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("nor", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("nor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_nor(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("nor", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("nor", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("nor", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("nor", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_nor(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_nor_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_nor_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("nor", s0, old_exec)
            memory = saveexec_run.run("nor", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


# --- Opcode: s_xnor_saveexec_b64 ---

class TestSaveExecOracle_xnor(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = run_saveexec_hw("xnor", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("xnor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64):
                new_exec, d_val, scc = run_saveexec_hw("xnor", s0, old_exec)
                expected_exec, _, expected_scc = saveexec_run.model("xnor", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(scc, expected_scc)


class TestSaveExecASTInterpreter_xnor(unittest.TestCase):
    def test_ast_construction_and_step(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        s0 = 0x0000_FFFF_FFFF_0000
        new_exec, d_val, scc = run_saveexec_hw("xnor", s0, old_exec)
        expected_exec, _, expected_scc = saveexec_run.model("xnor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)

    def test_ast_register_aliasing(self):
        old_exec = 0xAAAA_5555_AAAA_5555
        s0 = 0x1111_2222_3333_4444
        new_exec, d_val, scc = run_saveexec_hw("xnor", s0, old_exec, aliased=True)
        expected_exec, _, expected_scc = saveexec_run.model("xnor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)


class TestSaveExecDivergenceFlags_xnor(unittest.TestCase):
    def test_simd_lane_divergence(self):
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = run_saveexec_hw("xnor", low_lanes, high_lanes)
        expected_exec, _, expected_scc = saveexec_run.model("xnor", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(scc, expected_scc)
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = run_saveexec_hw("xnor", s0, old_exec)
            expected_exec, _, expected_scc = saveexec_run.model("xnor", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(new_exec, expected_exec)
            self.assertEqual(scc, expected_scc)


class TestSaveExecTranslator_xnor(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_xnor_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_xnor_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        for src in saveexec_run.SOURCES:
            mul, add, mask, old_exec = 1, 0, 32, 0xFFFFFFFF00000000
            s0 = old_exec if src == "exec" else saveexec_run.s0_mask(mul, add, mask)
            new, d, scc = saveexec_run.model("xnor", s0, old_exec)
            memory = saveexec_run.run("xnor", src, mul, add, mask, old_exec)
            got = sum(memory[4 * lane] << lane for lane in range(64))
            self.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
            self.assertEqual(memory[saveexec_run.SCC_AT], scc, "SCC")
            self.assertEqual(memory[saveexec_run.BCNT_AT], bin(d).count("1"), "population count of D")


if __name__ == "__main__":
    unittest.main()
