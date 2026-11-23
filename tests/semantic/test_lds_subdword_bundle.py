from __future__ import annotations

"""Semantic translator verification suite for RDNA2 LDS Sub-Dword Memory Suite.

Covers 6 opcodes with 100% translator execution via tools/rdna2spirv.py and spirv_mini_interp:
- ds_read_u8: 8-bit unsigned byte read from LDS, zero-extended to 32 bits.
- ds_read_i8: 8-bit signed byte read from LDS, sign-extended to 32 bits.
- ds_read_u16: 16-bit unsigned halfword read from LDS, zero-extended to 32 bits.
- ds_read_i16: 16-bit signed halfword read from LDS, sign-extended to 32 bits.
- ds_write_b8: 8-bit byte write to LDS, modifying only addressed byte within dword.
- ds_write_b16: 16-bit halfword write to LDS, modifying only addressed halfword within dword.

Standard library only (python -I, struct, unittest). Zero game data.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L
import support


# ==============================================================================
# 1. ds_read_u8 Tests
# ==============================================================================

class TestDsReadU8(unittest.TestCase):
    def test_read_u8_zero_extension(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xAB",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0xFF",
            "ds_write_b8 v1, v4 offset:1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v5, v1 offset:1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v5", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xAB)
        self.assertEqual(L.out(mem, 0, 1), 0xFF)

    def test_read_u8_with_offset(self):
        body = [
            "v_mov_b32_e32 v1, 4",
            "v_mov_b32_e32 v2, 0x42",
            "ds_write_b8 v1, v2 offset:6",  # writes to effective address 10
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v3, v1 offset:6",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x42)

    def test_read_u8_all_byte_positions(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xA1B2C3D4",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v10, v1 offset:0",
            "ds_read_u8 v11, v1 offset:1",
            "ds_read_u8 v12, v1 offset:2",
            "ds_read_u8 v13, v1 offset:3",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v10", 0) + L.store("v11", 1) + L.store("v12", 2) + L.store("v13", 3)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xD4)
        self.assertEqual(L.out(mem, 0, 1), 0xC3)
        self.assertEqual(L.out(mem, 0, 2), 0xB2)
        self.assertEqual(L.out(mem, 0, 3), 0xA1)

    def test_read_u8_dest_aliasing(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x7E",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v1, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x7E)

    def test_read_u8_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("cs_bad_u8", ["ds_read_u8 s0, v1", "s_endpgm"])


# ==============================================================================
# 2. ds_read_i8 Tests
# ==============================================================================

class TestDsReadI8(unittest.TestCase):
    def test_read_i8_sign_extension_negative(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xFF",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0x80",
            "ds_write_b8 v1, v4 offset:1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v5, v1 offset:1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v5", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFFFFFF)
        self.assertEqual(L.out(mem, 0, 1), 0xFFFFFF80)

    def test_read_i8_positive(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x7F",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0x01",
            "ds_write_b8 v1, v4 offset:1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v5, v1 offset:1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v5", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x7F)
        self.assertEqual(L.out(mem, 0, 1), 0x01)

    def test_read_i8_with_offset(self):
        body = [
            "v_mov_b32_e32 v1, 4",
            "v_mov_b32_e32 v2, 0xFE",
            "ds_write_b8 v1, v2 offset:8",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v3, v1 offset:8",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFFFFFE)

    def test_read_i8_dest_aliasing(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x85",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v1, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFFFF85)


# ==============================================================================
# 3. ds_read_u16 Tests
# ==============================================================================

class TestDsReadU16(unittest.TestCase):
    def test_read_u16_zero_extension(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x8000",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0xFFFF",
            "ds_write_b16 v1, v4 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v5, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v5", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x00008000)
        self.assertEqual(L.out(mem, 0, 1), 0x0000FFFF)

    def test_read_u16_positive(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x1234",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x1234)

    def test_read_u16_with_offset(self):
        body = [
            "v_mov_b32_e32 v1, 4",
            "v_mov_b32_e32 v2, 0xABCD",
            "ds_write_b16 v1, v2 offset:12",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:12",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xABCD)

    def test_read_u16_dest_aliasing(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x9876",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v1, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x9876)


# ==============================================================================
# 4. ds_read_i16 Tests
# ==============================================================================

class TestDsReadI16(unittest.TestCase):
    def test_read_i16_sign_extension_negative(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x8000",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0xFFFF",
            "ds_write_b16 v1, v4 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v5, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v5", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFF8000)
        self.assertEqual(L.out(mem, 0, 1), 0xFFFFFFFF)

    def test_read_i16_positive(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x7FFF",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0x0042",
            "ds_write_b16 v1, v4 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v5, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v5", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x7FFF)
        self.assertEqual(L.out(mem, 0, 1), 0x0042)

    def test_read_i16_with_offset(self):
        body = [
            "v_mov_b32_e32 v1, 4",
            "v_mov_b32_e32 v2, 0x8001",
            "ds_write_b16 v1, v2 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1 offset:16",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFF8001)

    def test_read_i16_dest_aliasing(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x8123",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v1, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFF8123)


# ==============================================================================
# 5. ds_write_b8 Tests
# ==============================================================================

class TestDsWriteB8(unittest.TestCase):
    def test_write_b8_preserves_adjacent_bytes(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xA1B2C3D4",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xFFFFFF55",
            "ds_write_b8 v1, v3 offset:1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xA1B255D4)

    def test_write_b8_all_byte_positions(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x00000000",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0x11",
            "ds_write_b8 v1, v3 offset:0",
            "v_mov_b32_e32 v4, 0x22",
            "ds_write_b8 v1, v4 offset:1",
            "v_mov_b32_e32 v5, 0x33",
            "ds_write_b8 v1, v5 offset:2",
            "v_mov_b32_e32 v6, 0x44",
            "ds_write_b8 v1, v6 offset:3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v7, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v7", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x44332211)

    def test_write_b8_ignores_high_source_bits(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xDEADBEEF",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xEF)

    def test_write_b8_with_offset(self):
        body = [
            "v_mov_b32_e32 v1, 4",
            "v_mov_b32_e32 v2, 0x77",
            "ds_write_b8 v1, v2 offset:5",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v3, v1 offset:5",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x77)


# ==============================================================================
# 6. ds_write_b16 Tests
# ==============================================================================

class TestDsWriteB16(unittest.TestCase):
    def test_write_b16_preserves_other_halfword(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x11223344",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xDEAD",
            "ds_write_b16 v1, v3 offset:0",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x1122DEAD)

    def test_write_b16_high_halfword(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x1122DEAD",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xBEEF",
            "ds_write_b16 v1, v3 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xBEEFDEAD)

    def test_write_b16_ignores_high_source_bits(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xFFFFCAFE",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xCAFE)

    def test_write_b16_with_offset(self):
        body = [
            "v_mov_b32_e32 v1, 8",
            "v_mov_b32_e32 v2, 0x5432",
            "ds_write_b16 v1, v2 offset:6",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1 offset:6",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x5432)


if __name__ == "__main__":
    unittest.main()
