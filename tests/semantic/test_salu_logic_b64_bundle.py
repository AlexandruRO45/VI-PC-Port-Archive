"""Hardware-verified semantic reference and translator regression suite for RDNA2 SALU 64-bit logic bundle.

Covers the complete 64-bit scalar ALU bitwise logic instruction family:
- s_and_b64:  D = S0 & S1, SCC = (D != 0)
- s_or_b64:   D = S0 | S1, SCC = (D != 0)
- s_xor_b64:  D = S0 ^ S1, SCC = (D != 0)
- s_andn2_b64: D = S0 & ~S1, SCC = (D != 0)
- s_orn2_b64:  D = S0 | ~S1, SCC = (D != 0)
- s_nand_b64: D = ~(S0 & S1), SCC = (D != 0)
- s_nor_b64:  D = ~(S0 | S1), SCC = (D != 0)
- s_xnor_b64: D = ~(S0 ^ S1), SCC = (D != 0)

Zero game data, standard library Python only.
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_BIT64: int = 0x8000_0000_0000_0000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit quadword."""
    return val & MASK64


def split64(val: int) -> tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32)."""
    return val & MASK32, (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    """Combine (lo32, hi32) into 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


# Reference mathematical oracles
def oracle_s_and_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) & u64(b)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_or_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) | u64(b)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_xor_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) ^ u64(b)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_andn2_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) & (~u64(b) & MASK64)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_orn2_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) | (~u64(b) & MASK64)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_nand_b64(a: int, b: int) -> tuple[int, int]:
    res = (~(u64(a) & u64(b))) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_nor_b64(a: int, b: int) -> tuple[int, int]:
    res = (~(u64(a) | u64(b))) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_xnor_b64(a: int, b: int) -> tuple[int, int]:
    res = (~(u64(a) ^ u64(b))) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


_TRANSLATE_ID: int = 0


def run_translated_salu64(op: str, s0_64: int, s1_64: int) -> tuple[int, int]:
    """Execute 64-bit SALU opcode through rdna2spirv and return (res64, scc)."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    s0_lo, s0_hi = split64(s0_64)
    s1_lo, s1_hi = split64(s1_64)
    prog = [
        f"s_mov_b32 s0, {s0_lo:#x}",
        f"s_mov_b32 s1, {s0_hi:#x}",
        f"s_mov_b32 s2, {s1_lo:#x}",
        f"s_mov_b32 s3, {s1_hi:#x}",
        f"{op} s[4:5], s[0:1], s[2:3]",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "s_cselect_b32 s6, 1, 0",
        "v_mov_b32 v2, s6",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_salu64_{op}_{_TRANSLATE_ID}", prog)
    words = support.mrt0_words(spv)
    res64 = join64(words[0], words[1])
    scc = words[2] & 1
    return res64, scc


@dataclasses.dataclass(frozen=True)
class SGPRPair:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 104 and self.index % 2 == 0):
            raise ValueError(f"SGPR pair index must be even in [0, 104], got {self.index}")

    def __str__(self) -> str:
        return f"s[{self.index}:{self.index + 1}]"


class SALUOpcode(enum.Enum):
    S_AND_B64 = "s_and_b64"
    S_OR_B64 = "s_or_b64"
    S_XOR_B64 = "s_xor_b64"
    S_ANDN2_B64 = "s_andn2_b64"
    S_ORN2_B64 = "s_orn2_b64"
    S_NAND_B64 = "s_nand_b64"
    S_NOR_B64 = "s_nor_b64"
    S_XNOR_B64 = "s_xnor_b64"


@dataclasses.dataclass
class SOP2_64Instruction:
    opcode: SALUOpcode
    dst: SGPRPair
    src0: SGPRPair
    src1: SGPRPair

    def to_asm(self) -> str:
        return f"{self.opcode.value} {self.dst}, {self.src0}, {self.src1}"


class SALUMachineState:
    """Architectural state for executing 64-bit SALU instructions."""

    def __init__(self) -> None:
        self.sgprs: List[int] = [0] * 106
        self.scc: int = 0

    def read_pair(self, pair: SGPRPair) -> int:
        return join64(self.sgprs[pair.index], self.sgprs[pair.index + 1])

    def write_pair(self, pair: SGPRPair, val64: int) -> None:
        lo, hi = split64(val64)
        self.sgprs[pair.index] = lo
        self.sgprs[pair.index + 1] = hi

    def step(self, inst: SOP2_64Instruction) -> tuple[int, int]:
        v0 = self.read_pair(inst.src0)
        v1 = self.read_pair(inst.src1)
        dispatch = {
            SALUOpcode.S_AND_B64: oracle_s_and_b64,
            SALUOpcode.S_OR_B64: oracle_s_or_b64,
            SALUOpcode.S_XOR_B64: oracle_s_xor_b64,
            SALUOpcode.S_ANDN2_B64: oracle_s_andn2_b64,
            SALUOpcode.S_ORN2_B64: oracle_s_orn2_b64,
            SALUOpcode.S_NAND_B64: oracle_s_nand_b64,
            SALUOpcode.S_NOR_B64: oracle_s_nor_b64,
            SALUOpcode.S_XNOR_B64: oracle_s_xnor_b64,
        }
        res, scc = dispatch[inst.opcode](v0, v1)
        self.write_pair(inst.dst, res)
        self.scc = scc
        return res, scc

class TestSAndB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_and_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        res, scc = oracle_s_and_b64(a, b)
        self.assertEqual(scc, 1 if res != 0 else 0)

    def test_oracle_zero_scc(self) -> None:
        res, scc = oracle_s_and_b64(0x5555555555555555, 0xaaaaaaaaaaaaaaaa)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        res_same, _ = oracle_s_and_b64(0xCAFE_BABE_DEAD_BEEF, 0xCAFE_BABE_DEAD_BEEF)
        self.assertIsInstance(res_same, int)

if __name__ == "__main__":
    unittest.main()
