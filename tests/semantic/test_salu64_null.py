"""Semantic tests for `null` destination and `null` source across 64-bit SALU instructions.

Every test assembles real RDNA2/GFX10 assembly with ``llvm-mc -mcpu=gfx1013 -show-encoding``,
translates it with ``tools/rdna2spirv.py``, assembles and validates the SPIR-V with
``spirv-as`` + ``spirv-val --target-env vulkan1.2``, and executes it on CPU via
``tests/semantic/spirv_mini_interp.py`` to verify register outputs, SCC, and EXEC.

Run directly:
    py -3.13 -I tests/semantic/test_salu64_null.py -v
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None


def _find_llvm_mc() -> str | None:
    env = os.environ.get("LLVM_MC")
    if env:
        return env if os.path.isfile(env) else shutil.which(env)
    return shutil.which("llvm-mc")


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    dis = os.path.join(_TMP_DIR.name, "disasm")
    fssm = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(dis)
    os.mkdir(fssm)
    _PS_PATH = os.path.join(dis, "ps_salu64_null.s")
    _CS_PATH = os.path.join(dis, "cs_salu64_null.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu64_null.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_salu64_null.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    # 4 user SGPRs (output V# in s[0:3]), TGID_X in s4, 64x1x1 threads.
    hdr = bytearray(0x148)
    struct.pack_into("<4I", hdr, 0x138, 0x8088, 64, 1, 1)
    with open(os.path.join(fssm, "cs_salu64_null.fssm"), "wb") as f:
        f.write(hdr)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def _assemble_and_translate(lines: list[str], path: str, header: str, *, wave64: bool = False) -> str:
    llvm_mc = _find_llvm_mc()
    if not llvm_mc:
        raise unittest.SkipTest("llvm-mc not found; set LLVM_MC or add it to PATH")
    if _TMP_DIR is None:
        setUpModule()
    asm = "".join(f"\t{ln}\n" for ln in lines)
    cmd = [llvm_mc, "-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
    if wave64:
        cmd.append("-mattr=+wavefrontsize64")
    cmd.append("-show-encoding")
    mc = subprocess.run(
        cmd,
        input=asm,
        capture_output=True,
        text=True,
    )
    if mc.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{mc.stderr}\nasm:\n{asm}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"; {header}\n{mc.stdout}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as and spirv_val:
        spvasm_file = os.path.join(_TMP_DIR.name, f"{header}.spvasm")
        spv_file = os.path.join(_TMP_DIR.name, f"{header}.spv")
        with open(spvasm_file, "w", encoding="utf-8", newline="") as f:
            f.write(spv)
        as_res = subprocess.run(
            [spirv_as, spvasm_file, "-o", spv_file, "--target-env", "vulkan1.2"],
            capture_output=True,
            text=True,
        )
        if as_res.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{as_res.stderr}")
        val_res = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", spv_file],
            capture_output=True,
            text=True,
        )
        if val_res.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{val_res.stderr}")
    return spv


def run_ps_lines(lines: list[str]) -> tuple[int, int, int]:
    """Run pixel shader lines, return (s[0:1] as uint64, SCC, s20_sentinel)."""
    tail = [
        "s_cselect_b32 s21, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "v_mov_b32 v2, s21",
        "v_mov_b32 v3, s20",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = _assemble_and_translate(lines + tail, _PS_PATH, "ps_salu64_null")
    w = mrt0_words(spv)
    return (w[0] | (w[1] << 32)), w[2], w[3]


def run_cs_lines(lines: list[str]) -> dict[int, int]:
    """Run 64-lane wave64 compute shader lines and return shared memory."""
    spv = _assemble_and_translate(lines + ["s_endpgm"], _CS_PATH, "cs_salu64_null", wave64=True)
    return si.run_workgroup(
        spv,
        64,
        64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
        memory={},
    )


def load64(reg: int, val: int) -> list[str]:
    return [
        f"s_mov_b32 s{reg}, {val & MASK32:#x}",
        f"s_mov_b32 s{reg + 1}, {(val >> 32) & MASK32:#x}",
    ]


def set_scc(val: int) -> str:
    return "s_cmp_eq_u32 1, 1" if val else "s_cmp_eq_u32 0, 1"


@unittest.skipUnless(_find_llvm_mc(), "needs llvm-mc")
class Salu64NullTests(unittest.TestCase):
    def test_non_scc_ops_null_destination_preserves_scc_and_regs(self):
        cases = [
            ("s_mov_b64", ["s_mov_b64 null, s[0:1]"]),
            ("s_cmov_b64", ["s_cmov_b64 null, s[0:1]"]),
            ("s_cselect_b64", ["s_cselect_b64 null, s[0:1], s[2:3]"]),
            ("s_brev_b64", ["s_brev_b64 null, s[0:1]"]),
            ("s_bitreplicate_b64_b32", ["s_bitreplicate_b64_b32 null, s4"]),
            ("s_bitset0_b64", ["s_bitset0_b64 null, s4"]),
            ("s_bitset1_b64", ["s_bitset1_b64 null, s4"]),
            ("s_bfm_b64", ["s_bfm_b64 null, s4, s5"]),
            ("s_flbit_i32_b64", ["s_flbit_i32_b64 null, s[0:1]"]),
            ("s_flbit_i32_i64", ["s_flbit_i32_i64 null, s[0:1]"]),
            ("s_ff0_i32_b64", ["s_ff0_i32_b64 null, s[0:1]"]),
            ("s_ff1_i32_b64", ["s_ff1_i32_b64 null, s[0:1]"]),
            ("s_movrels_b64", ["s_mov_b32 m0, 2", "s_movrels_b64 null, s[0:1]"]),
            ("s_getpc_b64", ["s_getpc_b64 null"]),
        ]
        seed0 = 0x11223344_55667788
        seed2 = 0x99AABBCC_DDEEFF00
        sentinel = 0xCAFEBABE
        for name, op_lines in cases:
            for scc_in in (0, 1):
                with self.subTest(op=name, scc_in=scc_in):
                    prog = (
                        load64(0, seed0)
                        + load64(2, seed2)
                        + [
                            "s_mov_b32 s4, 7",
                            "s_mov_b32 s5, 3",
                            f"s_mov_b32 s20, {sentinel:#x}",
                            set_scc(scc_in),
                        ]
                        + op_lines
                    )
                    got64, scc_out, sent_out = run_ps_lines(prog)
                    self.assertEqual(got64, seed0)
                    self.assertEqual(scc_out, scc_in)
                    self.assertEqual(sent_out, sentinel)

    def test_scc_writing_ops_null_destination_updates_scc(self):
        cases = [
            # (label, setup_lines, op_line, expected_scc)
            ("s_lshl_b64_nz_hi", load64(2, 0x00000001_00000000) + ["s_mov_b32 s4, 31"],
             "s_lshl_b64 null, s[2:3], s4", 1),
            ("s_lshl_b64_z_ovf", load64(2, 0x00000001_00000000) + ["s_mov_b32 s4, 32"],
             "s_lshl_b64 null, s[2:3], s4", 0),
            ("s_lshr_b64_nz_lo", load64(2, 0x80000000_00000000) + ["s_mov_b32 s4, 63"],
             "s_lshr_b64 null, s[2:3], s4", 1),
            ("s_lshr_b64_z", load64(2, 0x00000000_00000001) + ["s_mov_b32 s4, 1"],
             "s_lshr_b64 null, s[2:3], s4", 0),
            ("s_ashr_i64_nz", load64(2, 0x80000000_00000000) + ["s_mov_b32 s4, 63"],
             "s_ashr_i64 null, s[2:3], s4", 1),
            ("s_ashr_i64_z", load64(2, 0x7FFFFFFF_FFFFFFFF) + ["s_mov_b32 s4, 63"],
             "s_ashr_i64 null, s[2:3], s4", 0),
            ("s_bfe_u64_nz", load64(2, 0x00000123_45678000) + ["s_mov_b32 s4, 0x10001c"],
             "s_bfe_u64 null, s[2:3], s4", 1),
            ("s_bfe_u64_z", load64(2, 0x00000123_45678000) + ["s_mov_b32 s4, 0x00001c"],
             "s_bfe_u64 null, s[2:3], s4", 0),
            ("s_bfe_i64_nz", load64(2, 0x00000001_80000000) + ["s_mov_b32 s4, 0x02001f"],
             "s_bfe_i64 null, s[2:3], s4", 1),
            ("s_bfe_i64_z", load64(2, 0x00000001_00000000) + ["s_mov_b32 s4, 0x020021"],
             "s_bfe_i64 null, s[2:3], s4", 0),
            ("s_quadmask_b64_nz_hi", load64(2, 0x80000000_00000000),
             "s_quadmask_b64 null, s[2:3]", 1),
            ("s_quadmask_b64_z", load64(2, 0),
             "s_quadmask_b64 null, s[2:3]", 0),
            ("s_wqm_b64_nz_hi", load64(2, 0x10000000_00000000),
             "s_wqm_b64 null, s[2:3]", 1),
            ("s_wqm_b64_z", load64(2, 0),
             "s_wqm_b64 null, s[2:3]", 0),
            ("s_not_b64_nz_hi", load64(2, 0x7FFFFFFF_FFFFFFFF),
             "s_not_b64 null, s[2:3]", 1),
            ("s_not_b64_z", load64(2, MASK64),
             "s_not_b64 null, s[2:3]", 0),
            ("s_and_b64_nz_hi", load64(2, 0x80000000_00000000) + load64(4, 0x80000000_00000000),
             "s_and_b64 null, s[2:3], s[4:5]", 1),
            ("s_and_b64_z", load64(2, 0xFFFF0000_0000FFFF) + load64(4, 0x0000FFFF_FFFF0000),
             "s_and_b64 null, s[2:3], s[4:5]", 0),
            ("s_or_b64_nz_hi", load64(2, 0x00000001_00000000) + load64(4, 0),
             "s_or_b64 null, s[2:3], s[4:5]", 1),
            ("s_or_b64_z", load64(2, 0) + load64(4, 0),
             "s_or_b64 null, s[2:3], s[4:5]", 0),
            ("s_xor_b64_nz", load64(2, 0x00000001_00000000) + load64(4, 0),
             "s_xor_b64 null, s[2:3], s[4:5]", 1),
            ("s_xor_b64_z", load64(2, 0x12345678_9ABCDEF0) + load64(4, 0x12345678_9ABCDEF0),
             "s_xor_b64 null, s[2:3], s[4:5]", 0),
            ("s_andn2_b64_nz", load64(2, 0x00000001_00000000) + load64(4, 0),
             "s_andn2_b64 null, s[2:3], s[4:5]", 1),
            ("s_andn2_b64_z", load64(2, 0x12345678_9ABCDEF0) + load64(4, 0x12345678_9ABCDEF0),
             "s_andn2_b64 null, s[2:3], s[4:5]", 0),
            ("s_orn2_b64_nz", load64(2, 0) + load64(4, 0xFFFFFFFE_FFFFFFFF),
             "s_orn2_b64 null, s[2:3], s[4:5]", 1),
            ("s_orn2_b64_z", load64(2, 0) + load64(4, MASK64),
             "s_orn2_b64 null, s[2:3], s[4:5]", 0),
            ("s_nand_b64_nz", load64(2, MASK64) + load64(4, 0xFFFFFFFE_FFFFFFFF),
             "s_nand_b64 null, s[2:3], s[4:5]", 1),
            ("s_nand_b64_z", load64(2, MASK64) + load64(4, MASK64),
             "s_nand_b64 null, s[2:3], s[4:5]", 0),
            ("s_nor_b64_nz", load64(2, 0) + load64(4, 0xFFFFFFFE_FFFFFFFF),
             "s_nor_b64 null, s[2:3], s[4:5]", 1),
            ("s_nor_b64_z", load64(2, 0) + load64(4, MASK64),
             "s_nor_b64 null, s[2:3], s[4:5]", 0),
            ("s_xnor_b64_nz", load64(2, 0x12345678_9ABCDEF0) + load64(4, 0x12345678_9ABCDEF0),
             "s_xnor_b64 null, s[2:3], s[4:5]", 1),
            ("s_xnor_b64_z", load64(2, 0x12345678_9ABCDEF0) + load64(4, (~0x12345678_9ABCDEF0) & MASK64),
             "s_xnor_b64 null, s[2:3], s[4:5]", 0),
            ("s_bcnt0_i32_b64_nz", load64(2, 0xFFFFFFFE_FFFFFFFF),
             "s_bcnt0_i32_b64 null, s[2:3]", 1),
            ("s_bcnt0_i32_b64_z", load64(2, MASK64),
             "s_bcnt0_i32_b64 null, s[2:3]", 0),
            ("s_bcnt1_i32_b64_nz_hi", load64(2, 0x80000000_00000000),
             "s_bcnt1_i32_b64 null, s[2:3]", 1),
            ("s_bcnt1_i32_b64_z", load64(2, 0),
             "s_bcnt1_i32_b64 null, s[2:3]", 0),
        ]
        seed0 = 0xDEADBEEF_12345678
        for label, setup, op_line, want_scc in cases:
            # Start with the opposite SCC to prove the op overwrote SCC.
            scc_in = 1 - want_scc
            with self.subTest(case=label):
                prog = (
                    load64(0, seed0)
                    + setup
                    + [
                        "s_mov_b32 s20, 0x5555AAAA",
                        set_scc(scc_in),
                        op_line,
                    ]
                )
                got64, scc_out, sent_out = run_ps_lines(prog)
                self.assertEqual(got64, seed0)
                self.assertEqual(scc_out, want_scc)
                self.assertEqual(sent_out, 0x5555AAAA)

    def test_null_source_data_ops(self):
        v64 = 0x81234567_89ABCDEF
        # (label, setup, op_lines, want_s01, want_scc)
        cases = [
            # Non-SCC-writing ops (verify SCC preserved for both 0 and 1)
            ("s_mov_b64_src_null_scc0", load64(0, v64) + [set_scc(0)],
             ["s_mov_b64 s[0:1], null"], 0, 0),
            ("s_mov_b64_src_null_scc1", load64(0, v64) + [set_scc(1)],
             ["s_mov_b64 s[0:1], null"], 0, 1),
            ("s_cmov_b64_src_null_scc0", load64(0, v64) + [set_scc(0)],
             ["s_cmov_b64 s[0:1], null"], v64, 0),
            ("s_cmov_b64_src_null_scc1", load64(0, v64) + [set_scc(1)],
             ["s_cmov_b64 s[0:1], null"], 0, 1),
            ("s_cselect_b64_src0_null_scc1", load64(2, v64) + [set_scc(1)],
             ["s_cselect_b64 s[0:1], null, s[2:3]"], 0, 1),
            ("s_cselect_b64_src0_null_scc0", load64(2, v64) + [set_scc(0)],
             ["s_cselect_b64 s[0:1], null, s[2:3]"], v64, 0),
            ("s_cselect_b64_src1_null_scc1", load64(2, v64) + [set_scc(1)],
             ["s_cselect_b64 s[0:1], s[2:3], null"], v64, 1),
            ("s_cselect_b64_src1_null_scc0", load64(2, v64) + [set_scc(0)],
             ["s_cselect_b64 s[0:1], s[2:3], null"], 0, 0),
            ("s_brev_b64_src_null_scc1", load64(0, v64) + [set_scc(1)],
             ["s_brev_b64 s[0:1], null"], 0, 1),
            ("s_bitreplicate_b64_b32_src_null_scc1", load64(0, v64) + [set_scc(1)],
             ["s_bitreplicate_b64_b32 s[0:1], null"], 0, 1),
            ("s_bitset0_b64_src_null_scc1", load64(0, 0x00000000_00000003) + [set_scc(1)],
             ["s_bitset0_b64 s[0:1], null"], 0x00000000_00000002, 1),
            ("s_bitset1_b64_src_null_scc0", load64(0, 0x00000000_00000002) + [set_scc(0)],
             ["s_bitset1_b64 s[0:1], null"], 0x00000000_00000003, 0),
            ("s_flbit_i32_b64_src_null", [load64(0, 0)[1], set_scc(1)],
             ["s_flbit_i32_b64 s0, null"], MASK32, 1),
            ("s_flbit_i32_i64_src_null", [load64(0, 0)[1], set_scc(0)],
             ["s_flbit_i32_i64 s0, null"], MASK32, 0),
            ("s_ff0_i32_b64_src_null", ["s_mov_b32 s1, 0", set_scc(1)],
             ["s_ff0_i32_b64 s0, null"], 0, 1),
            ("s_ff1_i32_b64_src_null", ["s_mov_b32 s1, 0", set_scc(0)],
             ["s_ff1_i32_b64 s0, null"], MASK32, 0),

            # SCC-writing ops with null source
            ("s_lshl_b64_src0_null", ["s_mov_b32 s4, 5", set_scc(1)],
             ["s_lshl_b64 s[0:1], null, s4"], 0, 0),
            ("s_lshl_b64_src1_null", load64(2, v64) + [set_scc(0)],
             ["s_lshl_b64 s[0:1], s[2:3], null"], v64, 1),
            ("s_lshr_b64_src0_null", ["s_mov_b32 s4, 5", set_scc(1)],
             ["s_lshr_b64 s[0:1], null, s4"], 0, 0),
            ("s_lshr_b64_src1_null", load64(2, v64) + [set_scc(0)],
             ["s_lshr_b64 s[0:1], s[2:3], null"], v64, 1),
            ("s_ashr_i64_src0_null", ["s_mov_b32 s4, 5", set_scc(1)],
             ["s_ashr_i64 s[0:1], null, s4"], 0, 0),
            ("s_ashr_i64_src1_null", load64(2, v64) + [set_scc(0)],
             ["s_ashr_i64 s[0:1], s[2:3], null"], v64, 1),
            ("s_bfe_u64_src0_null", ["s_mov_b32 s4, 0x100004", set_scc(1)],
             ["s_bfe_u64 s[0:1], null, s4"], 0, 0),
            ("s_bfe_u64_src1_null", load64(2, v64) + [set_scc(1)],
             ["s_bfe_u64 s[0:1], s[2:3], null"], 0, 0),
            ("s_bfe_i64_src0_null", ["s_mov_b32 s4, 0x100004", set_scc(1)],
             ["s_bfe_i64 s[0:1], null, s4"], 0, 0),
            ("s_bfe_i64_src1_null", load64(2, v64) + [set_scc(1)],
             ["s_bfe_i64 s[0:1], s[2:3], null"], 0, 0),
            ("s_quadmask_b64_src_null", [set_scc(1)],
             ["s_quadmask_b64 s[0:1], null"], 0, 0),
            ("s_wqm_b64_src_null", [set_scc(1)],
             ["s_wqm_b64 s[0:1], null"], 0, 0),
            ("s_not_b64_src_null", [set_scc(0)],
             ["s_not_b64 s[0:1], null"], MASK64, 1),
            ("s_and_b64_src0_null", load64(2, v64) + [set_scc(1)],
             ["s_and_b64 s[0:1], null, s[2:3]"], 0, 0),
            ("s_and_b64_src1_null", load64(2, v64) + [set_scc(1)],
             ["s_and_b64 s[0:1], s[2:3], null"], 0, 0),
            ("s_or_b64_src0_null", load64(2, v64) + [set_scc(0)],
             ["s_or_b64 s[0:1], null, s[2:3]"], v64, 1),
            ("s_xor_b64_src1_null", load64(2, v64) + [set_scc(0)],
             ["s_xor_b64 s[0:1], s[2:3], null"], v64, 1),
            ("s_andn2_b64_src0_null", load64(2, v64) + [set_scc(1)],
             ["s_andn2_b64 s[0:1], null, s[2:3]"], 0, 0),
            ("s_andn2_b64_src1_null", load64(2, v64) + [set_scc(0)],
             ["s_andn2_b64 s[0:1], s[2:3], null"], v64, 1),
            ("s_orn2_b64_src0_null", load64(2, v64) + [set_scc(0)],
             ["s_orn2_b64 s[0:1], null, s[2:3]"], (~v64) & MASK64, 1),
            ("s_orn2_b64_src1_null", load64(2, 0) + [set_scc(0)],
             ["s_orn2_b64 s[0:1], s[2:3], null"], MASK64, 1),
            ("s_nand_b64_src_null", load64(2, v64) + [set_scc(0)],
             ["s_nand_b64 s[0:1], null, s[2:3]"], MASK64, 1),
            ("s_nor_b64_src_null", load64(2, v64) + [set_scc(0)],
             ["s_nor_b64 s[0:1], null, s[2:3]"], (~v64) & MASK64, 1),
            ("s_xnor_b64_src_null", load64(2, v64) + [set_scc(0)],
             ["s_xnor_b64 s[0:1], s[2:3], null"], (~v64) & MASK64, 1),
            ("s_bcnt0_i32_b64_src_null", ["s_mov_b32 s1, 0", set_scc(0)],
             ["s_bcnt0_i32_b64 s0, null"], 64, 1),
            ("s_bcnt1_i32_b64_src_null", ["s_mov_b32 s1, 0", set_scc(1)],
             ["s_bcnt1_i32_b64 s0, null"], 0, 0),
        ]
        for label, setup, op_lines, want_s01, want_scc in cases:
            with self.subTest(case=label):
                prog = ["s_mov_b32 s20, 0x13572468"] + setup + op_lines
                got64, scc_out, sent_out = run_ps_lines(prog)
                self.assertEqual(got64, want_s01)
                self.assertEqual(scc_out, want_scc)
                self.assertEqual(sent_out, 0x13572468)

    def test_compare_and_bitcmp_null_sources(self):
        cases = [
            ("s_cmp_eq_u64_null_zero", load64(2, 0), "s_cmp_eq_u64 null, s[2:3]", 1),
            ("s_cmp_eq_u64_null_hi", load64(2, 0x00000001_00000000), "s_cmp_eq_u64 null, s[2:3]", 0),
            ("s_cmp_eq_u64_hi_null", load64(2, 0x00000001_00000000), "s_cmp_eq_u64 s[2:3], null", 0),
            ("s_cmp_lg_u64_null_zero", load64(2, 0), "s_cmp_lg_u64 null, s[2:3]", 0),
            ("s_cmp_lg_u64_null_hi", load64(2, 0x00000001_00000000), "s_cmp_lg_u64 null, s[2:3]", 1),
            ("s_cmp_lg_u64_hi_null", load64(2, 0x00000001_00000000), "s_cmp_lg_u64 s[2:3], null", 1),
            ("s_bitcmp0_b64_null_src0", ["s_mov_b32 s4, 42"], "s_bitcmp0_b64 null, s4", 1),
            ("s_bitcmp1_b64_null_src0", ["s_mov_b32 s4, 42"], "s_bitcmp1_b64 null, s4", 0),
            ("s_bitcmp0_b64_null_src1", load64(2, 0xFFFFFFFF_FFFFFFFE), "s_bitcmp0_b64 s[2:3], null", 1),
            ("s_bitcmp1_b64_null_src1", load64(2, 0x00000000_00000001), "s_bitcmp1_b64 s[2:3], null", 1),
        ]
        for label, setup, op_line, want_scc in cases:
            with self.subTest(case=label):
                prog = load64(0, 0x11112222_33334444) + ["s_mov_b32 s20, 0"] + setup + [set_scc(1 - want_scc), op_line]
                got64, scc_out, _ = run_ps_lines(prog)
                self.assertEqual(got64, 0x11112222_33334444)
                self.assertEqual(scc_out, want_scc)

    def test_saveexec_wrexec_bfm_and_mask_ops_with_null_wave64(self):
        # Verify saveexec_b64 null, src updates EXEC and SCC while discarding dst,
        # wrexec_b64 null, src updates EXEC and SCC while discarding dst,
        # s_bfm_b64 works with null in dst/src0/src1, and s_cmp_eq_u64/s_or_b64 handle null masks.
        lines = [
            # Active lanes mask: lanes 0..7 in s[6:7]
            "v_cmp_gt_u32_e64 s[6:7], 8, v0",
            # s_and_saveexec_b64 null, s[6:7] -> EXEC becomes lanes 0..7, SCC = 1
            "s_and_saveexec_b64 null, s[6:7]",
            "s_cselect_b32 s10, 1, 0",
            # Each active lane (0..7) marks v1 = 0x11; inactive lanes stay 0x22
            "s_mov_b64 s[12:13], exec",
            "s_mov_b64 exec, -1",
            "v_mov_b32 v1, 0x22",
            "s_mov_b64 exec, s[12:13]",
            "v_mov_b32 v1, 0x11",
            "s_mov_b64 exec, -1",

            # s_and_saveexec_b64 null, null -> EXEC becomes 0, SCC = 0
            "s_and_saveexec_b64 null, null",
            "s_cselect_b32 s11, 1, 0",
            # Verify s_cmp_eq_u64 exec, null sees EXEC == 0 -> SCC = 1
            "s_cmp_eq_u64 exec, null",
            "s_cselect_b32 s12, 1, 0",
            # Restore EXEC = -1, then s_or_saveexec_b64 null, null -> EXEC stays -1, SCC = 1
            "s_mov_b64 exec, -1",
            "s_or_saveexec_b64 null, null",
            "s_cselect_b32 s13, 1, 0",

            # s_andn1_wrexec_b64 null, s[6:7] -> EXEC = ~s[6:7] & EXEC = lanes 8..63, SCC = 1
            "s_andn1_wrexec_b64 null, s[6:7]",
            "s_cselect_b32 s14, 1, 0",
            "s_mov_b64 s[26:27], exec",
            "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s[26:27]",
            # s_andn2_wrexec_b64 null, null -> EXEC = 0 & ~EXEC = 0, SCC = 0
            "s_andn2_wrexec_b64 null, null",
            "s_cselect_b32 s28, 1, 0",
            "s_mov_b64 exec, -1",

            # s_bfm_b64 with null operands
            "s_bfm_b64 s[16:17], 4, null",      # lanes 0..3
            "s_bfm_b64 s[18:19], null, 4",      # empty mask (size 0)
            set_scc(1),
            "s_bfm_b64 null, 4, 0",             # preserves SCC=1
            "s_cselect_b32 s15, 1, 0",
            "s_or_b64 s[20:21], s[16:17], null",
            "s_bcnt1_i32_b64 s22, s[20:21]",
            "s_bcnt1_i32_b64 s23, s[18:19]",

            # Store per-lane v1, v2 and scalar summary words at lane 0
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v1, v3, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v4, 256, v3",
            "buffer_store_dword v2, v4, s[0:3], 0 offen",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_and_saveexec_b64 s[24:25], vcc",
            "v_mov_b32 v5, s10",
            "v_mov_b32 v6, s11",
            "v_mov_b32 v7, s12",
            "v_mov_b32 v8, s13",
            "buffer_store_dwordx4 v[5:8], off, s[0:3], 0 offset:512",
            "v_mov_b32 v5, s14",
            "v_mov_b32 v6, s15",
            "v_mov_b32 v7, s22",
            "v_mov_b32 v8, s23",
            "buffer_store_dwordx4 v[5:8], off, s[0:3], 0 offset:528",
            "v_mov_b32 v5, s28",
            "buffer_store_dword v5, off, s[0:3], 0 offset:544",
            "s_mov_b64 exec, s[24:25]",
        ]
        mem = run_cs_lines(lines)
        for lane in range(64):
            self.assertEqual(mem[4 * lane], 0x11 if lane < 8 else 0x22, f"v1 lane {lane}")
            self.assertEqual(mem[256 + 4 * lane], 0 if lane < 8 else 1, f"v2 lane {lane}")
        self.assertEqual([mem[512 + 4 * k] for k in range(9)], [1, 0, 1, 1, 1, 1, 4, 0, 0])


if __name__ == "__main__":
    unittest.main()
