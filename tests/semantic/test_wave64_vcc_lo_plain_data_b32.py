"""Wave64: 32-bit bitwise/select SALU writes to vcc_lo stay plain data when VCC is not a lane mask.

`rdna2spirv.Gen._mask_b32_dst_half` only takes the per-lane mask path for wave64 `vcc_lo` / `vcc_hi`
when `"vcc" in self.cur_masks` or a source operand holds a lane mask; otherwise it computes and stores
a raw 32-bit scalar word into `%vcc`. `tools/lane_masks.py` must match that predicate under `wave64=True`:
unconditionally tagging `vcc` as `MASK` after a plain-data `s_not_b32` or 32-bit bitwise/select op into
`vcc_lo` causes later readers (`s_bcnt1_i32_b32`, `s_cmp_eq_u32`, chained `s_or_b32` / `s_and_b32`) to
misinterpret the raw scalar dword as a per-lane mask.

Covers:
  - Direct `lane_masks.analyse(..., wave64=True/False)` state checks across all 10 32-bit bitwise/select
    ops (`s_not_b32`, `s_and_b32`, `s_or_b32`, `s_xor_b32`, `s_andn2_b32`, `s_orn2_b32`, `s_nand_b32`,
    `s_nor_b32`, `s_xnor_b32`, `s_cselect_b32`), plus `wave32_pixel_scan.scan_shader`.
  - End-to-end translated wave64 compute shaders (native 64-lane subgroup and `--wave64-emulate`) plus
    binary SPIR-V assembly (`tools/spirv_asm.py`) and wave32/wave64 mask-preserving controls.

Standard library only; no GPU, no game files.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_wave64_emulate_opt import run_emulated_workgroup  # noqa: E402
import wave32_pixel_scan  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# s[0:3] = raw V# (OOB_SELECT 3); s4..s9 = push constants.
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

BIN_REFS = {
    "s_and_b32": lambda x, y: (x & y) & M32,
    "s_or_b32": lambda x, y: (x | y) & M32,
    "s_xor_b32": lambda x, y: (x ^ y) & M32,
    "s_andn2_b32": lambda x, y: (x & (~y & M32)) & M32,
    "s_orn2_b32": lambda x, y: (x | (~y & M32)) & M32,
    "s_nand_b32": lambda x, y: (~(x & y)) & M32,
    "s_nor_b32": lambda x, y: (~(x | y)) & M32,
    "s_xnor_b32": lambda x, y: (~(x ^ y)) & M32,
}

# Builds wave64 VCC from (s4, s5) and marks the program wave64 via a 64-bit EXEC/VCC op.
W64_BUILD_VCC = [
    "v_mov_b32_e32 v11, s5",
    "v_cmp_lt_u32_e64 vcc, v0, 32",
    "v_cndmask_b32_e64 v1, v11, s4, vcc",
    "v_and_b32_e32 v2, 31, v0",
    "v_lshrrev_b32_e32 v3, v2, v1",
    "v_and_b32_e32 v3, 1, v3",
    "v_cmp_ne_u32_e64 vcc, 0, v3",
    "s_and_b64 s[12:13], vcc, exec",
]


def translate_cs(name: str, lines: list[str], threads: int, emulate: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", f"cs_{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in lines:
                f.write(f"\t{ln}\n")
        with open(os.path.join(tmp, f"cs_{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", f"cs_{name}.fssm"), "wb") as f:
            # COMPUTE_PGM_RSRC2 0x8020: 16 user SGPRs (s[0:3] V#, s4..s15 push constants).
            f.write(bytes(0x138) + struct.pack("<4I", 0x8020, threads, 1, 1))
        old_emulate = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emulate
        return spv


def run_cs(
    spv: str,
    threads: int,
    *,
    emulate: bool = False,
    s4: int = 0,
    s5: int = 0,
    s6: int = 0,
    s7: int = 0,
    s8: int = 0,
    s9: int = 0,
    init_mem: dict[int, int] | None = None,
) -> dict[int, int]:
    pc = VSHARP + [s4 & M32, s5 & M32, s6 & M32, s7 & M32, s8 & M32, s9 & M32] + [0] * 22
    mem = dict(init_mem or {})
    if emulate:
        out_mem, _ = run_emulated_workgroup(spv, (threads, 1, 1), push_constants=pc, memory=mem)
        return out_mem
    return si.run_workgroup(
        spv,
        threads,
        threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory=mem,
    )


def bits_list(mask: int, width: int) -> list[int]:
    return [1 if (mask >> i) & 1 else 0 for i in range(width)]


class LaneMasksWave64VccLoPlainDataTests(unittest.TestCase):
    """Direct lane_masks.analyse checks for wave64 vs wave32 32-bit bitwise/select writes to vcc_lo."""

    @staticmethod
    def masks_after(*lines: str, wave64: bool = True):
        blk = [parse_line(ln) for ln in (*lines, "s_endpgm")]
        out, _, either = lane_masks.analyse([blk], [[]], include_either=True, wave64=wave64)
        return out[id(blk[-1])], either[id(blk[-1])]

    def test_wave64_plain_data_b32_writes_keep_vcc_plain_data(self):
        cases = [
            ("s_not_b32 vcc_lo, s4",),
            ("s_not_b32 vcc_lo, 0x0f",),
            ("s_cselect_b32 vcc_lo, s4, s5",),
            ("s_cselect_b32 vcc_lo, 0x05, 0x0a",),
            ("s_mov_b32 vcc_lo, 0", "s_or_b32 vcc_lo, vcc_lo, 0x0f"),
            ("v_cmp_lt_u32_e64 vcc, v0, 32", "s_buffer_load_dword vcc_lo, s[0:3], 0x0", "s_and_b32 vcc_lo, vcc_lo, 0x0f"),
            ("v_cmp_lt_u32_e64 vcc, v0, 32", "s_mov_b32 vcc_lo, s4", "s_not_b32 vcc_lo, vcc_lo"),
        ]
        for op in BIN_REFS:
            cases.append((f"{op} vcc_lo, s4, s5",))
            cases.append((f"{op} vcc_lo, 0x35, 0x0f",))
        for seq in cases:
            with self.subTest(seq=seq):
                masks, either = self.masks_after(*seq, wave64=True)
                self.assertNotIn("vcc", masks)
                self.assertNotIn("vcc", either)

    def test_wave64_mask_vcc_or_mask_source_keeps_vcc_mask(self):
        cases = [
            ("v_cmp_lt_u32_e64 vcc, v0, 32", "s_not_b32 vcc_lo, 0x0f"),
            ("v_cmp_lt_u32_e64 vcc, v0, 32", "s_and_b32 vcc_lo, vcc_lo, 0x0f"),
            ("v_cmp_lt_u32_e64 vcc, v0, 32", "s_cselect_b32 vcc_lo, s4, s5"),
            ("s_mov_b64 vcc, -1", "s_and_b32 vcc_lo, vcc_lo, 0xf0"),
            ("s_not_b32 vcc_lo, exec_lo",),
            ("s_and_b32 vcc_lo, exec_lo, 0x0f",),
            ("s_cselect_b32 vcc_lo, exec_lo, s4",),
        ]
        for seq in cases:
            with self.subTest(seq=seq):
                masks, _ = self.masks_after(*seq, wave64=True)
                self.assertEqual(masks.get("vcc"), lane_masks.MASK)

    def test_wave32_b32_writes_to_vcc_lo_still_produce_mask(self):
        for seq in (
            ("s_not_b32 vcc_lo, 1",),
            ("s_and_b32 vcc_lo, 0x0f, 0x03",),
            ("s_or_b32 vcc_lo, s4, 0x10",),
            ("s_cselect_b32 vcc_lo, s4, s5",),
        ):
            with self.subTest(seq=seq):
                masks, _ = self.masks_after(*seq, wave64=False)
                self.assertEqual(masks.get("vcc"), lane_masks.MASK)

    def test_wave32_pixel_scan_uses_wave32_lane_mask_rules(self):
        # Wave32 vcc_lo bitwise write stays a lane mask (s_bcnt1 reads mask_word), while wave32 s_bfm_b64 is data.
        insns_vcc = [parse_line(ln) for ln in (
            "s_and_b32 vcc_lo, 0x0f, 0x03",
            "s_bcnt1_i32_b32 s6, vcc_lo",
            "s_endpgm",
        )]
        self.assertEqual(wave32_pixel_scan.scan_shader(insns_vcc)["mask_word"], 1)

        insns_bfm = [parse_line(ln) for ln in (
            "s_bfm_b64 s[4:5], 2, 54",
            "s_bcnt1_i32_b64 s6, s[4:5]",
            "s_endpgm",
        )]
        self.assertEqual(wave32_pixel_scan.scan_shader(insns_bfm)["mask_word"], 0)


class Wave64VccLoPlainDataExecutionTests(unittest.TestCase):
    """End-to-end translated wave64 compute shaders writing plain scalar data into vcc_lo via 32-bit bitwise SALU."""

    def _run_plain_data_program(self, name: str, op_lines: list[str], *, emulate: bool, s4: int, s5: int,
                                s6: int, want: int, chain_op: str, want_chain: int, check_scc: bool = True):
        # s_mov_b64 s[14:15], exec marks the shader wave64 while leaving vcc uninitialized (plain data).
        asm = [
            "s_mov_b64 s[14:15], exec",
            *op_lines,
            "s_cselect_b32 s12, 1, 0",
            "s_bcnt1_i32_b32 s10, vcc_lo",
            "s_cmp_eq_u32 vcc_lo, s8",
            "s_cselect_b32 s11, 111, 222",
            chain_op,
            "v_mov_b32_e32 v4, vcc_lo",
            "v_mov_b32_e32 v5, s10",
            "v_mov_b32_e32 v7, s11",
            "v_mov_b32_e32 v8, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:512",
            "buffer_store_dword v8, v6, s[0:3], 0 offen offset:768",
            "s_endpgm",
        ]
        spv = translate_cs(f"{name}_e{int(emulate)}", asm, 64, emulate=emulate)
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertGreater(len(binary), 20)
        mem = run_cs(spv, 64, emulate=emulate, s4=s4, s5=s5, s6=s6, s8=want)
        self.assertEqual([mem[4 * i] for i in range(64)], [want_chain] * 64)
        self.assertEqual([mem[256 + 4 * i] for i in range(64)], [want.bit_count()] * 64)
        self.assertEqual([mem[512 + 4 * i] for i in range(64)], [111] * 64)
        if check_scc:
            self.assertEqual([mem[768 + 4 * i] for i in range(64)], [1 if want != 0 else 0] * 64)

    def test_all_b32_bitwise_ops_plain_data_into_vcc_lo_native_and_emulate(self):
        x, y = 0x15, 0x0F
        for emulate in (False, True):
            for op, ref in BIN_REFS.items():
                want = ref(x, y)
                want_chain = (want | 0x10) & M32
                with self.subTest(op=op, emulate=emulate):
                    self._run_plain_data_program(
                        f"w64_vcc_data_{op}",
                        [f"{op} vcc_lo, s4, s5"],
                        emulate=emulate,
                        s4=x,
                        s5=y,
                        s6=0,
                        want=want,
                        chain_op="s_or_b32 vcc_lo, vcc_lo, 0x10",
                        want_chain=want_chain,
                    )
            # s_not_b32 from scalar SGPR and from literal immediate (constant-folded scalar path).
            for tag, op_line, want in (
                ("sgpr", "s_not_b32 vcc_lo, s5", (~y) & M32),
                ("imm", "s_not_b32 vcc_lo, 0x0f", (~0x0F) & M32),
            ):
                with self.subTest(op=f"s_not_b32_{tag}", emulate=emulate):
                    self._run_plain_data_program(
                        f"w64_vcc_data_not_{tag}",
                        [op_line],
                        emulate=emulate,
                        s4=x,
                        s5=y,
                        s6=0,
                        want=want,
                        chain_op="s_and_b32 vcc_lo, vcc_lo, 0x3f",
                        want_chain=want & 0x3F,
                    )
            # Constant-folded immediate binary op into vcc_lo in wave64.
            with self.subTest(op="s_and_b32_imm", emulate=emulate):
                self._run_plain_data_program(
                    "w64_vcc_data_and_imm",
                    ["s_and_b32 vcc_lo, 0x35, 0x0f"],
                    emulate=emulate,
                    s4=0,
                    s5=0,
                    s6=0,
                    want=0x05,
                    chain_op="s_or_b32 vcc_lo, vcc_lo, 0x20",
                    want_chain=0x25,
                )
            # s_cselect_b32 into vcc_lo with plain scalar sources (both SCC=1 and SCC=0).
            for scc_flag, want in ((1, 0x15), (0, 0x0F)):
                with self.subTest(op="s_cselect_b32", scc=scc_flag, emulate=emulate):
                    self._run_plain_data_program(
                        f"w64_vcc_data_cselect_{scc_flag}",
                        ["s_cmp_eq_u32 s6, 1", "s_cselect_b32 vcc_lo, s4, s5"],
                        emulate=emulate,
                        s4=x,
                        s5=y,
                        s6=scc_flag,
                        want=want,
                        chain_op="s_or_b32 vcc_lo, vcc_lo, 0x40",
                        want_chain=want | 0x40,
                        check_scc=False,
                    )

    def test_plain_data_b32_after_vcmp_overwrite_and_after_either_zero(self):
        # 1) v_cmp makes vcc a mask, s_buffer_load_dword vcc_lo ends mask state, then s_and_b32 vcc_lo must stay data.
        asm_after_load = W64_BUILD_VCC + [
            "s_buffer_load_dword vcc_lo, s[0:3], 0x2000",
            "s_and_b32 vcc_lo, vcc_lo, 0x3f",
            "s_bcnt1_i32_b32 s10, vcc_lo",
            "s_or_b32 vcc_lo, vcc_lo, 0x80",
            "v_mov_b32_e32 v4, vcc_lo",
            "v_mov_b32_e32 v5, s10",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
            "s_endpgm",
        ]
        # 2) s_mov_b32 vcc_lo, 0 leaves vcc in EITHER (not MASK_KINDS), then s_or_b32 vcc_lo, vcc_lo, 0x0f stays data.
        asm_after_zero = [
            "s_mov_b64 s[14:15], exec",
            "s_mov_b32 vcc_lo, 0",
            "s_or_b32 vcc_lo, vcc_lo, 0x0f",
            "s_bcnt1_i32_b32 s10, vcc_lo",
            "s_or_b32 vcc_lo, vcc_lo, 0x10",
            "v_mov_b32_e32 v4, vcc_lo",
            "v_mov_b32_e32 v5, s10",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
            "s_endpgm",
        ]
        for emulate in (False, True):
            with self.subTest(case="after_load", emulate=emulate):
                spv = translate_cs(f"w64_vcc_after_load_e{int(emulate)}", asm_after_load, 64, emulate=emulate)
                mem = run_cs(spv, 64, emulate=emulate, s4=0xAAAAAAAA, s5=0x55555555, init_mem={0x2000: 0x12340015})
                self.assertEqual([mem[4 * i] for i in range(64)], [0x95] * 64)
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [3] * 64)
            with self.subTest(case="after_zero", emulate=emulate):
                spv = translate_cs(f"w64_vcc_after_zero_e{int(emulate)}", asm_after_zero, 64, emulate=emulate)
                mem = run_cs(spv, 64, emulate=emulate)
                self.assertEqual([mem[4 * i] for i in range(64)], [0x1F] * 64)
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [4] * 64)

    def test_positive_mask_controls_wave64_and_wave32(self):
        # Wave64 control 1: vcc is ALREADY a mask -> s_not_b32 vcc_lo updates lanes 0..31 and preserves vcc_hi.
        asm_w64_keep = W64_BUILD_VCC + [
            "s_not_b32 vcc_lo, 0x0000000f",
            "v_cndmask_b32_e64 v4, 0, 1, vcc",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv_keep = translate_cs("w64_vcc_mask_keep", asm_w64_keep, 64)
        mem_keep = run_cs(spv_keep, 64, s4=0x12345678, s5=0x80000005)
        want_vcc = ((~0x0000000F) & M32) | (0x80000005 << 32)
        self.assertEqual([mem_keep[4 * i] for i in range(64)], bits_list(want_vcc, 64))

        # Wave64 control 2: vcc starts plain data, mask source exec_lo promotes vcc_lo to a mask.
        asm_w64_promote = [
            "s_mov_b64 s[14:15], exec",
            "s_and_b32 vcc_lo, exec_lo, 0x0000000f",
            "s_bcnt1_i32_b32 s10, vcc_lo",
            "v_mov_b32_e32 v4, s10",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv_prom = translate_cs("w64_vcc_mask_promote", asm_w64_promote, 64)
        mem_prom = run_cs(spv_prom, 64)
        self.assertEqual([mem_prom[4 * i] for i in range(64)], [4] * 64)

        # Wave32 control: plain immediate bitwise op into vcc_lo still builds a 32-lane mask.
        asm_w32 = [
            "s_and_b32 vcc_lo, 0x0f, 0x05",
            "s_bcnt1_i32_b32 s10, vcc_lo",
            "v_cndmask_b32_e64 v4, 0, 1, vcc_lo",
            "v_mov_b32_e32 v5, s10",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:128",
            "s_endpgm",
        ]
        spv_w32 = translate_cs("w32_vcc_mask_ctl", asm_w32, 32)
        mem_w32 = run_cs(spv_w32, 32)
        self.assertEqual([mem_w32[4 * i] for i in range(32)], bits_list(0x05, 32))
        self.assertEqual([mem_w32[128 + 4 * i] for i in range(32)], [2] * 32)


if __name__ == "__main__":
    unittest.main()
