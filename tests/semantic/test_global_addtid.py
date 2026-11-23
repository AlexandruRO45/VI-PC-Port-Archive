"""Semantic, llvm-mc round-trip, and spirv-val tests for RDNA2 global_load/store_dword_addtid.

Covers AMD RDNA2 / GFX10 GLOBAL 0x016 (global_load_dword_addtid) and 0x017 (global_store_dword_addtid):
  - Operands: vDst/vData (1 VGPR), saddr (64-bit SGPR pair or `off` -> 0), optional signed 12-bit offset (-2048..2047).
  - Effective 64-bit address = saddr_u64 + (lane_id * 4) + sext12(offset), where lane_id is 0..31 in wave32 and
    0..63 in wave64; M0 is NOT added.
  - EXEC inactive-lane masking: inactive lanes neither load nor store and preserve vDst.
  - Unsupported forms: lds/tfe/d16, out-of-range offsets (< -2048 or > 2047), non-1-VGPR data, invalid saddr.
  - llvm-mc -mcpu=gfx1030 binary decode round-trip via tools/rdna2_decode.py.
  - shader_meta pointer-pair tracking and wave64_halves LANE taint classification.
  - spirv-val --target-env vulkan1.2 validation for both wave32 and wave64 compute shaders.
"""
from __future__ import annotations

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
TOOLS = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_meta  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import wave64_halves  # noqa: E402
from mc_command import mc_command  # noqa: E402
from shader_ir import parse_line  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def make_fssm(threads: int = 32, wave64: bool = False) -> bytes:
    """Synthetic compute FSSM header: rsrc2 = 0x88 (wave32) or 0x8088 (wave64)."""
    fssm = bytearray(0x148)
    rsrc2 = 0x8088 if wave64 else 0x88
    struct.pack_into("<4I", fssm, 0x138, rsrc2, threads, 1, 1)
    return bytes(fssm)


def translate(body: list[str], *, threads: int = 32, wave64: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_addtid.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_addtid.fssm"), "wb") as f:
            f.write(make_fssm(threads=threads, wave64=wave64))
        src = os.path.join(tmp, "disasm", "cs_addtid.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; cs_addtid\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run_cs(body: list[str], *, threads: int = 32, wave64: bool = False,
           memory: dict[int, int] | None = None, push_constants: list[int] | None = None) -> dict[int, int]:
    spv = translate(body, threads=threads, wave64=wave64)
    mem = dict(memory or {})
    pc = list(push_constants) if push_constants is not None else [0] * 36
    si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory=mem,
    )
    return mem


def _llvm_mc() -> list[str] | None:
    cmd = mc_command()
    try:
        r = subprocess.run(cmd + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                           input="s_endpgm\n", capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and "encoding:" in r.stdout else None


MC = _llvm_mc()


class TestGlobalAddtidSemantics(unittest.TestCase):
    def test_wave32_store_and_load_offsets_and_no_m0(self):
        """Wave32: stores and loads with offsets 0, 16, -4, 2047, -2048; M0 is NOT added."""
        base = 0x2000
        # Seed M0 with a non-zero value (0x300) to prove global_*_addtid ignores M0.
        for off in (0, 16, -4, -2048):
            with self.subTest(offset=off):
                mod = f" offset:{off}" if off else ""
                body = [
                    "s_mov_b32 m0, 0x300",
                    f"s_mov_b32 s4, {base:#x}",
                    "s_mov_b32 s5, 0",
                    "v_add_nc_u32_e32 v1, 0x1000, v0",
                    f"global_store_dword_addtid v1, s[4:5]{mod}",
                    "s_waitcnt_vscnt null, 0x0",
                    "v_mov_b32_e32 v2, 0xdeadbeef",
                    f"global_load_dword_addtid v2, s[4:5]{mod}",
                    "s_waitcnt vmcnt(0)",
                    "s_mov_b32 s6, 0x4000",
                    "s_mov_b32 s7, 0",
                    "global_store_dword_addtid v2, s[6:7]",
                ]
                mem = run_cs(body, threads=32, wave64=False)
                for lane in range(32):
                    expected_val = (0x1000 + lane) & M32
                    addr = base + lane * 4 + off
                    self.assertEqual(mem.get(addr), expected_val, f"lane {lane} store at {addr:#x}")
                    self.assertEqual(mem.get(0x4000 + lane * 4), expected_val, f"lane {lane} loadback")
                    # Ensure M0 was NOT added
                    self.assertNotIn(addr + 0x300, mem)

    def test_max_positive_offset_2047(self):
        """Offset +2047 with base aligned to 1 mod 4 lands on exact dword-aligned addresses."""
        base = 0x2001  # base + 2047 == 0x2800 (4-byte aligned)
        body = [
            f"s_mov_b32 s4, {base:#x}",
            "s_mov_b32 s5, 0",
            "v_add_nc_u32_e32 v1, 0x5500, v0",
            "global_store_dword_addtid v1, s[4:5] offset:2047",
            "s_waitcnt_vscnt null, 0x0",
            "global_load_dword_addtid v2, s[4:5] offset:2047",
            "s_waitcnt vmcnt(0)",
            "s_mov_b32 s6, 0x4000",
            "s_mov_b32 s7, 0",
            "global_store_dword_addtid v2, s[6:7]",
        ]
        mem = run_cs(body, threads=32, wave64=False)
        for lane in range(32):
            addr = base + lane * 4 + 2047
            self.assertEqual(addr, 0x2800 + lane * 4)
            self.assertEqual(mem.get(addr), 0x5500 + lane)
            self.assertEqual(mem.get(0x4000 + lane * 4), 0x5500 + lane)

    def test_wave64_all_64_lanes(self):
        """Wave64: lane_id spans 0..63 so addresses span base + 0..252 (+ offset)."""
        base = 0x3000
        body = [
            "s_mov_b64 s[10:11], exec",  # marks wave64 mode
            f"s_mov_b32 s4, {base:#x}",
            "s_mov_b32 s5, 0",
            "v_add_nc_u32_e32 v1, 0x6400, v0",
            "global_store_dword_addtid v1, s[4:5] offset:-8",
            "s_waitcnt_vscnt null, 0x0",
            "global_load_dword_addtid v2, s[4:5] offset:-8",
            "s_waitcnt vmcnt(0)",
            "s_mov_b32 s6, 0x5000",
            "s_mov_b32 s7, 0",
            "global_store_dword_addtid v2, s[6:7]",
        ]
        mem = run_cs(body, threads=64, wave64=True)
        self.assertTrue(rdna2spirv.translate.last_gen.wave64)
        for lane in range(64):
            val = 0x6400 + lane
            self.assertEqual(mem.get(base + lane * 4 - 8), val)
            self.assertEqual(mem.get(0x5000 + lane * 4), val)

    def test_saddr_off_base_zero(self):
        """saddr = off uses 64-bit base 0 -> address = (lane_id * 4) + sext12(offset)."""
        body = [
            "v_add_nc_u32_e32 v1, 0x7700, v0",
            "global_store_dword_addtid v1, off offset:256",
            "s_waitcnt_vscnt null, 0x0",
            "global_load_dword_addtid v2, off offset:256",
            "s_waitcnt vmcnt(0)",
            "global_store_dword_addtid v2, off offset:512",
        ]
        mem = run_cs(body, threads=32, wave64=False)
        for lane in range(32):
            self.assertEqual(mem.get(256 + lane * 4), 0x7700 + lane)
            self.assertEqual(mem.get(512 + lane * 4), 0x7700 + lane)

    def test_exec_inactive_lane_masking(self):
        """Inactive lanes (EXEC=0) neither store to memory nor overwrite vDst on load."""
        base = 0x2000
        out = 0x3000
        # Seed memory with recognizable values
        seed = {base + 4 * lane: 0xAA000000 | lane for lane in range(32)}
        body = [
            f"s_mov_b32 s4, {base:#x}",
            "s_mov_b32 s5, 0",
            f"s_mov_b32 s6, {out:#x}",
            "s_mov_b32 s7, 0",
            # Pre-fill v2 with sentinel so inactive lanes must keep it
            "v_add_nc_u32_e32 v2, 0xdead0000, v0",
            "v_add_nc_u32_e32 v1, 0xbb000000, v0",
            # Activate only even lanes (v0 & 1 == 0)
            "v_and_b32_e32 v3, 1, v0",
            "s_mov_b32 s8, exec_lo",
            "v_cmpx_eq_u32_e32 0, v3",
            "global_store_dword_addtid v1, s[4:5]",
            "s_waitcnt_vscnt null, 0x0",
            "global_load_dword_addtid v2, s[4:5]",
            "s_waitcnt vmcnt(0)",
            # Restore full EXEC and dump v2 to out
            "s_mov_b32 exec_lo, s8",
            "global_store_dword_addtid v2, s[6:7]",
        ]
        mem = run_cs(body, threads=32, wave64=False, memory=seed)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(mem[base + 4 * lane], 0xBB000000 | lane)
                self.assertEqual(mem[out + 4 * lane], 0xBB000000 | lane)
            else:
                self.assertEqual(mem[base + 4 * lane], 0xAA000000 | lane)
                self.assertEqual(mem[out + 4 * lane], 0xDEAD0000 + lane)

    def test_64bit_carry_and_borrow(self):
        """64-bit address arithmetic carries/borrows cleanly across the 32-bit boundary."""
        # Base = 0x00000001_FFFFFFFC -> lane 0 + offset:0 = 0x1FFFFFFFC, lane 1 + offset:0 = 0x200000000
        # Base = 0x00000002_00000000 -> lane 0 + offset:-4 = 0x1FFFFFFFC
        body = [
            "s_mov_b32 s4, 0xfffffffc",
            "s_mov_b32 s5, 1",
            "v_add_nc_u32_e32 v1, 0x900, v0",
            "global_store_dword_addtid v1, s[4:5]",
            "s_waitcnt_vscnt null, 0x0",
            "s_mov_b32 s6, 0",
            "s_mov_b32 s7, 2",
            "global_load_dword_addtid v2, s[6:7] offset:-4",
            "s_waitcnt vmcnt(0)",
            "s_mov_b32 s8, 0x1000",
            "s_mov_b32 s9, 0",
            "global_store_dword_addtid v2, s[8:9]",
        ]
        mem = run_cs(body, threads=4, wave64=False)
        self.assertEqual(mem[0x1FFFFFFFC], 0x900)
        self.assertEqual(mem[0x200000000], 0x901)
        self.assertEqual(mem[0x1000], 0x900)
        self.assertEqual(mem[0x1004], 0x901)


class TestGlobalAddtidUnsupported(unittest.TestCase):
    def test_rejected_forms(self):
        cases = [
            # Unsupported modifiers
            "global_load_dword_addtid v0, s[0:1] lds",
            "global_load_dword_addtid v0, s[0:1] tfe",
            "global_load_dword_addtid v0, s[0:1] d16",
            "global_store_dword_addtid v0, s[0:1] lds",
            "global_store_dword_addtid v0, s[0:1] tfe",
            "global_store_dword_addtid v0, s[0:1] d16",
            # Out-of-range / invalid offsets
            "global_load_dword_addtid v0, s[0:1] offset:2048",
            "global_load_dword_addtid v0, s[0:1] offset:-2049",
            "global_load_dword_addtid v0, s[0:1] offset:4096",
            "global_store_dword_addtid v0, s[0:1] offset:2048",
            "global_store_dword_addtid v0, s[0:1] offset:-2049",
            "global_load_dword_addtid v0, s[0:1] offset:bad",
            "global_load_dword_addtid v0, s[0:1] offset:4 offset:8",
            # Non-1-VGPR data operand
            "global_load_dword_addtid v[0:1], s[0:1]",
            "global_store_dword_addtid v[0:1], s[0:1]",
            "global_load_dword_addtid s2, s[0:1]",
            "global_store_dword_addtid s2, s[0:1]",
            # Invalid saddr
            "global_load_dword_addtid v0, s0",
            "global_load_dword_addtid v0, s[0:3]",
            "global_load_dword_addtid v0, v1",
            "global_store_dword_addtid v0, s0",
            "global_store_dword_addtid v0, s[0:3]",
            "global_store_dword_addtid v0, v1",
            "global_store_dword_addtid v0, 0",
        ]
        for line in cases:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate([line])


class TestGlobalAddtidDecodeAndAnalysis(unittest.TestCase):
    @unittest.skipUnless(MC is not None, "llvm-mc with AMDGPU gfx1030 not available")
    def test_llvm_mc_decode_round_trip(self):
        cases = [
            ("global_load_dword_addtid v5, s[4:5]", "global_load_dword_addtid", 5, ("sgpr", 4, 2), []),
            ("global_load_dword_addtid v5, s[4:5] offset:16", "global_load_dword_addtid", 5, ("sgpr", 4, 2), ["offset:16"]),
            ("global_load_dword_addtid v5, s[4:5] offset:-4", "global_load_dword_addtid", 5, ("sgpr", 4, 2), ["offset:-4"]),
            ("global_load_dword_addtid v3, off offset:2047", "global_load_dword_addtid", 3, ("special", "off"), ["offset:2047"]),
            ("global_store_dword_addtid v7, s[6:7]", "global_store_dword_addtid", 7, ("sgpr", 6, 2), []),
            ("global_store_dword_addtid v7, s[6:7] offset:-2048", "global_store_dword_addtid", 7, ("sgpr", 6, 2), ["offset:-2048"]),
            ("global_store_dword_addtid v9, off offset:12", "global_store_dword_addtid", 9, ("special", "off"), ["offset:12"]),
        ]
        src_text = "\n".join(c[0] for c in cases) + "\n"
        r = subprocess.run(MC + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                           input=src_text, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        enc_lines = [ln for ln in r.stdout.splitlines() if "encoding:" in ln]
        self.assertEqual(len(enc_lines), len(cases))
        for (src, want_op, want_v, want_s, want_mods), enc_ln in zip(cases, enc_lines):
            with self.subTest(src=src):
                raw = bytes(int(x, 16) for x in enc_ln.split("encoding:")[1].split("[")[1].split("]")[0].split(","))
                dec = rdna2_decode.decode_bytes(raw)
                self.assertEqual(dec["invalid"], 0)
                self.assertEqual(len(dec["insns"]), 1)
                ins = dec["insns"][0]
                self.assertEqual(ins["op"], want_op)
                self.assertEqual(ins["args"][0]["k"], "vgpr")
                self.assertEqual(ins["args"][0]["r"], want_v)
                self.assertEqual(ins["args"][0]["n"], 1)
                if want_s[0] == "sgpr":
                    self.assertEqual((ins["args"][1]["k"], ins["args"][1]["r"], ins["args"][1]["n"]), want_s)
                else:
                    self.assertEqual((ins["args"][1]["k"], ins["args"][1]["t"]), want_s)
                self.assertEqual(ins["mods"], want_mods)

    def test_shader_meta_tracks_saddr_pair(self):
        insns = [parse_line(ln) for ln in (
            "\tglobal_load_dword_addtid v1, s[0:1] offset:16",
            "\tglobal_store_dword_addtid v1, s[2:3] offset:-4",
            "\ts_endpgm",
        )]
        meta = shader_meta.analyse_instructions([insns], [[]])
        kinds = {r["sgpr"]: r["kind"] for r in meta["user_sgpr_reads"]}
        self.assertEqual(kinds.get(0), "ptr64_lo")
        self.assertEqual(kinds.get(1), "ptr64_hi")
        self.assertEqual(kinds.get(2), "ptr64_lo")
        self.assertEqual(kinds.get(3), "ptr64_hi")

    def test_wave64_halves_flags_addtid_lane_taint(self):
        for line in (
            "\tglobal_store_dword_addtid v1, s[0:1]",
            "\tglobal_load_dword_addtid v1, s[0:1]\n\tglobal_store_dword v2, v1, s[2:3]",
        ):
            with self.subTest(line=line):
                insns = [parse_line(ln) for ln in (line + "\n\ts_endpgm").splitlines()]
                res = wave64_halves.analyse([insns], [[]], set(), "cs")
                self.assertIn("lane-id", res["reasons"])


class TestGlobalAddtidSpirvVal(unittest.TestCase):
    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"),
                         "needs spirv-as and spirv-val on PATH")
    def test_spirv_val_wave32_and_wave64(self):
        body = [
            "s_mov_b32 s4, 0x2000",
            "s_mov_b32 s5, 0",
            "v_add_nc_u32_e32 v1, 0x100, v0",
            "global_store_dword_addtid v1, s[4:5]",
            "global_store_dword_addtid v1, s[4:5] offset:16",
            "global_store_dword_addtid v1, s[4:5] offset:-4",
            "global_store_dword_addtid v1, off offset:2047",
            "s_waitcnt_vscnt null, 0x0",
            "v_and_b32_e32 v3, 1, v0",
            "v_cmpx_eq_u32_e32 0, v3",
            "global_load_dword_addtid v2, s[4:5] offset:-2048",
            "global_load_dword_addtid v4, off offset:0",
            "s_waitcnt vmcnt(0)",
            "global_store_dword_addtid v2, s[4:5] offset:64",
        ]
        for label, threads, w64 in (("wave32", 32, False), ("wave64", 64, True)):
            with self.subTest(mode=label):
                prog = (["s_mov_b64 s[10:11], exec"] if w64 else []) + body
                spvasm = translate(prog, threads=threads, wave64=w64)
                self.assertEqual(rdna2spirv.translate.last_gen.wave64, w64)
                with tempfile.TemporaryDirectory() as tmp:
                    asm_p = os.path.join(tmp, f"{label}.spvasm")
                    spv_p = os.path.join(tmp, f"{label}.spv")
                    with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spvasm)
                    r1 = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm_p, "-o", spv_p],
                                        capture_output=True, text=True)
                    self.assertEqual(r1.returncode, 0, f"spirv-as failed ({label}): {r1.stderr}")
                    r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", spv_p],
                                        capture_output=True, text=True)
                    self.assertEqual(r2.returncode, 0, f"spirv-val failed ({label}): {r2.stderr}")


if __name__ == "__main__":
    unittest.main()
