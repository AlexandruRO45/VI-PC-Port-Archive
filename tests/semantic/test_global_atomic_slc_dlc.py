"""Semantic, rejection, and binary round-trip tests for slc/dlc on RDNA2 global_atomic_*.

Covers all three translator paths in tools/rdna2spirv.py:
  - Gen.global_atomic:       global_atomic_add, global_atomic_cmpswap, global_atomic_fcmpswap, global_atomic_inc
  - Gen.global_atomic_float: global_atomic_fmin, global_atomic_fmax
  - Gen.global_atomic_x2:    global_atomic_and_x2, global_atomic_or_x2, global_atomic_xor_x2, global_atomic_swap_x2

Tests every cache-policy modifier combination:
  slc, dlc, slc+dlc, glc+slc, glc+dlc, glc+slc+dlc

Also verifies:
  - lds and unknown modifier flags remain Unsupported on all three paths.
  - Carrying 64-bit x2 ops (e.g. global_atomic_add_x2) remain Unsupported even with slc/dlc.
  - llvm-mc (-arch=amdgcn -mcpu=gfx1030) -> rdna2_decode -> rdna2spirv -> spirv-val (--target-env vulkan1.2)
    binary round-trip when llvm-mc and spirv-as/spirv-val are available.
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
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import mc_command  # noqa: E402
import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from shader_ir import parse_line as parse  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)

TARGET = 0x2000
OUT = 0x3000

MOD_COMBOS = (
    ("slc",),
    ("dlc",),
    ("slc", "dlc"),
    ("glc", "slc"),
    ("glc", "dlc"),
    ("glc", "slc", "dlc"),
)

# IEEE 754 binary32 bit patterns
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
ONE = 0x3F800000
TWO = 0x40000000
THREE = 0x40400000


def _llvm_mc():
    cmd = mc_command.mc_command()
    try:
        r = subprocess.run(
            cmd + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
            input="s_endpgm\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and "encoding" in r.stdout else None


MC = _llvm_mc()
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def translate(body: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_gatomic_slc.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_gatomic_slc.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_gatomic_slc.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; cs_gatomic_slc\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def spirv_validate(spv: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "m.spvasm")
        bin_path = os.path.join(tmp, "m.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run(
            [SPIRV_AS, "--target-env", "vulkan1.2", asm_path, "-o", bin_path],
            capture_output=True,
            text=True,
        )
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run(
            [SPIRV_VAL, "--target-env", "vulkan1.2", bin_path],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def assemble_gfx1030(lines: list[str]) -> bytes:
    r = subprocess.run(
        MC + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{r.stderr}")
    blob = bytearray()
    for ln in r.stdout.splitlines():
        if "encoding: [" in ln:
            body = ln.split("encoding: [", 1)[1].split("]", 1)[0]
            blob.extend(int(b, 16) for b in body.split(","))
    return bytes(blob)


def run_cs(body: list[str], memory: dict[int, int], nlanes: int = 1) -> dict[int, int]:
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(
        spv,
        nlanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0] * 36,
        memory=mem,
    )
    return mem


class GlobalAtomicSlcDlcSemantics(unittest.TestCase):
    def test_32bit_add_and_cmpswap(self):
        for mods in MOD_COMBOS:
            glc = "glc" in mods
            mod_str = " ".join(mods)

            with self.subTest(op="global_atomic_add", mods=mods):
                body = [
                    f"v_mov_b32_e32 v1, {TARGET:#x}",
                    "v_mov_b32_e32 v2, 7",
                    "v_mov_b32_e32 v5, 0xdeadbeef",
                    f"global_atomic_add v5, v1, v2, s[0:1] {mod_str}"
                    if glc
                    else f"global_atomic_add v1, v2, s[0:1] {mod_str}",
                    "s_waitcnt vmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run_cs(body, {TARGET: 35})
                self.assertEqual(mem.get(TARGET), 42)
                self.assertEqual(mem.get(OUT), 35 if glc else 0xDEADBEEF)

            with self.subTest(op="global_atomic_cmpswap", mods=mods):
                body = [
                    f"v_mov_b32_e32 v1, {TARGET:#x}",
                    "v_mov_b32_e32 v2, 0x99",
                    "v_mov_b32_e32 v3, 0x55",
                    "v_mov_b32_e32 v5, 0xdeadbeef",
                    f"global_atomic_cmpswap v5, v1, v[2:3], s[0:1] {mod_str}"
                    if glc
                    else f"global_atomic_cmpswap v1, v[2:3], s[0:1] {mod_str}",
                    "s_waitcnt vmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run_cs(body, {TARGET: 0x55})
                self.assertEqual(mem.get(TARGET), 0x99)
                self.assertEqual(mem.get(OUT), 0x55 if glc else 0xDEADBEEF)

    def test_float_fmin_fmax_and_fcmpswap(self):
        for mods in MOD_COMBOS:
            glc = "glc" in mods
            mod_str = " ".join(mods)

            for kind, m_init, d_val, expected in (
                ("fmin", TWO, ONE, ONE),
                ("fmax", ONE, THREE, THREE),
            ):
                with self.subTest(op=f"global_atomic_{kind}", mods=mods):
                    body = [
                        f"v_mov_b32_e32 v1, {TARGET:#x}",
                        f"v_mov_b32_e32 v2, {d_val:#x}",
                        "v_mov_b32_e32 v5, 0xdeadbeef",
                        f"global_atomic_{kind} v5, v1, v2, s[0:1] {mod_str}"
                        if glc
                        else f"global_atomic_{kind} v1, v2, s[0:1] {mod_str}",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v6, {OUT:#x}",
                        "global_store_dword v6, v5, s[0:1]",
                    ]
                    mem = run_cs(body, {TARGET: m_init})
                    self.assertEqual(mem.get(TARGET), expected)
                    self.assertEqual(mem.get(OUT), m_init if glc else 0xDEADBEEF)

            # fcmpswap compares as binary32 (-0.0 == +0.0)
            with self.subTest(op="global_atomic_fcmpswap", mods=mods):
                body = [
                    f"v_mov_b32_e32 v1, {TARGET:#x}",
                    f"v_mov_b32_e32 v2, {TWO:#x}",
                    f"v_mov_b32_e32 v3, {POS_ZERO:#x}",
                    "v_mov_b32_e32 v5, 0xdeadbeef",
                    f"global_atomic_fcmpswap v5, v1, v[2:3], s[0:1] {mod_str}"
                    if glc
                    else f"global_atomic_fcmpswap v1, v[2:3], s[0:1] {mod_str}",
                    "s_waitcnt vmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run_cs(body, {TARGET: NEG_ZERO})
                self.assertEqual(mem.get(TARGET), TWO)
                self.assertEqual(mem.get(OUT), NEG_ZERO if glc else 0xDEADBEEF)

    def test_x2_and_or_xor_swap(self):
        m_lo, m_hi = 0xF0F00F0F, 0x12345678
        d_lo, d_hi = 0x0FF00FF0, 0xFF00FF00
        cases = (
            ("and_x2", m_lo & d_lo, m_hi & d_hi),
            ("or_x2", m_lo | d_lo, m_hi | d_hi),
            ("xor_x2", m_lo ^ d_lo, m_hi ^ d_hi),
            ("swap_x2", d_lo, d_hi),
        )
        for mods in MOD_COMBOS:
            glc = "glc" in mods
            mod_str = " ".join(mods)
            for kind, exp_lo, exp_hi in cases:
                with self.subTest(op=f"global_atomic_{kind}", mods=mods):
                    body = [
                        f"v_mov_b32_e32 v1, {TARGET:#x}",
                        f"v_mov_b32_e32 v2, {d_lo:#x}",
                        f"v_mov_b32_e32 v3, {d_hi:#x}",
                        "v_mov_b32_e32 v6, 0xdead",
                        "v_mov_b32_e32 v7, 0xbeef",
                        f"global_atomic_{kind} v[6:7], v1, v[2:3], s[0:1] {mod_str}"
                        if glc
                        else f"global_atomic_{kind} v1, v[2:3], s[0:1] {mod_str}",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v8, {OUT:#x}",
                        "global_store_dwordx2 v8, v[6:7], s[0:1]",
                    ]
                    mem = run_cs(body, {TARGET: m_lo, TARGET + 4: m_hi})
                    self.assertEqual((mem.get(TARGET), mem.get(TARGET + 4)), (exp_lo, exp_hi))
                    self.assertEqual(
                        (mem.get(OUT), mem.get(OUT + 4)),
                        (m_lo, m_hi) if glc else (0xDEAD, 0xBEEF),
                    )

    def test_offset_and_off_addressing_with_slc_dlc(self):
        body = [
            f"v_mov_b32_e32 v8, {TARGET + 8:#x}",
            "v_mov_b32_e32 v9, 0",
            "v_mov_b32_e32 v2, 5",
            "v_mov_b32_e32 v5, 0",
            "global_atomic_add v5, v[8:9], v2, off offset:-8 glc slc dlc",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run_cs(body, {TARGET: 10})
        self.assertEqual(mem.get(TARGET), 15)
        self.assertEqual(mem.get(OUT), 10)

    def test_read_before_write_vgprs_with_slc_dlc(self):
        for line in (
            "\tglobal_atomic_add v1, v2, s[0:1] slc dlc",
            "\tglobal_atomic_add v5, v1, v2, s[0:1] glc slc dlc",
            "\tglobal_atomic_fmin v1, v2, s[0:1] slc",
            "\tglobal_atomic_fmin v5, v1, v2, s[0:1] glc dlc",
            "\tglobal_atomic_and_x2 v1, v[2:3], s[0:1] slc dlc",
            "\tglobal_atomic_and_x2 v[6:7], v1, v[2:3], s[0:1] glc slc dlc",
        ):
            with self.subTest(line=line):
                expected = {1, 2, 3} if "_x2" in line else {1, 2}
                self.assertEqual(rdna2spirv.read_before_write_vgprs([parse(line)]), expected)


class GlobalAtomicSlcDlcRejection(unittest.TestCase):
    def test_lds_and_unknown_modifiers_rejected_across_all_paths(self):
        rejected = [
            # lds on 32-bit, float, and x2 paths
            "global_atomic_add v1, v2, s[0:1] lds",
            "global_atomic_add v5, v1, v2, s[0:1] glc lds",
            "global_atomic_add v5, v1, v2, s[0:1] glc slc dlc lds",
            "global_atomic_cmpswap v5, v1, v[2:3], s[0:1] glc lds",
            "global_atomic_fcmpswap v5, v1, v[2:3], s[0:1] glc lds",
            "global_atomic_fmin v1, v2, s[0:1] lds",
            "global_atomic_fmin v5, v1, v2, s[0:1] glc slc lds",
            "global_atomic_fmax v5, v1, v2, s[0:1] glc dlc lds",
            "global_atomic_and_x2 v1, v[2:3], s[0:1] lds",
            "global_atomic_and_x2 v[6:7], v1, v[2:3], s[0:1] glc slc dlc lds",
            # Unknown / invalid modifier flags on all three paths
            "global_atomic_add v5, v1, v2, s[0:1] glc tfe",
            "global_atomic_add v5, v1, v2, s[0:1] glc slc bogus",
            "global_atomic_fmin v5, v1, v2, s[0:1] glc tfe",
            "global_atomic_fmax v5, v1, v2, s[0:1] glc dlc bogus",
            "global_atomic_and_x2 v[6:7], v1, v[2:3], s[0:1] glc tfe",
            "global_atomic_swap_x2 v[6:7], v1, v[2:3], s[0:1] glc slc bogus",
        ]
        for line in rejected:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v3, 0", line])

    def test_unsupported_carrying_x2_still_rejected_with_slc_dlc(self):
        for mods in MOD_COMBOS:
            glc = "glc" in mods
            mod_str = " ".join(mods)
            line = (
                f"global_atomic_add_x2 v[6:7], v1, v[2:3], s[0:1] {mod_str}"
                if glc
                else f"global_atomic_add_x2 v1, v[2:3], s[0:1] {mod_str}"
            )
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v3, 0", line])


class GlobalAtomicSlcDlcRoundTrip(unittest.TestCase):
    @unittest.skipUnless(MC, "needs llvm-mc with AMDGPU gfx1030 target (set LLVM_MC)")
    def test_llvm_mc_decode_translate_and_spirv_val(self):
        ops = (
            ("global_atomic_add", "v5", "v2"),
            ("global_atomic_cmpswap", "v5", "v[2:3]"),
            ("global_atomic_fmin", "v5", "v2"),
            ("global_atomic_fcmpswap", "v5", "v[2:3]"),
            ("global_atomic_and_x2", "v[6:7]", "v[2:3]"),
        )
        for op, vdst, vdata in ops:
            for mods in MOD_COMBOS:
                glc = "glc" in mods
                mod_str = " ".join(mods)
                ins_line = f"{op} {vdst}, v1, {vdata}, s[0:1] {mod_str}" if glc else f"{op} v1, {vdata}, s[0:1] {mod_str}"
                with self.subTest(op=op, mods=mods):
                    asm_lines = [
                        f"v_mov_b32_e32 v1, {TARGET:#x}",
                        "v_mov_b32_e32 v2, 1",
                        "v_mov_b32_e32 v3, 2",
                        ins_line,
                        "s_endpgm",
                    ]
                    blob = assemble_gfx1030(asm_lines)
                    decoded, unknown = rdna2_decode.disasm_bytes(blob)
                    self.assertEqual(unknown, 0, decoded)
                    self.assertIn(op, decoded)
                    for m in mods:
                        self.assertIn(m, decoded)
                    decoded_lines = [
                        ln.strip()
                        for ln in decoded.splitlines()
                        if ln.strip() and not ln.strip().startswith(";") and ln.strip() != "s_endpgm"
                    ]
                    spv = translate(decoded_lines)
                    spirv_validate(spv)

    @unittest.skipUnless(MC, "needs llvm-mc with AMDGPU gfx1030 target (set LLVM_MC)")
    def test_llvm_mc_decode_add_x2_slc_dlc_roundtrip_rejected(self):
        # Verify rdna2_decode decodes global_atomic_add_x2 with slc/dlc cleanly,
        # and rdna2spirv still rejects the carrying 64-bit op as Unsupported.
        for mods in MOD_COMBOS:
            glc = "glc" in mods
            mod_str = " ".join(mods)
            ins_line = (
                f"global_atomic_add_x2 v[6:7], v1, v[2:3], s[0:1] {mod_str}"
                if glc
                else f"global_atomic_add_x2 v1, v[2:3], s[0:1] {mod_str}"
            )
            with self.subTest(mods=mods):
                blob = assemble_gfx1030(["v_mov_b32_e32 v1, 0", ins_line, "s_endpgm"])
                decoded, unknown = rdna2_decode.disasm_bytes(blob)
                self.assertEqual(unknown, 0, decoded)
                self.assertIn("global_atomic_add_x2", decoded)
                for m in mods:
                    self.assertIn(m, decoded)
                decoded_lines = [
                    ln.strip()
                    for ln in decoded.splitlines()
                    if ln.strip() and not ln.strip().startswith(";") and ln.strip() != "s_endpgm"
                ]
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(decoded_lines)


if __name__ == "__main__":
    unittest.main()
