"""Semantic tests for a compare whose destination is a VGPR (`v_cmp_* vN, ...`, the VOP3 SDST form).

AMD's compiler emits this form in the retail pipeline-library compute programs: the comparison lands in a VGPR as
-1/0 per lane instead of a lane mask in VCC or an SGPR pair, and the result is then used as a value (multiplied,
selected, or compared against -1). Before this rule the translator rejected every such program with
`v_cmp destination` (606 of the 2,684 compute programs of GTA VI 1.0.2.1's `shaders.rpf` pipeline library), which
is why the title's Xbox DeserializeComputePipelineStateX (d3d12_xs device slot 67) had no translated program for them.

Checked here, per lane, in wave32, wave64 and wave64 under --wave64-emulate:
  - `v_cmp_lt_u32 v2, lane, 16` writes 0xFFFFFFFF to v2 on lanes below 16 and 0 on the others (not 1/0);
  - `v_cmp_eq_u32 v3, lane, 3` does the same for a single lane;
  - the value is a value: `v_mul_lo_u32` of it by 2 is -2/0, and comparing it against -1 rebuilds the mask
    (`v_cmp_eq_u32_e64 s6, v2, -1`, `s_bcnt1_i32_b32` counts 16 active lanes);
  - a float compare into a VGPR (`v_cmp_lt_f32`, false on NaN) writes the same -1/0.
"""
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw): stores are in range up to num_records


def _translate_cs(name, body_lines, wave_size=32, emulate=False):
    """Translate a synthetic compute shader and return `(spv, gen)`."""
    prev_emu = rdna2spirv.WAVE64_EMULATE
    prev_emu_gfx = rdna2spirv.WAVE64_GRAPHICS
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n")
                for ln in body_lines:
                    f.write(f"\t{ln}\n")
                f.write("\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2 0x8088: 4 user SGPRs (s[0:3] = the output V#), TGID_X enabled.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
            gen = rdna2spirv.translate.last_gen
            return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu
        rdna2spirv.WAVE64_GRAPHICS = prev_emu_gfx


def _run_cs(name, body_lines, wave_size=32, emulate=False):
    """Translate a synthetic compute shader and execute every lane in spirv_mini_interp."""
    spv, _ = _translate_cs(name, body_lines, wave_size=wave_size, emulate=emulate)
    sg_size = 32 if emulate else wave_size
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )


def _compare_body(wave_size, emulate=False):
    """Compare into VGPRs, use the result as a value, and store the per-lane words plus a lane count."""
    lines = [] if (emulate or wave_size == 32) else ["s_mov_b64 exec, -1"]
    lines += [
        # lane id in v0 (already the thread id), copy so v0 stays the index
        "v_lshlrev_b32 v4, 4, v0",                 # byte offset of this lane: 4 dwords per lane
        "v_cmp_lt_u32 v2, v0, 16",                 # VGPR destination: -1 below lane 16, 0 above
        "v_cmp_eq_u32 v3, v0, 3",                  # -1 on lane 3 only
        "v_cmp_lt_f32 v5, v0, 8",                  # float compare into a VGPR (int bits compared, still 0/-1)
        "v_mul_lo_u32 v6, v2, 2",                  # the result is a value: -2 / 0
        "buffer_store_dword v2, v4, s[0:3], 0 offen",
        "v_add_nc_u32 v7, 4, v4",
        "buffer_store_dword v3, v7, s[0:3], 0 offen",
        "v_add_nc_u32 v7, 8, v4",
        "buffer_store_dword v6, v7, s[0:3], 0 offen",
        "v_add_nc_u32 v7, 12, v4",
        "buffer_store_dword v5, v7, s[0:3], 0 offen",
        # the whole-wave check: comparing the VGPR against -1 rebuilds the mask, and the count is 16
        "v_cmp_eq_u32_e64 s6, v2, -1",
        "s_bcnt1_i32_b32 s8, s6",
        "v_mov_b32 v9, s8",
        "v_mov_b32 v10, 0x1000",
        "buffer_store_dword v9, v10, s[0:3], 0 offen",
    ]
    return lines


def _words(mem, lane):
    off = lane * 16
    return tuple(mem[off + 4 * i] for i in range(4))


class _CompareIntoVgprMixin:
    wave_size = 32
    emulate = False

    def _check(self):
        mem = _run_cs(f"cs_vcmp_vgpr_w{self.wave_size}", _compare_body(self.wave_size),
                      wave_size=self.wave_size, emulate=self.emulate)
        for lane in range(self.wave_size):
            lt, eq, mul, flt = _words(mem, lane)
            want_lt = 0xFFFFFFFF if lane < 16 else 0
            want_eq = 0xFFFFFFFF if lane == 3 else 0
            want_mul = 0xFFFFFFFE if lane < 16 else 0
            want_flt = 0xFFFFFFFF if lane < 8 else 0
            self.assertEqual(lt, want_lt, f"lane {lane}: v_cmp_lt_u32 into a VGPR")
            self.assertEqual(eq, want_eq, f"lane {lane}: v_cmp_eq_u32 into a VGPR")
            self.assertEqual(mul, want_mul, f"lane {lane}: the compare result as a value")
            self.assertEqual(flt, want_flt, f"lane {lane}: v_cmp_lt_f32 into a VGPR")
        count = mem[0x1000]
        self.assertEqual(count, 16, "v_cmp_eq_u32_e64 of the VGPR against -1 must count the 16 lanes below 16")

    def test_compare_into_vgpr(self):
        self._check()


class TestWave32(_CompareIntoVgprMixin, unittest.TestCase):
    wave_size = 32


class TestWave64(_CompareIntoVgprMixin, unittest.TestCase):
    wave_size = 64


class TestWave64Emulated(unittest.TestCase):
    """--wave64-emulate: the rule must still translate (spirv_mini_interp rejects the emulation's barriers)."""

    def test_compare_into_vgpr_translates(self):
        spv, gen = _translate_cs("cs_vcmp_vgpr_emu", _compare_body(64, emulate=True), wave_size=64, emulate=True)
        self.assertFalse(gen.wave64)
        self.assertIn("%_ptr_Function_uint", spv)
        self.assertNotIn("OpSelect %uint %true", spv)


if __name__ == "__main__":
    unittest.main()
