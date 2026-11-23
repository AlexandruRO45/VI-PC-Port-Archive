"""--vs-compute: a wave64 vertex shader built as a compute module is exact on any subgroup size (issue #94, atom 1).

The shaders are the tier-C vertex fixtures of runtime/test/shaders (vs_w64_c_*), whose results depend on the 64 lanes of
the hardware wave. Each one is translated with rdna2spirv.VS_COMPUTE_PASS, which emulates the wave in workgroup
memory, and run on the CPU as one 64-invocation workgroup (test_wave64_emulate_opt.run_emulated_workgroup). The
expected values are the ISA's for one 64-lane wave of `count` vertices: the hardware leaves the lanes past the last
vertex out of EXEC. The exports go to the --vs-compute output buffer (rdna2spirv.vs_compute_layout).
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import support  # noqa: E402,F401
import rdna2spirv  # noqa: E402
from test_wave64_emulate_opt import run_emulated_workgroup  # noqa: E402

VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]     # raw buffer covering all offsets, at heap address 0
OUT = 0x100000                               # --vs-compute output buffer
USER = rdna2spirv.USER_SGPRS


def translate_vsc(name, insns):
    """Translate `(text, nbytes)` instructions as a wave64 VS with VS_COMPUTE_PASS. Returns (spvasm, gen)."""
    lines = [f"; {name}  synthetic", "; mattr: +wavefrontsize64,-wavefrontsize32"]
    for txt, nb in insns:
        lines.append(f"\t{txt:52s} ; encoding: [{','.join(['0x00'] * nb)}]")
    prev = rdna2spirv.VS_COMPUTE_PASS
    rdna2spirv.VS_COMPUTE_PASS = True
    try:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, name + ".s")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            spv = rdna2spirv.translate(path)
            return spv, rdna2spirv.translate.last_gen
    finally:
        rdna2spirv.VS_COMPUTE_PASS = prev


def push_constants(count):
    """s0..s3 = the V#, the rest of the user SGPRs 0, heap base 0, code base 0, output address, vertex count."""
    return VSHARP + [0] * (USER - 4) + [0, 0, 0, 0, OUT & 0xFFFFFFFF, OUT >> 32, count]


def run(spv, count, wg=(0, 0, 0), num_wg=(1, 1, 1)):
    mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=push_constants(count), memory={},
                                    wg_id=wg, num_wg=num_wg)
    return mem


# vs_w64_c_bcnt_export.asm: x = v0 & 63; r0 = s_bcnt1 of the lanes with x % 4 == 1, r1 = that lane's bit; the record
# of vertex v is at 0x1000 + 16 * v. Exports pos0 = param0 = 0.
BCNT = [("v_and_b32_e32 v2, 63, v0", 4), ("v_and_b32_e32 v4, 3, v2", 4), ("v_cmp_eq_u32_e32 vcc, 1, v4", 4),
        ("s_bcnt1_i32_b64 s6, vcc", 4), ("v_mov_b32_e32 v6, s6", 4), ("v_cndmask_b32_e64 v7, 0, 1, vcc", 8),
        ("v_mov_b32_e32 v8, 0", 4), ("v_mov_b32_e32 v9, 0", 4), ("v_lshlrev_b32_e32 v12, 4, v0", 4),
        ("v_add_nc_u32_e32 v12, 0x1000, v12", 8), ("buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen", 8),
        ("v_mov_b32_e32 v5, 0", 4), ("exp pos0 v5, v5, v5, v5 done", 8), ("exp param0 v5, v5, v5, v5", 8),
        ("s_endpgm", 4)]


class VsComputeTests(unittest.TestCase):
    def test_module_is_an_emulated_compute_shader(self):
        spv, gen = translate_vsc("vs_w64_c_bcnt_export", BCNT)
        self.assertTrue(gen.vsc and gen.emulate)
        self.assertIn("OpEntryPoint GLCompute %main", spv)
        self.assertIn("OpExecutionMode %main LocalSize 64 1 1", spv)
        self.assertNotIn("OpGroupNonUniform", spv)       # cross-lane data goes through workgroup memory
        self.assertNotIn("BuiltIn Position", spv)
        self.assertEqual(gen.vsc_layout["vertex_stride"], 32)     # position + param0
        self.assertEqual(gen.vsc_layout["params"], {0: 16})

    def test_lane_count_is_the_full_waves(self):
        """64 vertices: every lane counts the 16 lanes of its 64-lane wave with x % 4 == 1 (2 x 32 would give 8)."""
        spv, _ = translate_vsc("vs_w64_c_bcnt_export", BCNT)
        mem = run(spv, 64)
        for v in range(64):
            rec = [mem.get(0x1000 + 16 * v + 4 * k, 0) for k in range(4)]
            self.assertEqual(rec, [16, int(v % 4 == 1), 0, 0], f"vertex {v}")

    def test_partial_last_wave(self):
        """40 vertices: lanes 40..63 are not in EXEC, so they neither count nor store (the count is 10)."""
        spv, _ = translate_vsc("vs_w64_c_bcnt_export", BCNT)
        mem = run(spv, 40)
        for v in range(64):
            rec = [mem.get(0x1000 + 16 * v + 4 * k) for k in range(4)]
            if v < 40:
                self.assertEqual(rec, [10, int(v % 4 == 1), 0, 0], f"vertex {v}")
            else:
                self.assertEqual(rec, [None] * 4, f"vertex {v} past the count stored")

    def test_exports_land_in_the_vertex_slot(self):
        """Workgroup (1, 2) of a 3 x 4 grid: vertex x * 64 + i of instance 2 writes slot 2 * 192 + 64 + i."""
        spv, gen = translate_vsc("vs_w64_c_bcnt_export", BCNT)
        stride = gen.vsc_layout["vertex_stride"]
        mem = run(spv, 64 * 3, wg=(1, 2, 0), num_wg=(3, 4, 1))
        for i in range(64):
            base = OUT + (2 * 192 + 64 + i) * stride
            self.assertEqual([mem.get(base + 4 * k) for k in range(8)], [0] * 8, f"slot of lane {i}")
            # v0 is the vertex in the draw (x * 64 + i), so the store went to vertex 64 + i's record
            self.assertEqual(mem.get(0x1000 + 16 * (64 + i)), 16)

    def test_partial_wave_writes_every_slot_of_the_wave(self):
        """40 vertices: lanes 40..63 are inactive but still store their slot; their exports were never written, so
        the slots hold the null outputs (all zero words), and lanes 0..39 hold their own exports."""
        spv, gen = translate_vsc("vs_w64_c_bcnt_export", BCNT)
        stride = gen.vsc_layout["vertex_stride"]
        mem = run(spv, 40)
        for i in range(64):
            base = OUT + i * stride
            self.assertEqual([mem.get(base + 4 * k) for k in range(stride // 4)], [0] * (stride // 4), f"slot {i}")

    def test_refuses_a_wave32_vertex_shader(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            lines = [("v_mov_b32_e32 v5, 0", 4), ("exp pos0 v5, v5, v5, v5 done", 8), ("s_endpgm", 4)]
            prev = rdna2spirv.VS_COMPUTE_PASS
            rdna2spirv.VS_COMPUTE_PASS = True
            try:
                with tempfile.TemporaryDirectory() as td:
                    path = os.path.join(td, "vs_w32.s")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("; mattr: -wavefrontsize64,+wavefrontsize32\n" +
                                "\n".join(f"\t{t:52s} ; encoding: [{','.join(['0x00'] * n)}]" for t, n in lines))
                    rdna2spirv.translate(path)
            finally:
                rdna2spirv.VS_COMPUTE_PASS = prev


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class VsComputeCliTests(unittest.TestCase):
    """rdna2spirv.py --vs-compute as make_game.py would run it: the module next to the VS and its index entry."""

    def translate_dir(self, td, programs):
        disasm, out = os.path.join(td, "disasm"), os.path.join(td, "spirv")
        os.mkdir(disasm)
        for name, insns in programs.items():
            lines = [f"; {name}  synthetic", "; mattr: +wavefrontsize64,-wavefrontsize32"]
            lines += [f"\t{t:52s} ; encoding: [{','.join(['0x00'] * n)}]" for t, n in insns]
            with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        tool = os.path.join(os.path.dirname(rdna2spirv.__file__), "rdna2spirv.py")
        r = subprocess.run([sys.executable, "-I", tool, disasm, "--out", out, "--vs-compute"], capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            return out, json.load(f)

    def test_tier_c_vs_gets_a_validated_module_and_its_own_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            out, index = self.translate_dir(td, {"vs_w64_c_bcnt_export": BCNT})
            e = index["vs_w64_c_bcnt_export"]
            self.assertEqual(e.get("wave64_graphics"), "C")
            self.assertEqual(e.get("subgroup_size"), 64)                       # the VS itself is unchanged
            vc = e["vs_compute"]
            self.assertTrue(os.path.isfile(os.path.join(out, vc["module"])))
            self.assertEqual((vc["stage"], vc["local_size"], vc["subgroup_size"], vc["wave64_exchange"]),
                             ("cs", [64, 1, 1], None, "workgroup"))
            self.assertGreater(vc["lds_bytes"], 0)                              # the emulation's %x_buf
            self.assertIn("PhysicalStorageBufferAddresses", vc["capabilities"])
            self.assertNotIn("GroupNonUniform", vc["capabilities"])
            self.assertEqual((vc["vertex_stride"], vc["params"], vc["push_constant_bytes"]), (32, {"0": 16}, 156))
            v = subprocess.run(["spirv-val", "--target-env", "vulkan1.3", os.path.join(out, vc["module"])],
                               capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stderr)

    def test_refusal_is_recorded_and_writes_no_module(self):
        """build_vs_compute on a VS the compute pass refuses (wave32 here): vs_compute_error, no module written, and
        VS_COMPUTE_PASS is off again for the next shader."""
        wave32 = [("v_mov_b32_e32 v5, 0", 4), ("exp pos0 v5, v5, v5, v5 done", 8), ("s_endpgm", 4)]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "vs_w32.s")
            with open(path, "w", encoding="utf-8") as f:
                f.write("; mattr: -wavefrontsize64,+wavefrontsize32\n" +
                        "\n".join(f"\t{t:52s} ; encoding: [{','.join(['0x00'] * n)}]" for t, n in wave32) + "\n")

            class Args:
                out, assembler, no_validate, no_asm = td, "spirv-as", False, False

            m = {}
            rdna2spirv.build_vs_compute(path, "vs_w32", Args, m)
            self.assertIn("vs_compute_error", m)
            self.assertNotIn("vs_compute", m)
            self.assertFalse(os.path.exists(os.path.join(td, "vs_w32.vs_compute.spv")))
            self.assertFalse(rdna2spirv.VS_COMPUTE_PASS)


if __name__ == "__main__":
    unittest.main()
