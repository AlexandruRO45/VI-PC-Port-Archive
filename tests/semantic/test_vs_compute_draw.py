"""--vs-compute end to end on the CPU: the compute prepass writes the slots, tools/vs_compute_vs.py's vertex shader reads
them back (issue #94, atoms 1 and 2).

The wave64 VS below exports a lane count (tier C) next to v0 and v3. The prepass runs each workgroup of a 2 x 2 grid
(two waves of vertices, two instances) through run_emulated_workgroup into one memory; the draw shader then runs per
vertex through spirv_wave_interp with VertexIndex / InstanceIndex, and must export what the original VS exports in a
64-lane wave: pos = (v, n, 16, 16), param1 = (16, v, n, v) as bit patterns.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import support  # noqa: E402,F401
import rdna2spirv  # noqa: E402
import vs_compute_vs  # noqa: E402
from spirv_wave_interp import run_wave  # noqa: E402
from test_vs_compute import OUT, USER, run, translate_vsc  # noqa: E402

VS = [("v_and_b32_e32 v2, 63, v0", 4), ("v_and_b32_e32 v4, 3, v2", 4), ("v_cmp_eq_u32_e32 vcc, 1, v4", 4),
      ("s_bcnt1_i32_b64 s6, vcc", 4), ("v_mov_b32_e32 v6, s6", 4), ("v_mov_b32_e32 v7, v0", 4),
      ("v_mov_b32_e32 v8, v3", 4), ("exp pos0 v7, v8, v6, v6 done", 8), ("exp param1 v6, v7, v8, v0", 8),
      ("s_endpgm", 4)]


def words(v):
    return tuple(int(getattr(x, "bits", x)) for x in v)


class VsComputeDrawTests(unittest.TestCase):
    def test_draw_reads_back_what_the_wave_exported(self):
        spv, gen = translate_vsc("vs_w64_c_draw", VS)
        lay = dict(gen.vsc_layout)
        lay["params"] = {str(k): v for k, v in lay["params"].items()}
        self.assertEqual(lay["params"], {"1": 16})
        count, nx, instances = 128, 2, 2
        mem = {}
        for y in range(instances):
            for x in range(nx):
                mem.update(run(spv, count, wg=(x, y, 0), num_wg=(nx, instances, 1)))
        draw = vs_compute_vs.vs_spvasm(lay)
        row = nx * 64
        pcs = [0] * USER + [0, 0, 0, 0, OUT & 0xFFFFFFFF, OUT >> 32, row]
        for n in range(instances):
            for first in (0, 64):
                res = run_wave(draw, wave=64, push_constants=pcs, memory=dict(mem),
                               builtins={"VertexIndex": lambda lane, f=first: f + lane,
                                         "InstanceIndex": lambda lane, i=n: i})
                for lane, out in enumerate(res["outputs"]):
                    v = first + lane
                    self.assertEqual(words(out["%out_pos"]), (v, n, 16, 16), f"pos of vertex {v}, instance {n}")
                    self.assertEqual(words(out["%out1"]), (16, v, n, v), f"param1 of vertex {v}, instance {n}")

    def test_draw_module_validates_shape(self):
        lay = rdna2spirv.vs_compute_layout([0, 3])
        lay["params"] = {str(k): v for k, v in lay["params"].items()}
        asm = vs_compute_vs.vs_spvasm(lay)
        self.assertIn("OpEntryPoint Vertex %main", asm)
        self.assertIn("OpDecorate %out0 Location 0", asm)
        self.assertIn("OpDecorate %out3 Location 3", asm)
        with self.assertRaises(ValueError):
            vs_compute_vs.vs_spvasm(dict(lay, vertex_stride=64))


class VsComputeDrawEdgeTests(unittest.TestCase):
    """The cases a broken slot or base calculation would show (review on #8487)."""

    def test_partial_wave_and_draw_offsets(self):
        """40 vertices of one instance, drawn with BaseVertex 100 and BaseInstance 7 (VertexIndex 100.., InstanceIndex
        7): the draw VS reads slot (VertexIndex - BaseVertex) + (InstanceIndex - BaseInstance) * row, so vertex v reads
        the slot the prepass wrote for v. The wave has 40 active lanes: 10 of them with v % 4 == 1. The slots of lanes
        40..63, pre-filled with a sentinel, are overwritten with the null outputs."""
        spv, gen = translate_vsc("vs_w64_c_draw", VS)
        stride = gen.vsc_layout["vertex_stride"]
        sentinel = {OUT + a: 0xDEADBEEF for a in range(0, 64 * stride, 4)}
        mem = dict(sentinel)
        mem.update(run(spv, 40, wg=(0, 0, 0), num_wg=(1, 1, 1)))
        for i in range(40, 64):
            self.assertEqual([mem[OUT + i * stride + 4 * k] for k in range(stride // 4)], [0] * (stride // 4),
                             f"inactive slot {i}")
        lay = dict(gen.vsc_layout)
        lay["params"] = {str(k): v for k, v in lay["params"].items()}
        draw = vs_compute_vs.vs_spvasm(lay)
        pcs = [0] * USER + [0, 0, 0, 0, OUT & 0xFFFFFFFF, OUT >> 32, 64]
        # The interpreter runs whole subgroups: lanes 40..63 stand for no vertex of the draw and read the null slots.
        res = run_wave(draw, wave=64, push_constants=pcs, memory=dict(mem),
                       builtins={"VertexIndex": lambda lane: 100 + lane, "BaseVertex": lambda lane: 100,
                                 "InstanceIndex": lambda lane: 7, "BaseInstance": lambda lane: 7})
        for v, out in enumerate(res["outputs"]):
            if v < 40:
                self.assertEqual(words(out["%out_pos"]), (v, 0, 10, 10), f"pos of vertex {v}")
                self.assertEqual(words(out["%out1"]), (10, v, 0, v), f"param1 of vertex {v}")
            else:
                self.assertEqual(words(out["%out_pos"]) + words(out["%out1"]), (0,) * 8, f"null slot {v}")

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_build_for_index_writes_validated_modules(self):
        """tools/vs_compute_vs.py's folder mode: one validated <name>.vs_compute_draw.spv per vs_compute entry."""
        with tempfile.TemporaryDirectory() as td:
            index = {}
            for name, params in (("vs_a", []), ("vs_b", [0]), ("vs_c", [1, 2, 5])):
                lay = rdna2spirv.vs_compute_layout(params)
                lay["params"] = {str(k): v for k, v in lay["params"].items()}
                index[name] = {"stage": "vs", "vs_compute": lay}
            index["vs_plain"] = {"stage": "vs"}
            with open(os.path.join(td, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)
            built, failed = vs_compute_vs.build_for_index(td)
            self.assertEqual((sorted(built), failed), (["vs_a", "vs_b", "vs_c"], []))
            for name in built:
                path = os.path.join(td, name + vs_compute_vs.SUFFIX + ".spv")
                v = subprocess.run(["spirv-val", "--target-env", "vulkan1.3", path], capture_output=True, text=True)
                self.assertEqual(v.returncode, 0, f"{name}: {v.stderr}")
            self.assertFalse(os.path.exists(os.path.join(td, "vs_plain" + vs_compute_vs.SUFFIX + ".spv")))


if __name__ == "__main__":
    unittest.main()
