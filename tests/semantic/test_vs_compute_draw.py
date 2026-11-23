"""--vs-compute end to end on the CPU: the compute prepass writes the slots, tools/vs_compute_vs.py's vertex shader reads
them back (issue #94, atoms 1 and 2).

The wave64 VS below exports a lane count (tier C) next to v0 and v3. The prepass runs each workgroup of a 2 x 2 grid
(two waves of vertices, two instances) through run_emulated_workgroup into one memory; the draw shader then runs per
vertex through spirv_wave_interp with VertexIndex / InstanceIndex, and must export what the original VS exports in a
64-lane wave: pos = (v, n, 16, 16), param1 = (16, v, n, v) as bit patterns.
"""
import os
import sys
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


if __name__ == "__main__":
    unittest.main()
