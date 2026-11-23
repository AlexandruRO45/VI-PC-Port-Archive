"""Four-workgroup allocation test on the native and lowered vk_ms_sweep paths.

Set VK_MS_SWEEP to the built executable and LLVM_MC to an AMDGPU llvm-mc.
Run python -I tests/test_ms_sweep_gpu.py on a mesh-capable Vulkan device.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools")]
import ms_sweep
import run_gpu_tests


@unittest.skipUnless(os.environ.get("VK_MS_SWEEP"), "set VK_MS_SWEEP to the sweep executable")
class MeshAllocation(unittest.TestCase):
    def test_conditional_allocation_survives_rounds(self):
        name = "ms_wave0_alloc_rounds"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stage = out / "stage"
            disasm = stage / "disasm"
            disasm.mkdir(parents=True)
            fixture = ROOT / "runtime/test/shaders" / (name + ".asm")
            run_gpu_tests.assemble_to_s(os.environ.get("LLVM_MC"), fixture, disasm / (name + ".s"), name)
            psv = run_gpu_tests.make_ms_psv0(128, 64, 64)
            (stage / (name + ".dxbc")).write_bytes(run_gpu_tests.make_dxbc([(b"PSV0", psv)]))
            (out / "sample.json").write_text(json.dumps([{"name": name, "stratum": []}]), encoding="utf-8",
                                             newline="\n")
            args = ["--out", tmp]
            ms_sweep.main(["translate", *args])
            for folder, mode in (("native", "native"), ("fb", "compute")):
                subprocess.run(["spirv-val", "--target-env", "vulkan1.2", out / folder / (name + ".spv")], check=True)
                ms_sweep.main(["run", *args, "--mode", mode, "--exe", os.environ["VK_MS_SWEEP"]])
            ms_sweep.main(["compare", *args])
            row, = json.loads((out / "classes_z.json").read_text(encoding="utf-8"))
            self.assertEqual((row["native"], row["fallback"]), (0, 0), row)
            self.assertEqual(row["native_prims"], ms_sweep.GROUPS)
            layout = json.loads((out / "fb/index.json").read_text(encoding="utf-8"))[name]["mesh_compute"]
            data = (out / "compute_z/0.raw").read_bytes()
            pixels = ms_sweep.W * ms_sweep.H
            headers = [struct.unpack_from("<II", data, pixels * 16 + group * layout["slot_bytes"])
                       for group in range(ms_sweep.GROUPS)]
            self.assertEqual(headers, [(3, 1)] * ms_sweep.GROUPS)
            self.assertEqual((row["cov_native"], row["cov_fallback"], row["diff_px"]), (pixels, pixels, 0))


if __name__ == "__main__":
    unittest.main()
