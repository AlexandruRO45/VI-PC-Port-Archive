"""vk_pipeline_check: a --mesh-compute mesh entry is created as a compute pipeline, not skipped.

    python -I tests/test_vk_pipeline_check_mesh_compute.py

rdna2spirv.py --mesh-compute emits a mesh shader as a GLCompute module and records its buffer layout under
"mesh_compute" in index.json. vk_pipeline_check used to skip every such entry as "no VK_EXT_mesh_shader" on the GPUs
the flag exists for (MoltenVK has no mesh shaders), so the pipelines stage never created a third of the corpus there.
The module here is a hand-built empty compute shader; no game data. Needs vk_pipeline_check and a Vulkan device, and
skips without either. Set LEONIDA_VK_PIPELINE_CHECK to use a binary elsewhere.
"""
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_binary(root=ROOT):
    """LEONIDA_VK_PIPELINE_CHECK, else the newest vk_pipeline_check under runtime/build: a single-config build in
    runtime/build or one folder below it (linux/, Ninja on Windows with .exe), or a multi-config one in a
    configuration folder (Release/, RelWithDebInfo/, ...)."""
    env = os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
    if env:
        return Path(env) if Path(env).is_file() else None
    build = Path(root) / "runtime" / "build"
    found = [p for name in ("vk_pipeline_check", "vk_pipeline_check.exe")
             for p in (build / name, *build.glob(f"*/{name}")) if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def empty_compute_module(local_x=128):
    """SPIR-V 1.0: OpEntryPoint GLCompute "main", LocalSize local_x 1 1, a function that returns."""
    words = [0x07230203, 0x00010000, 0, 5, 0,
             (2 << 16) | 17, 1,                                   # OpCapability Shader
             (3 << 16) | 14, 0, 1,                                # OpMemoryModel Logical GLSL450
             (5 << 16) | 15, 5, 1, 0x6E69616D, 0,                 # OpEntryPoint GLCompute %1 "main"
             (6 << 16) | 16, 1, 17, local_x, 1, 1,                # OpExecutionMode %1 LocalSize
             (2 << 16) | 19, 2,                                   # %2 = OpTypeVoid
             (3 << 16) | 33, 3, 2,                                # %3 = OpTypeFunction %2
             (5 << 16) | 54, 2, 1, 0, 3,                          # %1 = OpFunction %2 None %3
             (2 << 16) | 248, 4,                                  # %4 = OpLabel
             (1 << 16) | 253,                                     # OpReturn
             (1 << 16) | 56]                                      # OpFunctionEnd
    return struct.pack(f"<{len(words)}I", *words)


# What rdna2spirv.py writes for a --mesh-compute mesh shader: the buffer layout, not a bool.
LAYOUT = {"slot_bytes": 12304, "vertex_stride": 80, "primitive_stride": 16, "max_vertices": 128,
          "max_primitives": 128}


class BinaryLookup(unittest.TestCase):
    """find_binary() on a hand-made build folder; no Vulkan needed."""

    def test_single_and_multi_config_layouts(self):
        saved = os.environ.pop("LEONIDA_VK_PIPELINE_CHECK", None)
        self.addCleanup(lambda: saved is None or os.environ.__setitem__("LEONIDA_VK_PIPELINE_CHECK", saved))
        for rel in ("runtime/build/vk_pipeline_check.exe", "runtime/build/RelWithDebInfo/vk_pipeline_check.exe",
                    "runtime/build/linux/vk_pipeline_check", "runtime/build/vk_pipeline_check"):
            with self.subTest(rel=rel), tempfile.TemporaryDirectory() as tmp:
                self.assertIsNone(find_binary(tmp))
                path = Path(tmp) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
                self.assertEqual(find_binary(tmp), path)

    def test_newest_build_wins(self):
        saved = os.environ.pop("LEONIDA_VK_PIPELINE_CHECK", None)
        self.addCleanup(lambda: saved is None or os.environ.__setitem__("LEONIDA_VK_PIPELINE_CHECK", saved))
        with tempfile.TemporaryDirectory() as tmp:
            old, new = (Path(tmp) / "runtime" / "build" / d / "vk_pipeline_check.exe" for d in ("Release", "RelWithDebInfo"))
            for i, path in enumerate((old, new)):
                path.parent.mkdir(parents=True)
                path.write_bytes(b"")
                os.utime(path, (1_000_000 + i, 1_000_000 + i))
            self.assertEqual(find_binary(tmp), new)


class MeshComputeEntries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built (cmake -S runtime -B runtime/build/linux)")

    def run_tool(self, index, *args):
        with tempfile.TemporaryDirectory() as d:
            for name in index:
                Path(d, name + ".spv").write_bytes(empty_compute_module())
            Path(d, "index.json").write_text(json.dumps(index), encoding="utf-8")
            r = subprocess.run([str(self.exe), d, "--jobs", "1", *args], capture_output=True, text=True, timeout=300)
        if "device:" not in r.stderr + r.stdout:
            self.skipTest("no Vulkan device: " + (r.stderr.strip().splitlines() or ["(no output)"])[-1])
        return r

    def require_compute_pipelines(self):
        """Skip a test that needs a pipeline to be created when the same module fails as an ordinary cs too.

        An ms-compute entry gets the checker's compute pipeline layout. Where the device refuses that layout itself
        (Intel UHD 770 on Windows, 101.7088: the per-stage sampler and image arrays exceed
        maxPerStageDescriptor*, so vkCreateComputePipelines returns -3 for every cs on main as well), the
        ms-compute result says nothing about this change. The control runs once per class."""
        cls = type(self)
        if not hasattr(cls, "_cs_control"):
            r = self.run_tool({"cs": {"stage": "cs", "subgroup_size": None}})
            cls._cs_control = r.stdout + r.stderr
        if "cs pipelines: 1 ok, 0 failed, 0 skipped" not in cls._cs_control:
            lines = [ln.strip() for ln in cls._cs_control.splitlines()]
            why = [ln for ln in lines if ln.startswith(("PIPELINE FAIL", "MODULE FAIL"))] or \
                  [ln for ln in lines if ln.startswith("cs pipelines:")] or ["no cs pipeline was reported"]
            self.skipTest("this device can't create an ordinary cs pipeline with the checker's layout either: "
                          + why[0])

    def test_mesh_compute_entry_is_a_compute_pipeline(self):
        self.require_compute_pipelines()
        r = self.run_tool({"mc": {"stage": "ms", "mesh_compute": LAYOUT, "subgroup_size": None}})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ms-compute: 1 modules ok, 0 failed", r.stdout)
        self.assertIn("ms-compute pipelines: 1 ok, 0 failed, 0 skipped", r.stdout)
        self.assertNotIn("no VK_EXT_mesh_shader", r.stdout)

    def test_counted_apart_from_compute_shaders(self):
        self.require_compute_pipelines()
        r = self.run_tool({"cs": {"stage": "cs", "subgroup_size": None},
                           "mc": {"stage": "ms", "mesh_compute": LAYOUT, "subgroup_size": None}})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cs pipelines: 1 ok, 0 failed, 0 skipped", r.stdout)
        self.assertIn("ms-compute pipelines: 1 ok, 0 failed, 0 skipped", r.stdout)

    def test_compute_only_includes_it_and_graphics_only_leaves_it_out(self):
        self.require_compute_pipelines()
        index = {"mc": {"stage": "ms", "mesh_compute": LAYOUT, "subgroup_size": None}}
        r = self.run_tool(index, "--compute-only")
        self.assertIn("ms-compute pipelines: 1 ok", r.stdout)
        r = self.run_tool(index, "--graphics-only")
        self.assertIn("checking 0 shaders", r.stdout + r.stderr)

    def test_workgroup_memory_is_checked_like_a_compute_shader(self):
        # BUILD-546's check of lds_bytes against maxComputeSharedMemorySize applies: the module is a compute pipeline.
        huge = 1 << 30
        r = self.run_tool({"mc": {"stage": "ms", "mesh_compute": LAYOUT, "subgroup_size": None, "lds_bytes": huge}})
        self.assertIn("MODULE FAIL mc (workgroup memory", r.stdout)
        self.assertIn("ms-compute: 0 modules ok, 1 failed", r.stdout)
        # A real mesh entry isn't held to the compute limit. With VK_EXT_mesh_shader it meets maxMeshSharedMemorySize
        # instead; without it the limit is 0 and the entry is skipped for the missing extension.
        r = self.run_tool({"m": {"stage": "ms", "lds_bytes": huge}})
        self.assertNotIn("maxComputeSharedMemorySize", r.stdout)
        if "workgroup memory" in r.stdout:
            self.assertIn("MODULE FAIL m (workgroup memory 1073741824 bytes exceeds maxMeshSharedMemorySize", r.stdout)
        else:
            self.assertIn("no VK_EXT_mesh_shader", r.stdout)

    def test_push_constant_layout_covers_the_index_size(self):
        self.require_compute_pipelines()
        # rdna2spirv.py records 152 bytes for a --mesh-compute module (the buffer address after the 144-byte user
        # SGPRs). The entry is built as compute, so it has to find the 152-byte layout, next to a 144-byte cs.
        r = self.run_tool({"mc": {"stage": "ms", "mesh_compute": LAYOUT, "subgroup_size": None,
                                  "push_constant_bytes": 152},
                           "cs": {"stage": "cs", "subgroup_size": None, "push_constant_bytes": 144}})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ms-compute pipelines: 1 ok, 0 failed, 0 skipped", r.stdout)
        self.assertIn("cs pipelines: 1 ok, 0 failed, 0 skipped", r.stdout)

    def test_push_constants_over_the_device_limit_are_skipped(self):
        r = self.run_tool({"mc": {"stage": "ms", "mesh_compute": LAYOUT, "subgroup_size": None,
                                  "push_constant_bytes": 1 << 20}})
        self.assertIn("needs 1048576 push-constant bytes", r.stdout)
        self.assertIn("ms-compute pipelines: 0 ok, 0 failed, 1 skipped", r.stdout)

    def test_bool_and_null_forms(self):
        self.require_compute_pipelines()
        r = self.run_tool({"mc": {"stage": "ms", "mesh_compute": True, "subgroup_size": None}})
        self.assertIn("ms-compute pipelines: 1 ok", r.stdout)
        # Absent or null: a real mesh shader as before (created with VK_EXT_mesh_shader, or skipped without it),
        # never counted as ms-compute. An empty compute module isn't a valid mesh shader, so only the label is checked.
        for value in (None, False):
            with self.subTest(mesh_compute=value):
                r = self.run_tool({"m": {"stage": "ms", "mesh_compute": value}})
                self.assertNotIn("ms-compute", r.stdout)


if __name__ == "__main__":
    unittest.main()
