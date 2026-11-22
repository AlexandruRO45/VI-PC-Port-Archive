"""Exercise vk_pipeline_check --graphics-manifest with driver-free input cases and synthetic Vulkan shaders.

Set LEONIDA_VK_PIPELINE_CHECK to the built executable. The positive driver tests also need glslangValidator.
No game data is used. The verifier builds its tiny fixtures in a temporary directory.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GLSL = {
    "vert": """#version 450
layout(location=0) in vec4 attr;
layout(location=0) out vec4 value;
void main() {
    gl_Position = vec4(0, 0, 0, 1);
    value = attr;
}
""",
    "geom": """#version 450
layout(lines) in;
layout(triangle_strip, max_vertices=3) out;
layout(location=0) in vec4 value_in[];
layout(location=0) out vec4 value;
void main() {
    for (int i=0; i<3; ++i) {
        gl_Position = gl_in[i % 2].gl_Position;
        value = value_in[i % 2];
        EmitVertex();
    }
    EndPrimitive();
}
""",
    "frag": """#version 450
layout(location=0) in vec4 value;
layout(location=0) out vec4 color;
void main() {
    gl_FragDepth = 0.5;
    color = value;
}
""",
    "mesh": """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x=1) in;
layout(triangles, max_vertices=3, max_primitives=1) out;
layout(location=0) out vec4 value[];
void main() {
    SetMeshOutputsEXT(3, 1);
    for (int i=0; i<3; ++i) {
        gl_MeshVerticesEXT[i].gl_Position = vec4(0, 0, 0, 1);
        value[i] = vec4(1);
    }
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
""",
}
PIXEL_STEM = ".pixel~clip!(&)"
INDEX = {
    "vert": {"stage": "vs", "vertex_inputs": [{"location": 0, "type": "float"}]},
    "geom": {"stage": "gs"},
    PIXEL_STEM: {"stage": "ps", "outputs": [0], "writes_depth": True},
    "mesh": {"stage": "ms"},
    "absent_blob": {"stage": "ps"},
    "wrong_model": {"stage": "ps"},
}
PAIR = {"name": "Paar 🎨", "vs": "vert", "ps": {"shader": PIXEL_STEM, "entry": "pixel"}}
TRIPLE = {"name": "geometry", "vs": "vert", "gs": "geom", "ps": PAIR["ps"]}
DISCARD = {"name": "geometry-only", "vs": "vert", "gs": "geom"}


def find_tool():
    override = os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
    if override:
        return Path(override)
    build = ROOT / "runtime" / "build"
    return next((p for folder in (build, build / "linux", build / "RelWithDebInfo", build / "Release")
                 for name in ("vk_pipeline_check", "vk_pipeline_check.exe")
                 if (p := folder / name).is_file()), None)


def manifest(*rows):
    return json.dumps({"version": 1, "pipelines": rows})


class GraphicsManifestTest(unittest.TestCase):
    def setUp(self):
        self.exe = find_tool()
        if not self.exe or not self.exe.is_file():
            self.skipTest("vk_pipeline_check isn't built")
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.folder = Path(scratch.name)
        self.index = self.folder / "index.json"
        self.index.write_text(json.dumps(INDEX), encoding="utf-8", newline="\n")
        self.path = self.folder / "chains.json"

    def run_tool(self, doc, *args):
        self.path.write_text(doc, encoding="utf-8", newline="\n")
        return subprocess.run([str(self.exe), str(self.folder), "--jobs", "2",
                               "--graphics-manifest", str(self.path), *map(str, args)],
                              capture_output=True, encoding="utf-8", errors="replace", timeout=90)

    def test_invalid_manifests_fail_before_vulkan_setup(self):
        for label, data, message in (
            ("missing pipelines", '{"version":1}', "nonempty"),
            ("empty list", '{"pipelines":[]}', "pipeline"),
            ("bad version", '{"version":2,"pipelines":[]}', "version 1"),
            ("boolean version", '{"version":true,"pipelines":[]}', "version 1"),
            ("trailing data", manifest(PAIR) + " {}", "trailing bytes"),
            ("duplicate names", manifest(PAIR, PAIR), "duplicate pipeline name"),
            ("duplicate fields", '{"pipelines":[{"name":"x","name":"y"}]}', "duplicate pipeline field"),
            ("missing fragment", manifest({"name": "bad", "vs": "vert"}), "expected vs+ps"),
            ("mesh plus vertex", manifest(dict(PAIR, ms="mesh")), "expected vs+ps"),
            ("unknown key", manifest(dict(PAIR, cs="anything")), "unknown pipeline field"),
            ("path traversal", manifest(dict(PAIR, vs="../vert")), "without directories"),
            ("bad name", manifest(dict(PAIR, name="line\nbreak")), "printable"),
            ("unknown shader", manifest(dict(PAIR, ps="missing")), "missing from index.json"),
            ("wrong stage", manifest(dict(PAIR, ps="geom")), "not a ps shader"),
            ("empty entry", manifest(dict(PAIR, ps={"shader": PIXEL_STEM, "entry": ""})), "entry point name"),
        ):
            with self.subTest(label=label):
                proc = self.run_tool(data)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("device:", proc.stderr)

    def test_incompatible_options_fail_before_vulkan_setup(self):
        for args in (("--isolate",), ("--compute-only",)):
            with self.subTest(args=args):
                proc = self.run_tool(manifest(PAIR), *args)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("requires graphics checks", proc.stderr)
        for option in ("--inflight-file", "--skip-file"):
            with self.subTest(option=option):
                proc = self.run_tool(manifest(PAIR), option, self.folder / "recovery.txt")
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("cannot be combined", proc.stderr)

    def test_output_cannot_replace_manifest_or_index(self):
        body = manifest(PAIR)
        for output, message in ((self.path, "would overwrite input"), (self.index, "must not overwrite index.json")):
            with self.subTest(output=output):
                proc = self.run_tool(body, output)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(message, proc.stderr)
        self.assertEqual(self.path.read_text(encoding="utf-8"), body)
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), INDEX)

    def make_shaders(self, *names, sources=GLSL):
        glslang = shutil.which("glslangValidator")
        if not glslang:
            self.skipTest("glslangValidator not installed")
        for name in names:
            src = self.folder / ("fixture." + name)
            src.write_text(sources[name], encoding="utf-8", newline="\n")
            entry = ["--source-entrypoint", "main", "-e", "pixel"] if name == "frag" else []
            stem = PIXEL_STEM if name == "frag" else name
            subprocess.run([glslang, "-V", "--target-env", "vulkan1.2", *entry,
                            "-o", str(self.folder / (stem + ".spv")), str(src)],
                           check=True, capture_output=True, timeout=30)

    def gpu_run(self, *rows, args=()):
        proc = self.run_tool(manifest(*rows), *args)
        if any(reason in proc.stderr for reason in ("no Vulkan physical devices found",
                   "could not load vkGetInstanceProcAddr", "could not load libMoltenVK")):
            self.skipTest("no usable Vulkan loader/device")
        return proc

    def test_driver_compiles_exact_chains_and_reloads_cache(self):
        self.make_shaders("vert", "geom", "frag")
        self.path.write_text(manifest(PAIR, TRIPLE, DISCARD), encoding="utf-8", newline="\n")
        static = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "shader_link_check.py"),
                                 "batch", str(self.path), "--shader-dir", str(self.folder)],
                                capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        self.assertEqual(static.returncode, 0, static.stdout + static.stderr)
        cache = self.folder / "session.cache"
        for run in range(2):
            proc = self.gpu_run(PAIR, TRIPLE, DISCARD, args=(cache, "--cache-feedback"))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            if "no geometryShader" in proc.stdout:
                self.skipTest("device lacks geometryShader")
            for row in (PAIR, TRIPLE, DISCARD):
                self.assertIn("GRAPHICS OK " + row["name"], proc.stdout)
            self.assertIn("graphics pipelines: 3 ok, 0 failed, 0 skipped", proc.stdout)
            self.assertTrue(cache.stat().st_size)
            if run:
                self.assertIn("seeded the pipeline cache", proc.stderr)

    def test_chain_combines_stage_layout_requirements(self):
        frag = GLSL["frag"].replace("void main()", """layout(set=1, binding=0) uniform Tail { vec4 scale; };
void main()""").replace("color = value;", "color = value * scale;")
        self.make_shaders("vert", "frag", sources={**GLSL, "frag": frag})
        metadata = {**INDEX,
                    "vert": dict(INDEX["vert"], push_constant_bytes=128),
                    PIXEL_STEM: dict(INDEX[PIXEL_STEM], push_constant_bytes=64, pc_tail={"bytes": 16})}
        self.index.write_text(json.dumps(metadata), encoding="utf-8")
        proc = self.gpu_run(PAIR)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("GRAPHICS OK " + PAIR["name"], proc.stdout)

    def test_slice_order_and_byte_limit(self):
        self.make_shaders("vert", "frag")
        second = dict(PAIR, name="second")
        largest_module = max((self.folder / (shader + ".spv")).stat().st_size
                             for shader in ("vert", PIXEL_STEM))
        proc = self.gpu_run(PAIR, second, args=("--start", "1", "--count", "1",
                                               "--large-module-bytes", largest_module, "--large-parallel", "1"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("GRAPHICS OK second", proc.stdout)
        self.assertNotIn("GRAPHICS OK " + PAIR["name"], proc.stdout)
        self.assertIn("large chains checked exclusively: 1", proc.stdout)
        proc = self.gpu_run(PAIR, args=("--max-bytes", "20"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("0 ok, 0 failed, 1 skipped", proc.stdout)

    def test_one_failed_chain_does_not_stop_other_chains(self):
        self.make_shaders("vert", "frag")
        shutil.copyfile(self.folder / "vert.spv", self.folder / "wrong_model.spv")
        wrong = {"name": "wrong-execution-model", "vs": "vert", "ps": "wrong_model"}
        missing = {"name": "missing-file", "vs": "vert", "ps": "absent_blob"}
        wrong_entry = dict(PAIR, name="wrong-entry", ps={"shader": PIXEL_STEM, "entry": "wrong"})
        proc = self.gpu_run(wrong, missing, wrong_entry, PAIR)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("wrong-execution-model: entry point not found", proc.stdout)
        self.assertIn("wrong-entry: entry point not found", proc.stdout)
        self.assertIn("missing-file: cannot read file size", proc.stdout)
        self.assertIn("GRAPHICS OK " + PAIR["name"], proc.stdout)
        self.assertIn("graphics pipelines: 1 ok, 3 failed, 0 skipped", proc.stdout)

    def test_mesh_chain(self):
        self.make_shaders("mesh", "frag")
        proc = self.gpu_run({"name": "native-mesh", "ms": "mesh", "ps": PAIR["ps"]})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        if "no VK_EXT_mesh_shader" in proc.stdout:
            self.assertIn("graphics pipelines: 0 ok, 0 failed, 1 skipped", proc.stdout)
        else:
            self.assertIn("GRAPHICS OK native-mesh", proc.stdout)
            self.assertIn("graphics pipelines: 1 ok, 0 failed, 0 skipped", proc.stdout)


if __name__ == "__main__":
    unittest.main()
