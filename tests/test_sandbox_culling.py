"""Synthetic sandbox draws; set LEONIDA_SANDBOX_BUILD to a build containing vk_sandbox for GPU checks."""
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sandbox


VERTEX = """#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(buffer_reference, std430, buffer_reference_align=4) readonly buffer Heap { uint words[]; };
layout(push_constant) uniform Push { uint words[36]; } pc;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;
layout(location=0) out vec3 colour;
vec4 read4(Heap heap, uint address) {
    uint word = address / 4;
    return uintBitsToFloat(uvec4(heap.words[word], heap.words[word+1],
                                heap.words[word+2], heap.words[word+3]));
}
void main() {
    Heap heap = Heap(uint64_t(pc.words[32]) | (uint64_t(pc.words[33]) << 32));
    uint object = heap.words[pc.words[8] / 4];
    uint view = heap.words[(pc.words[8] + 0x40) / 4] + 0x160;
    vec3 world = position.x * read4(heap, object).xyz
               + position.y * read4(heap, object + 16).xyz
               + position.z * read4(heap, object + 32).xyz
               + read4(heap, object + 48).xyz;
    gl_Position = world.x * read4(heap, view)
                + world.y * read4(heap, view + 16)
                + world.z * read4(heap, view + 32)
                + read4(heap, view + 48);
    colour = vec3(0.2 + uv.x * 0.5, 0.2 + uv.y * 0.5, abs(normal.x));
}
"""
FRAGMENT = """#version 460
layout(location=0) in vec3 colour;
layout(location=0) out vec4 result;
void main() { result = vec4(colour, 1); }
"""


def transform(x, y=0, z=0):
    return [1, 0, 0, 0, 1, 0, 0, 0, 1, x, y, z]


def scene_bytes(draws):
    # A YZ triangle, followed by a triangle crossing the near plane when translated to x=0.1.
    meshes = (
        ((0, -.6, -.6), (0, .6, -.6), (0, 0, .6)),
        ((-.2, -.1, -.1), (.2, .1, -.1), (.2, 0, .1)),
    )
    output = bytearray(struct.pack("<4s4I3f", b"LSB1", 1, 0, len(meshes), len(draws), 0, 0, 0))
    for positions in meshes:
        output += struct.pack("<2I", 3, 3)
        for position, uv in zip(positions, ((0, 0), (1, 0), (.5, 1))):
            output += struct.pack("<8f", *position, 1, 0, 0, *uv)
        output += struct.pack("<3I", 0, 1, 2)
    for mesh, rows in draws:
        output += struct.pack("<Ii12f", mesh, -1, *rows)
    return output


class SandboxCommand(unittest.TestCase):
    def test_no_cull_reaches_both_renderer_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            entities = Path(temporary) / "entities.jsonl"
            entities.write_text("", encoding="ascii", newline="\n")
            stats = {"placed": 1, "draws": 1, "textures": 0, "textured_draws": 0, "center": [0, 0, 0]}
            args = ["--work", temporary, "--entities", str(entities), "--models", temporary, "--no-cull"]
            with patch.object(sandbox, "find_tool", return_value="renderer"), \
                    patch.object(sandbox, "pick_pair", return_value=("vs.spv", "ps.spv")), \
                    patch.object(sandbox.sandbox_scene, "build", return_value=stats), \
                    patch.object(sandbox, "check_frame", return_value=(True, "synthetic frame")), \
                    patch.object(sandbox.subprocess, "run",
                                 return_value=subprocess.CompletedProcess([], 0, stdout="")) as run, \
                    patch.object(sandbox.subprocess, "call", return_value=0) as call:
                self.assertEqual(sandbox.main([*args, "--check"]), 0)
                self.assertIn("--no-cull", run.call_args.args[0])
                self.assertEqual(sandbox.main([*args, "--window"]), 0)
                self.assertIn("--no-cull", call.call_args.args[0])


@unittest.skipUnless(os.environ.get("LEONIDA_SANDBOX_BUILD"), "set LEONIDA_SANDBOX_BUILD for Vulkan readback")
class SandboxReadback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        name = "vk_sandbox.exe" if os.name == "nt" else "vk_sandbox"
        cls.renderer = Path(os.environ["LEONIDA_SANDBOX_BUILD"]).resolve() / name
        compiler = shutil.which("glslangValidator")
        if not compiler:
            raise RuntimeError("glslangValidator is required for the synthetic shaders")
        for stage, source in (("vert", VERTEX), ("frag", FRAGMENT)):
            path = cls.work / ("fixture." + stage)
            path.write_text(source, encoding="ascii", newline="\n")
            subprocess.run([compiler, "-V", "--target-env", "vulkan1.2", "-o", str(path) + ".spv", path],
                           check=True, capture_output=True, text=True)

    def draw(self, draws, cull, camera=(0, 0, 0, 0, 0)):
        scene = self.work / "scene.bin"
        image = self.work / "frame.ppm"
        scene.write_bytes(scene_bytes(draws))
        image.unlink(missing_ok=True)
        command = [self.renderer, scene, self.work / "fixture.vert.spv", self.work / "fixture.frag.spv",
                   "--out", image, "--size", "64", "64", "--cam", *map(str, camera), "--fov", "90"]
        if not cull:
            command.append("--no-cull")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        counts = re.search(r"culling: submitted=(\d+) culled=(\d+) total=(\d+)", result.stdout)
        self.assertIsNotNone(counts, result.stdout)
        return image.read_bytes(), tuple(map(int, counts.groups()))

    def test_clip_boundaries_and_instance_transforms(self):
        rotated = [0, -2, 0, 3, 0, 0, .3, 0, -.5, 3, 0, 0]
        cases = (
            ("inside", (0, transform(3)), 1),
            ("behind", (0, transform(-3)), 0),
            ("side", (0, transform(3, 12)), 0),
            ("above", (0, transform(3, 0, 12)), 0),
            ("near crossing", (1, transform(.1)), 1),
            ("before near", (0, transform(.05)), 0),
            ("far", (0, transform(6000)), 0),
            ("far crossing", (1, transform(5000)), 1),
            ("side crossing", (0, transform(3, 3)), 1),
            ("rotated reflected scaled", (0, rotated), 1),
        )
        for name, draw, submitted in cases:
            with self.subTest(name=name):
                baseline, all_draws = self.draw([draw], False)
                culled, counts = self.draw([draw], True)
                self.assertEqual(culled, baseline)
                self.assertEqual(all_draws, (1, 0, 1))
                self.assertEqual(counts, (submitted, 1 - submitted, 1))

    def test_draw_reduction_and_changed_visible_geometry(self):
        visible = [(0, transform(3, -.8)), (0, transform(3, .8)), (1, transform(.1, 0, .1))]
        hidden = [(0, transform(-10, index % 20, index // 20)) for index in range(1000)]
        draws = visible + hidden
        baseline, _ = self.draw(draws, False)
        culled, counts = self.draw(draws, True)
        self.assertEqual(culled, baseline)
        self.assertEqual(counts, (3, 1000, 1003))
        # Removing visible geometry must change the image, so two clear images cannot satisfy the comparison.
        missing, _ = self.draw(hidden, True)
        self.assertNotEqual(culled, missing)
        for camera in ((0, 0, 0, 90, 0), (0, 4, 1, -45, -10)):
            with self.subTest(camera=camera):
                baseline, _ = self.draw(draws, False, camera)
                culled, _ = self.draw(draws, True, camera)
                self.assertEqual(culled, baseline)


if __name__ == "__main__":
    unittest.main()
