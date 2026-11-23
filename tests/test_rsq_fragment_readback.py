"""Matched fragment readback for v_rsq_f64. Set VK_HARNESS to the built vk_harness
and run python -I tests/test_rsq_fragment_readback.py with Vulkan SDK tools on PATH.
"""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/semantic"))
from support import translate_lines

sys.path.insert(0, str(ROOT / "runtime/test/shaders"))


def translated(operand, runtime):
    from build import target_container  # typed target from #3544

    if runtime:
        load = ["buffer_load_dwordx2 v[4:5], off, s[0:3], 0", "s_waitcnt vmcnt(0)"]
    else:
        load = [f"v_mov_b32_e32 v4, 0x{operand & 0xffffffff:x}",
                f"v_mov_b32_e32 v5, 0x{operand >> 32:x}"]
    return translate_lines("ps_rsq_controls", load + [
        "v_rsq_f64_e32 v[6:7], v[4:5]",
        "exp mrt0 v4, v5, v6, v7 done vm", "s_endpgm",
    ], dxbc=target_container(1))


def reference(operand, runtime):
    # Same physical heap and byte offset as s[0:3] below. Keep the arithmetic
    # inside a function so glslang emits InverseSqrt even for literal callers.
    words = ("pc.heap.words[512]" if runtime else
             f"uvec2({operand & 0xffffffff}u, {operand >> 32}u)")
    return """#version 460
#extension GL_EXT_buffer_reference : require
layout(buffer_reference, std430, buffer_reference_align=8) readonly buffer Heap { uvec2 words[]; };
layout(push_constant) uniform Params { layout(offset=128) Heap heap; } pc;
layout(location=0) out uvec4 target;
uvec2 rsq(uvec2 x) { return unpackDouble2x32(inversesqrt(packDouble2x32(x))); }
""" + f"void main() {{ uvec2 x = {words}; target = uvec4(x, rsq(x)); }}\n"


def build_fragment(work, operand, runtime, direct):
    module = work / "pixel.spv"
    if direct:
        source = work / "reference.frag"
        source.write_text(reference(operand, runtime), encoding="utf-8", newline="\n")
        command = ["glslangValidator", "-V"]
    else:
        source = work / "translated.spvasm"
        source.write_text(translated(operand, runtime), encoding="utf-8", newline="\n")
        command = ["spirv-as"]
    subprocess.run([*command, "--target-env", "vulkan1.2", source, "-o", module],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", module], check=True)
    return module


HARNESS = os.environ.get("VK_HARNESS") or shutil.which("vk_harness")


@unittest.skipUnless(HARNESS, "set VK_HARNESS for the Vulkan readback")
class RsqFragmentReadback(unittest.TestCase):
    def test_operand_and_inverse_sqrt(self):
        one = 0x3ff0000000000000
        operands = range(one - 3, one + 4)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            vertex = work / "vertex.spv"
            subprocess.run(["glslangValidator", "-V", ROOT / "runtime/test/glsl/vs_fullscreen.vert",
                            "-o", vertex], check=True, stdout=subprocess.DEVNULL)
            for runtime in (False, True):
                for operand in operands:
                    with self.subTest(runtime=runtime, operand=hex(operand)):
                        frames = []
                        for direct in (False, True):
                            module = build_fragment(work, operand, runtime, direct)
                            output = work / "frame.raw"
                            output.unlink(missing_ok=True)
                            env = dict(os.environ, VKH_COLOR_FORMAT="107:16", VKH_SGPRS="4096,0,8,0x30000000",
                                       VKH_HEAP=f"4096={operand & 0xffffffff},4100={operand >> 32}")
                            draw = subprocess.run([HARNESS, vertex, module, output],
                                                  env=env, capture_output=True, text=True)
                            if draw.returncode == 77:
                                self.skipTest(draw.stderr.strip())
                            self.assertEqual(draw.returncode, 0, draw.stderr)
                            frame = output.read_bytes()
                            self.assertEqual(len(frame), 64 * 64 * 16)
                            observed, result = struct.unpack_from("<QQ", frame)
                            self.assertEqual(observed, operand, f"identity, direct={direct}")
                            self.assertTrue(frame == struct.pack("<QQ", operand, result) * (64 * 64),
                                            f"nonuniform frame, direct={direct}")
                            frames.append(result)
                        self.assertEqual(*frames, "translated and direct InverseSqrt disagree")


if __name__ == "__main__":
    unittest.main()
