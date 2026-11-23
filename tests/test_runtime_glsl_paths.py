"""Build real SPIR-V fixtures through the runtime's CMake rule in a Unicode checkout."""
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


RULE = Path(__file__).resolve().parents[1] / "runtime" / "cmake" / "compile_test_glsl.cmake"
COMPILER = shutil.which("glslangValidator") or shutil.which("glslang")


@unittest.skipUnless(COMPILER and shutil.which("cmake") and shutil.which("ninja"),
                     "requires glslang, CMake and Ninja")
class RuntimeGlslPaths(unittest.TestCase):
    def test_parallel_variants_in_unicode_source_and_output_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "checkout \u6771\u4eac")
            build = Path(temp, "build \u0394\u6e2c")
            source.mkdir()
            (source / "probe.comp").write_text(
                "#version 450\nlayout(local_size_x = LANES) in;\nvoid main() {}\n", encoding="ascii", newline="\n")
            (source / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.24)
project(probe NONE)
include("${RULE}")
foreach(lanes IN ITEMS 1 8)
  set(binary "${CMAKE_CURRENT_BINARY_DIR}/generated/${lanes}.spv")
  compile_test_glsl("${binary}" "${CMAKE_CURRENT_SOURCE_DIR}/probe.comp" vulkan1.2 "-DLANES=${lanes}")
  list(APPEND outputs "${binary}")
endforeach()
add_custom_target(shaders ALL DEPENDS ${outputs})
""", encoding="ascii", newline="\n")
            subprocess.run(["cmake", "-G", "Ninja", "-S", source, "-B", build, f"-DRULE={RULE.as_posix()}",
                            f"-DVulkan_GLSLANG_VALIDATOR_EXECUTABLE={COMPILER}"], check=True)
            subprocess.run(["cmake", "--build", build, "--parallel", "2"], check=True)
            for lanes in (1, 8):
                with self.subTest(lanes=lanes):
                    binary = (build / "generated" / f"{lanes}.spv").read_bytes()
                    self.assertEqual(struct.unpack_from("<II", binary), (0x07230203, 0x00010500))
                    # OpExecutionMode %entry LocalSize X Y Z in a SPIR-V 1.5 module.
                    words = struct.unpack(f"<{len(binary) // 4}I", binary)
                    offset = 5
                    sizes = []
                    while offset < len(words):
                        header = words[offset]
                        count, opcode = header >> 16, header & 0xffff
                        self.assertGreater(count, 0)
                        if opcode == 16 and words[offset + 2] == 17:
                            sizes.append(words[offset + 3:offset + count])
                        offset += count
                    self.assertEqual(sizes, [(lanes, 1, 1)])


if __name__ == "__main__":
    unittest.main()
