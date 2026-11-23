"""spirv-as and spirv-val in process (tools/spirv_tools_lib.py) give what the two programs give, and fall back to them.

    python -I tests/test_spirv_tools_lib.py

rdna2spirv.py used to launch spirv-as and spirv-val for every module, about half of the time per small shader. Where
the SPIRV-Tools shared library of the programs on PATH is found it now calls the library instead. These tests compare
the two paths on fuzz programs from four of the repository's fuzzers and, with llvm-mc, on every hand-written test
shader in runtime/test/shaders: the same bytes and the same verdict. A rejected module keeps spirv-val's message, and
without a library of the same build the programs run as before. Needs spirv-as and spirv-val (the Vulkan SDK) and
their shared library; no game files.
"""
import os
import gc
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import rdna2spirv  # noqa: E402
import spirv_tools_lib  # noqa: E402

LIB = spirv_tools_lib.load()
FUZZ_PROGRAMS = 1000
MINIMAL_CS = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
# Assembles, but spirv-val refuses it: OpReturnValue in a function that returns void.
INVALID_CS = MINIMAL_CS.replace("%fn = OpTypeFunction %void\n",
                                "%fn = OpTypeFunction %void\n%u32 = OpTypeInt 32 0\n%c1 = OpConstant %u32 1\n") \
    .replace("OpReturn\n", "OpReturnValue %c1\n")
UNKNOWN_OPCODE = "OpCapability Shader\nOpNotAnOpcode %x\n"


def no_launch():
    """Fails the test if rdna2spirv launches spirv-as or spirv-val: the library must have done the work."""
    return mock.patch.object(rdna2spirv.subprocess, "run", side_effect=AssertionError("a program was launched"))


def process_path(text, tmp):
    """(spirv-as failed, module bytes, spirv-val failed, stderr) from launching the two programs (exit codes as 0/1)."""
    sa, sv = os.path.join(tmp, "p.spvasm"), os.path.join(tmp, "p.spv")
    Path(sa).write_bytes(text)
    a = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", sv, sa], capture_output=True, text=True)
    if a.returncode:
        return 1, b"", None, a.stderr
    v = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
                       capture_output=True, text=True)
    return 0, Path(sv).read_bytes(), int(v.returncode != 0), v.stderr


def library_path(text):
    status, module, err = LIB.assemble(text)
    if status:
        return status, b"", None, err
    vstatus, verr = LIB.validate(module)
    return 0, module, vstatus, verr


def fuzz_texts(count):
    """SPIR-V assembly of `count` programs from valu_int_diff, salu_diff, mem_diff and shader_diff, in turn."""
    import mem_diff
    import salu_diff
    import shader_diff
    import valu_int_diff

    def valu(rng, path):
        lines, _ = valu_int_diff.build_program(rng, max_ops=rng.randint(4, 40))
        Path(path).write_text("\t" + "\n\t".join(lines) + "\n", encoding="utf-8")

    def salu(rng, path):
        lines = salu_diff.generate(rng, rng.randint(4, 40), sorted(salu_diff.OPS))
        Path(path).write_text("".join("\t" + ln + "\n" for ln in lines)
                              + "\texp mrt0 v8, v9, v10, v11 done vm\n\ts_endpgm\n", encoding="utf-8")

    def mem(rng, path):
        Path(path).write_text(mem_diff._listing(mem_diff.generate_program(rng, rng.randint(2, 20)).lines),
                              encoding="utf-8")

    def shader(rng, path):
        shader_diff._write_listing(path, shader_diff.generate_program(rng, rng.randint(4, 40))[0], with_export=True)

    makers, rng, texts, i = (valu, salu, mem, shader), random.Random(20261123), [], 0
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        while len(texts) < count and i < 3 * count:
            name, make = f"ps_fuzz_{i}", makers[i % len(makers)]
            i += 1
            path = os.path.join(tmp, "disasm", name + ".s")
            Path(tmp, name + ".dxbc").write_bytes(valu_int_diff.STUB_DXBC)
            make(rng, path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    texts.append((name, rdna2spirv.translate(path).encode("utf-8")))
            except rdna2spirv.Unsupported:
                continue
    return texts


def fixture_texts():
    """SPIR-V assembly of every runtime/test/shaders/*.asm the shader build makes, or [] without llvm-mc."""
    sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
    import build
    if not (os.path.isfile(build.MC[0]) or shutil.which(build.MC[0])):
        return []
    names = sorted(p.stem for p in Path(build.HERE).glob("*.asm") if not p.stem.startswith("ms_"))
    names += sorted(set(build.CLIP_CASES) - set(names))
    with tempfile.TemporaryDirectory() as tmp:
        if build.translate_tests(Path(tmp), names):
            raise AssertionError("runtime/test/shaders did not translate")
        return [(p.stem, p.read_bytes()) for p in sorted(Path(tmp, "spirv").glob("*.spvasm"))]


@unittest.skipUnless(LIB, "needs spirv-as, spirv-val and the SPIRV-Tools shared library of the same build")
class SameAsThePrograms(unittest.TestCase):
    def check_same(self, texts):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in texts:
                with self.subTest(name):
                    self.assertEqual(library_path(text), process_path(text, tmp))

    def test_fuzz_programs(self):
        texts = fuzz_texts(FUZZ_PROGRAMS)
        self.assertEqual(len(texts), FUZZ_PROGRAMS)
        self.check_same(texts)

    def test_hand_written_test_shaders(self):
        texts = fixture_texts()
        if not texts:
            self.skipTest("needs llvm-mc with the AMDGPU target")
        self.check_same(texts)

    def test_refusals_say_what_the_programs_say(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text in (INVALID_CS, UNKNOWN_OPCODE, ""):
                with self.subTest(text[-20:]):
                    got = library_path(text.encode())
                    self.assertTrue(got[0] or got[2], "refused")
                    self.assertEqual(got, process_path(text.encode(), tmp))


@unittest.skipUnless(LIB, "needs spirv-as, spirv-val and the SPIRV-Tools shared library of the same build")
class AssembleSpirv(unittest.TestCase):
    def test_an_invalid_module_is_refused_and_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            sv = os.path.join(tmp, "cs_bad.spv")
            Path(sv).write_bytes(b"stale")
            with no_launch():
                r = rdna2spirv.assemble_spirv(os.path.join(tmp, "unused.spvasm"), sv, asm_text=INVALID_CS)
            self.assertNotEqual(r.returncode, 0)
            self.assertTrue(r.stderr.startswith("error: line "), r.stderr)
            self.assertIn("OpReturnValue", r.stderr)
            self.assertEqual(os.listdir(tmp), [])

    def test_a_rejected_shader_is_counted_with_its_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(src)
            for name in ("ps_good", "ps_bad"):
                Path(src, name + ".s").write_text("", encoding="utf-8")
            texts = {"ps_good": MINIMAL_CS, "ps_bad": INVALID_CS}
            printed = []
            argv = ["rdna2spirv.py", src, "--out", out, "--no-asm"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(rdna2spirv, "translate", lambda p: texts[Path(p).stem]), \
                    mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))), \
                    no_launch():
                rdna2spirv.main()
            files = set(os.listdir(out))
            self.assertIn("ps_good.spv", files)
            self.assertNotIn("ps_good.spvasm", files)
            self.assertNotIn("ps_bad.spv", files)
            self.assertIn("2 shaders: 2 translated, 1 pass spirv-val", printed)
            self.assertTrue(any("INVALID: error: line " in p for p in printed), printed)


    @unittest.skipIf(os.name == "nt", "setrlimit")
    def test_a_staging_write_that_fails_refuses_the_shader_and_retires_the_old_module(self):
        # EFBIG (a file size limit) on the staging file used to escape as an exception that stopped the whole
        # shard and left the old module in place. Reported with the same reproduction by Nithin1777.
        import resource
        with tempfile.TemporaryDirectory() as tmp:
            sv = os.path.join(tmp, "cs_big.spv")
            Path(sv).write_bytes(b"stale")
            soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            old_handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (64, hard))
            try:
                with no_launch():
                    r = rdna2spirv.assemble_spirv(os.path.join(tmp, "unused.spvasm"), sv, asm_text=MINIMAL_CS)
            finally:
                resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
                signal.signal(signal.SIGXFSZ, old_handler)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("cannot write", r.stderr)
            self.assertEqual(os.listdir(tmp), [])


class Discovery(unittest.TestCase):
    def test_two_cold_loads_share_one_context(self):
        real_find, barrier, found = spirv_tools_lib.find, threading.Barrier(2), []

        def find_at_the_same_time(*args):
            try:
                barrier.wait(0.5)
            except threading.BrokenBarrierError:
                pass
            return real_find(*args)

        def ask():
            found.append(spirv_tools_lib.load("vulkan1.1"))

        with mock.patch.object(spirv_tools_lib, "find", find_at_the_same_time), \
                mock.patch.dict(spirv_tools_lib._cache, clear=True):
            threads = [threading.Thread(target=ask) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(found), 2)
        self.assertIs(found[0], found[1])

    @unittest.skipUnless(LIB, "needs the SPIRV-Tools shared library")
    def test_a_dropped_context_is_destroyed(self):
        destroyed = []

        class Counting:
            def __init__(self, lib):
                self.lib = lib

            def __getattr__(self, name):
                fn = getattr(self.lib, name)
                if name in ("spvContextDestroy", "spvValidatorOptionsDestroy"):
                    return lambda handle: (destroyed.append(name), fn(handle))[1]
                return fn

        tools = spirv_tools_lib.SpirvTools(Counting(LIB._lib), "vulkan1.2", True)
        del tools
        gc.collect()
        self.assertEqual(sorted(destroyed), ["spvContextDestroy", "spvValidatorOptionsDestroy"])

    def test_the_windows_dll_of_msys2_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("spirv-as", "spirv-val", "libSPIRV-Tools-shared.dll"):
                Path(tmp, name).write_bytes(b"")
            with mock.patch.object(sys, "platform", "win32"), \
                    mock.patch("ctypes.util.find_library", return_value=None), \
                    mock.patch.dict(os.environ, {spirv_tools_lib.ENV: ""}):
                found = list(spirv_tools_lib._candidates([os.path.join(tmp, "spirv-as")]))
            self.assertEqual(found, [os.path.join(tmp, "libSPIRV-Tools-shared.dll")])

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_a_library_the_sdk_ships_is_found(self):
        # Without this every other test here skips when discovery quietly fails.
        here = os.path.dirname(os.path.realpath(shutil.which("spirv-val")))
        shipped = [os.path.join(here, "..", d, name) for d in ("lib", "lib64") for name in spirv_tools_lib._lib_names()]
        if not any(os.path.isfile(p) for p in shipped):
            self.skipTest("the programs on PATH ship no library")
        self.assertIsNotNone(LIB, "the SDK ships the library but discovery found none")


@unittest.skipIf(os.name == "nt", "the stand-in programs are shell scripts")
@unittest.skipUnless(LIB, "needs spirv-as, spirv-val and the SPIRV-Tools shared library of the same build")
class Fallback(unittest.TestCase):
    """Stand-ins for spirv-as and spirv-val in a folder with no library: they run the real programs, and print
    `version` for --version."""

    def stand_ins(self, tmp, version, run="exec"):
        """`run` is the shell command a stand-in launches in place of the real program ("exec" runs it)."""
        for tool in ("spirv-as", "spirv-val"):
            path = os.path.join(tmp, tool)
            body = f"exec {shutil.which(tool)} \"$@\"" if run == "exec" or tool == "spirv-as" else run
            Path(path).write_text(f"#!/bin/sh\nif [ \"$1\" = --version ]; then echo '{version}'; exit 0; fi\n{body}\n",
                                  encoding="utf-8")
            os.chmod(path, 0o755)
        return {"PATH": tmp + os.pathsep + os.environ.get("PATH", "")}

    def real_version(self):
        return subprocess.run(["spirv-val", "--version"], capture_output=True, text=True).stdout.splitlines()[0]

    def test_a_library_of_another_build_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {spirv_tools_lib.ENV: LIB.path}
            env.update(self.stand_ins(tmp, self.real_version()))
            with mock.patch.dict(os.environ, env):
                self.assertIsNotNone(spirv_tools_lib.find(), "the override alone is accepted")
            env.update(self.stand_ins(tmp, "SPIRV-Tools v1999.1 another build"))
            with mock.patch.dict(os.environ, env):
                self.assertIsNone(spirv_tools_lib.find())

    def test_a_library_that_validates_differently_is_not_used(self):
        # The same banner, but spirv-val accepts the module with an ID bound over its limit and the library refuses it.
        with tempfile.TemporaryDirectory() as tmp:
            env = {spirv_tools_lib.ENV: LIB.path}
            env.update(self.stand_ins(tmp, self.real_version(), run="exit 0"))
            with mock.patch.dict(os.environ, env):
                self.assertIsNone(spirv_tools_lib.find())

    def test_a_relative_path_entry_names_the_same_programs_to_both_paths(self):
        # PATH=bin: discovery looked in the caller's bin and the launch, from the output folder, in the output's.
        with tempfile.TemporaryDirectory() as caller, tempfile.TemporaryDirectory() as out:
            os.mkdir(os.path.join(caller, "bin"))
            os.mkdir(os.path.join(out, "bin"))
            self.stand_ins(os.path.join(caller, "bin"), self.real_version())
            self.stand_ins(os.path.join(out, "bin"), self.real_version(), run="echo 'strict' >&2; exit 1")
            here = os.getcwd()
            os.chdir(caller)
            try:
                with mock.patch.dict(os.environ, {"PATH": "bin", spirv_tools_lib.ENV: "off"}):
                    r = rdna2spirv.assemble_spirv(os.path.join(out, "a.spvasm"), os.path.join(out, "a.spv"),
                                                  asm_text=MINIMAL_CS)
            finally:
                os.chdir(here)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(Path(out, "a.spv").exists())

    def test_without_the_library_the_programs_run(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            env = self.stand_ins(tmp, self.real_version())
            env[spirv_tools_lib.ENV] = os.path.join(tmp, "missing", spirv_tools_lib._lib_names()[0])
            with mock.patch.dict(os.environ, env), mock.patch("ctypes.util.find_library", return_value=None):
                self.assertIsNone(spirv_tools_lib.load())
                good = rdna2spirv.assemble_spirv(os.path.join(out, "a.spvasm"), os.path.join(out, "a.spv"),
                                                 asm_text=MINIMAL_CS)
                Path(out, "b.spvasm").write_text(INVALID_CS, encoding="utf-8")
                bad = rdna2spirv.assemble_spirv(os.path.join(out, "b.spvasm"), os.path.join(out, "b.spv"))
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(Path(out, "a.spv").read_bytes(), LIB.assemble(MINIMAL_CS.encode())[1])
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual(bad.stderr, LIB.validate(LIB.assemble(INVALID_CS.encode())[1])[1])
            self.assertFalse(Path(out, "b.spv").exists())


if __name__ == "__main__":
    unittest.main()
