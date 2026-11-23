"""Unit tests for tools/mesh_compute_vs.py (standard library; the assembly checks need SPIRV-Tools on PATH)."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mesh_compute_vs as vs  # noqa: E402
import rdna2spirv  # noqa: E402

HAVE_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))


def layout(maxv=64, maxp=3, clip=0, cull=0, params=()):
    """A real `mesh_compute` index entry, from the translator's own layout function."""
    return rdna2spirv.mesh_compute_layout(maxv, maxp, clip, cull, list(params))


def write_index(directory, index):
    path = os.path.join(directory, "index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return path


class Interface(unittest.TestCase):
    def test_entry_point_carries_the_mesh_shader_locations(self):
        asm = vs.vs_spvasm(layout(clip=5, cull=2, params=(2, 9)))
        entry = asm.split("OpEntryPoint Vertex", 1)[1].splitlines()[0]
        for name in ("%vertex_index", "%pc", "%out_pos", "%out_clip", "%out_cull", "%out2", "%out9"):
            self.assertIn(name, entry)
        for decoration in ("BuiltIn ClipDistance", "BuiltIn CullDistance", "Location 2", "Location 9"):
            self.assertIn(decoration, asm)

    def test_no_clip_or_cull_means_no_such_outputs(self):
        asm = vs.vs_spvasm(layout(params=(3,)))
        self.assertNotIn("ClipDistance", asm)
        self.assertNotIn("CullDistance", asm)
        self.assertNotIn("%out_clip", asm)
        self.assertIn("Location 3", asm)

    def test_pair_f16_halves_for_high_param_producer(self):
        """SHADER-378: a 16-param mesh-compute producer paired with a PS reading f16 on params (0, 4, 5, 6, 7) emits
        only those 5 decoded-half pairs within locations 0..29, with zero overlap."""
        lay = layout(params=tuple(range(16)))
        ps_halves = {"0": [28, 29], "4": [26, 27], "5": [24, 25], "6": [22, 23], "7": [20, 21]}
        paired = vs.pair_f16_halves(lay, ps_halves, ps_raw_inputs=range(15))
        self.assertEqual(paired, ps_halves)
        asm = vs.vs_spvasm(lay, f16_halves=paired)
        for loc in range(16):
            self.assertIn(f"OpDecorate %out{loc} Location {loc}", asm)
        for lo, hi in paired.values():
            self.assertIn(f"OpDecorate %out{lo} Location {lo}", asm)
            self.assertIn(f"OpDecorate %out{hi} Location {hi}", asm)
        self.assertNotIn("Location 30", asm)

        # Collision with producer param15 relocates both halves into free slots <= 29.
        reloc = vs.pair_f16_halves(lay, {"7": [15, 29]}, ps_raw_inputs=range(15))
        self.assertEqual(reloc, {"7": [28, 29]})
        asm_reloc = vs.vs_spvasm(lay, f16_halves=reloc)
        self.assertIn("OpDecorate %out28 Location 28", asm_reloc)
        self.assertIn("OpDecorate %out29 Location 29", asm_reloc)

        # Missing producer param and colliding explicit override are rejected cleanly.
        with self.assertRaises(ValueError):
            vs.pair_f16_halves(layout(params=(0, 1)), {"7": [28, 29]})
        with self.assertRaises(ValueError):
            vs.vs_spvasm(layout(params=(0, 1)), f16_halves={"0": [1, 29]})

    def test_the_index_drives_write_for_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(vs.write_for_index(tmp), [])       # no index.json: nothing to do
            write_index(tmp, {"ms_none": {"subgroup_size": 64}, "ms_null": None,
                              "ms_a": {"mesh_compute": layout(clip=1)}})
            written = vs.write_for_index(tmp)
            self.assertEqual([os.path.basename(p) for p in written], ["ms_a.mesh_vs.spvasm"])
            with open(written[0], encoding="utf-8") as f:
                self.assertEqual(f.read(), vs.vs_spvasm(layout(clip=1)))


class Assembly(unittest.TestCase):
    def test_every_layout_assembles_and_validates(self):
        if not HAVE_SPIRV_TOOLS:
            self.skipTest("spirv-as / spirv-val not on PATH")
        layouts = {
            "ms_plain": layout(),
            "ms_clip": layout(clip=1),
            "ms_full": layout(clip=5, cull=2, params=(2, 5)),
            "ms_small": layout(maxv=1, maxp=256),
            "ms_large": layout(maxv=128, maxp=64, clip=4, params=(0,)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_index(tmp, {name: {"mesh_compute": lay} for name, lay in layouts.items()})
            built = vs.build_for_index(tmp)
            self.assertEqual([os.path.basename(p) for p in built],
                             [name + ".mesh_vs.spv" for name in sorted(layouts)])
            for spv in built:
                self.assertGreater(os.path.getsize(spv), 0)
                self.assertTrue(os.path.isfile(spv[: -len(".spv")] + ".spvasm"))

    def test_only_validated_binary_reaches_the_final_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_index(tmp, {"ms_a": {"mesh_compute": layout()}})
            final = os.path.join(tmp, "ms_a.mesh_vs.spv")
            with open(final, "wb") as f:
                f.write(b"previous shader")

            def assemble_and_validate(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                if cmd[0] == "spirv-as":
                    with open(cmd[-1], "wb") as f:
                        f.write(b"new shader")
                    return subprocess.CompletedProcess(cmd, 0, "", "")
                self.assertEqual(cmd[0], "spirv-val")
                with open(cmd[-1], "rb") as f:
                    self.assertEqual(f.read(), b"new shader")
                return subprocess.CompletedProcess(cmd, validation_code, "", "validator output")

            with (mock.patch.object(vs, "tool_path", side_effect=lambda name: name),
                  mock.patch.object(vs.subprocess, "run", side_effect=assemble_and_validate)):
                validation_code = 1
                with self.assertRaisesRegex(RuntimeError, "spirv-val failed"):
                    vs.build_for_index(tmp)
                with open(final, "rb") as f:
                    self.assertEqual(f.read(), b"previous shader")

                validation_code = 0
                self.assertEqual(vs.build_for_index(tmp), [final])
                with open(final, "rb") as f:
                    self.assertEqual(f.read(), b"new shader")

            self.assertEqual(sorted(os.listdir(tmp)), ["index.json", "ms_a.mesh_vs.spv", "ms_a.mesh_vs.spvasm"])

    def test_a_rejected_shader_names_the_tool(self):
        if not HAVE_SPIRV_TOOLS:
            self.skipTest("spirv-as / spirv-val not on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            asm = os.path.join(tmp, "bad.mesh_vs.spvasm")
            with open(asm, "w", encoding="utf-8") as f:
                f.write("OpCapability NothingOfTheSort\n")
            with mock.patch.object(vs, "write_for_index", return_value=[asm]):
                with self.assertRaises(RuntimeError) as ctx:
                    vs.build_for_index(tmp)
        self.assertIn("spirv-as failed for", str(ctx.exception))


class Cli(unittest.TestCase):
    def run_main(self, argv, **env):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = vs.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_a_missing_spirv_tool_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_index(tmp, {"ms_a": {"mesh_compute": layout()}})
            rc, stdout, stderr = self.run_main(["mesh_compute_vs.py", tmp], PATH="")
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(len(stderr.splitlines()), 1)
        self.assertIn("spirv-as not found on PATH", stderr)

    def test_invalid_index_leaves_existing_assembly_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = os.path.join(tmp, "index.json")
            asm = os.path.join(tmp, "ms_a.mesh_vs.spvasm")
            for payload, message in ((b"{", "invalid JSON"), (b"[]", "expected a shader index object"),
                                     (b'{"ms_a": {"mesh_compute": {"params": {}}}}', "layout for 'ms_a'")):
                with self.subTest(payload=payload):
                    with open(asm, "wb") as f:
                        f.write(b"previous assembly")
                    with open(index, "wb") as f:
                        f.write(payload)
                    with mock.patch.object(vs, "tool_path", side_effect=lambda name: name):
                        rc, stdout, stderr = self.run_main(["mesh_compute_vs.py", tmp])
                    self.assertEqual((rc, stdout), (1, ""))
                    self.assertIn(index, stderr)
                    self.assertIn(message, stderr)
                    with open(asm, "rb") as f:
                        self.assertEqual(f.read(), b"previous assembly")

    def test_a_wrong_argument_count_is_a_usage_error(self):
        rc, stdout, stderr = self.run_main(["mesh_compute_vs.py"])
        self.assertEqual(rc, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Vertex shaders that draw", stderr)

    def test_help_prints_usage(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                rc, stdout, stderr = self.run_main(["mesh_compute_vs.py", flag])
                self.assertEqual(rc, 0)
                self.assertIn("mesh_compute_vs.py <spirv dir>", stdout)
                self.assertEqual(stderr, "")

    def test_a_missing_directory_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing")
            with mock.patch.object(vs, "build_for_index") as build:
                rc, stdout, stderr = self.run_main(["mesh_compute_vs.py", missing])
            build.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(len(stderr.splitlines()), 1)
        self.assertIn("not a directory", stderr)

    def test_an_unreadable_index_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_index(tmp, {})
            os.mkdir(os.path.join(tmp, "dir"))
            # tool_path() runs first, so stub it out too: otherwise a machine without SPIRV-Tools reports
            # the missing tool instead of the write error this test is about.
            with mock.patch.object(vs, "tool_path", return_value="spirv-as"), \
                    mock.patch.object(vs, "write_for_index",
                                      side_effect=OSError(13, "Permission denied", path)):
                rc, _, stderr = self.run_main(["mesh_compute_vs.py", tmp])
        self.assertEqual(rc, 1)
        self.assertEqual(len(stderr.splitlines()), 1)
        self.assertIn("Permission denied", stderr)

    def test_a_damaged_layout_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            lay = layout(params=(0,))
            lay["params"] = {"0": [16]}          # what a hand-edited index.json actually looks like
            write_index(tmp, {"ms_a": {"mesh_compute": lay}})
            with mock.patch.object(vs, "tool_path", return_value="spirv-as"):
                rc, stdout, stderr = self.run_main(["mesh_compute_vs.py", tmp])
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(len(stderr.splitlines()), 1)
        self.assertIn("ms_a.mesh_compute params location 0", stderr)


class DamagedLayout(unittest.TestCase):
    """index.json is a file on disk that other tools and people edit. A layout that cannot make a shader has to
    say which field of it, and must not reach the assembler as a KeyError or a TypeError traceback."""

    def test_a_wrong_typed_field_names_the_field(self):
        lay = layout()
        lay["max_vertices"] = "x"
        with self.assertRaisesRegex(vs.LayoutError, "max_vertices: expected an integer, got 'x'"):
            vs.vs_spvasm(lay)

    def test_the_params_offsets_are_checked(self):
        lay = layout(params=(0,))
        lay["params"] = {"0": [1, 2]}
        with self.assertRaisesRegex(vs.LayoutError, "params location 0"):
            vs.vs_spvasm(lay)

    def test_a_missing_field_is_named(self):
        lay = layout()
        del lay["slot_bytes"]
        with self.assertRaisesRegex(vs.LayoutError, "no slot_bytes"):
            vs.vs_spvasm(lay)

    def test_a_bool_is_not_an_integer(self):
        lay = layout()
        lay["max_primitives"] = True
        with self.assertRaisesRegex(vs.LayoutError, "max_primitives"):
            vs.vs_spvasm(lay)

    def test_the_distance_offsets_only_matter_with_distances(self):
        lay = layout()
        del lay["clip_offset"]                   # clip_count 0: those floats are never loaded
        self.assertTrue(vs.vs_spvasm(lay))
        lay = layout(clip=1)
        del lay["clip_offset"]
        with self.assertRaisesRegex(vs.LayoutError, "no clip_offset"):
            vs.vs_spvasm(lay)

    def test_f16_halves_must_be_locations_paired_with_two(self):
        for bad in (None, 5, {"0": 5}, {"0": ["a", "b"]}, {"x": [1, 2]}):
            with self.subTest(bad=bad):
                lay = layout(params=(0,))
                lay["f16_halves"] = bad
                with self.assertRaises(vs.LayoutError):
                    vs.vs_spvasm(lay)

    def test_a_checked_layout_builds_the_same_shader(self):
        for lay in (layout(), layout(clip=5, cull=2, params=(2, 9)), layout(maxv=1, maxp=256)):
            with self.subTest(maxv=lay["max_vertices"]):
                self.assertEqual(vs.vs_spvasm(vs.check_layout(lay)), vs.vs_spvasm(lay))

    def test_the_f16_halves_argument_may_come_back_with_int_keys(self):
        """pair_f16_halves answers with string keys because JSON has no int ones; a caller that already read them
        back as ints describes the same shader and gets the same shader."""
        lay = layout(params=(0, 4))
        halves = vs.pair_f16_halves(lay, {"0": [28, 29], "4": [26, 27]})
        self.assertEqual(vs.vs_spvasm(lay, f16_halves={int(k): v for k, v in halves.items()}),
                         vs.vs_spvasm(lay, f16_halves=halves))


class DamagedIndex(unittest.TestCase):
    def test_one_bad_entry_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = layout(), layout()
            bad["max_vertices"] = "x"
            write_index(tmp, {"ms_a": {"mesh_compute": good}, "ms_b": {"mesh_compute": bad}})
            kept = os.path.join(tmp, "ms_a.mesh_vs.spvasm")
            with open(kept, "w", encoding="utf-8") as f:
                f.write("previous run\n")
            with self.assertRaises(vs.LayoutError) as ctx:
                vs.write_for_index(tmp)
            self.assertIn("ms_b", str(ctx.exception))            # says which entry of the index is broken
            with open(kept, encoding="utf-8") as f:
                self.assertEqual(f.read(), "previous run\n")     # not truncated by the entry that failed
            self.assertFalse(os.path.exists(os.path.join(tmp, "ms_b.mesh_vs.spvasm")))

    def test_an_entry_names_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_index(tmp, {"ms_a": {"mesh_compute": [layout()]}})
            with self.assertRaisesRegex(vs.LayoutError, r"ms_a\.mesh_compute: expected an object"):
                vs.write_for_index(tmp)

    def test_an_index_that_is_not_json_is_one_named_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"ms_a": ')
            with self.assertRaises(vs.LayoutError) as ctx:
                vs.write_for_index(tmp)
            self.assertIn(path, str(ctx.exception))
            with open(path, "w", encoding="utf-8") as f:
                f.write("[]")
            with self.assertRaisesRegex(vs.LayoutError, "got list"):
                vs.write_for_index(tmp)


if __name__ == "__main__":
    unittest.main()
