"""tools/ped_assemble.py on a malformed JSON part or manifest: one `ERROR:` line and exit 2, not a traceback.

    python -I tests/test_ped_assemble_malformed_json.py

Every part and manifest here is written by the test (a two-triangle quad, one material, a one-joint skeleton); no
game files and no GPU.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import ped_assemble as pa  # noqa: E402

DELETE = object()
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def part() -> dict:
    """A valid skinned, coloured quad."""
    return {
        "name": "hand_fr1_000", "coord_space": "z-up", "translation": [0, 0, 0],
        "positions": [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]],
        "triangles": [[0, 1, 2], [1, 3, 2]],
        "normals": [[0, 0, 1]] * 4, "uvs": [[0, 0], [1, 0], [0, 1], [1, 1]], "colors": [[1, 1, 1, 1]] * 4,
        "joints_0": [[0, 0, 0, 0]] * 4, "weights_0": [[1, 0, 0, 0]] * 4, "joint_names": ["root"],
        "material": {"name": "m", "base_color_factor": [1, 1, 1, 1], "metallic_factor": 0.1,
                     "roughness_factor": 0.5, "alpha_cutoff": 0.5},
    }


def manifest() -> dict:
    return {
        "coord_space": "z-up",
        "parts": [part()],
        "skeleton": {"name": "sk", "standing_offset": 1.0, "joints": [
            {"name": "root", "parent": -1, "translation": [0, 0, 0], "rotation": [0, 0, 0, 1],
             "scale": [1, 1, 1], "inverse_bind_matrix": IDENTITY}]},
    }


def edited(doc: dict, path: tuple, value: object) -> dict:
    """`doc` with the value at `path` replaced (or removed, for DELETE)."""
    doc = copy.deepcopy(doc)
    cursor = doc
    for key in path[:-1]:
        cursor = cursor[key]
    if value is DELETE:
        del cursor[path[-1]]
    else:
        cursor[path[-1]] = value
    return doc


MALFORMED_PART = (
    # (label, path into the part, replacement, text the message must contain)
    ("positions is a number", ("positions",), 5, "positions must be a list"),
    ("positions is an object", ("positions",), {}, "positions must be a list"),
    ("a position has two values", ("positions", 0), [0, 0], "positions[0] must be a list of at least 3 numbers"),
    ("a position is a number", ("positions", 1), 7, "positions[1] must be a list of at least 3 numbers"),
    ("a position holds text", ("positions", 0, 0), "a", "positions[0] must hold numbers, got 'a'"),
    ("a position holds null", ("positions", 0, 0), None, "positions[0] must hold numbers, got None"),
    ("a position is past f32", ("positions", 0, 0), 1e308, "must hold finite 32-bit floats"),
    ("triangles is a number", ("triangles",), 7, "triangles must be a list"),
    ("a triangle has two indices", ("triangles", 0), [0, 1], "triangles[0] must be a list of at least 3 numbers"),
    ("a triangle holds text", ("triangles", 0, 0), "x", "triangles[0] must hold numbers, got 'x'"),
    ("a triangle index is past the positions", ("triangles", 0, 2), 9, "triangle index 9 is outside the part's 4"),
    ("a triangle index is negative", ("triangles", 0, 0), -1, "triangle index -1 is outside the part's 4"),
    ("flat indices hold a list", ("triangles",), [0, [1, 2, 3]], "triangles must hold numbers"),
    ("normals is a number", ("normals",), 3, "normals must be a list"),
    ("a normal has two values", ("normals", 0), [0, 1], "normals[0] must be a list of at least 3 numbers"),
    ("uvs holds numbers, not pairs", ("uvs",), [1, 2, 3], "uvs[0] must be a list of at least 2 numbers"),
    ("a colour has two values", ("colors", 0), [1, 1], "colors[0] must be a list of at least 3 numbers"),
    ("joints_0 is text", ("joints_0",), "0000", "joints_0 must be a list"),
    ("a joint row has three values", ("joints_0", 0), [0, 0, 0], "joints_0[0] must be a list of at least 4 numbers"),
    ("a joint row has five values", ("joints_0", 0), [0, 0, 0, 0, 0], "joints_0[0] must hold 4 values, got 5"),
    ("a joint index is negative", ("joints_0", 0, 0), -1, "joint indices from 0 to 65535"),
    ("a joint index is past 16 bits", ("joints_0", 0, 0), 65536, "joint indices from 0 to 65535"),
    ("a weight row has one value", ("weights_0", 0), [1], "weights_0[0] must be a list of at least 4 numbers"),
    ("a weight is not finite", ("weights_0", 0, 0), 1e308, "must hold finite 32-bit floats"),
    ("translation is a number", ("translation",), 4, "translation must be a list of at least 3 numbers"),
    ("translation has one value", ("translation",), [1], "translation must be a list of at least 3 numbers"),
    ("base_color_factor is a number", ("material", "base_color_factor"), 1, "base_color_factor must be a list"),
    ("base_color_factor holds text", ("material", "base_color_factor"), [1, "x", 1, 1], "must hold numbers"),
    ("metallic_factor is null", ("material", "metallic_factor"), None, "metallic_factor must hold numbers"),
    ("roughness_factor is a list", ("material", "roughness_factor"), [1], "roughness_factor must hold numbers"),
    ("alpha_cutoff is text", ("material", "alpha_cutoff"), "half", "alpha_cutoff must hold numbers"),
    ("a base64 texture that is not a PNG", ("material", "base_color_png"), "base64:@@@",
     "base_color_png is not base64 of a PNG file"),
    ("a base64 texture of other bytes", ("material", "normal_png"),
     "base64:" + base64.b64encode(b"not a png").decode("ascii"), "normal_png is not base64 of a PNG file"),
)

JOINT = ("skeleton", "joints", 0)
MALFORMED_MANIFEST = (
    ("parts is a number", ("parts",), 5, "parts must be a list"),
    ("parts is an object", ("parts",), {"a": 1}, "parts must be a list"),
    ("a part path is a number", ("parts", 0), {"path": 5}, "parts[0].path must be a string, got 5"),
    ("an inline part is malformed", ("parts", 0, "positions"), 5, "positions must be a list"),
    ("skeleton is a number", ("skeleton",), 3, "skeleton must be a JSON object"),
    ("skeleton is a list", ("skeleton",), [], "skeleton must be a JSON object"),
    ("joints is a number", ("skeleton", "joints"), 5, "joints must be a list"),
    ("joints is null", ("skeleton", "joints"), None, "joints must be a list"),
    ("a joint is a number", JOINT, 3, "joints[0] must be a JSON object"),
    ("a joint has no name", JOINT + ("name",), DELETE, "joints[0] has no name"),
    ("a joint parent is text", JOINT + ("parent",), "root", "joints[0]: parent must hold numbers"),
    ("a joint translation is short", JOINT + ("translation",), [0, 0], "translation must be a list of at least 3"),
    ("a joint rotation has three values", JOINT + ("rotation",), [0, 0, 0], "rotation must be a list of at least 4"),
    ("a joint scale is a number", JOINT + ("scale",), 1, "scale must be a list of at least 3"),
    ("an inverse bind matrix is text", JOINT + ("inverse_bind_matrix",), "identity", "inverse_bind_matrix must be"),
    ("an inverse bind matrix holds text", JOINT + ("inverse_bind_matrix", 0), "a", "must hold numbers, got 'a'"),
    ("standing_offset is a list", ("skeleton", "standing_offset"), [1], "standing_offset must hold numbers"),
)


class MalformedJson(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = self.enterContext(tempfile.TemporaryDirectory())

    def write(self, name: str, doc: object) -> str:
        path = os.path.join(self.folder, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh)
        return path

    def test_the_valid_part_and_manifest_still_load(self) -> None:
        (loaded,), skeleton = pa.load_parts_from_path(self.write("part.json", part()))
        self.assertIsNone(skeleton)
        self.assertEqual(loaded.triangles, [(0, 1, 2), (1, 3, 2)])
        self.assertEqual(loaded.joints_0[0], (0, 0, 0, 0))
        self.assertEqual(loaded.material.metallic_factor, 0.1)
        (loaded,), skeleton = pa.load_parts_from_path(self.write("manifest.json", manifest()))
        self.assertEqual(loaded.name, "hand_fr1_000")
        self.assertEqual([j.name for j in skeleton.joints], ["root"])
        self.assertEqual(skeleton.standing_offset, 1.0)

    def test_what_the_reader_took_before_it_still_takes(self) -> None:
        """Flat indices, a falsy `triangles`, numbers written as text, extra components, the camelCase keys and a
        material with only a 3-value colour all loaded on main and still do."""
        cases = (
            ("flat indices", edited(part(), ("triangles",), [0, 1, 2, 1, 3, 2])),
            ("the `indices` key", {**edited(part(), ("triangles",), DELETE), "indices": [0, 1, 2]}),
            ("triangles is null", edited(part(), ("triangles",), None)),
            ("triangles is empty text", edited(part(), ("triangles",), "")),
            ("a position written as text", edited(part(), ("positions", 0), ["0", "0.5", "1e-3"])),
            ("an index written as text", edited(part(), ("triangles", 0, 0), "0")),
            ("a position with a fourth value", edited(part(), ("positions", 0), [0, 0, 0, 1])),
            ("camelCase material keys", edited(part(), ("material",), {"baseColorFactor": [1, 0, 0],
                                                                         "metallicFactor": "0.5"})),
            ("no optional streams", {"positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "triangles": [[0, 1, 2]]}),
            ("no geometry at all", {"name": "empty"}),
        )
        for label, doc in cases:
            with self.subTest(label):
                (loaded,), _ = pa.load_parts_from_path(self.write("tolerated.json", doc))
                self.assertIsInstance(loaded, pa.MeshPart)
        (loaded,), _ = pa.load_parts_from_path(self.write("camel.json", cases[7][1]))
        self.assertEqual(loaded.material.base_color_factor, (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(loaded.material.metallic_factor, 0.5)

    def test_each_malformed_part_is_one_error(self) -> None:
        for index, (label, path, value, expected) in enumerate(MALFORMED_PART):
            with self.subTest(label):
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.load_parts_from_path(self.write(f"part{index}.json", edited(part(), path, value)))
                self.assertIn(expected, str(caught.exception))
                self.assertIn("hand_fr1_000: ", str(caught.exception))

    def test_each_malformed_manifest_is_one_error(self) -> None:
        for index, (label, path, value, expected) in enumerate(MALFORMED_MANIFEST):
            with self.subTest(label):
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.load_parts_from_path(self.write(f"manifest{index}.json", edited(manifest(), path, value)))
                self.assertIn(expected, str(caught.exception))

    def test_a_png_given_as_base64_still_loads(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"synthetic image bytes"
        doc = edited(part(), ("material", "base_color_png"), "base64:" + base64.b64encode(png).decode("ascii"))
        (loaded,), _ = pa.load_parts_from_path(self.write("png.json", doc))
        self.assertEqual(loaded.material.base_color_png, png)

    def test_cli_prints_one_error_line_and_exits_2(self) -> None:
        script = os.path.join(TOOLS, "ped_assemble.py")
        out = os.path.join(self.folder, "out.glb")
        cases = (
            ("positions is a number", edited(part(), ("positions",), 5)),
            ("a triangle index is past the positions", edited(part(), ("triangles", 0, 2), 9)),
            ("a joint has no name", edited(manifest(), JOINT + ("name",), DELETE)),
            ("skeleton is a number", edited(manifest(), ("skeleton",), 3)),
        )
        for index, (label, doc) in enumerate(cases):
            source = self.write(f"cli{index}.json", doc)
            for command in (["assemble", source, "-o", out], ["inspect-seams", source],
                            ["cull-overlap", "--target", source, "--ref", source]):
                with self.subTest(label, command=command[0]):
                    proc = subprocess.run([sys.executable, "-I", script, *command],
                                          capture_output=True, text=True, timeout=60)
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertEqual(len(proc.stderr.strip().splitlines()), 1, proc.stderr)
                    self.assertTrue(proc.stderr.startswith("ERROR: "), proc.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
