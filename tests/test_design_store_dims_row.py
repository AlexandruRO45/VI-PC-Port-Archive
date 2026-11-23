"""docs/design.md's "Typed image stores" row names every dimension the store path serves.

    python -I tests/test_design_store_dims_row.py

The translator contract table is where every new rule is recorded, with how it was confirmed (CONTRIBUTING.md,
AGENTS.md). A row that lists fewer dimensions than `Gen.STORE_DIMS` plus the aliases `Gen.image_store` maps onto
one of them reads as a refusal that no longer exists: the row said "(2D, 2D array, 3D)" while the store path also
served 1D (`STORE_DIMS` index 3, bindings 17-19) and, since the cube alias, `SQ_RSRC_IMG_CUBE` (bound as a
six-layer 2D array, `runtime/src/vulkan_bindless.h` "A cube is a 2D array of six layers"). The dimension names are
read from the translator, so a dimension added there cannot stay undocumented here.

Standard library only; no game data.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

DESIGN = os.path.join(ROOT, "docs", "design.md")
ROW_NAME = "Typed image stores"


def row_text(text, name):
    """The whole line of the table row whose first cell is `name`, or None."""
    for line in text.splitlines():
        if line.startswith("|") and line.split("|")[1].strip() == name:
            return line
    return None


def store_dimension_names():
    """Every dimension `image_store` accepts, as the row spells it: STORE_DIMS plus the aliases it maps onto one.

    STORE_DIMS' keys are the instruction's `dim:SQ_RSRC_IMG_*` names; the aliases are the extra `dim:` names in
    `image_store`'s own source, which it serves through one of those entries.
    """
    names = set()
    for dim in rdna2spirv.Gen.STORE_DIMS:
        names.add(dim.split("SQ_RSRC_IMG_")[-1].replace("_", " "))
    source = rdna2spirv.Gen.image_store.__code__.co_consts          # the literals the method compares against
    for const in source:
        if isinstance(const, str):
            names.update(m.split("SQ_RSRC_IMG_")[-1].replace("_", " ") for m in re.findall(r"dim:SQ_RSRC_IMG_\w+",
                                                                                            const))
    return sorted(names)


def unnamed_dimensions(text, names):
    """[name] for every dimension in `names` that `text` does not name (case-insensitive)."""
    lowered = text.lower()
    return [n for n in names if n.lower() not in lowered]


class StoreDimsRow(unittest.TestCase):
    def test_the_row_names_every_store_dimension(self):
        with open(DESIGN, encoding="utf-8") as f:
            row = row_text(f.read(), ROW_NAME)
        self.assertIsNotNone(row, f"docs/design.md has no '{ROW_NAME}' row")
        names = store_dimension_names()
        self.assertIn("CUBE", names, "the cube alias left image_store: drop it from the row and this test")
        self.assertEqual(unnamed_dimensions(row, names), [],
                         "docs/design.md's 'Typed image stores' row does not name these dimensions that "
                         "image_store serves (add them, with how each was confirmed)")

    def test_a_dimension_missing_from_a_row_is_reported(self):
        row = "| Typed image stores | `image_store` (2D, 2D array, 3D) writes the dmask components. |"
        self.assertEqual(unnamed_dimensions(row, ["2D", "2D ARRAY", "3D", "1D", "CUBE"]), ["1D", "CUBE"])


if __name__ == "__main__":
    unittest.main()
