"""The inventory table in docs/embedded_shaders.md, measured on the package in LEONIDA_GAME_DIR.

    python -I tests/test_embedded_shader_inventory.py

Section 2 of the page counts the DXBC containers embedded in the package's DLLs and in the executable: per image its
container count, its byte total, and its smallest / median / largest container. Links, flags and ids on that page have
tests that read it; these numbers are a measurement, so this one re-measures them: it scans the same images with
`tools/pe_shader_containers.inspect` and compares the table row by row, plus the "other N DLLs" row, which is the rest
of the package folder and has to add up to no container. Prints SKIP and exits 0 when LEONIDA_GAME_DIR is unset.
Counts only: no container is written, and none is quoted.
"""
import os
import re
import statistics
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAGE = os.path.join(ROOT, "docs", "embedded_shaders.md")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import pe_shader_containers as psc  # noqa: E402

GAME_DIR = os.environ.get("LEONIDA_GAME_DIR")
NAMED = ("XboxBvhBuilder.dll", "amd_fidelityfx_upscaler_xbox.dll", "game_scarlett_master.exe")
ROW = re.compile(r"^\| `([^`]+)` \| ([\d,]+) \| ([\d,]+) \| ([^|]+)\|")
OTHER_DLLS = re.compile(r"^\| other (\d+) DLLs \| ([\d,]+) \| ([\d,]+) \|")


def number(text):
    return int(text.replace(",", ""))


def spread(sizes):
    """The page's `min / median / max` cell for a sorted size list."""
    if not sizes:
        return "\u2014"
    middle = statistics.median(sizes)
    median = f"{middle:,.0f}" if float(middle).is_integer() else f"{middle:,.1f}"
    return f"{sizes[0]:,} / {median} / {sizes[-1]:,}"


def inventory():
    """{image: (containers, bytes, "min / median / max")} from the page's section 2 table, and the other-DLLs row."""
    rows, other = {}, None
    with open(PAGE, encoding="utf-8") as page:
        for line in page:
            match = ROW.match(line)
            if match:
                rows[match.group(1)] = (number(match.group(2)), number(match.group(3)), match.group(4).strip())
            match = OTHER_DLLS.match(line)
            if match:
                other = (number(match.group(1)), number(match.group(2)), number(match.group(3)))
    return rows, other


def measure(path):
    """(containers, bytes, "min / median / max") of the image at `path`, with the tool the page names."""
    with open(path, "rb") as f:
        data = f.read()
    found, _ = psc.inspect(data, os.path.basename(path))
    sizes = sorted(record["size"] for record in found)
    return len(sizes), sum(sizes), spread(sizes)


class InventoryTable(unittest.TestCase):
    @unittest.skipUnless(GAME_DIR, "set LEONIDA_GAME_DIR to the package folder to measure the page")
    def test_each_named_image_matches_the_package(self):
        rows, _ = inventory()
        self.assertEqual(sorted(rows), sorted(NAMED), "the page's per-image rows")
        for image in NAMED:
            with self.subTest(image=image):
                self.assertEqual(measure(os.path.join(GAME_DIR, image)), rows[image])

    @unittest.skipUnless(GAME_DIR, "set LEONIDA_GAME_DIR to the package folder to measure the page")
    def test_the_other_dlls_add_up_to_no_container(self):
        _, other = inventory()
        self.assertIsNotNone(other, "the page's 'other N DLLs' row")
        dlls = sorted(name for name in os.listdir(GAME_DIR)
                      if name.lower().endswith(".dll") and name not in NAMED)
        mine = [measure(os.path.join(GAME_DIR, name)) for name in dlls]
        self.assertEqual((len(dlls), sum(m[0] for m in mine), sum(m[1] for m in mine)), other)


if __name__ == "__main__":
    unittest.main()
