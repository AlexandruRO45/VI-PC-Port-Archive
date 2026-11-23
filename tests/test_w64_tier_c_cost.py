"""w64_tier_c_cost.py (SHADER-836): the count of pixels whose partition-dependent words move between two draws, and the
references it accepts. The GPU draws are replaced by stubs, so this needs no GPU, LLVM or Vulkan SDK."""
import contextlib
import io
import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, SHADERS)
import w64_graphics as g  # noqa: E402
import w64_tier_c_cost as cost  # noqa: E402

N = g.W * g.H


def image(name, dep_value, other_value=None):
    """A draw of `name`: partition-dependent words = dep_value, the others what the fixture derives."""
    out = []
    for i in range(N):
        want = g.FIXTURES[name][1](i)
        out.append(tuple(dep_value if w is None else (w if other_value is None else other_value) for w in want))
    return out


def test_only_the_tier_c_pixel_fixtures_are_measured():
    names = cost.tier_c_pixel_fixtures()
    assert names and all(n.startswith("ps_w64_c_") for n in names)
    assert "vs_w64_c_lane_id" not in names and "gs_w64_c_lane_id" not in names


def test_dependent_words_are_the_none_entries_of_the_header():
    assert cost.dependent_words("ps_w64_c_bcnt_export") == [0]
    assert cost.dependent_words("ps_w64_c_readlane") == [0, 2]


def test_moved_counts_pixels_not_words_and_separates_the_stable_words():
    name = "ps_w64_c_readlane"
    a, b = image(name, 1), image(name, 1)
    assert cost.moved(name, a, b) == (0, 0)
    b[5] = (b[5][0] + 1, b[5][1], b[5][2] + 1, b[5][3])        # two dependent words in one pixel: one pixel
    assert cost.moved(name, a, b) == (1, 0)
    b[6] = (b[6][0], b[6][1] + 1, b[6][2], b[6][3])            # a word that must not move
    assert cost.moved(name, a, b) == (1, 1)


def run_main(argv, draws):
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as build:
        spirv = os.path.join(build, "spirv")
        os.makedirs(spirv)
        for n in cost.tier_c_pixel_fixtures():
            Path(spirv, n + ".spv").write_bytes(b"")
        with patch.object(g, "draw", draws), patch.object(g, "live_width", lambda *a: 16), \
                contextlib.redirect_stdout(out):
            code = cost.main([build, build] + argv)
    return code, out.getvalue()


def test_required_sizes_compare_64_with_32():
    def draws(tools, spirv, name, size, tmp):
        return image(name, 7 if size == 64 else 9), None, True

    code, text = run_main([], draws)
    assert code == 0
    assert "required 64 vs required 32" in text
    assert text.count("required 64 vs required 32") == len(cost.tier_c_pixel_fixtures())
    assert f"{N} of {N} (100.0%)" in text                      # a fixture whose every pixel depends on the partition
    assert f"{N - 1} of {N}" in text                           # the "only pixel (5, 7)" fixtures fix that pixel's word
    assert "live lanes per fragment subgroup: 16" in text


def test_a_device_that_cannot_require_64_is_skipped_not_failed():
    def draws(tools, spirv, name, size, tmp):
        return image(name, 7), None, size is None

    code, text = run_main([], draws)
    assert code == 0 and text.count("skipped: this device cannot require 64 lanes") == len(cost.tier_c_pixel_fixtures())


def test_ref_directory_is_read_and_a_saved_draw_round_trips():
    with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as saved:
        for n in cost.tier_c_pixel_fixtures():
            Path(ref, n + ".ref64.raw").write_bytes(b"".join(struct.pack("<4I", *w) for w in image(n, 3)))

        def draws(tools, spirv, name, size, tmp):
            return image(name, 4), None, None

        code, text = run_main(["--ref", ref, "--save", saved], draws)
        assert code == 0 and "64-lane run from --ref" in text
        assert len(os.listdir(saved)) == len(cost.tier_c_pixel_fixtures())
        # a saved default draw is accepted as a reference, but is not called a 64-lane run
        code, text = run_main(["--ref", saved], draws)
        assert code == 0 and "(partition unknown)" in text
        assert text.count(f"    0 of {N} (  0.0%)") == len(cost.tier_c_pixel_fixtures())


def test_a_missing_reference_file_and_a_failed_draw_are_failures():
    with tempfile.TemporaryDirectory() as ref:
        code, text = run_main(["--ref", ref], lambda *a: (image("ps_w64_c_readlane", 1), None, None))
        assert code == 1 and "no ps_w64_c_" in text

    code, text = run_main([], lambda *a: (None, "exit 1: boom", None))
    assert code == 1 and "boom" in text


def test_no_arguments_prints_the_usage():
    with pytest.raises(SystemExit) as e:
        cost.main([])
    assert "w64_tier_c_cost.py" in str(e.value)
