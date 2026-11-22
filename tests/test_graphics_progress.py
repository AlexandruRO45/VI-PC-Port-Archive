"""Unit tests for tools/graphics_progress.py (the graphics-path progress page).

Every counter is run on a small synthetic tree written by the test: a few lines of made-up milestone table, CMake,
C++ and translator-shaped Python. No game data, GPU, network or SDK header is needed. Standard library only.

    python -I tests/test_graphics_progress.py
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "graphics_progress.py")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import graphics_progress as gp  # noqa: E402

MILESTONES = """# Plan

## 5. Something else

| **M9** | not a milestone | x | Met by `nothing_test` |

## 6. Boot milestones

| # | Scope | What it proves | Done when |
|---|---|---|---|
| **M0** | Device and `queue` | COM | A clear. **Synthetic part met** by `d3d12_a_test`; the rest is open |
| **M1** | Triangle | PSO | **Met.** Passes. `d3d12_b_test` and `d3d12_missing_test` |
| **M1.5** | Real code | shaders | Met by `d3d12_b_test` |
| **M2** | Textures | descriptors | **Synthetic part met** by `d3d12_a_test` |
| **M2** | Textures | descriptors | **Synthetic part met** by `d3d12_a_test` |
| **M3** | Streams | compute | Nothing yet |
| **M4** | Polish | present | **Partly met:** one piece, a | pipe | inside |

## 7. After

| **M8** | not counted | x | Met by `x_test` |
"""

CMAKE = """
add_test(NAME alpha COMMAND alpha_test)
# add_test(NAME commented_out COMMAND x)
  add_test(NAME d3d12_one COMMAND d3d12_one_test)
  add_test(NAME d3d12_m2_textured COMMAND d3d12_m2_textured_test)
  add_test(NAME vk_thing COMMAND vk_thing_test)
  add_test(NAME vk_thing COMMAND vk_thing_test)
  add_test(NAME xdxr_build COMMAND y)
  add_test(NAME run_tiny_game COMMAND z)
  add_test(NAME gta6rt_rejects_${name} COMMAND z)
add_custom_command(
  OUTPUT ${STUBS}
  COMMAND "${Python3_EXECUTABLE}" -I ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_d3d12_stubs.py --out ${STUBS}
          ${_stubs_header}
          ID3D12Alpha ID3D12Beta  # ID3D12InAComment
          ID3D12Missing
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_d3d12_stubs.py)
"""

COM_OBJECT = """
template<class Derived, class Stubs>
class Object : public Stubs {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) override { return S_OK; }
};
"""

COM_CPP = r"""
// class Fake : public ID3D12AlphaStubs { HRESULT STDMETHODCALLTYPE Hidden() override { return S_OK; } };
class Alpha final : public Object<Alpha, ID3D12AlphaStubs, ID3D12Alpha> {
public:
    HRESULT STDMETHODCALLTYPE Real(UINT a, void* (*cb)(int)) override { return call("{ not a brace"); }
    HRESULT STDMETHODCALLTYPE Refuses() override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE Empty() override {}
    HRESULT STDMETHODCALLTYPE Conditional() override { if (x) return E_NOTIMPL; return S_OK; }
};
class Beta final : public ID3D12BetaStubs {
    HRESULT STDMETHODCALLTYPE Only() override { return S_OK; }
};
class BetaSmaller final : public ID3D12BetaStubs {
};
"""

INTERFACES = {
    "ID3D12Object": {"base": "IUnknown", "methods": ["SetName"]},
    "ID3D12Alpha": {"base": "ID3D12Object", "methods": ["Real", "Refuses", "Empty", "Conditional", "Gen"]},
    "ID3D12Beta": {"base": "IUnknown", "methods": ["Only", "Other"]},
}

TRANSLATOR_SRC = '''
class Unsupported(Exception):
    pass

class Gen:
    FORMATS = {22: (4, "f32", (32,)), 1: (1, "unorm", (8,)),
               5: (1, "uint", (8,)), 5: (1, "uint", (8,)), 99: (4, "f32", (32,))}

    def fmt_field(self, lit3):
        return lit3 >> 12 & 0x7F

    def _insn(self, op):
        if op == "v_add_f32":
            return 1
        if op in ("s_mov_b32", "s_nop"):
            return 2
        if op.startswith("ds_"):
            raise Unsupported("ds form")
        if op == "s_denorm_mode":
            raise Unsupported(op)
        raise Unsupported(op)

    def image_op(self, op):
        raise Unsupported("image thing")

    def lds_op(self, op):
        raise Unsupported("lds thing")
        raise Unsupported("lds thing 2")

def helper():
    raise Unsupported("elsewhere")
    raise ValueError("not counted")
'''

GFX10 = {"0": "GFX10_FORMAT_INVALID", "1": "GFX10_FORMAT_8_UNORM", "5": "GFX10_FORMAT_8_UINT",
         "22": "GFX10_FORMAT_32_FLOAT", "23": "GFX10_FORMAT_16_16_UNORM",
         "128": "GFX10_FORMAT_8_SRGB", "129": "GFX10_FORMAT_8_8_SRGB"}

DESCRIPTORS_H = """
    {22, VK_FORMAT_R32_SFLOAT, channels_r},      // below 128: not an image-only code
    {128, VK_FORMAT_R8_SRGB, channels_r},        // GFX10_FORMAT_8_SRGB
    {128, VK_FORMAT_R8_SRGB, channels_bgr},      // the same code twice counts once
    // {129, VK_FORMAT_R8G8_SRGB, channels_rg},  commented out
"""

DEVICE_CPP = """
Format vulkan_format(DXGI_FORMAT format, bool depth) {
    switch (format) {
    case DXGI_FORMAT_R8_UNORM: return {VK_FORMAT_R8_UNORM};
    case DXGI_FORMAT_R32_FLOAT: return {VK_FORMAT_R32_SFLOAT};
    // case DXGI_FORMAT_COMMENTED: return {};
    }
}
bool vertex_format(DXGI_FORMAT format, VkFormat& out) {
    static const struct { DXGI_FORMAT f; VkFormat v; } table[] = {
        {DXGI_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UNORM},
        {DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM},
    };
}
bool unrelated() { return format == DXGI_FORMAT_R8G8_UNORM; }
"""

DXGI_LIST = ["R8_UNORM", "R8G8_UNORM", "R32_FLOAT", "R16G16_UNORM", "R16_UNORM"]

ISA_DOC = """# RDNA2 ISA coverage

## Result (translator at `abc12345`, AMD XML of 2026-08-06)

Every new rule moves these.

1,153 RDNA2 instructions: 436 handled-exact, 408 handled-family, 2 refused (`s_denorm_mode`, `s_round_mode`), 307
unknown. More text.
"""

RUNTIME_FILES = {
    "runtime/shims/d3d12_vulkan_device.cpp": DEVICE_CPP + """
HRESULT a() { return E_NOTIMPL; }
void b() { log("unsupported volume SRV format"); log("Not Implemented here"); }
// E_NOTIMPL in a comment, "unsupported" in a comment
const char* s = "plain"; /* E_NOTIMPL */
""",
    "runtime/shims/d3d12_trace.h": "x = E_NOTIMPL ? \"unsupported\" : 0;\n",
    "runtime/shims/registry.cpp": "return E_NOTIMPL; // not a graphics file\n",
    "runtime/src/vulkan_replay.h": 'void f() { fail("unsupported thing"); fail("fine"); auto r = R"(E_NOTIMPL)"; }\n',
    "runtime/shims/d3d12_com.cpp": COM_CPP,
    "runtime/shims/d3d12_com_object.h": COM_OBJECT,
    "runtime/shims/dxgi_com.cpp": "",
    "runtime/test/shaders/a.asm": "s_endpgm\n",
    "runtime/test/shaders/b.asm": "s_endpgm\n",
    "runtime/test/shaders/readme.md": "x\n",
    "runtime/test/d3d12_a_test.cpp": "",
    "runtime/test/d3d12_b_test.cpp": "",
    "runtime/CMakeLists.txt": CMAKE,
    "tests/semantic/test_a.py": "",
    "tests/semantic/test_b.py": "",
    "tests/semantic/fixtures/f1.asm": "",
    "tools/rdna2spirv.py": TRANSLATOR_SRC,
    "tools/gfx10_formats.json": json.dumps(GFX10),
    "runtime/src/gfx10_descriptors.h": DESCRIPTORS_H,
    "docs/d3d12_vulkan.md": MILESTONES,
    "docs/isa_coverage.md": ISA_DOC,
}


def write_tree(root, files=RUNTIME_FILES):
    for rel, text in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "repo")
        write_tree(self.root)
        self.tree = gp.WorkTree(self.root)
        self.data = {"d3d12": INTERFACES, "dxgi_formats": DXGI_LIST}


class Milestones(Fixture):
    def test_rows_of_section_six_only_each_id_once(self):
        result = gp.parse_milestones(MILESTONES, self.tree)
        ids = [m["id"] for m in result["items"]]
        self.assertEqual(ids, ["M0", "M1", "M1.5", "M2", "M3", "M4"])
        self.assertEqual(result["duplicate_rows"], 1)

    def test_status_is_read_from_the_done_when_words(self):
        status = {m["id"]: m["status"] for m in gp.parse_milestones(MILESTONES)["items"]}
        self.assertEqual(status, {"M0": "synthetic", "M1": "met", "M1.5": "met", "M2": "synthetic",
                                  "M3": "open", "M4": "partial"})

    def test_named_tests_are_checked_against_runtime_test(self):
        item = {m["id"]: m for m in gp.parse_milestones(MILESTONES, self.tree)["items"]}["M1"]
        self.assertEqual(item["named_tests"], ["d3d12_b_test", "d3d12_missing_test"])
        self.assertEqual(item["named_tests_found"], 1)

    def test_a_pipe_inside_the_last_cell_does_not_lose_the_row(self):
        row = [m for m in gp.parse_milestones(MILESTONES)["items"] if m["id"] == "M4"][0]
        self.assertEqual(row["scope"], "Polish")

    def test_missing_section_is_reported(self):
        self.assertEqual(gp.parse_milestones("# nothing\n"), {"found": False, "items": [], "duplicate_rows": 0})


class Interfaces(Fixture):
    def test_cmake_lists_the_generated_stub_interfaces_without_comments(self):
        self.assertEqual(gp.cmake_stub_interfaces(CMAKE), ["ID3D12Alpha", "ID3D12Beta", "ID3D12Missing"])

    def test_class_methods_ignore_comments_and_classify_bodies(self):
        classes = {c[0]: c for c in gp.class_methods(COM_CPP)}
        self.assertNotIn("Fake", classes)
        self.assertEqual(classes["Alpha"][1], "ID3D12Alpha")
        self.assertEqual(classes["Alpha"][2], {"Real": "real", "Refuses": "notimpl", "Empty": "empty",
                                               "Conditional": "real"})

    def test_counts_real_refusing_and_stub_methods_per_interface(self):
        classes = gp.class_methods(COM_CPP)
        common = {}
        for _, _, methods in gp.class_methods(COM_OBJECT):
            common.update(methods)
        report = {r["interface"]: r for r in
                  gp.count_interfaces(["ID3D12Alpha", "ID3D12Beta", "ID3D12Missing"], INTERFACES, classes, common)}
        alpha = report["ID3D12Alpha"]
        # SetName comes from the shared template; Real, Empty and Conditional are real; Refuses refuses; Gen is a stub
        self.assertEqual((alpha["methods"], alpha["real"], alpha["refuses"], alpha["stub"]), (6, 4, 1, 1))
        self.assertEqual(alpha["open"], ["ID3D12Alpha::Refuses", "ID3D12Alpha::Gen"])
        self.assertEqual(alpha["cells"], ["y", "y", "r", "y", "y", "n"])
        # two classes for Beta: the one with more overrides is the implementation
        beta = report["ID3D12Beta"]
        self.assertEqual((beta["methods"], beta["real"], beta["stub"], beta["implementation"]), (2, 1, 1, "Beta"))
        self.assertIsNone(report["ID3D12Missing"]["methods"])

    def test_an_interface_without_a_class_is_all_stub(self):
        report = gp.count_interfaces(["ID3D12Beta"], INTERFACES, [], {})
        self.assertEqual((report[0]["real"], report[0]["stub"], report[0]["implementation"]), (0, 2, None))


class Isa(Fixture):
    def test_snapshot_of_the_doc_is_parsed_across_a_line_break(self):
        snap = gp.isa_snapshot(ISA_DOC)
        self.assertEqual((snap["total"], snap["handled_exact"], snap["handled_family"], snap["refused"],
                          snap["unknown"]), (1153, 436, 408, 2, 307))
        self.assertEqual(snap["translator_commit"], "abc12345")
        self.assertIsNone(gp.isa_snapshot("no numbers here"))

    def test_live_counts_come_from_isa_coverage_on_the_translator_text(self):
        result = gp.count_isa(self.tree)
        live = result["live"]
        self.assertEqual(live["rule_names"], 3)          # v_add_f32, s_mov_b32, s_nop
        self.assertEqual(live["refused_names"], 1)       # s_denorm_mode
        self.assertEqual(live["prefix_families"], 1)     # ds_
        self.assertIsNone(result["fresh"])
        self.assertEqual(result["snapshot"]["total"], 1153)

    def test_translator_is_not_imported_or_run(self):
        before = "rdna2spirv" in sys.modules   # another test file may have imported it already under pytest
        gp.count_isa(self.tree)
        self.assertEqual("rdna2spirv" in sys.modules, before)


class Formats(Fixture):
    def test_gen_formats_reads_the_dict_keys_with_ast(self):
        self.assertEqual(gp.gen_formats(TRANSLATOR_SRC), {22, 1, 5, 99})

    def test_buffer_format_codes_against_the_json_table(self):
        g = gp.count_formats(self.tree, DXGI_LIST)["gfx10_buffer"]
        # INVALID (0) and the image-only codes 128 and 129 are not buffer codes; 99 is not in the table
        self.assertEqual((g["handled"], g["total"]), (3, 4))
        self.assertEqual(g["codes_outside_table"], [99])
        self.assertEqual([c["code"] for c in g["cells"]], [1, 5, 22, 23])
        self.assertEqual([c["name"] for c in g["cells"] if not c["handled"]], ["16_16_UNORM"])
        self.assertEqual(g["field_bits"], 7)
        self.assertTrue(g["field_bits_seen_in_translator"])
        self.assertEqual(g["codes_in_table_total"], 7)

    def test_image_only_codes_are_listed_apart_and_counted_from_the_runtime_table(self):
        im = gp.count_formats(self.tree, DXGI_LIST)["gfx10_buffer"]["image_only"]
        self.assertEqual((im["mapped"], im["total"]), (1, 2))
        self.assertEqual([(c["code"], c["mapped"]) for c in im["cells"]], [(128, True), (129, False)])

    def test_a_translator_covering_every_buffer_code_reports_all_of_them(self):
        table = {str(c): f"GFX10_FORMAT_X{c}" for c in range(1, 78)}
        table.update({"0": "GFX10_FORMAT_INVALID", "128": "GFX10_FORMAT_8_SRGB"})
        keys = ", ".join(f"{c}: (4, 'f32', (32,))" for c in range(1, 78))
        files = {"tools/gfx10_formats.json": json.dumps(table),
                 "tools/rdna2spirv.py": "class Gen:\n    FORMATS = {" + keys + "}\n"
                                        "    def f(self, w):\n        return w >> 12 & 0x7F\n"}
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(tmp, files)
            g = gp.count_formats(gp.WorkTree(tmp), [])["gfx10_buffer"]
        self.assertEqual((g["handled"], g["total"]), (77, 77))
        self.assertEqual(g["handled_outside_buffer_range"], [])

    def test_dxgi_formats_are_those_in_the_two_functions_only(self):
        d = gp.count_formats(self.tree, DXGI_LIST)["dxgi"]
        self.assertEqual((d["mapped"], d["total"]), (3, 5))
        self.assertEqual([c["name"] for c in d["cells"] if c["mapped"]], ["R8_UNORM", "R32_FLOAT", "R16G16_UNORM"])
        self.assertEqual(d["mapped_not_in_list"], [])


class Tests(Fixture):
    def test_names_are_distinct_uncommented_and_sorted_into_areas(self):
        t = gp.count_tests(CMAKE, ["a/x.asm", "a/y.asm", "a/z.md"], ["t/test_a.py", "t/fixtures/f.asm", "t/x.py"])
        self.assertEqual(t["registered"], 7)
        areas = {k: v["tests"] for k, v in t["areas"].items()}
        self.assertEqual(areas["Milestone tests (M1 to M4)"], ["d3d12_m2_textured"])
        self.assertEqual(areas["D3D12 layer on Vulkan"], ["d3d12_one"])
        self.assertEqual(areas["Vulkan backend and replay"], ["vk_thing"])
        self.assertEqual(areas["Ray tracing (XDXR, BVH)"], ["xdxr_build"])
        self.assertEqual(areas["Loader, shims and media (not graphics)"], ["alpha", "gta6rt_rejects_${name}", "run_tiny_game"])
        self.assertEqual(t["graphics_path"], 4)
        self.assertEqual((t["gpu_programs"], t["semantic_tests"], t["semantic_fixtures"]), (2, 1, 1))


class Unsupported(Fixture):
    def test_cpp_sites_skip_comments_and_raw_strings(self):
        notimpl, messages = gp.count_cpp_sites(RUNTIME_FILES["runtime/shims/d3d12_vulkan_device.cpp"])
        self.assertEqual((notimpl, messages), (1, 2))
        self.assertEqual(gp.count_cpp_sites(RUNTIME_FILES["runtime/src/vulkan_replay.h"]), (0, 1))

    def test_translator_sites_by_enclosing_function(self):
        result = gp.count_translator_sites(TRANSLATOR_SRC)
        self.assertEqual(result["total"], 7)
        self.assertEqual(result["areas"], {"Images and sampling": 1, "Instruction dispatch (_insn)": 3,
                                           "LDS and GS": 2, "Program structure and checks": 1})

    def test_runtime_areas_and_files_outside_the_graphics_path(self):
        u = gp.count_unsupported(self.tree)
        self.assertEqual(u["runtime_areas"]["D3D12 device on Vulkan"]["e_notimpl"], 1)
        self.assertEqual(u["runtime_areas"]["Vulkan backend and replay"]["messages"], 1)
        # registry.cpp is not a graphics file; d3d12_trace.h is a helper and counted nowhere
        self.assertNotIn("registry", json.dumps(u["runtime_areas"]))
        self.assertEqual(u["runtime_total"], 1 + 2 + 1 + 2)   # device, replay, and two E_NOTIMPL tokens in com


class History(unittest.TestCase):
    def test_one_point_per_date_and_sorted(self):
        points = gp.merge_history([], {"date": "2026-02-01", "x": 1})
        points = gp.merge_history(points, {"date": "2026-01-01", "x": 2})
        points = gp.merge_history(points, {"date": "2026-02-01", "x": 3})
        self.assertEqual(points, [{"date": "2026-01-01", "x": 2}, {"date": "2026-02-01", "x": 3}])

    def test_unreadable_history_is_empty(self):
        self.assertEqual(gp.read_history(os.path.join(tempfile.gettempdir(), "no-such-history.json")), [])


class EndToEnd(Fixture):
    def run_main(self, *extra):
        out = os.path.join(self.tmp.name, "out")
        code = gp.main(["--root", self.root, "--out-dir", out, *extra])
        self.assertEqual(code, 0)
        return out

    def read(self, out, name):
        with open(os.path.join(out, name), "rb") as f:
            return f.read()

    def setUp(self):
        super().setUp()
        # the interface table of the shipped data file does not know the fixture interfaces
        self._load = gp.load_data
        gp.load_data = lambda path=gp.DATA_FILE: self.data
        self.addCleanup(setattr, gp, "load_data", self._load)

    def test_three_outputs_with_the_disclaimer_first(self):
        out = self.run_main()
        for name in ("graphics.json", "graphics.html", "graphics.md"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
        md = self.read(out, "graphics.md").decode("utf-8")
        page = self.read(out, "graphics.html").decode("utf-8")
        self.assertIn(gp.DISCLAIMER, md)
        self.assertLess(page.index(gp.DISCLAIMER), page.index('class="tiles"'))
        self.assertLess(md.index(gp.DISCLAIMER), md.index("| Area |"))

    def test_html_is_self_contained(self):
        page = self.read(self.run_main(), "graphics.html").decode("utf-8")
        self.assertNotRegex(page, r"(?i)<script|<link\b|@import|url\(|https?://|src=")
        self.assertIn("prefers-color-scheme", page)

    def test_output_is_deterministic_and_has_no_timestamp_without_a_date(self):
        first = self.run_main()
        snapshot = {n: self.read(first, n) for n in ("graphics.json", "graphics.html", "graphics.md")}
        second = self.run_main()
        for name, data in snapshot.items():
            self.assertEqual(self.read(second, name), data, name)
        report = json.loads(snapshot["graphics.json"])
        self.assertNotIn("date", report["meta"])
        self.assertFalse(os.path.exists(os.path.join(first, "graphics-history.json")))
        self.assertNotIn(b"\r", snapshot["graphics.json"] + snapshot["graphics.md"])
        text = snapshot["graphics.json"].decode("ascii")
        self.assertEqual(text, json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n")

    def test_date_adds_one_history_point_and_replaces_the_same_day(self):
        out = self.run_main("--date", "2026-11-22")
        self.run_main("--date", "2026-11-22")
        self.run_main("--date", "2026-11-23")
        history = json.loads(self.read(out, "graphics-history.json"))
        self.assertEqual([p["date"] for p in history["points"]], ["2026-11-22", "2026-11-23"])
        self.assertEqual(json.loads(self.read(out, "graphics.json"))["meta"]["date"], "2026-11-23")
        self.assertIn("2026-11-22", self.read(out, "graphics.md").decode("utf-8"))

    def test_headline_numbers_of_the_fixture(self):
        report = json.loads(self.read(self.run_main(), "graphics.json"))
        tiles = {t["id"]: t for t in report["headline"]}
        self.assertEqual((tiles["milestones"]["value"], tiles["milestones"]["total"]), (2, 6))
        self.assertEqual((tiles["interfaces"]["value"], tiles["interfaces"]["total"]), (5, 8))
        self.assertEqual((tiles["isa"]["value"], tiles["isa"]["total"]), (844, 1153))
        self.assertEqual((tiles["formats"]["value"], tiles["formats"]["total"]), (3, 4))
        self.assertEqual(tiles["tests"]["value"], 4)
        self.assertEqual(tiles["unsupported"]["value"], 7 + 6)

    def test_bad_date_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gp.main(["--root", self.root, "--date", "yesterday"])


@unittest.skipUnless(shutil.which("git"), "git not installed")
class GitRef(Fixture):
    def git(self, *args):
        return subprocess.run(["git", "-C", self.root, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               "-c", "commit.gpgsign=false", *args], capture_output=True, check=True).stdout

    def test_tree_reads_the_ref_not_the_working_copy(self):
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "fixture")
        with open(os.path.join(self.root, "tools", "rdna2spirv.py"), "a", encoding="utf-8") as f:
            f.write("\ndef more():\n    raise Unsupported('uncommitted')\n")
        ref = gp.GitTree(self.root, "HEAD")
        self.assertEqual(len(ref.commit), 40)
        self.assertEqual(ref.list("runtime/test/shaders"),
                         ["runtime/test/shaders/a.asm", "runtime/test/shaders/b.asm", "runtime/test/shaders/readme.md"])
        self.assertEqual(gp.count_unsupported(ref)["translator"]["total"], 7)
        self.assertEqual(gp.count_unsupported(self.tree)["translator"]["total"], 8)
        self.assertEqual(ref.text("docs/isa_coverage.md"), ISA_DOC)
        self.assertIsNone(ref.text("no/such/file"))

    def test_unknown_ref_is_a_clean_exit(self):
        self.git("init", "-q")
        with self.assertRaises(SystemExit):
            gp.GitTree(self.root, "no-such-ref")


class NoNetworkNoExec(unittest.TestCase):
    ALLOWED_IMPORTS = {"argparse", "ast", "html", "json", "os", "re", "subprocess", "sys", "tempfile",
                       "isa_coverage", "gen_d3d12_stubs"}

    def setUp(self):
        with open(TOOL, encoding="utf-8") as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def test_only_standard_library_and_sibling_tools_are_imported(self):
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertLessEqual(names, self.ALLOWED_IMPORTS)
        self.assertNotIn("rdna2spirv", names)

    def test_no_dynamic_execution_and_subprocess_only_runs_git(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    self.assertNotIn(fn.id, {"exec", "eval", "compile", "__import__"}, node.lineno)
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "subprocess":
                    self.assertEqual(fn.attr, "run", node.lineno)
                    first = node.args[0]
                    self.assertIsInstance(first, ast.List)
                    self.assertEqual(first.elts[0].value, "git")
                if isinstance(fn, ast.Attribute) and fn.attr in {"system", "popen", "spawn", "urlopen", "connect"}:
                    self.fail(f"line {node.lineno}: {fn.attr}")

    def test_a_full_run_opens_no_socket_and_loads_no_measured_module(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = os.path.join(tmp.name, "repo")
        write_tree(root)
        real = socket.socket

        def refuse(*args, **kwargs):
            raise AssertionError("the tool opened a socket")

        socket.socket = refuse
        self.addCleanup(setattr, socket, "socket", real)
        before = set(sys.modules)
        code = gp.main(["--root", root, "--out-dir", os.path.join(tmp.name, "out")])
        self.assertEqual(code, 0)
        loaded = set(sys.modules) - before   # modules this run added
        self.assertFalse({"rdna2spirv", "urllib.request", "http.client"} & loaded)


class StdoutGuard(unittest.TestCase):
    def test_main_block_reconfigures_stdout_with_replace(self):
        with open(TOOL, encoding="utf-8") as f:
            text = f.read()
        block = text[re.search(r"^if __name__ == ['\"]__main__['\"]:\n", text, re.M).end():]
        self.assertIn('sys.stdout.reconfigure(errors="replace")', block[:400])
        self.assertIn("except (AttributeError, ValueError, OSError):", block[:400])

    def test_a_path_outside_the_code_page_does_not_crash_a_piped_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "repo-")
            write_tree(root)
            out = os.path.join(tmp, "out-")
            result = subprocess.run([sys.executable, "-I", TOOL, "--root", root, "--out-dir", out, "--print"],
                                    capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            self.assertNotIn("UnicodeEncodeError", result.stderr.decode("utf-8", "replace"))
            self.assertTrue(os.path.isfile(os.path.join(out, "graphics.json")))


class RealTree(unittest.TestCase):
    """The shipped data file and the real repository, as far as it can be read without the game."""

    def test_data_file_has_the_layer_interfaces(self):
        data = gp.load_data()
        for name in ("ID3D12Device11", "ID3D12GraphicsCommandList8", "ID3D12Resource", "ID3D12Object"):
            self.assertIn(name, data["d3d12"])
        self.assertIn("R8G8B8A8_UNORM", data["dxgi_formats"])
        self.assertNotIn("UNKNOWN", data["dxgi_formats"])

    def test_refresh_lists_each_method_once_and_drops_unknown_and_force_uint(self):
        header = ('MIDL_INTERFACE("00000000-0000-0000-0000-000000000001")\n'
                  "ID3D12Thing : public IUnknown\n{\npublic:\n"
                  "    virtual D3D12_DESC STDMETHODCALLTYPE GetDesc( void) = 0;\n"
                  "    virtual void STDMETHODCALLTYPE GetDesc( _Out_ D3D12_DESC *out) = 0;\n"
                  "    virtual void STDMETHODCALLTYPE Other( void) = 0;\n};\n")
        dxgi = "DXGI_FORMAT_UNKNOWN = 0,\nDXGI_FORMAT_R32_FLOAT = 41,\nDXGI_FORMAT_A8_UNORM = 65,\n" \
               "DXGI_FORMAT_FORCE_UINT = 0xffffffff\n"
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, text in (("d3d12.h", header), ("dxgiformat.h", dxgi)):
                paths[name] = os.path.join(tmp, name)
                with open(paths[name], "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            out = os.path.join(tmp, "data.json")
            self.assertEqual(gp.refresh_data(paths["d3d12.h"], paths["dxgiformat.h"], out), (1, 2))
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["d3d12"]["ID3D12Thing"], {"base": "IUnknown", "methods": ["GetDesc", "Other"]})
        self.assertEqual(data["dxgi_formats"], ["R32_FLOAT", "A8_UNORM"])

    def test_data_file_holds_names_only(self):
        data = gp.load_data()
        self.assertEqual(set(data), {"about", "source", "d3d12", "dxgi_formats"})
        self.assertIn("Windows SDK", data["source"])
        for name, entry in data["d3d12"].items():
            self.assertRegex(name, r"^ID3D12\w+$")
            self.assertEqual(set(entry), {"base", "methods"})
            self.assertRegex(entry["base"], r"^\w+$")
            self.assertTrue(all(re.fullmatch(r"[A-Za-z_]\w*", m) for m in entry["methods"]), name)
            self.assertEqual(len(entry["methods"]), len(set(entry["methods"])), name)
        self.assertTrue(all(re.fullmatch(r"[A-Z0-9_]+", n) for n in data["dxgi_formats"]))

    def test_every_interface_cmake_builds_is_in_the_data_file(self):
        with open(os.path.join(ROOT, "runtime", "CMakeLists.txt"), encoding="utf-8") as f:
            leaves = gp.cmake_stub_interfaces(f.read())
        self.assertGreaterEqual(len(leaves), 10)
        self.assertEqual([n for n in leaves if n not in gp.load_data()["d3d12"]], [])

    def test_the_real_milestone_table_parses(self):
        with open(os.path.join(ROOT, "docs", "d3d12_vulkan.md"), encoding="utf-8") as f:
            result = gp.parse_milestones(f.read())
        self.assertTrue(result["found"])
        self.assertGreaterEqual(len(result["items"]), 8)


if __name__ == "__main__":
    unittest.main()
