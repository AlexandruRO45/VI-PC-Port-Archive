#!/usr/bin/env python3
"""Progress of the graphics path, counted from the committed code and nothing else.

The graphics path is the Xbox D3D12 -> Vulkan layer under runtime/ and the RDNA2 -> SPIR-V translator
tools/rdna2spirv.py. This tool reads their source text and writes a page (HTML and Markdown) and a JSON file with
the raw numbers. It says nothing about whether the game runs: it never builds, runs or imports the code it measures.

    python -I tools/graphics_progress.py                       # the working tree -> docs/progress/graphics.*
    python -I tools/graphics_progress.py --tree origin/main    # a git ref, read with git show / git ls-tree
    python -I tools/graphics_progress.py --date 2026-11-22     # also append that day to the history file

What is counted (every number on the page says how, next to it):
  milestones   the table in docs/d3d12_vulkan.md section 6, one status per row, read from the row's own words;
  interfaces   per D3D12 interface of the layer, the methods of its chain (names from tools/graphics_progress_data.json,
               taken from the Windows SDK d3d12.h) against the overrides found in the class that implements it:
               a real body, a body that only returns E_NOTIMPL, or no override (a generated stub);
  RDNA2 ISA    tools/isa_coverage.py used as a library on the translator text; the instruction total comes from AMD's
               XML, which is not in the repository, so the share is the one docs/isa_coverage.md states (or a fresh
               one with --isa);
  formats      GFX10 buffer format codes the translator decodes (Gen.FORMATS) out of tools/gfx10_formats.json, and
               DXGI formats the layer maps to Vulkan out of the DXGI list in tools/graphics_progress_data.json;
  tests        add_test() names in runtime/CMakeLists.txt, per area (registered, not passing);
  unsupported  places that refuse work: raise Unsupported in the translator, E_NOTIMPL and "unsupported" / "not
               implemented" log strings in the runtime.

Standard library only. The output is deterministic: sorted keys, LF line endings, no clock. A date is written only
when --date is given. Needs no game data, no GPU and no network.
"""
import argparse
import ast
import html
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_FILE = os.path.join(HERE, "graphics_progress_data.json")
SCHEMA = 1

DISCLAIMER = "Counted from the code; says nothing about whether the game runs."

MILESTONE_DOC = "docs/d3d12_vulkan.md"
ISA_DOC = "docs/isa_coverage.md"
TRANSLATOR = "tools/rdna2spirv.py"
CMAKE = "runtime/CMakeLists.txt"
GFX10_JSON = "tools/gfx10_formats.json"
VK_DEVICE = "runtime/shims/d3d12_vulkan_device.cpp"
# Sources that implement the D3D12 COM objects; the common one carries ID3D12Object and GetDevice for every object.
COM_SOURCES = ["runtime/shims/d3d12_com.cpp", "runtime/shims/d3d12_vulkan_device.cpp", "runtime/shims/dxgi_com.cpp"]
COM_COMMON = ["runtime/shims/d3d12_com_object.h"]

# --------------------------------------------------------------------------------------------------------------
# Reading a tree: the working directory, or a git ref through git show / git ls-tree.
# --------------------------------------------------------------------------------------------------------------


def _decode(data):
    return data.decode("utf-8", "replace").replace("\r\n", "\n")


class WorkTree:
    label = "working tree"
    commit = None

    def __init__(self, root):
        self.root = root

    def text(self, path):
        try:
            with open(os.path.join(self.root, *path.split("/")), "rb") as f:
                return _decode(f.read())
        except OSError:
            return None

    def list(self, prefix):
        base = os.path.join(self.root, *prefix.split("/"))
        out = []
        for folder, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in (".git", "build", "__pycache__"))
            rel = os.path.relpath(folder, self.root).replace(os.sep, "/")
            out.extend(f"{rel}/{name}" for name in sorted(files))
        return sorted(out)


class GitTree:
    def __init__(self, root, ref):
        self.root, self.ref = root, ref
        self.label = ref
        self.commit = self._git("rev-parse", "--verify", ref + "^{commit}").decode("ascii").strip()

    def _git(self, *args):
        result = subprocess.run(["git", "-C", self.root, *args], capture_output=True)
        if result.returncode != 0:
            raise SystemExit(f"git {' '.join(args)}: {result.stderr.decode('utf-8', 'replace').strip()}")
        return result.stdout

    def text(self, path):
        result = subprocess.run(["git", "-C", self.root, "show", f"{self.commit}:{path}"], capture_output=True)
        return _decode(result.stdout) if result.returncode == 0 else None

    def list(self, prefix):
        out = self._git("ls-tree", "-r", "--name-only", "-z", self.commit, "--", prefix)
        return sorted(p for p in out.decode("utf-8", "replace").split("\0") if p)


# --------------------------------------------------------------------------------------------------------------
# C++ text helpers: mask comments and literals, match braces. Nothing here compiles or runs anything.
# --------------------------------------------------------------------------------------------------------------

_LITERALS = re.compile(
    r'//[^\n]*|/\*.*?\*/|R"([^\s()\\]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\\n])*"'
    r"|(?<![0-9A-Za-z_])'(?:\\.|[^'\\\n])*'", re.S)


def mask_cpp(text):
    """The same text with comments, strings and character literals blanked (newlines and length kept)."""
    return _LITERALS.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def match_close(masked, start, open_ch, close_ch):
    """Index of the bracket closing the one at masked[start], or -1."""
    depth = 0
    for i in range(start, len(masked)):
        c = masked[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


_CLASS = re.compile(r"\b(?:class|struct)\s+(\w+)\s*(?:final\s*)?:\s*([^{};]*)\{")
_LEAF = re.compile(r"\b(ID3D12\w+?)Stubs\b")
_METHOD = re.compile(r"\bSTDMETHODCALLTYPE\s+(\w+)\s*\(")
_TRIVIAL_NOTIMPL = re.compile(r"return\s+E_NOTIMPL\s*;")


def class_methods(text):
    """[(class name, leaf interface or None, {method: 'real' | 'notimpl' | 'empty'})] for the classes in a source.

    A class counts as the implementation of an interface when its base list names `<Interface>Stubs` (the generated
    stub class of tools/gen_d3d12_stubs.py). A method is `notimpl` when its body only returns E_NOTIMPL and `empty`
    when the body is `{}`; a declaration without a body here is taken as defined elsewhere (`real`)."""
    masked = mask_cpp(text)
    classes = []
    for m in _CLASS.finditer(masked):
        end = match_close(masked, m.end() - 1, "{", "}")
        if end < 0:
            continue
        body = masked[m.end():end]
        leaf = _LEAF.search(m.group(2))
        classes.append((m.group(1), leaf.group(1) if leaf else None, _methods_in(body)))
    return classes


def _methods_in(body):
    methods = {}
    for m in _METHOD.finditer(body):
        close = match_close(body, m.end() - 1, "(", ")")
        if close < 0:
            continue
        rest = re.match(r"(?:\s|const|noexcept|override|final)*", body[close + 1:]).end() + close + 1
        kind = "real"
        if body[rest:rest + 1] == "{":
            stop = match_close(body, rest, "{", "}")
            inner = body[rest + 1:stop].strip() if stop > 0 else "x"
            if inner == "":
                kind = "empty"
            elif _TRIVIAL_NOTIMPL.fullmatch(inner):
                kind = "notimpl"
        elif re.match(r"=\s*delete", body[rest:]):
            kind = "notimpl"
        # a method seen twice (overloads, #if branches): the real one wins
        if methods.get(m.group(1)) in (None, "notimpl", "empty") or kind == "real":
            methods[m.group(1)] = kind
    return methods


def find_function(masked, name):
    """The body text of the first function called `name` in masked source, or None."""
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", masked):
        close = match_close(masked, m.end() - 1, "(", ")")
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", masked[close + 1:]) if close > 0 else None
        if tail:
            start = close + 1 + tail.end() - 1
            stop = match_close(masked, start, "{", "}")
            if stop > 0:
                return masked[start:stop]
    return None


# --------------------------------------------------------------------------------------------------------------
# The counters. Each takes text (or a tree) and returns plain data.
# --------------------------------------------------------------------------------------------------------------

MET_RULES = [  # first match wins; the words are the ones the milestone rows use
    ("partial", re.compile(r"\bPartly met\b", re.I)),
    ("synthetic", re.compile(r"\bSynthetic part met\b", re.I)),
    ("met", re.compile(r"\*\*Met\b|\bMet by\b")),
]
STATUS_LABEL = {"met": "met", "synthetic": "synthetic part met", "partial": "partly met", "open": "open"}


def parse_milestones(markdown, tree=None):
    """Milestone rows of the section titled 'Boot milestones' (a `| **M0** | scope | proves | done when |` table)."""
    rows, seen, duplicates = [], set(), 0
    lines = markdown.split("\n")
    start = next((i for i, line in enumerate(lines) if re.match(r"##\s+(?:\d+\.\s*)?Boot milestones", line)), None)
    if start is None:
        return {"found": False, "items": [], "duplicate_rows": 0}
    for line in lines[start + 1:]:
        if re.match(r"##\s", line):
            break
        m = re.match(r"\|\s*\*\*(M[\d.]+)\*\*\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) < 3:
            continue
        mid, scope, proves, done = m.group(1), cells[0], cells[1], "|".join(cells[2:])
        if mid in seen:
            duplicates += 1
            continue
        seen.add(mid)
        status = next((s for s, rx in MET_RULES if rx.search(done)), "open")
        named = sorted(set(re.findall(r"`([a-z][a-z0-9_]*_test)`", done)))
        entry = {"id": mid, "status": status, "scope": _plain(scope), "named_tests": named}
        if tree is not None:
            entry["named_tests_found"] = sum(1 for t in named if tree.text(f"runtime/test/{t}.cpp") is not None)
        rows.append(entry)
    return {"found": True, "items": rows, "duplicate_rows": duplicates}


def _plain(text, limit=170):
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def cmake_stub_interfaces(cmake):
    """The interface names handed to tools/gen_d3d12_stubs.py in the CMake file (the interfaces the layer builds)."""
    cmake = re.sub(r"#[^\n]*", "", cmake)
    m = re.search(r"gen_d3d12_stubs\.py\s+--out(.*?)DEPENDS", cmake, re.S)
    return sorted(set(re.findall(r"\bID3D12\w+\b", m.group(1)))) if m else []


def chain_methods(interfaces, leaf):
    """Ordered [(owner, method)] along the base chain of leaf, base first, IUnknown left out."""
    names, name = [], leaf
    while name in interfaces and name not in names:
        names.append(name)
        name = interfaces[name].get("base")
    out = []
    for owner in reversed(names):
        out.extend((owner, method) for method in interfaces[owner]["methods"])
    return out


def count_interfaces(leaves, interfaces, class_list, common):
    """Per leaf: methods, real, refuses (E_NOTIMPL body), stub (no override), and the open names."""
    report = []
    for leaf in leaves:
        if leaf not in interfaces:
            report.append({"interface": leaf, "methods": None, "real": None, "refuses": None, "stub": None,
                           "implementation": None, "open": []})
            continue
        candidates = [c for c in class_list if c[1] == leaf]
        best = max(candidates, key=lambda c: (len(c[2]), c[0]), default=None)
        own = best[2] if best else {}
        real = refuses = stub = 0
        open_names = []
        for owner, method in chain_methods(interfaces, leaf):
            state = common.get(method) or own.get(method)
            if own.get(method) == "real" or common.get(method) == "real":
                state = "real"
            if state in ("real", "empty"):
                real += 1
            elif state == "notimpl":
                refuses += 1
                open_names.append(f"{owner}::{method}")
            else:
                stub += 1
                open_names.append(f"{owner}::{method}")
        total = real + refuses + stub
        report.append({"interface": leaf, "methods": total, "real": real, "refuses": refuses, "stub": stub,
                       "implementation": best[0] if best else None, "open": open_names,
                       "cells": [_cell_state(common, own, m) for _, m in chain_methods(interfaces, leaf)]})
    return report


def _cell_state(common, own, method):
    if own.get(method) == "real" or common.get(method) == "real":
        return "y"
    state = common.get(method) or own.get(method)
    return {"real": "y", "empty": "y", "notimpl": "r"}.get(state, "n")


def isa_snapshot(markdown):
    """The counts docs/isa_coverage.md states for its translator commit, or None."""
    flat = re.sub(r"\s+", " ", markdown or "")
    head = re.search(r"## Result \(translator at `([0-9a-f]+)`, AMD XML of ([\d-]+)\)", flat)
    m = re.search(r"([\d,]+) RDNA2 instructions: ([\d,]+) handled-exact, ([\d,]+) handled-family, ([\d,]+) refused"
                  r"(?: \([^)]*\))?, ([\d,]+) unknown", flat)
    if not m:
        return None
    n = [int(g.replace(",", "")) for g in m.groups()]
    return {"total": n[0], "handled_exact": n[1], "handled_family": n[2], "refused": n[3], "unknown": n[4],
            "translator_commit": head.group(1) if head else None, "xml_date": head.group(2) if head else None}


def load_isa_coverage():
    """tools/isa_coverage.py of this checkout, used as a library (never the translator itself)."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import isa_coverage
    return isa_coverage


def count_isa(tree, isa_xml=None):
    ic = load_isa_coverage()
    source = tree.text(TRANSLATOR)
    if source is None:
        return {"found": False}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rdna2spirv.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(source)
        scan = ic.scan_translator(path)
        # Names compared or tabled, minus those in a test whose whole body refuses. (isa_coverage's accepted_names()
        # drops names held in a tuple whenever the file has such a test, so it is not used for the live count.)
        refused = set(scan["refused"])
        accepted = set(scan["exact"]) - refused
        live = {"rule_names": len(accepted), "refused_names": len(refused),
                "branches_that_raise": len(scan["partial"]), "prefix_families": len(scan["prefixes"])}
        full = None
        if isa_xml:
            report = ic.build_report(ic.read_isa(isa_xml), scan, ic.accepted_names(path))
            t = report["totals"]
            full = {"total": report["instructions"], "handled_exact": t["handled-exact"],
                    "handled_family": t["handled-family"], "refused": t["refused"], "unknown": t["unknown"],
                    "architecture": report["architecture"]}
    snapshot = isa_snapshot(tree.text(ISA_DOC))
    return {"found": True, "live": live, "snapshot": snapshot, "fresh": full}


def gen_formats(source):
    """Format codes (ints) of the dict assigned to FORMATS in the translator's Gen class, read with ast."""
    codes = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "Gen":
            for item in node.body:
                if (isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FORMATS"
                                                         for t in item.targets) and isinstance(item.value, ast.Dict)):
                    codes.update(k.value for k in item.value.keys if isinstance(k, ast.Constant)
                                 and isinstance(k.value, int))
    return codes


def count_formats(tree, dxgi_list):
    names = tree.text(GFX10_JSON)
    translator = tree.text(TRANSLATOR)
    gfx = None
    if names is not None and translator is not None:
        table = {int(k): v for k, v in json.loads(names).items()}
        handled = gen_formats(translator)
        codes = sorted(c for c, n in table.items() if not n.endswith("_INVALID"))
        cells = [{"code": c, "name": table[c].replace("GFX10_FORMAT_", ""), "handled": c in handled} for c in codes]
        gfx = {"total": len(codes), "handled": sum(1 for c in cells if c["handled"]),
               "codes_outside_table": sorted(handled - set(table)), "cells": cells}
    dxgi = None
    device = tree.text(VK_DEVICE)
    if device is not None and dxgi_list:
        masked = mask_cpp(device)
        mapped = set()
        for fn in ("vulkan_format", "vertex_format"):
            body = find_function(masked, fn)
            if body:
                mapped.update(re.findall(r"\bDXGI_FORMAT_(\w+)", body))
        known = set(dxgi_list)
        cells = [{"name": n, "mapped": n in mapped} for n in dxgi_list]
        dxgi = {"total": len(dxgi_list), "mapped": sum(1 for c in cells if c["mapped"]),
                "mapped_not_in_list": sorted(mapped - known), "cells": cells}
    return {"gfx10_buffer": gfx, "dxgi": dxgi}


TEST_AREAS = [  # first match wins; (area, graphics path?, name pattern)
    ("Milestone tests (M1 to M4)", True, r"d3d12_(m\d.*|first_light|translated_draw)$"),
    ("Ray tracing (XDXR, BVH)", True, r"(d3d12_)?(xdxr.*|bvh_builder|d3d12_indirect_rt)$"),
    ("Presentation", True, r"(xbox_present|d3d12_present|dxgi_swapchain|vk_presenter_sync|frame_pacer)$"),
    ("Xbox proxies and root signatures", True,
     r"(xs_proxy.*|device_proxy|queue_proxy|allocator_proxy|root_sig.*|root_signature.*|d3d12_xs_root_signature)$"),
    ("D3D12 layer on Vulkan", True, r"d3d12_.*"),
    ("Vulkan backend and replay", True,
     r"(vk_.*|replay_.*|pipeline_cache.*|retire_tickets|bindless_sizing|depth24_convert|predicated_copy|typeless_rtv"
     r"|clear_color|subgroup_size|vulkan_device_filter|gfx10_descriptors)$"),
    ("Command lists, descriptors, shaders (CPU)", True,
     r"(command_list|resource_tiling|descriptor_table|shader_library.*|fsr_dispatch|runtime_types)$"),
    ("Loader, shims and media (not graphics)", False, r".*"),
]


def count_tests(cmake, shader_files, semantic_files):
    code = re.sub(r"#[^\n]*", "", cmake or "")
    names = sorted(set(re.findall(r"add_test\s*\(\s*NAME\s+([^\s)]+)", code)))
    areas = {}
    for name in names:
        for area, graphics, pattern in TEST_AREAS:
            if re.fullmatch(pattern, name):
                entry = areas.setdefault(area, {"graphics": graphics, "tests": []})
                entry["tests"].append(name)
                break
    graphics_total = sum(len(a["tests"]) for a in areas.values() if a["graphics"])
    return {"registered": len(names), "graphics_path": graphics_total,
            "areas": {k: {"graphics": v["graphics"], "count": len(v["tests"]), "tests": v["tests"]}
                      for k, v in sorted(areas.items())},
            "gpu_programs": sum(1 for p in shader_files if p.endswith(".asm")),
            "semantic_tests": sum(1 for p in semantic_files if re.search(r"/test_[^/]*\.py$", p)),
            "semantic_fixtures": sum(1 for p in semantic_files if "/fixtures/" in p)}


RUNTIME_AREAS = [  # first match wins; None = not a graphics-path file
    ("D3D12 device on Vulkan", r"runtime/shims/d3d12_vulkan_device\.cpp$"),
    (None, r"runtime/shims/d3d12_trace\.h$"),
    ("Command lists, queues, fences", r"runtime/(shims/d3d12_com.*|shims/d3d12_fence_waiter\.h|src/command_list\.h)$"),
    ("Xbox device and queue proxies", r"runtime/shims/d3d12_xs.*"),
    ("Presentation (DXGI, PresentX)", r"runtime/(shims/(dxgi_.*|xbox_present.*)|src/vulkan_present\.h)$"),
    ("Vulkan backend and replay", r"runtime/src/(vulkan_.*|retire_tickets\.h|cached_pso\.h|depth24.*|uav_clear\.h)$"),
    ("Root signatures and descriptors",
     r"runtime/src/(root_signature.*|descriptor_table\.h|gfx10_descriptors\.h|shader_library\.h|resource_tiling\.h)$"),
]
_SITE = re.compile(r'//[^\n]*|/\*.*?\*/|R"([^\s()\\]{0,16})\(.*?\)\1"|("(?:\\.|[^"\\\n])*")|\bE_NOTIMPL\b', re.S)
_REFUSAL_WORDS = re.compile(r"unsupported|not implemented|unimplemented", re.I)


def count_cpp_sites(text):
    """(E_NOTIMPL tokens, log/message strings that say unsupported or not implemented) outside comments."""
    notimpl = messages = 0
    for m in _SITE.finditer(text):
        if m.group(2):
            if _REFUSAL_WORDS.search(m.group(2)):
                messages += 1
        elif m.group(0) == "E_NOTIMPL":
            notimpl += 1
    return notimpl, messages


TRANSLATOR_AREAS = [  # first match on the enclosing function's name
    ("Images and sampling", r"image.*|gather.*|.*_lod$|.*bvh.*"),
    ("Buffers and formats", r"buffer.*|mtbuf.*|.*format.*"),
    ("Global memory and atomics", r"global.*|flat.*|scratch.*|.*atomic.*|addr.*"),
    ("LDS and GS", r"lds.*|gs_.*|ds_.*"),
    ("DPP and cross-lane", r"dpp.*|swizzle|selectors|mask_.*"),
    ("Interpolation and exports", r"interp.*|attribute|export|ms_export|target|x_spill"),
    ("Scalar loads and descriptors", r"scalar.*|desc_.*|sgpr.*"),
    ("64-bit, f16 and modes", r".*64.*|mode_.*|f64|pair.*|f16.*|.*half.*|src_[hu]|dst|number|compare|write_.*"),
    ("Instruction dispatch (_insn)", r"_insn"),
]


def count_translator_sites(source):
    """Sites that `raise Unsupported(...)`, by the area of the function they sit in (ast; the file is not run)."""
    areas, total = {}, 0

    def area_of(fn):
        for name, pattern in TRANSLATOR_AREAS:
            if re.fullmatch(pattern, fn or ""):
                return name
        return "Program structure and checks"

    def walk(node, fn):
        nonlocal total
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if (isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call)
                    and getattr(child.exc.func, "id", None) == "Unsupported"):
                total += 1
                areas[area_of(fn)] = areas.get(area_of(fn), 0) + 1
            walk(child, inner)

    walk(ast.parse(source), None)
    return {"total": total, "areas": dict(sorted(areas.items()))}


def count_unsupported(tree):
    areas, outside = {}, 0
    for prefix in ("runtime/src", "runtime/shims"):
        for path in tree.list(prefix):
            if not re.search(r"\.(cpp|h|inc)$", path):
                continue
            area = next((a for a, rx in RUNTIME_AREAS if re.match(rx, path)), "")
            if area == "" or area is None:
                outside += 1
                continue
            notimpl, messages = count_cpp_sites(tree.text(path) or "")
            entry = areas.setdefault(area, {"e_notimpl": 0, "messages": 0, "files": 0})
            entry["e_notimpl"] += notimpl
            entry["messages"] += messages
            entry["files"] += 1
    source = tree.text(TRANSLATOR)
    translator = count_translator_sites(source) if source else {"total": 0, "areas": {}}
    runtime_total = sum(a["e_notimpl"] + a["messages"] for a in areas.values())
    return {"runtime_total": runtime_total, "runtime_areas": dict(sorted(areas.items())),
            "runtime_files_outside_graphics_path": outside, "translator": translator}


def load_data(path=DATA_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(tree, data, isa_xml=None, d3d12_override=None):
    """All counts for one tree, as plain JSON-serialisable data."""
    interfaces = d3d12_override or data["d3d12"]
    cmake = tree.text(CMAKE) or ""
    leaves = cmake_stub_interfaces(cmake)
    class_list = []
    for path in COM_SOURCES:
        class_list.extend(class_methods(tree.text(path) or ""))
    common = {}
    for path in COM_COMMON:
        for _, _, methods in class_methods(tree.text(path) or ""):
            common.update(methods)
    milestones_text = tree.text(MILESTONE_DOC)
    shaders = tree.list("runtime/test/shaders")
    semantic = tree.list("tests/semantic")
    return {
        "milestones": parse_milestones(milestones_text, tree) if milestones_text else
        {"found": False, "items": [], "duplicate_rows": 0},
        "interfaces": count_interfaces(leaves, interfaces, class_list, common),
        "isa": count_isa(tree, isa_xml),
        "formats": count_formats(tree, data.get("dxgi_formats", [])),
        "tests": count_tests(cmake, shaders, semantic),
        "unsupported": count_unsupported(tree),
    }


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def headline(counts):
    """The overview tiles: id, label, value, total (or None), unit, note."""
    tiles = []
    ms = counts["milestones"]["items"]
    met = sum(1 for m in ms if m["status"] == "met")
    some = sum(1 for m in ms if m["status"] in ("synthetic", "partial"))
    tiles.append({"id": "milestones", "label": "Milestones", "value": met, "total": len(ms), "unit": "met",
                  "note": f"{some} more met in part"})
    ifaces = [i for i in counts["interfaces"] if i["methods"]]
    real, total = sum(i["real"] for i in ifaces), sum(i["methods"] for i in ifaces)
    tiles.append({"id": "interfaces", "label": "D3D12 methods", "value": real, "total": total, "unit": "real",
                  "note": f"{len(ifaces)} interfaces"})
    isa = counts["isa"]
    if isa.get("found"):
        ref = isa.get("fresh") or isa.get("snapshot")
        if ref:
            has_rule = ref["handled_exact"] + ref["handled_family"]
            source = "this run" if isa.get("fresh") else "docs/isa_coverage.md"
            tiles.append({"id": "isa", "label": "RDNA2 instructions", "value": has_rule, "total": ref["total"],
                          "unit": "with a rule", "note": f"share from {source}"})
        else:
            tiles.append({"id": "isa", "label": "RDNA2 rule names", "value": isa["live"]["rule_names"], "total": None,
                          "unit": "names", "note": "no total available"})
    fm = counts["formats"]
    if fm["gfx10_buffer"]:
        tiles.append({"id": "formats", "label": "Buffer formats", "value": fm["gfx10_buffer"]["handled"],
                      "total": fm["gfx10_buffer"]["total"], "unit": "decoded",
                      "note": "GFX10 codes in the translator"})
    t = counts["tests"]
    tiles.append({"id": "tests", "label": "Graphics tests", "value": t["graphics_path"], "total": None,
                  "unit": "registered", "note": f"{t['registered']} in all"})
    u = counts["unsupported"]
    tiles.append({"id": "unsupported", "label": "Refusal sites", "value": u["runtime_total"] + u["translator"]["total"],
                  "total": None, "unit": "in code", "note": f"{u['translator']['total']} translator, "
                                                             f"{u['runtime_total']} runtime"})
    for tile in tiles:
        tile["pct"] = pct(tile["value"], tile["total"]) if tile["total"] else None
    return tiles


def history_point(date, counts):
    """The numbers one history row keeps."""
    ifaces = [i for i in counts["interfaces"] if i["methods"]]
    fm = counts["formats"]
    isa = counts["isa"].get("live", {})
    return {"date": date,
            "milestones_met": sum(1 for m in counts["milestones"]["items"] if m["status"] == "met"),
            "milestones_total": len(counts["milestones"]["items"]),
            "methods_real": sum(i["real"] for i in ifaces), "methods_total": sum(i["methods"] for i in ifaces),
            "isa_rule_names": isa.get("rule_names"),
            "buffer_formats": fm["gfx10_buffer"]["handled"] if fm["gfx10_buffer"] else None,
            "dxgi_formats": fm["dxgi"]["mapped"] if fm["dxgi"] else None,
            "graphics_tests": counts["tests"]["graphics_path"],
            "refusal_sites": counts["unsupported"]["runtime_total"] + counts["unsupported"]["translator"]["total"]}


def read_history(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [p for p in data.get("points", []) if isinstance(p, dict) and "date" in p]
    except (OSError, ValueError, AttributeError):
        return []


def merge_history(points, point):
    """One point per date: a second run on the same date replaces the first."""
    kept = [p for p in points if p["date"] != point["date"]]
    return sorted(kept + [point], key=lambda p: p["date"])


def dumps(data):
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


# --------------------------------------------------------------------------------------------------------------
# The page: sections of blocks, rendered once as Markdown and once as HTML.
# --------------------------------------------------------------------------------------------------------------

def fmt(n):
    return "-" if n is None else f"{n:,}"


def sections(counts, history):
    """[{id, title, how, blocks}], blocks being ('table', ...), ('raster', ...), ('details', ...) or ('note', text)."""
    out = []

    ms = counts["milestones"]
    rows = [[m["id"], STATUS_LABEL[m["status"]], m["scope"],
             (f"{m['named_tests_found']} of {len(m['named_tests'])}" if m["named_tests"] else "-")]
            for m in ms["items"]]
    blocks = [("table", ["Id", "Status", "Scope", "Named tests found"], rows, {"num": [3], "status": 1})]
    if ms["duplicate_rows"]:
        blocks.append(("note", f"The table lists {ms['duplicate_rows']} row(s) twice; each id is counted once."))
    if not ms["found"]:
        blocks = [("note", f"No milestone table found in {MILESTONE_DOC}.")]
    out.append({"id": "milestones", "title": "Milestones",
                "how": f"Rows of the boot milestone table in {MILESTONE_DOC}, section 6, each id once. A row is "
                       "\"met\" when its Done-when cell says \"Met by\" or \"**Met\", \"synthetic part met\" when it "
                       "says \"Synthetic part met\", \"partly met\" when it says \"Partly met\", otherwise open. "
                       "The words are the authors'; this page does not judge them. Open pull requests are not "
                       "known here. Named tests are the names ending in _test in the row that have a source file in "
                       "runtime/test.", "blocks": blocks})

    ifaces = counts["interfaces"]
    rows = []
    rasters = []
    details = []
    for i in ifaces:
        if i["methods"] is None:
            rows.append([i["interface"], "-", "-", "-", "-", "-"])
            continue
        rows.append([i["interface"], fmt(i["real"]), fmt(i["refuses"]), fmt(i["stub"]), fmt(i["methods"]),
                     {"bar": i["real"] / i["methods"] if i["methods"] else 0, "t": f"{pct(i['real'], i['methods'])}%"}])
        rasters.append(("raster", i["interface"], [{"state": s, "title": n.split("::")[-1]}
                                                   for s, n in zip(i["cells"], _method_names(i, counts))],
                        f"{i['interface']}: {i['real']} of {i['methods']} methods have a real body"))
        if i["open"]:
            details.append(("details", f"{i['interface']}: {len(i['open'])} open", i["open"]))
    out.append({"id": "interfaces", "title": "D3D12 interface coverage",
                "how": "Interfaces are the ones handed to tools/gen_d3d12_stubs.py in runtime/CMakeLists.txt. Methods "
                       "are the names of each interface and its bases (IUnknown left out) in "
                       "tools/graphics_progress_data.json, from the Windows SDK d3d12.h. Real: an override with a body "
                       "in the class that derives from the interface's generated stub class, or in the shared object "
                       "template. Refuses: an override whose body only returns E_NOTIMPL. Stub: no override, so the "
                       "generated stub answers (E_NOTIMPL, or nothing) and logs once. Matching is by method name, "
                       "like gen_d3d12_stubs.py --report; a real body can still be partial.",
                "blocks": [("table", ["Interface", "Real", "Refuses", "Stub", "Methods", "Share"], rows,
                            {"num": [1, 2, 3, 4, 5]})] + rasters + [("legend", "interface")] + details})

    isa = counts["isa"]
    rows, notes = [], []
    if isa.get("found"):
        live = isa["live"]
        rows += [["Instruction names the translator accepts", fmt(live["rule_names"]), "counted from this tree"],
                 ["Names it refuses everywhere", fmt(live["refused_names"]), "counted from this tree"],
                 ["Branches that accept a name but can raise", fmt(live["branches_that_raise"]),
                  "counted from this tree"],
                 ["Prefix families (ds_, v_cmp_, ...)", fmt(live["prefix_families"]), "counted from this tree"]]
        ref = isa.get("fresh")
        label = "this run, AMD XML"
        if ref is None and isa.get("snapshot"):
            ref = isa["snapshot"]
            label = f"stated in {ISA_DOC}"
        if ref:
            rows += [["RDNA2 instructions in AMD's XML", fmt(ref["total"]), label],
                     ["with an exact rule", fmt(ref["handled_exact"]), label],
                     ["with a prefix rule", fmt(ref["handled_family"]), label],
                     ["refused", fmt(ref["refused"]), label], ["unknown", fmt(ref["unknown"]), label]]
            if isa.get("snapshot") and not isa.get("fresh"):
                s = isa["snapshot"]
                notes.append(f"The instruction total comes from AMD's XML, which is not in the repository. The share "
                             f"above is what {ISA_DOC} states for the translator at {s['translator_commit']}; this "
                             f"tool did not recount it. Run tools/isa_coverage.py or pass --isa for a fresh one.")
    else:
        notes.append(f"No translator found at {TRANSLATOR}.")
    out.append({"id": "isa", "title": "RDNA2 instruction coverage",
                "how": "tools/isa_coverage.py is used as a library on the translator text (an ast scan; the "
                       "translator is not run). \"Has a rule\" means the opcode name is compared or tabled, or starts "
                       "with a tested prefix; it does not mean the rule is right. Prefix rules are an upper bound.",
                "blocks": ([("table", ["Measure", "Count", "Source"], rows, {"num": [1]})] if rows else [])
                + [("note", n) for n in notes]})

    fm = counts["formats"]
    blocks = []
    g = fm["gfx10_buffer"]
    if g:
        blocks += [("table", ["Kind", "Handled", "Total", "Share"],
                    [["GFX10 buffer format codes (translator)", fmt(g["handled"]), fmt(g["total"]),
                      {"bar": g["handled"] / g["total"], "t": f"{pct(g['handled'], g['total'])}%"}]], {"num": [1, 2, 3]}),
                   ("raster", "GFX10 buffer format codes",
                    [{"state": "y" if c["handled"] else "n", "title": f"{c['code']} {c['name']}"} for c in g["cells"]],
                    f"GFX10 buffer format codes: {g['handled']} of {g['total']} decoded"),
                   ("legend", "format")]
        have = [f"{c['code']} {c['name']}" for c in g["cells"] if c["handled"]]
        missing = [f"{c['code']} {c['name']}" for c in g["cells"] if not c["handled"]]
        blocks += [("details", f"Decoded ({len(have)})", have), ("details", f"Not decoded ({len(missing)})", missing)]
    d = fm["dxgi"]
    if d:
        blocks += [("table", ["Kind", "Mapped", "Total", "Share"],
                    [["DXGI formats the layer maps to Vulkan", fmt(d["mapped"]), fmt(d["total"]),
                      {"bar": d["mapped"] / d["total"], "t": f"{pct(d['mapped'], d['total'])}%"}]], {"num": [1, 2, 3]}),
                   ("raster", "DXGI formats",
                    [{"state": "y" if c["mapped"] else "n", "title": c["name"]} for c in d["cells"]],
                    f"DXGI formats: {d['mapped']} of {d['total']} mapped"),
                   ("legend", "format")]
        unmapped = [c["name"] for c in d["cells"] if not c["mapped"]]
        blocks.append(("details", f"Not mapped ({len(unmapped)})", unmapped))
    out.append({"id": "formats", "title": "Formats",
                "how": "Buffer: the integer keys of the FORMATS dict in class Gen of the translator (read with ast), "
                       "out of the codes in tools/gfx10_formats.json except the INVALID one. Image formats are not "
                       "counted. DXGI: names in a case label or table row of vulkan_format() and vertex_format() in "
                       f"{VK_DEVICE}, out of the DXGI_FORMAT names (UNKNOWN left out) in "
                       "tools/graphics_progress_data.json. Mapped means a Vulkan format is chosen, not that every "
                       "usage of it works.", "blocks": blocks})

    t = counts["tests"]
    rows = [[name, fmt(a["count"]), "yes" if a["graphics"] else "no"] for name, a in t["areas"].items()]
    blocks = [("table", ["Area", "Registered", "Graphics path"], rows, {"num": [1]}),
              ("table", ["Also in the tree", "Count"],
               [["Hand-written RDNA2 programs in runtime/test/shaders (.asm)", fmt(t["gpu_programs"])],
                ["Translator regression tests in tests/semantic (test_*.py)", fmt(t["semantic_tests"])],
                ["Fixture files in tests/semantic/fixtures", fmt(t["semantic_fixtures"])]], {"num": [1]})]
    blocks += [("details", f"{n} ({a['count']})", a["tests"]) for n, a in t["areas"].items()]
    out.append({"id": "tests", "title": "GPU test inventory",
                "how": "Distinct names in add_test(NAME ...) calls of runtime/CMakeLists.txt, sorted into areas by "
                       "name. Registered under a condition (a GPU, a header, a tool) still counts once, and a name "
                       "with ${...} counts as one. Registered is not passing, and many need a GPU to do anything.",
                "blocks": blocks})

    u = counts["unsupported"]
    rows = [[a, fmt(v["e_notimpl"]), fmt(v["messages"]), fmt(v["e_notimpl"] + v["messages"])]
            for a, v in u["runtime_areas"].items()]
    blocks = [("table", ["Runtime area", "E_NOTIMPL", "Messages", "Sites"], rows, {"num": [1, 2, 3]}),
              ("table", ["Translator area", "raise Unsupported"],
               [[a, fmt(n)] for a, n in u["translator"]["areas"].items()], {"num": [1]})]
    out.append({"id": "unsupported", "title": "Unsupported sites",
                "how": "The places that refuse work, so the list of what is still needed. Runtime: each E_NOTIMPL "
                       "token and each string that says \"unsupported\", \"not implemented\" or \"unimplemented\", "
                       f"outside comments, in the graphics-path files of runtime/src and runtime/shims "
                       f"({u['runtime_files_outside_graphics_path']} other files are not counted). Translator: each "
                       "`raise Unsupported(...)` in tools/rdna2spirv.py, by the function it sits in. A site is a line "
                       "of code, not one missing feature: one site can guard many cases, and a site that fires "
                       "rarely counts the same as one that fires always.", "blocks": blocks})

    if history:
        last = history[-14:]
        rows = [[p["date"], f"{p['milestones_met']}/{p['milestones_total']}",
                 f"{fmt(p['methods_real'])}/{fmt(p['methods_total'])}", fmt(p.get("isa_rule_names")),
                 fmt(p.get("buffer_formats")), fmt(p.get("dxgi_formats")), fmt(p["graphics_tests"]),
                 fmt(p["refusal_sites"])] for p in reversed(last)]
        out.append({"id": "history", "title": "History",
                    "how": "One row per date this tool ran with --date, newest first, the last 14. The numbers are "
                           "the counts above for that day's tree.",
                    "blocks": [("table", ["Date", "Milestones", "Methods", "ISA names", "Buffer formats",
                                          "DXGI formats", "Tests", "Refusals"], rows, {"num": [1, 2, 3, 4, 5, 6, 7]})]})
    return out


def _method_names(iface, counts):
    # names in chain order are not stored per cell; the open list holds only the open ones, so rebuild from data
    return iface.get("method_names") or [""] * len(iface["cells"])


# ---- Markdown ------------------------------------------------------------------------------------------------

def md_cell(c):
    if isinstance(c, dict):
        return c["t"]
    return str(c).replace("|", "\\|")


def render_md(report, secs):
    meta = report["meta"]
    lines = ["# Graphics path progress", "", f"**{DISCLAIMER}**", "",
             "The Xbox D3D12 to Vulkan layer under `runtime/` and the RDNA2 to SPIR-V translator "
             "`tools/rdna2spirv.py`. Each number is a count over source text; nothing was built or run.", "",
             f"Source: {meta['source']}" + (f", {meta['date']}" if meta.get("date") else "") + ".", "",
             "| Area | Count | Of | Share |", "|---|---:|---:|---:|"]
    for t in report["headline"]:
        lines.append(f"| [{t['label']}](#{t['id']}) | {fmt(t['value'])} {t['unit']} | {fmt(t['total'])} | "
                     f"{'' if t['pct'] is None else str(t['pct']) + '%'} |")
    lines.append("")
    for s in secs:
        lines += [f"<a id=\"{s['id']}\"></a>", "", f"## {s['title']}", "", f"*How this is counted.* {s['how']}", ""]
        for block in s["blocks"]:
            kind = block[0]
            if kind == "table":
                _, head, rows, opts = block
                lines.append("| " + " | ".join(head) + " |")
                lines.append("|" + "|".join("---:" if i in opts.get("num", []) else "---" for i in range(len(head))) + "|")
                lines += ["| " + " | ".join(md_cell(c) for c in row) + " |" for row in rows]
                lines.append("")
            elif kind == "note":
                lines += [block[1], ""]
            elif kind == "details":
                _, summary, items = block
                if items:
                    lines += ["<details><summary>" + html.escape(summary) + "</summary>", "",
                              ", ".join(f"`{i}`" for i in items), "", "</details>", ""]
            # rasters and legends are page-only; the tables carry the same numbers
    lines += ["---", "", "Generated by `tools/graphics_progress.py`. Raw numbers: `graphics.json`. "
              "Counts are over committed text, so they move only when the code does.", ""]
    return "\n".join(lines)


# ---- HTML ----------------------------------------------------------------------------------------------------

CSS = """
:root{color-scheme:light dark;--paper:#f5f2ea;--panel:#fbf9f4;--ink:#1c1e23;--ink-2:#4f535c;--rule:#d8d3c5;
--track:#e2ddcf;--warn:#a63d27;--font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
--serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
--mono:ui-monospace,"Cascadia Mono","SF Mono",Consolas,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#131519;--panel:#1a1d22;--ink:#ebe7dd;
--ink-2:#a3a7af;--rule:#2d3138;--track:#2a2e35;--warn:#ef8d72}}
:root[data-theme="dark"]{--paper:#131519;--panel:#1a1d22;--ink:#ebe7dd;--ink-2:#a3a7af;--rule:#2d3138;
--track:#2a2e35;--warn:#ef8d72}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 var(--font);
-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
a{color:inherit;text-decoration-thickness:from-font;text-underline-offset:.18em}
a:focus-visible,summary:focus-visible,.scroll:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.skip{position:absolute;inset-inline-start:16px;top:-48px;background:var(--ink);color:var(--paper);padding:8px 12px}
.skip:focus{top:8px}
main{max-width:64rem;margin:0 auto;padding:32px 16px 64px}
.eyebrow{font:600 12px/1.2 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-2);margin:0 0 12px}
h1{font:600 clamp(1.75rem,5vw,2.75rem)/1.1 var(--serif);letter-spacing:-.015em;margin:0 0 16px;text-wrap:balance}
h2{font:600 1.375rem/1.2 var(--serif);letter-spacing:-.01em;margin:0 0 8px;text-wrap:balance}
p{margin:0 0 12px;max-inline-size:68ch;text-wrap:pretty}
.disclaimer{border-inline-start:4px solid var(--ink);padding:4px 0 4px 16px;margin:0 0 16px;font-weight:600;
font-size:1.125rem;line-height:1.4}
.lede,.meta{color:var(--ink-2)}
.meta{font:13px/1.4 var(--mono);margin-bottom:32px}
.tiles{display:grid;grid-template-columns:1fr;gap:12px;margin:0 0 48px;padding:0;list-style:none}
@media (min-width:480px){.tiles{grid-template-columns:repeat(2,1fr)}}
@media (min-width:760px){.tiles{grid-template-columns:repeat(3,1fr)}}
.tile{display:block;height:100%;padding:16px;background:var(--panel);border:1px solid var(--rule);border-radius:6px;
text-decoration:none;transition:transform 140ms cubic-bezier(.23,1,.32,1)}
@media (hover:hover) and (pointer:fine){.tile:hover .tile-label{text-decoration:underline}}
.tile:active{transform:scale(.98)}
.tile-label{display:block;font:600 12px/1.3 var(--mono);letter-spacing:.08em;text-transform:uppercase;
color:var(--ink-2);margin-bottom:8px}
.tile-num{display:block;font:600 2rem/1.1 var(--mono);font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile-of{font-size:.9rem;color:var(--ink-2);font-weight:400;letter-spacing:0}
.tile-note{display:block;font-size:13px;line-height:1.4;color:var(--ink-2);margin-top:4px;min-height:1.4em}
.bar{display:block;height:4px;background:var(--track);border-radius:2px;margin-top:12px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ink);border-radius:2px}
section{margin:0 0 48px;scroll-margin-top:16px}
.how{color:var(--ink-2);font-size:.9375rem}
.how b{font-weight:600;color:var(--ink)}
.scroll{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:.9375rem;font-variant-numeric:tabular-nums}
caption{text-align:start;font-size:13px;color:var(--ink-2);padding-bottom:4px}
th,td{padding:8px 12px 8px 0;border-bottom:1px solid var(--rule);text-align:start;vertical-align:top;
overflow-wrap:break-word}
th.rowh{font:inherit;text-transform:none;letter-spacing:0;color:inherit;white-space:normal;
border-bottom:1px solid var(--rule);overflow-wrap:anywhere}
th.rowh.short{white-space:nowrap}
th{font:600 12px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);
border-bottom:1px solid var(--ink);white-space:nowrap}
td.n,th.n{text-align:end;font-family:var(--mono);padding-inline:12px 0}
td.n+td,th.n+th{padding-inline-start:16px}
td .bar{margin:0;min-width:56px}
td.share{white-space:nowrap}
td.share span{display:block;font:13px/1.3 var(--mono);color:var(--ink-2);margin-bottom:4px;text-align:end}
td.warn{color:var(--warn)}
.raster{display:flex;flex-wrap:wrap;gap:2px;margin:8px 0 4px}
.raster-wrap{margin:16px 0}
.raster-label{font:600 12px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}
.c{display:block;width:10px;height:10px;border:1px solid var(--ink-2);border-radius:1px;background:transparent}
.c.y{background:var(--ink);border-color:var(--ink)}
.c.r{background:repeating-linear-gradient(135deg,var(--warn) 0 2px,transparent 2px 4px);border-color:var(--warn)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--ink-2);margin:4px 0 16px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend .c{display:inline-block}
details{margin:8px 0}
summary{cursor:pointer;font-size:.9375rem;padding:4px 0}
details p{font:13px/1.6 var(--mono);color:var(--ink-2);overflow-wrap:anywhere;max-inline-size:none}
footer{border-top:1px solid var(--rule);padding-top:16px;color:var(--ink-2);font-size:.875rem}
@media (prefers-reduced-motion:reduce){.tile{transition:none}.tile:active{transform:none}}
@media print{.skip{display:none}}
"""


def _h(text):
    return html.escape(str(text), quote=True)


def html_cell(c, numeric, share_col):
    if isinstance(c, dict):
        return (f'<td class="share n"><span>{_h(c["t"])}</span><i class="bar" role="presentation">'
                f'<i style="width:{c["bar"] * 100:.1f}%"></i></i></td>')
    return f'<td class="{"n" if numeric else ""}">{_h(c)}</td>'


LEGENDS = {"interface": [("y", "real body"), ("r", "returns E_NOTIMPL"), ("n", "generated stub")],
           "format": [("y", "handled"), ("n", "not handled")]}


def render_html(report, secs):
    meta = report["meta"]
    out = ['<!doctype html>', '<html lang="en">', "<head>", '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           "<title>Graphics path progress</title>",
           '<meta name="description" content="Progress of the Leonida graphics path, counted from the committed code.">',
           f"<style>{CSS}</style>", "</head>", "<body>", '<a class="skip" href="#main">Skip to content</a>',
           '<main id="main">', '<p class="eyebrow">Leonida / graphics path</p>', "<h1>Graphics path progress</h1>",
           f'<p class="disclaimer">{_h(DISCLAIMER)}</p>',
           '<p class="lede">The Xbox D3D12 to Vulkan layer under <code>runtime/</code> and the RDNA2 to SPIR-V '
           "translator <code>tools/rdna2spirv.py</code>. Every number is a count over source text. Nothing was "
           "built, run or imported to get it.</p>",
           f'<p class="meta">Source: {_h(meta["source"])}' + (f" &middot; {_h(meta['date'])}" if meta.get("date") else "")
           + "</p>", '<ul class="tiles" aria-label="Overview">']
    for t in report["headline"]:
        total = f' <span class="tile-of">of {fmt(t["total"])}</span>' if t["total"] else ""
        unit = f' <span class="tile-of">{_h(t["unit"])}</span>' if not t["total"] else ""
        bar = ""
        label = f'{t["label"]}: {fmt(t["value"])}' + (f' of {fmt(t["total"])}, {t["pct"]}%' if t["total"] else "")
        if t["total"]:
            bar = f'<i class="bar" aria-hidden="true"><i style="width:{min(100.0, t["pct"]):.1f}%"></i></i>'
        pct_text = f" &middot; {t['pct']}%" if t["pct"] is not None else ""
        out.append(f'<li><a class="tile" href="#{t["id"]}" aria-label="{_h(label)}"><span class="tile-label">'
                   f'{_h(t["label"])}</span><span class="tile-num">{fmt(t["value"])}{total}{unit}</span>'
                   f'<span class="tile-note">{_h(t["note"])}{pct_text}</span>{bar}</a></li>')
    out.append("</ul>")
    for s in secs:
        out += [f'<section id="{s["id"]}" aria-labelledby="h-{s["id"]}">', f'<h2 id="h-{s["id"]}">{_h(s["title"])}</h2>',
                f'<p class="how"><b>How this is counted.</b> {_h(s["how"])}</p>']
        for block in s["blocks"]:
            kind = block[0]
            if kind == "table":
                _, head, rows, opts = block
                num = set(opts.get("num", []))
                out.append(f'<div class="scroll" tabindex="0" role="region" aria-label="{_h(s["title"])} table">'
                           "<table><thead><tr>"
                           + "".join(f'<th scope="col"{" class=\"n\"" if i in num else ""}>{_h(h)}</th>'
                                     for i, h in enumerate(head)) + "</tr></thead><tbody>")
                for row in rows:
                    cells = []
                    for i, c in enumerate(row):
                        if i == 0:
                            nowrap = ' class="rowh short"' if len(str(c)) <= 8 else ' class="rowh"'
                            cells.append(f'<th scope="row"{nowrap}>{_h(c)}</th>')
                        else:
                            cells.append(html_cell(c, i in num, False))
                    out.append("<tr>" + "".join(cells) + "</tr>")
                out.append("</tbody></table></div>")
            elif kind == "raster":
                _, label, cells, aria = block
                out.append(f'<div class="raster-wrap"><div class="raster-label">{_h(label)}</div>'
                           f'<div class="raster" role="img" aria-label="{_h(aria)}">'
                           + "".join(f'<i class="c {c["state"]}" title="{_h(c["title"])}"></i>' for c in cells)
                           + "</div></div>")
            elif kind == "legend":
                out.append('<div class="legend" aria-hidden="true">'
                           + "".join(f'<span><i class="c {k}"></i>{_h(v)}</span>' for k, v in LEGENDS[block[1]])
                           + "</div>")
            elif kind == "note":
                out.append(f"<p>{_h(block[1])}</p>")
            elif kind == "details":
                _, summary, items = block
                if items:
                    out.append(f"<details><summary>{_h(summary)}</summary><p>{_h(', '.join(items))}</p></details>")
        out.append("</section>")
    out += ["<footer><p>Generated by <code>tools/graphics_progress.py</code>. Raw numbers are in "
            "<code>graphics.json</code>; the same page as Markdown is <code>graphics.md</code>. The counts move only "
            "when the committed code does.</p></footer>", "</main>", "</body>", "</html>", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------------------
# Refreshing the data file from the SDK headers (a maintainer step; the CI run never needs the headers).
# --------------------------------------------------------------------------------------------------------------

def refresh_data(d3d12_header, dxgi_header, out_path=DATA_FILE):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import gen_d3d12_stubs as gen
    with open(d3d12_header, encoding="utf-8", errors="replace") as f:
        parsed = gen.parse_interfaces(f.read())
    d3d12 = {name: {"base": base, "methods": list(dict.fromkeys(it[2] for it in items if it[0] == "m"))}
             for name, (base, items) in sorted(parsed.items())}
    with open(dxgi_header, encoding="utf-8", errors="replace") as f:
        enum = re.findall(r"\bDXGI_FORMAT_(\w+)\s*=\s*(\d+)", gen.COMMENTS_OR_LITERALS.sub(" ", f.read()))
    formats = [n for n, v in sorted(set(enum), key=lambda x: int(x[1])) if n not in ("UNKNOWN", "FORCE_UINT")]
    data = {"about": "Method names of the ID3D12 interfaces (Windows SDK d3d12.h) and the DXGI_FORMAT names "
                     "(dxgiformat.h). Names only, no declarations. Regenerate with tools/graphics_progress.py "
                     "--refresh-data --d3d12-header <d3d12.h> --dxgi-header <dxgiformat.h>.",
            "d3d12": d3d12, "dxgi_formats": formats}
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return len(d3d12), len(formats)


# --------------------------------------------------------------------------------------------------------------

def build_report(tree, data, isa_xml=None, date=None):
    counts = collect(tree, data, isa_xml)
    names = {}
    for i in counts["interfaces"]:
        if i["methods"]:
            names[i["interface"]] = [m for _, m in chain_methods(data["d3d12"], i["interface"])]
    for i in counts["interfaces"]:
        if i["methods"]:
            i["method_names"] = names[i["interface"]]
    meta = {"source": tree.label + (f" ({tree.commit[:12]})" if tree.commit else ""), "schema": SCHEMA,
            "disclaimer": DISCLAIMER}
    if tree.commit:
        meta["commit"] = tree.commit
    if date:
        meta["date"] = date
    report = {"meta": meta, "counts": counts}
    report["headline"] = headline(counts)
    return report


def put_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tree", metavar="REF", help="read this git ref instead of the working tree")
    parser.add_argument("--root", default=ROOT, help="repository root (default: this checkout)")
    parser.add_argument("--out-dir", default=None, help="folder for graphics.json/.html/.md (default docs/progress)")
    parser.add_argument("--date", metavar="YYYY-MM-DD", help="stamp the output and add this day to the history")
    parser.add_argument("--history", metavar="FILE", help="history file (default <out-dir>/graphics-history.json)")
    parser.add_argument("--isa", metavar="XML", help="AMD's amdgpu_isa_rdna2.xml, for a fresh instruction total")
    parser.add_argument("--d3d12-header", metavar="H", help="count methods from this d3d12.h instead of the data file")
    parser.add_argument("--refresh-data", action="store_true",
                        help="rewrite tools/graphics_progress_data.json from --d3d12-header and --dxgi-header")
    parser.add_argument("--dxgi-header", metavar="H")
    parser.add_argument("--print", action="store_true", help="print the headline numbers")
    args = parser.parse_args(argv)

    if args.refresh_data:
        if not (args.d3d12_header and args.dxgi_header):
            parser.error("--refresh-data needs --d3d12-header and --dxgi-header")
        n, f = refresh_data(args.d3d12_header, args.dxgi_header)
        print(f"wrote {DATA_FILE}: {n} interfaces, {f} DXGI formats")
        return 0
    if args.date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        parser.error("--date must look like 2026-11-22")
    root = os.path.abspath(args.root)
    tree = GitTree(root, args.tree) if args.tree else WorkTree(root)
    data = load_data()
    override = None
    if args.d3d12_header:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import gen_d3d12_stubs as gen
        with open(args.d3d12_header, encoding="utf-8", errors="replace") as f:
            parsed = gen.parse_interfaces(f.read())
        override = {n: {"base": b, "methods": list(dict.fromkeys(i[2] for i in items if i[0] == "m"))} for n, (b, items) in parsed.items()}
        data = dict(data, d3d12=override)
    report = build_report(tree, data, args.isa, args.date)

    out_dir = args.out_dir or os.path.join(root, "docs", "progress")
    history_path = args.history or os.path.join(out_dir, "graphics-history.json")
    history = read_history(history_path)
    if args.date:
        history = merge_history(history, history_point(args.date, report["counts"]))
        put_text(history_path, dumps({"points": history, "schema": SCHEMA}))
    secs = sections(report["counts"], history)
    # the JSON keeps the raw counts; the per-interface cell strings are page-only
    json_report = json.loads(json.dumps(report))
    for i in json_report["counts"]["interfaces"]:
        i.pop("cells", None)
        i.pop("method_names", None)
    for key in ("gfx10_buffer", "dxgi"):
        block = json_report["counts"]["formats"].get(key)
        if block:
            block["cells"] = [c for c in block["cells"]]
    json_report["history_points"] = len(history)
    put_text(os.path.join(out_dir, "graphics.json"), dumps(json_report))
    put_text(os.path.join(out_dir, "graphics.md"), render_md(report, secs))
    put_text(os.path.join(out_dir, "graphics.html"), render_html(report, secs))
    if args.print:
        for t in report["headline"]:
            of = f" of {t['total']}" if t["total"] else ""
            print(f"{t['label']}: {t['value']}{of} {t['unit']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
