#!/usr/bin/env python3
"""Coverage dashboard: Automated visual SVG treemap of Leonida instruction & codebase coverage.

    python -I tools/coverage_dashboard.py [--output docs/coverage/dashboard.svg] [--check]

Reports static RDNA2 (GFX1030) and x64 (Zen 2) instruction estimates from
hard-coded dashboard baselines with no documented measurement source.
Runtime and shim figures are source-text estimates with fallback counts.
CLI tool figures count matching test filenames.

Renders a dark-mode SVG treemap for the project README and docs.
In --check mode, verifies the existing SVG's panel totals and their stated basis.
"""
import argparse
import datetime as dt
import html
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "docs", "coverage", "dashboard.svg")


# ---------------------------------------------------------------------------
# Metric Collectors
# ---------------------------------------------------------------------------

def get_git_metadata(repo_root=REPO_ROOT):
    """Retrieve the current commit short hash and date deterministically."""
    commit_sha = "unknown"
    commit_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%h %cI"],
            cwd=repo_root, capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(" ", 1)
            commit_sha = parts[0]
            if len(parts) > 1:
                commit_date = parts[1]
    except Exception:
        pass

    return commit_sha, commit_date


def get_rdna2_coverage(repo_root=REPO_ROOT):
    """Return the static GFX1030 baseline independently of the checkout."""
    catalog = [
        # (Category, Estimated covered, Total)
        ("VOP3", 150, 425),
        ("SDWA", 54, 220),
        ("FLAT", 0, 133),
        ("MIMG", 3, 111),
        ("DPP8", 43, 100),
        ("VOP1", 34, 81),
        ("MUBUF", 0, 73),
        ("SOP2", 51, 51),
        ("VOP2", 30, 47),
        ("SOPP", 6, 37),
        ("SOPK", 22, 27),
        ("SOP1", 31, 65),
        ("DS", 0, 154),
        ("SMEM", 5, 84),
        ("VOPC", 62, 190),
    ]

    small_items = [
        ("VOP3P", 29, 29),
        ("SOPC", 18, 18),
        ("MTBUF", 0, 16),
        ("VINTRP", 2, 3),
        ("EXP", 1, 1),
    ]

    tot_cov = sum(c for _, c, _ in catalog) + sum(c for _, c, _ in small_items)
    tot_all = sum(t for _, _, t in catalog) + sum(t for _, _, t in small_items)
    return {
        "title": "gfx1030 (RDNA2) forms",
        "covered": tot_cov,
        "total": tot_all,
        "basis": "static estimate from hard-coded dashboard baseline",
        "categories": catalog,
        "small_items": small_items,
    }


def get_x64_coverage(repo_root=REPO_ROOT):
    """Return the static x64 (Zen 2 user-mode mnemonics) baseline."""
    catalog = [
        ("AVX", 208, 256),
        ("Integer", 147, 191),
        ("x87", 68, 100),
        ("SSE", 52, 69),
        ("FMA", 60, 60),
        ("MMX", 47, 48),
        ("SSE3-4", 58, 86),
        ("SSE2", 52, 64),
    ]

    small_items = [
        ("AVX2", 28, 28),
        ("Crypto", 21, 21),
        ("BMI/ADX", 16, 16),
        ("XSAVE", 0, 9),
    ]

    tot_cov = sum(c for _, c, _ in catalog) + sum(c for _, c, _ in small_items)
    tot_all = sum(t for _, _, t in catalog) + sum(t for _, _, t in small_items)
    return {
        "title": "x64 (Zen 2 user-mode mnemonics)",
        "covered": tot_cov,
        "total": tot_all,
        "basis": "static estimate from hard-coded dashboard baseline",
        "categories": catalog,
        "small_items": small_items,
    }


def get_runtime_coverage(repo_root=REPO_ROOT):
    """Collect Runtime and Win32/Vulkan shim coverage."""
    shims_dir = os.path.join(repo_root, "runtime", "shims")
    src_dir = os.path.join(repo_root, "runtime", "src")

    groups = {
        "D3D12 Core & Device": ["d3d12_xs_device.cpp", "d3d12_com.cpp", "d3d12_vulkan_device.cpp", "d3d12_xs.cpp", "d3d12_xs_rootsig.cpp"],
        "GameInput & Controls": ["gameinput.cpp", "xbox_present.cpp"],
        "Vulkan Translation": ["dxgi_com.cpp", "root_signature.cpp", "tls_host.cpp", "pe_loader.cpp"],
        "System & Process": ["file_system.cpp", "process.cpp", "registry.cpp", "library_loader.cpp"],
        "XGameRuntime": ["xgameruntime.cpp", "xgameruntime_user.cpp", "xmem.cpp"],
        "WinHTTP & Network": ["winhttp_offline.cpp"],
    }

    categories = []
    small_items = []

    stub_patterns = ["E_NOTIMPL", "unimplemented", "STUB", "abort", "return 0;", "return nullptr;"]

    for cat_name, file_list in groups.items():
        total_fns = 0
        stubs = 0
        for f in file_list:
            p = os.path.join(shims_dir, f)
            if not os.path.exists(p):
                p = os.path.join(src_dir, f)
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                matches = re.findall(
                    r"(?:STDMETHODIMP|HRESULT|void|BOOL|HANDLE|DWORD|uint32_t|int|size_t)\s+([A-Za-z0-9_:]+)\s*\([^)]*\)\s*\{([^}]*)\}",
                    text
                )
                for name, body in matches:
                    total_fns += 1
                    if any(sp in body for sp in stub_patterns):
                        stubs += 1
            except Exception:
                pass

        if total_fns == 0:
            total_fns = 10
            stubs = 1

        covered = max(0, total_fns - stubs)
        if total_fns < 15:
            small_items.append((cat_name, covered, total_fns))
        else:
            categories.append((cat_name, covered, total_fns))

    tot_cov = sum(c for _, c, _ in categories) + sum(c for _, c, _ in small_items)
    tot_all = sum(t for _, _, t in categories) + sum(t for _, _, t in small_items)
    return {
        "title": "Runtime & Win32/Vulkan Shims",
        "covered": tot_cov,
        "total": tot_all,
        "basis": "source-text estimate with fallback counts",
        "categories": categories,
        "small_items": small_items,
    }


def get_tools_coverage(repo_root=REPO_ROOT):
    """Collect CLI tools & regression test suite coverage."""
    tools_dir = os.path.join(repo_root, "tools")
    tests_dir = os.path.join(repo_root, "tests")

    tool_files = []
    if os.path.isdir(tools_dir):
        tool_files = [f for f in os.listdir(tools_dir) if f.endswith(".py")]

    test_files = set()
    if os.path.isdir(tests_dir):
        test_files = set(os.listdir(tests_dir))

    groups = {
        "Shader & Translator": ["rdna2spirv.py", "disasm.py", "isa_coverage.py", "cf_diff.py", "wave64_halves.py", "shader_meta.py", "f16_fuzz.py"],
        "Asset & Archive": ["anim_index.py", "skinned_glb.py", "bvh_rehydrate.py", "manifest_verifier.py", "world_archetypes.py", "rpf9dump.py"],
        "Texture & Format": ["tex_export.py", "amd_swizzle.py", "bc_feature.py", "tile_probe.py", "swizzle_search.py", "calibrated_texture.py"],
        "Runtime & Build": ["make_game.py", "vk_pipeline_check.py", "serve.py", "progress_dashboard.py", "exe_checklist.py"],
        "Fuzzers & Diagnostics": ["valu_fuzz.py", "lds_fuzz.py", "image_fuzz.py", "salu_diff.py", "f32_fuzz.py"],
    }

    categories = []
    assigned_tools = set()

    for cat_name, file_list in groups.items():
        cat_tools = [f for f in file_list if f in tool_files]
        assigned_tools.update(cat_tools)
        cov = sum(1 for f in cat_tools if f"test_{f}" in test_files or f"test_{f[:-3]}.py" in test_files)
        tot = len(cat_tools)
        if tot > 0:
            categories.append((cat_name, cov, tot))

    # Remaining tools
    other_tools = [f for f in tool_files if f not in assigned_tools]
    if other_tools:
        cov = sum(1 for f in other_tools if f"test_{f}" in test_files or f"test_{f[:-3]}.py" in test_files)
        categories.append(("General Tools", cov, len(other_tools)))

    tot_cov = sum(c for _, c, _ in categories)
    tot_all = sum(t for _, _, t in categories)
    return {
        "title": "CLI Tools",
        "covered": tot_cov,
        "total": tot_all,
        "basis": "matching test filenames",
        "categories": categories,
        "small_items": [],
    }


# ---------------------------------------------------------------------------
# SVG Layout and Rendering
# ---------------------------------------------------------------------------

def render_category_grid(name, covered, total, x, y, width, height):
    """Render an individual category container with a cell grid."""
    lines = []
    # Outer category border
    lines.append(f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" stroke="#30363d" stroke-width="1" />')

    # Label
    label_text = html.escape(f"{name} {covered}/{total}")
    lines.append(f'  <text x="{x + 6}" y="{y + 13}" class="cat-label">{label_text}</text>')

    # Cell grid setup
    grid_x = x + 6
    grid_y = y + 20
    avail_w = width - 12
    avail_h = height - 26

    cell_size = 6
    cell_pitch = 8
    rx = 1

    # Calculate optimal number of columns based on width
    cols = max(1, int(avail_w // cell_pitch))
    max_cells_visible = int((avail_w // cell_pitch) * (avail_h // cell_pitch))

    cells_to_draw = min(total, max_cells_visible) if total > 0 else 0

    for i in range(cells_to_draw):
        row = i // cols
        col = i % cols
        cx = grid_x + col * cell_pitch
        cy = grid_y + row * cell_pitch

        if i < covered:
            color = "#2ea043"  # Green
        else:
            color = "#30363d"  # Dark grey

        lines.append(f'  <rect x="{cx}" y="{cy}" width="{cell_size}" height="{cell_size}" rx="{rx}" fill="{color}" />')

    return "\n".join(lines)


def panel_summary(panel):
    """Format the counts and their basis for both rendering and drift checks."""
    return f"{panel['title']}: {panel['covered']} of {panel['total']} ({panel['basis']})"


def render_card(panel_data, card_x, card_y, card_w, card_h):
    """Render a section card containing multiple categories in a clean multi-column layout."""
    lines = []
    # Card background
    lines.append(f'  <rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" class="card-bg" />')

    # Card title
    card_title_text = html.escape(panel_summary(panel_data))
    lines.append(f'  <text x="{card_x + 16}" y="{card_y + 24}" class="card-title">{card_title_text}</text>')

    # Arrange categories inside card
    cats = panel_data.get("categories", [])
    if cats:
        grid_top = card_y + 36
        grid_left = card_x + 16
        grid_w = card_w - 32
        grid_h = card_h - 62  # leave space for small items

        n_cats = len(cats)
        # Select layout columns based on category count
        if n_cats <= 4:
            n_cols = n_cats
            col_w = (grid_w - (n_cols - 1) * 8) / n_cols
            for idx, (cat_name, cat_cov, cat_tot) in enumerate(cats):
                cx = grid_left + idx * (col_w + 8)
                lines.append(render_category_grid(cat_name, cat_cov, cat_tot, cx, grid_top, col_w, grid_h))
        elif n_cats <= 8:
            n_cols = 4
            col_w = (grid_w - (n_cols - 1) * 8) / n_cols
            row_h = (grid_h - 8) / 2
            for idx, (cat_name, cat_cov, cat_tot) in enumerate(cats):
                r = idx // n_cols
                c = idx % n_cols
                cx = grid_left + c * (col_w + 8)
                cy = grid_top + r * (row_h + 8)
                lines.append(render_category_grid(cat_name, cat_cov, cat_tot, cx, cy, col_w, row_h))
        else:
            # 5-column layout for larger panels
            n_cols = 5
            col_w = (grid_w - (n_cols - 1) * 6) / n_cols
            n_rows = (n_cats + n_cols - 1) // n_cols
            row_h = (grid_h - (n_rows - 1) * 6) / n_rows
            for idx, (cat_name, cat_cov, cat_tot) in enumerate(cats):
                r = idx // n_cols
                c = idx % n_cols
                cx = grid_left + c * (col_w + 6)
                cy = grid_top + r * (row_h + 6)
                lines.append(render_category_grid(cat_name, cat_cov, cat_tot, cx, cy, col_w, row_h))

    # Small items at bottom
    small_items = panel_data.get("small_items", [])
    if small_items:
        items_str = ", ".join(f"{name} {c}/{t}" for name, c, t in small_items)
        small_text = html.escape(f"Too small to label: {items_str}")
        lines.append(f'  <text x="{card_x + 16}" y="{card_y + card_h - 12}" class="too-small">{small_text}</text>')

    return "\n".join(lines)


def render_dashboard_svg(panels, commit_sha, commit_date):
    """Render the full SVG document containing all panels."""
    viewbox_w = 980
    card_x = 20
    card_w = 940

    card_heights = []
    for p in panels:
        num_cats = len(p.get("categories", []))
        if num_cats <= 4:
            card_heights.append(170)
        elif num_cats <= 8:
            card_heights.append(230)
        else:
            card_heights.append(280)

    header_height = 80
    gap = 16
    total_height = header_height + sum(card_heights) + gap * len(card_heights) + 16

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox_w} {total_height}" width="100%" height="100%">',
        '  <defs>',
        '    <style>',
        '      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }',
        '      .bg { fill: #0d1117; }',
        '      .card-bg { fill: #161b22; stroke: #30363d; stroke-width: 1px; rx: 6px; }',
        '      .title { font-size: 18px; font-weight: 600; fill: #f0f6fc; }',
        '      .subtitle { font-size: 12px; fill: #8b949e; }',
        '      .meta { font-size: 11px; fill: #8b949e; }',
        '      .card-title { font-size: 13px; font-weight: 600; fill: #f0f6fc; }',
        '      .cat-label { font-size: 10px; font-weight: 500; fill: #c9d1d9; }',
        '      .too-small { font-size: 10px; fill: #8b949e; }',
        '    </style>',
        '  </defs>',
        f'  <rect width="{viewbox_w}" height="{total_height}" class="bg" />',
        '  <!-- Header -->',
        '  <text x="20" y="32" class="title">Leonida — Codebase &amp; Instruction Coverage</text>',
        '  <text x="20" y="52" class="subtitle">Instruction baselines have no documented measurement source. Green cells represent each panel’s numerator.</text>',
        f'  <text x="20" y="68" class="meta">Generated from checkout {html.escape(commit_sha)} ({html.escape(commit_date)}).</text>',
    ]

    curr_y = header_height
    for p, ch in zip(panels, card_heights):
        svg_parts.append(render_card(p, card_x, curr_y, card_w, ch))
        curr_y += ch + gap

    svg_parts.append('</svg>')
    return "\n".join(svg_parts) + "\n"


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output SVG path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--check", action="store_true", help="Check that the output includes current metric totals and their basis")
    args = parser.parse_args()

    commit_sha, commit_date = get_git_metadata()

    # Collect all coverage data
    rdna2_data = get_rdna2_coverage()
    x64_data = get_x64_coverage()
    runtime_data = get_runtime_coverage()
    tools_data = get_tools_coverage()

    panels = [rdna2_data, x64_data, runtime_data, tools_data]

    # Generate the SVG string
    svg_content = render_dashboard_svg(panels, commit_sha, commit_date)

    # Validate SVG XML well-formedness
    try:
        ET.fromstring(svg_content)
    except Exception as e:
        sys.stderr.write(f"Error: Generated SVG is malformed XML: {e}\n")
        sys.exit(1)

    if args.check:
        if not os.path.exists(args.output):
            sys.stderr.write(f"Check failed: {args.output} does not exist. Run without --check to generate it.\n")
            sys.exit(1)

        try:
            with open(args.output, "r", encoding="utf-8") as fh:
                existing = fh.read()
        except Exception as e:
            sys.stderr.write(f"Check failed: could not read {args.output}: {e}\n")
            sys.exit(1)

        # Check the displayed totals and their stated basis.
        for p in panels:
            expected_snippet = html.escape(panel_summary(p))
            if expected_snippet not in existing:
                sys.stderr.write(f"Check failed: metric drift in {args.output} (expected '{expected_snippet}').\n")
                sys.exit(1)

        print(f"{args.output} is up to date.")
        sys.exit(0)

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg_content)

    print(f"Wrote {args.output} ({len(svg_content.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    main()
