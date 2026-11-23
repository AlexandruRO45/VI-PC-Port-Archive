#!/usr/bin/env python3
"""Generate or verify visual SVG coverage treemap dashboard for Leonida.

Distinct translator mnemonics mentioned in semantic-test fixtures (counted against a fixed RDNA2 catalog; not a measure of game completion).
"""
import argparse
import ast
import html
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "docs", "coverage", "dashboard.svg")
METRIC_INPUT_PATHS = ["tools", "tests", "shims", "runtime", "src"]


# ---------------------------------------------------------------------------
# Metric Collectors & Provenance
# ---------------------------------------------------------------------------

def get_git_metadata(repo_root=REPO_ROOT):
    """Retrieve the provenance commit short hash and ISO timestamp deterministically.

    Filters git log to only commits touching metric-input paths (tools, tests, shims, runtime, src).
    Guards against shallow clones or commits that only touched docs/coverage/dashboard.svg or .github/:
    checks if the candidate commit touched actual metric-input files and falls back cleanly
    to ("no-metrics", "1970-01-01T00:00:00Z") if no metric-touching commit exists in local history.
    """
    commit_sha = "no-metrics"
    commit_date = "1970-01-01T00:00:00Z"

    try:
        cmd = ["git", "log", "-1", "--format=%h|%cI", "--"] + METRIC_INPUT_PATHS
        r = subprocess.run(
            cmd,
            cwd=repo_root, capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("|", 1)
            candidate_sha = parts[0]
            candidate_date = parts[1] if len(parts) > 1 else "1970-01-01T00:00:00Z"

            # Check if repo is shallow and whether candidate_sha has a parent
            r_shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo_root, capture_output=True, text=True, timeout=5)
            is_shallow = (r_shallow.returncode == 0 and r_shallow.stdout.strip() == "true")

            r_parent = subprocess.run(["git", "rev-parse", "--verify", f"{candidate_sha}^"], cwd=repo_root, capture_output=True, text=True, timeout=5)
            has_parent = (r_parent.returncode == 0)

            if has_parent:
                diff_cmd = ["git", "diff-tree", "-m", "-r", "--no-commit-id", "--name-only", candidate_sha]
                rdiff = subprocess.run(diff_cmd, cwd=repo_root, capture_output=True, text=True, timeout=5)
                if rdiff.returncode == 0 and rdiff.stdout.strip():
                    touched = [f.strip().replace("\\", "/") for f in rdiff.stdout.splitlines() if f.strip()]
                    if any(any(f == p or f.startswith(p + "/") for p in METRIC_INPUT_PATHS) for f in touched):
                        commit_sha = candidate_sha
                        commit_date = candidate_date
            elif not is_shallow:
                diff_cmd = ["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", candidate_sha]
                rdiff = subprocess.run(diff_cmd, cwd=repo_root, capture_output=True, text=True, timeout=5)
                if rdiff.returncode == 0 and rdiff.stdout.strip():
                    touched = [f.strip().replace("\\", "/") for f in rdiff.stdout.splitlines() if f.strip()]
                    if any(any(f == p or f.startswith(p + "/") for p in METRIC_INPUT_PATHS) for f in touched):
                        commit_sha = candidate_sha
                        commit_date = candidate_date
    except Exception:
        pass

    return commit_sha, commit_date


MNEMONIC = re.compile(r"^((s|v|ds|buffer|tbuffer|global|flat|scratch|image)(_[a-z0-9]+)+|exp)$")

# Documented catalog of GFX1030 / RDNA2 instruction-form encoding totals (fixed 1,865 denominator)
CATEGORY_TOTALS = [
    ("VOP3", 425),
    ("VOP2", 47),
    ("VOP1", 81),
    ("VOPC", 190),
    ("VOP3P", 29),
    ("SOP1", 65),
    ("SOP2", 51),
    ("SOPC", 18),
    ("SOPK", 27),
    ("SOPP", 37),
    ("SMEM", 84),
    ("DS", 154),
    ("MUBUF", 73),
    ("MTBUF", 16),
    ("FLAT", 133),
    ("MIMG", 111),
    ("SDWA", 220),
    ("DPP8", 100),
    ("VINTRP", 3),
    ("EXP", 1),
]

# Load architectural RDNA2 decoder tables
try:
    import rdna2_isa_tables as _T
except ImportError:
    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    try:
        import rdna2_isa_tables as _T
    except ImportError:
        _T = None


def _extract_table_opcodes(tbl):
    if not tbl:
        return set()
    s = set()
    for val in tbl.values():
        name = val[0] if isinstance(val, (tuple, list)) else val
        s.add(name.lower())
    return s


VOP2_OPCODES = _extract_table_opcodes(_T.VOP2) if _T and hasattr(_T, "VOP2") else set()
VOP1_OPCODES = _extract_table_opcodes(_T.VOP1) if _T and hasattr(_T, "VOP1") else set()
VOPC_OPCODES = _extract_table_opcodes(_T.VOPC) if _T and hasattr(_T, "VOPC") else set()
VOP3P_OPCODES = _extract_table_opcodes(_T.VOP3P) if _T and hasattr(_T, "VOP3P") else set()
VOP3_OPCODES = (_extract_table_opcodes(_T.VOP3) | _extract_table_opcodes(getattr(_T, "VOP3B", {}))) if _T and hasattr(_T, "VOP3") else set()

SOP1_OPCODES = _extract_table_opcodes(_T.SOP1) if _T and hasattr(_T, "SOP1") else set()
SOP2_OPCODES = _extract_table_opcodes(_T.SOP2) if _T and hasattr(_T, "SOP2") else set()
SOPC_OPCODES = _extract_table_opcodes(_T.SOPC) if _T and hasattr(_T, "SOPC") else set()
SOPK_OPCODES = _extract_table_opcodes(_T.SOPK) if _T and hasattr(_T, "SOPK") else set()
SOPP_OPCODES = _extract_table_opcodes(_T.SOPP) if _T and hasattr(_T, "SOPP") else set()
SMEM_OPCODES = _extract_table_opcodes(_T.SMEM) if _T and hasattr(_T, "SMEM") else set()
DS_OPCODES = _extract_table_opcodes(_T.DS) if _T and hasattr(_T, "DS") else set()
MUBUF_OPCODES = _extract_table_opcodes(_T.MUBUF) if _T and hasattr(_T, "MUBUF") else set()
MTBUF_OPCODES = _extract_table_opcodes(_T.MTBUF) if _T and hasattr(_T, "MTBUF") else set()
MIMG_OPCODES = _extract_table_opcodes(_T.MIMG) if _T and hasattr(_T, "MIMG") else set()
FLAT_OPCODES = (_extract_table_opcodes(_T.FLAT) | _extract_table_opcodes(getattr(_T, "GLOBAL", {})) | _extract_table_opcodes(getattr(_T, "SCRATCH", {}))) if _T and hasattr(_T, "FLAT") else set()
VINTRP_OPCODES = _extract_table_opcodes(_T.VINTRP) if _T and hasattr(_T, "VINTRP") else set()


def classify_rdna2_instruction(op):
    """Classify an RDNA2 opcode mnemonic into its instruction family category.

    Derives instruction encoding classes directly from repository RDNA2 decoder tables:
    - Any mnemonic ending in _e64 (such as v_cmp_eq_f32_e64) is always VOP3.
    - Suffixes _sdwa -> SDWA, _dpp / _dpp8 / _dpp16 -> DPP8.
    - Suffix _e32 strips to base mnemonic.
    - Multi-operand instructions (v_bfe_u32, v_med3_f32, v_min3_f32, v_sad_u8, v_mul_hi_u32, etc.) classify as VOP3.
    - v_cvt_pkrtz_f16_f32 classifies as VOP2.
    - Strictly validates against table sets; never guesses VOP2 for arbitrary v_ names.
    """
    op = op.lower()

    if op.endswith("_sdwa"):
        return "SDWA"
    if op.endswith("_dpp8") or op.endswith("_dpp16") or op.endswith("_dpp"):
        return "DPP8"

    # _e64 forms are architecturally VOP3 encodings
    if op.endswith("_e64"):
        return "VOP3"

    # Strip _e32 for scalar-encoded variants
    if op.endswith("_e32"):
        op = op[:-4]

    # Explicit lookup against architectural table sets
    if op in VOPC_OPCODES:
        return "VOPC"
    if op in VOP1_OPCODES:
        return "VOP1"
    if op in VOP2_OPCODES:
        return "VOP2"
    if op in VOP3P_OPCODES:
        return "VOP3P"
    if op in VOP3_OPCODES:
        return "VOP3"

    if op in SOP1_OPCODES:
        return "SOP1"
    if op in SOP2_OPCODES:
        return "SOP2"
    if op in SOPC_OPCODES:
        return "SOPC"
    if op in SOPK_OPCODES:
        return "SOPK"
    if op in SOPP_OPCODES:
        return "SOPP"
    if op in SMEM_OPCODES:
        return "SMEM"
    if op in DS_OPCODES:
        return "DS"
    if op in MUBUF_OPCODES:
        return "MUBUF"
    if op in MTBUF_OPCODES:
        return "MTBUF"
    if op in MIMG_OPCODES:
        return "MIMG"
    if op in FLAT_OPCODES:
        return "FLAT"
    if op in VINTRP_OPCODES:
        return "VINTRP"
    if op == "exp":
        return "EXP"

    # Prefixes for remaining forms
    if op.startswith("ds_"):
        return "DS"
    if op.startswith("buffer_"):
        return "MUBUF"
    if op.startswith("tbuffer_"):
        return "MTBUF"
    if op.startswith("image_"):
        return "MIMG"
    if op.startswith("flat_") or op.startswith("global_") or op.startswith("scratch_"):
        return "FLAT"
    if op.startswith("v_interp_"):
        return "VINTRP"
    if op.startswith("v_cmp_") or op.startswith("v_cmpx_"):
        return "VOPC"
    if op.startswith("v_pk_"):
        return "VOP3P"
    if op.startswith("s_cmp_"):
        return "SOPC"
    if op.startswith("s_load_") or op.startswith("s_buffer_load_"):
        return "SMEM"
    if (op.startswith("s_cbranch_") or op.startswith("s_branch") or
        op.startswith("s_endpgm") or op in ("s_nop", "s_waitcnt", "s_barrier", "s_trap", "s_sethalt", "s_sleep", "s_sendmsg", "s_sendmsghalt")):
        return "SOPP"
    if op.startswith("s_movk_") or op.startswith("s_setreg_") or op.startswith("s_getreg_"):
        return "SOPK"
    if op.startswith("s_"):
        if any(op.startswith(p) for p in (
            "s_add_", "s_sub_", "s_and_", "s_or_", "s_xor_", "s_nand_", "s_nor_",
            "s_xnor_", "s_lshl_", "s_lshr_", "s_ashr_", "s_mul_", "s_bfe_", "s_bfm_"
        )):
            return "SOP2"
        return "SOP1"
    if any(op.startswith(p) for p in ("v_add3_", "v_lshl_add_", "v_fma_", "v_mad_", "v_alignbit_", "v_alignbyte_", "v_cube", "v_perm_")):
        return "VOP3"

    # Do not guess VOP2 for arbitrary unknown v_ mnemonics
    return None


def extract_exercised_semantic_opcodes(semantic_dir):
    """Extract distinct genuine RDNA2 instruction mnemonics exercised in semantic test fixtures.

    - Scans .asm and .s fixtures in semantic_dir (stripping ; and // comments, extracting opcode tokens).
    - Scans .py test fixtures, stripping module/class/function docstrings and standalone string expressions.
    - Files containing only comments or docstrings produce zero extracted instruction tokens.
    """
    exercised = set()
    if not os.path.isdir(semantic_dir):
        return exercised

    for root, _, files in os.walk(semantic_dir):
        # 1. Assembly fixtures (.asm, .s)
        for fname in files:
            if fname.endswith(".asm") or fname.endswith(".s"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            line = line.split(";", 1)[0].split("//", 1)[0].strip()
                            if not line or line.endswith(":"):
                                continue
                            if ":" in line:
                                line = line.split(":", 1)[1].strip()
                            tokens = line.split()
                            if tokens:
                                mnemonic = tokens[0].lower()
                                if MNEMONIC.match(mnemonic):
                                    exercised.add(mnemonic)
                except Exception:
                    pass

        # 2. Python semantic test fixtures (test_*.py)
        for fname in files:
            if not (fname.startswith("test_") and fname.endswith(".py")):
                continue
            fpath = os.path.join(root, fname)
            try:
                if os.path.getsize(fpath) == 0:
                    continue
                with open(fpath, "rb") as fh:
                    raw_bytes = fh.read()
                if not raw_bytes.strip():
                    continue

                tree = ast.parse(raw_bytes, fpath)
                docstring_ids = set()

                # Leading module docstring
                if (tree.body and isinstance(tree.body[0], ast.Expr) and
                        isinstance(tree.body[0].value, ast.Constant) and
                        isinstance(tree.body[0].value.value, str)):
                    docstring_ids.add(id(tree.body[0].value))

                # Function and class docstrings, and standalone string expressions
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if (node.body and isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str)):
                            docstring_ids.add(id(node.body[0].value))
                    elif (isinstance(node, ast.Expr) and
                          isinstance(node.value, ast.Constant) and
                          isinstance(node.value.value, str)):
                        docstring_ids.add(id(node.value))

                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if id(node) in docstring_ids:
                            continue
                        for word in re.findall(r"\b[a-z0-9_]+\b", node.value.lower()):
                            if MNEMONIC.match(word):
                                exercised.add(word)
            except Exception:
                pass

    return exercised


def get_translator_lowered_opcodes(repo_root=REPO_ROOT):
    """Extract opcodes lowered/handled by tools/rdna2spirv.py."""
    translator_path = os.path.join(repo_root, "tools", "rdna2spirv.py")
    if not os.path.exists(translator_path):
        translator_path = os.path.join(REPO_ROOT, "tools", "rdna2spirv.py")
    if not os.path.exists(translator_path):
        return set()

    try:
        tools_dir = os.path.dirname(translator_path)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from isa_coverage import scan_translator
        info = scan_translator(translator_path)
        return set(info["exact"].keys())
    except Exception:
        pass

    opcodes = set()
    try:
        with open(translator_path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read(), translator_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.lower()
                if MNEMONIC.match(val):
                    opcodes.add(val)
    except Exception:
        pass
    return opcodes


def get_rdna2_coverage(repo_root=REPO_ROOT):
    """Collect GFX1030 (RDNA2) instruction-form coverage dynamically.

    Starts strictly at 0 covered across every category on an empty tree.
    Derives covered counts dynamically from documented RDNA2 encoding-form mappings
    in tools/rdna2spirv.py exercised by test fixtures in tests/semantic/.
    The denominator is strictly fixed at 1,865 catalog forms (no max(tot, cov) inflation; covered <= total always holds).
    """
    semantic_dir = os.path.join(repo_root, "tests", "semantic")
    exercised = extract_exercised_semantic_opcodes(semantic_dir)
    translator_ops = get_translator_lowered_opcodes(repo_root)

    valid_exercised = exercised & translator_ops

    # Count distinct exercised opcodes per category starting at 0
    cat_counts = {cat: set() for cat, _ in CATEGORY_TOTALS}
    for op in valid_exercised:
        cat = classify_rdna2_instruction(op)
        if cat in cat_counts:
            cat_counts[cat].add(op)

    categories = []
    small_items = []
    for cat, tot in CATEGORY_TOTALS:
        cov = min(len(cat_counts[cat]), tot)
        if tot <= 30 and cat in ("VINTRP", "EXP", "MTBUF", "SOPC", "VOP3P"):
            small_items.append((cat, cov, tot))
        else:
            categories.append((cat, cov, tot))

    tot_cov = sum(c for _, c, _ in categories) + sum(c for _, c, _ in small_items)
    tot_all = sum(t for _, _, t in categories) + sum(t for _, _, t in small_items)

    return {
        "title": "gfx1030 (RDNA2) forms",
        "covered": tot_cov,
        "total": tot_all,
        "skipped": 0,
        "categories": categories,
        "small_items": small_items,
    }


# ---------------------------------------------------------------------------
# SVG Layout and Rendering
# ---------------------------------------------------------------------------

def render_category_grid(name, covered, total, x, y, width, height):
    """Render an individual category container with a cell grid."""
    lines = []
    lines.append(f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" stroke="#30363d" stroke-width="1" />')

    label_text = html.escape(f"{name} {covered}/{total}")
    lines.append(f'  <text x="{x + 6}" y="{y + 13}" class="cat-label">{label_text}</text>')

    grid_x = x + 6
    grid_y = y + 20
    avail_w = width - 12
    avail_h = height - 26

    cell_pitch = 8
    cols = max(1, int(avail_w // cell_pitch))
    max_cells_visible = int((avail_w // cell_pitch) * (avail_h // cell_pitch))

    cells_to_draw = min(total, max_cells_visible) if total > 0 else 0

    for i in range(cells_to_draw):
        row = i // cols
        col = i % cols
        cx = grid_x + col * cell_pitch
        cy = grid_y + row * cell_pitch

        if i < covered:
            fill_color = "#238636"
        else:
            fill_color = "#30363d"

        lines.append(f'  <rect x="{cx}" y="{cy}" width="6" height="6" rx="1" fill="{fill_color}" />')

    return "\n".join(lines)


def render_card(panel_data, card_x, card_y, card_w, card_h):
    """Render a section card containing multiple categories in a clean multi-column layout."""
    lines = []
    lines.append(f'  <rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" class="card-bg" />')

    title = panel_data["title"]
    cov = panel_data["covered"]
    tot = panel_data["total"]
    skips = panel_data.get("skipped", 0)
    card_title_text = html.escape(f"{title}: {cov} of {tot} covered, {skips} held back by skips")
    lines.append(f'  <text x="{card_x + 16}" y="{card_y + 24}" class="card-title">{card_title_text}</text>')

    cats = panel_data.get("categories", [])
    if cats:
        grid_top = card_y + 36
        grid_left = card_x + 16
        grid_w = card_w - 32
        grid_h = card_h - 62

        n_cats = len(cats)
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
        '  <text x="20" y="32" class="title">Leonida — Coverage Dashboard</text>',
        '  <text x="20" y="52" class="subtitle">Distinct translator mnemonics mentioned in semantic-test fixtures (counted against a fixed RDNA2 catalog; not a measure of game completion).</text>',
        f'  <text x="20" y="68" class="meta">Measured on {html.escape(commit_sha)}, {html.escape(commit_date)}. Green: covered. Grey: not covered.</text>',
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
    parser.add_argument("--check", action="store_true", help="Check that the output file exists and matches current metrics without drift")
    parser.add_argument("--repo-root", default=REPO_ROOT, help=f"Repository root directory (default: {REPO_ROOT})")
    args = parser.parse_args()

    commit_sha, commit_date = get_git_metadata(args.repo_root)

    # Collect RDNA2 coverage data dynamically
    rdna2_data = get_rdna2_coverage(args.repo_root)
    panels = [rdna2_data]

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

        # Check top-line metrics parity
        for p in panels:
            title = p["title"]
            cov = p["covered"]
            tot = p["total"]
            expected_snippet = html.escape(f"{title}: {cov} of {tot} covered")
            if expected_snippet not in existing:
                sys.stderr.write(f"Check failed: metric drift in {args.output} (expected '{expected_snippet}').\n")
                sys.exit(1)

            # Check individual category parity to catch count shifts that preserve total
            for cat_name, cat_cov, cat_tot in p.get("categories", []):
                cat_label = html.escape(f"{cat_name} {cat_cov}/{cat_tot}")
                if cat_label not in existing:
                    sys.stderr.write(f"Check failed: category metric drift in {args.output} (expected '{cat_label}').\n")
                    sys.exit(1)

            for cat_name, cat_cov, cat_tot in p.get("small_items", []):
                small_str = html.escape(f"{cat_name} {cat_cov}/{cat_tot}")
                if small_str not in existing:
                    sys.stderr.write(f"Check failed: small item drift in {args.output} (expected '{small_str}').\n")
                    sys.exit(1)

        # Full SVG equality check (ignoring transient commit metadata line)
        existing_nometa = re.sub(r'<text [^>]*class="meta"[^>]*>.*?</text>', '', existing, flags=re.S)
        svg_nometa = re.sub(r'<text [^>]*class="meta"[^>]*>.*?</text>', '', svg_content, flags=re.S)
        if existing_nometa.strip() != svg_nometa.strip():
            sys.stderr.write(f"Check failed: content mismatch / metric drift in {args.output}.\n")
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
