"""Compare two structured GPU test reports produced by scripts/run_gpu_tests.py --json.

Usage:
    python -I scripts/compare_gpu_reports.py BASELINE CANDIDATE [--json FILE] [--allow-incompatible]

Distinguishes hard regressions (pass -> fail), coverage regressions (pass -> skip), missing tests,
new failures (skip -> fail, added fail), improvements (fail -> pass, skip -> pass), and fail -> skip
transitions, while surfacing incompatible run settings, GPU, or driver mismatches between runs.

Exit codes:
    0  No regressions, complete/unambiguous reports, and runs are compatible (or --allow-incompatible was given).
    1  One or more regressions, run-level error on either side, empty report, duplicate test names, or incompatible runs.
    2  Usage error, unreadable/invalid/inconsistent report JSON, or --json overwriting an input.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from report_paths import same_report_path  # noqa: E402

SCHEMA_VERSION = 1
VALID_STATUSES = ("pass", "fail", "skip")
STATUS_SEVERITY = {"pass": 0, "skip": 1, "fail": 2}

# Effective run settings that change test selection, shader translation, or pass/fail/skip judgment.
COMPARABLE_SETTINGS = (
    "filter",
    "disasm_mode",
    "assembler",
    "validate",
    "wave64_emulate",
    "bary_fallback",
    "f16_interp_fallback",
    "texel_offset_fallback",
    "mesh_compute",
    "pc_tail_ubo",
    "fma",
    "vk_validation",
)

GPU_IDENTITY_FIELDS = ("name", "vendor_id", "device_id")
DRIVER_IDENTITY_FIELDS = ("id", "name", "info", "version")


class ReportError(ValueError):
    """Raised when an input JSON report is unreadable or does not match schema v1."""


def load_report(path: str) -> dict:
    """Read and validate one GPU test JSON report."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            doc = json.load(f)
    except OSError as exc:
        raise ReportError(f"cannot read report {path}: {exc.strerror or exc}") from None
    except ValueError as exc:
        raise ReportError(f"invalid JSON in {path}: {exc}") from None

    if not isinstance(doc, dict):
        raise ReportError(f"{path}: report root must be a JSON object")
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ReportError(f"{path}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION})")
    for section in ("revision", "gpu", "driver", "settings", "summary"):
        if not isinstance(doc.get(section), dict):
            raise ReportError(f"{path}: missing or invalid {section!r} object")
    if "error" in doc and doc["error"] is not None:
        if not isinstance(doc["error"], str) or not doc["error"].strip():
            raise ReportError(f"{path}: 'error' must be a non-empty string or null")
    tests = doc.get("tests")
    if not isinstance(tests, list):
        raise ReportError(f"{path}: missing or invalid 'tests' list")
    for idx, entry in enumerate(tests):
        if not isinstance(entry, dict):
            raise ReportError(f"{path}: tests[{idx}] must be an object")
        name = entry.get("name")
        status = entry.get("status")
        if not isinstance(name, str) or not name:
            raise ReportError(f"{path}: tests[{idx}].name must be a non-empty string")
        if status not in VALID_STATUSES:
            raise ReportError(f"{path}: tests[{idx}].status must be one of {VALID_STATUSES}, got {status!r}")
        detail = entry.get("detail", "")
        if not isinstance(detail, str):
            raise ReportError(f"{path}: tests[{idx}].detail must be a string")

    summary = doc["summary"]
    for key in ("total", "passed", "failed", "skipped"):
        val = summary.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ReportError(f"{path}: summary.{key} must be a non-negative integer, got {val!r}")
    if "exit_code" in summary:
        ec = summary["exit_code"]
        if not isinstance(ec, int) or isinstance(ec, bool):
            raise ReportError(f"{path}: summary.exit_code must be an integer, got {ec!r}")
    actual_total = len(tests)
    actual_passed = sum(1 for entry in tests if entry["status"] == "pass")
    actual_failed = sum(1 for entry in tests if entry["status"] == "fail")
    actual_skipped = sum(1 for entry in tests if entry["status"] == "skip")
    if (summary["total"], summary["passed"], summary["failed"], summary["skipped"]) != (
        actual_total, actual_passed, actual_failed, actual_skipped
    ):
        raise ReportError(
            f"{path}: summary counts ({summary['total']} total, {summary['passed']} pass, "
            f"{summary['failed']} fail, {summary['skipped']} skip) disagree with test rows "
            f"({actual_total} total, {actual_passed} pass, {actual_failed} fail, {actual_skipped} skip)"
        )
    return doc


def index_tests(tests: list[dict], label: str) -> tuple[dict[str, dict], list[str], list[str]]:
    """Index tests by name, keeping worst status on duplicates (fail > skip > pass) and noting duplicates."""
    by_name: dict[str, dict] = {}
    duplicates: list[str] = []
    seen_dup: set[str] = set()
    for entry in tests:
        name = entry["name"]
        if name in by_name:
            if name not in seen_dup:
                seen_dup.add(name)
                duplicates.append(name)
            prev = by_name[name]
            if STATUS_SEVERITY[entry["status"]] > STATUS_SEVERITY[prev["status"]]:
                by_name[name] = entry
        else:
            by_name[name] = entry
    warnings = [f"{label} report contains duplicate entries for test {name!r}" for name in duplicates]
    return by_name, duplicates, warnings


def find_incompatibilities(baseline: dict, candidate: dict) -> list[dict]:
    """Return structured mismatches in runner, GPU, driver, or effective run settings."""
    mismatches: list[dict] = []

    if baseline.get("runner") != candidate.get("runner"):
        mismatches.append({
            "scope": "runner",
            "field": "runner",
            "baseline": baseline.get("runner"),
            "candidate": candidate.get("runner"),
        })

    base_gpu, cand_gpu = baseline["gpu"], candidate["gpu"]
    for field in GPU_IDENTITY_FIELDS:
        bv, cv = base_gpu.get(field), cand_gpu.get(field)
        if bv != cv:
            mismatches.append({"scope": "gpu", "field": field, "baseline": bv, "candidate": cv})

    base_drv, cand_drv = baseline["driver"], candidate["driver"]
    for field in DRIVER_IDENTITY_FIELDS:
        bv, cv = base_drv.get(field), cand_drv.get(field)
        if bv != cv:
            mismatches.append({"scope": "driver", "field": field, "baseline": bv, "candidate": cv})

    base_set, cand_set = baseline["settings"], candidate["settings"]
    for field in COMPARABLE_SETTINGS:
        bv, cv = base_set.get(field), cand_set.get(field)
        if bv != cv:
            mismatches.append({"scope": "settings", "field": field, "baseline": bv, "candidate": cv})

    # If GPU identity could not be probed on either side, surface a differing LEONIDA_VK_DEVICE filter.
    if base_gpu.get("name") is None or cand_gpu.get("name") is None:
        bv, cv = base_set.get("vk_device_filter"), cand_set.get("vk_device_filter")
        if bv != cv:
            mismatches.append({"scope": "settings", "field": "vk_device_filter", "baseline": bv, "candidate": cv})

    return mismatches


def compare_reports(
    baseline: dict,
    candidate: dict,
    *,
    baseline_path: str | None = None,
    candidate_path: str | None = None,
    allow_incompatible: bool = False,
) -> dict:
    """Compare validated baseline and candidate GPU test reports and return a structured result dict."""
    incompatibilities = find_incompatibilities(baseline, candidate)
    compatible = not incompatibilities

    base_tests, base_dups, base_warnings = index_tests(baseline["tests"], "baseline")
    cand_tests, cand_dups, cand_warnings = index_tests(candidate["tests"], "candidate")
    warnings = base_warnings + cand_warnings

    pass_to_fail: list[dict] = []
    pass_to_skip: list[dict] = []
    missing: list[dict] = []
    skip_to_fail: list[dict] = []
    added_fail: list[dict] = []

    fail_to_pass: list[dict] = []
    skip_to_pass: list[dict] = []
    fail_to_skip: list[dict] = []
    added_pass: list[dict] = []
    added_skip: list[dict] = []

    unchanged_pass = 0
    unchanged_skip = 0
    unchanged_fail: list[dict] = []

    all_names = sorted(set(base_tests) | set(cand_tests))
    for name in all_names:
        b = base_tests.get(name)
        c = cand_tests.get(name)
        if b is not None and c is None:
            missing.append({
                "name": name,
                "baseline_status": b["status"],
                "baseline_detail": b.get("detail", ""),
            })
            continue
        if b is None and c is not None:
            row = {
                "name": name,
                "candidate_status": c["status"],
                "candidate_detail": c.get("detail", ""),
            }
            if c["status"] == "fail":
                added_fail.append(row)
            elif c["status"] == "pass":
                added_pass.append(row)
            else:
                added_skip.append(row)
            continue

        assert b is not None and c is not None
        bs, cs = b["status"], c["status"]
        if bs == cs:
            if bs == "pass":
                unchanged_pass += 1
            elif bs == "skip":
                unchanged_skip += 1
            else:
                unchanged_fail.append({
                    "name": name,
                    "baseline_detail": b.get("detail", ""),
                    "candidate_detail": c.get("detail", ""),
                })
            continue

        diff_row = {
            "name": name,
            "baseline_status": bs,
            "candidate_status": cs,
            "baseline_detail": b.get("detail", ""),
            "candidate_detail": c.get("detail", ""),
        }
        if bs == "pass" and cs == "fail":
            pass_to_fail.append(diff_row)
        elif bs == "pass" and cs == "skip":
            if "skip_reason" in c:
                diff_row["candidate_skip_reason"] = c["skip_reason"]
            pass_to_skip.append(diff_row)
        elif bs == "skip" and cs == "fail":
            skip_to_fail.append(diff_row)
        elif bs == "fail" and cs == "pass":
            fail_to_pass.append(diff_row)
        elif bs == "skip" and cs == "pass":
            skip_to_pass.append(diff_row)
        elif bs == "fail" and cs == "skip":
            if "skip_reason" in c:
                diff_row["candidate_skip_reason"] = c["skip_reason"]
            fail_to_skip.append(diff_row)

    baseline_error = baseline.get("error")
    candidate_error = candidate.get("error")
    incomplete_reasons: list[str] = []
    if baseline_error:
        incomplete_reasons.append(f"baseline run aborted with error: {baseline_error}")
    elif not baseline["tests"]:
        incomplete_reasons.append("baseline report contains no test rows")
    if candidate_error:
        incomplete_reasons.append(f"candidate run aborted with error: {candidate_error}")
    elif not candidate["tests"]:
        incomplete_reasons.append("candidate report contains no test rows")

    incomplete = bool(incomplete_reasons)
    ambiguous = bool(base_dups or cand_dups)
    all_skipped = bool(
        baseline["tests"]
        and candidate["tests"]
        and all(entry["status"] == "skip" for entry in baseline["tests"])
        and all(entry["status"] == "skip" for entry in candidate["tests"])
    )

    has_regressions = bool(
        pass_to_fail or pass_to_skip or missing or skip_to_fail or added_fail or candidate_error
    )
    exit_code = 1 if (
        has_regressions
        or incomplete
        or ambiguous
        or (not compatible and not allow_incompatible)
    ) else 0

    def side_meta(doc: dict, path: str | None) -> dict:
        return {
            "path": path,
            "revision": doc["revision"],
            "gpu": doc["gpu"],
            "driver": doc["driver"],
            "settings": doc["settings"],
            "summary": doc["summary"],
            "error": doc.get("error"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "compatible": compatible,
        "incompatibilities": incompatibilities,
        "incomplete": incomplete,
        "incomplete_reasons": incomplete_reasons,
        "ambiguous": ambiguous,
        "duplicates": {"baseline": base_dups, "candidate": cand_dups},
        "all_skipped": all_skipped,
        "baseline": side_meta(baseline, baseline_path),
        "candidate": side_meta(candidate, candidate_path),
        "counts": {
            "pass_to_fail": len(pass_to_fail),
            "pass_to_skip": len(pass_to_skip),
            "missing": len(missing),
            "skip_to_fail": len(skip_to_fail),
            "added_fail": len(added_fail),
            "fail_to_pass": len(fail_to_pass),
            "skip_to_pass": len(skip_to_pass),
            "fail_to_skip": len(fail_to_skip),
            "added_pass": len(added_pass),
            "added_skip": len(added_skip),
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": len(unchanged_fail),
            "unchanged_skip": unchanged_skip,
        },
        "regressions": {
            "pass_to_fail": pass_to_fail,
            "pass_to_skip": pass_to_skip,
            "missing": missing,
            "skip_to_fail": skip_to_fail,
            "added_fail": added_fail,
        },
        "changes": {
            "fail_to_pass": fail_to_pass,
            "skip_to_pass": skip_to_pass,
            "fail_to_skip": fail_to_skip,
            "added_pass": added_pass,
            "added_skip": added_skip,
        },
        "unchanged_fail": unchanged_fail,
        "warnings": warnings,
        "has_regressions": has_regressions,
        "exit_code": exit_code,
    }


def _one_line(text: str, limit: int = 160) -> str:
    clean = " | ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(clean) > limit:
        return clean[: limit - 3] + "..."
    return clean


def _format_header(label: str, side: dict) -> str:
    rev = side["revision"]
    commit = rev.get("short_commit") or rev.get("commit") or "unknown"
    dirty = rev.get("dirty")
    rev_text = f"{commit} ({'dirty' if dirty else 'clean'})" if dirty is not None else commit

    gpu_name = side["gpu"].get("name") or "unknown GPU"
    drv = side["driver"]
    drv_name = drv.get("name")
    drv_info = drv.get("info")
    if drv_name and drv_info:
        drv_text = f"{drv_name} ({drv_info})"
    else:
        drv_text = drv_name or drv_info or "unknown driver"

    s = side["summary"]
    counts = f"{s.get('passed', 0)} pass, {s.get('failed', 0)} fail, {s.get('skipped', 0)} skip ({s.get('total', 0)} total)"
    return f"{label:<10s} {rev_text} | {gpu_name} | {drv_text} | {counts}"


def format_comparison(result: dict) -> str:
    """Format a comparison result dict as concise human-readable text."""
    lines = [
        _format_header("Baseline:", result["baseline"]),
        _format_header("Candidate:", result["candidate"]),
    ]

    if result["baseline"].get("error"):
        lines.append(f"\nBASELINE ERROR: {result['baseline']['error']}")

    if result["candidate"].get("error"):
        lines.append(f"\nCANDIDATE ERROR: {result['candidate']['error']}")

    if result.get("incomplete_reasons"):
        lines.append("\nINCOMPLETE COMPARISON:")
        for reason in result["incomplete_reasons"]:
            lines.append(f"  - {reason}")

    if result["incompatibilities"]:
        lines.append("\nINCOMPATIBLE RUN SETTINGS / ENVIRONMENT:")
        for item in result["incompatibilities"]:
            lines.append(
                f"  - {item['scope']}.{item['field']}: {item['baseline']!r} -> {item['candidate']!r}"
            )

    if result.get("ambiguous"):
        lines.append("\nAMBIGUOUS COMPARISON (DUPLICATE TEST NAMES):")
        for warning in result["warnings"]:
            lines.append(f"  - {warning}")
    elif result["warnings"]:
        lines.append("\nWARNINGS:")
        for warning in result["warnings"]:
            lines.append(f"  - {warning}")

    if result.get("all_skipped"):
        lines.append("\nNOTE: all tests were skipped (no tests ran)")

    reg = result["regressions"]
    if any(reg.values()):
        lines.append("\nREGRESSIONS:")
        if reg["pass_to_fail"]:
            lines.append(f"  pass -> fail ({len(reg['pass_to_fail'])}):")
            for row in reg["pass_to_fail"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if reg["pass_to_skip"]:
            lines.append(f"  pass -> skip ({len(reg['pass_to_skip'])}):")
            for row in reg["pass_to_skip"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if reg["missing"]:
            lines.append(f"  missing from candidate ({len(reg['missing'])}):")
            for row in reg["missing"]:
                detail = _one_line(row["baseline_detail"])
                suffix = f": {detail}" if detail else ""
                lines.append(f"    {row['name']} (was {row['baseline_status']}{suffix})")
        if reg["skip_to_fail"]:
            lines.append(f"  skip -> fail ({len(reg['skip_to_fail'])}):")
            for row in reg["skip_to_fail"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if reg["added_fail"]:
            lines.append(f"  added fail ({len(reg['added_fail'])}):")
            for row in reg["added_fail"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")

    chg = result["changes"]
    if any(chg.values()):
        lines.append("\nOTHER CHANGES:")
        if chg["fail_to_pass"]:
            lines.append(f"  fail -> pass ({len(chg['fail_to_pass'])}):")
            for row in chg["fail_to_pass"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if chg["skip_to_pass"]:
            lines.append(f"  skip -> pass ({len(chg['skip_to_pass'])}):")
            for row in chg["skip_to_pass"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if chg["fail_to_skip"]:
            lines.append(f"  fail -> skip ({len(chg['fail_to_skip'])}):")
            for row in chg["fail_to_skip"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if chg["added_pass"]:
            lines.append(f"  added pass ({len(chg['added_pass'])}):")
            for row in chg["added_pass"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")
        if chg["added_skip"]:
            lines.append(f"  added skip ({len(chg['added_skip'])}):")
            for row in chg["added_skip"]:
                lines.append(f"    {row['name']}: {_one_line(row['candidate_detail'])}")

    c = result["counts"]
    unchanged_total = c["unchanged_pass"] + c["unchanged_fail"] + c["unchanged_skip"]
    improvements = c["fail_to_pass"] + c["skip_to_pass"]
    unchanged_note = f"{unchanged_total} unchanged"
    if c["unchanged_fail"]:
        unchanged_note += f" ({c['unchanged_fail']} pre-existing fail)"
    compat_note = "compatible" if result["compatible"] else f"INCOMPATIBLE ({len(result['incompatibilities'])} mismatches)"
    extra_notes = []
    if result.get("incomplete"):
        extra_notes.append("INCOMPLETE")
    if result.get("ambiguous"):
        extra_notes.append("AMBIGUOUS (duplicates)")
    if result.get("all_skipped"):
        extra_notes.append("all skipped (no tests ran)")
    status_note = " | ".join([compat_note, *extra_notes])
    lines.append(
        "\nSummary: "
        f"{c['pass_to_fail']} pass->fail, {c['pass_to_skip']} pass->skip, {c['missing']} missing, "
        f"{c['skip_to_fail']} skip->fail, {c['added_fail']} added-fail | "
        f"{improvements} improved, {c['fail_to_skip']} fail->skip | "
        f"{unchanged_note} | {status_note}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare baseline and candidate JSON reports from scripts/run_gpu_tests.py --json."
    )
    ap.add_argument("baseline", help="baseline GPU test report JSON")
    ap.add_argument("candidate", help="candidate (PR) GPU test report JSON")
    ap.add_argument("--json", dest="json_out", metavar="FILE", default=None,
                    help="also write the structured comparison as JSON")
    ap.add_argument("--allow-incompatible", action="store_true",
                    help="surface incompatible run settings/GPU/driver without failing the exit code for them alone")
    args = ap.parse_args(argv)

    if args.json_out is not None:
        if os.path.isdir(args.json_out):
            print(f"error: --json path is a directory: {args.json_out}", file=sys.stderr)
            return 2
        for input_path in (args.baseline, args.candidate):
            if same_report_path(args.json_out, input_path):
                print(f"error: --json output must not overwrite the source report: {input_path}", file=sys.stderr)
                return 2

    try:
        baseline = load_report(args.baseline)
        candidate = load_report(args.candidate)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = compare_reports(
        baseline,
        candidate,
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        allow_incompatible=args.allow_incompatible,
    )
    print(format_comparison(result))

    if args.json_out is not None:
        try:
            parent = os.path.dirname(os.path.abspath(args.json_out))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
        except OSError as exc:
            print(f"error: cannot write comparison report {args.json_out}: {exc.strerror or exc}", file=sys.stderr)
            return 2

    return result["exit_code"]


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
