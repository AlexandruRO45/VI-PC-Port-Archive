#!/usr/bin/env python3
"""shim_aliases: find imports that skip a shim because they name another DLL than the one the shim is registered under.

The shim registry keys each entry on the exact lower-cased `dll!name` (`runtime/shims/registry.cpp`, `find`), and the
loader sends an `api-ms-*` or system-DLL import that has no entry to the host. A function shimmed under
`api-ms-win-core-processthreads-l1-1-0.dll` is therefore not shimmed when an image imports it through
`api-ms-win-core-processthreads-l1-1-2.dll`, `kernel32.dll` or another contract: the host's own function runs instead,
with no message. Windows resolves an `api-ms-`/`ext-ms-` name without its last version segment: on Windows 11
`ApiSetQueryApiSetPresence` reports `...-l1-1-99` present wherever `...-l1-1-0` is, and `...-l1-2-0` as a contract of
its own. So `-l1-1-0` and `-l1-1-2` reach the same host function, while the registry treats them as unrelated.

This tool reads the `SHIM_TABLE(...)` rows from the runtime sources (literal `{"dll", "Name", ...}` rows and the
function-like macros the same file defines, such as `EXIT_PROCESS(dll)`), then reads PE images' import and
delay-import tables and reports every import whose function name is shimmed, but not under the DLL the image names.

Standard library only. Run with `python -I tools/shim_aliases.py <command> ...`.

Commands
  table [--names] [--json]        the parsed shim table: rows per DLL (with --names: every function and the DLLs it
                                  is registered under)
  contracts [--json]              every API-set contract the table registers, with the version names it lists and
                                  how many functions each holds. An import through any other version of the same
                                  contract reaches the host. No image needed.
  check <image>... [--json]       one `bypass` line per import that names a shimmed function under an unregistered
                                  DLL (`version alias` when only the contract's last segment differs), then a
                                  summary. Works on the executable and on the package's DLLs.
  --runtime DIR                   read the shim sources from DIR (default: this checkout's runtime/shims)

Exit status: 0 when no import bypasses a shim, 1 when one does, 2 when an input can't be read (one ERROR line).
The output names imported functions and DLLs only (the same names `gta6rt --dry-run` prints); no code or data of an
image is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe_inspect  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNTIME = os.path.join(ROOT, "runtime", "shims")
SOURCE_EXT = (".cpp", ".h", ".inc")

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)\(([^)]*)\)[ \t]*((?:[^\n]*\\\n)*[^\n]*)", re.M)
ROW = re.compile(r'\{\s*"([^"\\]+)"\s*,\s*"([^"\\]+)"')
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
MAX_EXPANSION = 16


class SourceError(ValueError):
    """The runtime sources can't be read as shim tables."""


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments, keeping string literals (a "//" inside a DLL path stays)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def split_args(text: str, start: int) -> tuple[list[str], int]:
    """The comma-separated arguments of the call whose '(' is at text[start - 1]; returns (args, index after ')')."""
    depth, i, args, cur = 0, start, [], []
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            cur.append(text[i:j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                args.append("".join(cur).strip())
                return args, i + 1
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    raise SourceError("unbalanced parentheses")


def macros_of(text: str) -> dict[str, tuple[list[str], str]]:
    out = {}
    for m in DEFINE.finditer(text):
        params = [p.strip() for p in m.group(2).split(",") if p.strip()]
        out[m.group(1)] = (params, m.group(3).replace("\\\n", " "))
    return out


def expand(text: str, macros: dict, depth: int = 0) -> str:
    """Expand the function-like macros of `macros` in `text` (argument substitution and #param stringizing)."""
    if depth > MAX_EXPANSION:
        raise SourceError("macro expansion deeper than %d levels" % MAX_EXPANSION)
    out, pos = [], 0
    for m in CALL.finditer(text):
        if m.start() < pos or m.group(1) not in macros:
            continue
        params, body = macros[m.group(1)]
        args, end = split_args(text, m.end())
        if len(args) != len(params) and not (params and params[-1] == "..."):
            continue
        sub = body
        for p, a in zip(params, args):
            sub = re.sub(r"#\s*%s\b" % re.escape(p), '"%s"' % a.strip('"'), sub)
            sub = re.sub(r"\b%s\b" % re.escape(p), lambda _m, a=a: a, sub)
        out.append(text[pos:m.start()])
        out.append(expand(sub, macros, depth + 1))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def table_rows(text: str, shared: dict | None = None) -> list[tuple[str, str]]:
    """(dll lower-cased, function) for every row of every SHIM_TABLE(...) in one source file."""
    text = strip_comments(text)
    macros = dict(shared or {})
    macros.update(macros_of(text))
    macros.pop("SHIM_TABLE", None)
    rows = []
    for m in re.finditer(r"\bSHIM_TABLE\s*\(", text):
        if text[:m.start()].rstrip().endswith("define"):
            continue
        args, _ = split_args(text, m.end())
        body = expand(",".join(args[1:]), macros)
        rows += [(dll.lower(), name) for dll, name in ROW.findall(body)]
    return rows


def read_table(runtime_dir: str) -> dict[str, set[str]]:
    """{function name: {dll, ...}} over every source file under runtime_dir."""
    if not os.path.isdir(runtime_dir):
        raise SourceError("%s is not a folder" % runtime_dir)
    texts = {}
    for base, dirs, files in os.walk(runtime_dir):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(SOURCE_EXT):
                with open(os.path.join(base, f), encoding="utf-8", errors="replace") as fh:
                    texts[os.path.join(base, f)] = fh.read()
    shared = {}
    for path, text in texts.items():  # macros a header defines for its includers
        if path.endswith(".h"):
            shared.update(macros_of(strip_comments(text)))
    table: dict[str, set[str]] = {}
    for text in texts.values():
        if "SHIM_TABLE" not in text:
            continue
        for dll, name in table_rows(text, shared):
            table.setdefault(name, set()).add(dll)
    if not table:
        raise SourceError("no SHIM_TABLE rows under %s" % runtime_dir)
    return table


def image_imports(path: str) -> tuple[list[tuple[str, str, str]], int]:
    """(dll, function, 'import' or 'delay') for every named import of one PE image, and the number of imports by
    ordinal, which this tool does not match (the table names functions)."""
    with open(path, "rb") as f:
        img = pe_inspect.PeImage(f.read())
    out, ordinals = [], 0
    for kind, index, delay in (("import", 1, False), ("delay", 13, True)):
        for entry in pe_inspect._imports(img, kind, index, delay):
            for fn in entry["functions"]:
                if fn and fn.startswith("#"):
                    ordinals += 1
                elif fn and fn != "?":
                    out.append((entry["dll"], fn, kind))
    return out, ordinals


def contract(dll: str) -> str | None:
    """The API-set contract a DLL name resolves as: the name without `.dll` and its last version segment, or None
    for a name that is not an API set (`api-ms-win-core-file-l1-2-0.dll` -> `api-ms-win-core-file-l1-2`)."""
    d = dll.lower()
    if d.endswith(".dll"):
        d = d[:-4]
    if not d.startswith(("api-ms-", "ext-ms-")):
        return None
    m = re.match(r"^(.*-l\d+-\d+)-\d+$", d)
    return m.group(1) if m else None


def contracts(table):
    """{contract: {version name: number of functions}} for every API-set contract the table registers."""
    out: dict[str, dict[str, int]] = {}
    for dlls in table.values():
        for d in dlls:
            c = contract(d)
            if c:
                versions = out.setdefault(c, {})
                versions[d] = versions.get(d, 0) + 1
    return out


def bypasses(imports, table):
    """[(dll, function, kind, sorted registered dlls, version alias?)] for imports that skip a shim, and the number of
    imports that reach one."""
    found, shimmed = [], 0
    for dll, fn, kind in imports:
        where = table.get(fn)
        if not where:
            continue
        if dll.lower() in where:
            shimmed += 1
        else:
            alias = contract(dll) is not None and any(contract(w) == contract(dll) for w in where)
            found.append((dll, fn, kind, sorted(where), alias))
    return found, shimmed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runtime", default=DEFAULT_RUNTIME, help="folder of shim sources (default: runtime/shims)")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table", help="print the parsed shim table")
    t.add_argument("--names", action="store_true", help="list every function and its DLLs")
    t.add_argument("--json", action="store_true", help="one JSON document")
    g = sub.add_parser("contracts", help="the API-set contracts and versions the table registers")
    g.add_argument("--json", action="store_true", help="one JSON document")
    c = sub.add_parser("check", help="report imports that bypass a shim")
    c.add_argument("images", nargs="+")
    c.add_argument("--json", action="store_true", help="one JSON document")
    args = ap.parse_args(argv)
    try:
        table = read_table(args.runtime)
        if args.command == "table":
            per_dll: dict[str, int] = {}
            for name, dlls in table.items():
                for d in dlls:
                    per_dll[d] = per_dll.get(d, 0) + 1
            if args.json:
                json.dump({"functions": {n: sorted(d) for n, d in sorted(table.items())}, "rows_per_dll": per_dll},
                          sys.stdout, indent=1)
                sys.stdout.write("\n")
                return 0
            for d in sorted(per_dll):
                print("%5d  %s" % (per_dll[d], d))
            if args.names:
                for n in sorted(table):
                    print("%s: %s" % (n, ", ".join(sorted(table[n]))))
            print("summary: %d rows, %d functions, %d DLL names" % (sum(per_dll.values()), len(table), len(per_dll)))
            return 0
        if args.command == "contracts":
            rows = contracts(table)
            if args.json:
                json.dump(rows, sys.stdout, indent=1, sort_keys=True)
                sys.stdout.write("\n")
            else:
                for c in sorted(rows):
                    print("%s: %s" % (c, ", ".join("-%s (%d)" % (v[len(c) + 1:-4], n) for v, n in sorted(rows[c].items()))))
                print("summary: %d contracts, %d version names" % (len(rows), sum(len(v) for v in rows.values())))
            return 0
        report, total = [], 0
        for path in args.images:
            try:
                imports, ordinals = image_imports(path)
            except pe_inspect.PeFormatError as e:
                raise SourceError("%s is not a PE image: %s" % (path, e)) from None
            found, shimmed = bypasses(imports, table)
            total += len(found)
            report.append({"image": os.path.basename(path), "imports": len(imports), "ordinal_imports": ordinals,
                           "shimmed": shimmed,
                           "bypass": [{"dll": d, "function": f, "kind": k, "registered": r, "version_alias": a}
                                      for d, f, k, r, a in found]})
        if args.json:
            json.dump(report, sys.stdout, indent=1)
            sys.stdout.write("\n")
        else:
            for r in report:
                for b in r["bypass"]:
                    print("bypass  %s  %s!%s%s  (shim under: %s%s)" % (
                        r["image"], b["dll"], b["function"], " [delay]" if b["kind"] == "delay" else "",
                        ", ".join(b["registered"]), "; version alias" if b["version_alias"] else ""))
                print("summary: %s: %d named imports (%d by ordinal, not matched), %d through a shim, %d bypass one"
                      % (r["image"], r["imports"], r["ordinal_imports"], r["shimmed"], len(r["bypass"])))
        return 1 if total else 0
    except (OSError, SourceError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
