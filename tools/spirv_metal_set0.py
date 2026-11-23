#!/usr/bin/env python3
"""Give set 0's descriptor arrays fixed, non-overlapping Metal argument-buffer IDs (SHADER-161).

SPIRV-Cross (1.4.363.0) puts every runtime descriptor array of an argument buffer at one ID: with Leonida's
set 0 (sampled images at binding 0, samplers at 1, more image arrays at 3+) it comments all but one out as
"Overlapping binding", so the shader can't reach them. This rewrites a translated module for a native Metal
path only (the Vulkan module and layout stay as they are):

- each set-0 runtime descriptor array becomes a sized array with that binding's capacity from
  runtime/src/vulkan_bindless.h (`bindings`), and
- each set-0 Binding becomes the first ID of its range: the capacities of all lower bindings added up.

Convert the result with `spirv-cross --msl --msl-argument-buffers --msl-argument-buffer-tier 1
--msl-decoration-binding`, so the IDs are the same in every shader, whichever bindings it uses. The argument
buffer then holds binding b's element i at ID `metal_id(b) + i` (`--print-layout` prints the table).

Usage: spirv_metal_set0.py in.spv out.spv     |     spirv_metal_set0.py --print-layout
Needs spirv-dis and spirv-as (SPIRV-Tools) on PATH, or SPIRV_DIS / SPIRV_AS.
"""
from collections import deque
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDLESS_H = os.path.join(ROOT, "runtime", "src", "vulkan_bindless.h")


def capacities(path=BINDLESS_H):
    """Descriptor counts by binding number, from the `bindings` table of vulkan_bindless.h."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"std::array<Binding,\s*(\d+)>\s*bindings\s*=\s*\{\{(.*?)\}\};", src, re.S)
    if not m:
        raise SystemExit(f"{path}: no `bindings` table")
    counts = [int(c) for c in re.findall(r"\{VK_DESCRIPTOR_TYPE_\w+,\s*(\d+)\}", m.group(2))]
    if len(counts) != int(m.group(1)):
        raise SystemExit(f"{path}: `bindings` declares {m.group(1)} entries, found {len(counts)}")
    return counts


def metal_ids(counts):
    """First argument-buffer ID of each binding: the capacities of the lower bindings added up."""
    ids, total = [], 0
    for c in counts:
        ids.append(total)
        total += c
    return ids


def _fresh_id(base, used):
    """Return `base` if unused, otherwise `base_1`, `base_2`, ... and record it in `used`."""
    cand = base
    i = 1
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


_ID_RE = re.compile(r'"(?:\\.|[^"\\])*"|(%[^\s"]+)')


def _replace_ids(line, id_map):
    """Replace %id tokens outside quoted string literals using `id_map`."""
    return _ID_RE.sub(lambda m: m.group(0) if m.group(1) is None else id_map.get(m.group(1), m.group(1)), line)


def _specialize_helper_calls(out, retype, used_ids):
    """Clone helper functions for each distinct retyped set-0 argument signature and rewrite OpFunctionCall sites."""
    fnty_by_sig = {}
    for ln in out:
        m = re.match(r"\s*(%\S+) = OpTypeFunction (%\S+)(.*)$", ln)
        if m:
            fnty_by_sig[(m.group(2), tuple(m.group(3).split()))] = m.group(1)

    orig_funcs = {}
    i = 0
    while i < len(out):
        m_fn = re.match(r"(\s*)(%\S+) = OpFunction (%\S+) (\S+) (%\S+)$", out[i])
        if not m_fn:
            i += 1
            continue
        start = i
        params = []
        i += 1
        while i < len(out):
            m_p = re.match(r"\s*(%\S+) = OpFunctionParameter (%\S+)$", out[i])
            if m_p:
                params.append((m_p.group(1), m_p.group(2)))
            if out[i].strip() == "OpFunctionEnd":
                break
            i += 1
        end = i + 1
        orig_funcs[m_fn.group(2)] = {
            "indent": m_fn.group(1),
            "ret": m_fn.group(3),
            "ctrl": m_fn.group(4),
            "ft": m_fn.group(5),
            "params": params,
            "lines": out[start:end],
            "start": start,
            "end": end,
        }
        i = end

    if not orig_funcs:
        return out

    extra_fntys = []
    clones = {}
    cloned_ids = {}
    cloned_blocks = []
    updated_orig = {}
    worklist = deque()
    for fn_id, fn in orig_funcs.items():
        fn_lines = list(fn["lines"])
        updated_orig[fn_id] = fn_lines
        worklist.append((fn_lines, dict(retype)))

    while worklist:
        fn_lines, fn_retype = worklist.popleft()
        changed = True
        while changed:
            changed = False
            for idx, ln in enumerate(fn_lines):
                m_cp = re.match(r"(\s*(%\S+) = OpCopyObject )%\S+ (%\S+)$", ln)
                if m_cp and m_cp.group(3) in fn_retype:
                    ptr = fn_retype[m_cp.group(3)]
                    if fn_retype.get(m_cp.group(2)) != ptr:
                        fn_retype[m_cp.group(2)] = ptr
                        fn_lines[idx] = f"{m_cp.group(1)}{ptr} {m_cp.group(3)}"
                        changed = True

        for idx, ln in enumerate(fn_lines):
            m_call = re.match(r"(\s*%\S+ = OpFunctionCall %\S+ )(%\S+)((?:\s+%\S+)*)$", ln)
            if not m_call:
                continue
            callee = m_call.group(2)
            args = m_call.group(3).split()
            fn = orig_funcs.get(callee)
            if not fn or len(args) != len(fn["params"]):
                continue
            orig_param_tys = tuple(p_ty for _, p_ty in fn["params"])
            call_sig = tuple(fn_retype.get(arg, orig_param_tys[k]) for k, arg in enumerate(args))
            if call_sig == orig_param_tys:
                continue
            key = (callee, call_sig)
            if key not in clones:
                clone_fn = _fresh_id(f"{callee}_msl", used_ids)
                clones[key] = clone_fn
                ft_key = (fn["ret"], call_sig)
                ft_new = fnty_by_sig.get(ft_key)
                if ft_new is None:
                    ft_new = _fresh_id(f"{fn['ft']}_msl", used_ids)
                    fnty_by_sig[ft_key] = ft_new
                    extra_fntys.append(f"{ft_new} = OpTypeFunction {fn['ret']} {' '.join(call_sig)}")
                id_map = {callee: clone_fn}
                for cln in fn["lines"][1:]:
                    m_def = re.match(r"\s*(%\S+) = ", cln)
                    if m_def:
                        id_map[m_def.group(1)] = _fresh_id(f"{m_def.group(1)}_msl", used_ids)
                for orig_id, new_id in id_map.items():
                    cloned_ids.setdefault(orig_id, []).append(new_id)
                clone_lines = [f"{fn['indent']}{clone_fn} = OpFunction {fn['ret']} {fn['ctrl']} {ft_new}"]
                p_idx = 0
                for cln in fn["lines"][1:]:
                    m_p = re.match(r"(\s*)(%\S+) = OpFunctionParameter %\S+$", cln)
                    if m_p:
                        clone_lines.append(f"{m_p.group(1)}{id_map[m_p.group(2)]} = OpFunctionParameter {call_sig[p_idx]}")
                        p_idx += 1
                    else:
                        clone_lines.append(_replace_ids(cln, id_map))
                clone_retype = dict(retype)
                for p_id, orig_pty in fn["params"]:
                    idx_p = fn["params"].index((p_id, orig_pty))
                    if call_sig[idx_p] != orig_pty:
                        clone_retype[id_map[p_id]] = call_sig[idx_p]
                cloned_blocks.append(clone_lines)
                worklist.append((clone_lines, clone_retype))
            fn_lines[idx] = f"{m_call.group(1)}{clones[key]}{m_call.group(3)}"

    result = []
    fn_by_start = {fn["start"]: (fn_id, fn["end"]) for fn_id, fn in orig_funcs.items()}
    idx = 0
    inserted_fntys = False
    while idx < len(out):
        if idx in fn_by_start:
            if not inserted_fntys and extra_fntys:
                result.extend(extra_fntys)
                inserted_fntys = True
            fn_id, end = fn_by_start[idx]
            result.extend(updated_orig[fn_id])
            idx = end
        else:
            ln = out[idx]
            m_dec = re.match(r"^(\s*OpDecorate )(%\S+)(\s+.*)$", ln)
            m_grp = re.match(r"^\s*OpGroupDecorate %\S+(.*)$", ln)
            if m_dec and m_dec.group(2) in cloned_ids:
                result.append(ln)
                for cid in cloned_ids[m_dec.group(2)]:
                    result.append(f"{m_dec.group(1)}{cid}{m_dec.group(3)}")
            elif m_grp:
                extra = [cid for t in m_grp.group(1).split() for cid in cloned_ids.get(t, ())]
                result.append(f"{ln} {' '.join(extra)}" if extra else ln)
            else:
                result.append(ln)
            idx += 1
    for blk in cloned_blocks:
        result.extend(blk)
    return result


def rewrite(text, counts):
    """Rewrite the disassembly (spirv-dis --raw-id is not needed; names are kept) of one module."""
    ids = metal_ids(counts)
    lines = text.splitlines()
    sets, binds = {}, {}
    for ln in lines:
        m = re.match(r"\s*OpDecorate (%\S+) DescriptorSet (\d+)$", ln)
        if m:
            sets[m.group(1)] = int(m.group(2))
        m = re.match(r"\s*OpDecorate (%\S+) Binding (\d+)$", ln)
        if m:
            binds[m.group(1)] = int(m.group(2))
    set0 = {v: b for v, b in binds.items() if sets.get(v) == 0}
    for v, b in set0.items():
        if b >= len(counts):
            raise SystemExit(f"{v}: binding {b} isn't in vulkan_bindless.h")

    defs = {}       # result id -> instruction text
    definition_positions: dict[str, int] = {}
    for idx, ln in enumerate(lines):
        m = re.match(r"\s*(%\S+) = (.*)$", ln)
        if m:
            defs[m.group(1)] = m.group(2)
            definition_positions[m.group(1)] = idx
    uint = next((r for r, d in defs.items() if d == "OpTypeInt 32 0"), None)

    candidates = []
    for order, (v, b) in enumerate(set0.items()):
        var = re.match(r"OpVariable (%\S+) UniformConstant$", defs.get(v, ""))
        if not var:
            continue    # the hash table (a storage buffer) and other non-array resources keep their type
        old_ptr = var.group(1)
        ptr = re.match(r"OpTypePointer UniformConstant (%\S+)$", defs.get(old_ptr, ""))
        rt = re.match(r"OpTypeRuntimeArray (%\S+)$", defs.get(ptr.group(1), "")) if ptr else None
        if not rt:
            continue
        candidates.append((b, definition_positions[old_ptr], order, v, old_ptr, rt.group(1)))

    if candidates:
        if uint is None:
            raise SystemExit("module has no 32-bit uint type for the array lengths")
        first_pointer = min(candidate[1] for candidate in candidates)
        uint_position = definition_positions[uint]
        if uint_position > first_pointer:
            # The generated length may precede an original declaration of its integer type.
            lines.insert(first_pointer, lines.pop(uint_position))

    used_ids = set(defs) | set(sets) | set(binds)
    cap_by_bind = {}    # binding -> OpConstant id (emitted once per binding)
    ptr_by_key = {}     # (binding, elem_type) -> new OpTypePointer id
    new_types = []      # inserted right after the variable's pointer type is declared
    retype = {}         # resized variable or copied pointer ID -> new pointer type
    for b, _, _, v, old_ptr, elem in sorted(candidates):
        if (b, elem) in ptr_by_key:
            retype[v] = ptr_by_key[(b, elem)]
            continue
        decl = []
        cap = cap_by_bind.get(b)
        if cap is None:
            cap = _fresh_id(f"%msl_cap_b{b}", used_ids)
            cap_by_bind[b] = cap
            decl.append(f"{cap} = OpConstant {uint} {counts[b]}")
        arr = _fresh_id(f"%msl_arr_b{b}", used_ids)
        nptr = _fresh_id(f"%msl_ptr_b{b}", used_ids)
        decl.extend([
            f"{arr} = OpTypeArray {elem} {cap}",
            f"{nptr} = OpTypePointer UniformConstant {arr}",
        ])
        new_types.append((old_ptr, decl))
        ptr_by_key[(b, elem)] = nptr
        retype[v] = nptr

    out = []
    for ln in lines:
        m = re.match(r"(\s*OpDecorate (%\S+) Binding )(\d+)$", ln)
        if m and m.group(2) in set0:
            ln = f"{m.group(1)}{ids[int(m.group(3))]}"
        m = re.match(r"(\s*(%\S+) = OpVariable )(%\S+)( UniformConstant)$", ln)
        if m and m.group(2) in retype:
            ln = f"{m.group(1)}{retype[m.group(2)]}{m.group(4)}"
        # Retype copies by source ID so arrays sharing a type in other sets are unaffected.
        m = re.match(r"(\s*(%\S+) = OpCopyObject )%\S+ (%\S+)$", ln)
        if m and m.group(3) in retype:
            ptr = retype[m.group(3)]
            retype[m.group(2)] = ptr
            ln = f"{m.group(1)}{ptr} {m.group(3)}"
        out.append(ln)
        m = re.match(r"\s*(%\S+) = OpTypePointer", ln)
        if m:
            for after, decl in new_types:
                if after == m.group(1):
                    out.extend(decl)
    if retype:
        out = _specialize_helper_calls(out, retype, used_ids)
    return "\n".join(out) + "\n"


def main(argv):
    rest = argv[1:]
    if rest and rest[0] in ("-h", "--help"):
        for line in __doc__.strip().splitlines()[-2:]:
            print(line)
        return 0
    counts = capacities()
    if argv[1:] == ["--print-layout"]:
        for b, (c, i) in enumerate(zip(counts, metal_ids(counts))):
            print(f"binding {b:2}: {c:5} descriptors at IDs {i}..{i + c - 1}")
        return 0
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    dis, asm = _tools()
    # errors="replace": on Windows text=True decodes with the ANSI code page, and a program that answers in UTF-8
    # (a path with non-ASCII letters in its error message) would otherwise kill the reader thread and lose stderr.
    text = _run([dis, argv[1]], capture_output=True, text=True, errors="replace").stdout
    env = re.search(r"; Version: (\d+)\.(\d+)", text)
    target = f"spv{env.group(1)}.{env.group(2)}" if env else "vulkan1.2"
    _run([asm, "--target-env", target, "-", "-o", argv[2]], input=rewrite(text, counts), text=True)
    return 0


def _tools():
    return os.environ.get("SPIRV_DIS", "spirv-dis"), os.environ.get("SPIRV_AS", "spirv-as")


def _run(cmd, **kwargs):
    """subprocess.run(check=True). Windows does not name the program it could not start, so name it here."""
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except OSError as e:
        if e.filename is None:
            e.filename = cmd[0]
        raise


def cli(argv):
    """main() for the command line: a SPIRV-Tools program that cannot be started, or that rejects its input, is
    one ERROR line and exit 1 instead of a traceback. main() itself still raises, for callers that catch it."""
    try:
        return main(argv)
    except subprocess.CalledProcessError as e:
        program = e.cmd[0] if isinstance(e.cmd, (list, tuple)) else e.cmd
        said = next((line.strip() for line in (e.stderr or "").splitlines() if line.strip()), "")
        print(f"ERROR: {program} failed with exit status {e.returncode}" + (f": {said}" if said else ""),
              file=sys.stderr)
    except OSError as e:
        hint = " (SPIRV-Tools must be on PATH, or set SPIRV_DIS / SPIRV_AS)" if e.filename in _tools() else ""
        print(f"ERROR: {e.filename or 'spirv_metal_set0.py'}: {e.strerror or e}{hint}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(cli(sys.argv))
