"""Loop liveness check for the mesh-shader input registers (docs/design.md, "Mesh shaders" row).

`rdna2spirv.read_before_write_vgprs` is a linear scan: a VGPR counts as an input when its first read comes before
its first write in listing order. This tool redoes the count with liveness over the basic blocks and successor
edges `translate()` itself builds (it stops `translate()` where those are handed to `_shader_meta.collect`, so the
code is the code translate keeps: re-anchored, cut to reachable code, irreducible regions split). For each mesh
listing it computes the VGPRs live at entry with every edge, without the back edges (DFS retreating edges from the
entry block), and without the taken edge of `s_cbranch_execz`, and prints counts only:

    python -I tools/mesh_v5_liveness.py <work>/shaders/dxbc/disasm [-j 8]

Listings are found with os.listdir (names starting with `.` included); the stage comes from the carve index next
to the folder (`stage_of`). A write is any instruction with a VGPR destination, as in read_before_write_vgprs
(a write under a partial EXEC counts as a full write). Under that rule the taken edge of `s_cbranch_execz` lets a
later read see past writes that, with no lane active, write nothing; per lane the skip is the fall-through, so the
third count leaves those edges out. Standard library only; it reads the folder and writes nothing.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdna2spirv  # noqa: E402
from rdna2spirv import BRANCHES, VGPR_READ_ONLY_PREFIXES, base_op, read_before_write_vgprs, vgpr_dest_registers  # noqa: E402


def block_code(blk):
    """The instructions of a block that run: up to and including the first s_endpgm."""
    for k, ins in enumerate(blk):
        if base_op(ins["op"]) == "s_endpgm":
            return blk[:k + 1]
    return blk


def block_writes(blk):
    """VGPRs a block writes (read_before_write_vgprs' rule for a write)."""
    out = set()
    for ins in blk:
        a, op = ins["args"], ins["op"]
        if not a:
            continue
        reads_all = op.startswith(VGPR_READ_ONLY_PREFIXES) \
            or (op.startswith("global_atomic_") and "glc" not in ins.get("mods", []))
        if not reads_all and a[0]["k"] == "vgpr":
            out.update(vgpr_dest_registers(ins))
    return out


def back_edges(succ):
    """Edges (u, v) that a depth-first search from block 0 finds going back to a block still on its stack."""
    found, state = set(), {}                       # state: 1 = on the stack, 2 = done
    stack = [(0, iter(succ[0]))] if succ else []
    if succ:
        state[0] = 1
    while stack:
        u, it = stack[-1]
        v = next(it, None)
        if v is None:
            state[u] = 2
            stack.pop()
        elif state.get(v) == 1:
            found.add((u, v))
        elif v not in state:
            state[v] = 1
            stack.append((v, iter(succ[v])))
    return found


def branch_target_pc(ins):
    """The PC a branch goes to (translate()'s rule: an explicit target, else next PC + 4 * simm16)."""
    if "target" in ins:
        return ins["target"]
    imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
    v = (imm["v"] & 0xFFFF) if imm else 0
    return ins["pc"] + len(ins["enc"]) // 2 + 4 * (v - 0x10000 if v & 0x8000 else v)


def execz_skip_edges(blocks, succ):
    """Edges (u, v) where block u ends in s_cbranch_execz and v is its taken target (not also the fall-through)."""
    out = set()
    for u, blk in enumerate(blocks):
        last = next((i for i in blk if base_op(i["op"]) in BRANCHES or base_op(i["op"]) == "s_endpgm"), None)
        if last is None or base_op(last["op"]) != "s_cbranch_execz":
            continue
        tgt = branch_target_pc(last)
        if tgt == last["pc"] + len(last["enc"]) // 2:
            continue
        out.update((u, v) for v in succ[u] if blocks[v] and blocks[v][0]["pc"] == tgt)
    return out


def live_at_entry(blocks, succ, drop=frozenset()):
    """VGPRs live at the entry of block 0, by backward dataflow over succ without the edges in `drop`."""
    code = [block_code(b) for b in blocks]
    use = [read_before_write_vgprs(c) for c in code]
    defs = [block_writes(c) for c in code]
    edges = [[v for v in succ[u] if (u, v) not in drop] for u in range(len(blocks))]
    live_in = [set(u) for u in use]
    changed = True
    while changed:
        changed = False
        for u in reversed(range(len(blocks))):
            out = set().union(*(live_in[v] for v in edges[u])) if edges[u] else set()
            new = use[u] | (out - defs[u])
            if new != live_in[u]:
                live_in[u], changed = new, True
    return live_in[0] if blocks else set()


def analyse(blocks, succ):
    """(has a back edge, live at entry with all edges, without the back edges, without the execz skip edges)."""
    be = back_edges(succ)
    return (bool(be), live_at_entry(blocks, succ), live_at_entry(blocks, succ, frozenset(be)),
            live_at_entry(blocks, succ, frozenset(execz_skip_edges(blocks, succ))))


class _Captured(Exception):
    pass


class _CaptureMeta:
    @staticmethod
    def collect(path, name, blocks, succ, **kw):
        raise _Captured(blocks, succ)


def translate_cfg(path):
    """The blocks and successor lists translate() builds for a listing, or None if it stops before that."""
    saved = rdna2spirv._shader_meta
    rdna2spirv._shader_meta = _CaptureMeta
    try:
        rdna2spirv.translate(path)
    except _Captured as c:
        return c.args
    finally:
        rdna2spirv._shader_meta = saved
    return None


def one(path):
    """('skip',) for a listing that is not a mesh shader, else a result tuple of counts for main()."""
    name = os.path.basename(path)[:-2]
    if rdna2spirv.stage_of(name, path) != "ms":
        return ("skip",)
    try:
        cfg = translate_cfg(path)
    except Exception as e:                         # Unsupported or a crash: counted, not hidden
        return ("error", type(e).__name__)
    if cfg is None:
        return ("nocfg",)
    loop, every, no_back, no_execz = analyse(*cfg)
    return ("ok", loop, sorted(every), sorted(no_back), sorted(no_execz))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disasm", help="folder of .s listings (<work>/shaders/dxbc/disasm)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="processes (at most 8)")
    a = ap.parse_args(argv)
    paths = [os.path.join(a.disasm, n) for n in sorted(os.listdir(a.disasm)) if n.endswith(".s")]
    jobs = max(1, min(8, a.jobs))
    if jobs == 1:
        res = [one(p) for p in paths]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(one, paths, chunksize=64))
    ok = [r for r in res if r[0] == "ok"]
    errors = [r for r in res if r[0] == "error"]
    print(f"listings: {len(paths)}")
    print(f"mesh listings: {len(res) - sum(1 for r in res if r[0] == 'skip')}")
    print(f"  analysed: {len(ok)}, translate() refused or failed: {len(errors)}, no blocks reached: "
          f"{sum(1 for r in res if r[0] == 'nocfg')}")
    print(f"  with a back edge: {sum(1 for r in ok if r[1])}")
    print(f"  live-at-entry VGPRs differ without the back edges: {sum(1 for r in ok if r[2] != r[3])}")
    print(f"  live at entry exactly v5 (all edges): {sum(1 for r in ok if r[2] == [5])}")
    print(f"  live at entry exactly v5 (no back edges): {sum(1 for r in ok if r[3] == [5])}")
    print(f"  live at entry exactly v5 (no execz skip edges): {sum(1 for r in ok if r[4] == [5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
