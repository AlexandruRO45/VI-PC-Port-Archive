"""Arrange a control-flow graph as nested loops and conditionals driven by a block-number variable.

    structure(succ) -> tree, or None when the graph is irreducible

`succ[b]` lists the blocks that block b can go to next (none = the program ends there); block 0 is the entry.
Every block ends by storing its successor's number in a variable `pc` (END when it has none). The tree says where
each block's code goes:

    ("block", b)             run block b
    ("guard", b, [items])    if pc == b: run the items (they start with block b, or with the loop headed by b)
    ("loop", b, [items])     repeat the items, which start with block b, for as long as pc == b afterwards

Why this instead of one loop around a switch: a shader compiler has to rediscover the real control flow from a
switch on a variable that every block writes, which takes it minutes on shaders with a thousand blocks. Here the
loops are loops, and the guards are tests against a value that was just assigned, which fold at once.

How it is built (reducible graphs only, i.e. every loop has a single entry):
  - Each natural loop becomes a "loop" item. Leaving it needs no break: pc simply no longer names the header.
  - Inside a loop (and at the top level) the members are nested by dominance: after a member come the members it
    immediately dominates, each behind a guard, in an order where every forward edge points later. A jump to a
    member that isn't below the current one falls out of the guards it is in, to the level where a later guard
    matches. A jump out of the loop falls out of all of them.
A block is never skipped or run twice in one pass, so any path through the graph is followed exactly.
"""
import collections

END = -1


def _order_rpo(succ):
    """Reverse postorder from block 0, its RPO index table, retreating edges, and RPO-indexed predecessors."""
    n = len(succ)
    state = [0] * n
    state[0] = 1
    order, back = [], []
    stack = [(0, 0, succ[0])]
    while stack:
        v, idx, sv = stack[-1]
        n_sv = len(sv)
        while idx < n_sv:
            w = sv[idx]
            idx += 1
            st = state[w]
            if st == 0:
                state[w] = 1
                stack[-1] = (v, idx, sv)
                stack.append((w, 0, succ[w]))
                break
            if st == 1:
                back.append((v, w))
        else:
            state[v] = 2
            order.append(v)
            stack.pop()
    rpo = order[::-1]
    m = len(rpo)
    pos = [-1] * n
    for i, v in enumerate(rpo):
        pos[v] = i
    preds_i = [[] for _ in range(m)]
    for vi, v in enumerate(rpo):
        for w in succ[v]:
            preds_i[pos[w]].append(vi)
    return rpo, pos, back, preds_i


def _order(succ):
    """Reverse postorder from block 0, the retreating edges of that walk, and the predecessors of each block."""
    rpo, _, back, _ = _order_rpo(succ)
    preds = collections.defaultdict(list)
    for v in rpo:
        for w in succ[v]:
            preds[w].append(v)
    return rpo, back, preds


def _dominators_rpo(m, preds_i, has_back=True):
    """Immediate dominator of each reachable block in RPO index space (0..m-1, Cooper, Harvey and Kennedy)."""
    idom_i = [-1] * m
    idom_i[0] = 0
    if not has_back:
        for vi in range(1, m):
            new = -1
            for pi in preds_i[vi]:
                if new < 0:
                    new = pi
                    if new == 0:
                        break
                    continue
                a, b = pi, new
                while a != b:
                    while a > b:
                        a = idom_i[a]
                    while b > a:
                        b = idom_i[b]
                new = a
                if new == 0:
                    break
            idom_i[vi] = new
        return idom_i

    need_repeat = False
    for vi in range(1, m):
        new = -1
        for pi in preds_i[vi]:
            if idom_i[pi] < 0:
                need_repeat = True
                continue
            if new < 0:
                new = pi
                continue
            if new == 0:
                continue
            a, b = pi, new
            while a != b:
                while a > b:
                    a = idom_i[a]
                while b > a:
                    b = idom_i[b]
            new = a
        idom_i[vi] = new
    while need_repeat:
        need_repeat = False
        for vi in range(1, m):
            new = -1
            for pi in preds_i[vi]:
                if idom_i[pi] < 0:
                    continue
                if new < 0:
                    new = pi
                    if new == 0:
                        break
                    continue
                a, b = pi, new
                while a != b:
                    while a > b:
                        a = idom_i[a]
                    while b > a:
                        b = idom_i[b]
                new = a
                if new == 0:
                    break
            if idom_i[vi] != new:
                idom_i[vi] = new
                need_repeat = True
    return idom_i


def _dominators(rpo, preds):
    """Immediate dominator of each reachable block (Cooper, Harvey and Kennedy)."""
    pos = {v: i for i, v in enumerate(rpo)}
    preds_i = [[pos[p] for p in preds[v] if p in pos] for v in rpo]
    idom_i = _dominators_rpo(len(rpo), preds_i)
    return {rpo[vi]: rpo[idom_i[vi]] for vi in range(len(rpo))}, pos


def structure(succ):
    n = len(succ)
    rpo, pos, back, preds_i = _order_rpo(succ)
    m = len(rpo)
    idom_i = _dominators_rpo(m, preds_i, bool(back))

    # Dominance among the members of each region. A member's parent is the member holding the immediate
    # dominator of its first block; that dominator is always inside the same region.
    # Member encoding: integer node >= 0 is block node; node < 0 is loop headed by ~node (-1 - node).
    # Iterating xi in RPO order (1..m-1) appends children in strictly increasing RPO position.
    if back:
        for v, w in back:
            wi = pos[w]
            bi = pos[v]
            while bi > wi:
                bi = idom_i[bi]
            if bi != wi:
                return None                               # a loop with two entries: no nesting expresses it

        # Natural loops in RPO index space, and the innermost / enclosing loop for each block / header.
        bodies = {}
        for v, h in back:
            hi = pos[h]
            body = bodies.setdefault(hi, {hi})
            work = [pos[v]]
            while work:
                xi = work.pop()
                if xi not in body:
                    body.add(xi)
                    work.extend(preds_i[xi])
        inner = [-1] * m
        outer = {hi: -1 for hi in bodies}
        for hi in sorted(bodies, key=lambda k: len(bodies[k])):
            for xi in bodies[hi]:
                if inner[xi] < 0:
                    inner[xi] = hi
                elif xi != hi and xi in outer and outer[xi] < 0:
                    outer[xi] = hi

        children = [[] for _ in range(2 * n)]
        for xi in range(1, m):
            v = rpo[xi]
            reg = inner[xi]
            di = idom_i[xi]
            if reg == xi:                                 # a loop header: placed as its loop, in the outer region
                t_reg = outer[xi]
                child = ~v
            else:
                t_reg = reg
                child = v
            loop = inner[di]
            if loop == t_reg:
                parent = rpo[di]
            else:
                while outer[loop] != t_reg:
                    loop = outer[loop]
                parent = ~rpo[loop]
            children[parent].append(child)
        start = ~0 if 0 in bodies else 0
    else:
        children = [[] for _ in range(n)]
        for xi in range(1, m):
            children[rpo[idom_i[xi]]].append(rpo[xi])
        start = 0

    # Flatten dominated tail continuations out of enclosing guards and build iteratively so long sequential
    # diamond/loop chains stay O(1) deep instead of nesting O(N) OpSelectionMerge constructs (and never hit
    # Python's recursion limit on multi-thousand-block shaders).
    # Hoisting the last child c_last of a guarded member m into m's enclosing list is exact: m's head strictly
    # dominates c_last's head, so no block outside m's subtree can jump to c_last's head. When m's guard is skipped
    # (pc != m_head), pc cannot equal c_last_head either, so c_last's guard is skipped too; when m's guard runs,
    # c_last runs right after m's earlier children, in the same order as before.
    max_guard_depth = 16
    root = []
    # Stack frame: (out_list, parent_list, member, depth). out_list is None when opening a new guard.
    stack = [(root, None, start, 0)]
    while stack:
        out_list, parent_list, node, d = stack.pop()
        if out_list is None:
            gbody = []
            parent_list.append(("guard", node if node >= 0 else ~node, gbody))
            out_list = gbody
        below = children[node]
        if node < 0:
            x = ~node
            lbody = []
            out_list.append(("loop", x, lbody))
            if below:
                if parent_list is not None:
                    stack.append((None, parent_list, below[-1], d))
                    for c in reversed(below[:-1]):
                        stack.append((None, out_list, c, d + 1))
                else:
                    for c in reversed(below):
                        stack.append((None, out_list, c, d + 1))
            stack.append((lbody, None, x, 0))
        else:
            out_list.append(("block", node))
            if below:
                n_below = len(below)
                sx = succ[node]
                c0 = below[0]
                if n_below == 1 and len(sx) == 1 and sx[0] == (c0 if c0 >= 0 else ~c0):
                    stack.append((out_list, parent_list, c0, d))
                elif parent_list is not None and (n_below >= 2 or d >= max_guard_depth):
                    stack.append((None, parent_list, below[-1], d))
                    for c in reversed(below[:-1]):
                        stack.append((None, out_list, c, d + 1))
                else:
                    for c in reversed(below):
                        stack.append((None, out_list, c, d + 1))
    return root



def depth(tree):
    """Nesting depth of a tree from structure(): a block is depth 0 and each guard or loop adds one."""
    best = 0
    stack = [(tree, 0)]
    while stack:
        items, d = stack.pop()
        for it in items:
            if it[0] == "block":
                best = max(best, d)
            else:
                best = max(best, d + 1)
                stack.append((it[2], d + 1))
    return best


def _sccs(nodes, adj):
    """Iterative Tarjan's algorithm over `nodes` with adjacency `adj`; returns SCCs with >= 2 nodes."""
    index, low, on_stack, stack, out = {}, {}, set(), [], []
    nxt = 0
    for start in nodes:
        if start in index:
            continue
        index[start] = low[start] = nxt
        nxt += 1
        stack.append(start)
        on_stack.add(start)
        call = [(start, iter(adj[start]))]
        while call:
            v, it = call[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = nxt
                    nxt += 1
                    stack.append(w)
                    on_stack.add(w)
                    call.append((w, iter(adj[w])))
                    break
                if w in on_stack and index[w] < low[v]:
                    low[v] = index[w]
            else:
                if low[v] == index[v]:
                    comp = []
                    while True:
                        x = stack.pop()
                        on_stack.remove(x)
                        comp.append(x)
                        if x == v:
                            break
                    if len(comp) >= 2:
                        out.append(set(comp))
                call.pop()
                if call:
                    p = call[-1][0]
                    if low[v] < low[p]:
                        low[p] = low[v]
    return out


def split_irreducible(succ, max_factor=2, max_splits=64):
    """Make an irreducible CFG reducible by controlled node splitting (cloning the closed region R reachable from a secondary entry e without passing through primary entry h
    in an irreducible SCC, and redirecting e's external predecessors to clone[e]).

    Returns `(new_succ, orig_of)` where `orig_of[b]` is the original block index that block `b` executes
    and `new_succ[b]` preserves the exact positional successor order of `succ[orig_of[b]]`.
    Returns `(succ, list(range(len(succ))))` unchanged if `succ` is already reducible, or `None` if
    splitting would exceed `max_factor` * len(succ) blocks or `max_splits` duplications.
    """
    n0 = len(succ)
    if n0 == 0:
        return succ, []
    max_blocks = min(int(n0 * max_factor), n0 + max_splits)
    g = None
    orig = None
    while True:
        cur = succ if g is None else g
        rpo, back, preds = _order(cur)
        idom, pos = _dominators(rpo, preds)

        def dominates(a, b):
            while pos[b] > pos[a]:
                b = idom[b]
            return a == b

        if all(dominates(w, v) for v, w in back):
            if g is None:
                return succ, list(range(n0))
            return g, orig
        if g is None:
            g = [list(s) for s in succ]
            orig = list(range(n0))
        if len(g) >= max_blocks:
            return None

        # Strip dominating back-edges so nested irreducible cycles inside reducible loops surface as SCCs.
        adj = {v: [w for w in g[v] if not dominates(w, v)] for v in rpo}
        sccs = _sccs(rpo, adj)
        if not sccs:
            return None
        # Pick the smallest irreducible SCC (tie-break by earliest RPO position).
        scc = min(sccs, key=lambda c: (len(c), min(pos[x] for x in c)))
        entries = [x for x in scc if x == 0 or any(p not in scc for p in preds[x])]
        # Choose a pair (h, e) of SCC entries (e != h, e != 0) minimizing |R|, where R is the closed region of
        # SCC nodes reachable from e without passing through h. Cloning all of R at once keeps internal cycles
        # inside the clone instead of re-entering the original SCC and unrolling infinitely.
        best = None
        for h in entries:
            for e in entries:
                if e == h or e == 0:
                    continue
                region = {e}
                work = [e]
                while work:
                    u = work.pop()
                    for w in g[u]:
                        if w in scc and w != h and w not in region:
                            region.add(w)
                            work.append(w)
                key = (len(region), pos[h], -pos[e])
                if best is None or key < best[0]:
                    best = (key, h, e, region)
        if best is None:
            return None
        _, h, e, region = best
        if len(g) + len(region) > max_blocks:
            return None
        # Order cloned nodes deterministically by RPO position.
        reg_ordered = sorted(region, key=lambda x: pos[x])
        clone_of = {}
        for u in reg_ordered:
            clone_of[u] = len(g)
            orig.append(orig[u])
            g.append([])
        for u in reg_ordered:
            g[clone_of[u]] = [clone_of[w] if w in clone_of else w for w in g[u]]
        ext_preds = {p for p in preds[e] if p not in scc}
        ce = clone_of[e]
        for p in ext_preds:
            g[p] = [ce if w == e else w for w in g[p]]

def run(tree, choose, limit):
    """Interpret a tree: the blocks it executes, in order, stopping after `limit`. `choose(b)` gives the block
    that b goes to next (END to finish) and stands for whatever b stores in pc."""
    trace, pc = [], [0]
    stack = [("items", tree, 0)]
    while stack:
        frame = stack[-1]
        if frame[0] == "items":
            _, items, idx = frame
            if idx >= len(items):
                stack.pop()
                continue
            stack[-1] = ("items", items, idx + 1)
            it = items[idx]
            if it[0] == "block":
                if len(trace) >= limit:
                    break
                trace.append(it[1])
                pc[0] = choose(it[1])
            elif it[0] == "guard":
                if pc[0] == it[1]:
                    stack.append(("items", it[2], 0))
            else:
                stack.append(("loop", it[1], it[2]))
                stack.append(("items", it[2], 0))
        else:
            _, head, body = frame
            if pc[0] == head:
                stack.append(("items", body, 0))
            else:
                stack.pop()
    return trace
