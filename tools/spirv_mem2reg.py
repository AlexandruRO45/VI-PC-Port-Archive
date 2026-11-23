"""Pruned SSA promotion (Mem2Reg / per-slot Function-array scalarization) for tools/spirv_opt.py (#884).

Promotes non-escaping, statically-indexed, undecorated Function-scope OpVariables and
their static OpAccessChain slots (such as rdna2spirv's %vgpr, %sgpr, %active, %vcc,
%scc, %pc, %done, %m0) from load/store memory traffic into pruned SSA form with minimal
OpPhi nodes, collapses trivial OpPhi chains (strict-dominance checked), hoists common
side-effect-free expressions across sibling blocks to their dominator-tree LCA
(GVN-Hoist), and runs mark-and-sweep pure DCE (removing dead OpPhi cycles).

Soundness invariants (SPIR-V 1.5 / Vulkan 1.2):
  - Only Function-scope OpVariables declared in the function entry block (block 0) are
    considered, and only when block 0 has no predecessors.
  - Pointer chains (OpAccessChain, OpInBoundsAccessChain, OpCopyObject) are resolved to
    a fixed point regardless of block text order. Any pointer use outside plain
    non-Volatile OpLoad / OpStore (e.g. OpFunctionCall, OpAtomic*, OpCopyMemory,
    OpPtrAccessChain, OpBitcast, OpSelect, OpPhi, or storing a pointer as a value) marks
    the entire root variable as escaped.
  - Any dynamic OpAccessChain index marks the root variable as dynamically indexed and
    excludes it from promotion (other variables in the same function still promote).
  - Any mixed access depth (e.g. accessing both a whole array and an element) excludes
    the root variable.
  - Any variable or pointer touched in an unreachable block, or targeted by OpDecorate /
    OpDecorateId / OpDecorateString / OpGroupDecorate, is excluded conservatively.
  - Each promoted slot must have a single consistent non-pointer value type matching the
    pointee type of its pointer.
  - Backward bitset liveness prunes OpPhi placement strictly to join blocks where the
    slot is live-in, and rejects any slot that is live-in at function entry without a
    known constant initializer (so OpUndef is never synthesized).
  - Join blocks with unreachable predecessors, duplicate edges from a predecessor
    (e.g. OpBranchConditional %c %L %L or repeated OpSwitch targets), or more than
    max_phi_preds predecessors are excluded from OpPhi placement per slot: read-only /
    prologue slots and local temporaries still promote with 0 phis even inside huge
    dispatch-loop switches, while switch-carried mutable slots stay safely in memory.
  - CFG structure (OpLabel, OpSelectionMerge, OpLoopMerge, OpBranch*, OpSwitch,
    OpControlBarrier) is never modified.

Standard library only.
"""
import re

_M2R_ID_RE = re.compile(r"%m2r_\d+")


def _parse_inst_for_header(line: str):
    """Parse one header line for module-level metadata; preserves quoted strings safely."""
    stripped = line.lstrip()
    if not stripped or stripped.startswith(";"):
        return None
    if '"' in stripped:
        q = stripped.find('"')
        prefix = stripped[:q].split()
        if len(prefix) >= 2 and prefix[0] in ("OpDecorateString", "OpDecorateStringGOOGLE"):
            return None, prefix[0], prefix[1:]
        return None
    semi = stripped.find(";")
    toks = (stripped[:semi] if semi >= 0 else stripped).split()
    if not toks:
        return None
    if len(toks) >= 3 and toks[1] == "=":
        return toks[0], toks[2], toks[3:]
    return None, toks[0], toks[1:]


def collect_module_info(lines: list[str], const_ids: set[str], const_type: dict[str, str],
                        const_int_val: dict[str, tuple[str, int, int, int]],
                        const_null_elem_zero: dict[str, str],
                        spvasm: str = "") -> dict:
    """Collect pointer types, composite constant elements, all decorations, and colliding %m2r_N IDs."""
    ptr_pointee_type: dict[str, str] = {}
    ptr_types: set[str] = set()
    const_composite_elems: dict[str, tuple[str, list[str]]] = {}
    decorated_ids: set[str] = set()

    # Only scan the module header (before the first OpFunction); function bodies have no
    # decorations, type declarations, or global constants in valid SPIR-V.
    for raw in lines:
        if "OpFunction" in raw and "=" in raw:
            p = _parse_inst_for_header(raw)
            if p is not None and p[1] == "OpFunction":
                break
        parsed = _parse_inst_for_header(raw)
        if parsed is None:
            continue
        res, op, args = parsed
        if op in ("OpDecorate", "OpDecorateId", "OpDecorateString", "OpDecorateStringGOOGLE") and args:
            decorated_ids.add(args[0])
        elif op == "OpGroupDecorate" and len(args) >= 2:
            for t in args[1:]:
                decorated_ids.add(t)
        elif op == "OpDecorationGroup" and res:
            decorated_ids.add(res)
        elif op == "OpTypePointer" and res and len(args) >= 2:
            ptr_types.add(res)
            ptr_pointee_type[res] = args[1]
        elif op == "OpConstantComposite" and res and len(args) >= 2:
            const_composite_elems[res] = (args[0], args[1:])

    # Find any pre-existing %m2r_N IDs in the entire module in one C-speed regex pass.
    used_ids: set[str] = set(_M2R_ID_RE.findall(spvasm)) if spvasm else set()
    if not spvasm:
        for raw in lines:
            if "%m2r_" in raw:
                used_ids.update(_M2R_ID_RE.findall(raw))

    return {
        "const_ids": const_ids,
        "const_type": const_type,
        "const_int_val": const_int_val,
        "const_null_elem_zero": const_null_elem_zero,
        "const_composite_elems": const_composite_elems,
        "ptr_pointee_type": ptr_pointee_type,
        "ptr_types": ptr_types,
        "decorated_ids": decorated_ids,
        "used_ids": used_ids,
    }


def _dom_euler(dom_children: list[list[int]], n_blocks: int) -> tuple[list[int], list[int]]:
    """Compute iterative Euler-tour (tin, tout) timestamps on the dominator tree (O(N)).

    For reachable blocks: `a` dominates `b` iff `tin[a] <= tin[b] <= tout[a]`.
    """
    tin = [-1] * n_blocks
    tout = [-1] * n_blocks
    if n_blocks == 0:
        return tin, tout
    timer = 0
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        b, state = stack.pop()
        if state == 0:
            tin[b] = timer
            timer += 1
            stack.append((b, 1))
            for ch in reversed(dom_children[b]):
                stack.append((ch, 0))
        else:
            tout[b] = timer - 1
    return tin, tout


def _dom_depths(rpo: list[int], idom: dict[int, int]) -> dict[int, int]:
    depth: dict[int, int] = {0: 0}
    for b in rpo[1:]:
        p = idom.get(b)
        if p is not None and p in depth:
            depth[b] = depth[p] + 1
    return depth


def _dominates(a: int, b: int, idom: dict[int, int], depth: dict[int, int]) -> bool:
    if a == b:
        return True
    da, db = depth.get(a), depth.get(b)
    if da is None or db is None or da >= db:
        return False
    cur = b
    while depth[cur] > da:
        cur = idom[cur]
    return cur == a


def _terminator_targets(insts: list[list], b_insts: list[int], label_to_block: dict[str, int]) -> list[int]:
    """Return raw CFG target block indices from a block's terminator (preserving duplicates)."""
    out: list[int] = []
    for idx in reversed(b_insts):
        _, op, args, _ = insts[idx]
        if op is None:
            continue
        if op == "OpBranch" and args:
            tb = label_to_block.get(args[0])
            if tb is not None:
                out.append(tb)
        elif op == "OpBranchConditional" and len(args) >= 3:
            for t in (args[1], args[2]):
                tb = label_to_block.get(t)
                if tb is not None:
                    out.append(tb)
        elif op == "OpSwitch" and len(args) >= 2:
            for t in [args[1]] + [args[k] for k in range(3, len(args), 2)]:
                tb = label_to_block.get(t)
                if tb is not None:
                    out.append(tb)
        break
    return out


def promote_function(insts: list[list], build_cfg_and_dom, format_inst, module_info: dict,
                     deleted_ids: set[str], max_phi_preds: int = 32) -> tuple[list[list], bool]:
    """Run pruned SSA promotion (Mem2Reg) on one parsed function body.

    Returns (new_insts, changed). Never alters CFG edges or structured merge annotations.
    """
    blocks, succs, all_preds, rpo, idom, dom_children = build_cfg_and_dom(insts)
    n_blocks = len(blocks)
    if n_blocks == 0 or all_preds[0]:
        return insts, False

    block_label = [insts[b_insts[0]][0] for b_insts in blocks]
    label_to_block = {lbl: b for b, lbl in enumerate(block_label) if lbl is not None}
    reachable_set = set(rpo)
    postorder = list(reversed(rpo))

    inst_block: list[int] = [-1] * len(insts)
    for b, b_insts in enumerate(blocks):
        for idx in b_insts:
            inst_block[idx] = b

    # Compute deterministic reachable predecessors and flag blocks unsafe for new OpPhi insertion.
    reach_preds: list[list[int]] = [[] for _ in range(n_blocks)]
    unsafe_phi_blocks: set[int] = {0}
    for b in rpo:
        raw_tgts = _terminator_targets(insts, blocks[b], label_to_block)
        seen_t: set[int] = set()
        for tb in raw_tgts:
            if tb in seen_t:
                unsafe_phi_blocks.add(tb)
            else:
                seen_t.add(tb)
                reach_preds[tb].append(b)
    for b in rpo:
        if any(p not in reachable_set for p in all_preds[b]):
            unsafe_phi_blocks.add(b)

    const_ids: set[str] = module_info["const_ids"]
    const_type: dict[str, str] = module_info["const_type"]
    const_int_val: dict[str, tuple[str, int, int, int]] = module_info["const_int_val"]
    const_null_elem_zero: dict[str, str] = module_info["const_null_elem_zero"]
    const_composite_elems: dict[str, tuple[str, list[str]]] = module_info["const_composite_elems"]
    ptr_pointee_type: dict[str, str] = module_info["ptr_pointee_type"]
    ptr_types: set[str] = module_info["ptr_types"]
    decorated_ids: set[str] = module_info["decorated_ids"]
    used_ids: set[str] = module_info["used_ids"]

    # 1. Discover Function-scope variables in entry block and SSA result types.
    func_vars: dict[str, tuple[str, str | None]] = {}
    ptr_root: dict[str, str] = {}
    ptr_slot: dict[str, tuple | None] = {}
    id_type: dict[str, str] = dict(const_type)
    escaped: set[str] = set()
    dyn_indexed: set[str] = set()
    ptr_derivations: list[tuple[str, str, list[str]]] = []

    for idx, (res, op, args, _) in enumerate(insts):
        if op is None:
            continue
        if res is not None and args and op != "OpLabel":
            id_type[res] = args[0]
        if op == "OpVariable" and len(args) >= 2 and args[1] == "Function" and res:
            if inst_block[idx] == 0:
                init = args[2] if len(args) >= 3 else None
                func_vars[res] = (args[0], init)
                ptr_root[res] = res
                ptr_slot[res] = (res,)
                if res in decorated_ids:
                    escaped.add(res)
        elif op in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain",
                    "OpInBoundsPtrAccessChain", "OpCopyObject") and res and len(args) >= 2:
            ptr_derivations.append((res, op, args))

    if not func_vars:
        return insts, False

    # Resolve nested pointer derivations to a fixed point regardless of block text order.
    fp_changed = True
    while fp_changed:
        fp_changed = False
        for res, op, args in ptr_derivations:
            if res in ptr_root:
                continue
            base = args[1]
            if base not in ptr_root:
                continue
            root = ptr_root[base]
            ptr_root[res] = root
            fp_changed = True
            if op in ("OpPtrAccessChain", "OpInBoundsPtrAccessChain"):
                ptr_slot[res] = None
                escaped.add(root)
            elif op == "OpCopyObject":
                ptr_slot[res] = ptr_slot.get(base)
                if res in decorated_ids:
                    escaped.add(root)
            else:
                idxs = args[2:]
                base_slot = ptr_slot.get(base)
                if base_slot is not None and all(i in const_int_val for i in idxs):
                    cidxs = tuple(const_int_val[i][1] for i in idxs)
                    ptr_slot[res] = base_slot + cidxs
                else:
                    ptr_slot[res] = None
                    dyn_indexed.add(root)
                if res in decorated_ids:
                    escaped.add(root)

    # 2. Scan all instructions for escapes, unreachable accesses, mixed access depths, and slot types.
    slot_depths: dict[str, set[int]] = {}
    slot_types: dict[tuple, set[str]] = {}
    bad_slots: set[tuple] = set()
    unreachable_roots: set[str] = set()

    for idx, (res, op, args, _) in enumerate(insts):
        if op is None:
            continue
        b = inst_block[idx]
        reachable = b in reachable_set
        if op == "OpLoad":
            if len(args) >= 2 and args[1] in ptr_root:
                ptr = args[1]
                root = ptr_root[ptr]
                if not reachable:
                    unreachable_roots.add(root)
                if any("Volatile" in m for m in args[2:]) or (res and res in decorated_ids) or ptr in decorated_ids:
                    escaped.add(root)
                sl = ptr_slot.get(ptr)
                if sl is not None:
                    slot_depths.setdefault(root, set()).add(len(sl))
                    val_ty = args[0]
                    slot_types.setdefault(sl, set()).add(val_ty)
                    p_ty = id_type.get(ptr)
                    pointee = ptr_pointee_type.get(p_ty) if p_ty is not None else None
                    if pointee is not None and pointee != val_ty:
                        bad_slots.add(sl)
            for tok in args[2:]:
                if tok in ptr_root:
                    escaped.add(ptr_root[tok])
        elif op == "OpStore":
            if len(args) >= 2 and args[0] in ptr_root:
                ptr, val = args[0], args[1]
                root = ptr_root[ptr]
                if not reachable:
                    unreachable_roots.add(root)
                if any("Volatile" in m for m in args[2:]) or ptr in decorated_ids:
                    escaped.add(root)
                sl = ptr_slot.get(ptr)
                if sl is not None:
                    slot_depths.setdefault(root, set()).add(len(sl))
                    vty = id_type.get(val)
                    if vty is None:
                        bad_slots.add(sl)
                    else:
                        slot_types.setdefault(sl, set()).add(vty)
                        p_ty = id_type.get(ptr)
                        pointee = ptr_pointee_type.get(p_ty) if p_ty is not None else None
                        if pointee is not None and pointee != vty:
                            bad_slots.add(sl)
            for tok in args[1:]:
                if tok in ptr_root:
                    escaped.add(ptr_root[tok])
        elif op in ("OpAccessChain", "OpInBoundsAccessChain"):
            for tok in args[2:]:
                if tok in ptr_root:
                    escaped.add(ptr_root[tok])
        elif op == "OpCopyObject":
            pass
        elif op != "OpVariable":
            for tok in args:
                if tok in ptr_root:
                    escaped.add(ptr_root[tok])

    for root, depths in slot_depths.items():
        if len(depths) > 1:
            escaped.add(root)

    clean_roots = {v for v in func_vars if v not in escaped and v not in dyn_indexed and v not in unreachable_roots}
    if not clean_roots:
        return insts, False

    # 3. Assign dense integer IDs to candidate slots and resolve entry-block initializers.
    slot_to_sid: dict[tuple, int] = {}
    sid_val_type: list[str] = []
    sid_init: list[str | None] = []
    sid_has_unknown_init: list[bool] = []

    for sl, tys in slot_types.items():
        root = sl[0]
        if root not in clean_roots or sl in bad_slots or len(tys) != 1:
            continue
        val_ty = next(iter(tys))
        if val_ty in ptr_types:
            continue
        _, raw_init = func_vars[root]
        init_val: str | None = None
        unknown_init = False
        if raw_init is not None:
            if len(sl) == 1:
                if raw_init in const_ids and const_type.get(raw_init) == val_ty:
                    init_val = raw_init
                else:
                    unknown_init = True
            elif len(sl) == 2:
                z = const_null_elem_zero.get(raw_init)
                if z is not None and const_type.get(z) == val_ty:
                    init_val = z
                elif raw_init in const_composite_elems:
                    elems = const_composite_elems[raw_init][1]
                    k = sl[1]
                    if 0 <= k < len(elems) and const_type.get(elems[k]) == val_ty:
                        init_val = elems[k]
                    else:
                        unknown_init = True
                else:
                    unknown_init = True
            else:
                unknown_init = True
        sid = len(sid_val_type)
        slot_to_sid[sl] = sid
        sid_val_type.append(val_ty)
        sid_init.append(init_val)
        sid_has_unknown_init.append(unknown_init)

    n_slots = len(sid_val_type)
    if n_slots == 0:
        return insts, False

    ptr_sid: dict[str, int] = {}
    for ptr, sl in ptr_slot.items():
        if sl is not None and sl in slot_to_sid:
            ptr_sid[ptr] = slot_to_sid[sl]

    # 4. Per-block upward-exposed uses, defs, and access counts; backward bitset liveness.
    use_mask = [0] * n_blocks
    def_mask = [0] * n_blocks
    def_blocks: list[set[int]] = [set() for _ in range(n_slots)]
    slot_loads = [0] * n_slots
    slot_stores = [0] * n_slots

    for b in rpo:
        um = 0
        dm = 0
        for idx in blocks[b]:
            _, op, args, _ = insts[idx]
            if op == "OpLoad" and len(args) >= 2:
                sid = ptr_sid.get(args[1])
                if sid is not None:
                    slot_loads[sid] += 1
                    bit = 1 << sid
                    if not (dm & bit):
                        um |= bit
            elif op == "OpStore" and len(args) >= 2:
                sid = ptr_sid.get(args[0])
                if sid is not None:
                    slot_stores[sid] += 1
                    bit = 1 << sid
                    dm |= bit
                    def_blocks[sid].add(b)
        use_mask[b] = um
        def_mask[b] = dm

    live_in = [0] * n_blocks
    live_out = [0] * n_blocks
    l_changed = True
    while l_changed:
        l_changed = False
        for b in postorder:
            lo = 0
            for s in succs[b]:
                lo |= live_in[s]
            li = use_mask[b] | (lo & ~def_mask[b])
            if lo != live_out[b] or li != live_in[b]:
                live_out[b] = lo
                live_in[b] = li
                l_changed = True

    # 5. Cooper-Harvey-Kennedy dominance frontiers with early-stop deduplication.
    df: list[set[int]] = [set() for _ in range(n_blocks)]
    for b in rpo:
        preds = reach_preds[b]
        if len(preds) >= 2:
            ib = idom[b]
            for p in preds:
                runner = p
                while runner != ib and b not in df[runner]:
                    df[runner].add(b)
                    runner = idom[runner]

    # 6. Pruned Iterated Dominance Frontier + per-slot safety and SPIR-V size cost model.
    promoted_sids: set[int] = set()
    block_phis: list[list[tuple[int, str, str]]] = [[] for _ in range(n_blocks)]
    phi_args: dict[str, list[str]] = {}
    fresh_counter = 0

    def fresh_id() -> str:
        nonlocal fresh_counter
        while True:
            cand = f"%m2r_{fresh_counter}"
            fresh_counter += 1
            if cand not in used_ids:
                used_ids.add(cand)
                return cand

    entry_live_in = live_in[0]
    for sid in range(n_slots):
        bit = 1 << sid
        if (entry_live_in & bit) and (sid_init[sid] is None or sid_has_unknown_init[sid]):
            continue
        if slot_loads[sid] == 0 and slot_stores[sid] == 0:
            continue

        phi_blocks: set[int] = set()
        if slot_loads[sid] > 0 and def_blocks[sid]:
            worklist = list(def_blocks[sid])
            ever_queued = set(worklist)
            abort = False
            while worklist:
                d = worklist.pop()
                for y in df[d]:
                    if not (live_in[y] & bit):
                        continue
                    if y in phi_blocks:
                        continue
                    if y in unsafe_phi_blocks or len(reach_preds[y]) > max_phi_preds:
                        abort = True
                        break
                    phi_blocks.add(y)
                    if y not in ever_queued:
                        ever_queued.add(y)
                        worklist.append(y)
                if abort:
                    break
            if abort:
                continue

        if phi_blocks:
            phi_words = sum(3 + 2 * len(reach_preds[y]) for y in phi_blocks)
            mem_words = 4 * slot_loads[sid] + 3 * slot_stores[sid]
            if phi_words > max(mem_words * 3, 128):
                continue

        promoted_sids.add(sid)
        val_ty = sid_val_type[sid]
        for y in sorted(phi_blocks):
            pid = fresh_id()
            block_phis[y].append((sid, pid, val_ty))
            phi_args[pid] = [val_ty]

    if not promoted_sids:
        return insts, False

    # 7. Iterative dominator-tree preorder SSA renaming.
    stacks: dict[int, list[str]] = {
        sid: ([sid_init[sid]] if sid_init[sid] is not None else [])
        for sid in promoted_sids
    }
    alias: dict[str, str] = {}

    def resolve(tok: str) -> str:
        cur = tok
        while cur in alias:
            cur = alias[cur]
        if cur != tok:
            alias[tok] = cur
        return cur

    dead_idx: set[int] = set()
    dfs: list[tuple] = [("enter", 0)]
    while dfs:
        frame = dfs.pop()
        if frame[0] == "enter":
            b = frame[1]
            pushed: list[int] = []
            for sid, pid, _ in block_phis[b]:
                stacks[sid].append(pid)
                pushed.append(sid)

            for idx in blocks[b]:
                res, op, args, _ = insts[idx]
                if op == "OpLoad" and len(args) >= 2:
                    sid = ptr_sid.get(args[1])
                    if sid is not None and sid in promoted_sids:
                        alias[res] = stacks[sid][-1]
                        dead_idx.add(idx)
                        deleted_ids.add(res)
                elif op == "OpStore" and len(args) >= 2:
                    sid = ptr_sid.get(args[0])
                    if sid is not None and sid in promoted_sids:
                        stacks[sid].append(resolve(args[1]))
                        pushed.append(sid)
                        dead_idx.add(idx)

            lbl_b = block_label[b]
            for s in succs[b]:
                for sid, pid, _ in block_phis[s]:
                    phi_args[pid].append(stacks[sid][-1])
                    phi_args[pid].append(lbl_b)

            dfs.append(("exit", pushed))
            for ch in reversed(dom_children[b]):
                dfs.append(("enter", ch))
        else:
            for sid in reversed(frame[1]):
                stacks[sid].pop()

    # 8. Resolve aliases everywhere (including backedges of pre-existing and newly inserted OpPhis).
    if alias:
        for pid, pargs in phi_args.items():
            for k in range(1, len(pargs), 2):
                pargs[k] = resolve(pargs[k])
        for idx, entry in enumerate(insts):
            if idx in dead_idx:
                continue
            _, op, args, _ = entry
            if op is None or not args:
                continue
            changed_a = False
            new_args = []
            for t in args:
                if t in alias:
                    new_args.append(resolve(t))
                    changed_a = True
                else:
                    new_args.append(t)
            if changed_a:
                entry[2] = new_args

    # 9. Splice newly created OpPhis immediately after each block's OpLabel and drop dead loads/stores.
    label_idx_to_block = {blocks[b][0]: b for b in range(n_blocks) if block_phis[b]}
    new_insts: list[list] = []
    for idx, entry in enumerate(insts):
        if idx in dead_idx:
            continue
        new_insts.append(entry)
        b = label_idx_to_block.get(idx)
        if b is not None:
            for _sid, pid, _ty in block_phis[b]:
                pargs = phi_args[pid]
                new_insts.append([pid, "OpPhi", pargs, format_inst(pid, "OpPhi", pargs)])

    return new_insts, True


def collapse_trivial_phis(insts: list[list], build_cfg_and_dom, decorated_ids: set[str],
                          deleted_ids: set[str]) -> tuple[list[list], bool]:
    """Iteratively collapse trivial OpPhis with O(1) Euler-tour strict-dominance checks."""
    blocks, _succs, _all_preds, rpo, _idom, dom_children = build_cfg_and_dom(insts)
    if not blocks:
        return insts, False

    reachable_set = set(rpo)
    tin, tout = _dom_euler(dom_children, len(blocks))
    def_block: dict[str, int] = {}
    phi_indices: list[tuple[int, int]] = []
    for b, b_insts in enumerate(blocks):
        for idx in b_insts:
            res, op, _args, _ = insts[idx]
            if res is not None:
                def_block[res] = b
            if op == "OpPhi" and b in reachable_set and res is not None and res not in decorated_ids:
                phi_indices.append((b, idx))

    if not phi_indices:
        return insts, False

    alias: dict[str, str] = {}

    def resolve(tok: str) -> str:
        cur = tok
        while cur in alias:
            cur = alias[cur]
        if cur != tok:
            alias[tok] = cur
        return cur

    dead_idx: set[int] = set()
    progress = True
    while progress:
        progress = False
        for b, idx in phi_indices:
            if idx in dead_idx:
                continue
            res, _op, args, _ = insts[idx]
            if len(args) < 3:
                continue
            non_self: set[str] = set()
            for k in range(1, len(args), 2):
                v = resolve(args[k])
                if v != res:
                    non_self.add(v)
                    if len(non_self) > 1:
                        break
            if len(non_self) == 1:
                cand = next(iter(non_self))
                cb = def_block.get(cand)
                if cb is None or (cb != b and tin[cb] >= 0 and tin[cb] <= tin[b] <= tout[cb]):
                    alias[res] = cand
                    dead_idx.add(idx)
                    deleted_ids.add(res)
                    progress = True

    if not dead_idx:
        return insts, False

    out: list[list] = []
    for idx, entry in enumerate(insts):
        if idx in dead_idx:
            continue
        _, op, args, _ = entry
        if op is not None and args:
            changed_a = False
            new_args = []
            for t in args:
                if t in alias:
                    new_args.append(resolve(t))
                    changed_a = True
                else:
                    new_args.append(t)
            if changed_a:
                entry[2] = new_args
        out.append(entry)
    return out, True


def mark_and_sweep_dce(insts: list[list], trackable: set[str], readonly_global_vars: set[str],
                       ptr_root: dict[str, str], glsl_sets: set[str], pure_ops: frozenset[str],
                       pure_glsl_ext: frozenset[str], deleted_ids: set[str]) -> tuple[list[list], bool]:
    """Mark-and-sweep pure DCE over a function body; eliminates dead OpPhi cycles as well."""
    n = len(insts)
    def_idx: dict[str, int] = {}
    for idx, (res, op, _args, _raw) in enumerate(insts):
        if res is not None and op is not None:
            def_idx[res] = idx

    def is_pure_removable(res: str | None, op: str | None, args: list[str] | None) -> bool:
        if res is None or op is None or args is None:
            return False
        if op in pure_ops or op == "OpPhi":
            return True
        if op == "OpExtInst" and len(args) >= 3 and args[1] in glsl_sets and args[2] in pure_glsl_ext:
            return True
        if op == "OpLoad" and len(args) >= 2 and not any("Volatile" in m for m in args[2:]):
            ptr = args[1]
            root = ptr_root.get(ptr)
            return root in trackable or root in readonly_global_vars or ptr in readonly_global_vars
        if op == "OpVariable" and len(args) >= 2 and args[1] == "Function" and res in trackable:
            return True
        return False

    live = [False] * n
    worklist: list[int] = []
    for idx, (res, op, args, _raw) in enumerate(insts):
        if not is_pure_removable(res, op, args):
            live[idx] = True
            worklist.append(idx)

    while worklist:
        idx = worklist.pop()
        _, op, args, _ = insts[idx]
        if op is None or not args:
            continue
        for tok in args:
            d = def_idx.get(tok)
            if d is not None and not live[d]:
                live[d] = True
                worklist.append(d)

    dead_count = 0
    for idx, is_live in enumerate(live):
        if not is_live:
            dead_count += 1
            res = insts[idx][0]
            if res is not None:
                deleted_ids.add(res)

    if dead_count == 0:
        return insts, False

    return [e for idx, e in enumerate(insts) if live[idx]], True


_HOISTABLE_PURE_OPS = frozenset((
    "OpIAdd", "OpISub", "OpIMul",
    "OpBitwiseOr", "OpBitwiseAnd", "OpBitwiseXor", "OpNot",
    "OpBitReverse", "OpBitCount",
    "OpIEqual", "OpINotEqual",
    "OpUGreaterThan", "OpUGreaterThanEqual", "OpULessThan", "OpULessThanEqual",
    "OpSGreaterThan", "OpSGreaterThanEqual", "OpSLessThan", "OpSLessThanEqual",
    "OpLogicalAnd", "OpLogicalOr", "OpLogicalNot", "OpLogicalEqual", "OpLogicalNotEqual",
    "OpUConvert", "OpSConvert", "OpBitcast", "OpSelect",
    "OpCompositeConstruct", "OpCompositeExtract", "OpVectorShuffle",
))

_HOIST_COMMUTATIVE = frozenset((
    "OpIAdd", "OpIMul", "OpBitwiseOr", "OpBitwiseAnd", "OpBitwiseXor",
    "OpIEqual", "OpINotEqual", "OpLogicalAnd", "OpLogicalOr",
    "OpLogicalEqual", "OpLogicalNotEqual",
))

_BLOCK_END_CONTROL = frozenset((
    "OpSelectionMerge", "OpLoopMerge",
    "OpBranch", "OpBranchConditional", "OpSwitch",
    "OpReturn", "OpReturnValue", "OpKill", "OpTerminateInvocation", "OpUnreachable",
))


def _is_hoistable(op: str, args: list[str], const_int_val: dict[str, tuple[str, int, int, int]],
                  int_type_info: dict[str, tuple[int, int]]) -> bool:
    if op in _HOISTABLE_PURE_OPS:
        return True
    if op in ("OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic") and len(args) == 3:
        info = int_type_info.get(args[0])
        c2 = const_int_val.get(args[2])
        return info is not None and c2 is not None and c2[1] < info[0]
    if op in ("OpBitFieldUExtract", "OpBitFieldSExtract") and len(args) == 4:
        info = int_type_info.get(args[0])
        co = const_int_val.get(args[2])
        cc = const_int_val.get(args[3])
        return info is not None and co is not None and cc is not None and co[1] + cc[1] <= info[0]
    return False


def hoist_common_pure_exprs(insts: list[list], build_cfg_and_dom, format_inst,
                            const_int_val: dict[str, tuple[str, int, int, int]],
                            int_type_info: dict[str, tuple[int, int]],
                            decorated_ids: set[str], deleted_ids: set[str],
                            max_rounds: int = 12) -> tuple[list[list], bool]:
    """Hoist identical side-effect-free integer/bit/boolean expressions that occur in >=2
    sibling reachable blocks up to their dominator-tree LCA (GVN-Hoist).

    CFG structure and the dominator tree never change across rounds, so we compute
    dominators and Euler-tour timestamps once (O(N)), find the LCA of any block set in
    O(tree height) via min/max tin, and check operand dominance in O(1).
    """
    blocks, _succs, _all_preds, rpo, idom, dom_children = build_cfg_and_dom(insts)
    n_blocks = len(blocks)
    if n_blocks < 3:
        return insts, False

    tin, tout = _dom_euler(dom_children, n_blocks)

    # Split insts into prologue (before block 0), per-block lists, and epilogue (OpFunctionEnd).
    first_blk_idx = blocks[0][0]
    last_blk_idx = blocks[-1][-1]
    prologue = insts[:first_blk_idx]
    epilogue = insts[last_blk_idx + 1:]
    blk_entries: list[list[list]] = [[insts[idx] for idx in b_insts] for b_insts in blocks]

    def_block: dict[str, int] = {}
    for b, entries in enumerate(blk_entries):
        for entry in entries:
            res = entry[0]
            if res is not None:
                def_block[res] = b

    any_changed = False
    for _round in range(max_rounds):
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for b in rpo:
            entries = blk_entries[b]
            for pos, (res, op, args, _) in enumerate(entries):
                if res is None or op is None or not args or res in decorated_ids:
                    continue
                if not _is_hoistable(op, args, const_int_val, int_type_info):
                    continue
                if op in _HOIST_COMMUTATIVE and len(args) == 3 and args[1] > args[2]:
                    key = (op, (args[0], args[2], args[1]))
                else:
                    key = (op, tuple(args))
                groups.setdefault(key, []).append((b, pos))

        hoist_insert: dict[int, list[list]] = {}
        dead_in_blk: dict[int, set[int]] = {}
        alias: dict[str, str] = {}

        for _key, sites in groups.items():
            if len(sites) < 2:
                continue
            # Find min-tin and max-tin blocks in one pass; LCA of a set = LCA(min_tin, max_tin).
            min_b = sites[0][0]
            max_b = min_b
            min_t = tin[min_b]
            max_t = min_t
            same_blk = True
            for sb, _ in sites[1:]:
                if sb != min_b:
                    same_blk = False
                st = tin[sb]
                if st < min_t:
                    min_t = st
                    min_b = sb
                if st > max_t:
                    max_t = st
                    max_b = sb
            if same_blk:
                continue

            # Walk up from min_b until tout[lca] >= max_t (exact tree LCA, O(depth) once per key).
            lca = min_b
            while tout[lca] < max_t:
                lca = idom[lca]
            lca_tin = tin[lca]

            first_b, first_pos = sites[0]
            first_res, first_op, first_args, _ = blk_entries[first_b][first_pos]
            operands_ok = True
            for tok in first_args[1:]:
                db = def_block.get(tok)
                if db is not None and not (tin[db] <= lca_tin <= tout[db]):
                    operands_ok = False
                    break
            if not operands_ok:
                continue

            # If a copy already exists inside lca, reuse the earliest copy in lca without moving it.
            winner_pos = None
            for sb, spos in sites:
                if sb == lca and (winner_pos is None or spos < winner_pos):
                    winner_pos = spos
            if winner_pos is not None:
                winner_res = blk_entries[lca][winner_pos][0]
                for sb, spos in sites:
                    if sb == lca and spos == winner_pos:
                        continue
                    r = blk_entries[sb][spos][0]
                    alias[r] = winner_res
                    dead_in_blk.setdefault(sb, set()).add(spos)
                    deleted_ids.add(r)
                    def_block.pop(r, None)
            else:
                winner_res = first_res
                hoist_insert.setdefault(lca, []).append(
                    [winner_res, first_op, list(first_args), format_inst(winner_res, first_op, first_args)]
                )
                def_block[winner_res] = lca
                dead_in_blk.setdefault(first_b, set()).add(first_pos)
                for sb, spos in sites[1:]:
                    r = blk_entries[sb][spos][0]
                    alias[r] = winner_res
                    dead_in_blk.setdefault(sb, set()).add(spos)
                    deleted_ids.add(r)
                    def_block.pop(r, None)

        if not dead_in_blk:
            break

        any_changed = True

        def resolve(tok: str) -> str:
            cur = tok
            while cur in alias:
                cur = alias[cur]
            if cur != tok:
                alias[tok] = cur
            return cur

        for b in range(n_blocks):
            entries = blk_entries[b]
            dead_pos = dead_in_blk.get(b)
            moved = hoist_insert.get(b)

            if dead_pos is not None:
                entries = [e for pos, e in enumerate(entries) if pos not in dead_pos]

            if moved is not None:
                # Insert right before OpSelectionMerge / OpLoopMerge / terminator of block b.
                ins_pos = len(entries) - 1
                for pos in range(len(entries) - 1, -1, -1):
                    op = entries[pos][1]
                    if op in _BLOCK_END_CONTROL:
                        ins_pos = pos
                    elif op is not None:
                        break
                entries[ins_pos:ins_pos] = moved

            if alias:
                for entry in entries:
                    op, args = entry[1], entry[2]
                    if op is None or not args:
                        continue
                    changed_a = False
                    new_args = []
                    for t in args:
                        if t in alias:
                            new_args.append(resolve(t))
                            changed_a = True
                        else:
                            new_args.append(t)
                    if changed_a:
                        entry[2] = new_args

            blk_entries[b] = entries

    if not any_changed:
        return insts, False

    out_insts = list(prologue)
    for entries in blk_entries:
        out_insts.extend(entries)
    out_insts.extend(epilogue)
    return out_insts, True
