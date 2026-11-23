"""Which scalar registers hold a lane mask?

On hardware a lane mask is one register for the whole wave, a bit per lane. The translator runs one invocation
per lane and keeps only that lane's bit in the register (all ones or zero), which is all that and/or/not, selects
and EXEC need. A few instructions look at the mask as a whole instead: s_bcnt1 (how many lanes), s_ff1 (the first
lane), v_mbcnt (how many lanes below this one). For those the translator has to collect the other lanes' bits.
The same instructions also run on plain 64-bit data, such as a light mask read from a buffer, and nothing in the
instruction says which of the two it has, so this module works it out from what wrote the register.

    masks, writes = analyse(blocks, succ)   # blocks: lists of instructions; succ: successor block numbers per block
    masks[id(ins)]                      # dict mapping each register holding a lane mask on every path to ins
                                        # (SGPR number, "vcc", ("lane", vgpr, lane) parked in a VGPR lane, or
                                        # ("v", vgpr) passing through a VGPR) to its kind (MASK, MASK_LO, MASK_HI)
    id(ins) in writes                   # ins computes a mask into a VGPR (a plain copy does not count)

A register counts as a mask only if every path to the instruction wrote a mask to it. Anything else is treated
as data, which is what the translator did for all of them before.

Parked masks: a shader that runs out of SGPRs moves some into single lanes of a spare VGPR (v_writelane_b32) and
fetches them back later (v_readlane_b32). When the SGPR is a saved EXEC, the value that comes back has to be this
lane's bit again, not the bit of the lane it was parked in.

Masks through a VGPR: the compiler does some scalar work in the vector unit, such as an OR of three SGPRs
(v_mov_b32 / v_or3_b32, then v_readfirstlane_b32 to get the scalar back). On hardware every lane computes the same
whole mask and the first lane's copy is read. Here each lane computes its own bit, so it has to read its own
copy, and it has to compute it whether or not it is active. A plain copy of a mask (v_mov_b32) is different:
it moves an already-computed value, so it is an ordinary VALU write and inactive lanes keep the old VGPR
(SHADER-21 part 2).
"""

MASK = "mask"           # full lane mask (64-bit pair, vcc, or wave32 single register)
MASK_LO = "mask_lo"     # low 32-bit half of a 64-bit lane mask (lanes 0-31)
MASK_HI = "mask_hi"     # high 32-bit half of a 64-bit lane mask (lanes 32-63)
MASK_KINDS = (MASK, MASK_LO, MASK_HI)
EITHER = "either"       # a constant 0 or -1: no lanes or all lanes, and the same bits when read as data
# Absent from the state: data.

COPIES = ("s_mov_b", "s_not_b", "s_wqm_b")
BITWISE = ("s_and_b", "s_or_b", "s_xor_b", "s_andn2_b", "s_orn2_b", "s_nand_b", "s_nor_b", "s_xnor_b",
           "s_cselect_b")
NO_DESTINATION = (                      # their first operand (if any) is a source, not a written register
    "s_cmp", "s_bitcmp", "v_cmpx", "s_set", "s_rfe",
    "s_store_", "s_buffer_store_", "s_dcache_", "s_icache_",
    "s_sendmsg", "s_branch", "s_cbranch", "s_waitcnt", "s_nop", "s_sleep", "s_barrier", "s_endpgm",
    "exp",
    "buffer_store", "tbuffer_store", "global_store", "flat_store", "image_store", "ds_write",
)
DS_HAS_DST = ("ds_read", "ds_swizzle", "ds_permute", "ds_bpermute",
              "ds_append", "ds_consume", "ds_ordered_count", "ds_nop")
VGPR_REL_DST = ("v_movreld_b32", "v_movrelsd_b32", "v_movrelsd_2_b32", "v_swaprel_b32")
SGPR_REL_DST = ("s_movreld_b32", "s_movreld_b64", "s_movrelsd_2_b32")
VECTOR_BITWISE = ("v_mov_b32", "v_or3_b32", "v_or_b32", "v_and_b32", "v_and_or_b32",
                  "v_xor_b32", "v_xor3_b32", "v_xnor_b32", "v_bfi_b32", "v_not_b32")

_OPCODE_CACHE_LIMIT = 4096
_BASE_OP_CACHE = {}
_OP_INFO_CACHE = {}
_VCC_OPERAND = {"k": "special", "t": "vcc"}

_OP_NO_DST = 0
_OP_BUF_LOAD = 1
_OP_ATOMIC = 2
_OP_V_READFIRSTLANE = 3
_OP_V_WRITELANE = 4
_OP_V_READLANE = 5
_OP_VCMP = 6
_OP_VSWAP = 7
_OP_VGPR_REL = 8
_OP_SGPR_REL = 9
_OP_GENERAL = 10

_SALU_OTHER = 0
_SALU_SAVEEXEC = 1
_SALU_SAVEEXEC_B32 = 2
_SALU_COPY = 3
_SALU_BITWISE = 4
_SALU_BFM = 5


def base_op(op):
    cached = _BASE_OP_CACHE.get(op)
    if cached is not None:
        return cached
    res = op
    for suffix in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suffix):
            res = op[:-len(suffix)]
            break
    if len(_BASE_OP_CACHE) < _OPCODE_CACHE_LIMIT:
        _BASE_OP_CACHE[op] = res
    return res


def _op_info(raw_op):
    cached = _OP_INFO_CACHE.get(raw_op)
    if cached is not None:
        return cached
    op = base_op(raw_op)
    if op.startswith(NO_DESTINATION):
        info = (_OP_NO_DST, op, False, False, _SALU_OTHER, False, False, False)
    elif op.startswith(("buffer_load", "tbuffer_load")):
        info = (_OP_BUF_LOAD, op, False, False, _SALU_OTHER, False, False, False)
    elif op.startswith(("buffer_atomic", "global_atomic", "flat_atomic", "image_atomic",
                        "s_atomic", "s_buffer_atomic")):
        info = (_OP_ATOMIC, op, False, False, _SALU_OTHER, False, False, False)
    elif op.startswith("ds_") and not op.startswith(DS_HAS_DST) and "_rtn" not in op:
        info = (_OP_NO_DST, op, False, False, _SALU_OTHER, False, False, False)
    elif op == "v_readfirstlane_b32":
        info = (_OP_V_READFIRSTLANE, op, False, False, _SALU_OTHER, False, False, False)
    elif op == "v_writelane_b32":
        info = (_OP_V_WRITELANE, op, False, False, _SALU_OTHER, False, False, False)
    elif op == "v_readlane_b32":
        info = (_OP_V_READLANE, op, False, False, _SALU_OTHER, False, False, False)
    elif op.startswith("v_cmp"):
        info = (_OP_VCMP, op, False, False, _SALU_OTHER, False, False, False)
    elif op == "v_swap_b32":
        info = (_OP_VSWAP, op, False, False, _SALU_OTHER, False, False, False)
    elif op in VGPR_REL_DST:
        info = (_OP_VGPR_REL, op, False, False, _SALU_OTHER, False, False, False)
    elif op in SGPR_REL_DST:
        info = (_OP_SGPR_REL, op, False, False, _SALU_OTHER, False, False, False)
    else:
        vec_bitwise_ok = op in VECTOR_BITWISE and not raw_op.endswith(("_sdwa", "_dpp", "_dpp8"))
        is_v_mov = op == "v_mov_b32"
        if "_saveexec_b" in op or "_wrexec_b" in op:
            salu_kind = _SALU_SAVEEXEC_B32 if op.endswith("_b32") else _SALU_SAVEEXEC
        elif op == "s_bfm_b64":
            salu_kind = _SALU_BFM
        elif op.startswith(COPIES):
            salu_kind = _SALU_COPY
        elif op.startswith(BITWISE):
            salu_kind = _SALU_BITWISE
        else:
            salu_kind = _SALU_OTHER
        is_b64 = op.endswith("b64")
        is_b32 = op.endswith("b32")
        is_vop3b_mask = ((op.startswith("v_") and "_co_" in op)
                         or op in ("v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64"))
        info = (_OP_GENERAL, op, vec_bitwise_ok, is_v_mov, salu_kind, is_b64, is_b32, is_vop3b_mask)
    if len(_OP_INFO_CACHE) < _OPCODE_CACHE_LIMIT:
        _OP_INFO_CACHE[raw_op] = info
    return info


def kind(state, a):
    ak = a["k"]
    if ak == "sgpr":
        r = a["r"]
        n = a.get("n", 1)
        if n == 1:
            return state.get(r)
        if n == 2:
            k0 = state.get(r)
            if k0 is None:
                return None
            k1 = state.get(r + 1)
            if k1 is None:
                return None
            if k0 in MASK_KINDS or k1 in MASK_KINDS:
                return MASK
            return k0 if k0 == k1 else None
        ks = [state.get(r + i) for i in range(n)]
        if None in ks:
            return None
        return ks[0] if len(set(ks)) == 1 else None
    if ak == "vgpr":
        return state.get(("v", a["r"])) if a.get("n", 1) == 1 else None
    if ak == "imm":
        return EITHER if a["v"] in (0, -1) else None
    if ak == "special":
        t = a.get("t", "")
        if t == "exec_lo":
            return MASK_LO
        if t == "exec_hi":
            return MASK_HI
        if t == "exec":
            return MASK
        if t == "vcc":
            return state.get("vcc")
        if t in ("vcc_lo", "vcc_hi"):
            vk = state.get("vcc")
            if vk in MASK_KINDS:
                return MASK_LO if t == "vcc_lo" else MASK_HI
            return vk if t == "vcc_lo" else None
        return None
    return None


def _put(state, a, k):
    """Write `k` (or clear when None) to register operand `a` in `state`. Returns True if `state` changed."""
    ak = a["k"]
    if ak == "sgpr":
        r = a["r"]
        n = a.get("n", 1)
        if n == 1:
            if k is None:
                return state.pop(r, None) is not None
            if state.get(r) == k:
                return False
            state[r] = k
            return True
        if k == MASK and n == 2:
            if state.get(r) == MASK_LO and state.get(r + 1) == MASK_HI:
                return False
            state[r] = MASK_LO
            state[r + 1] = MASK_HI
            return True
        changed = False
        if k is None:
            for i in range(n):
                if state.pop(r + i, None) is not None:
                    changed = True
        else:
            for i in range(n):
                ri = r + i
                if state.get(ri) != k:
                    state[ri] = k
                    changed = True
        return changed
    if ak == "slist":
        changed = False
        for r in a["r"]:
            key = r if isinstance(r, int) else "vcc"
            if k is None:
                if state.pop(key, None) is not None:
                    changed = True
            else:
                if state.get(key) != k:
                    state[key] = k
                    changed = True
        return changed
    if ak == "special":
        t = a.get("t")
        if t == "vcc":
            if k is None:
                return state.pop("vcc", None) is not None
            if state.get("vcc") == k:
                return False
            state["vcc"] = k
            return True
        if t in ("vcc_lo", "vcc_hi"):
            # A mask / 0 / -1 half-write updates that half of %vcc in wave64 (put_mask), keeping VCC a lane mask;
            # a plain-data half-write (k is None) ends VCC's life as one lane mask.
            vk = state.get("vcc")
            if k is not None and vk in MASK_KINDS:
                if vk == MASK:
                    return False
                state["vcc"] = MASK
                return True
            if t == "vcc_lo" and k is not None:
                if vk == k:
                    return False
                state["vcc"] = k
                return True
            return state.pop("vcc", None) is not None
    return False


def put(state, a, k):
    _put(state, a, k)


def _maskable_b32(x):
    return x["k"] in ("sgpr", "imm") or (
        x["k"] == "special" and (x.get("t", "").startswith(("exec", "vcc")) or x.get("t") == "null")
    )


def lane_slot(vgpr, lane):
    """State key for a mask parked in one lane of a VGPR, or None unless both operands are plain."""
    if vgpr["k"] != "vgpr" or lane["k"] != "imm":
        return None
    return ("lane", vgpr["r"], lane["v"] & 63)


def _step_tracked(state, ins, vgpr_keys, wave64=True):
    """Update `state` (and optional `vgpr_keys` set of tuple keys in `state`) for one instruction.

    Returns `(wrote_vgpr_mask, state_changed)`. `wave64` False: the program runs in 32-lane waves, where a 64-bit
    s_bfm_b64 result can't be a lane mask.
    """
    a = ins["args"]
    if not a:
        return (False, False)
    tag, op, vec_bitwise_ok, is_v_mov, salu_kind, is_b64, is_b32, is_vop3b_mask = _op_info(ins["op"])
    if tag == _OP_NO_DST:
        return (False, False)
    mods = ins.get("mods", ())
    if tag == _OP_BUF_LOAD:
        if "lds" in mods:
            return (False, False)
    elif tag == _OP_ATOMIC:
        if "glc" not in mods:
            return (False, False)
    elif tag == _OP_V_READFIRSTLANE:
        if len(a) == 2:                                                         # sD = vS, the same in every lane
            return (False, _put(state, a[0], kind(state, a[1])))
    elif tag == _OP_V_WRITELANE:
        if len(a) == 3:
            slot = lane_slot(a[0], a[2])
            if slot:                                                            # vD[lane] = sS
                vkey = ("v", a[0]["r"])
                changed = state.pop(vkey, None) is not None
                if changed and vgpr_keys is not None:
                    vgpr_keys.discard(vkey)
                k = kind(state, a[1])
                if k is None:
                    if state.pop(slot, None) is not None:
                        changed = True
                        if vgpr_keys is not None:
                            vgpr_keys.discard(slot)
                else:
                    if state.get(slot) != k:
                        state[slot] = k
                        changed = True
                    if vgpr_keys is not None:
                        vgpr_keys.add(slot)
                return (False, changed)
    elif tag == _OP_V_READLANE:
        if len(a) == 3:                                                         # sD = vS[lane]
            return (False, _put(state, a[0], state.get(lane_slot(a[1], a[2]))))
    elif tag == _OP_VCMP:
        # Handle v_cmp before VGPR clobber (defensive for 2-operand forms with implicit vcc).
        return (False, _put(state, _VCC_OPERAND if len(a) == 2 else a[0], MASK))
    elif tag == _OP_VSWAP:
        if vgpr_keys is not None:
            if not vgpr_keys:
                return (False, False)
            keys_iter = vgpr_keys
        elif not state:
            return (False, False)
        else:
            keys_iter = state
        written = set()
        for x in a[:2]:
            if x["k"] == "vgpr":
                written.update(range(x["r"], x["r"] + x.get("n", 1)))
        to_del = [k for k in keys_iter if isinstance(k, tuple) and k[1] in written]
        for key in to_del:
            del state[key]
            if vgpr_keys is not None:
                vgpr_keys.discard(key)
        return (False, bool(to_del))
    elif tag == _OP_VGPR_REL:
        if vgpr_keys is not None:
            if not vgpr_keys:
                return (False, False)
            keys_iter = vgpr_keys
        elif not state:
            return (False, False)
        else:
            keys_iter = state
        bases = [x["r"] for x in (a[:2] if op == "v_swaprel_b32" else a[:1]) if x["k"] == "vgpr"]
        if bases:
            lo = min(bases)
            to_del = [k for k in keys_iter if isinstance(k, tuple) and k[1] >= lo]
            for key in to_del:
                del state[key]
                if vgpr_keys is not None:
                    vgpr_keys.discard(key)
            return (False, bool(to_del))
        return (False, False)
    elif tag == _OP_SGPR_REL:
        if a[0]["k"] == "sgpr" and state:
            lo = a[0]["r"]
            to_del = [k for k in state if isinstance(k, int) and k >= lo]
            for key in to_del:
                del state[key]
            return (False, bool(to_del))
        return (False, False)

    a0 = a[0]
    if a0["k"] == "vgpr":                               # any other write to a VGPR ends what was parked in it
        # A mask only when every source is one (or a constant 0 / -1) on every path: no EITHER for VGPRs, whose
        # plain writes skip inactive lanes. The sources are looked at first: the destination can be one of them.
        # SDWA/DPP or modified forms alter sub-dwords or read other lanes, so they are never plain mask ops.
        changed = False
        r0 = a0["r"]
        n0 = a0.get("n", 1)
        plain_valu = (vec_bitwise_ok and n0 == 1 and not mods
                      and not any(x.get("neg") or x.get("abs") or x.get("sext") for x in a[1:]))
        kinds = [kind(state, x) for x in a[1:]] if plain_valu else None
        if vgpr_keys is not None:
            if vgpr_keys:
                hi0 = r0 + n0
                to_del = [k for k in vgpr_keys if r0 <= k[1] < hi0]
                for key in to_del:
                    del state[key]
                    vgpr_keys.discard(key)
                if to_del:
                    changed = True
        elif state:
            hi0 = r0 + n0
            to_del = [k for k in state if isinstance(k, tuple) and r0 <= k[1] < hi0]
            for key in to_del:
                del state[key]
            if to_del:
                changed = True
        if kinds is not None and None not in kinds and any(k in MASK_KINDS for k in kinds):
            ms = {k for k in kinds if k in MASK_KINDS}
            vkey = ("v", r0)
            new_k = ms.pop() if len(ms) == 1 else MASK
            state[vkey] = new_k
            if vgpr_keys is not None:
                vgpr_keys.add(vkey)
            # A plain copy (v_mov_b32) is an ordinary VALU write: inactive lanes keep the old
            # VGPR (SHADER-21 part 2). Only per-lane mask computations take the all-lanes store,
            # because whole-wave readers (wave_mask, v_readfirstlane) read every lane's bit.
            return (not is_v_mov, True)
        if is_vop3b_mask and len(a) > 1:
            if _put(state, a[1], MASK):
                changed = True
        return (False, changed)

    if salu_kind == _SALU_SAVEEXEC:
        return (False, _put(state, a0, MASK))
    if salu_kind == _SALU_SAVEEXEC_B32:
        return (False, _put(state, a0, MASK_LO))
    if salu_kind == _SALU_BFM:
        # In wave32 code it is 64-bit data, e.g. the first two words of an S# built in place (the pipeline library's
        # wave32 compute programs: s_bfm_b64 s[88:89], 2, 54); rdna2spirv computes the value there.
        return (False, _put(state, a0, MASK if wave64 else None))
    if salu_kind == _SALU_COPY and len(a) > 1:
        k = kind(state, a[1])
        dst_t = a0.get("t") if a0["k"] == "special" else None
        if k in (None, EITHER) and is_b64 and dst_t in ("vcc", "vcc_lo") and a[1]["k"] == "imm":
            k = MASK
        elif op == "s_not_b32" and dst_t in ("vcc", "vcc_lo") and _maskable_b32(a[1]):
            k = MASK
        elif op == "s_not_b32" and dst_t == "vcc_hi" and state.get("vcc") in MASK_KINDS and _maskable_b32(a[1]):
            return (False, False)
        return (False, _put(state, a0, k))
    if salu_kind == _SALU_BITWISE:
        kinds = [kind(state, x) for x in a[1:]]
        ms = {k for k in kinds if k in MASK_KINDS}
        dst_t = a0.get("t") if a0["k"] == "special" else None
        if None not in kinds:
            if is_b32 and dst_t == "vcc_hi" and state.get("vcc") in MASK_KINDS:
                return (False, False)
            fallback = MASK if is_b64 and dst_t in ("vcc", "vcc_lo") else EITHER
            return (False, _put(state, a0, (ms.pop() if len(ms) == 1 else MASK) if ms else fallback))
        if is_b64 and (ms or dst_t in ("vcc", "vcc_lo")) and \
                all(k is not None or x["k"] == "imm" for k, x in zip(kinds, a[1:])):
            return (False, _put(state, a0, (ms.pop() if len(ms) == 1 else MASK) if ms else MASK))
        if is_b32 and dst_t in ("vcc", "vcc_lo") and all(_maskable_b32(x) for x in a[1:]):
            return (False, _put(state, a0, MASK))
        if is_b32 and dst_t == "vcc_hi" and state.get("vcc") in MASK_KINDS and \
                all(_maskable_b32(x) for x in a[1:]):
            return (False, False)
        return (False, _put(state, a0, None))
    changed = _put(state, a0, None)
    if is_vop3b_mask and len(a) > 1:
        if _put(state, a[1], MASK):                      # carry/sign/scale lane mask of VOP3b ops
            changed = True
    return (False, changed)


def step(state, ins, wave64=True):
    """Update `state` (register -> MASK | MASK_LO | MASK_HI | EITHER) for one instruction. True if it writes a mask to a VGPR.

    `wave64` False: the program runs in 32-lane waves, where a 64-bit s_bfm_b64 result can't be a lane mask."""
    wrote, _ = _step_tracked(state, ins, None, wave64)
    return True if wrote else None


_DEFAULT_STEP = step


def parked(blocks, masks):
    """The (vgpr, lane) slots some v_readlane_b32 fetches a mask from, given analyse()'s result."""
    found = set()
    for blk in blocks:
        for ins in blk:
            a = ins["args"]
            if base_op(ins["op"]) == "v_readlane_b32" and len(a) == 3:
                slot = lane_slot(a[1], a[2])
                if slot in masks.get(id(ins), ()):
                    found.add(slot[1:])
    return sorted(found)


def analyse(blocks, succ, *, include_either=False, wave64=True):
    n_blocks = len(blocks)
    entry = [None] * n_blocks                           # None: not reached yet
    if n_blocks:
        entry[0] = {}
    use_fast = step is _DEFAULT_STEP
    if n_blocks > 1 or (n_blocks == 1 and succ[0]):
        work = [0]
        while work:
            n = work.pop()
            sn = succ[n]
            if not sn:
                continue
            state = dict(entry[n])
            if use_fast:
                vkeys = {k for k in state if isinstance(k, tuple)}
                for ins in blocks[n]:
                    _step_tracked(state, ins, vkeys, wave64)
            else:
                for ins in blocks[n]:
                    step(state, ins, wave64)
            for s in sn:
                es = entry[s]
                if es is None:
                    merged = dict(state)
                else:                                   # a mask on both paths, or a mask on one and 0 / -1 on the other
                    merged = {}
                    for r in es.keys() & state.keys():
                        a_k, b_k = es[r], state[r]
                        if a_k == b_k:
                            merged[r] = a_k
                        elif a_k == EITHER:
                            merged[r] = b_k
                        elif b_k == EITHER:
                            merged[r] = a_k
                        else:
                            merged[r] = MASK
                if merged != es:
                    entry[s] = merged
                    work.append(s)
    out, writes, either = {}, set(), {}
    for n, blk in enumerate(blocks):
        state = dict(entry[n] or {})
        if use_fast:
            vkeys = {k for k in state if isinstance(k, tuple)}
            mask_snap = None
            either_snap = None
            for ins in blk:
                if mask_snap is None:
                    mask_snap = {r: k for r, k in state.items() if k in MASK_KINDS}
                ins_id = id(ins)
                out[ins_id] = mask_snap
                if include_either:
                    if either_snap is None:
                        either_snap = {r for r, k in state.items() if k == EITHER}
                    either[ins_id] = either_snap
                wrote, changed = _step_tracked(state, ins, vkeys, wave64)
                if wrote:
                    writes.add(ins_id)
                if changed:
                    mask_snap = None
                    either_snap = None
        else:
            for ins in blk:
                ins_id = id(ins)
                out[ins_id] = {r: k for r, k in state.items() if k in MASK_KINDS}
                if include_either:
                    either[ins_id] = {r for r, k in state.items() if k == EITHER}
                if step(state, ins, wave64):
                    writes.add(ins_id)
    return (out, writes, either) if include_either else (out, writes)
