"""Opt-in x87 leaf-function profile for ffx_version_query.lower(x87=True).

The profile accepts FLD and FSTP with an m32, m64 or m80 operand; FADDP ST(i), ST(0) for i from 1 to 7;
and FMUL with an m32 or m64 operand, FMUL ST(0), ST(i), FMUL ST(i), ST(0) and FMULP ST(i), ST(0) for i
from 1 to 7. A function starts with control word 0x027F, TOP 0, eight empty registers and a clear status
word, and must return with the stack empty. The stack depth before every instruction is fixed at
translation time, so each register maps to a C++ slot and no TOP is needed at run time. Translation
rejects any x87 instruction outside the profile (including control-word and status-word access, and
FADDP or FMULP with ST(0) as the destination), a stack overflow or underflow, a depth that differs where
control flow joins, an x87 instruction it cannot reach from the entry through direct edges, and x87
state that is live at a call, a return, an indirect branch or an external edge. The helpers are in
tools/x87_soft.h, which the generated source must include before the lowered functions (X87_HEADER).
"""
from pathlib import Path

from iced_x86 import FlowControl, MemorySize as S, Mnemonic as M, OpKind as K, Register as R

X87_HEADER = Path(__file__).with_name("x87_soft.h").read_text(encoding="ascii")
X87_OPS = frozenset({M.FLD, M.FSTP, M.FADDP, M.FMUL, M.FMULP})
DECLARATION = "X87Value x87_slot[8]{};"

LOADS = {S.FLOAT32: "x87_from_f32(load<uint32_t>(a))", S.FLOAT64: "x87_from_f64(load<uint64_t>(a))",
         S.FLOAT80: "x87_from_m80(load<uint64_t>(a), load<uint16_t>(a + 8))"}
# An arithmetic memory operand keeps a signaling NaN signaling until the NaN rules have seen it.
OPERANDS = {S.FLOAT32: "x87_operand_f32(load<uint32_t>(a))", S.FLOAT64: "x87_operand_f64(load<uint64_t>(a))"}
# Two-operand register arithmetic: mnemonic -> helper. The P forms pop after writing ST(i).
ARITHMETIC = {M.FADDP: "x87_add", M.FMUL: "x87_mul", M.FMULP: "x87_mul"}


# Every x87 mnemonic in iced starts with F (including FXSAVE/FXRSTOR, which store the x87 state, and
# FEMMS, which empties it); WAIT reports pending x87 exceptions. All of them touch the modelled state.
X87_STATE_OPS = frozenset(value for name, value in vars(M).items() if name.startswith("F")) | {M.WAIT}


def is_x87(i):
    return i.mnemonic in X87_STATE_OPS


def _effect(i, depth):
    """Stack depth after i, or a ValueError for an instruction or operand outside the profile."""
    op = i.mnemonic
    if op in (M.FLD, M.FSTP):
        if i.op0_kind != K.MEMORY or i.memory_size not in LOADS:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if op == M.FLD:
            if depth == 8:
                raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
            return depth + 1
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    if op == M.FMUL and i.op_count == 1:
        if i.op0_kind != K.MEMORY or i.memory_size not in OPERANDS:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op == M.FMUL and i.op_count == 2 and R.ST0 in (i.op0_register, i.op1_register):
        if depth <= max(i.op0_register, i.op1_register) - R.ST0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in (M.FADDP, M.FMULP) and i.op_count == 2 and i.op1_register == R.ST0 and i.op0_register != R.ST0:
        if depth <= i.op0_register - R.ST0 or depth < 2:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    raise ValueError(f"unsupported x87 operation at {i.ip:#x}: {i}")


def depths(body, exits):
    """The x87 stack depth before each instruction of body, from an empty stack at body[0]."""
    index = {i.ip: i for i in body}
    depth = {body[0].ip: 0}
    work = [body[0].ip]
    while work:
        i = index[work.pop()]
        before = depth[i.ip]
        after = _effect(i, before) if is_x87(i) else before
        flow = i.flow_control
        if flow == FlowControl.RETURN and before:
            raise ValueError(f"x87 stack not empty at return {i.ip:#x}")
        if flow in (FlowControl.CALL, FlowControl.INDIRECT_CALL) and before:
            raise ValueError(f"x87 state live across the call at {i.ip:#x}")
        if flow == FlowControl.INDIRECT_BRANCH and before:
            raise ValueError(f"x87 state live at the indirect branch {i.ip:#x}")
        successors = []
        if flow not in (FlowControl.RETURN, FlowControl.UNCONDITIONAL_BRANCH, FlowControl.INDIRECT_BRANCH):
            successors.append(i.next_ip)
        if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
            successors.append(i.near_branch_target)
        for target in successors:
            if target in index:
                if target not in depth:
                    depth[target] = after
                    work.append(target)
                elif depth[target] != after:
                    raise ValueError(f"x87 stack depth {depth[target]} and {after} meet at {target:#x}")
            elif target in exits and after:
                raise ValueError(f"x87 state live at the external edge {i.ip:#x} -> {target:#x}")
    for i in body:
        if i.ip not in depth and is_x87(i):
            raise ValueError(f"unknown x87 stack depth at {i.ip:#x}: {i}")
    return depth


def statement(i, depth, address):
    """C++ for one profile instruction at a known stack depth; address() gives the memory operand."""
    op = i.mnemonic
    _effect(i, depth)
    top = depth - 1  # ST(k) is x87_slot[top - k]
    if op == M.FLD:
        return f"{{ const word a = {address()}; x87_slot[{depth}] = {LOADS[i.memory_size]}; }}"
    if op == M.FSTP:
        v = f"x87_slot[{top}]"
        store = {S.FLOAT32: f"store<uint32_t>(a, x87_to_f32({v}));", S.FLOAT64: f"store<uint64_t>(a, x87_to_f64({v}));",
                 S.FLOAT80: f"store<uint64_t>(a, {v}.sig); store<uint16_t>(a + 8, {v}.se);"}[i.memory_size]
        return f"{{ const word a = {address()}; {store} }}"
    if i.op_count == 1:  # FMUL m32/m64: ST(0) = ST(0) * operand
        return (f"{{ const word a = {address()}; "
                f"x87_slot[{top}] = x87_mul(x87_slot[{top}], {OPERANDS[i.memory_size]}); }}")
    # The destination is the first operand: ST(i) for FMUL ST(i), ST(0) and the P forms, ST(0) otherwise.
    destination, source = (top - (i.op0_register - R.ST0), top - (i.op1_register - R.ST0))
    return f"x87_slot[{destination}] = {ARITHMETIC[op]}(x87_slot[{destination}], x87_slot[{source}]);"
