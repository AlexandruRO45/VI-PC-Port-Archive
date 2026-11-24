"""Deterministic differential fuzz smoke test for the shared CPU lifter."""

import argparse
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import random
import re
import sys

WIDTHS = {
    "b": ("al", "dl", "BYTE PTR", 1),
    "w": ("ax", "dx", "WORD PTR", 2),
    "d": ("eax", "edx", "DWORD PTR", 4),
    "q": ("rax", "rdx", "QWORD PTR", 8),
}
CMP_CONDITIONS = ("e", "ne", "l", "le", "g", "ge", "s", "ns", "o", "no", "p", "np", "b", "ae", "be", "a")
TEST_CONDITIONS = ("e", "ne", "l", "le", "g", "ge", "s", "ns", "p", "np")


@dataclass
class Function:
    name: str
    lines: list[str]
    prolog: list[str] = field(default_factory=list)
    # The functions this one calls or tail-jumps to, in order of first use, printed with it on a mismatch.
    callees: list["Function"] = field(default_factory=list)
    # For a negative control: the reason the translator must give for refusing it. It is assembled
    # and exported but neither translated with the others nor run.
    refused: str | None = None
    # A callee that only other functions of the image reach is not exported, so it is not run on its own.
    export: bool = True
    # Only AL is defined in RAX when it returns, so a caller widens it and no tail jump reaches it.
    narrow: bool = False

    def asm(self):
        head = [f"    {line}" for line in self.prolog]
        body = [f"    {line}" if not line.endswith(":") else line for line in self.lines]
        return "\n".join([f"{self.name} PROC FRAME", *head, "    .endprolog", *body, f"{self.name} ENDP"])


def alu(name, width, variant, rng):
    reg, rhs, ptr, size = WIDTHS[width]
    imm = rng.choice((1, 3, 7, 0x1f, 0x7f, 0xff))
    lines = ["mov rax, rcx"]
    if variant == 0:
        lines += [
            f"add {reg}, {rhs}",
            f"sub {reg}, {imm}",
            f"and {reg}, {rhs}",
            f"xor {reg}, {rng.choice((1, 3, 5, 9))}",
        ]
    else:
        lines += [
            f"or {reg}, {rhs}",
            f"add {reg}, {imm}",
            f"sub {reg}, {rhs}",
            f"xor {reg}, {rng.choice((1, 3, 5, 9))}",
        ]
    lines += [f"mov {ptr} [r8], {reg}", f"shr {reg}, {rng.choice((1, 3, 5))}", f"mov {ptr} [r8+{size}], {reg}", "ret"]
    return Function(name, lines)


def shift(name, width, rng):
    reg, _, ptr, size = WIDTHS[width]
    op = rng.choice(("shl", "shr", "sar", "rol", "ror"))
    count = rng.choice((1, 3, 5, 7))
    lines = ["mov rax, rcx", f"{op} {reg}, {count}", f"mov {ptr} [r8], {reg}", "ret"]
    return Function(name, lines)


def flags(name, width, rng):
    reg, rhs, ptr, _ = WIDTHS[width]
    producer = rng.choice(("cmp", "test"))
    cc = rng.choice(CMP_CONDITIONS if producer == "cmp" else TEST_CONDITIONS)
    label = f"{name}_taken"
    lines = [
        "mov rax, rcx",
        f"{producer} {reg}, {rhs}",
        "mov rax, 0",
        "mov rdx, 0",
        f"set{cc} BYTE PTR [r8]",
        "mov eax, 42",
        "mov r10d, 17",
        f"cmov{cc} eax, r10d",
        "mov DWORD PTR [r8+4], eax",
        f"j{cc} {label}",
        "mov eax, 1",
        "mov DWORD PTR [r8+8], eax",
        "ret",
        f"{label}:",
        "mov eax, 2",
        "mov DWORD PTR [r8+8], eax",
        "ret",
    ]
    return Function(name, lines)


def branch_diamond(name, rng):
    label = f"{name}_even"
    add = rng.choice((1, 3, 7))
    lines = [
        "mov eax, ecx",
        "test eax, 1",
        f"jz {label}",
        f"add eax, {add}",
        "mov DWORD PTR [r8], eax",
        "mov eax, 3",
        "ret",
        f"{label}:",
        "sub eax, edx",
        "mov DWORD PTR [r8], eax",
        "mov eax, 4",
        "ret",
    ]
    return Function(name, lines)


def branch_fallthrough(name, rng):
    label = f"{name}_next"
    add = rng.choice((1, 3, 7))
    lines = [
        "mov eax, ecx",
        "cmp eax, edx",
        f"jne {label}",
        f"{label}:",
        f"add eax, {add}",
        "mov DWORD PTR [r8], eax",
        "ret",
    ]
    return Function(name, lines)


def mixed(name, width, rng):
    reg, rhs, _, _ = WIDTHS[width]
    cc = rng.choice(TEST_CONDITIONS)
    lines = [
        "mov rax, rcx",
        f"test {reg}, {rhs}",
        "mov rax, 0",
        "mov rdx, 0",
        f"set{cc} BYTE PTR [r8]",
        "mov eax, 123456789",
        "mov r10d, 987654321",
        f"cmov{cc} eax, r10d",
        "mov DWORD PTR [r8+4], eax",
        "ret",
    ]
    return Function(name, lines)


def generate(seed, count):
    rng = random.Random(seed)
    recipes = [
        ("alu", "b", 0), ("alu", "b", 1),
        ("alu", "w", 0), ("alu", "w", 1),
        ("alu", "d", 0), ("alu", "d", 1),
        ("alu", "q", 0), ("alu", "q", 1),
        ("shift", "d", 0), ("shift", "d", 1),
        ("shift", "q", 0), ("shift", "q", 1),
        ("flags", "b", 0), ("flags", "w", 0),
        ("flags", "d", 0), ("flags", "q", 0),
        ("branch_diamond", None, None),
        ("branch_fallthrough", None, None),
        ("mixed", "b", 0), ("mixed", "d", 0),
    ]
    functions = []
    for index in range(count):
        kind, width, variant = recipes[index % len(recipes)]
        name = f"fuzz_{index}"
        if kind == "alu":
            functions.append(alu(name, width, variant, rng))
        elif kind == "shift":
            functions.append(shift(name, width, rng))
        elif kind == "flags":
            functions.append(flags(name, width, rng))
        elif kind == "branch_diamond":
            functions.append(branch_diamond(name, rng))
        elif kind == "branch_fallthrough":
            functions.append(branch_fallthrough(name, rng))
        else:
            functions.append(mixed(name, width, rng))
    return functions


REGISTER_PARTS = {
    "rax": ("eax", "ax", "al"), "rcx": ("ecx", "cx", "cl"), "rdx": ("edx", "dx", "dl"),
    "rbx": ("ebx", "bx", "bl"), "rsi": ("esi", "si", "sil"), "rdi": ("edi", "di", "dil"),
    **{f"r{n}": (f"r{n}d", f"r{n}w", f"r{n}b") for n in range(9, 16)},
}
PTR = {8: "QWORD PTR", 4: "DWORD PTR", 2: "WORD PTR", 1: "BYTE PTR"}
NONVOLATILE = ("rbx", "rsi", "rdi", "r12", "r13", "r14", "r15")
# The partitions of a 16-byte spill region into consecutive stores, so stores of every width
# meet at unaligned boundaries and a reload can straddle two or three of them.
SPILL_WIDTHS = (8, 4, 2, 1)


def part(register, width):
    return register if width == 8 else REGISTER_PARTS[register][{4: 0, 2: 1, 1: 2}[width]]


def partition(rng, total):
    widths = []
    while total:
        width = rng.choice([width for width in SPILL_WIDTHS if width <= total])
        widths.append(width)
        total -= width
    return widths


def frame_function(name, rng):
    """A fixed RSP frame: saved nonvolatile registers, a 16-byte spill region and partial reloads.

    Each saved register is overwritten from the arguments before it is read, so the function never
    reads a nonvolatile register's entry value, which cpu_switch_aot rejects.
    """
    prolog, lines = [], []
    home = rng.random() < 0.3
    if home:
        # The MSVC /O2 shape: RBX and RSI saved into home space before the allocation, restored by MOV.
        pushed, moved = ["rdi"], ["rbx", "rsi"]
        prolog += ["mov QWORD PTR [rsp+8], rbx", "mov QWORD PTR [rsp+10h], rsi"]
    else:
        pushed, moved = rng.sample(NONVOLATILE, rng.choice((1, 2, 3))), []
    for register in pushed:
        prolog += [f"push {register}", f".pushreg {register}"]
    # Entry RSP is 8 mod 16; keep RSP 16-byte aligned after the allocation.
    size = rng.choice((16, 32)) + (8 if len(pushed) % 2 == 0 else 0)
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    for index, register in enumerate(moved):
        prolog.append(f".savereg {register}, {size + 8 * len(pushed) + 8 * (index + 1)}")

    saved = pushed + moved
    for register in saved:
        lines.append(rng.choice((f"mov {register}, rcx", f"mov {register}, rdx",
                                 f"lea {register}, [rcx+rdx*2]", f"lea {register}, [rdx+rcx*4+7]")))
        lines.append(rng.choice((f"xor {register}, rdx", f"rol {register}, 13", f"add {register}, rcx")))
    sources = ["rcx", "rdx", *saved]
    base = rng.choice([offset for offset in (0, 8) if offset + 16 <= size])

    offset, boundaries = base, []
    for width in partition(rng, 16):
        if offset != base:
            boundaries.append(offset)
        if rng.random() < 0.2:
            lines.append(f"mov {PTR[width]} [rsp+{offset}], {rng.choice((0x5A, 0x7F, 0x81, 0x33))}")
        else:
            lines.append(f"mov {PTR[width]} [rsp+{offset}], {part(rng.choice(sources), width)}")
        offset += width

    if rng.random() < 0.5:
        width = rng.choice((1, 2, 4, 8))
        at = base + rng.randrange(16 - width + 1)
        lines.append(f"add {PTR[width]} [rsp+{at}], {part(rng.choice(('rdx', *saved)), width)}")
    if rng.random() < 0.5:
        label, join = f"{name}_odd", f"{name}_join"
        lines += [
            "test cl, 1",
            f"jnz {label}",
            f"mov BYTE PTR [rsp+{base + rng.randrange(16)}], 11h",
            f"jmp {join}",
            f"{label}:",
            f"mov WORD PTR [rsp+{base + rng.randrange(15)}], dx",
            f"{join}:",
        ]

    lines.append("mov rax, rcx")
    for slot in range(rng.choice((4, 5, 6))):
        if slot == 0:
            # At least one reload crosses a boundary between two of the stores above.
            boundary = rng.choice(boundaries)
            width = rng.choice((2, 4, 8))
            at = rng.randrange(max(base, boundary - width + 1), min(boundary - 1, base + 16 - width) + 1)
        else:
            width = rng.choice((1, 2, 4, 8))
            at = base + rng.randrange(16 - width + 1)
        signed = rng.random() < 0.5
        if width == 8:
            lines.append(f"mov r10, QWORD PTR [rsp+{at}]")
        elif width == 4:
            lines.append(f"movsxd r10, DWORD PTR [rsp+{at}]" if signed else f"mov r10d, DWORD PTR [rsp+{at}]")
        else:
            lines.append(f"{'movsx r10' if signed else 'movzx r10d'}, {PTR[width]} [rsp+{at}]")
        lines += [f"mov QWORD PTR [r8+{8 * slot}], r10", rng.choice(("xor rax, r10", "add rax, r10", "sub rax, r10")),
                  "rol rax, 7"]

    for index, register in enumerate(moved):
        lines.append(f"mov {register}, QWORD PTR [rsp+{size + 8 * len(pushed) + 8 * (index + 1)}]")
    lines.append(f"add rsp, {size}")
    lines += [f"pop {register}" for register in reversed(pushed)]
    lines.append("ret")
    return Function(name, lines, prolog)


def generate_frames(seed, count):
    rng = random.Random(seed)
    return [frame_function(f"fuzz_frame_{index}", rng) for index in range(count)]


# Each call writes its callee's output into its own 48-byte segment of the buffer (a frame function
# writes at most six words); the caller stores RAX after each call from byte 192 on.
CALL_SEGMENT = 48
CALL_RESULTS = 4 * CALL_SEGMENT


def clobber_function(name):
    """A callee that overwrites every volatile general-purpose register (RAX, RCX, RDX and R8 to R11)
    with known values derived from its arguments.

    It writes each value to its output segment before the next one replaces it, so the clobbers are
    observed through memory and RAX, the only results a caller may read after the call.
    """
    lines = [
        "lea rax, [rcx+rdx*2]",
        "lea r9, [rcx+11h]",
        "lea r10, [rdx+2222h]",
        "lea r11, [rcx+rdx+333h]",
        "mov QWORD PTR [r8], r9",
        "mov QWORD PTR [r8+8], r10",
        "mov QWORD PTR [r8+16], r11",
        "mov rcx, r10",
        "xor rcx, rax",
        "mov rdx, r11",
        "sub rdx, r9",
        "mov QWORD PTR [r8+24], rcx",
        "mov QWORD PTR [r8+32], rdx",
        "mov r8, rax",
        "ret",
    ]
    return Function(name, lines)


def clobber_read(name, clobber, register):
    """A negative control: a caller that reads a volatile register after the call, which the
    translator must refuse because the callee may have overwritten it."""
    prolog = ["push rbx", ".pushreg rbx", "sub rsp, 32", ".allocstack 32"]
    lines = ["mov rbx, r8", f"call {clobber.name}", f"mov QWORD PTR [rbx+{CALL_RESULTS}], rax",
             f"mov rax, {register}", "add rsp, 32", "pop rbx", "ret"]
    return Function(name, lines, prolog, [clobber], refused=f"undefined register {register}")


def caller_function(name, callees, first, rng, displacements):
    """A fixed RSP frame that makes one to three direct calls and may end with a tail jump.

    Every callee may overwrite the volatile registers, and a frame callee overwrites the nonvolatile
    registers it saves, so the caller keeps its state in three saved registers and defines RCX, RDX
    and R8 again before each call. After a call it reads only RAX: cpu_switch_aot treats the other
    volatile registers and the flags as undefined there. The displacement added for RDX is unique in
    the program, so callers that share a callee pass it different arguments.
    """
    output, left, right = rng.sample(NONVOLATILE, 3)
    # Three pushes and a 32- or 48-byte allocation leave RSP 8 mod 16 from entry with the callee's
    # home space at RSP, below the return address.
    size = rng.choice((32, 48))
    prolog = []
    for register in (output, left, right):
        prolog += [f"push {register}", f".pushreg {register}"]
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    lines = [f"mov {output}, r8", f"mov {left}, rcx", f"mov {right}, rdx"]
    # A spill above the callee's home space is kept across the calls.
    spill = size == 48 and rng.random() < 0.5
    if spill:
        lines.append(f"mov QWORD PTR [rsp+32], {right}")

    def arguments(index, after_call):
        displacement = rng.randrange(1, 0x8000)
        while displacement in displacements:
            displacement = rng.randrange(1, 0x8000)
        displacements.add(displacement)
        sources = [left, right, f"{left}+{right}*2"] + (["rax"] if after_call else [])
        source = rng.choice(sources)
        return [f"lea rcx, [{source}]" if "+" in source else f"mov rcx, {source}",
                f"lea rdx, [{rng.choice((left, right, 'rax') if after_call else (left, right))}+{displacement}]",
                f"lea r8, [{output}+{CALL_SEGMENT * index}]" if index else f"mov r8, {output}"]

    targets = []
    for index in range(rng.choice((1, 2, 3))):
        if index == 0:
            target = first
        elif rng.random() < 0.3:
            target = targets[-1]  # the same callee again, with the previous result as an argument
        else:
            target = rng.choice(callees)
        lines += arguments(index, index > 0)
        lines += [f"call {target.name}", f"mov QWORD PTR [{output}+{CALL_RESULTS + 8 * index}], rax",
                  rng.choice((f"xor {left}, rax", f"add {right}, rax", f"sub {left}, rax")), f"rol {left}, 9"]
        targets.append(target)
    if spill:
        lines += ["mov r10, QWORD PTR [rsp+32]", f"xor {left}, r10"]

    tail = rng.random() < 0.5
    if tail:
        target = rng.choice(callees)
        targets.append(target)
        lines += arguments(len(targets) - 1, True)
    else:
        lines += [f"mov rax, {left}", f"xor rax, {right}"]
    lines.append(f"add rsp, {size}")
    lines += [f"pop {register}" for register in (right, left, output)]
    lines.append(f"jmp {target.name}" if tail else "ret")
    return Function(name, lines, prolog, list({target.name: target for target in targets}.values()))


def generate_calls(seed, count):
    """Callers with direct calls and tail jumps to callees from the leaf and frame generators.

    Half as many callees as callers, alternating leaf and frame functions. Each callee is the first
    call of one or two callers, and later calls and tail jumps reach it again.

    The first two callers instead start by calling fuzz_clobber with different arguments, which
    overwrites every volatile general-purpose register with known values. Two controls read R10 and RCX after
    calling it, and the translator must refuse both.
    """
    rng = random.Random(seed)
    pool = max(2, count // 2)
    leaves = rng.sample(generate(seed ^ 0x1EAF, max(20, pool)), (pool + 1) // 2)
    frames = generate_frames(seed ^ 0xF4A3E, pool // 2)
    callees = [function for pair in zip(leaves, frames + [None]) for function in pair if function]
    clobber = clobber_function("fuzz_clobber")
    displacements = set()
    callers = [caller_function(f"fuzz_call_{index}", callees, clobber if index < 2 else callees[index % pool],
                               rng, displacements)
               for index in range(count)]
    controls = [clobber_read(f"fuzz_clobber_read_{register}", clobber, register) for register in ("r10", "rcx")]
    return [clobber, *callees, *callers, *controls]


# Chain shapes: deeper call chains, skipped calls and unexported callees, beyond the direct calls of
# generate_calls. Each exported function is fn(x, y, out) with x in RCX, y in RDX and the output buffer in R8.
# The callees are not exported: they live in a pool that every exported function shares, so one callee
# is reached by several callers, by calls at different depths and by tail jumps. Callee i only calls a
# callee below it, so the pool has no cycles.
CALLEE_KINDS = ("mix", "branch", "narrow", "store", "dword", "mid_registers", "mid_spill")
# Output words 0 to 7 belong to the exported functions. A storing callee writes its own word above them.
CALLEE_STORE_OFFSET = 0x40
MAX_CHAIN_CALLS = 4


def reached(targets):
    """The targets and everything they call, in order of first use, for the failure report."""
    found = {}
    for target in targets:
        for function in (target, *target.callees):
            found.setdefault(function.name, function)
    return list(found.values())


def callee_leaf(name, kind, index, rng):
    imm = rng.choice((1, 5, 0x21, 0x7f))
    count = rng.choice((1, 3, 5, 9))
    if kind == "mix":
        lines = ["lea rax, [rcx+rcx*2]", "add rax, rdx", f"xor rax, {imm}", f"rol rax, {count}", "ret"]
    elif kind == "branch":
        label = f"{name}_below"
        lines = ["cmp rcx, rdx", f"jb {label}", "mov rax, rcx", "sub rax, rdx", f"add rax, {imm}", "ret",
                 f"{label}:", "mov rax, rdx", "sub rax, rcx", f"xor rax, {imm}", "ret"]
    elif kind == "narrow":
        # Only AL is defined at the RET, so every caller must widen it before use.
        lines = ["mov al, cl", "add al, dl", f"xor al, {imm}", "ret"]
    elif kind == "store":
        lines = ["mov rax, rcx", "add rax, rdx", f"mov QWORD PTR [r8+{CALLEE_STORE_OFFSET + 8 * index}], rax",
                 f"xor rax, {imm}", "ret"]
    else:
        # Writing EAX clears the upper half of RAX.
        lines = ["lea eax, [rcx+rdx*4]", f"rol eax, {count}", "ret"]
    return Function(name, lines, narrow=kind == "narrow", export=False)


def call_lines(callee):
    """A call followed by the widening that a byte-sized return needs."""
    return [f"call {callee.name}", *(["movzx eax, al"] if callee.narrow else [])]


def callee_mid(name, kind, pool, rng):
    """A callee with its own frame that calls one or two callees below it and combines the results."""
    targets = [rng.choice(pool) for _ in range(rng.choice((1, 2)))]
    if kind == "mid_registers":
        pushed, size, keep, spill = ["rbx", "rsi"], 0x28, "rbx", None
    else:
        pushed, size, keep, spill = ["rsi"], 0x30, None, "[rsp+20h]"
    prolog = []
    for register in pushed:
        prolog += [f"push {register}", f".pushreg {register}"]
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    lines = [f"mov {keep}, rdx" if keep else f"mov QWORD PTR {spill}, rdx", "mov rsi, r8"]
    held = keep or "r10"
    for position, callee in enumerate(targets):
        if position == 0:
            lines.append(rng.choice(("xor rcx, rdx", "lea rcx, [rcx+rdx*2]", "add rcx, rdx")))
        else:
            lines += ["mov rcx, rax", f"mov rdx, {keep}" if keep else f"mov rdx, QWORD PTR {spill}"]
        lines += ["mov r8, rsi", *call_lines(callee)]
        if keep is None:
            lines.append(f"mov r10, QWORD PTR {spill}")
        lines.append(rng.choice((f"add rax, {held}", f"xor rax, {held}", f"sub rax, {held}")))
    lines += [f"add rsp, {size}", *[f"pop {register}" for register in reversed(pushed)], "ret"]
    return Function(name, lines, prolog, reached(targets), export=False)


def callee_pool(rng):
    pool = []
    for index, kind in enumerate(CALLEE_KINDS):
        name = f"fuzz_callee_{index}"
        if kind.startswith("mid"):
            pool.append(callee_mid(name, kind, pool, rng))
        else:
            pool.append(callee_leaf(name, kind, index, rng))
    return pool


def argument_setup(rng, held_x, held_y, first):
    """RCX and RDX for the next call: from the previous result in RAX, or from the saved arguments."""
    if first:
        rcx = rng.choice((f"mov rcx, {held_x}", f"lea rcx, [{held_x}+7]", f"lea rcx, [{held_x}+{held_x}*2]"))
    else:
        rcx = rng.choice(("mov rcx, rax", f"lea rcx, [rax+{held_x}]", "lea rcx, [rax+rax*4]"))
    rdx = [f"mov rdx, {held_y}"]
    if rng.random() < 0.5:
        rdx.append("add rdx, 11")
    return [rcx, *rdx]


def call_chain(name, pool, rng, conditional):
    """Two to four calls from a frame that keeps the output pointer, x and y across them."""
    registers = rng.sample(NONVOLATILE, 3)
    use_registers = rng.random() < 0.6
    pushed = registers if use_registers else registers[:2]
    out, held_x = pushed[0], pushed[1]
    held_y = pushed[2] if use_registers else "QWORD PTR [rsp+20h]"
    size = 0x20 if use_registers else 0x28
    prolog = []
    for register in pushed:
        prolog += [f"push {register}", f".pushreg {register}"]
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    lines = [f"mov {out}, r8", f"mov {held_x}, rcx", f"mov {held_y}, rdx"]
    calls = rng.randint(2, MAX_CHAIN_CALLS)
    used = []
    for position in range(calls):
        callee = rng.choice(pool)
        used.append(callee)
        body = [*argument_setup(rng, held_x, held_y, position == 0), f"mov r8, {out}", *call_lines(callee),
                f"mov QWORD PTR [{out}+{8 * position}], rax",
                rng.choice((f"add {held_x}, rax", f"xor {held_x}, rax", f"lea {held_x}, [{held_x}+rax*2]"))]
        if conditional and position > 0 and rng.random() < 0.6:
            skip = f"{name}_skip{position}"
            body = [f"test {part(held_x, 1)}, 1", f"jz {skip}", *body, f"{skip}:"]
        lines += body
    lines += [f"mov rax, {held_x}", f"add rsp, {size}", *[f"pop {register}" for register in reversed(pushed)], "ret"]
    return Function(name, lines, prolog, reached(used))


def tail_function(name, pool, rng, shape):
    """A function that ends in a jump to another function's entry instead of a RET."""
    tails = [callee for callee in pool if not callee.narrow]
    first, second = rng.choice(tails), rng.choice(tails)
    imm = rng.choice((3, 5, 9, 0x10))
    if shape == "plain":
        lines = [f"add rcx, {imm}", f"xor rdx, {imm + 1}", f"jmp {first.name}"]
        return Function(name, lines, callees=reached([first]))
    if shape == "frame":
        # The saved register is popped before the jump, so the callee returns straight to the caller.
        lines = ["mov rbx, rcx", "mov QWORD PTR [r8], rbx", f"lea rcx, [rbx+{imm}]", "xor rdx, rbx", "pop rbx",
                 f"jmp {first.name}"]
        return Function(name, lines, ["push rbx", ".pushreg rbx"], reached([first]))
    odd = f"{name}_odd"
    lines = ["test cl, 1", f"jnz {odd}", f"add rcx, {imm}", f"jmp {first.name}",
             f"{odd}:", "lea rdx, [rdx+rcx*2]", f"jmp {second.name}"]
    return Function(name, lines, callees=reached([first, second]))


def generate_chains(seed, count):
    """Exports that call and jump to a pool of unexported callees that nest two levels deep."""
    rng = random.Random(seed)
    pool = callee_pool(rng)
    recipes = (("chain", False), ("tail", "plain"), ("chain", True), ("tail", "frame"), ("chain", True),
               ("tail", "arms"))
    functions = []
    for index in range(count):
        kind, option = recipes[index % len(recipes)]
        name = f"fuzz_call_{index}"
        functions.append(call_chain(name, pool, rng, option) if kind == "chain" else tail_function(name, pool, rng, option))
    return [*pool, *functions]


# Switch shapes: a jump table behind an unsigned bound check, in the layouts cpu_switch_aot accepts (see
# tests/cpu/switch.asm and tests/cpu/switch_inverted.asm). Each export is fn(selector, y, out): the
# selector in RCX picks an arm that rewrites RDX, every arm writes a marker word of its own, and the common
# tail stores RDX at the start of the output and returns the selector plus RDX. Every combination of table
# layout, bound branch and index source appears once in 48 exports.
SWITCH_COUNTS = (2, 3, 4, 5, 6, 8, 9, 12, 16, 24)
SWITCH_LAYOUTS = ("rva", "rel")
# JA and JAE branch to the default; JBE and JB branch to the dispatch block, which a JMP to the default precedes.
SWITCH_BOUNDS = ("ja", "jae", "jbe", "jb")
SWITCH_SOURCES = ("plain", "wide", "mask", "byte", "word", "floor")
SWITCH_MARKERS = 8
# Selectors 0 to 39 reach every arm, and the default, of the largest table.
SWITCH_SELECTORS = 40
# Negative controls: the shape that the translator must refuse, and the reason it must give.
SWITCH_CONTROLS = {
    "unbounded": "unbounded jump-table index",
    "signed": "jump-table bound is not an unsigned upper limit",
    "unextended": "is not the bounded value",
    "base": "is not a RIP-relative address",
}


def arm_operations(rng, wide, count):
    """Operations on RDX that read only RDX and the selector in RAX, so that no arm depends on the
    registers that the dispatch sequence overwrites."""
    d, a = ("rdx", "rax") if wide else ("edx", "eax")
    imm = lambda: rng.choice((1, 3, 7, 17, 127, 23130))
    shifts = ("shl", "shr", "sar", "rol")
    choices = (lambda: f"add {d}, {imm()}", lambda: f"sub {d}, {imm()}", lambda: f"xor {d}, {imm()}",
               lambda: f"imul {d}, {d}, {rng.choice((3, 5, 7, 11))}",
               lambda: f"{rng.choice(shifts)} {d}, {rng.choice((1, 3, 5))}", lambda: f"neg {d}", lambda: f"not {d}",
               lambda: f"lea {d}, [rdx+rdx*{rng.choice((2, 4, 8))}]",
               lambda: f"lea {d}, [rdx+rax*{rng.choice((2, 4))}+{imm()}]",
               lambda: f"{rng.choice(('add', 'xor', 'sub'))} {d}, {a}")
    return [rng.choice(choices)() for _ in range(count)]


def switch_entries(rng, count):
    """The arm of each table entry and the order the arms are laid out in: every entry its own arm in
    order, runs of entries that share an arm in order, or shared arms with entries and arms shuffled."""
    kind = rng.choice(("distinct", "grouped", "shuffled") if count > 2 else ("distinct", "shuffled"))
    if kind == "distinct":
        return list(range(count)), list(range(count))
    arms = rng.randint(max(2, count // 2), count - 1 if kind == "grouped" else count)
    entries = list(range(arms)) + [rng.randrange(arms) for _ in range(count - arms)]
    layout = list(range(arms))
    if kind == "grouped":
        return sorted(entries), layout
    rng.shuffle(entries)
    rng.shuffle(layout)
    return entries, layout


def switch_function(name, rng, layout, bound, source):
    count = rng.choice(SWITCH_COUNTS)
    wide = rng.random() < 0.3
    d, a, ptr = ("rdx", "rax", "QWORD PTR") if wide else ("edx", "eax", "DWORD PTR")
    entries, order = switch_entries(rng, count)
    arms = max(entries) + 1
    # JA and JBE admit selectors up to the compared limit, JAE and JB stop one short of it.
    limit = count - 1 if bound in ("ja", "jbe") else count
    default, store, table = f"{name}_default", f"{name}_store", f"{name}_table"
    epilogue = [f"mov {ptr} [r8], {d}", f"add {a}, {d}", "ret"]

    def block(label, slot, tail):
        lines = [f"{label}:", f"mov DWORD PTR [r8+{SWITCH_MARKERS + 4 * slot}], {0x100 + slot}",
                 *arm_operations(rng, wide, rng.choice((1, 2)))]
        return lines + {"jump": [f"jmp {store}"], "fall": [], "ret": epilogue, "const": [f"mov {a}, -1", "ret"]}[tail]

    # What the selector becomes before the bound check, the register that is compared, and which
    # 32-bit register the dispatch block copies the index from.
    mask = (1 << count.bit_length()) - 1
    prepare, compared = {
        "plain": (["mov eax, ecx"], "ecx"), "wide": (["mov eax, ecx"], "rcx"),
        "mask": (["mov eax, ecx", f"and eax, {mask}"], "eax"), "byte": (["movzx eax, cl"], "eax"),
        "word": (["movzx eax, cx"], "eax"),
        "floor": (["mov eax, ecx", "xor r11d, r11d", "test eax, eax", "cmovs eax, r11d"], "eax"),
    }[source]
    hoisted = []
    if layout == "rva":
        # MSVC: image-relative entries, loaded through a register that holds the image base.
        index = rng.choice((f"movsxd r9, {'ecx' if source in ('plain', 'wide') else 'eax'}",
                            f"mov r9d, {'ecx' if source in ('plain', 'wide') else 'eax'}"))
        if rng.random() < 0.25:
            hoisted, dispatch = ["lea r10, __ImageBase"], [index]
        else:
            dispatch = rng.sample([index, "lea r10, __ImageBase"], 2)
        dispatch += [f"mov r9d, DWORD PTR [(IMAGEREL {table}) + r10 + r9*4]", "add r9, r10", "jmp r9"]
        table_lines = [f"{table} LABEL DWORD", *(f"DD IMAGEREL {name}_arm{arm}" for arm in entries)]
    else:
        # LLVM: signed offsets from the table, whose address is added back. A 64-bit compare of RCX
        # needs no copy; a 32-bit one zero-extends the compared value into RCX first.
        base = rng.choice(("r9", "r10"))
        dispatch = [*([] if source == "wide" else ["mov ecx, eax"]), f"lea {base}, {table}"]
        rng.shuffle(dispatch)
        dispatch += [f"movsxd rcx, DWORD PTR [{base}+rcx*4]", f"add rcx, {base}", "jmp rcx"]
        table_lines = [f"{table}:", *(f"DD {name}_arm{arm} - {table}" for arm in entries)]
    check = [f"cmp {compared}, {limit}"]
    if bound in ("jbe", "jb"):
        check += [f"{bound} {name}_in_range", f"jmp {default}", f"{name}_in_range:"]
    else:
        check.append(f"{bound} {default}")

    guard = rng.random() < 0.25
    default_tail = rng.choice(("jump", "jump", "ret", "const"))
    default_lines = block(default, arms, default_tail)
    lines = [*hoisted]
    if guard:
        # The default in front of the check: the bound branch goes backward.
        lines += [f"jmp {name}_guard", *default_lines, f"{name}_guard:"]
    lines += [*prepare, *check, *dispatch]
    for position, arm in enumerate(order):
        last = position == len(order) - 1
        tail = "fall" if not last and rng.random() < 0.25 else rng.choice(("jump",) * 4 + ("ret",))
        arm_lines = block(f"{name}_arm{arm}", arm, tail)
        if guard and last and tail == "jump":
            arm_lines.pop()  # falls into the common tail
        lines += arm_lines
    if not guard:
        # The default after the arms; as the last block it falls into the common tail.
        lines += default_lines[:-1] if default_tail == "jump" else default_lines
    return Function(name, [*lines, f"{store}:", *epilogue, "ALIGN 4", *table_lines])


def switch_control(name, kind):
    """A negative control: a two-entry LLVM-layout switch that breaks one rule the translator checks."""
    check = {"unbounded": [], "signed": ["cmp ecx, 1", f"jg {name}_default"]}.get(kind, ["cmp ecx, 1", f"ja {name}_default"])
    copy = [] if kind in ("unbounded", "unextended") else ["mov ecx, eax"]
    base = "mov r9, rdx" if kind == "base" else f"lea r9, {name}_table"
    lines = ["mov eax, ecx", *check, *copy, base, "movsxd rcx, DWORD PTR [r9+rcx*4]", "add rcx, r9", "jmp rcx",
             f"{name}_arm0:", "add edx, 1", f"jmp {name}_store", f"{name}_arm1:", "xor edx, 5", f"jmp {name}_store",
             f"{name}_default:", "not edx", f"{name}_store:", "mov DWORD PTR [r8], edx", "add eax, edx", "ret",
             "ALIGN 4", f"{name}_table:", f"DD {name}_arm0 - {name}_table", f"DD {name}_arm1 - {name}_table"]
    return Function(name, lines, refused=SWITCH_CONTROLS[kind])


def generate_switches(seed, count):
    """Exports that switch on the selector in RCX through a jump table, then four refused controls.

    The first 48 exports take the 48 combinations of layout, bound branch and index source in a
    seeded order; later ones repeat them with other tables.
    """
    rng = random.Random(seed)
    shapes = [(layout, bound, source) for layout in SWITCH_LAYOUTS for bound in SWITCH_BOUNDS
              for source in SWITCH_SOURCES]
    rng.shuffle(shapes)
    functions = [switch_function(f"fuzz_switch_{index}", rng, *shapes[index % len(shapes)]) for index in range(count)]
    return [*functions, *(switch_control(f"fuzz_switch_refused_{kind}", kind) for kind in SWITCH_CONTROLS)]


AREAS = {"leaf": (generate, "cpu_fuzz"), "frames": (generate_frames, "cpu_fuzz_frames"),
         "calls": (generate_calls, "cpu_fuzz_calls"), "chains": (generate_chains, "cpu_fuzz_chains"),
         "switches": (generate_switches, "cpu_fuzz_switches")}


def inputs_for(seed, area="leaf"):
    boundary = [
        0, 1, 2, 3, 7, 0x0F, 0x10, 0x1F, 0x20, 0x3F, 0x40, 0x7F, 0x80, 0x81, 0xFF, 0x100,
        0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100000000,
        0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF, 0xDEADBEEF, 0x12345678,
        0x0FEDCBA9, 0xA5A5A5A5A5A5A5A5, 0x5A5A5A5A5A5A5A5A,
    ]
    values = [(value, 0) for value in boundary] + [(0, value) for value in boundary] + [(value, value) for value in boundary]
    rng = random.Random(seed ^ 0x5EED)
    while len(values) < 313:
        values.append((rng.getrandbits(64), rng.getrandbits(64)))
    if area == "switches":
        # Selectors that reach every arm and the default, then the same selectors with upper bits set: a
        # 32-bit compare ignores them, a 64-bit compare does not, and a signed clamp sees bit 31.
        rng = random.Random(seed ^ 0x5E1EC7)
        values = ([(selector, rng.getrandbits(64)) for selector in range(SWITCH_SELECTORS)]
                  + [(upper | selector, rng.getrandbits(64)) for selector in range(4)
                     for upper in (1 << 32, 0x80000000, 0xFFFFFFFF00000000)] + values)
    return values[:313]


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def buffer_text(data, start, stop):
    return " ".join(f"{byte:02x}" for byte in data[start:stop])


def run_check(seed, count, native, lifted, o2, only=None, input_index=None, area="leaf", mutant=None):
    if first_difference(b"\x01\x02", b"\x01\x02") is not None:
        raise SystemExit("comparator self-test failed on equal bytes")
    if first_difference(b"\x01\x02", b"\x01\x03") != 1:
        raise SystemExit("comparator self-test failed on a changed byte")
    if mutant is not None and area not in MUTANTS:
        raise SystemExit(f"no planted translator fault for --area {area}")

    functions = AREAS[area][0](seed, count)
    controls = [function for function in functions if function.refused]
    functions = [function for function in functions if not function.refused and function.export]
    inputs = inputs_for(seed, area)
    libraries = [(native, ""), (lifted, "lifted_"), (o2, "lifted_")]
    if mutant is not None:
        libraries.append((mutant, "lifted_"))
    loaded = []
    for path, prefix in libraries:
        library = ctypes.CDLL(str(path))
        for function in functions:
            symbol = getattr(library, prefix + function.name)
            symbol.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            symbol.restype = ctypes.c_uint64
        loaded.append((library, prefix))

    native_library, lifted_library, o2_library = (item[0] for item in loaded[:3])
    if count <= 0:
        raise SystemExit("CPU_FUZZ_FUNCTIONS must be positive")
    selected = [function for function in functions if only is None or function.name == only]
    selected_inputs = [(index, value) for index, value in enumerate(inputs) if input_index is None or index == input_index]
    if not selected:
        raise SystemExit(f"no generated function matches --only {only!r}")
    if not selected_inputs:
        raise SystemExit(f"no input matches --input {input_index}")
    checks = 0
    for function in selected:
        native_symbol = getattr(native_library, function.name)
        for variant, library in (("default", lifted_library), ("/O2", o2_library)):
            lifted_symbol = getattr(library, "lifted_" + function.name)
            for index, (x, y) in selected_inputs:
                native_buffer = ctypes.create_string_buffer(288)
                lifted_buffer = ctypes.create_string_buffer(288)
                ctypes.memset(native_buffer, 0x5A, len(native_buffer))
                ctypes.memset(lifted_buffer, 0x5A, len(lifted_buffer))
                native_return = native_symbol(x, y, ctypes.addressof(native_buffer) + 16)
                lifted_return = lifted_symbol(x, y, ctypes.addressof(lifted_buffer) + 16)
                checks += 1
                difference = first_difference(native_buffer.raw, lifted_buffer.raw)
                if native_return != lifted_return or difference is not None:
                    print(f"seed={seed:#x} functions={count} function={function.name} input={index} x={x:#x} y={y:#x}", file=sys.stderr)
                    print(f"native return={native_return:#x} {variant} return={lifted_return:#x}", file=sys.stderr)
                    if difference is not None:
                        low = max(0, difference - 8)
                        high = min(len(native_buffer), difference + 8)
                        print(f"first differing byte={difference} (output offset {difference - 16})", file=sys.stderr)
                        print(f"native {buffer_text(native_buffer.raw, low, high)}", file=sys.stderr)
                        print(f"lifted {buffer_text(lifted_buffer.raw, low, high)}", file=sys.stderr)
                    print(function.asm(), file=sys.stderr)
                    for callee in function.callees:
                        print(callee.asm(), file=sys.stderr)
                    command = (f"python -B tools/fuzz_cpu.py check {seed:#x} {count} {native} {lifted} {o2} "
                               f"--area {area} --only {function.name} --input {index}")
                    print(f"reproduce: {command}", file=sys.stderr)
                    return 1
    prefix = "" if area == "leaf" else f"area={area} "
    print(f"{prefix}seed={seed:#x} functions={len(selected)} inputs={len(selected_inputs)} "
          f"comparisons={checks} default and /O2 matched")
    if controls and only is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import cpu_switch_aot

        for control in controls:
            try:
                cpu_switch_aot.translate(Path(native), [control.name])
            except ValueError as error:
                if control.refused not in str(error):
                    print(f"{control.name} refused for another reason: {error}", file=sys.stderr)
                    return 1
            else:
                print(f"{control.name} was translated; expected refusal: {control.refused}", file=sys.stderr)
                print(control.asm(), file=sys.stderr)
                return 1
        print(f"controls refused: {', '.join(f'{control.name} ({control.refused})' for control in controls)}")
    if mutant is not None:
        mutant_library = loaded[3][0]
        mismatches = total = 0
        failing = set()
        for function in selected:
            native_symbol = getattr(native_library, function.name)
            mutant_symbol = getattr(mutant_library, "lifted_" + function.name)
            for _, (x, y) in selected_inputs:
                native_buffer = ctypes.create_string_buffer(288)
                mutant_buffer = ctypes.create_string_buffer(288)
                ctypes.memset(native_buffer, 0x5A, len(native_buffer))
                ctypes.memset(mutant_buffer, 0x5A, len(mutant_buffer))
                native_return = native_symbol(x, y, ctypes.addressof(native_buffer) + 16)
                mutant_return = mutant_symbol(x, y, ctypes.addressof(mutant_buffer) + 16)
                total += 1
                if native_return != mutant_return or native_buffer.raw != mutant_buffer.raw:
                    mismatches += 1
                    failing.add(function.name)
        print(f"mutant {MUTANTS[area][0]}: {mismatches}/{total} mismatches in {len(failing)}/{len(selected)} functions")
        if not mismatches:
            print("the planted translator fault was not detected", file=sys.stderr)
            return 1
    return 0


def emit(seed, count, asm_path, def_path, area="leaf"):
    generator, library = AREAS[area]
    functions = generator(seed, count)
    body = "\n\n".join(function.asm() for function in functions)
    asm = ("EXTERN __ImageBase:BYTE\n" if "__ImageBase" in body else "") + ".code\n\n" + body + "\n\nEND\n"
    definition = (f"LIBRARY {library}\nEXPORTS\n"
                  + "".join(f"    {function.name}\n" for function in functions if function.export))
    Path(asm_path).write_text(asm, encoding="utf-8")
    Path(def_path).write_text(definition, encoding="utf-8")


def plant_reload_shift(cpu_switch_aot, image, functions):
    """Make every frame reload narrower than 8 bytes read one byte higher in the frame array.

    This is a planted translator fault for the negative control: a partial reload at the wrong
    index changes only the value read back, which the return value and output buffer show.
    """
    from iced_x86 import Mnemonic as M, MemorySizeExt, OpKind

    original = cpu_switch_aot.Lifter.slot

    def slot(self, ins):
        text = original(self, ins)
        if (ins.mnemonic in (M.MOV, M.MOVZX, M.MOVSX, M.MOVSXD) and ins.op0_kind == OpKind.REGISTER
                and ins.op1_kind == OpKind.MEMORY and MemorySizeExt.size(ins.memory_size) < 8):
            index = self.frame[ins.ip]
            text = text.replace(f"frame + {index})", f"frame + {index + 1})")
        return text

    cpu_switch_aot.Lifter.slot = slot


def plant_call_retarget(cpu_switch_aot, image, functions):
    """Send the first call of the first caller to the next callee in the generated order.

    This is a planted translator fault for the negative control: the caller's own instructions are
    unchanged, but the wrong callee computes RAX and writes the call's output segment.
    """
    caller = next(function for function in functions if function.callees)
    callees = [function for function in functions if not function.callees]
    wrong = callees[(callees.index(caller.callees[0]) + 1) % len(callees)]
    with cpu_switch_aot.pefile.PE(str(image), fast_load=True) as pe:
        caller_rva = cpu_switch_aot.export_rva(pe, caller.name)
        wrong_rva = cpu_switch_aot.export_rva(pe, wrong.name)
    original = cpu_switch_aot.Program.callee
    planted = []

    def callee(self, target, ip):
        # Callees are resolved in address order, so the first one asked for by the caller is its first call.
        if not planted and self.active[-1] == caller_rva:
            planted.append(ip)
            target = wrong_rva
        return original(self, target, ip)

    cpu_switch_aot.Program.callee = callee


CALL_SITE = re.compile(r"\b(rax = |return )(fn_[0-9a-f]+)\(rcx, rdx, ")


def plant_swapped_arguments(cpu_switch_aot, image, functions):
    """Pass RDX first and RCX second at every direct call and tail jump of the translation.

    This is a planted translator fault for the negative control: every callee that is not
    symmetric in its two arguments returns another value.
    """
    original = cpu_switch_aot.translate

    def translate(*args, **kwargs):
        source, rest = original(*args, **kwargs)
        swapped = CALL_SITE.sub(lambda match: f"{match.group(1)}{match.group(2)}(rdx, rcx, ", source)
        if swapped == source:
            raise SystemExit("the swapped_arguments fault found nothing to change")
        return swapped, rest

    cpu_switch_aot.translate = translate


def plant_table_rotate(cpu_switch_aot, image, functions):
    """Read every jump-table entry one slot late, so each selector reaches the arm of the next entry.

    This is a planted translator fault for the negative control: the bound check and the control flow
    are unchanged, but a selector runs another arm, which the markers, RDX and the return value show.
    """
    original = cpu_switch_aot.Function.dispatch

    def dispatch(self, ip):
        found = original(self, ip)
        index, targets, *rest = self.dispatches[ip]
        self.dispatches[ip] = (index, targets[1:] + targets[:1], *rest)
        return found

    cpu_switch_aot.Function.dispatch = dispatch


MUTANTS = {"frames": ("reload_shift", plant_reload_shift), "calls": ("call_retarget", plant_call_retarget),
           "chains": ("swapped_arguments", plant_swapped_arguments),
           "switches": ("table_rotate", plant_table_rotate)}


def translate(seed, count, image, output, area="leaf", mutant=False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cpu_switch_aot

    functions = AREAS[area][0](seed, count)
    if mutant:
        if area not in MUTANTS:
            raise SystemExit(f"no planted translator fault for --area {area}")
        MUTANTS[area][1](cpu_switch_aot, image, functions)
    # Negative controls stay in the image for the check to translate on their own.
    functions = [function for function in functions if not function.refused and function.export]
    names = [function.name for function in functions]
    try:
        source, _ = cpu_switch_aot.translate(Path(image), names)
    except ValueError as error:
        print(f"seed={seed:#x} functions={count}: {error}", file=sys.stderr)
        for function in functions:
            try:
                cpu_switch_aot.translate(Path(image), [function.name])
            except ValueError as inner:
                print(f"refused {function.name}: {inner}", file=sys.stderr)
                print(function.asm(), file=sys.stderr)
                break
        return 1
    Path(output).write_text(source, encoding="utf-8")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emit_parser = commands.add_parser("emit")
    emit_parser.add_argument("seed", type=lambda value: int(value, 0))
    emit_parser.add_argument("count", type=int)
    emit_parser.add_argument("asm", type=Path)
    emit_parser.add_argument("definition", type=Path)
    emit_parser.add_argument("--area", choices=AREAS, default="leaf")
    translate_parser = commands.add_parser("translate")
    translate_parser.add_argument("seed", type=lambda value: int(value, 0))
    translate_parser.add_argument("count", type=int)
    translate_parser.add_argument("image", type=Path)
    translate_parser.add_argument("output", type=Path)
    translate_parser.add_argument("--area", choices=AREAS, default="leaf")
    translate_parser.add_argument("--mutant", action="store_true",
                                  help="plant the area's translator fault for the negative control: "
                                       + ", ".join(f"{name} for {area}" for area, (name, _) in MUTANTS.items()))
    check_parser = commands.add_parser("check")
    check_parser.add_argument("seed", type=lambda value: int(value, 0))
    check_parser.add_argument("count", type=int)
    check_parser.add_argument("native", type=Path)
    check_parser.add_argument("lifted", type=Path)
    check_parser.add_argument("o2", type=Path)
    check_parser.add_argument("--only")
    check_parser.add_argument("--input", type=int, dest="input_index")
    check_parser.add_argument("--area", choices=AREAS, default="leaf")
    check_parser.add_argument("--mutant", type=Path, help="a translation with the planted fault, which must mismatch")
    args = parser.parse_args(argv)

    if args.command == "emit":
        emit(args.seed, args.count, args.asm, args.definition, args.area)
        return 0
    if args.command == "translate":
        return translate(args.seed, args.count, args.image, args.output, args.area, args.mutant)
    return run_check(args.seed, args.count, args.native, args.lifted, args.o2, args.only, args.input_index,
                     args.area, args.mutant)


if __name__ == "__main__":
    raise SystemExit(main())