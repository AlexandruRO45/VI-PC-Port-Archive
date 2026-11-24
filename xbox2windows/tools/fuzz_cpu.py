"""Deterministic differential fuzz smoke test for the shared CPU lifter."""

import argparse
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import random
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


AREAS = {"leaf": (generate, "cpu_fuzz"), "frames": (generate_frames, "cpu_fuzz_frames")}


def inputs_for(seed):
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

    functions = AREAS[area][0](seed, count)
    inputs = inputs_for(seed)
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
                    command = (f"python -B tools/fuzz_cpu.py check {seed:#x} {count} {native} {lifted} {o2} "
                               f"--area {area} --only {function.name} --input {index}")
                    print(f"reproduce: {command}", file=sys.stderr)
                    return 1
    prefix = "" if area == "leaf" else f"area={area} "
    print(f"{prefix}seed={seed:#x} functions={len(selected)} inputs={len(selected_inputs)} "
          f"comparisons={checks} default and /O2 matched")
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
        print(f"mutant {MUTANT_NAME}: {mismatches}/{total} mismatches in {len(failing)}/{len(selected)} functions")
        if not mismatches:
            print("the planted translator fault was not detected", file=sys.stderr)
            return 1
    return 0


def emit(seed, count, asm_path, def_path, area="leaf"):
    generator, library = AREAS[area]
    functions = generator(seed, count)
    asm = ".code\n\n" + "\n\n".join(function.asm() for function in functions) + "\n\nEND\n"
    definition = f"LIBRARY {library}\nEXPORTS\n" + "".join(f"    {function.name}\n" for function in functions)
    Path(asm_path).write_text(asm, encoding="utf-8")
    Path(def_path).write_text(definition, encoding="utf-8")


MUTANT_NAME = "reload_shift"


def plant_reload_shift(cpu_switch_aot):
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


def translate(seed, count, image, output, area="leaf", mutant=False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cpu_switch_aot

    if mutant:
        plant_reload_shift(cpu_switch_aot)
    functions = AREAS[area][0](seed, count)
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
                                  help=f"plant the {MUTANT_NAME} translator fault for the negative control")
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