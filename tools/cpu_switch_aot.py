"""Recompile bounded Windows x64 leaf functions with jump-table dispatch to C++.

Each export takes three integer words in RCX, RDX and R8 and returns RAX. The
translator follows branches and jump-table entries from the export, so data
placed after the code (as MSVC does with its tables) is never decoded. An
indirect JMP is accepted only at the end of one of these sequences, reached by
the fallthrough of an unsigned bound check on the index:

    mov E32, [B+I*4+table]; add E, B; jmp E   B = image base (MSVC)
    movsxd E, [B+I*4];      add E, B; jmp E   B = table      (LLVM)

B must hold the same RIP-relative address on every path, and the index must be
shown to carry the compared value. The generated C++ replaces the dispatch with
a switch over the table targets.
"""

import argparse
from pathlib import Path
import struct

import pefile
from iced_x86 import (ConditionCode as CC, Decoder, FlowControl, MemorySizeExt, Mnemonic as M, OpKind,
                      Register as R, RegisterExt)

from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges, image_bytes


NAMES = {R.RAX: "rax", R.RCX: "rcx", R.RDX: "rdx", R.RBX: "rbx", R.RSP: "rsp", R.RBP: "rbp",
         R.RSI: "rsi", R.RDI: "rdi", R.R8: "r8", R.R9: "r9", R.R10: "r10", R.R11: "r11",
         R.R12: "r12", R.R13: "r13", R.R14: "r14", R.R15: "r15"}
ARGUMENTS = (R.RCX, R.RDX, R.R8)
VOLATILE = {R.RAX, R.RCX, R.RDX, R.R8, R.R9, R.R10, R.R11}
HIGH_BYTES = {R.AH, R.BH, R.CH, R.DH}
ALU = {M.ADD: "+", M.SUB: "-", M.AND: "&", M.OR: "|", M.XOR: "^"}
SHIFTS = {M.SHL: "<<", M.SHR: ">>", M.SAR: ">>"}
CONDITIONS = {CC.E: "e", CC.NE: "ne", CC.B: "b", CC.AE: "ae", CC.BE: "be", CC.A: "a",
              CC.L: "l", CC.GE: "ge", CC.LE: "le", CC.G: "g", CC.S: "s", CC.NS: "ns"}
MAX_CASES = 1024
MAX_INSTRUCTIONS = 4096


def unsigned(width):
    return f"std::uint{width * 8}_t"


def signed(width):
    return f"std::int{width * 8}_t"


def full(register):
    full_register = RegisterExt.full_register(register)
    if full_register not in NAMES:
        raise ValueError(f"unsupported register {register}")
    return full_register


def runtime_functions(pe):
    table = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    if not table.Size:
        return []
    if table.Size % 12:
        raise ValueError("incomplete runtime-function record")
    return [(begin, end) for begin, end, _ in
            struct.iter_unpack("<III", image_bytes(pe, table.VirtualAddress, table.Size))]


def owned_ranges(pe, entry):
    """Unwind-owned intervals, or the export's section for a leaf without unwind data."""
    records = runtime_functions(pe)
    if any(begin <= entry < end for begin, end in records):
        return function_ranges(pe, entry), []
    # Windows x64 leaf functions may omit .pdata. Such a function may use any
    # file-backed bytes of its section that no runtime-function entry owns.
    section = pe.get_section_by_rva(entry)
    if not section or not section.Characteristics & 0x20000000:
        raise ValueError(f"export at RVA {entry:#x} is not in an executable section")
    end = section.VirtualAddress + min(section.Misc_VirtualSize, section.SizeOfRawData)
    return [(section.VirtualAddress, end)], records


class Function:
    def __init__(self, pe, entry):
        self.pe = pe
        self.entry = entry
        ranges, self.foreign = owned_ranges(pe, entry)
        self.ranges = [(begin, end, image_bytes(pe, begin, end - begin, executable=True))
                       for begin, end in ranges]
        self.instructions = {}
        self.owner = {}
        self.ending = {}
        self.successors = {}
        self.dispatches = {}

    def decode(self, ip):
        for begin, end, data in self.ranges:
            if begin <= ip < end:
                ins = Decoder(64, data[ip - begin:], ip=ip).decode()
                if ins.is_invalid or ins.next_ip > end:
                    raise ValueError(f"invalid or truncated instruction at RVA {ip:#x}")
                if any(ip < stop and start < ins.next_ip for start, stop in self.foreign):
                    raise ValueError(f"edge outside owned ranges at RVA {ip:#x}: another function's unwind interval")
                return ins
        raise ValueError(f"edge outside owned ranges at RVA {ip:#x}")

    def claim(self, begin, end, owner, what):
        for address in range(begin, end):
            if self.owner.get(address, owner) != owner:
                raise ValueError(f"{what} at RVA {begin:#x} overlaps RVA {self.owner[address]:#x}")
        for address in range(begin, end):
            self.owner[address] = owner

    def explore(self):
        pending = [self.entry]
        indirect = []
        while pending or indirect:
            if not pending:
                ip = indirect.pop()
                pending.extend(self.dispatch(ip))
                continue
            ip = pending.pop()
            if ip in self.instructions:
                continue
            if self.owner.get(ip, ip) != ip:
                raise ValueError(f"edge at RVA {ip:#x} overlaps an instruction")
            if len(self.instructions) >= MAX_INSTRUCTIONS:
                raise ValueError(f"instruction limit reached at RVA {ip:#x}")
            ins = self.decode(ip)
            self.claim(ip, ins.next_ip, ip, "instruction")
            self.instructions[ip] = ins
            self.ending[ins.next_ip] = ins
            flow = ins.flow_control
            if flow == FlowControl.NEXT:
                targets = [ins.next_ip]
            elif flow == FlowControl.CONDITIONAL_BRANCH and ins.is_jcc_short_or_near:
                targets = [ins.near_branch_target, ins.next_ip]
            elif flow == FlowControl.UNCONDITIONAL_BRANCH and ins.is_jmp_short_or_near:
                targets = [ins.near_branch_target]
            elif flow == FlowControl.RETURN and ins.mnemonic == M.RET and ins.op_count == 0:
                targets = []
            elif flow == FlowControl.INDIRECT_BRANCH and ins.op0_kind == OpKind.REGISTER:
                targets = []
                indirect.append(ip)
            else:
                raise ValueError(f"unsupported control flow at RVA {ip:#x}: {ins}")
            self.successors[ip] = targets
            pending.extend(reversed(targets))
        edges = {target for ip, targets in self.successors.items() for target in targets
                 if target != self.instructions[ip].next_ip
                 or self.instructions[ip].flow_control not in (FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH)}
        constants = self.constants()
        for ip in self.dispatches:
            self.prove(ip, edges, constants)
        return edges

    def previous(self, ins, what):
        before = self.ending.get(ins.ip)
        if before is None or before.flow_control != FlowControl.NEXT:
            raise ValueError(f"missing {what} before RVA {ins.ip:#x}")
        return before

    def constants(self):
        """RIP-relative addresses that every explored path leaves in a register, per instruction."""
        state = {self.entry: {}}
        work = [self.entry]
        while work:
            ip = work.pop()
            ins = self.instructions.get(ip)
            if ins is None:
                continue
            known = dict(state[ip])
            if ip in self.dispatches:
                known.pop(self.dispatches[ip][4], None)
            elif ins.mnemonic == M.LEA and ins.is_ip_rel_memory_operand and RegisterExt.is_gpr64(ins.op0_register):
                known[ins.op0_register] = ins.ip_rel_memory_address
            elif ins.op_count and ins.op0_kind == OpKind.REGISTER and ins.mnemonic not in (M.CMP, M.TEST):
                known.pop(full(ins.op0_register), None)
            for successor in self.successors.get(ip, ()):
                if successor not in self.instructions:
                    continue
                if successor == self.entry:
                    merged = {}
                elif successor in state:
                    merged = {r: v for r, v in state[successor].items() if known.get(r) == v}
                else:
                    merged = known
                if successor not in state or merged != state[successor]:
                    state[successor] = merged
                    work.append(successor)
        return state

    def dispatch(self, ip):
        jump = self.instructions[ip]
        add = self.previous(jump, "table address add")
        load = self.previous(add, "table entry load")
        target = full(jump.op0_register)
        base = add.op1_register if add.op1_kind == OpKind.REGISTER else None
        if (add.mnemonic != M.ADD or add.op0_kind != OpKind.REGISTER or add.op0_register != target
                or not RegisterExt.is_gpr64(target) or base is None or not RegisterExt.is_gpr64(base)
                or base == target):
            raise ValueError(f"unrecognized indirect jump at RVA {ip:#x}")
        index = load.memory_index
        if (load.op1_kind != OpKind.MEMORY or load.memory_base != base or load.memory_index_scale != 4
                or not RegisterExt.is_gpr64(index) or index == base or load.has_segment_prefix
                or full(load.op0_register) != target):
            raise ValueError(f"unrecognized table load at RVA {load.ip:#x}")
        anchor = self.constants().get(load.ip, {}).get(base)
        if anchor is None:
            raise ValueError(f"table base {NAMES[base]} is not a RIP-relative address at RVA {load.ip:#x}")
        if load.mnemonic == M.MOV and RegisterExt.is_gpr32(load.op0_register):
            # Image-relative entries: the base register holds the image base.
            table, kind = load.memory_displacement, "image"
            if anchor != 0 or not 0 < table < 1 << 31:
                raise ValueError(f"image-relative table without an image base at RVA {load.ip:#x}")
        elif load.mnemonic == M.MOVSXD and load.memory_displacement == 0:
            table, kind = anchor, "relative"
        else:
            raise ValueError(f"unrecognized table load at RVA {load.ip:#x}")

        # The bound check must fall through into the dispatch block.
        bound = load
        between = []
        while True:
            bound = self.ending.get(bound.ip)
            if bound is None:
                raise ValueError(f"unbounded jump-table index at RVA {ip:#x}")
            if bound.flow_control == FlowControl.CONDITIONAL_BRANCH:
                break
            if bound.flow_control != FlowControl.NEXT or len(between) >= 8:
                raise ValueError(f"unbounded jump-table index at RVA {ip:#x}")
            between.append(bound)
        compare = self.ending.get(bound.ip)
        if bound.condition_code not in (CC.A, CC.AE):
            raise ValueError(f"jump-table bound is not an unsigned upper limit at RVA {bound.ip:#x}")
        if (compare is None or compare.mnemonic != M.CMP or compare.op0_kind != OpKind.REGISTER
                or compare.op1_kind not in (OpKind.IMMEDIATE8TO32, OpKind.IMMEDIATE8TO64,
                                            OpKind.IMMEDIATE32, OpKind.IMMEDIATE32TO64)
                or RegisterExt.size(compare.op0_register) not in (4, 8)):
            raise ValueError(f"unbounded jump-table index at RVA {bound.ip:#x}")
        limit = compare.immediate(1) & ((1 << RegisterExt.size(compare.op0_register) * 8) - 1)
        count = limit + 1 if bound.condition_code == CC.A else limit
        if not 0 < count <= MAX_CASES:
            raise ValueError(f"unsupported jump-table size {count} at RVA {compare.ip:#x}")
        entries = struct.unpack(f"<{count}{'I' if kind == 'image' else 'i'}",
                                image_bytes(self.pe, table, 4 * count))
        self.claim(table, table + 4 * count, -table, "jump table")
        targets = [entry if kind == "image" else (table + entry) & 0xffffffff for entry in entries]
        self.dispatches[ip] = (index, targets, (load.ip, add.ip), list(reversed(between)),
                               target, (compare, bound, base, anchor))
        self.successors[ip] = targets
        return list(reversed(targets))

    def prove(self, ip, edges, constants):
        index, _, sequence, window, _, (compare, bound, base, anchor) = self.dispatches[ip]
        if any(address in edges for address in (bound.ip, *(i.ip for i in window), *sequence, ip)):
            raise ValueError(f"second entry into the dispatch block at RVA {ip:#x}")
        if constants.get(sequence[0], {}).get(base) != anchor:
            raise ValueError(f"table base {NAMES[base]} differs on another path to RVA {sequence[0]:#x}")
        # Extend the straight-line window back from the compare while it has a single entry.
        start = compare
        prefix = []
        while start.ip not in edges and start.ip != self.entry and len(prefix) < 16:
            before = self.ending.get(start.ip)
            if before is None or before.flow_control != FlowControl.NEXT:
                break
            prefix.append(before)
            start = before
        values = {}
        counter = iter(range(1 << 30))

        def value(reg):
            if reg not in values:
                values[reg] = (next(counter), "full")
            return values[reg]

        def step(ins):
            if ins.op_count == 0 or ins.op0_kind != OpKind.REGISTER:
                return
            dst = ins.op0_register
            width = RegisterExt.size(dst)
            if ins.mnemonic == M.CMP:
                return
            if ins.mnemonic == M.MOV and ins.op1_kind == OpKind.REGISTER and width in (4, 8):
                number, form = value(full(ins.op1_register))
                values[full(dst)] = (number, form) if width == 8 else (number, "zext32")
            elif (ins.mnemonic == M.MOVSXD and ins.op1_kind == OpKind.REGISTER
                  and RegisterExt.size(ins.op1_register) == 4):
                values[full(dst)] = (value(full(ins.op1_register))[0], "sext32")
            elif ins.mnemonic not in (M.CMP, M.TEST):
                values[full(dst)] = (next(counter), "zext32" if width == 4 else "full")

        for ins in reversed(prefix):
            step(ins)
        bounded, bounded_form = value(full(compare.op0_register))
        compared_width = RegisterExt.size(compare.op0_register)
        for ins in window:
            step(ins)
        number, form = value(index)
        if number != bounded or not (form in ("zext32", "sext32") or (compared_width == 8 and form == bounded_form)):
            raise ValueError(f"jump-table index {NAMES[index]} is not the bounded value at RVA {ip:#x}")


class Lifter:
    def __init__(self):
        self.used = set()
        self.flags = False

    def reg(self, register):
        width = RegisterExt.size(register)
        if register in HIGH_BYTES or width not in (1, 2, 4, 8):
            raise ValueError(f"unsupported register {register}")
        name = NAMES[full(register)]
        self.used.add(full(register))
        return name if width == 8 else f"static_cast<{unsigned(width)}>({name})"

    def address(self, ins):
        if (ins.memory_base == R.RIP or ins.memory_base == R.RSP or ins.memory_index == R.RSP
                or ins.has_segment_prefix or ins.memory_base == R.NONE):
            raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}")
        parts = [self.reg(ins.memory_base)]
        if ins.memory_index != R.NONE:
            if not RegisterExt.is_gpr64(ins.memory_index):
                raise ValueError(f"unsupported memory index at RVA {ins.ip:#x}")
            parts.append(f"{self.reg(ins.memory_index)} * {ins.memory_index_scale}u")
        displacement = ins.memory_displacement & ((1 << 64) - 1)
        if displacement:
            parts.append(f"{displacement:#x}ull")
        return "(" + " + ".join(parts) + ")"

    def operand(self, ins, number, width):
        kind = ins.op_kind(number)
        if kind == OpKind.REGISTER:
            return self.reg(ins.op_register(number))
        if kind == OpKind.MEMORY:
            return f"load<{unsigned(width)}>({self.address(ins)})"
        if kind in (OpKind.IMMEDIATE8, OpKind.IMMEDIATE8TO16, OpKind.IMMEDIATE8TO32, OpKind.IMMEDIATE8TO64,
                    OpKind.IMMEDIATE16, OpKind.IMMEDIATE32, OpKind.IMMEDIATE32TO64, OpKind.IMMEDIATE64):
            return f"static_cast<{unsigned(width)}>({ins.immediate(number) & ((1 << width * 8) - 1):#x}ull)"
        raise ValueError(f"unsupported operand at RVA {ins.ip:#x}")

    def write(self, ins, width, expression):
        if ins.op0_kind == OpKind.MEMORY:
            return f"store<{unsigned(width)}>({self.address(ins)}, static_cast<{unsigned(width)}>({expression}));"
        register = ins.op0_register
        if width not in (4, 8) or RegisterExt.size(register) != width:
            raise ValueError(f"unsupported partial register write at RVA {ins.ip:#x}")
        self.used.add(full(register))
        return f"{NAMES[full(register)]} = static_cast<{unsigned(width)}>({expression});"

    def width(self, ins):
        if ins.op0_kind == OpKind.REGISTER:
            return RegisterExt.size(ins.op0_register)
        if ins.op0_kind == OpKind.MEMORY:
            return _memory_width(ins)
        return 0

    def condition(self, producer, jcc):
        code = CONDITIONS.get(jcc.condition_code)
        width = self.width(producer)
        u, s = unsigned(width), signed(width)
        if producer.mnemonic in (M.CMP, M.SUB):
            a, b = f"static_cast<{u}>(fa)", f"static_cast<{u}>(fb)"
            result = f"static_cast<{s}>(static_cast<{u}>({a} - {b}))"
            table = {"e": f"{a} == {b}", "ne": f"{a} != {b}", "b": f"{a} < {b}", "ae": f"{a} >= {b}",
                     "be": f"{a} <= {b}", "a": f"{a} > {b}", "l": f"static_cast<{s}>({a}) < static_cast<{s}>({b})",
                     "ge": f"static_cast<{s}>({a}) >= static_cast<{s}>({b})",
                     "le": f"static_cast<{s}>({a}) <= static_cast<{s}>({b})",
                     "g": f"static_cast<{s}>({a}) > static_cast<{s}>({b})",
                     "s": f"{result} < 0", "ns": f"{result} >= 0"}
        else:
            result = f"static_cast<{u}>(fa)"
            table = {"e": f"{result} == 0", "ne": f"{result} != 0",
                     "s": f"static_cast<{s}>({result}) < 0", "ns": f"static_cast<{s}>({result}) >= 0"}
            if producer.mnemonic in (M.TEST, M.AND, M.OR, M.XOR):
                # Logical operations clear CF and OF.
                table.update({"b": "false", "ae": "true", "be": f"{result} == 0", "a": f"{result} != 0",
                              "l": table["s"], "ge": table["ns"],
                              "le": f"{result} == 0 || {table['s']}", "g": f"{result} != 0 && {table['ns']}"})
        if code not in table:
            raise ValueError(f"unsupported condition after {producer.mnemonic} at RVA {jcc.ip:#x}")
        return table[code]

    def statement(self, ins, next_ins):
        """Return C++ for one instruction and the registers it reads and writes."""
        mnemonic = ins.mnemonic
        width = self.width(ins)
        reads, writes = set(), set()

        def note(*numbers):
            for number in numbers:
                kind = ins.op_kind(number)
                if kind == OpKind.REGISTER:
                    reads.add(full(ins.op_register(number)))
                elif kind == OpKind.MEMORY:
                    reads.update(full(r) for r in (ins.memory_base, ins.memory_index) if r != R.NONE)

        if ins.op_count and ins.op0_kind == OpKind.REGISTER:
            writes.add(full(ins.op0_register))
        elif ins.op_count and ins.op0_kind == OpKind.MEMORY:
            note(0)
        if mnemonic == M.NOP:
            return "", set(), set()
        if mnemonic == M.MOV:
            note(1)
            return self.write(ins, width, self.operand(ins, 1, width)), reads, writes
        if mnemonic in (M.MOVZX, M.MOVSX, M.MOVSXD):
            note(1)
            source_width = (RegisterExt.size(ins.op1_register) if ins.op1_kind == OpKind.REGISTER
                            else _memory_width(ins))
            value = self.operand(ins, 1, source_width)
            if mnemonic != M.MOVZX:
                value = f"static_cast<{unsigned(width)}>(static_cast<{signed(width)}>(static_cast<{signed(source_width)}>({value})))"
            return self.write(ins, width, value), reads, writes
        if mnemonic == M.LEA:
            note(1)
            return self.write(ins, width, self.address(ins)), reads, writes
        if mnemonic in (*ALU, M.CMP, M.TEST):
            same = (mnemonic in (M.XOR, M.SUB) and ins.op0_kind == ins.op1_kind == OpKind.REGISTER
                    and ins.op0_register == ins.op1_register)
            if not same:
                note(0, 1)
            a = self.operand(ins, 0, width) if not same else f"static_cast<{unsigned(width)}>(0)"
            b = self.operand(ins, 1, width) if not same else a
            operator = ALU.get(mnemonic, "&")
            lines = []
            fused = next_ins is not None and next_ins.flow_control == FlowControl.CONDITIONAL_BRANCH
            if fused:
                self.flags = True
                if mnemonic in (M.CMP, M.SUB):
                    lines.append(f"fa = {a}; fb = {b};")
                else:
                    lines.append(f"fa = static_cast<{unsigned(width)}>({a} {operator} {b});")
            if mnemonic in ALU:
                lines.append(self.write(ins, width, f"{a} {operator} {b}"))
            else:
                writes.clear()
            return " ".join(lines), reads, writes
        if mnemonic in (M.INC, M.DEC, M.NEG, M.NOT):
            note(0)
            a = self.operand(ins, 0, width)
            value = {M.INC: f"{a} + 1u", M.DEC: f"{a} - 1u", M.NEG: f"0u - {a}", M.NOT: f"~{a}"}[mnemonic]
            line = self.write(ins, width, value)
            if next_ins is not None and next_ins.flow_control == FlowControl.CONDITIONAL_BRANCH:
                if mnemonic not in (M.INC, M.DEC):
                    raise ValueError(f"unsupported flag producer at RVA {ins.ip:#x}")
                self.flags = True
                line = f"fa = static_cast<{unsigned(width)}>({value}); " + line
            return line, reads, writes
        if mnemonic in SHIFTS:
            note(0)
            a = self.operand(ins, 0, width)
            # x86 masks the count to six bits for 64-bit operands and to five bits otherwise, so an
            # 8- or 16-bit destination can be shifted by up to 31. The shift is done in 64 bits
            # (count < 64, no host overflow) and the write truncates the result to the destination.
            mask = 63 if width == 8 else 31
            if ins.op1_kind == OpKind.REGISTER:
                if ins.op1_register != R.CL:
                    raise ValueError(f"unsupported shift count at RVA {ins.ip:#x}")
                reads.add(R.RCX)
                count = f"({self.reg(R.RCX)} & {mask}u)"
            else:
                count = f"{ins.immediate(1) & mask}u"
            if mnemonic == M.SAR:
                a = f"static_cast<std::int64_t>(static_cast<{signed(width)}>({a}))"
            else:
                a = f"static_cast<std::uint64_t>({a})"
            return self.write(ins, width, f"{a} {SHIFTS[mnemonic]} {count}"), reads, writes
        if mnemonic == M.IMUL and ins.op_count in (2, 3):
            note(*range(ins.op_count))
            if ins.op_count == 2:
                a, b = self.operand(ins, 0, width), self.operand(ins, 1, width)
            else:
                a, b = self.operand(ins, 1, width), self.operand(ins, 2, width)
            return self.write(ins, width, f"{a} * {b}"), reads, writes
        if mnemonic == M.RET:
            return "return rax;", {R.RAX}, set()
        raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")


def _memory_width(ins):
    width = MemorySizeExt.size(ins.memory_size)
    if width not in (1, 2, 4, 8):
        raise ValueError(f"unsupported memory width at RVA {ins.ip:#x}")
    return width


def lift(pe, name):
    entry = export_rva(pe, name)
    function = Function(pe, entry)
    edges = function.explore()
    lifter = Lifter()
    order = sorted(function.instructions)
    instructions = function.instructions
    loads = {sequence[0]: index for index, _, sequence, _, _, _ in function.dispatches.values()}
    skipped = {ip for _, _, sequence, _, _, _ in function.dispatches.values() for ip in sequence}
    body = {}
    effects = {}
    for ip in order:
        ins = instructions[ip]
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix:
            raise ValueError(f"unsupported prefix at RVA {ip:#x}")
        following = instructions.get(ins.next_ip) if ins.flow_control == FlowControl.NEXT else None
        body[ip] = ""
        if ins.flow_control == FlowControl.CONDITIONAL_BRANCH:
            producer = function.ending.get(ip)
            if ip in edges or producer is None or producer.flow_control != FlowControl.NEXT:
                raise ValueError(f"flags cross a join before RVA {ip:#x}")
            if producer.mnemonic not in (*ALU, M.CMP, M.TEST, M.INC, M.DEC):
                raise ValueError(f"unsupported flag producer at RVA {producer.ip:#x}")
            body[ip] = f"if ({lifter.condition(producer, ins)}) goto loc_{ins.near_branch_target:x};"
            effects[ip] = (set(), set(), set())
        elif ins.flow_control == FlowControl.UNCONDITIONAL_BRANCH:
            body[ip] = f"goto loc_{ins.near_branch_target:x};"
            effects[ip] = (set(), set(), set())
        elif ip in function.dispatches:
            index, targets, _, _, target, _ = function.dispatches[ip]
            cases = {}
            for number, destination in enumerate(targets):
                cases.setdefault(destination, []).append(number)
            lines = [f"switch ({lifter.reg(index)}) {{"]
            for destination, numbers in cases.items():
                labels = " ".join(f"case {n}:" for n in numbers)
                lines.append(f"    {labels} goto loc_{destination:x};")
            lines += ["    default: std::abort();", "}"]
            body[ip] = "\n    ".join(lines)
            # The guest jump target has no host meaning after the dispatch.
            effects[ip] = (set(), set(), {target})
        elif ip in skipped:
            # The table load reads the index; it and the add leave guest addresses behind.
            effects[ip] = ({loads[ip]} if ip in loads else set(), set(), {full(ins.op0_register)})
        elif ins.mnemonic == M.LEA and ins.is_ip_rel_memory_operand and RegisterExt.is_gpr64(ins.op0_register):
            # A guest address is only usable as a jump-table base, which the switch replaces.
            effects[ip] = (set(), set(), {full(ins.op0_register)})
        else:
            if ins.mnemonic != M.RET and ins.is_stack_instruction:
                raise ValueError(f"stack operation in leaf profile at RVA {ip:#x}")
            text, reads, writes = lifter.statement(ins, following)
            body[ip] = text
            effects[ip] = (reads, writes, set())
        touched = set().union(*effects[ip])
        if touched - VOLATILE:
            raise ValueError(f"nonvolatile register {NAMES[min(touched - VOLATILE)]} at RVA {ip:#x}")

    # Must-defined registers: arguments at entry, intersected at every join.
    defined = {ip: set(NAMES) for ip in order}
    defined[entry] = set(ARGUMENTS)
    changed = True
    while changed:
        changed = False
        for ip in order:
            _, writes, kills = effects[ip]
            out = (defined[ip] | writes) - kills
            for successor in function.successors[ip]:
                merged = defined[successor] & out
                if successor == entry:
                    merged &= set(ARGUMENTS)
                if merged != defined[successor]:
                    defined[successor] = merged
                    changed = True
    for ip in order:
        missing = effects[ip][0] - defined[ip]
        if missing:
            raise ValueError(f"undefined register {NAMES[min(missing)]} at RVA {ip:#x}")

    labels = set(edges)
    for position, ip in enumerate(order):
        ins = instructions[ip]
        following = order[position + 1] if position + 1 < len(order) else None
        if ins.flow_control in (FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH) and following != ins.next_ip:
            labels.add(ins.next_ip)
    locals_ = sorted((lifter.used | {R.RAX}) - set(ARGUMENTS), key=lambda r: NAMES[r])
    declarations = [f"word {NAMES[r]} = 0;" for r in locals_]
    if lifter.flags:
        declarations.append("word fa = 0, fb = 0;")
    text = []
    for position, ip in enumerate(order):
        if ip in labels:
            text.append(f"loc_{ip:x}:")
        if body[ip]:
            text.append("    " + body[ip])
        ins = instructions[ip]
        following = order[position + 1] if position + 1 < len(order) else None
        if ins.flow_control in (FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH) and following != ins.next_ip:
            text.append(f"    goto loc_{ins.next_ip:x};")
    signature = (f'extern "C" __declspec(dllexport) word lifted_{name}'
                 f'({", ".join(f"word {NAMES[r]}" for r in ARGUMENTS)})')
    source = [signature + " {", *("    " + d for d in declarations), *text, "}"]
    return "\n".join(source), function


PRELUDE = """#include <cstdint>
#include <cstdlib>
#include <cstring>
using word = std::uint64_t;
template <class T> static T load(word address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return value;
}
template <class T> static void store(word address, T value) {
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
}
"""


def translate_pe(pe, names):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    functions = [lift(pe, name) for name in names]
    return PRELUDE + "\n".join(f"\n{source}" for source, _ in functions) + "\n", [f for _, f in functions]


def translate(image, names):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        source, functions = translate(args.image, args.exports.split(","))
        args.output.write_text(source, encoding="utf-8")
        for function in functions:
            print(f"RVA {function.entry:#x}: {len(function.instructions)} instructions, "
                  f"{sum(len(d[1]) for d in function.dispatches.values())} table entries")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
