"""Native/AOT differential checks for integer flag producers and conditional consumers."""
import ctypes
import os
from pathlib import Path
import random
import re
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder, Mnemonic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from cpu_switch_aot import Lifter, translate
from cpu_leaf_aot import export_rva

OPS = ('cmp', 'sub', 'add', 'test', 'and', 'or', 'xor', 'inc', 'dec', 'neg')
BITS = (8, 16, 32, 64)
CONDITIONS = ('o', 'no', 'p', 'np')
ARITH_CONDITIONS = CONDITIONS + ('e', 'ne', 's', 'ns', 'l', 'ge', 'le', 'g')
ALL_CONDITIONS = ARITH_CONDITIONS + ('b', 'ae', 'be', 'a')


def conditions(op):
    if op in ('add', 'neg'):
        return ALL_CONDITIONS
    if op in ('inc', 'dec'):
        return CONDITIONS + ('l', 'ge', 'le', 'g')
    return CONDITIONS


NAMES = [f'flags_{op}_{bits}_{cc}' for op in OPS for bits in BITS for cc in conditions(op)] + ['flags_neg_mem_32_b']
MASK64 = (1 << 64) - 1


def reference(op, bits, a, b):
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    a, b = a & mask, b & mask
    signed = lambda v: v - (1 << bits) if v >= sign else v
    if op in ('cmp', 'sub'):
        result, exact = a - b, signed(a) - signed(b)
        carry = a < b
    elif op == 'add':
        result, exact = a + b, signed(a) + signed(b)
        carry = result > mask
    elif op == 'inc':
        result, exact = a + 1, signed(a) + 1
        carry = False
    elif op == 'dec':
        result, exact = a - 1, signed(a) - 1
        carry = False
    elif op == 'neg':
        result, exact = (-a) & mask, -signed(a)
        carry = a != 0
    else:
        result = {'test': a & b, 'and': a & b, 'or': a | b, 'xor': a ^ b}[op]
        exact = 0  # Logical producers clear overflow.
        carry = False
    zero = (result & mask) == 0
    sf = bool(result & sign)
    overflow = not -sign <= exact < sign
    parity = ((result & 0xff).bit_count() % 2) == 0
    return dict(o=overflow, no=not overflow, p=parity, np=not parity,
                e=zero, ne=not zero, s=sf, ns=not sf,
                b=carry, ae=not carry, be=carry or zero, a=not carry and not zero,
                l=sf != overflow, ge=sf == overflow,
                le=zero or sf != overflow, g=not zero and sf == overflow)


def inputs(bits):
    sign, mask = 1 << (bits - 1), (1 << bits) - 1
    edges = sorted({0, 1, 2, 3, 0x7f, 0x80, 0xff, 0x100, 0x101, sign - 1, sign, sign + 1, mask - 1, mask})
    pairs = [(a, b) for a in edges for b in edges]
    # Every low result byte, both operand signs, and arbitrary upper input bytes.
    for a in range(256):
        pairs.extend((a | (0x123456789abc0000 & ~mask & MASK64), b)
                     for b in (0, 1, 3, sign - 1, sign, mask))
    rng = random.Random(0x0f02 + bits)
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(128))
    return pairs


def rejected(image):
    raw = Path(image).read_bytes()
    with pefile.PE(data=raw, fast_load=True) as pe:
        def decode(name):
            entry = export_rva(pe, name)
            return list(Decoder(64, pe.get_data(entry, 96), ip=entry))

        add_body = decode('flags_add_64_o')
        producer = next(i for i in add_body if i.mnemonic == Mnemonic.ADD)
        add_at = pe.get_offset_from_rva(producer.ip)
        if producer.len != 3:
            raise AssertionError('unexpected ADD fixture encoding')

        inc_dec_patches = []
        for name, mnemonic in (('flags_inc_32_o', Mnemonic.INC), ('flags_dec_32_o', Mnemonic.DEC)):
            body = decode(name)
            seto = next(i for i in body if i.mnemonic == Mnemonic.SETO)
            cmovo = next(i for i in body if i.mnemonic == Mnemonic.CMOVO)
            jo = next(i for i in body if i.mnemonic == Mnemonic.JO)
            for consumer, orig, replacements in (
                (seto, 0x90, (('setb', 0x92),)),
                (cmovo, 0x40, (('cmovb', 0x42),)),
                (jo, 0x70 if jo.len == 2 else 0x80, (('jb', 2), ('jae', 3), ('jbe', 6), ('ja', 7))),
            ):
                off = pe.get_offset_from_rva(consumer.ip)
                idx = next((k for k in range(consumer.len) if raw[off + k] == orig), None)
                if idx is None:
                    raise AssertionError(f'unexpected encoding for {name} {consumer}')
                for cc_name, delta in replacements:
                    val = delta if orig in (0x90, 0x40) else ((orig & 0xf0) | delta)
                    inc_dec_patches.append((name, mnemonic, cc_name, off + idx, val))

        neg_body = decode('flags_neg_32_o')
        neg_ins = next(i for i in neg_body if i.mnemonic == Mnemonic.NEG)
        neg_off = pe.get_offset_from_rva(neg_ins.ip)
        if raw[neg_off:neg_off + neg_ins.len] != bytes.fromhex('f7 d8'):
            raise AssertionError('unexpected NEG EAX encoding')

    with TemporaryDirectory() as tmp:
        target = Path(tmp) / 'control.dll'

        # Shift flags cannot be substituted for those of the ADD. Variable-count
        # shifts have flag-preserving and undefined-OF cases outside this profile.
        data = bytearray(raw)
        data[add_at:add_at + 3] = bytes.fromhex('48 d3 e0')  # SHL RAX, CL
        target.write_bytes(data)
        try:
            translate(target, ['flags_add_64_o'])
        except ValueError as exc:
            if 'unsupported flag producer' not in str(exc):
                raise
        else:
            raise AssertionError('accepted a shift as the OF/PF producer')

        # INC/DEC preserve CF, so CF-dependent conditions must be rejected.
        for name, mnemonic, cc_name, patch_off, patch_val in inc_dec_patches:
            data = bytearray(raw)
            data[patch_off] = patch_val
            target.write_bytes(data)
            expected_msg = f'unsupported condition after {mnemonic}'
            try:
                translate(target, [name])
            except ValueError as exc:
                if expected_msg not in str(exc):
                    raise AssertionError(f'{name} -> {cc_name}: {exc}') from exc
            else:
                raise AssertionError(f'accepted {cc_name} after {name}')

        # NOT does not write flags: replacing NEG with NOT leaves the consumer without a producer,
        # and lowering NOT directly with sets_flags=True rejects it as an unsupported flag producer.
        data = bytearray(raw)
        data[neg_off:neg_off + 2] = bytes.fromhex('f7 d0')  # NOT EAX
        target.write_bytes(data)
        try:
            translate(target, ['flags_neg_32_o'])
        except ValueError as exc:
            if 'flags cross a join' not in str(exc):
                raise AssertionError(f'NOT control: {exc}') from exc
        else:
            raise AssertionError('accepted NOT without a preceding flag producer')

        not_ins = next(Decoder(64, bytes.fromhex('f7 d0'), ip=0x1000))
        try:
            Lifter().lower(not_ins, sets_flags=True)
        except ValueError as exc:
            if 'unsupported flag producer' not in str(exc):
                raise AssertionError(f'NOT sets_flags control: {exc}') from exc
        else:
            raise AssertionError('accepted NOT with sets_flags=True')

    print(f'rejection controls passed (SHL RAX, CL; {len(inc_dec_patches)} INC/DEC CF conditions; NOT)')


def check_neg_mem(original, libraries, buffer):
    source, _ = translate(original, ['flags_neg_mem_32_b'])
    body = source.split('extern "C" __declspec(dllexport) word lifted_')[1]
    if body.count('load_always<std::uint32_t>') != 1 or body.count('load_always<') != 1 or re.search(r'\bload<', body):
        raise AssertionError(f'unexpected load count in flags_neg_mem_32_b:\n{body}')
    statement = next(line for line in body.splitlines() if 'load_always<' in line)
    if len(re.findall(r'\bmem\b', statement)) < 3:
        raise AssertionError(f'flags_neg_mem_32_b did not reuse captured memory operand:\n{statement}')

    functions = [getattr(lib, prefix + 'flags_neg_mem_32_b') for lib, prefix in zip(libraries, ('', 'lifted_'))]
    for function in functions:
        function.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
        function.restype = ctypes.c_uint64
    count = 0
    for a, b in inputs(32):
        take = int(reference('neg', 32, a, b)['b'])
        expected = bytearray(b'\xa5' * 16)
        expected[0] = take
        struct.pack_into('<I', expected, 4, 17 if take else 42)
        struct.pack_into('<I', expected, 8, (-a) & 0xffffffff)
        for label, function in zip(('native', 'lifted'), functions):
            ctypes.memset(buffer, 0xa5, 16)
            result = function(a, b, buffer)
            if result != take or bytes(buffer) != expected:
                raise AssertionError(f'{label} flags_neg_mem_32_b({a:#x},{b:#x}): '
                                     f'return={result}, trace={bytes(buffer).hex()}, expected={expected.hex()}')
        count += 1
    print(f'memory-operand NEG [mem] single-read & {count} differential comparisons passed')
    return count


def check(original, lifted):
    libraries = [ctypes.CDLL(str(Path(p).resolve())) for p in (original, lifted)]
    buffer = (ctypes.c_ubyte * 16)()
    count = 0
    for op in OPS:
        for bits in BITS:
            pairs = inputs(bits)
            for cc in conditions(op):
                name = f'flags_{op}_{bits}_{cc}'
                functions = [getattr(lib, prefix + name) for lib, prefix in zip(libraries, ('', 'lifted_'))]
                for function in functions:
                    function.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
                    function.restype = ctypes.c_uint64
                for a, b in pairs:
                    take = int(reference(op, bits, a, b)[cc])
                    expected = bytearray(b'\xa5' * 16)
                    expected[0] = take
                    selected = (17 if take else 42) | ((1 << 32) if bits == 64 else 0)
                    struct.pack_into('<Q' if bits == 64 else '<I', expected, 4, selected)
                    for label, function in zip(('native', 'lifted'), functions):
                        ctypes.memset(buffer, 0xa5, 16)
                        result = function(a, b, buffer)
                        if result != take or bytes(buffer) != expected:
                            raise AssertionError(f'{label} {name}({a:#x},{b:#x}): '
                                                 f'return={result}, trace={bytes(buffer).hex()}, expected={expected.hex()}')
                    count += 1
    count += check_neg_mem(original, libraries, buffer)
    rejected(original)
    record = os.environ.get('FORM_COVERAGE_RECORD')
    if record:
        from iced_x86 import Formatter, FormatterSyntax
        formatter = Formatter(FormatterSyntax.INTEL)
        _, translated = translate(original, NAMES)
        forms = {formatter.format_mnemonic(ins).lower()
                 for function in translated for ins in function.instructions.values()}
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        Path(record).write_text(''.join(word + '\n' for word in sorted(forms)), encoding='utf-8')
    print(f'{len(NAMES)} exports: {count} native/AOT/reference comparisons passed '
          '(Jcc, SETcc, CMOVcc; 8/16/32/64-bit producers; output guards intact)')


if __name__ == '__main__':
    mode, original, destination = sys.argv[1:]
    if mode == 'emit':
        Path(destination).write_text(translate(original, NAMES)[0], encoding='utf-8')
    elif mode == 'check':
        if os.environ.get('FORM_COVERAGE_RECORD'):
            Path(os.environ['FORM_COVERAGE_RECORD']).unlink(missing_ok=True)
        check(original, destination)
    else:
        raise SystemExit('expected emit or check')
