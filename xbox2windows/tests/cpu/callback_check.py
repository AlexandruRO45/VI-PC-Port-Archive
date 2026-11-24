"""Compare native and lifted object-table dispatch with two source-owned providers."""

import ctypes as C
from pathlib import Path
import sys

from direct_check import GUARDS, MASK, Word
from cpu_chained_aot import translate


Callback = C.CFUNCTYPE(Word, Word, Word)


class Binding(C.Structure):
    _fields_ = [("table", Word), ("target", Callback)]


def check(image, compiled):
    native = C.CDLL(str(image)).call_context
    native.argtypes = [Word, Word]
    native.restype = Word
    lifted = C.CDLL(str(compiled)).try_lifted_callback
    lifted.argtypes = [Word, Word, C.POINTER(Word), C.POINTER(Binding)]
    lifted.restype = Word
    calls = []

    def provider(name, salt):
        @Callback
        def run(context, value):
            calls.append((name, context, value))
            return value ^ salt
        return run

    a = provider("A", GUARDS[0])
    b = provider("B", GUARDS[1])
    tables = [(Callback * 2)(b, a), (Callback * 2)(a, b)]
    context = Word()
    handle = Word(C.addressof(context))

    def compare(handle_address, value, binding, expected, events):
        calls.clear()
        assert native(handle_address, value) == expected and calls == events
        calls.clear()
        output = (Word * 3)(GUARDS[0], MASK, GUARDS[1])
        middle = C.cast(C.byref(output, C.sizeof(Word)), C.POINTER(Word))
        assert lifted(handle_address, value, middle, binding) == 1
        assert tuple(output) == (GUARDS[0], expected, GUARDS[1]) and calls == events

    compare(0, MASK, None, 6, [])
    empty = Word(0)
    compare(C.addressof(empty), MASK, None, 6, [])
    for table, target, salt, name in zip(tables, (a, b), GUARDS, ("A", "B")):
        context.value = C.addressof(table)
        binding = Binding(context.value, target)
        for value in (0, 1, 1 << 32, 1 << 63, MASK):
            compare(C.addressof(handle), value, C.byref(binding), value ^ salt,
                    [(name, C.addressof(context), value)])

    calls.clear()
    for table, binding in ((tables[0], None), (tables[1], Binding(C.addressof(tables[0]), a)),
                           (tables[0], Binding(C.addressof(tables[0]), b)),
                           (tables[0], Binding(C.addressof(tables[0]), Callback()))):
        context.value = C.addressof(table)
        output = Word(MASK)
        assert lifted(C.addressof(handle), 7, C.byref(output), binding) == 0
        assert output.value == MASK and not calls

    for offset in (None, 0):
        _, _, returns, stops = translate(image, "call_context", partial=True, callback_offset=offset)
        assert len(returns) == 2 and len(stops) == 1
    print("Two providers matched native selection, both arguments and 64-bit returns; unknown bindings stopped.")


if __name__ == "__main__":
    check(*map(Path, sys.argv[1:]))
