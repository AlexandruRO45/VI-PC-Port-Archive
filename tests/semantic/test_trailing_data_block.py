"""A listing whose code range ends before a trailing constant-data block translates like one without dead padding.

shader_disasm.py cuts a blob's code range at the count dword of a block laid out as `s_endpgm, N, N data dwords,
s_endpgm`. The range then holds no undecodable word, so translate() no longer re-anchors it, and it used to keep
every instruction in it, including dead padding: an s_nop after an unconditional s_branch and the s_nop run /
second s_endpgm after the last reachable s_endpgm. A dead instruction after a branch is a block of its own, which
under --wave64-emulate shifts the block-state constants by one. translate()
now keeps only the code reachable from entry for such a listing, as it did when the data words were still in the
range. Hand-built listing and blob, no game data.

    python -I tests/semantic/test_trailing_data_block.py -v
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import rdna2spirv, STUB_DXBC  # noqa: E402

ENDPGM = b"\x00\x00\x81\xbf"
NOP = b"\x00\x00\x80\xbf"
CODE_START = 0x150
NAME = "cs_trailing_data_block"
DATA = [0x3F800000, 0x40000000, 0x40400000, 0x40800000]

# (text, bytes): a wave64 compute body with a forward branch, then the real s_endpgm. Only the s_endpgm / s_nop bytes
# are real encodings; the others are placeholders the printed text and the blob share.
CLEAN = [
    ("s_mov_b32 s20, 9", bytes(4)),
]
HEAD = [("s_and_b64 s[12:13], exec, exec", bytes(4)), ("s_cmp_eq_u32 s0, 0", bytes(4))]
JUMP = [("s_branch 0", bytes(4))]
STORE = [
    ("v_mov_b32_e32 v6, s20", bytes(4)),
    ("v_lshlrev_b32_e32 v5, 2, v0", bytes(4)),
    ("buffer_store_dword v6, v5, s[0:3], 0 offen", bytes(8)),
    ("s_endpgm", ENDPGM),
]
BODY = HEAD + [("s_cbranch_scc1 2", bytes(4))] + CLEAN + JUMP + STORE
# The same body with an alignment s_nop after the unconditional s_branch: dead, and a block of its own.
BODY_DEAD_NOP = HEAD + [("s_cbranch_scc1 3", bytes(4))] + CLEAN + [("s_branch 1", bytes(4)), ("s_nop 0", NOP)] + STORE
DEAD_TAIL = [("s_nop 0", NOP)] * 3 + [("s_endpgm", ENDPGM)]


def block(count=None):
    """The constant-data block after the code: the count, the data, the closing s_endpgm, s_code_end padding."""
    return (struct.pack("<I", len(DATA) if count is None else count) + b"".join(struct.pack("<I", d) for d in DATA)
            + ENDPGM + b"\x1f\x00\x9f\xbf" * 4)


def translate(program, after=None):
    """Translate `program` as a wave64 CS under --wave64-emulate; `after` is what the blob holds past its range."""
    code = b"".join(raw for _, raw in program)
    end = CODE_START + len(code)
    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{end:x} (header), gfx1013, 0 invalid words"]
    for text, raw in program:
        lines.append(f"\t{text} ; encoding: [{','.join(f'0x{b:02x}' for b in raw)}]")
    blob = bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1)
    blob += bytes(CODE_START - len(blob)) + code + (block() if after is None else after)
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", NAME + ".s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(tmp, NAME + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", NAME + ".fssm"), "wb") as f:
            f.write(blob)
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        return spv


class TrailingDataBlockTest(unittest.TestCase):
    def test_dead_padding_around_the_code_is_dropped(self):
        self.assertEqual(translate(BODY_DEAD_NOP + DEAD_TAIL), translate(BODY))

    def test_tail_variants_are_dropped_too(self):
        self.assertEqual(translate(BODY_DEAD_NOP + DEAD_TAIL[:1] + DEAD_TAIL[-1:]), translate(BODY))
        self.assertEqual(translate(BODY_DEAD_NOP + DEAD_TAIL[-1:]), translate(BODY))
        self.assertEqual(translate(BODY + DEAD_TAIL), translate(BODY))

    def test_padding_is_kept_when_no_data_block_follows_the_range(self):
        # The dword after the range is not a count that reaches a closing s_endpgm: not a block, nothing trimmed.
        plain = translate(BODY_DEAD_NOP + DEAD_TAIL, after=b"\x1f\x00\x9f\xbf" * 8)
        self.assertNotEqual(plain, translate(BODY))
        self.assertIn("OpEntryPoint GLCompute", plain)


if __name__ == "__main__":
    unittest.main()
