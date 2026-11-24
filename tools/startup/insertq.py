"""Emulate the six-byte SSE4a immediate INSERTQ encoding in a Windows AMD64 CONTEXT."""
import struct


def apply_insertq(code, context, rip):
    if len(code) != 6 or code[:3] != b"\xf2\x0f\x78" or code[3] < 0xc0:
        return False
    width = (code[4] & 63) or 64
    offset = code[5] & 63
    if offset + width > 64 or struct.unpack_from("<Q", context, 0xf8)[0] != rip:
        return False
    # FltSave.XmmRegisters starts at offset 0x1a0 in an AMD64 CONTEXT.
    dest = 0x1a0 + 16 * ((code[3] >> 3) & 7)
    src = 0x1a0 + 16 * (code[3] & 7)
    original = struct.unpack_from("<Q", context, dest)[0]
    incoming = struct.unpack_from("<Q", context, src)[0]
    mask = (1 << width) - 1
    result = (original & ~(mask << offset)) | ((incoming & mask) << offset)
    struct.pack_into("<Q", context, dest, result)
    struct.pack_into("<Q", context, 0xf8, rip + 6)
    return True
