# PSIN root descriptors in the retail executable

The two root IDs reported by local PSIN container dumps also occur in static
registration records in the AMD64 retail image (SHA-256
`e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993`).
The addresses below are image-relative.

| Root ID | Static record | Registration function | Object bytes | Alignment | Members |
|---|---:|---:|---:|---:|---:|
| 0xde396fe2 | 0x68449e8 | 0x5a33f30 | 80 | 8 | 5 |
| 0xf4fc446b | 0x69691d0 | 0x597e700 | 608 | 8 | 34 |

Both registration functions pass their static record to RVA 0xc884b0.
It constructs a runtime descriptor and caches its pointer at static offset 0.
The allocation callbacks at 0x2210420 and 0x2992fa0 independently pass the
tabulated sizes and alignment to the same allocator. RVA 0xa04368 compares
the script ID against runtime descriptor offset 0x68.

## Measured representation

| Static offset | Interpretation and evidence |
|---:|---|
| 0x08 | Absolute pointer to packed member metadata. Its byte 0 holds flags, followed by a little-endian u16 count at +1 and u16 offsets at +3, relative to the metadata base. |
| 0x10 | Root ID. RVA 0xc88713 copies it to runtime +0x68. |
| 0x14 | Object byte size, copied to runtime +0x70. |
| 0x18 | Packed flags. Bits 7..9 give log2 alignment at 0xc884dd. |
| 0x1a | Unaligned u32 member offsets, one per metadata entry. RVA 0xc88897 indexes this vector and assigns the offset to the constructed member. |

Within an indexed member record, +0x18 holds a 32-bit identifier and +0x1c
holds the byte used by the member-constructor dispatch at RVA 0xdeee45.
Run the following to emit every field offset, identifier and dispatch kind:

    python -I tools/psin_root_descriptor.py image.exe 0x68449e8 0x69691d0

All five script fields have dispatch kind 0x0d, at byte offsets 0, 16, 32,
48 and 64. The story fields use six dispatch kinds. The last 20 start at
0x80 and repeat every 24 bytes through 0x248, all with kind 0x10.

## Container cross-check

The IDs matched 1,009/1,009 script and 2,270/2,270 story dump headers.
Comparing pairs of member identifier and offset against the first decoded
item of each populated block, all five script pairs appeared in 871 files
and none appeared in the other 138. None of the 34 story pairs appeared in
those decoded items in any of the 2,270 story files. A match across blocks
does not establish a root instance at a particular payload offset.

This recovers the in-memory member layout. Human-readable field names,
gameplay meaning and serialized payload offsets remain unresolved. The
packed static registration record has different offsets from an LSCH entry.
