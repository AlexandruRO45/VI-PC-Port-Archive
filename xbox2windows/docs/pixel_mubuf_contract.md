# Raw MUBUF binding contract for pixel_aot

This is a proposal for the first vector-memory lowering in pixel_aot, tracked by
[issue #319](https://github.com/R0Y00/Xbox2Windows/issues/319). The issue remains open and
no maintainer decision resolves the binding, descriptor-provenance or lookup-miss questions.
The proposal therefore adds no lowerer or passing instruction.

## Bounded first case

The first implementation should accept exactly:

~~~text
buffer_load_dword vN, vM, s[A:A+3], 0 offen
~~~

with the following conditions:

- OFFEN is present and IDXEN is absent.
- The V# word 1 stride field is zero.
- The scalar offset is zero.
- The V# uses raw-buffer bounds mode, OOB_SELECT = 3.
- The V#, vector offset and immediate offset are 4-byte aligned.
- The instruction's result is one VGPR and the V# is the result of one tracked,
  completed s_load_dwordx4 from the constant descriptor table.

Wider loads, idxen, nonzero strides or scalar offsets, stores, swizzles and the other
OOB_SELECT modes remain rejected. This keeps the first review to one address rule.

## V# provenance

The V# is not a desktop descriptor and should not become a second allocation map. An
s_load_dwordx4 result carries provenance as candidate V# words. The provenance remains
usable only by a MUBUF consumer. If the same words feed an image instruction, they retain
the existing texture-descriptor contract. The pair and consumer must be unambiguous.

For a gfx1013 four-word V#:

- DW0 bits [31:0] hold address bits [31:0];
- DW1 bits [15:0] hold address bits [47:32];
- DW1 bits [29:16] hold the 14-bit byte stride;
- DW1 bits [31:30] hold the cache-swizzle and swizzle-enable flags;
- DW2 holds NUM_RECORDS in bytes;
- DW3 bits [29:28] hold OOB_SELECT.

DW1 must not be shifted wholesale into the high half of a 64-bit address. The
address and the remaining descriptor fields use distinct masks:

~~~hlsl
static const uint V_BASE_LO_MASK = 0xffffffffu; // DW0[31:0]
static const uint V_BASE_HI_MASK = 0x0000ffffu; // DW1[15:0] -> address[47:32]
static const uint V_STRIDE_MASK  = 0x3fff0000u; // DW1[29:16]
static const uint V_DW1_FLAG_MASK = 0xc0000000u; // DW1[31:30]

uint64_t base = (uint64_t(dw1 & V_BASE_HI_MASK) << 32) | (dw0 & V_BASE_LO_MASK);
~~~

This is the Buffer Resource Descriptor layout in the
[AMD RDNA 2 instruction-set reference](https://gpuopen.com/download/rdna2-shader-instruction-set-architecture.pdf).
The first case requires stride zero and OOB_SELECT = 3, so it consumes the 48-bit
base, NUM_RECORDS and the raw mode. The runtime copies the four V# words into the
existing constant buffer where the test program's s_load_dwordx4 fetches them.
Reads before the matching s_waitcnt lgkmcnt(0) are rejected.

## Address and bounds

With OFFEN, no IDXEN, stride zero and scalar offset zero, the byte offset is:

~~~text
uoff = vM + offset12
address = base64 + zero_extend(uoff)
~~~

The address addition is 64-bit and carries into the high word. The proposal requires
uoff and offset12 to be multiples of four, leaving the hardware's non-aligned rounding
outside this patch.

The host runtime measured RDNA2 raw bounds as a whole-access check:

~~~text
in_range = (offen + soffset + 4 <= NUM_RECORDS)
~~~

Leonida's 732-case measurement on gfx1036 found the per-access four-byte bound,
including soffset in raw mode. Its OOB_SELECT = 0 and 1 rules differ, and whether
gfx1013 matches gfx1036 remains open. This proposal accepts mode 3 only and records that
console-specific premise.

The guest boundary and the host binding are separate outcomes and must remain
observable independently:

- A guest bound miss means the aligned access is not wholly inside NUM_RECORDS. The
  load returns zero and performs no aliased buffer access.
- A host binding failure means the guest address is inside the V# bound but no
  published title range maps it. The draw must report a binding failure through a
  separately observable channel, such as a rejection or a status word. It must not be
  recorded as a successful read that returned zero.

A translation-time refusal cannot cover a dynamic V# address that the host failed to
publish, so the runtime is the only layer that can distinguish these cases.

## D3D12 binding

The pixel path should remain ps_6_0. The existing embedded pixel root signature has one
texture SRV table in slot 0, a CBV in slot 1 and the static point-clamp sampler. Add a
second descriptor-table parameter in slot 2:

~~~hlsl
DescriptorTable(
    SRV(t0, space = 1, numDescriptors = 1),
    SRV(t1, space = 1, numDescriptors = 8),
    visibility = SHADER_VISIBILITY_PIXEL)
~~~

The shader declarations follow the same shape as the existing compute alias fixture:

~~~hlsl
StructuredBuffer<TitleRange> Ranges : register(t0, space1);
ByteAddressBuffer Aliases[8] : register(t1, space1);
~~~

shader_view supplies the sorted ranges and slot-ordered aliases. The runtime sets the
new table with SetGraphicsRootDescriptorTable(2, ...). It must reject a publication with
more than the root signature's visible descriptor count. A successful DWORD load is:

~~~hlsl
uint slot, byte_offset;
if (TitleLookup(Ranges, range_count, address, 4, slot, byte_offset))
    value = Aliases[NonUniformResourceIndex(slot)].Load(byte_offset);
~~~

The bounded array keeps the limit visible in the root signature and preserves the desktop
shader model. It also reuses TitleLookup, rather than translating a V# into an unrelated
descriptor identity.

## Test plan

The source-built fixture should place a known DWORD pattern at word offsets 0, 1, 2 and
3. Set NUM_RECORDS = 16 for the complete-access cases and draw a scalar offset across the
viewport with:

1. offset 0 for the first word;
2. offset 12 for the last complete word;
3. offset 4 for an interior word;
4. offset 16 for a complete guest bound miss.

Keep the aligned partial tail in its own case. Set NUM_RECORDS = 14 and use offset 12,
which is aligned but has only two bytes left, so the four-byte access must miss. Keep
offsets 13, 14 and 15 in a separate rejection group because they are not 4-byte aligned
and never enter the accepted first case.

Two further descriptor cases belong to the plan:

5. change DW1[15:0] so the 48-bit base points at a different published range, and check
   that the high base word participates in the address;
6. use a V# base inside the guest bound but outside every published range, and require a
   host binding failure rather than zero data.

Add an address-extraction control that feeds two descriptor words with the same DW0 and
DW1[15:0] but different DW1[29:16] or DW1[31:30] bits. Both must produce the same 48-bit
base. A wrong expression that shifts DW1 wholesale must produce two different addresses.
This control runs against the extraction helper and does not need a draw.

The CPU reference applies the guest V# bound before lookup. Cases 1 to 3 must return the
four distinct fixture words. Case 4 and the aligned partial tail must return zero without
reading a neighboring word. Case 6 must surface a binding failure. Negative controls must
fail if the lowering:

- drops the guest V# bound check;
- treats the byte offset as a DWORD index;
- drops the low base word or high base word;
- copies DW1 wholesale into address[63:32];
- counts a host mapping miss as a successful zero read;
- treats an unaligned offset as a partial-tail bound miss;
- selects OOB_SELECT 0 semantics for a raw buffer;
- reads the first range when TitleLookup misses, or crosses a range end; and
- accepts a V# whose provenance is an image descriptor, or accepts the MUBUF before its
  scalar load completes.

The first implementation should cover the first word, last valid word and bound miss.
Multi-dword loads, stores, idxen, nonzero stride and signed scalar offsets stay in
follow-up patches. Assemble and disassemble the fixture with LLVM 15, then run the
integer readback on WARP and Radeon. The GPU tests should run serially and record the
adapter, driver, Windows build, commit and debug-layer state.

## Decisions requested for #319

1. Confirm ps_6_0 with the bounded space1 table and choose the published alias and
   range bound.
2. Confirm the failure channel for an unmapped title address. It must be separately
   observable from a guest bound miss and must not count a zero result as a successful
   read. A draw rejection and a status word are the candidates in this proposal.
3. Confirm that the first case accepts only OOB_SELECT 3, stride zero, no idxen and a
   zero scalar offset.
4. Confirm the gfx1013 raw-bounds rule against an RDNA1 target. The available 732-case
   measurement is gfx1036.

## References

- [Xbox2Windows issue #319](https://github.com/R0Y00/Xbox2Windows/issues/319)
- [Leonida's measured buffer bounds](https://github.com/ItsNotRexed/Leonida/blob/4eb6155d/docs/buffer_bounds.md)
- [AMD RDNA 2 instruction-set reference](https://gpuopen.com/download/rdna2-shader-instruction-set-architecture.pdf)
- [TomBox's independent direct-buffer regression](https://github.com/brcly/TomBox/blob/ca6cf6a6b44e36d3ca4420db55a660a7bc6d6d5a/crates/tuxbox-xshader/tests/mem_conformance.rs)
