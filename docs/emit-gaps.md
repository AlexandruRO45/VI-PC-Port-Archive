# The instruction tail: what the emitter still refuses, and how many programs it costs

The counts below come from the shader archive, not from a run's dump set.
`victory shader-info D:\Grand Test Auto6.Xbox.v1.0.2.1\shaders.rpf` finds
**26,503 containers and 26,698 native programs** (Vs 2,236, Ps 12,101,
Ms 9,282, Cs 2,689, Es/Gs 195 each) in about 9 seconds: the file is SGD5, and
each record is one deflate-packed container. A `VIctory-data\failed-shaders`
dump describes one session, and one session touches a fraction of a percent
of them.

19,506 programs over 20,000 containers, translated one slice of 500
containers at a time (`victory translate <slice> --list`; a single run over
the whole archive outlives the tool's per-call budget):

| Stage | emitted | of | missing |
| --- | ---: | ---: | ---: |
| vertex + pixel | 7,504 | 8,416 | 912 |
| mesh | 8,383 | 8,713 | 330 |
| compute | 2,245 | 2,377 | 132 |

Refusals by family and by owner:

| Programs | Refusal | Owner |
| ---: | --- | --- |
| 808 | emit: a descriptor's address or format read as data | resources |
| 207 | emit: a lane mask read as data | this file, below |
| 195 | resources: an address in an SGPR that is no root parameter | resources |
| 174 | emit: a packed input half no element holds | I/O |
| 167 | I/O: an interpolation with computed barycentrics | I/O |
| 64 | resources: an address not from a root parameter | resources |
| 63 | emit: a descriptor read only as data | resources |
| 41 | emit: a descriptor chosen by control flow | resources |
| 38 | resources: a descriptor read from the first lane of computed VGPRs | resources |
| 15 | resources: a waterfall loop whose compares do not select the access's lanes | resources |
| 15 | emit: a hardware SGPR input | instruction |
| 13 | emit: a constant buffer's descriptor read as data | resources |
| 13 | emit: a constant lane mask | instruction |
| 12 | emit: a lane mask holding lanes elsewhere read as data | instruction |
| 8 | resources: a descriptor partly joined by control flow | resources |
| 8 | emit: the first half of an interpolation | I/O |
| 5 | emit: a whole-wave section step the simulation does not follow | instruction |
| 4 | emit: `s_getpc_b64` | instruction |
| 4 | emit: an LDS operation | instruction |
| 4 | emit: `v_cvt_f32_f64` | instruction |
| 4 | resources: an address from a phi | resources |
| 3 | resources: a descriptor neither loaded nor built | resources |
| 1 | emit: `v_permlanex16_b32` | instruction |
| 1 | emit: LDS the program does not allocate | instruction |
| 1 | emit: a lane-mask join of an untracked mask | instruction |

## Closed on this branch

**A scalar spilled through a VGPR lane, read back at a constant lane.** The
console compiler moves an SGPR through a VGPR lane and reads it back:

```
v_writelane_b32 v158, s14, 1
v_readlane_b32 s0, v158, 1
s_load_dwordx8 s[72:79], s[90:91], s0
```

`pixel-ee41ebc3eb3fbbdc` feeds a descriptor offset this way, and a pixel
program refused the lane write with `a cross-lane operation outside compute`
even though nothing crosses lanes: at a *constant* lane the read back is the
scalar the write stored. `lower::valu_op` resolves it through `spilled_value`
and `spilled_in_vgpr`, the pair the lane-mask spill already uses.
`victory translate pixel-ee41ebc3eb3fbbdc.dxbc --validate` reports
`1 emitted of 1, 1 validated`, and `a_pixel_shader_reads_a_scalar_it_spilled_through_a_lane`
fails on the tree without the change. With this, every pixel container in both
dump sets translates; the only refusals left there are five mesh programs in
the resource layer.

**A mesh shader no longer asks for a wave size.** DXIL allows the wave-size
attribute only on compute and node shaders:

```
error: Function: main: error: WaveSize only allowed on compute or node shaders
```

Every mesh program in this archive is wave64 (920 sampled, none wave32), and
the emitter set `wave_size = Some(64)` for each of them, so **every emitted
mesh shader failed `--validate`** — 8,383 programs — and the container's
`PSV0` lane counts disagreed with the module even after the attribute was
dropped (`DXIL container mismatch for 'PSVRuntimeInfo'`). Neither is now
emitted for a mesh stage, and `chunks/d010` and `chunks/d011` validate
448 of 450 and 427 of 427 mesh programs. `a_mesh_shader_declares_no_wave_size`
holds the container to it.

The consequence is stated where it belongs: a mesh program cannot ask the
runtime for the console's wave, so its lanes are the device's.

## Open, with what would settle each

**A lane mask read as data (207 mesh programs).** The console saves a mask to
a scalar (a register pair in wave64) and reads it back as a number:
`v_cmp_eq_u32 vcc, s14, v80` / `s_xor_b64 s[12:13], vcc, exec` /
`s_bcnt1_i32_b64 s12, s[12:13]` / `s_cbranch_scc1` is a wave-wide "does any
lane differ from the first lane's value", and the count decides a uniform
branch. Compute programs already do this exactly (`MaskData::Console`,
because they declare the console's wave size). For mesh, DXIL cannot declare
one — see above — so with a host wave narrower than the console's the two
halves of a console wave would answer separately and the "uniform" branch
would stop being uniform. The honest fix is group-shared memory: each thread
writes its bit to LDS by its flattened thread id, one barrier, then each
thread counts the bits of its console wave. That is exact and costs a barrier
per site. What would settle whether it is needed at all: the device's mesh
wave size, read back from a compute shader that writes `WaveGetLaneCount()` to
a UAV in the same run — if the driver runs 64-wide waves for the mesh stages
in these pipelines, the ballots are already the console's and no emulation is
needed.

**A packed input half no element holds (174 pixel programs).** A 16-bit pixel
input is packed two per dword and the host's inputs are 32-bit floats, so
`lower` rebuilds each half from the input elements (`packed_half`). It returns
`None` when the signature element for that half does not exist, which happens
for an interpolation of a channel the input signature does not describe. What
would settle it: the ISG1 element list of one failing program against its
interpolation's `attr`/`chan` (e.g. `chunks/c005/11190.dxbc@0x0`), to see
whether the element is missing from the signature or merely from the
`halves` pairing.

**An interpolation with computed barycentrics (167 pixel programs).** The
interpolation's barycentric operand is not an input VGPR but a value joined
by control flow: `chunks/c005/10550.dxbc@0x0` at 0x3790 reads a phi whose
incoming values are `v1` (`PerspectiveCenter`, index 1) and a value the io
pass cannot name, which is neither an input nor an instruction result. If the
arms name the same barycentric, resolving the phi is enough; if they name
different ones, one attribute is interpolated two ways and `io`'s
one-interp-per-attribute model needs a select over two interpolations. What
would settle it: the two arms of that phi from a disassembly of the block
before 0x3790.

**The rest** are single programs each (`s_getpc_b64`, `v_cvt_f32_f64`, an LDS
operation, `v_permlanex16_b32`, a whole-wave section step); each is a small,
self-contained instruction to add once something depends on it.
