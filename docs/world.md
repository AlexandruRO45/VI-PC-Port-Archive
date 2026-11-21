# World placement and streaming data

What is between the exported models/textures and a rendered world: the world metadata resources, the map
placement records, their transforms, bounds and LOD/streaming fields. Everything below was measured on the
Xbox Series 1.0.2.1 package. No game data is in this file; counts, aggregate bounds and format constants only.

## How much was read

`levels.rpf` is 76,155,065,056 bytes (70.92 GiB). The first pass below sampled four windows; the full census
later in this file scanned the whole data area. `packs.rpf`, `data.rpf` and `common.rpf` are small enough to
index completely. All indexes were made with the corrected RPF9 carver from PR #90 (`--index-only`), which
advances by the actual stream length. File and system-page streams are zlib; some graphics pages and four
continuation pages are stored raw, so "unread" is non-zlib payload between streams, not a second file format.

| Area | Bytes covered | Streams | Stored | Inflated | Zero | Unread | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `packs.rpf` (whole archive) | 4,798,213,264 | 144,364 | 4.269 GiB | 9.067 GiB | 3.08% | 1.39% | 95.5% |
| `levels.rpf` 0–2 GiB | 2,147,483,648 | 3,185 | 1.994 GiB | 13.063 GiB | 0.28% | 0.02% | 99.7% |
| `levels.rpf` 20–21 GiB | 1,073,741,824 | 16,683 | 0.982 GiB | 2.359 GiB | 1.48% | 0.42% | 98.2% |
| `levels.rpf` 40–41 GiB | 1,073,741,824 | 14,314 | 0.985 GiB | 1.774 GiB | 1.17% | 0.35% | 98.5% |
| `levels.rpf` 70–70.8 GiB | 858,993,459 | 14,445 | 0.787 GiB | 1.923 GiB | 1.50% | 0.09% | 98.4% |
| `data.rpf` (whole archive) | 481,544,304 | 28,509 | 0.394 GiB | 1.307 GiB | 7.88% | 4.21% | 87.9% |
| `common.rpf` (whole archive) | 6,642,192 | 954 | 0.005 GiB | 0.047 GiB | 5.85% | 8.64% | 85.5% |

The four `levels.rpf` windows cover 5,153,960,755 bytes, 6.77% of the archive. The sampled stream split is
~2.36 GiB in 20–21 GiB, 1.77 GiB in 40–41 GiB and 1.92 GiB in 70–70.8 GiB; the first two GiB are anomalous (see
the metadata family below). These are sample measurements; the full census below replaces their totals.

## Full census

`tools/world_census.py` scanned the whole `levels.rpf` data area in 36 independent 2 GiB shards with 12
processes. Each shard has an 8 MiB warm-up overlap so a resource that starts just before a boundary is still
parsed; 1,140 resource starts were re-read in the overlap and counted once by offset. The walker inflates
every zlib stream, validates the 0x7a header/schema, joins contiguous zlib continuation pages, and joins a raw
(uncompressed) continuation page when the system pointers resolve past the first page. Every map payload is
SHA-1 hashed for duplicate detection. The merge step measures every byte and audits the unread ranges for a
0x7a signature or an unaligned zlib stream that inflates to one.

The run took 241.4 s wall (238.2 s inside the shard pool, 2,093.9 s summed across shards), read all 36 shards,
and used no partial result. The archive is 76,155,065,056 bytes; its data area is 76,154,954,352 bytes.

| Class | Bytes | Share |
|---|---:|---:|
| zlib stream payload | 74,877,281,159 | 98.322% |
| raw system continuation | 44,304 | <0.001% |
| raw graphics page (declared 0x6 extent) | 2,762,232 | 0.004% |
| zero padding | 1,058,806,784 | 1.390% |
| unread non-zlib | 216,059,873 | 0.284% |
| **accounted** | **76,154,954,352** | **100%** |

Zlib coverage is 98.322%, but every byte is accounted for. The 216.1 MB unread range is non-zlib payload
between streams. The audit found 23 additional `0DRP` + 0x7a hits there; none has a valid 0x7a schema, so none
is a map or map metadata. It also found zero raw map/meta resources and zero unaligned zlib streams that
inflate to a 0x7a header. No map resource is in the unread range.

| Item | Count |
|---|---:|
| placement map files (`CMapData` + `CEntityDef`) | 23,535 |
| entities (`CEntityDef`) | 1,670,442 |
| map files that join a continuation | 2,868 |
| map files whose continuation page is raw | 3 |
| `CMapData`-only metadata resources (no `CEntityDef`) | 138,839 |
| other 0x7a meta resources | 26,075 |
| zlib stream visits (includes warm-up re-reads) | 1,122,614 |
| byte-identical duplicate maps | 0 |
| duplicate 32-bit map-name hashes | 0 |
| entity blocks in graphics pages | 0 |
| unresolved or truncated map resources | 0 |

The 23,535 maps contain 1,670,442 entities: HD 1,153,122; LOD 100,960; SLOD1 16,828; SLOD2 1,708; SLOD3 601;
SLOD4 352; ORPHAN_HD 396,871. Mean density is 70.98 entities per map (maximum 1,851); the largest map payload
is 506,080 bytes. The global entity bounds are x −16,237.0…5,550.0, y −8,672.0…10,300.0, z −367.3941…1,663.4242
metres. Maps span 0x1e0bed010 (8,065,568,784) to 0xfb883d010 (67,520,155,664); the 0–6 GiB and 64–70.92 GiB
windows contain none. No map payload repeats: 23,535 unique SHA-1s and 23,535 unique map-name hashes.

Four resources need an uncompressed system continuation page: three placement maps (7 entities) and one
`CMapData`-only metadata resource. They are complete after the raw page is joined. The maximum page count for
a map is two. The two earlier passes sampled 9,918 maps / 680,228 entities (~55%). If that byte share is exact,
a linear extrapolation gives 18,033 maps / 1,236,778 entities, 23.4% / 26.0% below the measured total. The
sample's 68.6 entities per map is close to the full 71.0, but its map density (254/GiB at 55% coverage) is
below the full 331.8/GiB, so byte coverage alone did not predict the map count. The measured total sits inside
the second pass's block range of 16k–39.6k maps and 1.03M–2.64M entities.

SixNG's published external measurement is 30,785 map files / 2,249,490 entities; this package measures 76.5% /
74.3% of those. That figure is external context, not a correction to this census: this census measures this
1.0.2.1 package with the map/entity definition above.

The 138,839 `CMapData`-only resources are metadata, not placement: they carry a `CMapData` root and
container-LOD blocks but no `CEntityDef` struct, so they have no entities to count. They account for 1.23 GiB
inflated and are reported separately, not included in the 23,535.

### Per-shard census

Each row is one 2 GiB scan window. The first/last map columns are exact byte offsets of the first and last
placement resource owned by that window; `s` is that shard's elapsed time. The total row's `s` is the sum
across shards, not wall time.

| # | Window (GiB) | Maps | Entities | First map | Last map | s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000–2.000 | 0 | 0 | — | — | 68.1 |
| 1 | 2.000–4.000 | 0 | 0 | — | — | 78.5 |
| 2 | 4.000–6.000 | 0 | 0 | — | — | 64.3 |
| 3 | 6.000–8.000 | 498 | 36,480 | 0x1e0bed010 | 0x1e26f7010 | 60.7 |
| 4 | 8.000–10.000 | 2,466 | 163,572 | 0x20ac57010 | 0x27ffec010 | 69.4 |
| 5 | 10.000–12.000 | 1,992 | 142,180 | 0x2800b4010 | 0x2f923f010 | 65.0 |
| 6 | 12.000–14.000 | 1,318 | 115,571 | 0x34901a010 | 0x363067010 | 58.0 |
| 7 | 14.000–16.000 | 2,451 | 173,999 | 0x39511f010 | 0x3e06a0010 | 70.1 |
| 8 | 16.000–18.000 | 2,174 | 145,118 | 0x417ec1010 | 0x46b205010 | 68.1 |
| 9 | 18.000–20.000 | 1,201 | 87,603 | 0x4a9ac4010 | 0x4fef80010 | 60.0 |
| 10 | 20.000–22.000 | 1,811 | 147,930 | 0x506c7b010 | 0x57e5f8010 | 67.9 |
| 11 | 22.000–24.000 | 1,773 | 120,843 | 0x5a80f3010 | 0x5fb646010 | 67.6 |
| 12 | 24.000–26.000 | 1,048 | 84,407 | 0x60e69c010 | 0x65b3e7010 | 58.4 |
| 13 | 26.000–28.000 | 1,967 | 153,904 | 0x68111d010 | 0x6f9038010 | 69.3 |
| 14 | 28.000–30.000 | 1,405 | 73,294 | 0x715a52010 | 0x767442010 | 66.8 |
| 15 | 30.000–32.000 | 259 | 19,093 | 0x787216010 | 0x7d5231010 | 55.3 |
| 16 | 32.000–34.000 | 244 | 17,946 | 0x804a96010 | 0x86c32e010 | 52.2 |
| 17 | 34.000–36.000 | 221 | 16,061 | 0x885536010 | 0x8cfcbb010 | 50.9 |
| 18 | 36.000–38.000 | 237 | 23,186 | 0x909b84010 | 0x959df8010 | 49.6 |
| 19 | 38.000–40.000 | 209 | 13,476 | 0x98982e010 | 0x9ffbbd010 | 49.7 |
| 20 | 40.000–42.000 | 198 | 17,346 | 0xa003c6010 | 0xa4693f010 | 48.1 |
| 21 | 42.000–44.000 | 188 | 15,215 | 0xa8e4cc010 | 0xaf8f2c010 | 44.6 |
| 22 | 44.000–46.000 | 165 | 10,637 | 0xb2395f010 | 0xb51351010 | 46.0 |
| 23 | 46.000–48.000 | 105 | 7,999 | 0xb879e2010 | 0xb91311010 | 37.1 |
| 24 | 48.000–50.000 | 130 | 8,537 | 0xc1c336010 | 0xc703a7010 | 42.6 |
| 25 | 50.000–52.000 | 262 | 13,100 | 0xc81c50010 | 0xcec270010 | 56.0 |
| 26 | 52.000–54.000 | 214 | 11,942 | 0xd14132010 | 0xd59837010 | 55.2 |
| 27 | 54.000–56.000 | 226 | 10,658 | 0xd86d83010 | 0xdf546f010 | 52.3 |
| 28 | 56.000–58.000 | 276 | 16,731 | 0xe07218010 | 0xe7a774010 | 60.1 |
| 29 | 58.000–60.000 | 215 | 9,842 | 0xe9bda2010 | 0xef0989010 | 55.4 |
| 30 | 60.000–62.000 | 178 | 9,766 | 0xf0f08b010 | 0xf52595010 | 54.1 |
| 31 | 62.000–64.000 | 104 | 4,006 | 0xf80eb4010 | 0xfb883d010 | 59.1 |
| 32 | 64.000–66.000 | 0 | 0 | — | — | 75.0 |
| 33 | 66.000–68.000 | 0 | 0 | — | — | 68.3 |
| 34 | 68.000–70.000 | 0 | 0 | — | — | 61.4 |
| 35 | 70.000–70.925 | 0 | 0 | — | — | 28.7 |
| **total** | **0.000–70.925** | **23,535** | **1,670,442** | **0x1e0bed010** | **0xfb883d010** | **2,093.9** |

## Composition

Resource classes after joining contiguous graphics pages to their system page. "model" means a system page that
carries a `.#dr`, `.#ft` or `.#dd` name; it is an upper bound on model/fragment resources, not proof that
`rpf9dump` decodes geometry from it. "respage" is a standalone `42 40 00 00` graphics page. "system" is a
zero-head `0x5...` system page with no drawable name found in the stream. Counts are resources; bytes are
inflated.

| Area | model | respage | system | 0x7a meta (first page) | 0x7a map (first page) | PSIN | other/text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `packs.rpf` | 27,224 / 5.350 GiB | 205 / 0.082 | 47,933 / 0.417 | 4,678 / 0.530 | 0 | 586 / 0.010 | 35,229 / 2.678 |
| `levels` 0–2 GiB | 0 | 0 | 0 | 2,855 / 13.054 | 0 | 8 / <0.001 | 322 / 0.009 |
| `levels` 20–21 GiB | 2,956 / 1.170 | 500 / 0.798 | 2,207 / 0.214 | 2,728 / 0.053 | 630 / 0.019 | 46 / <0.001 | 5,218 / 0.106 |
| `levels` 40–41 GiB | 1,172 / 0.442 | 782 / 0.672 | 1,984 / 0.305 | 3,282 / 0.026 | 96 / 0.002 | 32 / <0.001 | 6,014 / 0.328 |
| `levels` 70–70.8 GiB | 5,782 / 0.931 | 395 / 0.877 | 271 / 0.065 | 67 / 0.001 | 0 | 10 / <0.001 | 1,622 / 0.049 |
| `data.rpf` | 2,917 / 0.390 | 5 / 0.001 | 229 / 0.065 | 15,768 / 0.375 | 0 | 2,281 / 0.035 | 2,758 / 0.427 |
| `common.rpf` | — | — | — | — | — | — | 954 / 0.047 (257 XML docs) |

`data.rpf` also has 1,263 `RBF0` metadata resources (0.008 GiB) and 87 `GFX*` shader blobs (0.002 GiB).
`common.rpf` is XML config, not placement: 3,315 `<filename>` paths, 2,267 unique `.rpf` paths, including
519 `_metadata.rpf`, 117 `_merged.rpf`, 93 `_interiors.rpf`, 51 `_occl.rpf` and 2 `_occl_2.rpf` map-cell
archives. Those names explain the cell layout, but the outer `levels.rpf` does not expose them; its directory
table stays encrypted.

The 20–21 GiB map count needs a correction: a first system page of exactly 65,536 bytes is often followed by a
small continuation stream (usually 1–2 KiB) that the next stream scan alone would miss. `tools/world_index.py`
joins that continuation when the schema still has unresolved pointers. The join turns 630 first-page maps into
698 complete map files and 55,564 entities into 75,551. The other sampled windows do not change. This is a
format rule, not a heuristic: the continuation is reached by system pointers in the first page, and the joined
schema then validates. The full census also found four resources whose continuation page is stored raw rather
than zlib. The extent is still the unresolved system pointer range; joining 44,304 raw bytes completes three
placement maps (7 entities) and one `CMapData`-only metadata resource.

## Signatures and strings

The class names are not stored as text: `CMapData` and `CInstance` had zero literal hits in `common.rpf` and
the sampled `levels.rpf` streams. The map roots are found by their struct hashes (`d3593fa6`, `ce501483`),
which are the case-sensitive joaat of `CMapData` and `CEntityDef`. Readable strings in the world streams are
model/material names (`.#dr`, `.#ft`) and, in `data.rpf`, UI/dialogue text; `ipl`, `stream`, `lod`, `cell`,
`zone` hits were substrings of those names or of unrelated text, not chunk tags. No repeated 4×4 or 3×4 matrix
table was found.

Byte offsets used: the four archive windows in the table above. The first map resource in the 20–21 GiB window
is at 21,588,586,512 (`0x506c7b010`); the first in the 40–41 GiB window is at 42,953,629,712
(`0xa003c6010`). The `0DRP` signature is at +0x10 of a map/meta stream; in a `PSIN` resource the same tag is in
the footer.

## The 0x7a resource

World metadata and map placement are **self-describing `0x7a` resources**, not `PSIN` and not `RBF0`. A carved
stream that starts a resource has 8 zero bytes, a `0x5...` page-map pointer at +0x08, then:

```
+0x10  "0DRP" + u8 0x7a          magic + version
+0x20  u32 root object count
+0x28  u64 -> struct table       32 bytes per struct
+0x30  u64 -> enum table
+0x38  u64 -> instance table     16 bytes per block
+0x48  u16 structs, u16 enums, u16 blocks, u16 reserved
```

Struct record (32 bytes): name hash, signature, flags, 0, `u64 -> member table`, size, 0, member count.
Member record (16 bytes): name hash, byte offset, type, subtype, reference, extra. Instance block (16 bytes):
struct hash, byte size, `u64` pointer. The pointers are `0x50000000 | file offset`; a `0x6...` pointer would
reach an appended graphics page. Fields are not named on disk. Their 32-bit hashes are the **case-sensitive
joaat** of the engine names; matching those hashes is how the fields below were identified.

Every 0x7a header is checked the same way: the struct/instance table pointers must lie inside the stream, every
struct's member table must lie inside the stream, and at least one block must have a known struct and a
resolvable pointer. A block whose struct hash is not in the table is kept opaque rather than guessed (some
resources use a page block there). Entity blocks must divide cleanly by the entity struct size before entities
are counted. A stream whose tables point past its end is treated as a split resource and its continuation is
joined, rather than rejected.

Two structs identify a map-placement file: root `d3593fa6` (`CMapData`, 0x460 bytes in the sample) and entity
`ce501483` (`CEntityDef`, 0x90 bytes). Both are the case-sensitive joaat of those class names, which confirms
both the names and the hash function. Files with no entity/root struct are counted as `meta`, not placement.

`0DRP` appears in one other place: at the end of a `PSIN` resource it is the page-directory chunk, followed by
the same `0x7a` u32 and a `CHKS` chunk. It is not a map file; it describes the pages of the model/texture
resource. `data.rpf` contains 15,768 such 0x7a/page-directory resources (most are page directories),
`packs.rpf` 4,678, and none of them has a `CMapData` or `CEntityDef` struct. The task's "0DRP page
directories" and a map file share the magic, not the payload type.

### The first two GiB are a different family

The 0–2 GiB window is 89.6% 0x7a resources (2,855 first pages; 2,824 validate completely), 13.05 GiB inflated,
and **none contain an entity block**. All 2,824 carry one identical 7-struct schema:

```
6c4d9553 71ca4525 ba51bd1e c2b80676 c69192ec e33df1c6 fb42c06a
```

Their blocks include 9,546 blocks of exactly 1,048,576 bytes, which at 16 bytes per slot is 65,536 slots: a
hash-table family, like the slot tables in the text containers. There are also many 16/576/1152/1728-byte
blocks. The header and schema are readable; the field semantics are not. This family is **not** placement and
should not be decoded as entity tables. It is the main reason the first two GiB look so different from the rest.

## Map placement records

Decoded from the entity schema; offsets below are the observed member offsets in the sampled files. `position`
is RAGE world space in metres, Z up. `rotation` is the stored quaternion and is the inverse of the entity's
world rotation (conjugate it for rendering). `scaleXY` applies to X/Y, `scaleZ` to Z.

| Field | Hash | Offset | Type | Meaning |
|---|---|---:|---|---|
| `archetypeName` | `a023a02c` | +0x44 | u32 | hash of the archetype/model name |
| `position` | `01166164` | +0x38 | 3×f32 | world position, metres |
| `rotation` | `77dd4010` | +0x10 | 4×f32 | quaternion x,y,z,w, stored inverse |
| `scaleXY` | `9ca32637` | +0x54 | f32 | XY scale |
| `scaleZ` | `10fb7c42` | +0x58 | f32 | Z scale |
| `flags` | `67d26872` | +0x48 | u32 | entity flags |
| `guid` | `9a7b6e6d` | +0x20 | u64 | placement GUID |
| 64-bit hash | — | +0x08 | u64 | per-archetype hash, function unknown |
| `lodLevel` | `6c8f1715` | +0x7e | u8 | 0 HD, 1 LOD, 2 SLOD1, 3 SLOD2, 4 SLOD3, 5 ORPHAN_HD, 6 SLOD4 |
| `parentIndex` | `d89231bd` | +0x5c | i16 | parent entity index (signed; -1 = none) |
| `parentInstance` | `e74c3c23` | +0x5e | i16 | parent batch instance |
| `numChildren` | `a687ac89` | +0x68 | u32 | linked children |
| `lodDist` | `bf74dfa7` | +0x60 | f32 | LOD distance; -1 = use archetype default |
| `childLodDist` | `ca974bcd` | +0x64 | f32 | child LOD distance |
| `priorityLevel` | `2691f019` | +0x7c | u8 | priority |

The map root carries `name` (`e6e17f14`, +0x08, u32 hash), `parent` (`a99c4651`, +0x68, u32 hash),
`streamingExtentsMin` (`dd22821f`) / `streamingExtentsMax` (`a22ea335`), `entitiesExtentsMin` (`1c75bcf1`) /
`entitiesExtentsMax` (`6d074437`), `flags` and `contentFlags`. The extents are reached through the root's member
table, not fixed offsets.

### Measured placement summary

| Area | Map files | Entities | Unique archetypes | Entity position bbox (x, y, z) |
|---|---:|---:|---:|---|
| 0–2 GiB | 0 | 0 | 0 | — |
| 20–21 GiB | 698 | 75,551 | 8,758 | (-9866.3, 993.5, -146.2) … (-717.2, 5285.8, 123.7) |
| 40–41 GiB | 96 | 6,912 | 2,898 | (-4094.2, 4004.7, -3.6) … (-1.9, 6140.3, 81.3) |
| 70–70.8 GiB | 0 | 0 | 0 | — |
| **sampled** | **794** | **82,463** | — | — |

This table is the four-window sample retained to show the per-window method; the full census above measures
23,535 map files and 1,670,442 entities.

The 20–21 GiB LOD split: HD 53,551; LOD 2,119; SLOD1 230; SLOD2 40; SLOD3 11; SLOD4 10; ORPHAN_HD 19,590.
`lodDist` is -1 for 74,375 entities (98.4%, meaning archetype default); the rest are 10–1200 m.
`parentIndex` is -1 for 19,874 entities (26.3%); the rest point at another entity. The 40–41 GiB window splits
HD 5,354; LOD 1,248; SLOD1 278; SLOD2 25; SLOD3 6; SLOD4 1.

Every sampled map file's three streaming extents are equal, i.e. a cube. Cube sizes range from 20 m to
42,744 m; common values in the 20–21 GiB window are 240, 300, 500, 600 and 2000 m. That per-file cube is the
cell/zone metadata a streamer needs. **No global cell table, zone table or 4×4/3×4 matrix array was found** in
the sampled streams. Placement is per-entity position + quaternion + scale, not a matrix; the repeated 16-byte
pattern in the 0–2 GiB family is a hash-table slot, not a matrix row.

For context, SixNG's published full-archive measurement is 30,785 map files and 2,249,490 entities in
`levels.rpf` (https://github.com/holdmysocks/SixNG). The full census above is 23,535 map files and 1,670,442
entities, 76.5% and 74.3% of that external count. The gap audit found no 0x7a resource in the unread bytes, so
the difference is not a coverage hole in this scan; a different package build or a different map/entity
definition would produce the external total. That number is context, not a correction to this census.

## What is readable, what is hash-only

Readable:
- the 0x7a header, schema, struct/member tables and instance blocks;
- map file count, per-file streaming/entity extents;
- every entity's position, rotation, scale, flags, GUID, 64-bit hash, archetype hash and LOD fields;
- in `common.rpf`, the full XML content-definition path list and cell names.

Hash-only or opaque:
- the RPF directory table and therefore original file names; resource names must be recovered from payloads;
- map file `name`/`parent` (32-bit hashes; `parent` is nonzero for 2,089 of the 23,535 maps);
- most `archetypeName` values. The repo's model manifest reports 155,598 models with names/bounds; this light
  check used 61,861 name hashes recovered from sampled drawables, fragments and `packs.rpf`, not the full
  manifest join. 5,538 of the 75,551 entities in the 20–21 GiB window matched (7.3%). The remaining 92.7% are
  hash-only without a full name index and the bbox/archive-order linking step used by SixNG. This is a lower
  bound, not a claim that the names do not exist;
- texture keys do not appear in placement records. They are 64-bit asset keys in model materials, so a runtime
  resolves placement → archetype → model → material asset key;
- the 0–2 GiB 7-struct family's field semantics and most of its payload;
- `PSIN` page-directory internals beyond the chunk tags (`MNEL`/`LSNS`, `LGID`/`LRSC`, `0DRP`/`0x7a`, `CHKS`);
- texture dictionaries: no `.#td` system page was found in the sampled `levels.rpf` windows. World textures in
  this package are the loose RDS5 pools under `data/ttdb` plus `textures.rpf`, so placement records do not
  carry texture names, only material/asset keys in the model resources.

## What a runtime needs

1. **RPF9 page reader.** Read the `9FPR` header; the data area starts at the u32 at +0x04. Streams start on a
   16-byte boundary; file/system pages are zlib and some graphics pages are raw. Join a system page and its
   contiguous graphics pages; for 0x7a resources also join the continuation page when the schema's pointers
   reach it, whether that page is zlib or raw.
2. **0x7a schema parser.** Validate the header and tables, then read instance blocks. Do not assume fixed
   offsets: field offsets come from the resource's member table. The map root/entity hashes above identify
   placement files.
3. **Transforms.** Apply `position`, conjugate the stored quaternion for world rotation, and use `scaleXY` /
   `scaleZ`. The source is RAGE Z-up; glTF-style Y-up is `(x, z, -y)` with the vector part rotated accordingly.
4. **Archetype keys.** Resolve the 32-bit `archetypeName` hash to an exported model. Keep the 64-bit +0x08 hash
   and GUID as secondary keys. Hash-only records are normal; the runtime needs a hash→model table, with the
   bbox/archive-order fallback for names that were not recovered.
5. **Streaming cells.** Load map files by their streaming AABB (cube) and use `lodLevel`, `lodDist`,
   `childLodDist`, `parentIndex`, `parentInstance` and `numChildren` to switch HD/LOD/SLOD. Load the HD model
   first, then its parent/children; do not use one global LOD distance.
6. **Model/texture layer.** The existing manifest gives names/bounds; texture keys are 64-bit asset keys in
   model materials. Placement itself contains neither geometry nor texture names.

## Tool

`tools/world_index.py` is the single-window scanner and the 0x7a library. It scans an RPF9 archive or a
directory of carved streams and writes JSON/CSV with `offset, size, signature, version, kind, structs, blocks,
entities, extents, position bounds, LOD counts`. It implements the header/schema validation above and the
continuation join. It does not decrypt anything, does not extract models/textures, and keeps the output
aggregate (one row per resource, no entity dump).

`tools/world_census.py` is the whole-archive driver. It shards the data area, scans shards in parallel, joins
zlib and raw continuation pages, hashes map payloads for duplicate detection, measures every byte, and audits
the unread ranges. It writes `summary.json`, `maps.json` and one `shard_*.json` per window; no payload is kept.

```powershell
python -I tools\world_census.py <levels.rpf> --shard-size 2147483648 --jobs 12 --out work\world\census
python -I tools\world_census.py <levels.rpf> --out work\world\census --merge-only
python -I tools\world_index.py <levels.rpf> --start 21474836480 --limit 1073741824 `
    --out work\world\levels-20-21.json --csv work\world\levels-20-21.csv
python -I tests\test_world_census.py
python -I tests\test_world_index.py
```

`world_index.py` still needs `--start/--limit` for a large archive. `world_census.py` is the whole-archive path;
`--resume` reuses complete shards and `--merge-only` reports the coverage already on disk. A map whose entity
block lives in a 0x6 graphics page is counted with fewer entities (the census found 0 such blocks); a resource
needing more than 64 pages or a page above the 64 MiB cap is treated as incomplete rather than guessed (the
census found 0).

## No-sharing rule

Everything here is aggregate. Do not commit archives, carved streams, extracted models/textures, entity
dumps, or logs that quote them. `work/` and `reports/` are git-ignored; the tool output contains only resource
headers, counts and aggregate bounds, but it is still generated from the user's copy and stays local.
