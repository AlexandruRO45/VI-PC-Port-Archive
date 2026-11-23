# SGD5 pass render-state indices

In the Xbox 1.0.2.1 shader archive, bytes 6, 7 and 8 of a 24-byte pass select rows in the effect's rasterizer, depth/stencil and blend tables. These byte positions and state indices are zero-based. The six bytes immediately before them use one-based shader program selectors, with zero meaning unused. A renderer looking up a stored pass can therefore identify the three explicit state records alongside its shader programs. [Program lookup and effect boundaries](https://github.com/ItsNotRexed/Leonida/pull/2519)

| Pass byte | Effect-local table | Bytes per row | Stored rows across both databases | Effects with passes and at least two rows | Effects in which the pass byte selects every row |
|---:|---|---:|---:|---:|---:|
| 6 | Rasterizer | 16 | 64 | 4 | 4 |
| 7 | Depth/stencil | 24 | 124 | 33 | 33 |
| 8 | Blend | 68 | 398 | 7 | 7 |

All **28,798 passes** in the two databases satisfy the same rule. For a nonempty table, the byte is strictly less than the table's row count. If the count is zero, the byte is zero and there is no stored row to read. The state the runtime supplies in that case is unknown. An explicit row at index zero and an empty table can be distinguished by retaining the table count.

## Reproduce on a local package

Use the `shaders.rpf` listed in the [Xbox 1.0.2.1 manifest](../game/xbox-1.0.2.1.json). The two SGD5 databases start at `0x9000` and `0x10287000`. They contain 910 and 22 effects. Use each effect's own six program lists and its own state tables. [Named-pass reader and package test](https://github.com/ItsNotRexed/Leonida/pull/3036)

1. From the end of the six program ordinal arrays, skip 80 bytes. For each row width in `(16, 24, 68, 36)`, read one little-endian u32 count, record the offset of the first row, and advance by `count * width`. The fourth table is needed to locate the techniques.
2. Read the u32 technique count and skip four bytes per technique. The next u32 is the pass count. For every 24-byte pass, check offsets 6, 7 and 8 against counts 0, 1 and 2 respectively. For a nonempty table the selected row starts at `first_row + pass_byte * width`.
3. For an independent check of the byte positions, repeat the test for **each whole byte** at offsets 6 through 23. For each table, require every pass to satisfy the bounds above. Within each effect with passes and at least two rows, also require the set of byte values across its passes to equal the set of row indices. Exactly one whole-byte position per table satisfies both conditions across the archive. The table above gives those positions and the number of effects that distinguish them.

## Interpretation of the rows

The field positions below are the byte distributions over every stored row (64 rasterizer, 124 depth/stencil, 398 blend rows; the SGD2 reader only served as a hint for the field order). [SGD2 format comparison](https://github.com/Kalmarv/rage-engine-formats/blob/91e5abb68476709fac80f256bc64f928c6fc2d19/src/sgd2_effect.rs)

| Row | Offset | Measured values | Decoded as |
|---|---:|---|---|
| Rasterizer (16 B) | 0 | 3 in 64 rows | `fill_mode`, raw integer (the single value stored) |
| | 1 | 1 in 25 rows, 2 in 35, 3 in 4 | `cull_mode`; named none / front / back with the D3D11 values 1 / 2 / 3. Which winding counts as front depends on the front-face flag below, so the winding a `front` or `back` row culls is not established |
| | 2..15 | bytes 2, 12 and 13 are 1 in every row, the rest 0 | raw: which of the three is the front-face flag is not settled, because none of them varies |
| Depth/stencil (24 B) | 0, 1 | each in `0..1` (0 in 9 and 17 rows) | `depth_enable`, `depth_write`, in the D3D11 order. The pairs are (1, 1) in 106 rows, (1, 0) in 9, (0, 0) in 8 and (0, 1) in 1. In this order the nine mixed rows test depth without writing it, the usual read-only depth pass; swapped, nine rows would write depth without testing it. The order is the likelier reading, not a proof |
| | 2 | 2 in 7 rows, 3 in 30, 5 in 6, 7 in 80, 8 in 1 | `depth_func`, compare function; 1..8 named never .. always in the D3D11 order. The range fits that order, and greater_equal being the most used is what a reversed depth range gives, but the names are the D3D11 reading, not a measurement |
| | 3 | `0..1` | `stencil_enable`: it is 0 in exactly the 18 rows whose bytes 8 and 12 are 1 (stencil pass operation) and 1 in the other 106 rows, which have 3 there |
| | 4..23 | 4 is 128 or 255, 5 is 0, 127 or 255, 6, 7, 10, 11, 22, 23 are 1, 8 and 12 are 1 or 3, 9 and 13 are 6 or 8, 14..21 are 0 | raw; bytes 6..13 have the shape of two stencil faces, but the values do not fix the names |
| Blend (68 B) | 0..3 | 0 / `0..1` / 0 / 0 | raw prefix |
| | `4 + 8 * i` | enable in `0..1` for all eight targets | `enable` |
| | `5 + 8 * i` .. `10 + 8 * i` | small values (`1..6`) in a fixed per-position pattern | raw; blend factors and operations are not named |
| | `11 + 8 * i` | `0..15`: 15 in 369 of 398 rows for target 0 | `write_mask` |

Over the 28,798 stored passes the decoded values are: fill mode 3 in all 1,833 passes with a rasterizer row; cull none 999, front 776, back 58. Depth enable 1 in 2,027 and 0 in 253 of the 2,280 passes with a depth/stencil row; depth write 1 in 1,702 and 0 in 578; compare function less 249, equal 74, greater 64, greater_equal 1,891, always 2; stencil enable 1 in 1,701 and 0 in 579. Of the 17,116 passes with a blend row, 4,716 target entries (over all eight targets) are enabled and 132,212 are not; write mask 15 in 108,236, 3 in 16,415, 0 in 9,139, 7 in 1,528, 1 in 959, 8 in 490, 2 in 156, 11 in 5. The passes with an empty table, where the runtime state is unknown: 26,965 rasterizer, 26,518 depth/stencil, 11,682 blend. There are 3 distinct rasterizer rows, 10 depth/stencil rows and 21 blend rows in use, plus the empty table in each (`--states` lists that as `"state": null`).

`tools/sgd5_pass_pipelines.py` reads every stored pass of both databases, joins each program to the carve index by carve offset and writes the distinct graphics program sets that have a PS as a manifest for `tools/shader_link_check.py batch`; `--passes` adds one row per pass with its programs, stages, the raster, depth and blend bytes and the decoded rows as `raster_state`, `depth_state` and `blend_state` (null for an empty table, an out-of-range selector is an error), `--states FILE` writes the distinct decoded rows with the number of passes using each, and `--spirv-dir` leaves out sets whose `.spv` is missing. On our package it found 28,798 passes and 16,325 distinct graphics sets with a PS. Run `python -I tools/sgd5_pass_pipelines.py <shaders.rpf> <carve>/index.json --out manifest.json --spirv-dir <spirv>`, then `python -I tools/shader_link_check.py batch manifest.json --shader-dir <spirv>`.

The measurements identify the indices stored in the package. Runtime selection of a technique, application of overrides, and the resulting image still require execution evidence.
