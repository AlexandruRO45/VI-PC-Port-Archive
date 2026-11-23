# Band-3 `grid` octree coordinates and leaf payloads (`navmeshes.rpf`)

Measured over all `27,482 / 27,482` Band-3 `grid` resources (`53,835,068` entries: `8,617,353` inner nodes and
`45,217,715` leaves) in `navmeshes.rpf` (`6,707,512,496` bytes) by `tools/navmesh_grid.py`.

## Octant axis order and sector placement

Each `u64` key is a sentinel-bit linear octree key (`1` = root, `(parent << 3) | octant`). Across the 3 bits of
each octant:

- **Bit 1** = inverted X (`~X`)
- **Bit 0** = inverted Y (`~Y`)
- **Bit 2** = inverted Z (`~Z`)

Equivalently, `bit = ((octant >> shift) & 1) ^ 1`, so a cell at `(depth, ix, iy, iz)` covers
`[origin_c + i_c * size, origin_c + (i_c + 1) * size)` where `size = extent / (1 << depth)`. When `extent > 128`
(`256`, `512`, `1024`, `4096`), inverting all three axes places the owning `128 m x 128 m` sector at `[0, 128)` from
`(origin_x, origin_y)` (which corresponds to `[extent - 128, extent)` in uninverted octant indices).

Confirmed against Band-2 `0x00010015` tile vertices on five varied-Z sectors: `sec=(192,289)` (`extent=128`, `71.5%`
vertices inside `0x03` cells vs `15.7%` unswapped/uninverted), `sec=(189,274)` (`extent=256`, `74.2%` vs `0.6%`),
`sec=(192,202)` (`extent=256`), `sec=(180,282)` (`extent=256`, `21.1%` vs `6.8%`), and `sec=(187,203)` (`extent=256`,
`12.5%` vs `0.7%`).

## Leaf `u32` payload bitfields (`45,217,715 / 45,217,715` leaves)

Every leaf payload falls into one of three formats (`0` failures across `27,482 / 27,482` resources):

| Low byte / value | Leaves | Mask (`w & ~mask == 0`) | Verified fields & constraints (`27,482 / 27,482`) |
|---|---:|---|---|
| `0x0000FF00` | `1,791,096` | `0x0000FF00` | Uniform / empty octant leaf (also the value of all `8,617,353` inner nodes) |
| `0x01` | `26,659,574` | `0x00060F01` | `bits 8..11` in `{0x0 (18,731,172), 0x5 (49), 0x7 (5,182,528), 0x9 (6,943), 0xB (33,765), 0xD (170,495), 0xF (2,534,622)}`; whenever `bit 11` (`0x8`) is set (`2,745,825 / 2,745,825` leaves), `cell_z0 >= 0.0`. `bits 17..18` take values `0..3` (`9,415,535`, `7,057,386`, `4,653,767`, `5,532,886`), increasing with height above `0x03` surface cells |
| `0x03` | `16,767,045` | `0x00007903` | Surface/ground cells; cell size `extent >> depth` is strictly `4 m` (`16,763,936`) or `8 m` (`3,109`). `bit 8` (`0`: `9,771,338`; `1`: `6,995,707`) separates dry/navmesh-bearing cells from submerged/seabed cells. `bits 11..14` (`0..15`) encode a 4-bit sub-cell Z height in `cell_size / 16` (`0.25 m` at `4 m` resolution) |

On `sec=(192,289)` (`12,491` Band-2 vertices matched inside `4 m` `0x03` cells), `bit 8 == 0` on `12,491 / 12,491`
vertices, and `max(vz - cell_z0)` increases monotonically across `bits 11..14` from `0.150 m` (`0`) to `3.950 m` (`15`)
in `0.25 m` steps.

## Verification

```powershell
python -I tests/test_navmesh_grid.py -v
python -I tools/navmesh_grid.py <package>\navmeshes.rpf --start 6253445120 --out <temp>/navmesh-grid.json
```
