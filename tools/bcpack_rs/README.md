# bcpack_rs

Standalone Rust tool for offline BCPack → DDS export from Xbox Series GTA VI TTPF texture packs.

Addresses issue #4949.

**AI-assisted (Claude).** Tested on: build environment (Windows, Rust stable-x86_64-pc-windows-msvc). No GPU required.

## What it does

Reads a `.ttpf` file (Rockstar DirectStorage texture pack), decodes the BCPack entropy layer (interleaved 16-bit rANS), reconstructs BC1/BC3/BC7 blocks, de-tiles the GFX10 swizzle pattern, and writes one `.dds` per entry.

Includes the `single_mip_layout` fix for 4×4 textures that otherwise decode as all-zero (black/chrome artifacts).

## Build

```
cargo build --release
```

No external dependencies. Standard library only.

## Usage

```
bcpack_rs <ttpf_file> [--out <dir>] [--mip <n>] [--entry <idx>]

  --out     output directory (default: same dir as input)
  --mip     mip level to extract (default: 0 = highest resolution)
  --entry   single entry index (default: all entries)
```

Example:

```
bcpack_rs data/ttdb/vehicles/ttdb_vehicles.ttpf --out dds_out --mip 0
```

## Status

| Feature | Status |
|---|---|
| TTPF header parse | done |
| rANS 16-bit interleaved decode | done |
| BC1 block reconstruction | done (symbols → endpoint/index layout) |
| BC3 block reconstruction | done |
| BC7 block reconstruction | done |
| GFX10 de-tile (tile_mode 1-3) | done (approximation; full addrlib patterns TODO) |
| single_mip_layout fix | done |
| rANS probability tables (q=0..15) | partial (q=0 exact; q>0 approximated) |
| BC3 history table (confirmed entries) | partial (6 confirmed pairs; full calibration TODO) |
| BC7 history table | partial (4 confirmed entries; full calibration TODO) |
| BC6H | not implemented |

## Notes

The rANS probability tables and history tables are the hardest part to get right. The confirmed entries in `src/tables.rs` were recovered by statistical analysis of the dump. Full table calibration requires running against a large corpus — contributions welcome.

The GFX10 de-tile uses a simplified row-major tile approximation. The full addrlib pattern tables (from Mesa `gfx10SwizzlePattern.h`) give exact results and would be the correct next step.
