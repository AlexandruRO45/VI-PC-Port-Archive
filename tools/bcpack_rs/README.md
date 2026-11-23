# bcpack_rs

Standalone Rust CLI to export BCPack-compressed TTPF textures to DDS offline.

Zero external dependencies. AI-assisted.

## Build

```
cargo build --release --manifest-path tools/bcpack_rs/Cargo.toml
```

## Usage

```
bcpack_rs <file.ttpf> [--out <dir>] [--mip <n>] [--entry <idx>]
```

## Coverage

| Format | Status |
|--------|--------|
| BC1 (0x04) | Implemented |
| BC3 (0x0C) | Implemented, 6 confirmed history pairs |
| BC7 (0x1C) | Implemented, 4 confirmed history entries |
| single_mip_layout fix | Included |
| DDS DX10 output | Implemented |
