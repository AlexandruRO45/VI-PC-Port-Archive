//! Textures de la Xbox Series X : blocs BC7, agencement GFX10 en tuiles de 4 Kio, export PNG.
//!
//! Le décodage BC7 et le désentrelacement sont portés de Xbox2Windows (R0Y00, licence MIT,
//! <https://github.com/R0Y00/Xbox2Windows>, `tools/bc7_blocks.py` et `tools/bc7_tiles.py`).
pub mod bc7;
pub mod png;
pub mod tiles;

/// Image RGBA8 (ligne par ligne) d'une surface de blocs BC7 rangés ligne par ligne.
pub fn decode_surface(blocks: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    if blocks.len() != columns * rows * bc7::BLOCK_BYTES {
        return Err(format!("{} octets reçus, {} attendus pour {width}×{height}", blocks.len(),
                           columns * rows * bc7::BLOCK_BYTES));
    }
    let mut image = vec![0u8; width * height * 4];
    for (index, block) in blocks.chunks_exact(bc7::BLOCK_BYTES).enumerate() {
        let texels = bc7::decode(block)?;
        let (bx, by) = (index % columns * 4, index / columns * 4);
        for texel in 0..bc7::TEXELS {
            let (x, y) = (bx + texel % 4, by + texel / 4);
            if x < width && y < height {
                image[(y * width + x) * 4..][..4].copy_from_slice(&texels[texel * 4..][..4]);
            }
        }
    }
    Ok(image)
}
