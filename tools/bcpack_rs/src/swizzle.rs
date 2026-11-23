#![deny(clippy::all)]

pub struct DetileParams {
    pub width: u32,
    pub height: u32,
    pub bytes_per_block: u32,
    pub block_dim: u32,
    pub tile_mode: u32,
    pub d0: u32,
    pub slices: u32,
}

pub fn detile(tiled: &[u8], p: &DetileParams) -> Result<Vec<u8>, String> {
    let w_blocks = p.width.div_ceil(p.block_dim);
    let h_blocks = p.height.div_ceil(p.block_dim);
    let n_blocks = (w_blocks * h_blocks) as usize;
    let bpb = p.bytes_per_block as usize;

    if tiled.len() < n_blocks * bpb {
        return Err(format!(
            "tiled data too short: got {} need {}",
            tiled.len(),
            n_blocks * bpb
        ));
    }

    if is_single_mip_layout(p.tile_mode, p.d0, p.slices, p.bytes_per_block) {
        return Ok(tiled[..n_blocks * bpb].to_vec());
    }

    if p.tile_mode == 0 {
        return Ok(tiled[..n_blocks * bpb].to_vec());
    }

    let mut out = vec![0u8; n_blocks * bpb];

    let tile_w: u32 = match p.tile_mode {
        1 => 8,
        2 => 4,
        3 => 2,
        _ => 8,
    };
    let tile_h: u32 = 8 / tile_w;

    let w_tiles = w_blocks.div_ceil(tile_w);
    let h_tiles = h_blocks.div_ceil(tile_h);

    for ty in 0..h_tiles {
        for tx in 0..w_tiles {
            let tile_idx = ty * w_tiles + tx;
            for ly in 0..tile_h {
                for lx in 0..tile_w {
                    let bx = tx * tile_w + lx;
                    let by = ty * tile_h + ly;
                    if bx >= w_blocks || by >= h_blocks {
                        continue;
                    }
                    let src_idx = (tile_idx * tile_w * tile_h + ly * tile_w + lx) as usize;
                    let dst_idx = (by * w_blocks + bx) as usize;
                    if src_idx * bpb + bpb <= tiled.len() && dst_idx * bpb + bpb <= out.len() {
                        out[dst_idx * bpb..dst_idx * bpb + bpb]
                            .copy_from_slice(&tiled[src_idx * bpb..src_idx * bpb + bpb]);
                    }
                }
            }
        }
    }

    Ok(out)
}

fn is_single_mip_layout(tile_mode: u32, d0: u32, slices: u32, bpb: u32) -> bool {
    if !(1..=3).contains(&tile_mode) {
        return false;
    }
    if d0 >> 28 != 1 {
        return false;
    }
    if slices != 1 {
        return false;
    }
    bpb <= 1
}
