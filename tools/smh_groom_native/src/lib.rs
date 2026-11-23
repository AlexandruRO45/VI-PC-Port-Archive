//! CPU kernels for blending and packing synthetic or user-provided polylines.

mod geometry;
mod strands;

use std::slice;

type Vec3 = [f64; 3];

// The compensated sum of squares and square-root correction used by CPython
// 3.12–3.14 math.hypot. Keep each rounding step, especially the final correction,
// because the result can affect trimming, frame selection, and float32 rounding.
#[inline]
fn hypot3(v: Vec3) -> f64 {
    let abs = v.map(f64::abs);
    let max = abs[0].max(abs[1]).max(abs[2]);
    if max == 0.0 || !max.is_finite() {
        return max;
    }
    // Below this threshold the power-of-two scale would overflow. The only
    // callers compare such lengths with a much larger floor or reject them.
    if max < f64::MIN_POSITIVE {
        return max;
    }
    let exponent = (max.to_bits() >> 52) as u32;
    let scale = if exponent >= 2045 {
        f64::from_bits(1u64 << (2096 - exponent))
    } else {
        f64::from_bits(u64::from(2045 - exponent) << 52)
    };

    let mut sum = 1.0;
    let mut square_error = 0.0;
    let mut sum_error = 0.0;
    let mut accumulate = |x: f64, sign: f64| {
        // Split into 26 and 27 significant bits to recover the rounding
        // error without a library FMA on targets that lack a hardware one.
        let hi = f64::from_bits(x.to_bits() & !((1u64 << 27) - 1));
        let lo = x - hi;
        let square = x * x;
        let error = ((hi * hi - square) + (2.0 * hi) * lo) + lo * lo;
        let term = sign * square;
        let next = sum + term;
        square_error += sign * error;
        sum_error += (sum - next) + term;
        sum = next;
        (sum - 1.0) + (square_error + sum_error)
    };

    let [x, y, z] = abs.map(|x| x * scale);
    accumulate(x, 1.0);
    accumulate(y, 1.0);
    let root = accumulate(z, 1.0).sqrt();
    let residue = accumulate(root, -1.0);
    (root + residue / (2.0 * root)) / scale
}

#[no_mangle]
pub extern "C" fn smh_groom_abi_version() -> u32 {
    1
}

// All buffers are owned by Python and have the lengths indicated by the
// scalar arguments. A nonzero status discards all partially written output.
/// # Safety
/// The input buffers must be aligned, readable for their specified lengths,
/// and unmodified during the call. The output and six bounds must be writable
/// and must not overlap another buffer.
#[no_mangle]
pub unsafe extern "C" fn smh_groom_geometry(
    points: *const Vec3,
    counts: *const u32,
    rows: usize,
    nodes: usize,
    mode: u32,
    trig: *const [f64; 2],
    sides: usize,
    radius: f64,
    tip: f64,
    z_up: bool,
    output: *mut u8,
    output_len: usize,
    bounds: *mut [f32; 6],
) -> i32 {
    let Some(point_count) = rows.checked_mul(nodes) else {
        return 1;
    };
    let input = slice::from_raw_parts(points, point_count);
    let counts = slice::from_raw_parts(counts, rows);
    let trig = if sides == 0 {
        &[]
    } else {
        slice::from_raw_parts(trig, sides)
    };
    let output = slice::from_raw_parts_mut(output, output_len);
    match geometry::pack(input, counts, nodes, mode, trig, radius, tip, z_up, output) {
        Some(range) => {
            *bounds = range;
            0
        }
        None => 1,
    }
}

/// # Safety
/// The input buffers must be aligned, readable for their specified lengths,
/// and unmodified during the call. The output buffer must be writable for
/// `rows * (nodes * 3 + 4)` doubles, without aliasing the input buffers.
#[no_mangle]
pub unsafe extern "C" fn smh_groom_strands(
    points: *const Vec3,
    guides: *const Vec3,
    guide_count: usize,
    nodes: usize,
    indices: *const [u32; 3],
    weights: *const [f64; 4],
    rows: usize,
    amount: f64,
    low: f64,
    output: *mut f64,
) -> i32 {
    let Some(point_count) = guide_count.checked_mul(nodes) else {
        return 1;
    };
    let Some(stride) = nodes.checked_mul(3).and_then(|n| n.checked_add(4)) else {
        return 1;
    };
    let Some(output_count) = rows.checked_mul(stride) else {
        return 1;
    };

    let points = slice::from_raw_parts(points, point_count);
    let guides = slice::from_raw_parts(guides, guide_count);
    let indices = slice::from_raw_parts(indices, rows);
    let weights = slice::from_raw_parts(weights, rows);
    let output = slice::from_raw_parts_mut(output, output_count);
    if strands::blend(points, guides, nodes, indices, weights, amount, low, output).is_some() {
        0
    } else {
        1
    }
}
