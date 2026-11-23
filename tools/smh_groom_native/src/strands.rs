use crate::{hypot3, Vec3};

#[inline]
fn dot([a, b, c]: Vec3, [x, y, z]: Vec3) -> f64 {
    (a * x + b * y) + c * z
}

#[inline]
fn point(a: &[Vec3], b: &[Vec3], c: &[Vec3], i: usize, bary: Vec3) -> Vec3 {
    std::array::from_fn(|axis| dot(bary, [a[i][axis], b[i][axis], c[i][axis]]))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn blend(
    points: &[Vec3],
    guides: &[Vec3],
    nodes: usize,
    indices: &[[u32; 3]],
    weights: &[[f64; 4]],
    amount: f64,
    low: f64,
    output: &mut [f64],
) -> Option<()> {
    if nodes == 0
        || !amount.is_finite()
        || !low.is_finite()
        || !points.iter().chain(guides).flatten().all(|x| x.is_finite())
    {
        return None;
    }
    let one_minus_low = 1.0 - low;
    let stride = nodes * 3 + 4;
    for ((ids, &[b0, b1, b2, random]), row) in indices
        .iter()
        .zip(weights)
        .zip(output.chunks_exact_mut(stride))
    {
        let [g0, g1, g2] = ids.map(|i| i as usize);
        let [l0, u0, v0] = *guides.get(g0)?;
        let [l1, u1, v1] = *guides.get(g1)?;
        let [l2, u2, v2] = *guides.get(g2)?;
        let bary = [b0, b1, b2];
        let a = &points[g0 * nodes..(g0 + 1) * nodes];
        let b = &points[g1 * nodes..(g1 + 1) * nodes];
        let c = &points[g2 * nodes..(g2 + 1) * nodes];

        let guide_len = dot(bary, [l0, l1, l2]);
        let factor = low + random * one_minus_low;
        let mut remaining = guide_len * (1.0 + (factor - 1.0) * amount);
        let mut prev = point(a, b, c, 0, bary);
        let u = dot(bary, [u0, u1, u2]);
        let v = dot(bary, [v0, v1, v2]);
        if ![remaining, prev[0], prev[1], prev[2], u, v]
            .into_iter()
            .all(f64::is_finite)
        {
            return None;
        }
        let mut cur = prev;
        let mut length = 0.0;
        let mut live = 1usize;
        let (coords, stats) = row.split_at_mut(nodes * 3);
        for (i, dest) in coords.chunks_exact_mut(3).enumerate() {
            if i > 0 && remaining > 0.0 {
                let next = point(a, b, c, i, bary);
                let delta = std::array::from_fn(|axis| next[axis] - prev[axis]);
                if !delta.into_iter().all(f64::is_finite) {
                    return None;
                }
                prev = next;
                let seg = hypot3(delta).max(f64::from(f32::EPSILON));
                if !seg.is_finite() {
                    return None;
                }
                let used = remaining.min(seg);
                if used >= 1e-9 {
                    let scale = used / seg;
                    for axis in 0..3 {
                        cur[axis] += delta[axis] * scale;
                    }
                    length += used;
                    live += 1;
                }
                remaining -= seg;
            }
            dest.copy_from_slice(&cur);
        }
        if !cur.into_iter().chain([length]).all(f64::is_finite) {
            return None;
        }
        stats.copy_from_slice(&[live as f64, length, u, v]);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_and_repeats_tip() {
        let guide = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let mut output = [0.0; 13];
        assert!(blend(
            &guide,
            &[[2.0, 0.25, 0.5]],
            3,
            &[[0; 3]],
            &[[1.0, 0.0, 0.0, 0.5]],
            0.0,
            1.0,
            &mut output
        )
        .is_some());
        assert_eq!(
            output,
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.25, 0.5]
        );
    }
}
