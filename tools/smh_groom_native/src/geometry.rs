use crate::{hypot3, Vec3};

const CURVES: u32 = 1;
const STRIPS: u32 = 2;
const RIBBONS: u32 = 3;
const TUBES: u32 = 4;

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|i| a[i] - b[i])
}

#[inline]
fn frame(poly: &[Vec3], i: usize) -> Option<(Vec3, Vec3)> {
    let tangent = if i == 0 {
        sub(poly[1], poly[0])
    } else if i + 1 == poly.len() {
        sub(poly[i], poly[i - 1])
    } else {
        sub(poly[i + 1], poly[i - 1]).map(|v| 0.5 * v)
    };
    let length = hypot3(tangent);
    if !length.is_finite() {
        return None;
    }
    let [tx, ty, tz] = if length > 1e-9 {
        let inv = 1.0 / length;
        tangent.map(|v| v * inv)
    } else {
        [0.0, 0.0, 1.0]
    };

    let side = if tz.abs() < 0.9 {
        [ty, -tx, 0.0]
    } else {
        [0.0, tz, -ty]
    };
    let length = hypot3(side);
    let [sx, sy, sz] = if length > 1e-9 {
        let inv = 1.0 / length;
        side.map(|v| v * inv)
    } else {
        [1.0, 0.0, 0.0]
    };
    Some((
        [sx, sy, sz],
        [ty * sz - tz * sy, tz * sx - tx * sz, tx * sy - ty * sx],
    ))
}

#[inline]
fn clean_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

#[inline]
fn convert([x, y, z]: Vec3, z_up: bool) -> Vec3 {
    if z_up { [x, z, -y] } else { [x, y, z] }.map(clean_zero)
}

struct Vertices<'a> {
    positions: &'a mut [u8],
    normals: &'a mut [u8],
    lo: [f32; 3],
    hi: [f32; 3],
}

impl Vertices<'_> {
    #[inline]
    fn position(&mut self, offset: usize, point: Vec3, z_up: bool) -> Option<()> {
        let xyz = convert(point, z_up).map(|v| v as f32);
        for (axis, x) in xyz.into_iter().enumerate() {
            if !x.is_finite() {
                return None;
            }
            let at = offset + 4 * axis;
            self.positions[at..at + 4].copy_from_slice(&x.to_le_bytes());
            self.lo[axis] = self.lo[axis].min(x);
            self.hi[axis] = self.hi[axis].max(x);
        }
        Some(())
    }

    #[inline]
    fn normal(&mut self, offset: usize, normal: Vec3, z_up: bool) -> Option<()> {
        let normal = convert(normal, z_up);
        let length = hypot3(normal);
        let unit = if length == 0.0 {
            [0.0, 1.0, 0.0]
        } else if length.is_normal() {
            normal.map(|v| clean_zero(v / length))
        } else {
            return None;
        };
        for (axis, x) in unit.into_iter().enumerate() {
            let at = offset + 4 * axis;
            self.normals[at..at + 4].copy_from_slice(&(x as f32).to_le_bytes());
        }
        Some(())
    }
}

struct Indices<'a> {
    bytes: &'a mut [u8],
    offset: usize,
    width: usize,
}

impl Indices<'_> {
    #[inline]
    fn extend(&mut self, values: impl IntoIterator<Item = usize>) {
        for value in values {
            let end = self.offset + self.width;
            self.bytes[self.offset..end]
                .copy_from_slice(&(value as u32).to_le_bytes()[..self.width]);
            self.offset = end;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pack(
    points: &[Vec3],
    counts: &[u32],
    nodes: usize,
    mode: u32,
    trig: &[[f64; 2]],
    radius: f64,
    tip: f64,
    z_up: bool,
    output: &mut [u8],
) -> Option<[f32; 6]> {
    if nodes < 2 || !points.iter().flatten().all(|x| x.is_finite()) {
        return None;
    }
    let (per_node, per_segment) = match mode {
        CURVES | STRIPS => (1, 2),
        RIBBONS => (2, 6),
        TUBES if trig.len() >= 3 => (trig.len(), trig.len().checked_mul(6)?),
        _ => return None,
    };
    let mesh = mode == TUBES || mode == RIBBONS;
    if mesh && !(radius.is_finite() && tip.is_finite()) {
        return None;
    }
    let count_sum = counts.iter().try_fold(0usize, |sum, &n| {
        let n = n as usize;
        if n > nodes || (mode == STRIPS && n == 1) {
            return None;
        }
        sum.checked_add(if mode == STRIPS {
            n
        } else {
            n.saturating_sub(1)
        })
    })?;
    let vertex_count = if mode == STRIPS {
        count_sum
    } else {
        points.len().checked_mul(per_node)?
    };
    if vertex_count > u32::MAX as usize || count_sum == 0 {
        return None;
    }
    let index_count = if mode == STRIPS {
        count_sum
    } else {
        count_sum.checked_mul(per_segment)?
    };
    let width = if vertex_count <= 65535 { 2 } else { 4 };
    let pos_bytes = vertex_count.checked_mul(12)?;
    let nrm_bytes = if mesh { pos_bytes } else { 0 };
    let index_bytes = index_count.checked_mul(width)?;
    if output.len() != pos_bytes.checked_add(nrm_bytes)?.checked_add(index_bytes)? {
        return None;
    }

    let (positions, rest) = output.split_at_mut(pos_bytes);
    let (normals, index_bytes) = rest.split_at_mut(nrm_bytes);
    let mut vertices = Vertices {
        positions,
        normals,
        lo: [f32::INFINITY; 3],
        hi: [f32::NEG_INFINITY; 3],
    };
    let mut indices = Indices {
        bytes: index_bytes,
        offset: 0,
        width,
    };
    let tapers: Vec<_> = if mesh {
        (0..nodes)
            .map(|i| (1.0 - (1.0 - tip) * (i as f64 / (nodes - 1) as f64)) * radius)
            .collect()
    } else {
        Vec::new()
    };

    let mut base = 0;
    for (poly, &count) in points.chunks_exact(nodes).zip(counts) {
        let count = count as usize;
        if mode == STRIPS {
            for (j, &point) in poly[..count].iter().enumerate() {
                vertices.position((base + j) * 12, point, z_up)?;
            }
            indices.extend(base..base + count);
            base += count;
            continue;
        }
        for (j, &point) in poly.iter().enumerate() {
            let ring = base + j * per_node;
            if mesh {
                let (side, up) = frame(poly, j)?;
                let taper = tapers[j];
                if mode == TUBES {
                    for (s, &[ca, sa]) in trig.iter().enumerate() {
                        let radial = std::array::from_fn(|i| ca * side[i] + sa * up[i]);
                        let position = std::array::from_fn(|i| point[i] + radial[i] * taper);
                        let at = (ring + s) * 12;
                        vertices.position(at, position, z_up)?;
                        vertices.normal(at, radial, z_up)?;
                    }
                } else {
                    let left = std::array::from_fn(|i| point[i] - side[i] * taper);
                    let right = std::array::from_fn(|i| point[i] + side[i] * taper);
                    vertices.position(ring * 12, left, z_up)?;
                    vertices.position((ring + 1) * 12, right, z_up)?;
                    vertices.normal(ring * 12, up, z_up)?;
                    let at = ring * 12;
                    vertices.normals.copy_within(at..at + 12, at + 12);
                }
            } else {
                vertices.position(ring * 12, point, z_up)?;
            }
        }
        for j in 0..count.saturating_sub(1) {
            let a = base + j * per_node;
            let b = a + per_node;
            match mode {
                TUBES => {
                    for s in 0..per_node {
                        let next = if s + 1 == per_node { 0 } else { s + 1 };
                        indices.extend([a + s, a + next, b + s, a + next, b + next, b + s]);
                    }
                }
                RIBBONS => indices.extend([a, a + 1, b, a + 1, b + 1, b]),
                _ => indices.extend([a, b]),
            }
        }
        base += nodes * per_node;
    }

    let [lx, ly, lz] = vertices.lo.map(|v| clean_zero(f64::from(v)) as f32);
    let [hx, hy, hz] = vertices.hi.map(|v| clean_zero(f64::from(v)) as f32);
    Some([lx, ly, lz, hx, hy, hz])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_indices_and_zeroes() {
        let mut output = [0xff; 28];
        let points = [[-0.0, 0.0, 0.0], [1.0, 0.0, -2.0]];
        let bounds = pack(&points, &[2], 2, CURVES, &[], 0.0, 0.25, true, &mut output);
        assert_eq!(bounds, Some([0.0, -2.0, 0.0, 1.0, 0.0, 0.0]));
        assert_eq!(&output[..12], &[0; 12]);
        assert_eq!(&output[24..], &[0, 0, 1, 0]);
    }

    #[test]
    fn refuses_non_finite_points_and_invalid_counts() {
        let mut output = [0; 28];
        let points = [[0.0; 3], [f64::INFINITY; 3]];
        assert!(pack(&points, &[2], 2, CURVES, &[], 0.0, 0.25, false, &mut output).is_none());
        assert!(pack(
            &[[0.0; 3]; 2],
            &[3],
            2,
            CURVES,
            &[],
            0.0,
            0.25,
            false,
            &mut output
        )
        .is_none());
    }
}
