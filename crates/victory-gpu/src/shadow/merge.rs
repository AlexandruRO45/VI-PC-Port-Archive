//! Merge GPU writes into the current title bytes without losing CPU writes.
//! The caller has already read title memory and identified a changed piece.
//! Keep the original bytewise precedence, including when both sides wrote.

/// Returns whether title bytes changed, and synchronizes the host copy.
pub(super) fn merge(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    assert_eq!(title.len(), host.len());
    assert_eq!(title.len(), old.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2; all three slices have equal length.
        return unsafe { avx2(title, host, old) };
    }
    scalar(title, host, old)
}

fn scalar(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut changed = false;
    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
        if *h != *o && *t != *h {
            *t = *h;
            changed = true;
        }
        *h = *t;
    }
    changed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let end = title.len() / 32 * 32;
    let mut changed = _mm256_setzero_si256();
    for at in (0..end).step_by(32) {
        // SAFETY: each offset has 32 accessible bytes in all three
        // equally sized slices. Unaligned loads/stores are intentional.
        // The mutable slices are disjoint by Rust's borrowing rules.
        unsafe {
            let t = _mm256_loadu_si256(title.as_ptr().add(at).cast());
            let h = _mm256_loadu_si256(host.as_ptr().add(at).cast());
            let o = _mm256_loadu_si256(old.as_ptr().add(at).cast());
            // A byte the GPU did not change keeps the CPU's current byte.
            let unchanged = _mm256_cmpeq_epi8(h, o);
            let merged = _mm256_blendv_epi8(h, t, unchanged);
            changed = _mm256_or_si256(changed, _mm256_xor_si256(merged, t));
            _mm256_storeu_si256(title.as_mut_ptr().add(at).cast(), merged);
            _mm256_storeu_si256(host.as_mut_ptr().add(at).cast(), merged);
        }
    }
    let tail = scalar(&mut title[end..], &mut host[end..], &old[end..]);
    _mm256_testz_si256(changed, changed) == 0 || tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(seed: &mut u32) -> u8 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed as u8
    }

    #[test]
    fn merge_preserves_cpu_bytes_and_gpu_write_precedence() {
        let mut title = [7, 8, 9, 10, 11, 12];
        let mut host = [1, 2, 9, 4, 99, 12];
        let old = [1, 3, 3, 4, 5, 6];
        assert!(merge(&mut title, &mut host, &old));
        assert_eq!(title, [7, 2, 9, 10, 99, 12]);
        assert_eq!(host, title);
    }

    #[test]
    fn merge_reports_no_title_change_when_gpu_matches_title() {
        let mut title = [5; 65];
        let mut host = [5; 65];
        assert!(!merge(&mut title, &mut host, &[0; 65]));
        let mut host = [1; 65];
        assert!(!merge(&mut title, &mut host, &[1; 65]));
        assert_eq!(host, title);
    }

    #[test]
    fn merge_matches_scalar_with_unaligned_vectors_and_tails() {
        let mut seed = 0x1234_abcd;
        for len in (0..=129).chain([255, 256, 257, 4095, 4096, 4097, 65535, 65536]) {
            for offset in 0..32 {
                let mut title = vec![0x5a; len + 64];
                let mut host = vec![0xa5; len + 64];
                let mut old = vec![0; len + 64];
                for i in 0..len {
                    let at = offset + i;
                    old[at] = next(&mut seed);
                    title[at] = next(&mut seed);
                    host[at] = match i % 4 {
                        0 => old[at],
                        1 => title[at],
                        _ => next(&mut seed),
                    };
                }
                let mut expected_title = title.clone();
                let mut expected_host = host.clone();
                let range = offset..offset + len;
                let changed = scalar(
                    &mut expected_title[range.clone()],
                    &mut expected_host[range.clone()],
                    &old[range.clone()],
                );
                assert_eq!(
                    merge(&mut title[range.clone()], &mut host[range.clone()], &old[range]),
                    changed
                );
                // Also checks that bytes outside the view remain untouched.
                assert_eq!(title, expected_title);
                assert_eq!(host, expected_host);
            }
        }
    }
}
