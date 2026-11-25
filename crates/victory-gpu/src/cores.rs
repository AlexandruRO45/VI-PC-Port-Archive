//! Hybrid CPU core preference helpers for victory worker threads.

use std::ffi::OsStr;
use std::sync::OnceLock;

/// Parses `SYSTEM_CPU_SET_INFORMATION` records returned by `GetSystemCpuSetInformation`,
/// keeping `(id, logical_processor_index, efficiency_class)` for group 0.
pub(crate) fn parse_cpu_sets(bytes: &[u8]) -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::SYSTEM_CPU_SET_INFORMATION;
    let mut sets = Vec::new();
    let mut at = 0usize;
    let record_size = size_of::<SYSTEM_CPU_SET_INFORMATION>();
    while at + record_size <= bytes.len() {
        // SAFETY: `at + record_size <= bytes.len()` ensures a full record is readable.
        let info = unsafe {
            bytes[at..]
                .as_ptr()
                .cast::<SYSTEM_CPU_SET_INFORMATION>()
                .read_unaligned()
        };
        if info.Size == 0 {
            break;
        }
        // SAFETY: `SYSTEM_CPU_SET_INFORMATION` on Windows carries the `CpuSet` variant.
        let set = unsafe { info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.LogicalProcessorIndex, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

/// Logical CPU sets of processor group 0 as `(id, logical_processor, efficiency_class)`.
pub(crate) fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::GetSystemCpuSetInformation;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: size query (`null_mut`, 0 bytes) writes the required byte length into `bytes`.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u8; bytes as usize];
    // SAFETY: `buffer` holds `bytes` writable bytes; `GetCurrentProcess()` is always valid.
    let ok = unsafe {
        GetSystemCpuSetInformation(
            buffer.as_mut_ptr().cast(),
            bytes,
            &mut bytes,
            GetCurrentProcess(),
            0,
        )
    };
    if ok == 0 {
        return Vec::new();
    }
    let valid = (bytes as usize).min(buffer.len());
    parse_cpu_sets(&buffer[..valid])
}

/// CPU set IDs of the highest efficiency class when the CPU has multiple classes;
/// empty if all cores belong to the same efficiency class.
pub(crate) fn fast_ids(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    let Some(best) = sets.iter().map(|c| c.2).max() else {
        return Vec::new();
    };
    let fast: Vec<u32> = sets.iter().filter(|c| c.2 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() {
        Vec::new()
    } else {
        fast
    }
}

/// CPU set IDs of slower efficiency classes when at least 8 slow logical
/// processors are available for background disk work.
pub(crate) fn slow_ids(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    let Some(best) = sets.iter().map(|c| c.2).max() else {
        return Vec::new();
    };
    let slow: Vec<u32> = sets.iter().filter(|c| c.2 < best).map(|c| c.0).collect();
    if slow.len() < 8 { Vec::new() } else { slow }
}

fn fast_cores_on_from(val: Option<&OsStr>) -> bool {
    val.is_none_or(|v| v != "0")
}

fn fast_cores_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| fast_cores_on_from(std::env::var_os("VICTORY_FAST_CORES").as_deref()))
}

fn set_thread_cpu_sets(ids: &[u32]) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    // SAFETY: `GetCurrentThread()` pseudo-handle and `ids` live across the Win32 call.
    unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), ids.len() as u32) != 0 }
}

fn apply_fast(
    what: &str,
    on: bool,
    sets: &[(u32, u8, u8)],
    pin: impl FnOnce(&[u32]) -> bool,
) -> bool {
    if !on {
        return false;
    }
    let fast = fast_ids(sets);
    if fast.is_empty() {
        return false;
    }
    if pin(&fast) {
        tracing::info!(
            what,
            fast = fast.len(),
            of = sets.len(),
            "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)"
        );
        true
    } else {
        tracing::warn!(
            what,
            error = %std::io::Error::last_os_error(),
            "the thread could not be kept on the fast cores"
        );
        false
    }
}

fn apply_slow(what: &str, sets: &[(u32, u8, u8)], pin: impl FnOnce(&[u32]) -> bool) -> bool {
    let slow = slow_ids(sets);
    if slow.is_empty() {
        return false;
    }
    let ok = pin(&slow);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            what,
            slow = slow.len(),
            of = sets.len(),
            ok,
            "helper threads are kept on the slow cores: the first one"
        );
    });
    ok
}

/// Hints the Windows scheduler to run the calling thread on performance cores
/// when running on a hybrid CPU (`VICTORY_FAST_CORES=0` disables).
pub fn keep_on_fast_cores(what: &str) {
    let _ = apply_fast(what, fast_cores_on(), &cpu_sets(), set_thread_cpu_sets);
}

/// Hints the Windows scheduler to run the calling thread on efficiency cores
/// when a hybrid CPU has at least 8 slow logical processors.
pub fn keep_on_slow_cores(what: &str) -> bool {
    apply_slow(what, &cpu_sets(), set_thread_cpu_sets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::SystemInformation::SYSTEM_CPU_SET_INFORMATION;

    fn raw_record(id: u32, group: u16, logical: u8, efficiency: u8, size: u32) -> Vec<u8> {
        let mut info: SYSTEM_CPU_SET_INFORMATION =
            // SAFETY: POD Win32 struct, zero-initialized before setting fields.
            unsafe { std::mem::zeroed() };
        info.Size = size;
        info.Anonymous.CpuSet.Id = id;
        info.Anonymous.CpuSet.Group = group;
        info.Anonymous.CpuSet.LogicalProcessorIndex = logical;
        info.Anonymous.CpuSet.EfficiencyClass = efficiency;
        let len = size_of::<SYSTEM_CPU_SET_INFORMATION>();
        // SAFETY: `info` is `len` initialized bytes on the stack.
        unsafe { std::slice::from_raw_parts((&raw const info).cast::<u8>(), len) }.to_vec()
    }

    #[test]
    fn parses_cpu_sets_and_respects_boundaries_and_groups() {
        assert!(parse_cpu_sets(&[]).is_empty());
        let sz = size_of::<SYSTEM_CPU_SET_INFORMATION>() as u32;
        let mut bytes = Vec::new();
        bytes.extend(raw_record(0x100, 0, 0, 1, sz));
        bytes.extend(raw_record(0x101, 1, 1, 0, sz)); // non-group-0 skipped
        bytes.extend(raw_record(0x102, 0, 2, 0, sz)); // exact end boundary `<=`
        assert_eq!(parse_cpu_sets(&bytes), vec![(0x100, 0, 1), (0x102, 2, 0)]);

        // Trailing truncated bytes stop cleanly; `Size == 0` breaks loop.
        bytes.push(0);
        assert_eq!(parse_cpu_sets(&bytes), vec![(0x100, 0, 1), (0x102, 2, 0)]);
        let mut zero_size = raw_record(0x200, 0, 0, 0, 0);
        zero_size.extend(raw_record(0x201, 0, 1, 1, sz));
        assert!(parse_cpu_sets(&zero_size).is_empty());
    }

    #[test]
    fn partitions_fast_and_slow_sets_only_for_hybrid_cpus_with_enough_slow_cores() {
        assert!(fast_ids(&[]).is_empty());
        assert!(slow_ids(&[]).is_empty());

        let symmetric = vec![(1, 0, 0), (2, 1, 0), (3, 2, 0)];
        assert!(fast_ids(&symmetric).is_empty());
        assert!(slow_ids(&symmetric).is_empty());

        // Hybrid CPU with 7 slow cores: too few slow cores (`< 8`), fast still partitioned.
        let mut seven_slow: Vec<(u32, u8, u8)> = (0..4).map(|i| (10 + i, i as u8, 1)).collect();
        seven_slow.extend((0..7).map(|i| (20 + i, (4 + i) as u8, 0)));
        assert_eq!(fast_ids(&seven_slow), vec![10, 11, 12, 13]);
        assert!(slow_ids(&seven_slow).is_empty());

        // 8 slow cores triggers `slow_ids` as well.
        let mut eight_slow = seven_slow;
        eight_slow.push((27, 11, 0));
        assert_eq!(
            slow_ids(&eight_slow),
            vec![20, 21, 22, 23, 24, 25, 26, 27]
        );
    }

    #[test]
    fn env_switch_and_apply_helpers_handle_opt_out_pin_success_and_failure() {
        if std::env::var_os("VICTORY_FAST_CORES_CHILD").is_some() {
            assert!(!fast_cores_on());
            return;
        }
        assert!(fast_cores_on());
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .env("VICTORY_FAST_CORES", "0")
            .env("VICTORY_FAST_CORES_CHILD", "1")
            .args([
                "--exact",
                "cores::tests::env_switch_and_apply_helpers_handle_opt_out_pin_success_and_failure",
                "--nocapture",
            ])
            .status()
            .unwrap();
        assert!(status.success());

        assert!(fast_cores_on_from(None));
        assert!(fast_cores_on_from(Some(OsStr::new("1"))));
        assert!(!fast_cores_on_from(Some(OsStr::new("0"))));

        let hybrid: Vec<(u32, u8, u8)> = (0..4)
            .map(|i| (100 + i, i as u8, 1))
            .chain((0..8).map(|i| (200 + i, (4 + i) as u8, 0)))
            .collect();
        assert!(!apply_fast("test", false, &hybrid, |_| unreachable!()));
        assert!(!apply_fast("test", true, &[(1, 0, 1)], |_| unreachable!()));

        let mut pinned = Vec::new();
        assert!(apply_fast("test", true, &hybrid, |ids| {
            pinned.extend_from_slice(ids);
            true
        }));
        assert_eq!(pinned, vec![100, 101, 102, 103]);
        assert!(!apply_fast("test", true, &hybrid, |_| false));

        pinned.clear();
        assert!(!apply_slow("disk", &[(1, 0, 1), (2, 1, 0)], |_| unreachable!()));
        assert!(apply_slow("disk", &hybrid, |ids| {
            pinned.extend_from_slice(ids);
            true
        }));
        assert_eq!(pinned, vec![200, 201, 202, 203, 204, 205, 206, 207]);
        assert!(!apply_slow("disk", &hybrid, |_| false));

        // Live Windows query + SetThreadSelectedCpuSets round-trip on a worker thread.
        let sets = cpu_sets();
        assert!(!sets.is_empty());
        let expected_slow = !slow_ids(&sets).is_empty();
        let first_id = sets[0].0;
        let (ok, invalid, slow) = std::thread::spawn(move || {
            keep_on_fast_cores("test-fast");
            let slow = keep_on_slow_cores("test-slow");
            (
                set_thread_cpu_sets(&[first_id]),
                set_thread_cpu_sets(&[u32::MAX]),
                slow,
            )
        })
        .join()
        .unwrap();
        assert!(ok);
        assert!(!invalid);
        assert_eq!(slow, expected_slow);
    }
}
