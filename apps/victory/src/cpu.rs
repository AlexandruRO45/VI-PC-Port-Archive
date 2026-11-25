//! An optional CPU ceiling for the whole process: a job object's hard rate
//! cap, so a long translate or run leaves the machine responsive.
//! `--cpu-percent` (`VICTORY_CPU_PERCENT`), 1 to 100; unset means no cap.

/// Caps this process at `percent` of the machine's CPU, if asked.
#[cfg(windows)]
pub fn limit(percent: Option<u8>) {
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_CPU_RATE_CONTROL_ENABLE,
        JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP, JOBOBJECT_CPU_RATE_CONTROL_INFORMATION,
        JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0, JobObjectCpuRateControlInformation,
        SetInformationJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let Some(percent) = percent else { return };
    let info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
        ControlFlags: JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        Anonymous: JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0 {
            // In hundredths of a percent, over the whole machine.
            CpuRate: u32::from(percent) * 100,
        },
    };
    // SAFETY: a job object handle and a sized information structure; the
    // handle is kept for the process's life so the cap does.
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            tracing::warn!(percent, "no CPU limit: no job object");
            return;
        }
        let set = SetInformationJobObject(
            job,
            JobObjectCpuRateControlInformation,
            std::ptr::from_ref(&info).cast(),
            size_of_val(&info) as u32,
        );
        if set == 0 {
            tracing::warn!(percent, "no CPU limit: the job object refused the rate");
            let _ = windows_sys::Win32::Foundation::CloseHandle(job);
            return;
        }
        if AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
            // A process already in a job this one cannot nest under (some
            // launchers, CI): the limit cannot be set.
            tracing::warn!(
                percent,
                "no CPU limit: this process is already in a job that does not allow it"
            );
            let _ = windows_sys::Win32::Foundation::CloseHandle(job);
            return;
        }
        // The handle stays open (nothing closes it): closing the last one
        // would end the job and its limit.
    }
    tracing::info!(percent, "CPU use capped");
}

#[cfg(not(windows))]
pub fn limit(_percent: Option<u8>) {}
