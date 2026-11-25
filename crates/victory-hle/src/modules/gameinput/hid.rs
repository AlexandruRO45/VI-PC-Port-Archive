//! The host's HID gamepads that XInput does not cover, read on one thread and
//! offered to `gameinput` as a pad in the first slot XInput leaves empty. One
//! HID pad at a time: the first device found, native drivers (`playstation`)
//! before anything generic.
//!
//! Reading is overlapped with a short wait, so a device that never sends a
//! report cannot pin the thread, and rumble goes out between reads over a
//! second handle of the same device (a write on the handle with a read
//! pending would wait behind the read).

use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    CM_GET_DEVICE_INTERFACE_LIST_PRESENT, CM_Get_Device_Interface_List_SizeW,
    CM_Get_Device_Interface_ListW, CR_SUCCESS,
};
use windows_sys::Win32::Devices::HumanInterfaceDevice::{
    HIDP_CAPS, HidD_FreePreparsedData, HidD_GetFeature, HidD_GetHidGuid, HidD_GetPreparsedData,
    HidP_GetCaps,
};
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE, GetLastError, HANDLE,
    INVALID_HANDLE_VALUE, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    WriteFile,
};
use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, WaitForSingleObject};

use super::{PadState, playstation};

/// How often to look for a pad while none is open.
const RESCAN: Duration = Duration::from_secs(1);
/// How long a read waits before the thread looks at rumble and the device.
const WAIT_MS: u32 = 20;

/// A reading and the pad's vendor and product ids.
pub(super) type Reading = (PadState, (u16, u16));

static LATEST: Mutex<Option<Reading>> = Mutex::new(None);
/// What the title asked of the rumble motors: low frequency, high frequency.
static RUMBLE: Mutex<(u16, u16)> = Mutex::new((0, 0));

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The pad's newest reading while one is open; starts the reader thread the
/// first time it is asked.
pub(super) fn connected_pad() -> Option<Reading> {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if let Err(e) = std::thread::Builder::new().name("hid-pad".into()).spawn(run) {
            tracing::warn!(error = %e, "no HID pad reader thread");
        }
    });
    *lock(&LATEST)
}

/// Rumble for the open pad (the title's 16-bit speeds); sent by the reader
/// thread, which sees it within a few milliseconds.
pub(super) fn set_rumble(low: u16, high: u16) {
    *lock(&RUMBLE) = (low, high);
}

/// What drives a device.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Driver {
    Sony(playstation::Kind),
}

struct Found {
    path: Vec<u16>,
    driver: Driver,
    vendor: u16,
    product: u16,
    bluetooth: bool,
}

/// `vid_054c&pid_0ce6` over USB, `vid&0002054c_pid&0ce6` over Bluetooth.
fn ids_of(path: &str) -> Option<(u16, u16)> {
    let path = path.to_ascii_lowercase();
    let hex = |at: usize| u16::from_str_radix(path.get(at..at + 4)?, 16).ok();
    let vid = path.find("vid_").map(|i| i + 4).or_else(|| path.find("vid&").map(|i| i + 4 + 4))?;
    let pid = path.find("pid_").map(|i| i + 4).or_else(|| path.find("pid&").map(|i| i + 4))?;
    Some((hex(vid)?, hex(pid)?))
}

/// Bluetooth HID devices carry the HID service's UUID in their path.
fn is_bluetooth(path: &str) -> bool {
    path.to_ascii_lowercase().contains("{00001124-0000-1000-8000-00805f9b34fb}")
}

fn devices() -> Vec<Found> {
    // SAFETY: plain queries into buffers we size from the answer.
    unsafe {
        let mut guid = std::mem::zeroed();
        HidD_GetHidGuid(&mut guid);
        let mut len = 0u32;
        let flags = CM_GET_DEVICE_INTERFACE_LIST_PRESENT;
        if CM_Get_Device_Interface_List_SizeW(&mut len, &guid, std::ptr::null(), flags) != CR_SUCCESS
            || len < 2
        {
            return Vec::new();
        }
        let mut buf = vec![0u16; len as usize];
        if CM_Get_Device_Interface_ListW(&guid, std::ptr::null(), buf.as_mut_ptr(), len, flags)
            != CR_SUCCESS
        {
            return Vec::new();
        }
        buf.split(|&c| c == 0)
            .filter(|p| !p.is_empty())
            .filter_map(|p| {
                let text = String::from_utf16_lossy(p);
                let (vendor, product) = ids_of(&text)?;
                let driver = Driver::Sony(playstation::kind_of(vendor, product)?);
                Some(Found {
                    path: p.iter().copied().chain(Some(0)).collect(),
                    driver,
                    vendor,
                    product,
                    bluetooth: is_bluetooth(&text),
                })
            })
            .collect()
    }
}

fn run() {
    loop {
        for found in devices() {
            if read_pad(&found) {
                break;
            }
        }
        *lock(&LATEST) = None;
        std::thread::sleep(RESCAN);
    }
}

fn open(path: &[u16], access: u32, flags: u32) -> HANDLE {
    // SAFETY: a NUL-terminated path.
    unsafe {
        CreateFileW(
            path.as_ptr(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            flags,
            std::ptr::null_mut(),
        )
    }
}

/// The report sizes of a device: input, output.
fn report_sizes(handle: HANDLE) -> Option<(usize, usize)> {
    // SAFETY: preparsed data from this handle, freed after the query.
    unsafe {
        let mut pp = 0isize;
        if !HidD_GetPreparsedData(handle, &mut pp) {
            return None;
        }
        let mut caps: HIDP_CAPS = std::mem::zeroed();
        let ok = HidP_GetCaps(pp, &mut caps) >= 0;
        HidD_FreePreparsedData(pp);
        ok.then(|| (usize::from(caps.InputReportByteLength), usize::from(caps.OutputReportByteLength)))
    }
}

/// A feature report, `size` bytes with `id` first; `None` when the device
/// does not answer.
fn feature(handle: HANDLE, id: u8, size: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; size];
    buf[0] = id;
    // SAFETY: a buffer of `size` bytes.
    let ok = unsafe { HidD_GetFeature(handle, buf.as_mut_ptr().cast(), size as u32) };
    ok.then_some(buf)
}

/// Sends rumble to a Sony pad.
struct Rumble {
    control: HANDLE,
    kind: playstation::Kind,
    bluetooth: bool,
    new_vibration: bool,
    output_len: usize,
    sequence: u8,
    sent: (u16, u16),
}

impl Rumble {
    fn send(&mut self, low: u16, high: u16) {
        let mut report = playstation::rumble_report(
            self.kind,
            self.bluetooth,
            low,
            high,
            self.sequence,
            self.new_vibration,
        );
        self.sequence = (self.sequence + 1) & 0x0F;
        // The device wants its own report size.
        report.resize(self.output_len.max(report.len()), 0);
        report.truncate(self.output_len.max(1));
        let mut written = 0u32;
        // SAFETY: the report's bytes and a synchronous handle of our own.
        let ok = unsafe {
            WriteFile(
                self.control,
                report.as_ptr(),
                report.len() as u32,
                &mut written,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            tracing::debug!(error = unsafe { GetLastError() }, "rumble write failed");
        }
        self.sent = (low, high);
    }

    /// Sends what the title wants if it changed.
    fn update(&mut self) {
        let wanted = *lock(&RUMBLE);
        if wanted != self.sent {
            self.send(wanted.0, wanted.1);
        }
    }
}

/// Reads one pad until it fails (unplugged, switched off); whether it could be
/// opened and read at all.
fn read_pad(found: &Found) -> bool {
    let read = open(&found.path, GENERIC_READ, FILE_FLAG_OVERLAPPED);
    if read == INVALID_HANDLE_VALUE {
        return false;
    }
    // Feature reports and writes: a second, synchronous handle.
    let mut control = open(&found.path, GENERIC_READ | GENERIC_WRITE, 0);
    if control == INVALID_HANDLE_VALUE {
        control = open(&found.path, GENERIC_WRITE, 0);
    }
    let close_all = |read: HANDLE, control: HANDLE| {
        // SAFETY: our handles.
        unsafe {
            CloseHandle(read);
            if control != INVALID_HANDLE_VALUE {
                CloseHandle(control);
            }
        }
    };
    let Some((size, output_len)) = report_sizes(read).filter(|&(input, _)| input > 0) else {
        close_all(read, control);
        return false;
    };
    let Driver::Sony(kind) = found.driver;
    let mut rumble = None;
    if control != INVALID_HANDLE_VALUE {
        // Over Bluetooth a pad sends a short report, and ignores output,
        // until a title reads its calibration feature report (0x05).
        if found.bluetooth || kind == playstation::Kind::DualSense {
            let _ = feature(control, 0x05, 41);
        }
        // DualSense firmware before update 2.21 wants the old rumble bit.
        let new_vibration = match (kind, found.product) {
            (playstation::Kind::DualSense, 0x0CE6) => feature(control, 0x20, 64)
                .map_or(true, |f| u16::from_le_bytes([f[44], f[45]]) >= 0x0215),
            _ => true,
        };
        if output_len > 0 {
            rumble = Some(Rumble {
                control,
                kind,
                bluetooth: found.bluetooth,
                new_vibration,
                output_len,
                sequence: 0,
                sent: (0, 0),
            });
        }
    }
    tracing::info!(
        ?kind,
        vendor = found.vendor,
        product = found.product,
        bluetooth = found.bluetooth,
        report_bytes = size,
        rumble = rumble.is_some(),
        "HID pad connected"
    );
    let mut buf = vec![0u8; size];
    let mut state: Option<PadState> = None;
    let mut first = true;
    // SAFETY: an event of our own, closed below.
    let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
    'reading: loop {
        let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
        overlapped.hEvent = event;
        // SAFETY: the buffer and `overlapped` outlive the read: it is waited
        // for or cancelled and waited for before they go out of scope.
        let started = unsafe {
            ResetEvent(event);
            ReadFile(read, buf.as_mut_ptr().cast(), size as u32, std::ptr::null_mut(), &mut overlapped)
        };
        // SAFETY: a plain query.
        if started == 0 && unsafe { GetLastError() } != ERROR_IO_PENDING {
            break;
        }
        loop {
            // SAFETY: our event.
            match unsafe { WaitForSingleObject(event, WAIT_MS) } {
                WAIT_OBJECT_0 => break,
                WAIT_TIMEOUT => {
                    if let Some(r) = rumble.as_mut() {
                        r.update();
                    }
                }
                _ => {
                    cancel(read, &mut overlapped);
                    break 'reading;
                }
            }
        }
        let mut got = 0u32;
        // SAFETY: the finished read of this handle.
        if unsafe { GetOverlappedResult(read, &overlapped, &mut got, 0) } == 0 {
            break;
        }
        let report = &buf[..got as usize];
        if first {
            tracing::info!(id = report.first().copied().unwrap_or(0), bytes = got, "HID pad first report");
            first = false;
        }
        let Driver::Sony(kind) = found.driver;
        let now = playstation::after(kind, report, state);
        if state.is_some() && now.is_none() {
            tracing::info!("PlayStation adapter: its pad left");
        }
        state = now;
        *lock(&LATEST) = state.map(|pad| (pad, (found.vendor, found.product)));
        if let Some(r) = rumble.as_mut() {
            r.update();
        }
    }
    // Leave the motors still, then let go.
    if let Some(r) = rumble.as_mut()
        && r.sent != (0, 0)
    {
        r.send(0, 0);
    }
    tracing::info!("HID pad disconnected");
    // SAFETY: our event.
    unsafe { CloseHandle(event) };
    close_all(read, control);
    true
}

/// Cancels the read of `overlapped` and waits for it to finish, so its buffer
/// can go.
fn cancel(handle: HANDLE, overlapped: &mut OVERLAPPED) {
    let mut got = 0u32;
    // SAFETY: the pending read of this handle.
    unsafe {
        CancelIoEx(handle, overlapped);
        GetOverlappedResult(handle, overlapped, &mut got, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_come_out_of_usb_and_bluetooth_paths() {
        assert_eq!(ids_of(r"\\?\hid#vid_054c&pid_0ce6#7&1#{4d1e55b2}"), Some((0x054C, 0x0CE6)));
        assert_eq!(
            ids_of(r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&05c4#a"),
            Some((0x054C, 0x05C4))
        );
        assert_eq!(ids_of(r"\\?\hid#nothing"), None);
    }

    #[test]
    fn bluetooth_paths_are_told_apart() {
        assert!(is_bluetooth(r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&0ce6#a"));
        assert!(!is_bluetooth(r"\\?\hid#vid_054c&pid_0ce6#7&1"));
    }
}
