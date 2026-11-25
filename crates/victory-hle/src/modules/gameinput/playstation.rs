//! DualSense (PS5) and DualShock 4 (PS4) pads, read straight from HID over USB
//! or Bluetooth, so no XInput translator (DS4Windows, Steam Input) is needed.
//! (Xbox pads already come through XInput.) A reader thread keeps the newest
//! reading of the first pad it finds; `gameinput` lists it in the first host
//! slot XInput leaves empty.
//!
//! Reports (`data` below is the report with its id byte, and for the
//! Bluetooth 0x31 report its extra header byte, removed):
//! - USB `0x01` (64 bytes) and Bluetooth `0x31` (78 bytes): left stick x, y,
//!   right stick x, y, L2, R2, counter, then three button bytes.
//! - Bluetooth `0x01`, the simple report a pad sends until a title asks for
//!   its calibration (we never do): left stick x, y, right stick x, y, three
//!   button bytes, L2, R2.
//! - DualShock 4: the same simple layout in its USB `0x01` (64 bytes) and its
//!   Bluetooth `0x01` (10 bytes) and `0x11` (78 bytes, two header bytes after
//!   the id) reports.
//!
//! Sticks are 0 to 255 with 128 in the middle and up at 0.

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

/// The pads we read, by product id: DualSense, DualSense Edge; DualShock 4
/// (two revisions and the USB adapter).
#[cfg_attr(not(windows), allow(dead_code))]
const PRODUCTS: [(&str, Kind); 5] = [
    ("0ce6", Kind::DualSense),
    ("0df2", Kind::DualSense),
    ("05c4", Kind::DualShock4),
    ("09cc", Kind::DualShock4),
    ("0ba0", Kind::DualShock4),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Kind {
    DualSense,
    DualShock4,
}

/// The report's three button bytes, as the pad lays them out.
struct Buttons {
    face: u8,
    shoulder: u8,
    system: u8,
}

fn stick(v: u8) -> f32 {
    ((f32::from(v) - 128.0) / 127.0).clamp(-1.0, 1.0)
}

fn decode(sticks: [u8; 4], triggers: [u8; 2], b: &Buttons) -> PadState {
    // Hat in the low nibble: 0 up, then clockwise by 45 degrees; 8 is centred.
    let hat = match b.face & 0x0F {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => DOWN | RIGHT,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => UP | LEFT,
        _ => 0,
    };
    let table = [
        (b.face, 0x10, X), // square
        (b.face, 0x20, A), // cross
        (b.face, 0x40, B), // circle
        (b.face, 0x80, Y), // triangle
        (b.shoulder, 0x01, LEFT_SHOULDER),
        (b.shoulder, 0x02, RIGHT_SHOULDER),
        (b.shoulder, 0x10, VIEW), // create
        (b.shoulder, 0x20, MENU), // options
        (b.shoulder, 0x40, LEFT_STICK),
        (b.shoulder, 0x80, RIGHT_STICK),
    ];
    let buttons = table
        .iter()
        .filter(|(byte, mask, _)| byte & mask != 0)
        .fold(hat, |all, (_, _, bit)| all | bit);
    let _ = b.system; // PS and touchpad buttons: the title has no gamepad bit for them
    PadState {
        buttons,
        left_trigger: f32::from(triggers[0]) / 255.0,
        right_trigger: f32::from(triggers[1]) / 255.0,
        left_x: stick(sticks[0]),
        left_y: -stick(sticks[1]),
        right_x: stick(sticks[2]),
        right_y: -stick(sticks[3]),
    }
}

/// A reading from one input report; `None` for a report that is no pad state
/// (feedback, a short or unknown one).
pub(super) fn parse(kind: Kind, report: &[u8]) -> Option<PadState> {
    let (&id, rest) = report.split_first()?;
    match (kind, id) {
        // DualSense over USB: the data follows the id.
        (Kind::DualSense, 0x01) if report.len() == 64 => full(rest),
        // DualSense over Bluetooth, full: a header byte follows the id.
        (Kind::DualSense, 0x31) if report.len() >= 12 => full(&rest[1..]),
        // The simple layout: a DualSense over Bluetooth, a DualShock 4 over
        // USB or Bluetooth (the read may be padded to the longest report).
        (_, 0x01) if report.len() >= 10 => Some(simple(rest)),
        // DualShock 4 over Bluetooth, full: two header bytes follow the id.
        (Kind::DualShock4, 0x11) if report.len() >= 14 => Some(simple(&rest[2..])),
        _ => None,
    }
}

fn simple(d: &[u8]) -> PadState {
    decode(
        [d[0], d[1], d[2], d[3]],
        [d[7], d[8]],
        &Buttons {
            face: d[4],
            shoulder: d[5],
            system: d[6],
        },
    )
}

fn full(data: &[u8]) -> Option<PadState> {
    let d = data.get(..10)?;
    Some(decode(
        [d[0], d[1], d[2], d[3]],
        [d[4], d[5]],
        &Buttons {
            face: d[7],
            shoulder: d[8],
            system: d[9],
        },
    ))
}

#[cfg(windows)]
pub(super) use reader::connected_pad;

#[cfg(windows)]
mod reader {
    use std::sync::{Mutex, OnceLock, PoisonError};
    use std::time::Duration;

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_GET_DEVICE_INTERFACE_LIST_PRESENT, CM_Get_Device_Interface_List_SizeW,
        CM_Get_Device_Interface_ListW, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDP_CAPS, HidD_FreePreparsedData, HidD_GetHidGuid, HidD_GetPreparsedData, HidP_GetCaps,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    };

    use super::*;

    /// How often to look for a pad while none is open.
    const RESCAN: Duration = Duration::from_secs(1);

    /// Sony's USB vendor id.
    const SONY: u16 = 0x054C;

    /// A reading and the pad's vendor and product ids.
    type Reading = (PadState, (u16, u16));

    static LATEST: Mutex<Option<Reading>> = Mutex::new(None);

    fn set(state: Option<Reading>) {
        *LATEST.lock().unwrap_or_else(PoisonError::into_inner) = state;
    }

    /// The pad's newest reading while one is open; starts the reader thread
    /// the first time it is asked.
    pub(in super::super) fn connected_pad() -> Option<Reading> {
        static STARTED: OnceLock<()> = OnceLock::new();
        STARTED.get_or_init(|| {
            let spawned = std::thread::Builder::new()
                .name("dualsense".into())
                .spawn(run);
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no PlayStation pad reader thread");
            }
        });
        *LATEST.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The kind of pad a HID device path names, if it is one we read.
    pub(super) fn kind_of(path: &str) -> Option<(Kind, u16)> {
        let path = path.to_ascii_lowercase();
        // `vid_054c&pid_0ce6` over USB, `vid&0002054c_pid&0ce6` over Bluetooth.
        if !path.contains("054c") {
            return None;
        }
        PRODUCTS
            .iter()
            .find(|(id, _)| {
                path.contains(&format!("pid_{id}")) || path.contains(&format!("pid&{id}"))
            })
            .and_then(|&(id, kind)| Some((kind, u16::from_str_radix(id, 16).ok()?)))
    }

    /// Device paths of the present HID devices that look like a supported pad.
    fn paths() -> Vec<(Vec<u16>, Kind, u16)> {
        // SAFETY: plain queries into buffers we size from the answer.
        unsafe {
            let mut guid = std::mem::zeroed();
            HidD_GetHidGuid(&mut guid);
            let mut len = 0u32;
            let flags = CM_GET_DEVICE_INTERFACE_LIST_PRESENT;
            if CM_Get_Device_Interface_List_SizeW(&mut len, &guid, std::ptr::null(), flags)
                != CR_SUCCESS
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
                    let (kind, product) = kind_of(&String::from_utf16_lossy(p))?;
                    Some((p.iter().copied().chain(Some(0)).collect(), kind, product))
                })
                .collect()
        }
    }

    fn run() {
        loop {
            for (path, kind, product) in paths() {
                if read_from(&path, kind, product) {
                    break;
                }
            }
            set(None);
            std::thread::sleep(RESCAN);
        }
    }

    /// Reads one pad until it fails (unplugged, switched off); whether it
    /// could be opened and read at all.
    fn read_from(path: &[u16], kind: Kind, product: u16) -> bool {
        // SAFETY: a NUL-terminated path; the handle is closed below.
        let handle = unsafe {
            let open = |access| {
                CreateFileW(
                    path.as_ptr(),
                    access,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    0,
                    std::ptr::null_mut(),
                )
            };
            let h = open(GENERIC_READ | GENERIC_WRITE);
            if h == INVALID_HANDLE_VALUE {
                open(GENERIC_READ)
            } else {
                h
            }
        };
        if handle == INVALID_HANDLE_VALUE {
            return false;
        }
        // The size every read must have: the longest input report.
        let mut size = 0usize;
        // SAFETY: preparsed data from this handle, freed after the query.
        unsafe {
            let mut pp = 0isize;
            if HidD_GetPreparsedData(handle, &mut pp) {
                let mut caps: HIDP_CAPS = std::mem::zeroed();
                if HidP_GetCaps(pp, &mut caps) >= 0 {
                    size = usize::from(caps.InputReportByteLength);
                }
                HidD_FreePreparsedData(pp);
            }
        }
        if size == 0 {
            // SAFETY: our handle.
            unsafe { CloseHandle(handle) };
            return false;
        }
        tracing::info!(?kind, report_bytes = size, "PlayStation pad connected");
        let mut buf = vec![0u8; size];
        let mut first = true;
        loop {
            let mut read = 0u32;
            // SAFETY: a buffer of `size` bytes and our handle; blocks until a report.
            let ok = unsafe {
                ReadFile(
                    handle,
                    buf.as_mut_ptr().cast(),
                    size as u32,
                    &mut read,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                break;
            }
            let report = &buf[..read as usize];
            if first {
                tracing::info!(
                    id = report.first().copied().unwrap_or(0),
                    bytes = read,
                    "PlayStation pad first report"
                );
                first = false;
            }
            if let Some(state) = parse(kind, report) {
                set(Some((state, (SONY, product))));
            }
        }
        tracing::info!("PlayStation pad disconnected");
        // SAFETY: our handle.
        unsafe { CloseHandle(handle) };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usb(sticks: [u8; 4], triggers: [u8; 2], face: u8, shoulder: u8) -> Vec<u8> {
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&sticks);
        r[5..7].copy_from_slice(&triggers);
        r[8] = face;
        r[9] = shoulder;
        r
    }

    #[test]
    fn centred_pad_reads_neutral() {
        let pad = parse(Kind::DualSense, &usb([128; 4], [0, 0], 0x08, 0)).unwrap();
        assert_eq!(pad, PadState::default());
    }

    #[test]
    fn usb_buttons_sticks_and_triggers() {
        // cross + circle, L1 + options, hat up-right; left stick left, right stick up
        let pad = parse(
            Kind::DualSense,
            &usb([0, 128, 128, 0], [255, 51], 0x61, 0x21),
        )
        .unwrap();
        assert_eq!(pad.buttons, A | B | UP | RIGHT | LEFT_SHOULDER | MENU);
        assert_eq!(pad.left_x, -1.0);
        assert_eq!(pad.left_y, 0.0);
        assert_eq!(pad.right_y, 1.0);
        assert_eq!(pad.left_trigger, 1.0);
        assert!((pad.right_trigger - 0.2).abs() < 1e-6);
    }

    #[test]
    fn bluetooth_full_report_has_a_header_byte() {
        let mut r = vec![0u8; 78];
        r[0] = 0x31;
        r[2..6].copy_from_slice(&[255, 128, 128, 255]);
        r[6..8].copy_from_slice(&[10, 20]);
        r[9] = 0x08 | 0x80; // triangle, hat centred
        r[10] = 0x10 | 0x80; // create + R3
        let pad = parse(Kind::DualSense, &r).unwrap();
        assert_eq!(pad.buttons, Y | VIEW | RIGHT_STICK);
        assert_eq!(pad.left_x, 1.0);
        assert_eq!(pad.right_y, -1.0);
        assert!((pad.left_trigger - 10.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn bluetooth_simple_report_with_and_without_padding() {
        let mut r = vec![0u8; 10];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&[128, 0, 128, 128]);
        r[5] = 0x08 | 0x10; // square, hat centred
        r[6] = 0x02; // R1
        r[8] = 255;
        let pad = parse(Kind::DualSense, &r).unwrap();
        assert_eq!(pad.buttons, X | RIGHT_SHOULDER);
        assert_eq!(pad.left_y, 1.0);
        assert_eq!(pad.left_trigger, 1.0);
        r.resize(78, 0);
        assert_eq!(parse(Kind::DualSense, &r).unwrap(), pad);
    }

    #[test]
    fn every_hat_position() {
        let want = [
            UP,
            UP | RIGHT,
            RIGHT,
            DOWN | RIGHT,
            DOWN,
            DOWN | LEFT,
            LEFT,
            UP | LEFT,
            0,
        ];
        for (hat, buttons) in want.into_iter().enumerate() {
            assert_eq!(
                parse(Kind::DualSense, &usb([128; 4], [0, 0], hat as u8, 0))
                    .unwrap()
                    .buttons,
                buttons
            );
        }
    }

    #[test]
    fn other_reports_are_no_pad_state() {
        assert_eq!(parse(Kind::DualSense, &[]), None);
        assert_eq!(parse(Kind::DualSense, &[0x02; 78]), None);
        assert_eq!(parse(Kind::DualSense, &[0x01; 5]), None);
    }

    #[test]
    fn dualshock_4_reports() {
        // USB: 64 bytes, the simple layout. Square + options, hat down, L2 half.
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&[128, 128, 255, 128]);
        r[5] = 0x04 | 0x10;
        r[6] = 0x20;
        r[8] = 128;
        let usb = parse(Kind::DualShock4, &r).unwrap();
        assert_eq!(usb.buttons, X | DOWN | MENU);
        assert_eq!(usb.right_x, 1.0);
        assert!((usb.left_trigger - 128.0 / 255.0).abs() < 1e-6);
        // Bluetooth full: two header bytes after the id, then the same.
        let mut b = vec![0u8; 78];
        b[0] = 0x11;
        b[3..7].copy_from_slice(&[128, 128, 255, 128]);
        b[7] = 0x04 | 0x10;
        b[8] = 0x20;
        b[10] = 128;
        assert_eq!(parse(Kind::DualShock4, &b).unwrap(), usb);
        // The same 64 bytes read as a DualSense are another layout.
        assert_ne!(parse(Kind::DualSense, &r).unwrap(), usb);
    }

    #[cfg(windows)]
    #[test]
    fn device_paths_name_the_pad() {
        use super::reader::kind_of;
        assert_eq!(
            kind_of(r"\\?\hid#vid_054c&pid_0ce6#7&1#{4d1e55b2}"),
            Some((Kind::DualSense, 0x0CE6))
        );
        assert_eq!(
            kind_of(r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&05c4#a"),
            Some((Kind::DualShock4, 0x05C4))
        );
        assert_eq!(kind_of(r"\\?\hid#vid_046d&pid_c52b#7&1"), None);
        assert_eq!(kind_of(r"\\?\hid#vid_054c&pid_0000#7&1"), None);
    }
}
