//! DualShock 4 pads (and the many third-party pads that report themselves as
//! one: Sony's vendor id 0x054C) read straight from HID, for hosts where they
//! are not XInput devices. Each pad gets a reader thread doing blocking reads;
//! `pad(n)` hands out the n-th pad's latest state in the console's layout.
//! Off with `VICTORY_DUALSHOCK=0`.

use super::*;

/// Sony's vendor id and the DualShock 4 product ids: first model, second
/// model, the wireless adapter (and its firmware-update mode).
const VENDOR: u16 = 0x054C;
const PRODUCTS: [u16; 4] = [0x05C4, 0x09CC, 0x0BA0, 0x0BA1];

/// Face and system buttons in the report's fifth and sixth data bytes:
/// (byte, bit, console button). Cross/circle/square/triangle sit where
/// A/B/X/Y do; Options is Menu; Share is View.
const BUTTONS: [(usize, u8, u32); 10] = [
    (4, 0x10, X), // square
    (4, 0x20, A), // cross
    (4, 0x40, B), // circle
    (4, 0x80, Y), // triangle
    (5, 0x01, LEFT_SHOULDER),
    (5, 0x02, RIGHT_SHOULDER),
    (5, 0x10, VIEW), // share
    (5, 0x20, MENU), // options
    (5, 0x40, LEFT_STICK),
    (5, 0x80, RIGHT_STICK),
];

/// The d-pad hat (low nibble of the fifth data byte): 0 is up, then
/// clockwise in eighths; 8 and above is released.
const HAT: [u32; 8] = [UP, UP | RIGHT, RIGHT, DOWN | RIGHT, DOWN, DOWN | LEFT, LEFT, UP | LEFT];

/// One input report as `ReadFile` hands it over (report id first) to a
/// console pad state. USB and short Bluetooth reports (id 0x01) carry the
/// data from byte 1; full Bluetooth reports (id 0x11) from byte 3.
pub(super) fn parse(report: &[u8]) -> Option<PadState> {
    let start = match report.first()? {
        0x01 => 1,
        0x11 => 3,
        _ => return None,
    };
    let d = report.get(start..start + 9)?;
    // Sticks: 0..255 with 128 at rest, down positive; the console's up is.
    let axis = |v: u8| ((f32::from(v) - 128.0) / 127.0).clamp(-1.0, 1.0);
    let mut buttons = BUTTONS
        .iter()
        .filter(|&&(byte, bit, _)| d[byte] & bit != 0)
        .fold(0, |all, &(_, _, button)| all | button);
    if let Some(&dpad) = HAT.get(usize::from(d[4] & 0x0F)) {
        buttons |= dpad;
    }
    // The touchpad's click (seventh data byte, bit 1) is the map on the
    // console's own pad; View is the closest button.
    if d[6] & 0x02 != 0 {
        buttons |= VIEW;
    }
    Some(PadState {
        buttons,
        left_trigger: f32::from(d[7]) / 255.0,
        right_trigger: f32::from(d[8]) / 255.0,
        left_x: axis(d[0]),
        left_y: -axis(d[1]),
        right_x: axis(d[2]),
        right_y: -axis(d[3]),
    })
}

pub(super) fn is_dualshock(vendor: u16, product: u16) -> bool {
    vendor == VENDOR && PRODUCTS.contains(&product)
}

#[cfg(windows)]
pub(super) use host::pad;

#[cfg(windows)]
mod host {
    use std::collections::HashSet;
    use std::sync::{LazyLock, Mutex};
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA,
        SP_DEVICE_INTERFACE_DETAIL_DATA_W, SetupDiDestroyDeviceInfoList,
        SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW, SetupDiGetDeviceInterfaceDetailW,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HidD_GetAttributes, HidD_GetHidGuid,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    };

    use super::*;

    /// How often to look for pads plugged in since.
    const RESCAN: Duration = Duration::from_secs(2);

    struct Pads {
        /// Latest state per reader, `None` once its pad is gone.
        states: Vec<Option<PadState>>,
        /// Device paths with a reader running.
        open: HashSet<Vec<u16>>,
        scanned: Option<Instant>,
    }

    static PADS: LazyLock<Mutex<Pads>> =
        LazyLock::new(|| Mutex::new(Pads { states: Vec::new(), open: HashSet::new(), scanned: None }));

    fn enabled() -> bool {
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("VICTORY_DUALSHOCK").is_none_or(|v| v != "0"));
        *ON
    }

    /// The `n`-th connected DualShock 4's state, in the order they were
    /// found (a pad that went away leaves the others where they are).
    pub(in super::super) fn pad(n: usize) -> Option<PadState> {
        if !enabled() {
            return None;
        }
        let mut pads = PADS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if pads.scanned.is_none_or(|at| at.elapsed() >= RESCAN) {
            pads.scanned = Some(Instant::now());
            scan(&mut pads);
        }
        pads.states.iter().flatten().nth(n).copied()
    }

    fn scan(pads: &mut Pads) {
        for path in hid_paths() {
            if pads.open.contains(&path) {
                continue;
            }
            let Some(handle) = open(&path) else { continue };
            let mut attributes = HIDD_ATTRIBUTES { Size: size_of::<HIDD_ATTRIBUTES>() as u32, ..Default::default() };
            // SAFETY: an open device and room for its attributes.
            let known = unsafe { HidD_GetAttributes(handle, &mut attributes) }
                && is_dualshock(attributes.VendorID, attributes.ProductID);
            if !known {
                // SAFETY: ours to close.
                unsafe { CloseHandle(handle) };
                continue;
            }
            let index = pads.states.len();
            pads.states.push(None);
            pads.open.insert(path.clone());
            tracing::info!(
                product = format_args!("{:04x}", attributes.ProductID),
                index,
                "DualShock 4 pad found; read as a console pad"
            );
            let handle = handle as usize;
            let spawned = std::thread::Builder::new()
                .name("victory-dualshock".into())
                .spawn(move || read(handle as HANDLE, index, path));
            if spawned.is_err() {
                // SAFETY: the thread that would have owned it did not start.
                unsafe { CloseHandle(handle as HANDLE) };
            }
        }
    }

    /// Blocking reads until the pad goes away, then forgets it so that a
    /// later scan opens it again.
    fn read(handle: HANDLE, index: usize, path: Vec<u16>) {
        // Larger than any DualShock 4 input report (USB 64, Bluetooth 547).
        let mut report = vec![0u8; 1024];
        loop {
            let mut got = 0u32;
            // SAFETY: a device opened for synchronous reads; the buffer is ours.
            let ok = unsafe {
                ReadFile(handle, report.as_mut_ptr(), report.len() as u32, &mut got, std::ptr::null_mut())
            } != 0;
            if !ok {
                break;
            }
            if let Some(state) = parse(&report[..got as usize]) {
                let mut pads = PADS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pads.states[index] = Some(state);
            }
        }
        tracing::info!(index, "DualShock 4 pad gone");
        let mut pads = PADS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        pads.states[index] = None;
        pads.open.remove(&path);
        drop(pads);
        // SAFETY: ours to close; no other thread uses it.
        unsafe { CloseHandle(handle) };
    }

    fn open(path: &[u16]) -> Option<HANDLE> {
        let try_open = |access: u32| {
            // SAFETY: a nul-terminated path; no security attributes or template.
            let handle = unsafe {
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
            (handle != INVALID_HANDLE_VALUE).then_some(handle)
        };
        // Some drivers refuse write access to a pad another program holds.
        try_open(GENERIC_READ | GENERIC_WRITE).or_else(|| try_open(GENERIC_READ))
    }

    /// Device paths (nul-terminated) of every present HID interface.
    fn hid_paths() -> Vec<Vec<u16>> {
        let mut paths = Vec::new();
        let mut guid = Default::default();
        // SAFETY: room for the GUID.
        unsafe { HidD_GetHidGuid(&mut guid) };
        // SAFETY: a valid class GUID, no enumerator or window.
        let set = unsafe {
            SetupDiGetClassDevsW(&guid, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
        };
        if set == INVALID_HANDLE_VALUE as isize {
            return paths;
        }
        for member in 0.. {
            let mut interface =
                SP_DEVICE_INTERFACE_DATA { cbSize: size_of::<SP_DEVICE_INTERFACE_DATA>() as u32, ..Default::default() };
            // SAFETY: a device set of ours and room for the interface.
            if unsafe { SetupDiEnumDeviceInterfaces(set, std::ptr::null(), &guid, member, &mut interface) } == 0 {
                break;
            }
            let mut needed = 0u32;
            // SAFETY: asks only for the size.
            unsafe {
                SetupDiGetDeviceInterfaceDetailW(set, &interface, std::ptr::null_mut(), 0, &mut needed, std::ptr::null_mut())
            };
            if needed < 8 {
                continue;
            }
            // u32 words keep the detail structure aligned.
            let mut buffer = vec![0u32; (needed as usize).div_ceil(4)];
            let detail = buffer.as_mut_ptr().cast::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>();
            // SAFETY: the buffer holds `needed` bytes; cbSize is the fixed
            // part's size (8 on x64), as the API wants.
            let ok = unsafe {
                (*detail).cbSize = size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() as u32;
                SetupDiGetDeviceInterfaceDetailW(set, &interface, detail, needed, std::ptr::null_mut(), std::ptr::null_mut())
            } != 0;
            if !ok {
                continue;
            }
            // The path follows the 4-byte size, nul-terminated UTF-16.
            // SAFETY: inside the buffer the API filled.
            let words = unsafe {
                std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>().add(4).cast::<u16>(), (needed as usize - 4) / 2)
            };
            let len = words.iter().position(|&w| w == 0).unwrap_or(words.len());
            let mut path = words[..len].to_vec();
            path.push(0);
            paths.push(path);
        }
        // SAFETY: ours to free.
        unsafe { SetupDiDestroyDeviceInfoList(set) };
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A USB report: id 0x01, then sticks, buttons, triggers.
    fn usb(lx: u8, ly: u8, rx: u8, ry: u8, b4: u8, b5: u8, b6: u8, l2: u8, r2: u8) -> Vec<u8> {
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..10].copy_from_slice(&[lx, ly, rx, ry, b4, b5, b6, l2, r2]);
        r
    }

    #[test]
    fn a_pad_at_rest_reads_as_nothing_pressed() {
        let pad = parse(&usb(128, 128, 128, 128, 0x08, 0, 0, 0, 0)).unwrap();
        assert_eq!(pad.buttons, 0);
        assert_eq!((pad.left_x, pad.left_y, pad.right_x, pad.right_y), (0.0, 0.0, 0.0, 0.0));
        assert_eq!((pad.left_trigger, pad.right_trigger), (0.0, 0.0));
    }

    #[test]
    fn face_buttons_land_where_the_console_pad_has_them() {
        let pad = parse(&usb(128, 128, 128, 128, 0x08 | 0x20 | 0x40, 0x20, 0, 0, 0)).unwrap();
        assert_eq!(pad.buttons, A | B | MENU);
        let pad = parse(&usb(128, 128, 128, 128, 0x08 | 0x10 | 0x80, 0x10 | 0x03, 0x02, 0, 0)).unwrap();
        assert_eq!(pad.buttons, X | Y | VIEW | LEFT_SHOULDER | RIGHT_SHOULDER);
    }

    #[test]
    fn the_hat_gives_diagonals_and_eight_is_released() {
        assert_eq!(parse(&usb(128, 128, 128, 128, 0, 0, 0, 0, 0)).unwrap().buttons, UP);
        assert_eq!(parse(&usb(128, 128, 128, 128, 3, 0, 0, 0, 0)).unwrap().buttons, DOWN | RIGHT);
        assert_eq!(parse(&usb(128, 128, 128, 128, 8, 0, 0, 0, 0)).unwrap().buttons, 0);
    }

    #[test]
    fn sticks_are_up_positive_and_triggers_run_to_one() {
        let pad = parse(&usb(255, 0, 0, 255, 8, 0, 0, 255, 0)).unwrap();
        assert_eq!((pad.left_x, pad.left_y), (1.0, 1.0));
        assert_eq!((pad.right_x, pad.right_y), (-1.0, -1.0));
        assert_eq!((pad.left_trigger, pad.right_trigger), (1.0, 0.0));
    }

    #[test]
    fn full_bluetooth_reports_carry_the_data_two_bytes_later() {
        let mut r = vec![0u8; 78];
        r[0] = 0x11;
        r[3..12].copy_from_slice(&[128, 128, 128, 128, 0x08 | 0x20, 0, 0, 0, 0]);
        assert_eq!(parse(&r).unwrap().buttons, A);
    }

    #[test]
    fn other_reports_and_short_ones_are_not_read() {
        assert_eq!(parse(&[]), None);
        assert_eq!(parse(&[0x05, 1, 2, 3]), None);
        assert_eq!(parse(&[0x01, 128, 128]), None);
    }

    #[test]
    fn only_sonys_dualshock_ids_are_taken() {
        assert!(is_dualshock(0x054C, 0x09CC));
        assert!(is_dualshock(0x054C, 0x05C4));
        assert!(!is_dualshock(0x054C, 0x0CE6)); // DualSense: another report layout
        assert!(!is_dualshock(0x045E, 0x09CC));
    }
}
