//! `ImageInfo` against a small PE32+ image built in the test, so these checks
//! run on every host. The unit tests that parse the test binary itself only
//! run where that binary is a PE image (Windows).

use victory_pe::layout::{self, dir};
use victory_pe::{ImageInfo, ImportSymbol, MACHINE_AMD64, PeError};

const IMAGE_BASE: u64 = 0x1_4000_0000;
/// The one section: RVA 0x1000, file offset 0x200, 0x400 bytes.
const SECTION_RVA: u32 = 0x1000;
const SECTION_FILE_OFFSET: u32 = 0x200;
const SECTION_SIZE: u32 = 0x400;
const SIZE_OF_IMAGE: u32 = 0x2000;
const OPT: usize = 0x58;
const DIR_TLS: usize = 9;

/// Section contents, by RVA.
const ENTRY: u32 = 0x1000;
const IMPORT_DESCRIPTORS: u32 = 0x1100;
const NAME_TABLE: u32 = 0x1180;
const IAT: u32 = 0x11A0;
const DLL_NAME: u32 = 0x11C0;
const HINT_NAME: u32 = 0x11D0;
const TLS_DIRECTORY: u32 = 0x1200;
const TLS_CALLBACKS: u32 = 0x1240;

fn put16(d: &mut [u8], at: usize, v: u16) {
    d[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(d: &mut [u8], at: usize, v: u32) {
    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(d: &mut [u8], at: usize, v: u64) {
    d[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// File offset of an RVA inside the section.
fn at(rva: u32) -> usize {
    (rva - SECTION_RVA + SECTION_FILE_OFFSET) as usize
}

/// An x86-64 executable importing `KERNEL32.dll!ExitProcess` (hint 0x123)
/// and ordinal 5 from it, with two TLS callbacks.
fn image() -> Vec<u8> {
    let mut d = vec![0u8; (SECTION_FILE_OFFSET + SECTION_SIZE) as usize];
    d[0..2].copy_from_slice(b"MZ");
    put32(&mut d, layout::DOS_E_LFANEW, 0x40);
    d[0x40..0x44].copy_from_slice(b"PE\0\0");
    put16(&mut d, 0x44, MACHINE_AMD64);
    put16(&mut d, 0x46, 1); // sections
    put32(&mut d, 0x48, 0x6000_0000); // timestamp
    put16(&mut d, 0x54, 240); // optional header size
    put16(&mut d, 0x56, 0x0022); // executable, large address aware

    put16(&mut d, OPT, 0x20B);
    put32(&mut d, OPT + 16, ENTRY);
    put64(&mut d, OPT + 24, IMAGE_BASE);
    put32(&mut d, OPT + 32, 0x1000); // section alignment
    put32(&mut d, OPT + 36, 0x200); // file alignment
    put32(&mut d, OPT + 56, SIZE_OF_IMAGE);
    put32(&mut d, OPT + 60, 0x200); // size of headers
    put16(&mut d, OPT + 68, layout::subsystem::WINDOWS_CUI);
    put64(&mut d, OPT + 72, 0x10_0000); // stack reserve
    put64(&mut d, OPT + 80, 0x1000); // stack commit
    put32(&mut d, OPT + 108, 16);
    let dd = |index: usize| OPT + layout::OPT_DATA_DIRECTORY + index * 8;
    put32(&mut d, dd(dir::IMPORT), IMPORT_DESCRIPTORS);
    put32(&mut d, dd(dir::IMPORT) + 4, 40);
    put32(&mut d, dd(DIR_TLS), TLS_DIRECTORY);
    put32(&mut d, dd(DIR_TLS) + 4, 40);

    let sec = OPT + 240;
    d[sec..sec + 5].copy_from_slice(b".text");
    put32(&mut d, sec + 8, SECTION_SIZE);
    put32(&mut d, sec + 12, SECTION_RVA);
    put32(&mut d, sec + 16, SECTION_SIZE);
    put32(&mut d, sec + 20, SECTION_FILE_OFFSET);
    put32(&mut d, sec + 36, 0xE000_0020); // code, read, write, execute

    d[at(ENTRY)] = 0xC3; // ret

    // One descriptor, then the all-zero terminator.
    let desc = at(IMPORT_DESCRIPTORS);
    put32(&mut d, desc, NAME_TABLE);
    put32(&mut d, desc + 12, DLL_NAME);
    put32(&mut d, desc + 16, IAT);
    for table in [NAME_TABLE, IAT] {
        put64(&mut d, at(table), u64::from(HINT_NAME));
        put64(&mut d, at(table) + 8, (1 << 63) | 5);
    }
    let name = b"KERNEL32.dll\0";
    d[at(DLL_NAME)..at(DLL_NAME) + name.len()].copy_from_slice(name);
    put16(&mut d, at(HINT_NAME), 0x123);
    let func = b"ExitProcess\0";
    d[at(HINT_NAME) + 2..at(HINT_NAME) + 2 + func.len()].copy_from_slice(func);

    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(TLS_CALLBACKS),
    );
    put64(&mut d, at(TLS_CALLBACKS), IMAGE_BASE + 0x1010);
    put64(&mut d, at(TLS_CALLBACKS) + 8, IMAGE_BASE + 0x1020);
    d
}

#[test]
fn reads_headers_and_sections() {
    let info = ImageInfo::parse(&image()).unwrap();
    assert_eq!(info.machine, MACHINE_AMD64);
    assert!(!info.is_dll());
    assert_eq!(info.timestamp, 0x6000_0000);
    assert_eq!(info.image_base, IMAGE_BASE);
    assert_eq!(info.size_of_image, SIZE_OF_IMAGE);
    assert_eq!(info.entry_rva, ENTRY);
    assert_eq!(info.subsystem_name(), "windows-cui");
    assert_eq!((info.stack_reserve, info.stack_commit), (0x10_0000, 0x1000));
    assert_eq!(info.sections.len(), 1);
    let s = &info.sections[0];
    assert!(s.name.starts_with(".text"));
    assert_eq!((s.rva, s.virtual_size), (SECTION_RVA, SECTION_SIZE));
    assert_eq!(
        (s.file_offset, s.file_size),
        (SECTION_FILE_OFFSET, SECTION_SIZE)
    );
    assert!(!info.has_relocations);
    assert!(!info.has_exception_table);
    assert!(info.exports.is_empty());
    assert!(info.delay_imports.is_empty());
    assert!(info.codeview.is_none());
}

#[test]
fn reads_imports_with_their_iat_slots() {
    let info = ImageInfo::parse(&image()).unwrap();
    assert_eq!(info.imports.len(), 1);
    let kernel32 = &info.imports[0];
    assert_eq!(kernel32.dll, "KERNEL32.dll");
    let entries: Vec<_> = kernel32
        .entries
        .iter()
        .map(|e| (e.symbol.clone(), e.iat_slot_rva))
        .collect();
    assert_eq!(
        entries,
        [
            (
                ImportSymbol::Name {
                    hint: 0x123,
                    name: "ExitProcess".into()
                },
                IAT
            ),
            (ImportSymbol::Ordinal(5), IAT + 8),
        ]
    );
    assert_eq!(info.import_count(), 2);
}

#[test]
fn reads_tls_callbacks_in_order() {
    let tls = ImageInfo::parse(&image()).unwrap().tls.unwrap();
    assert_eq!(tls.directory_rva, TLS_DIRECTORY);
    assert_eq!(tls.callbacks_va, IMAGE_BASE + u64::from(TLS_CALLBACKS));
    assert_eq!(tls.callback_rvas, [0x1010, 0x1020]);
}

#[test]
fn tls_callback_below_the_image_base_is_malformed() {
    let mut d = image();
    put64(&mut d, at(TLS_CALLBACKS) + 8, IMAGE_BASE - 0x1000);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn rva_to_file_offset_maps_section_data() {
    let data = image();
    let mut info = ImageInfo::parse(&data).unwrap();
    let entry = info.rva_to_file_offset(info.entry_rva).unwrap() as usize;
    assert_eq!(data[entry], 0xC3);
    assert_eq!(
        info.rva_to_file_offset(HINT_NAME),
        Some(at(HINT_NAME) as u32)
    );
    // Headers and the end of the section are not section data.
    assert_eq!(info.rva_to_file_offset(0x40), None);
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + SECTION_SIZE), None);
    // Raw data past the virtual size is padding, not part of the section.
    info.sections[0].virtual_size = 0x100;
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0xFF), Some(0x2FF));
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0x100), None);
    // An offset near 4 GiB does not wrap.
    info.sections[0].file_offset = u32::MAX - 0x10;
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0x20), None);
}

#[test]
fn rejects_iat_outside_image() {
    for iat in [SIZE_OF_IMAGE - 8, 0xFFFF_FFF8] {
        let mut d = image();
        // The second slot (or the first, at the 4 GiB limit) runs past the end.
        put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, iat);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "IAT at {iat:#x}"
        );
    }
}

#[test]
fn rejects_other_machines() {
    let mut d = image();
    put16(&mut d, 0x44, 0xAA64); // ARM64
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Unsupported(_))));
}

#[test]
fn rejects_truncated_images_without_panicking() {
    let d = image();
    for len in [0, 2, 0x40, 0x58, 0x150, 0x200] {
        assert!(ImageInfo::parse(&d[..len]).is_err(), "{len} bytes");
    }
}
