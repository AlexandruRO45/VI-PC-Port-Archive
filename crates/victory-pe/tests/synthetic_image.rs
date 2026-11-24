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

// Optional directories, each added to the image by the test that needs it.
// They live in the free tail of the section (0x1280..0x1400).
const DIR_EXPORT: usize = 0;
const DIR_DEBUG: usize = 6;
const DIR_DELAY_IMPORT: usize = 13;

const DELAY_DESCRIPTORS: u32 = 0x1280;
const DELAY_NAME_TABLE: u32 = 0x12C0;
const DELAY_IAT: u32 = 0x12D0;
const DELAY_DLL_NAME: u32 = 0x12E0;
const DELAY_HINT_NAME: u32 = 0x12F0;
const DEBUG_DIRECTORY: u32 = 0x1300;
const CODEVIEW: u32 = 0x1320;
const EXPORT_DIRECTORY: u32 = 0x1350;
const EXPORT_DIRECTORY_END: u32 = 0x1400;

fn put_bytes(d: &mut [u8], rva: u32, bytes: &[u8]) {
    d[at(rva)..at(rva) + bytes.len()].copy_from_slice(bytes);
}

fn set_directory(d: &mut [u8], index: usize, rva: u32, size: u32) {
    let entry = OPT + layout::OPT_DATA_DIRECTORY + index * layout::DATA_DIRECTORY_SIZE;
    put32(d, entry, rva);
    put32(d, entry + 4, size);
}

/// Delay-loads `USER32.dll!MessageBoxW` (hint 7) with an RVA-based descriptor.
fn add_delay_import(d: &mut [u8]) {
    set_directory(d, DIR_DELAY_IMPORT, DELAY_DESCRIPTORS, 64);
    let desc = at(DELAY_DESCRIPTORS);
    put32(d, desc, 1); // attributes: RVA based
    put32(d, desc + 4, DELAY_DLL_NAME);
    put32(d, desc + 12, DELAY_IAT);
    put32(d, desc + 16, DELAY_NAME_TABLE);
    put64(d, at(DELAY_NAME_TABLE), u64::from(DELAY_HINT_NAME));
    put_bytes(d, DELAY_DLL_NAME, b"USER32.dll\0");
    put16(d, at(DELAY_HINT_NAME), 7);
    put_bytes(d, DELAY_HINT_NAME + 2, b"MessageBoxW\0");
}

/// A CodeView debug directory entry whose record has signature `sig`.
fn add_codeview(d: &mut [u8], sig: &[u8; 4]) {
    set_directory(d, DIR_DEBUG, DEBUG_DIRECTORY, 28);
    let dbg = at(DEBUG_DIRECTORY);
    put32(d, dbg + 12, 2); // IMAGE_DEBUG_TYPE_CODEVIEW
    put32(d, dbg + 16, 4 + 16 + 4 + 6); // size of data
    put32(d, dbg + 20, CODEVIEW);
    put32(d, dbg + 24, at(CODEVIEW) as u32); // pointer to raw data
    put_bytes(d, CODEVIEW, sig);
    let guid: Vec<u8> = (0..16).collect();
    put_bytes(d, CODEVIEW + 4, &guid);
    put32(d, at(CODEVIEW) + 20, 0x2A); // age
    put_bytes(d, CODEVIEW + 24, b"t.pdb\0");
}

/// Exports ordinal 1 `local` at the entry point, ordinal 2 `fwdname` forwarded
/// to `NTDLL.RtlFoo` and ordinal 3 `fwdord` forwarded to `NTDLL.#7`.
fn add_exports(d: &mut [u8]) {
    set_directory(
        d,
        DIR_EXPORT,
        EXPORT_DIRECTORY,
        EXPORT_DIRECTORY_END - EXPORT_DIRECTORY,
    );
    let (addresses, names, ordinals) = (0x1378, 0x1384, 0x1390);
    let (dll, local, fwdname, fwdord) = (0x1398, 0x13A0, 0x13A8, 0x13B0);
    let (to_name, to_ordinal) = (0x13B8, 0x13C8);
    let ed = at(EXPORT_DIRECTORY);
    put32(d, ed + 12, dll);
    put32(d, ed + 16, 1); // ordinal base
    put32(d, ed + 20, 3); // functions
    put32(d, ed + 24, 3); // names
    put32(d, ed + 28, addresses);
    put32(d, ed + 32, names);
    put32(d, ed + 36, ordinals);
    // A forwarder is an address inside the export directory.
    for (i, rva) in [ENTRY, to_name, to_ordinal].into_iter().enumerate() {
        put32(d, at(addresses) + i * 4, rva);
    }
    // The name pointer table is sorted; each name maps to its ordinal index.
    for (i, (name, index)) in [(fwdname, 1u16), (fwdord, 2), (local, 0)]
        .into_iter()
        .enumerate()
    {
        put32(d, at(names) + i * 4, name);
        put16(d, at(ordinals) + i * 2, index);
    }
    put_bytes(d, dll, b"t.dll\0");
    put_bytes(d, local, b"local\0");
    put_bytes(d, fwdname, b"fwdname\0");
    put_bytes(d, fwdord, b"fwdord\0");
    put_bytes(d, to_name, b"NTDLL.RtlFoo\0");
    put_bytes(d, to_ordinal, b"NTDLL.#7\0");
}

#[test]
fn reads_delay_imports_with_their_iat_slots() {
    let mut d = image();
    add_delay_import(&mut d);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.delay_imports.len(), 1);
    let user32 = &info.delay_imports[0];
    assert_eq!(user32.dll, "USER32.dll");
    assert_eq!(user32.entries.len(), 1);
    assert_eq!(
        user32.entries[0].symbol,
        ImportSymbol::Name {
            hint: 7,
            name: "MessageBoxW".into()
        }
    );
    assert_eq!(user32.entries[0].iat_slot_rva, DELAY_IAT);
    // Delay imports are kept apart from the ordinary imports.
    assert_eq!(info.import_count(), 2);
}

#[test]
fn rejects_delay_iat_outside_image() {
    for iat in [SIZE_OF_IMAGE - 4, 0xFFFF_FFFC] {
        let mut d = image();
        add_delay_import(&mut d);
        put32(&mut d, at(DELAY_DESCRIPTORS) + 12, iat);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "delay IAT at {iat:#x}"
        );
    }
}

#[test]
fn reads_exports_and_forwarders() {
    let mut d = image();
    add_exports(&mut d);
    let info = ImageInfo::parse(&d).unwrap();
    let mut exports: Vec<_> = info
        .exports
        .iter()
        .map(|e| (e.ordinal, e.name.as_deref(), e.rva, e.forward.as_deref()))
        .collect();
    exports.sort();
    assert_eq!(
        exports,
        [
            (1, Some("local"), Some(ENTRY), None),
            (2, Some("fwdname"), None, Some("NTDLL.RtlFoo")),
            (3, Some("fwdord"), None, Some("NTDLL.#7")),
        ]
    );
}

#[test]
fn reads_codeview_record_and_symbol_key() {
    let mut d = image();
    add_codeview(&mut d, b"RSDS");
    let cv = ImageInfo::parse(&d).unwrap().codeview.unwrap();
    assert_eq!(cv.pdb_path, "t.pdb");
    assert_eq!(cv.age, 0x2A);
    let guid: Vec<u8> = (0..16).collect();
    assert_eq!(cv.guid[..], guid[..]);
    // The first three GUID fields are little endian; the age is hex, unpadded.
    assert_eq!(cv.symbol_key(), "030201000504070608090A0B0C0D0E0F2A");
}

#[test]
fn ignores_codeview_records_that_are_not_rsds() {
    let mut d = image();
    add_codeview(&mut d, b"NB10");
    assert!(ImageInfo::parse(&d).unwrap().codeview.is_none());
}

#[test]
fn reports_relocation_exception_and_load_config_directories() {
    let mut d = image();
    // Only the directory entries matter; their contents are not parsed.
    set_directory(&mut d, dir::BASERELOC, 0x1380, 12);
    set_directory(&mut d, dir::EXCEPTION, 0x1390, 12);
    set_directory(&mut d, dir::LOAD_CONFIG, 0x13A0, 0x40);
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.has_relocations);
    assert!(info.has_exception_table);
    assert!(info.has_load_config);

    // Stripped relocations win over a leftover directory entry.
    let fh = layout::file_header_offset(&d).unwrap();
    let at_flags = fh + layout::FILE_HEADER_CHARACTERISTICS;
    let flags = u16::from_le_bytes([d[at_flags], d[at_flags + 1]]);
    put16(
        &mut d,
        at_flags,
        flags | layout::file_flags::RELOCS_STRIPPED,
    );
    assert!(!ImageInfo::parse(&d).unwrap().has_relocations);

    // An entry with an RVA but no size is absent.
    set_directory(&mut d, dir::EXCEPTION, 0x1390, 0);
    assert!(!ImageInfo::parse(&d).unwrap().has_exception_table);
}

#[test]
fn names_subsystems_and_dll_flag() {
    use layout::subsystem::*;
    for (value, name) in [
        (WINDOWS_GUI, "windows-gui"),
        (WINDOWS_CUI, "windows-cui"),
        (XBOX, "xbox"),
        (XBOX_CODE_CATALOG, "xbox-code-catalog"),
        (1, "other"),
    ] {
        let mut d = image();
        put16(&mut d, OPT + 68, value);
        let info = ImageInfo::parse(&d).unwrap();
        assert_eq!((info.subsystem, info.subsystem_name()), (value, name));
    }
    let mut d = image();
    put16(&mut d, 0x56, 0x0022 | layout::file_flags::DLL);
    assert!(ImageInfo::parse(&d).unwrap().is_dll());
}

#[test]
fn layout_offsets_match_the_image() {
    let mut d = image();
    assert_eq!(layout::file_header_offset(&d), Some(0x44));
    assert_eq!(layout::optional_header_offset(&d), Some(OPT));
    assert_eq!(
        layout::data_directory_offset(&d, DIR_TLS),
        Some(OPT + layout::OPT_DATA_DIRECTORY + DIR_TLS * layout::DATA_DIRECTORY_SIZE)
    );
    assert_eq!(layout::data_directory_offset(&d, 16), None);
    let tls = layout::data_directory_offset(&d, DIR_TLS).unwrap();
    assert_eq!(layout::read_u32(&d, tls), Some(TLS_DIRECTORY));

    // Fewer directories than the index: none.
    put32(&mut d, OPT + layout::OPT_NUMBER_OF_RVA_AND_SIZES, 2);
    assert!(layout::data_directory_offset(&d, dir::IMPORT).is_some());
    assert_eq!(layout::data_directory_offset(&d, DIR_TLS), None);

    // Headers that point past the data, or nowhere at all, give None.
    assert_eq!(layout::file_header_offset(&d[..0x3C]), None);
    put32(&mut d, layout::DOS_E_LFANEW, u32::MAX);
    assert_eq!(layout::data_directory_offset(&d, dir::IMPORT), None);
}

#[test]
fn reads_names_from_the_iat_when_the_name_table_is_absent() {
    // Old linkers leave OriginalFirstThunk at 0; the IAT holds the names.
    let mut d = image();
    put32(&mut d, at(IMPORT_DESCRIPTORS), 0);
    d[at(NAME_TABLE)..at(NAME_TABLE) + 16].fill(0);
    let info = ImageInfo::parse(&d).unwrap();
    let entries: Vec<_> = info.imports[0]
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
}

#[test]
fn tls_directory_without_callbacks_or_without_an_rva() {
    // A TLS directory with no callback array is still TLS (static data).
    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        0,
    );
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!((tls.callbacks_va, tls.callback_rvas.len()), (0, 0));

    // A directory entry with RVA 0 means no TLS at all.
    let mut d = image();
    set_directory(&mut d, DIR_TLS, 0, 40);
    assert!(ImageInfo::parse(&d).unwrap().tls.is_none());

    // So does an image whose header lists no TLS directory slot.
    let mut d = image();
    put32(
        &mut d,
        OPT + layout::OPT_NUMBER_OF_RVA_AND_SIZES,
        DIR_TLS as u32,
    );
    assert!(ImageInfo::parse(&d).unwrap().tls.is_none());
}

#[test]
fn tls_callback_array_stops_at_the_end_of_the_section() {
    // The last slot of the section, with no null terminator after it.
    let last = SECTION_RVA + SECTION_SIZE - 8;
    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(last),
    );
    put64(&mut d, at(last), IMAGE_BASE + 0x1030);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, [0x1030]);
}

#[test]
fn rejects_tls_outside_the_sections() {
    let malformed = |d: &[u8], what: &str| {
        assert!(
            matches!(ImageInfo::parse(d), Err(PeError::Malformed(_))),
            "{what}"
        );
    };
    let callbacks = at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS;

    // Inside the image but past the only section.
    let mut d = image();
    set_directory(&mut d, DIR_TLS, SECTION_RVA + SECTION_SIZE + 0x100, 40);
    malformed(&d, "directory past the section");

    // Too close to the end of the section to hold AddressOfCallBacks.
    let mut d = image();
    set_directory(&mut d, DIR_TLS, SECTION_RVA + SECTION_SIZE - 0x10, 40);
    malformed(&d, "truncated directory");

    let mut d = image();
    put64(
        &mut d,
        callbacks,
        IMAGE_BASE + u64::from(SECTION_RVA + SECTION_SIZE + 0x100),
    );
    malformed(&d, "callback array past the section");

    let mut d = image();
    put64(&mut d, callbacks, IMAGE_BASE - 8);
    malformed(&d, "callback array below the image base");

    // A callback 4 GiB or more above the base has no RVA.
    let mut d = image();
    put64(&mut d, at(TLS_CALLBACKS) + 8, IMAGE_BASE + (1 << 32));
    malformed(&d, "callback past 4 GiB");
}

#[test]
fn rejects_pe32_images() {
    // Same image, 32-bit optional header magic: PE32 is not supported.
    let mut d = image();
    put16(&mut d, OPT, 0x10B);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Unsupported(_))));
}

#[test]
fn import_symbols_display_as_their_name_or_ordinal() {
    let info = ImageInfo::parse(&image()).unwrap();
    let shown: Vec<_> = info.imports[0]
        .entries
        .iter()
        .map(|e| e.symbol.to_string())
        .collect();
    assert_eq!(shown, ["ExitProcess", "#5"]);
}

#[test]
fn tls_callbacks_are_capped() {
    // Grow the section so a 1100-entry callback array, with no terminator,
    // fits after the existing contents.
    const SIZE: u32 = 0x3000;
    const ARRAY: u32 = 0x1400;
    const COUNT: usize = 1100;
    let mut d = image();
    d.resize((SECTION_FILE_OFFSET + SIZE) as usize, 0);
    let sec = OPT + 240;
    put32(&mut d, sec + 8, SIZE);
    put32(&mut d, sec + 16, SIZE);
    put32(&mut d, OPT + 56, SECTION_RVA + SIZE);
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(ARRAY),
    );
    for i in 0..COUNT {
        put64(&mut d, at(ARRAY) + i * 8, IMAGE_BASE + 0x1000 + i as u64);
    }
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas.len(), 1024);
    assert_eq!(tls.callback_rvas[0], 0x1000);
    assert_eq!(tls.callback_rvas[1023], 0x1000 + 1023);
}

#[test]
fn reads_exports_without_a_name() {
    // Only two names: the export at ordinal 1 (`local`) is by ordinal only.
    let mut d = image();
    add_exports(&mut d);
    put32(&mut d, at(EXPORT_DIRECTORY) + 24, 2);
    let info = ImageInfo::parse(&d).unwrap();
    let mut exports: Vec<_> = info
        .exports
        .iter()
        .map(|e| (e.ordinal, e.name.as_deref(), e.rva, e.forward.as_deref()))
        .collect();
    exports.sort();
    assert_eq!(
        exports,
        [
            (1, None, Some(ENTRY), None),
            (2, Some("fwdname"), None, Some("NTDLL.RtlFoo")),
            (3, Some("fwdord"), None, Some("NTDLL.#7")),
        ]
    );
}

#[test]
fn reads_delay_imports_by_ordinal() {
    let mut d = image();
    add_delay_import(&mut d);
    put64(&mut d, at(DELAY_NAME_TABLE), (1 << 63) | 9);
    let info = ImageInfo::parse(&d).unwrap();
    let user32 = &info.delay_imports[0];
    assert_eq!(user32.entries.len(), 1);
    assert_eq!(user32.entries[0].symbol, ImportSymbol::Ordinal(9));
    assert_eq!(user32.entries[0].symbol.to_string(), "#9");
    assert_eq!(user32.entries[0].iat_slot_rva, DELAY_IAT);
}

#[test]
fn rva_to_file_offset_picks_the_section_that_holds_the_rva() {
    // A second section, `.data` at RVA 0x2000 and file offset 0x600, with a
    // virtual size of 0: its raw size is then its size in memory.
    const DATA_RVA: u32 = 0x2000;
    const DATA_FILE_OFFSET: u32 = SECTION_FILE_OFFSET + SECTION_SIZE;
    const DATA_SIZE: u32 = 0x200;
    let mut d = image();
    d.resize((DATA_FILE_OFFSET + DATA_SIZE) as usize, 0);
    put16(&mut d, 0x46, 2);
    put32(&mut d, OPT + 56, DATA_RVA + 0x1000);
    let sec = OPT + 240 + 40;
    d[sec..sec + 5].copy_from_slice(b".data");
    put32(&mut d, sec + 12, DATA_RVA);
    put32(&mut d, sec + 16, DATA_SIZE);
    put32(&mut d, sec + 20, DATA_FILE_OFFSET);
    put32(&mut d, sec + 36, 0xC000_0040); // initialised data, read, write

    let info = ImageInfo::parse(&d).unwrap();
    let sections: Vec<_> = info
        .sections
        .iter()
        .map(|s| (s.name.trim_end_matches('\0'), s.rva, s.virtual_size))
        .collect();
    assert_eq!(
        sections,
        [(".text", SECTION_RVA, SECTION_SIZE), (".data", DATA_RVA, 0)]
    );
    assert_eq!(
        info.rva_to_file_offset(HINT_NAME),
        Some(at(HINT_NAME) as u32)
    );
    assert_eq!(info.rva_to_file_offset(DATA_RVA), Some(DATA_FILE_OFFSET));
    assert_eq!(
        info.rva_to_file_offset(DATA_RVA + DATA_SIZE - 1),
        Some(DATA_FILE_OFFSET + DATA_SIZE - 1)
    );
    // The gap between the sections and the end of `.data` map nowhere.
    assert_eq!(
        info.rva_to_file_offset(SECTION_RVA + SECTION_SIZE + 0x10),
        None
    );
    assert_eq!(info.rva_to_file_offset(DATA_RVA + DATA_SIZE), None);
}

#[test]
fn rejects_damaged_dos_and_nt_signatures() {
    // No `MZ`: not a PE file at all.
    let mut d = image();
    d[0..2].copy_from_slice(b"ZM");
    assert!(ImageInfo::parse(&d).is_err());

    // `MZ`, but the NT signature is wrong.
    let mut d = image();
    d[0x40..0x44].copy_from_slice(b"NE\0\0");
    assert!(ImageInfo::parse(&d).is_err());

    // `e_lfanew` points past the end of the file, or right at its last bytes.
    for lfanew in [u32::MAX, d.len() as u32, d.len() as u32 - 2] {
        let mut d = image();
        put32(&mut d, layout::DOS_E_LFANEW, lfanew);
        assert!(ImageInfo::parse(&d).is_err(), "e_lfanew {lfanew:#x}");
    }
}

#[test]
fn rejects_an_import_name_outside_the_sections() {
    for name in [SIZE_OF_IMAGE + 0x100, 0xFFFF_FFF0] {
        let mut d = image();
        put32(&mut d, at(IMPORT_DESCRIPTORS) + 12, name);
        assert!(ImageInfo::parse(&d).is_err(), "DLL name at {name:#x}");
    }
    for hint_name in [SIZE_OF_IMAGE + 0x100, 0x7FFF_FFF0] {
        let mut d = image();
        put64(&mut d, at(NAME_TABLE), u64::from(hint_name));
        assert!(ImageInfo::parse(&d).is_err(), "hint/name at {hint_name:#x}");
    }
}

#[test]
fn import_and_tls_directories_pointing_outside_the_image_do_not_panic() {
    for rva in [SIZE_OF_IMAGE, SIZE_OF_IMAGE + 0x1000, 0xFFFF_FFF0] {
        for index in [
            dir::IMPORT,
            DIR_TLS,
            DIR_EXPORT,
            DIR_DEBUG,
            DIR_DELAY_IMPORT,
        ] {
            let mut d = image();
            set_directory(&mut d, index, rva, 0x40);
            // Either outcome is acceptable; the parse must simply return.
            let _ = ImageInfo::parse(&d);
        }
    }
}

#[test]
fn corrupting_any_header_byte_does_not_panic() {
    // Every byte of the headers, the section table and the directories in
    // the section, set to each of a few hostile values.
    let clean = image();
    let ranges = [
        0..OPT + 240 + 40,
        at(IMPORT_DESCRIPTORS)..at(HINT_NAME) + 16,
        at(TLS_DIRECTORY)..at(TLS_CALLBACKS) + 16,
    ];
    for range in ranges {
        for i in range {
            for v in [0x00, 0x7F, 0x80, 0xFF] {
                if clean[i] == v {
                    continue;
                }
                let mut d = clean.clone();
                d[i] = v;
                let _ = ImageInfo::parse(&d);
            }
        }
    }
}

#[test]
fn every_truncation_of_the_image_returns() {
    let d = image();
    for len in 0..d.len() {
        let _ = ImageInfo::parse(&d[..len]);
    }
}

/// A second import descriptor, `XAPI.dll!Sleep` (hint 9), in the free gap
/// between the first descriptor's terminator and the first name table.
#[test]
fn reads_every_import_module_in_descriptor_order() {
    const SECOND_DESCRIPTOR: u32 = IMPORT_DESCRIPTORS + 20;
    // Past the new terminator, which ends at 0x113C.
    const NAMES: u32 = 0x1140;
    const SLOTS: u32 = 0x1150;
    const DLL: u32 = 0x1160;
    const HINT: u32 = 0x1170;
    let mut d = image();
    set_directory(&mut d, dir::IMPORT, IMPORT_DESCRIPTORS, 60);
    let desc = at(SECOND_DESCRIPTOR);
    put32(&mut d, desc, NAMES);
    put32(&mut d, desc + 12, DLL);
    put32(&mut d, desc + 16, SLOTS);
    for table in [NAMES, SLOTS] {
        put64(&mut d, at(table), u64::from(HINT));
    }
    put_bytes(&mut d, DLL, b"XAPI.dll\0");
    put16(&mut d, at(HINT), 9);
    put_bytes(&mut d, HINT + 2, b"Sleep\0");

    let info = ImageInfo::parse(&d).unwrap();
    let modules: Vec<_> = info
        .imports
        .iter()
        .map(|m| (m.dll.as_str(), m.entries.len()))
        .collect();
    assert_eq!(modules, [("KERNEL32.dll", 2), ("XAPI.dll", 1)]);
    let sleep = &info.imports[1].entries[0];
    assert_eq!(
        sleep.symbol,
        ImportSymbol::Name {
            hint: 9,
            name: "Sleep".into()
        }
    );
    assert_eq!(sleep.iat_slot_rva, SLOTS);
    assert_eq!(info.import_count(), 3);
}

#[test]
fn section_names_fill_all_eight_bytes_and_keep_their_flags() {
    let mut d = image();
    let sec = OPT + 240;
    d[sec..sec + 8].copy_from_slice(b".textbss");
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.sections[0].name, ".textbss");
    assert_eq!(info.sections[0].characteristics, 0xE000_0020);
    // A shorter name loses its NUL padding.
    let info = ImageInfo::parse(&image()).unwrap();
    assert_eq!(info.sections[0].name, ".text");
}

#[test]
fn rva_to_file_offset_stops_at_the_virtual_size_and_never_overflows() {
    // Raw data padded past a smaller virtual size: the padding is not mapped.
    let mut d = image();
    put32(&mut d, OPT + 240 + 8, 0x300);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(
        info.rva_to_file_offset(SECTION_RVA + 0x2FF),
        Some(at(SECTION_RVA + 0x2FF) as u32)
    );
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0x300), None);
    // Below every section, and far past it.
    assert_eq!(info.rva_to_file_offset(0), None);
    assert_eq!(info.rva_to_file_offset(u32::MAX), None);

    // A section claiming a file offset near 4 GiB maps nothing instead of
    // wrapping around. Built by hand: such an image does not parse.
    let mut info = ImageInfo::parse(&image()).unwrap();
    info.sections[0].file_offset = u32::MAX - 0x10;
    assert_eq!(info.rva_to_file_offset(SECTION_RVA), Some(u32::MAX - 0x10));
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0x10), Some(u32::MAX));
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0x11), None);
}

#[test]
fn delay_import_slots_follow_the_name_table_eight_bytes_apart() {
    // Move the delay name table and IAT to the free tail so each holds two
    // entries: a name, then an ordinal, then the terminator.
    const NAMES: u32 = 0x1350;
    const SLOTS: u32 = 0x1370;
    let mut d = image();
    add_delay_import(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 12, SLOTS);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 16, NAMES);
    put64(&mut d, at(NAMES), u64::from(DELAY_HINT_NAME));
    put64(&mut d, at(NAMES) + 8, (1 << 63) | 4);
    let info = ImageInfo::parse(&d).unwrap();
    let entries: Vec<_> = info.delay_imports[0]
        .entries
        .iter()
        .map(|e| (e.symbol.clone(), e.iat_slot_rva))
        .collect();
    assert_eq!(
        entries,
        [
            (
                ImportSymbol::Name {
                    hint: 7,
                    name: "MessageBoxW".into()
                },
                SLOTS
            ),
            (ImportSymbol::Ordinal(4), SLOTS + 8),
        ]
    );
    // Ordinary imports are reported apart from delay imports.
    assert_eq!(info.import_count(), 2);
}

#[test]
fn an_iat_that_ends_exactly_at_the_size_of_image_is_inside_it() {
    // Two slots, the second's last byte the image's last byte: accepted.
    // One slot further is the case `rejects_iat_outside_image` covers.
    let mut d = image();
    let iat = SIZE_OF_IMAGE - 16;
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, iat);
    let info = ImageInfo::parse(&d).unwrap();
    let slots: Vec<_> = info.imports[0]
        .entries
        .iter()
        .map(|e| e.iat_slot_rva)
        .collect();
    assert_eq!(slots, [iat, iat + 8]);
}

#[test]
fn section_names_that_are_not_utf8_are_read_lossily() {
    let mut d = image();
    let name = OPT + 240;
    d[name..name + 8].copy_from_slice(b".t\xFFxt\0\0\0");
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.sections[0].name, ".t\u{FFFD}xt");
    // The rest of the image is unaffected by the name.
    assert_eq!(info.rva_to_file_offset(ENTRY), Some(SECTION_FILE_OFFSET));
}

#[test]
fn tls_vas_more_than_4_gib_above_the_image_base_are_malformed() {
    // An RVA is 32 bits: a VA that far above the base names nothing in the
    // image and must not be truncated to some low RVA inside it.
    let far = IMAGE_BASE + (1 << 32);
    for callback in [far + 0x1010, u64::MAX] {
        let mut d = image();
        put64(&mut d, at(TLS_CALLBACKS), callback);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "callback at {callback:#x}"
        );
    }
    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        far + u64::from(TLS_CALLBACKS),
    );
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
    // The last RVA that fits is still read as an RVA (and then found to be
    // outside the sections), not rejected as a VA.
    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(u32::MAX),
    );
    match ImageInfo::parse(&d) {
        Err(PeError::Malformed(message)) => assert!(
            message.contains("outside of sections"),
            "unexpected message: {message}"
        ),
        other => panic!("expected a malformed image, got {other:?}"),
    }
}

#[test]
fn export_ordinals_start_at_the_ordinal_base() {
    let mut d = image();
    add_exports(&mut d);
    put32(&mut d, at(EXPORT_DIRECTORY) + 16, 10);
    let info = ImageInfo::parse(&d).unwrap();
    let mut exports: Vec<_> = info
        .exports
        .iter()
        .map(|e| (e.ordinal, e.name.as_deref()))
        .collect();
    exports.sort();
    assert_eq!(
        exports,
        [
            (10, Some("local")),
            (11, Some("fwdname")),
            (12, Some("fwdord"))
        ]
    );
}

#[test]
fn unused_export_slots_are_left_out() {
    // A zero address marks an ordinal the DLL does not export; it is gone
    // even though the name table still points at it.
    let mut d = image();
    add_exports(&mut d);
    put32(&mut d, at(0x1378), 0);
    let info = ImageInfo::parse(&d).unwrap();
    let mut exports: Vec<_> = info
        .exports
        .iter()
        .map(|e| (e.ordinal, e.name.as_deref(), e.rva))
        .collect();
    exports.sort();
    assert_eq!(
        exports,
        [(2, Some("fwdname"), None), (3, Some("fwdord"), None)]
    );
    assert!(info.exports.iter().all(|e| e.forward.is_some()));
}

#[test]
fn a_directory_with_a_size_but_no_rva_is_absent() {
    let mut d = image();
    set_directory(&mut d, dir::BASERELOC, 0, 12);
    set_directory(&mut d, dir::EXCEPTION, 0, 12);
    set_directory(&mut d, dir::LOAD_CONFIG, 0, 0x40);
    let info = ImageInfo::parse(&d).unwrap();
    assert!(!info.has_relocations);
    assert!(!info.has_exception_table);
    assert!(!info.has_load_config);
}

#[test]
fn header_flag_words_are_passed_through_unchanged() {
    let mut d = image();
    // High entropy VA, dynamic base, NX compatible, terminal server aware.
    put16(&mut d, OPT + 70, 0x8160);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.characteristics, 0x0022);
    assert_eq!(info.dll_characteristics, 0x8160);
    assert!(!info.is_dll());
}

#[test]
fn names_come_from_the_name_table_when_the_iat_is_bound() {
    // A bound image has absolute addresses already written into its IAT;
    // the unbound name table still says what each slot imports.
    let mut d = image();
    put64(&mut d, at(IAT), 0x7FF8_0000_1000);
    put64(&mut d, at(IAT) + 8, 0x7FF8_0000_2000);
    let info = ImageInfo::parse(&d).unwrap();
    let entries: Vec<_> = info.imports[0]
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
}

#[test]
fn an_import_module_with_an_empty_name_table_has_no_entries() {
    let mut d = image();
    d[at(NAME_TABLE)..at(NAME_TABLE) + 16].fill(0);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.imports.len(), 1);
    assert_eq!(info.imports[0].dll, "KERNEL32.dll");
    assert!(info.imports[0].entries.is_empty());
    assert_eq!(info.import_count(), 0);
}

#[test]
fn an_import_directory_holding_only_the_terminator_has_no_modules() {
    let mut d = image();
    d[at(IMPORT_DESCRIPTORS)..at(IMPORT_DESCRIPTORS) + 20].fill(0);
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.imports.is_empty());
    assert_eq!(info.import_count(), 0);
}

#[test]
fn an_address_at_the_end_of_the_export_directory_is_not_a_forwarder() {
    // Only addresses inside [directory, directory + size) are forwarder
    // strings; the first byte past it is ordinary code or data.
    let mut d = image();
    add_exports(&mut d);
    put32(&mut d, at(0x1378), EXPORT_DIRECTORY_END);
    let info = ImageInfo::parse(&d).unwrap();
    let local = info
        .exports
        .iter()
        .find(|e| e.name.as_deref() == Some("local"))
        .unwrap();
    assert_eq!(local.rva, Some(EXPORT_DIRECTORY_END));
    assert_eq!(local.forward, None);
}

#[test]
fn an_export_directory_without_functions_has_no_exports() {
    let mut d = image();
    add_exports(&mut d);
    let ed = at(EXPORT_DIRECTORY);
    put32(&mut d, ed + 20, 0); // functions
    put32(&mut d, ed + 24, 0); // names
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.exports.is_empty());
}

#[test]
fn rejects_a_delay_import_name_outside_the_sections() {
    for name in [SIZE_OF_IMAGE + 0x100, 0xFFFF_FFF0] {
        let mut d = image();
        add_delay_import(&mut d);
        put32(&mut d, at(DELAY_DESCRIPTORS) + 4, name);
        assert!(ImageInfo::parse(&d).is_err(), "DLL name at {name:#x}");
    }
    for hint_name in [SIZE_OF_IMAGE + 0x100, 0x7FFF_FFF0] {
        let mut d = image();
        add_delay_import(&mut d);
        put64(&mut d, at(DELAY_NAME_TABLE), u64::from(hint_name));
        assert!(ImageInfo::parse(&d).is_err(), "hint/name at {hint_name:#x}");
    }
}

#[test]
fn a_delay_import_directory_holding_only_the_terminator_has_no_modules() {
    let mut d = image();
    add_delay_import(&mut d);
    d[at(DELAY_DESCRIPTORS)..at(DELAY_DESCRIPTORS) + 32].fill(0);
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.delay_imports.is_empty());
}

#[test]
fn ordinal_imports_keep_only_the_low_16_bits() {
    // IMAGE_ORDINAL64 is the low word; bits 16..62 are not part of it.
    let mut d = image();
    for table in [NAME_TABLE, IAT] {
        put64(&mut d, at(table) + 8, (1 << 63) | 0x7FFF_1234_0005);
    }
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.imports[0].entries[1].symbol, ImportSymbol::Ordinal(5));
}

#[test]
fn name_imports_read_the_hint_name_rva_from_the_low_32_bits() {
    // Without the ordinal flag the thunk holds an RVA; the loader reads it as
    // a 32-bit value, so the bits above it do not move the name.
    let mut d = image();
    put64(
        &mut d,
        at(NAME_TABLE),
        0x1234_5678_0000_0000 | u64::from(HINT_NAME),
    );
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(
        info.imports[0].entries[0].symbol,
        ImportSymbol::Name {
            hint: 0x123,
            name: "ExitProcess".into()
        }
    );
}

#[test]
fn tls_callbacks_after_a_null_slot_are_not_read() {
    // The loader stops at the first null pointer; what follows is not a
    // callback even when it looks like one.
    let mut d = image();
    put64(&mut d, at(TLS_CALLBACKS) + 16, 0);
    put64(&mut d, at(TLS_CALLBACKS) + 24, IMAGE_BASE + 0x1030);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, vec![0x1010, 0x1020]);

    put64(&mut d, at(TLS_CALLBACKS), 0);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callbacks_va, IMAGE_BASE + u64::from(TLS_CALLBACKS));
    assert!(tls.callback_rvas.is_empty());
}

#[test]
fn rva_to_file_offset_maps_the_raw_size_when_the_virtual_size_is_zero() {
    // Some linkers leave VirtualSize at zero; the raw size is all there is.
    let mut info = ImageInfo::parse(&image()).unwrap();
    info.sections[0].virtual_size = 0;
    let last = SECTION_RVA + SECTION_SIZE - 1;
    assert_eq!(info.rva_to_file_offset(last), Some(at(last) as u32));
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + SECTION_SIZE), None);
}

#[test]
fn rva_to_file_offset_leaves_the_zero_filled_tail_unmapped() {
    // A virtual size past the raw size is zero fill in memory, not file data.
    let mut info = ImageInfo::parse(&image()).unwrap();
    info.sections[0].virtual_size = 0x1000;
    info.sections[0].file_size = 0x100;
    assert_eq!(
        info.rva_to_file_offset(SECTION_RVA + 0xFF),
        Some(at(SECTION_RVA + 0xFF) as u32)
    );
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0x100), None);
    assert_eq!(info.rva_to_file_offset(SECTION_RVA + 0xFFF), None);

    // A section with no file data maps nothing at all.
    info.sections[0].file_size = 0;
    assert_eq!(info.rva_to_file_offset(SECTION_RVA), None);
}

#[test]
fn symbol_keys_pad_every_guid_field_but_not_the_age() {
    use victory_pe::CodeViewInfo;
    let key = |guid: [u8; 16], age: u32| {
        CodeViewInfo {
            pdb_path: String::new(),
            guid,
            age,
        }
        .symbol_key()
    };
    assert_eq!(key([0; 16], 0), format!("{}0", "0".repeat(32)));
    assert_eq!(key([0; 16], 1), format!("{}1", "0".repeat(32)));
    assert_eq!(key([0; 16], 0x10), format!("{}10", "0".repeat(32)));
    assert_eq!(
        key([0xFF; 16], u32::MAX),
        format!("{}FFFFFFFF", "F".repeat(32))
    );
    // Data1 is four little-endian bytes, Data2 and Data3 two each, Data4 is
    // eight bytes in order.
    let mut guid = [0u8; 16];
    guid[0] = 0x01;
    guid[4] = 0x02;
    guid[6] = 0x03;
    guid[15] = 0x04;
    assert_eq!(key(guid, 2), "000000010002000300000000000000042");
}

#[test]
fn a_tls_directory_too_short_for_its_callback_field_is_malformed() {
    // The directory starts 16 bytes before the end of the section, so the
    // AddressOfCallBacks field at offset 24 is past the section data.
    let short = SECTION_RVA + SECTION_SIZE - 16;
    let mut d = image();
    set_directory(&mut d, DIR_TLS, short, 40);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    // With exactly the 32 bytes up to and including the field it is read.
    let fits = SECTION_RVA + SECTION_SIZE - 32;
    let mut d = image();
    set_directory(&mut d, DIR_TLS, fits, 40);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!((tls.directory_rva, tls.callbacks_va), (fits, 0));
    assert!(tls.callback_rvas.is_empty());
}

#[test]
fn a_partial_tls_slot_at_the_end_of_the_section_is_not_read() {
    // One whole slot, then four non-zero bytes the section ends on.
    let array = SECTION_RVA + SECTION_SIZE - 12;
    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(array),
    );
    put64(&mut d, at(array), IMAGE_BASE + 0x1030);
    put32(&mut d, at(array + 8), 0x1040);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, [0x1030]);
}

#[test]
fn names_that_are_not_utf8_are_kept_with_replacement_characters() {
    let mut d = image();
    put_bytes(&mut d, DLL_NAME, b"KERN\xFFL32.dll\0");
    put_bytes(&mut d, HINT_NAME + 2, b"Exit\xC3Process\0");
    add_delay_import(&mut d);
    put_bytes(&mut d, DELAY_DLL_NAME, b"USER\x8032.dll\0");
    add_codeview(&mut d, b"RSDS");
    // Same length as "t.pdb\0", so it still ends inside the record.
    put_bytes(&mut d, CODEVIEW + 24, b"\xFE.pdb\0");

    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.imports[0].dll, "KERN\u{FFFD}L32.dll");
    assert_eq!(
        info.imports[0].entries[0].symbol,
        ImportSymbol::Name {
            hint: 0x123,
            name: "Exit\u{FFFD}Process".into()
        }
    );
    assert_eq!(info.delay_imports[0].dll, "USER\u{FFFD}32.dll");
    assert_eq!(info.codeview.unwrap().pdb_path, "\u{FFFD}.pdb");
}

#[test]
fn a_codeview_path_without_its_nul_inside_the_record_is_malformed() {
    // The record is 30 bytes, leaving six for "t.pdb\0"; a seven-byte path
    // runs its terminator past the size the debug directory gives.
    let mut d = image();
    add_codeview(&mut d, b"RSDS");
    put_bytes(&mut d, CODEVIEW + 24, b"tt.pdb\0");
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    // Growing the record by that byte makes it readable again.
    put32(&mut d, at(DEBUG_DIRECTORY) + 16, 4 + 16 + 4 + 7);
    assert_eq!(
        ImageInfo::parse(&d).unwrap().codeview.unwrap().pdb_path,
        "tt.pdb"
    );
}

#[test]
fn a_tls_directory_entry_without_a_size_is_still_read() {
    // Unlike the relocation, exception and load-config entries, which need
    // both an RVA and a size, TLS is found by its RVA alone: the directory's
    // own layout says how much to read.
    let mut d = image();
    set_directory(&mut d, DIR_TLS, TLS_DIRECTORY, 0);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.directory_rva, TLS_DIRECTORY);
    assert_eq!(tls.callback_rvas, [0x1010, 0x1020]);
}

#[test]
fn directory_slots_past_number_of_rva_and_sizes_are_ignored() {
    // Values left in slots the header does not count are not directories.
    let mut d = image();
    set_directory(&mut d, dir::EXCEPTION, 0x1390, 12);
    set_directory(&mut d, dir::BASERELOC, 0x1380, 12);
    set_directory(&mut d, dir::LOAD_CONFIG, 0x13A0, 0x40);
    put32(&mut d, OPT + layout::OPT_NUMBER_OF_RVA_AND_SIZES, 1);
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.imports.is_empty());
    assert_eq!(info.import_count(), 0);
    assert!(info.tls.is_none());
    assert!(!info.has_exception_table);
    assert!(!info.has_relocations);
    assert!(!info.has_load_config);
    // The section table still follows the full optional header.
    assert_eq!(info.sections.len(), 1);
    assert_eq!(info.rva_to_file_offset(ENTRY), Some(SECTION_FILE_OFFSET));

    // Counting the import slot brings the imports back, but not TLS.
    put32(&mut d, OPT + layout::OPT_NUMBER_OF_RVA_AND_SIZES, 2);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.import_count(), 2);
    assert!(info.tls.is_none());
}

#[test]
fn tls_vas_are_taken_relative_to_the_image_base_in_the_header() {
    const DLL_BASE: u64 = 0x1_8000_0000;
    let rebase = |d: &mut [u8]| put64(d, OPT + 24, DLL_BASE);

    // VAs written for the old base now lie below the image: malformed.
    let mut d = image();
    rebase(&mut d);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    // Rewritten for the new base, the same RVAs come back.
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        DLL_BASE + u64::from(TLS_CALLBACKS),
    );
    put64(&mut d, at(TLS_CALLBACKS), DLL_BASE + 0x1010);
    put64(&mut d, at(TLS_CALLBACKS) + 8, DLL_BASE + 0x1020);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.image_base, DLL_BASE);
    let tls = info.tls.unwrap();
    assert_eq!(tls.callbacks_va, DLL_BASE + u64::from(TLS_CALLBACKS));
    assert_eq!(tls.callback_rvas, [0x1010, 0x1020]);
}

#[test]
fn a_tls_callback_at_the_image_base_is_rva_zero_not_the_terminator() {
    // Only a null VA ends the array; the image base itself is a (strange but
    // representable) callback at RVA 0, and the walk continues past it.
    let mut d = image();
    put64(&mut d, at(TLS_CALLBACKS), IMAGE_BASE);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, [0, 0x1020]);
}

#[test]
fn an_image_without_sections_or_directories_has_only_headers() {
    let mut d = image();
    put16(&mut d, 0x46, 0); // no sections
    for index in [dir::IMPORT, DIR_TLS] {
        set_directory(&mut d, index, 0, 0);
    }
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.sections.is_empty());
    assert!(info.imports.is_empty() && info.delay_imports.is_empty());
    assert!(info.exports.is_empty());
    assert!(info.tls.is_none() && info.codeview.is_none());
    assert_eq!(info.entry_rva, ENTRY);
    // Nothing is file-backed without a section to map it.
    assert_eq!(info.rva_to_file_offset(ENTRY), None);
    assert_eq!(info.rva_to_file_offset(0), None);
}

#[test]
fn an_ordinal_thunk_for_ordinal_zero_ends_the_name_table() {
    // Only the ordinal flag set: the address bits are zero, which is the
    // terminator, so the slot and anything after it are not imports.
    let mut d = image();
    put64(&mut d, at(NAME_TABLE) + 8, 1 << 63);
    let info = ImageInfo::parse(&d).unwrap();
    let symbols: Vec<_> = info.imports[0]
        .entries
        .iter()
        .map(|e| e.symbol.to_string())
        .collect();
    assert_eq!(symbols, ["ExitProcess"]);
}

/// The forwarder string of ordinal 2, rewritten in place (16 bytes free).
fn with_forwarder(forward: &[u8]) -> Vec<u8> {
    let mut d = image();
    add_exports(&mut d);
    put_bytes(&mut d, 0x13B8, &[0; 16]);
    put_bytes(&mut d, 0x13B8, forward);
    d
}

#[test]
fn forwarders_split_at_the_last_dot_and_keep_a_dotted_dll_name() {
    for (forward, kept) in [
        (&b"x.dll.Bar\0"[..], "x.dll.Bar"),
        (b"api-ms-x.Foo\0", "api-ms-x.Foo"),
        // Nothing before the dot is still a forwarder with an empty DLL.
        (b".Foo\0", ".Foo"),
    ] {
        let info = ImageInfo::parse(&with_forwarder(forward)).unwrap();
        let fwd = info.exports.iter().find(|e| e.ordinal == 2).unwrap();
        assert_eq!((fwd.rva, fwd.forward.as_deref()), (None, Some(kept)));
    }
}

#[test]
fn a_forwarder_without_a_name_or_a_valid_ordinal_makes_the_image_malformed() {
    // One bad forwarder fails the whole parse, not just its own export.
    for forward in [
        &b"NTDLLRtlFoo\0"[..],
        b"NTDLL.\0",
        b"NTDLL.#\0",
        b"NTDLL.#x\0",
        b"NTDLL.#70000\0",
    ] {
        assert!(
            matches!(
                ImageInfo::parse(&with_forwarder(forward)),
                Err(PeError::Malformed(_))
            ),
            "{}",
            String::from_utf8_lossy(forward)
        );
    }
}

#[test]
fn an_ordinal_with_two_names_is_listed_once() {
    // `fwdord` is pointed at ordinal 1 as well, so ordinal 3 loses its name.
    let mut d = image();
    add_exports(&mut d);
    put16(&mut d, at(0x1390) + 2, 0);
    let info = ImageInfo::parse(&d).unwrap();
    let ordinals: Vec<_> = info.exports.iter().map(|e| e.ordinal).collect();
    assert_eq!(ordinals, [1, 2, 3]);
    let first = &info.exports[0];
    assert_eq!(first.rva, Some(ENTRY));
    assert!(matches!(first.name.as_deref(), Some("local" | "fwdord")));
    let third = &info.exports[2];
    assert_eq!(
        (third.name.as_deref(), third.forward.as_deref()),
        (None, Some("NTDLL.#7"))
    );
}

#[test]
fn a_delay_iat_that_ends_exactly_at_the_size_of_image_is_inside_it() {
    let mut d = image();
    add_delay_import(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 12, SIZE_OF_IMAGE - 8);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(
        info.delay_imports[0].entries[0].iat_slot_rva,
        SIZE_OF_IMAGE - 8
    );
}

#[test]
fn delay_descriptor_attributes_are_not_checked() {
    // Attributes 0 marks an old VA-based descriptor; its fields are still
    // read as RVAs, as for the RVA-based kind.
    let mut d = image();
    add_delay_import(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS), 0);
    let info = ImageInfo::parse(&d).unwrap();
    let user32 = &info.delay_imports[0];
    assert_eq!(user32.dll, "USER32.dll");
    assert_eq!(user32.entries[0].iat_slot_rva, DELAY_IAT);
}

#[test]
fn a_codeview_entry_after_other_debug_entries_is_found() {
    // A POGO entry first, then the CodeView one: entries of other types are
    // skipped, not taken as the end of the directory.
    const TWO_ENTRIES: u32 = 0x1280;
    let mut d = image();
    add_codeview(&mut d, b"RSDS");
    let codeview_entry = d[at(DEBUG_DIRECTORY)..at(DEBUG_DIRECTORY) + 28].to_vec();
    put32(&mut d, at(TWO_ENTRIES) + 12, 13); // IMAGE_DEBUG_TYPE_POGO
    put_bytes(&mut d, TWO_ENTRIES + 28, &codeview_entry);
    set_directory(&mut d, DIR_DEBUG, TWO_ENTRIES, 56);
    let cv = ImageInfo::parse(&d).unwrap().codeview.unwrap();
    assert_eq!((cv.pdb_path.as_str(), cv.age), ("t.pdb", 0x2A));
}

#[test]
fn a_debug_directory_that_is_not_whole_entries_is_malformed() {
    let mut d = image();
    add_codeview(&mut d, b"RSDS");
    for size in [27, 30, 55] {
        set_directory(&mut d, DIR_DEBUG, DEBUG_DIRECTORY, size);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "debug directory of {size} bytes"
        );
    }
}

#[test]
fn the_codeview_record_is_read_at_its_file_offset_not_its_rva() {
    let mut d = image();
    add_codeview(&mut d, b"RSDS");
    // AddressOfRawData is ignored: a zero there changes nothing.
    put32(&mut d, at(DEBUG_DIRECTORY) + 20, 0);
    let cv = ImageInfo::parse(&d).unwrap().codeview.unwrap();
    assert_eq!(cv.pdb_path, "t.pdb");
    // PointerToRawData past the end of the file is malformed.
    let past_end = d.len() as u32;
    put32(&mut d, at(DEBUG_DIRECTORY) + 24, past_end);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn files_that_are_not_pe_images_are_unsupported() {
    // An import library and a bare AMD64 COFF object are recognised for what
    // they are and turned away, not parsed as a PE.
    let mut coff = vec![0u8; 20];
    put16(&mut coff, 0, MACHINE_AMD64);
    // The kind is read from the first 16 bytes, so the archive is padded.
    let mut archive = b"!<arch>\n".to_vec();
    archive.resize(68, b' ');
    for (what, data) in [("archive", archive), ("COFF object", coff)] {
        assert!(
            matches!(ImageInfo::parse(&data), Err(PeError::Unsupported(_))),
            "{what}: {:?}",
            ImageInfo::parse(&data)
        );
    }
}

#[test]
fn two_descriptors_for_the_same_dll_stay_separate_modules() {
    // A second KERNEL32.dll descriptor, as linkers emit when objects import
    // from the same DLL through different import libraries.
    const SECOND_DESCRIPTOR: u32 = IMPORT_DESCRIPTORS + 20;
    const NAMES: u32 = 0x1140;
    const SLOTS: u32 = 0x1150;
    let mut d = image();
    set_directory(&mut d, dir::IMPORT, IMPORT_DESCRIPTORS, 60);
    let desc = at(SECOND_DESCRIPTOR);
    put32(&mut d, desc, NAMES);
    put32(&mut d, desc + 12, DLL_NAME);
    put32(&mut d, desc + 16, SLOTS);
    for table in [NAMES, SLOTS] {
        put64(&mut d, at(table), (1 << 63) | 9);
    }
    let info = ImageInfo::parse(&d).unwrap();
    let modules: Vec<_> = info
        .imports
        .iter()
        .map(|m| (m.dll.as_str(), m.entries.len()))
        .collect();
    assert_eq!(modules, [("KERNEL32.dll", 2), ("KERNEL32.dll", 1)]);
    let second = &info.imports[1].entries[0];
    assert_eq!(
        (second.symbol.clone(), second.iat_slot_rva),
        (ImportSymbol::Ordinal(9), SLOTS)
    );
    assert_eq!(info.import_count(), 3);
}

#[test]
fn a_name_table_outside_the_sections_is_rejected() {
    for table in [SECTION_RVA + SECTION_SIZE + 0x100, SIZE_OF_IMAGE + 0x100] {
        let mut d = image();
        put32(&mut d, at(IMPORT_DESCRIPTORS), table);
        assert!(ImageInfo::parse(&d).is_err(), "name table at {table:#x}");

        let mut d = image();
        add_delay_import(&mut d);
        put32(&mut d, at(DELAY_DESCRIPTORS) + 16, table);
        assert!(
            ImageInfo::parse(&d).is_err(),
            "delay name table at {table:#x}"
        );
    }
}

#[test]
fn import_descriptors_that_run_off_the_section_are_rejected() {
    // The only descriptor fills the last 20 bytes, leaving no terminator.
    const LAST: u32 = SECTION_RVA + SECTION_SIZE - 20;
    let mut d = image();
    let desc = at(IMPORT_DESCRIPTORS);
    let copy: Vec<u8> = d[desc..desc + 20].to_vec();
    put_bytes(&mut d, LAST, &copy);
    set_directory(&mut d, dir::IMPORT, LAST, 20);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn the_import_directory_size_does_not_limit_the_descriptors_read() {
    // Descriptors are read up to the all-zero terminator, as the loader does,
    // even when the directory entry covers only the first one.
    let mut d = image();
    set_directory(&mut d, dir::IMPORT, IMPORT_DESCRIPTORS, 20);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.imports.len(), 1);
    assert_eq!(info.import_count(), 2);
}

#[test]
fn an_export_past_the_raw_data_is_kept_as_its_rva() {
    // Exported data in the zero-filled part of the image has no file bytes,
    // but its RVA is still what the export table gives.
    const BSS: u32 = SECTION_RVA + SECTION_SIZE + 0x200;
    let mut d = image();
    add_exports(&mut d);
    put32(&mut d, at(0x1378), BSS);
    let info = ImageInfo::parse(&d).unwrap();
    let local = info.exports.iter().find(|e| e.ordinal == 1).unwrap();
    assert_eq!(
        (local.name.as_deref(), local.rva, local.forward.as_deref()),
        (Some("local"), Some(BSS), None)
    );
    assert_eq!(info.rva_to_file_offset(BSS), None);
}

#[test]
fn an_import_descriptor_with_any_field_set_is_not_the_terminator() {
    // The descriptor after KERNEL32's ends the list only when all five
    // fields are zero. A stray timestamp or forwarder chain makes it a
    // descriptor of its own, whose missing DLL name is malformed.
    for field in [4, 8] {
        let mut d = image();
        put32(&mut d, at(IMPORT_DESCRIPTORS) + 20 + field, 0x1234);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "terminator field at +{field}"
        );
    }
    // The same fields on a real descriptor (a bound import's -1 markers)
    // are not checked.
    let mut d = image();
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 4, u32::MAX);
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 8, u32::MAX);
    assert_eq!(ImageInfo::parse(&d).unwrap().import_count(), 2);
}

#[test]
fn a_delay_descriptor_with_any_field_set_is_not_the_terminator() {
    for field in [0, 8] {
        let mut d = image();
        add_delay_import(&mut d);
        put32(&mut d, at(DELAY_DESCRIPTORS) + 32 + field, 1);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "terminator field at +{field}"
        );
    }
}

#[test]
fn a_descriptor_without_a_dll_name_or_name_table_is_malformed() {
    let mut d = image();
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 12, 0);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    // With both the name table and the IAT at 0 there is nothing to read.
    let mut d = image();
    put32(&mut d, at(IMPORT_DESCRIPTORS), 0);
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, 0);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    let mut d = image();
    add_delay_import(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 4, 0);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    let mut d = image();
    add_delay_import(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 16, 0);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn an_iat_at_rva_zero_is_taken_as_written() {
    // Only the end of each slot is checked against SizeOfImage, so an IAT
    // at RVA 0 is accepted and its slots overlap the headers. The loader
    // that fills them must not trust these RVAs to be inside a section.
    let mut d = image();
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, 0);
    let info = ImageInfo::parse(&d).unwrap();
    let slots: Vec<_> = info.imports[0]
        .entries
        .iter()
        .map(|e| e.iat_slot_rva)
        .collect();
    assert_eq!(slots, [0, 8]);

    let mut d = image();
    add_delay_import(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 12, 0);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.delay_imports[0].entries[0].iat_slot_rva, 0);
}

#[test]
fn a_hint_name_entry_without_room_for_the_name_is_malformed() {
    // The hint fits in the last two bytes of the section; the name does not.
    let mut d = image();
    put64(
        &mut d,
        at(NAME_TABLE),
        u64::from(SECTION_RVA + SECTION_SIZE - 2),
    );
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
    // A hint of 0xFFFF is kept as it is.
    let mut d = image();
    put16(&mut d, at(HINT_NAME), 0xFFFF);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(
        info.imports[0].entries[0].symbol,
        ImportSymbol::Name {
            hint: 0xFFFF,
            name: "ExitProcess".into()
        }
    );
}

#[test]
fn a_name_ordinal_past_the_address_table_is_malformed() {
    // Three functions, so name ordinal indexes 3 and up have no address.
    for index in [3, 7, u16::MAX] {
        let mut d = image();
        add_exports(&mut d);
        put16(&mut d, at(0x1390), index);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "name ordinal index {index}"
        );
    }
}

#[test]
fn export_ordinals_must_fit_in_16_bits() {
    let ed = at(EXPORT_DIRECTORY);
    // Base 0 is allowed: the first export is ordinal 0.
    let mut d = image();
    add_exports(&mut d);
    put32(&mut d, ed + 16, 0);
    let ordinals: Vec<_> = ImageInfo::parse(&d)
        .unwrap()
        .exports
        .iter()
        .map(|e| e.ordinal)
        .collect();
    assert_eq!(ordinals, [0, 1, 2]);
    // A base past 0xFFFF, or one that pushes the last of the three
    // functions past it, has no valid ordinals.
    for base in [0xFFFE, 0xFFFF, 0x1_0000] {
        let mut d = image();
        add_exports(&mut d);
        put32(&mut d, ed + 16, base);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "ordinal base {base:#x}"
        );
    }
}

#[test]
fn export_tables_outside_the_sections_are_malformed() {
    let ed = at(EXPORT_DIRECTORY);
    let past = SECTION_RVA + SECTION_SIZE + 0x100;
    for (what, at_field) in [
        ("address table", ed + 28),
        ("name pointer table", ed + 32),
        ("a name", at(0x1384)),
    ] {
        let mut d = image();
        add_exports(&mut d);
        put32(&mut d, at_field, past);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "{what}"
        );
    }
    // Counts too large for the section are caught the same way.
    for at_field in [ed + 20, ed + 24] {
        let mut d = image();
        add_exports(&mut d);
        put32(&mut d, at_field, 0x1000_0000);
        assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
    }
}

#[test]
fn the_export_directory_dll_name_is_not_read() {
    // The image's own name in the export directory is not used, so a bad
    // one does not stop the exports from being read.
    for name in [0, SIZE_OF_IMAGE + 0x100] {
        let mut d = image();
        add_exports(&mut d);
        put32(&mut d, at(EXPORT_DIRECTORY) + 12, name);
        let info = ImageInfo::parse(&d).unwrap();
        assert_eq!(info.exports.len(), 3, "DLL name at {name:#x}");
    }
}

#[test]
fn tls_callbacks_past_size_of_image_are_kept_as_written() {
    // Only the 32-bit RVA range is checked: a callback below 4 GiB but past
    // SizeOfImage is reported as written, for the caller to judge.
    let mut d = image();
    let past = SIZE_OF_IMAGE + 0x10;
    put64(&mut d, at(TLS_CALLBACKS), IMAGE_BASE + u64::from(past));
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, [past, 0x1020]);
}

#[test]
fn header_values_are_not_checked_against_each_other() {
    // An entry point past the image, a stack commit larger than its
    // reserve and an image base that is not 64 KiB aligned are all taken
    // as written; the loader decides what to make of them.
    let mut d = image();
    put32(&mut d, OPT + 16, SIZE_OF_IMAGE + 0x1000);
    put64(&mut d, OPT + 24, IMAGE_BASE + 0x1000);
    put64(&mut d, OPT + 72, 0x1000);
    put64(&mut d, OPT + 80, 0x10_0000);
    // Keep the TLS VAs relative to the new base.
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + 0x1000 + u64::from(TLS_CALLBACKS),
    );
    put64(&mut d, at(TLS_CALLBACKS), IMAGE_BASE + 0x1000 + 0x1010);
    put64(&mut d, at(TLS_CALLBACKS) + 8, IMAGE_BASE + 0x1000 + 0x1020);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.entry_rva, SIZE_OF_IMAGE + 0x1000);
    assert_eq!(info.image_base, IMAGE_BASE + 0x1000);
    assert_eq!((info.stack_reserve, info.stack_commit), (0x1000, 0x10_0000));
    assert_eq!(info.rva_to_file_offset(info.entry_rva), None);
    assert_eq!(info.tls.unwrap().callback_rvas, [0x1010, 0x1020]);
}

#[test]
fn data_appended_after_the_last_section_is_not_mapped() {
    // An overlay (a signature, an installer payload) follows the raw data of
    // the last section. It does not change the parse, and no RVA reaches it.
    let plain = ImageInfo::parse(&image()).unwrap();
    let mut d = image();
    let overlay_offset = d.len() as u32;
    d.extend_from_slice(&[0xCC; 0x300]);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(format!("{info:?}"), format!("{plain:?}"));
    for rva in [
        SECTION_RVA + SECTION_SIZE,
        SECTION_RVA + SECTION_SIZE + 0x2FF,
    ] {
        assert_eq!(info.rva_to_file_offset(rva), None, "rva {rva:#x}");
    }
    assert!(
        (0..SIZE_OF_IMAGE)
            .step_by(0x10)
            .filter_map(|rva| info.rva_to_file_offset(rva))
            .all(|offset| offset < overlay_offset)
    );
}

#[test]
fn flag_directories_are_judged_by_their_entry_not_their_target() {
    // Relocation, exception and load-config presence come from the directory
    // entry alone: an entry past SizeOfImage, past the section data, or one
    // byte long still counts, because nothing reads what it points at.
    for (rva, size) in [
        (SIZE_OF_IMAGE + 0x1000, 12),
        (SECTION_RVA + SECTION_SIZE + 0x10, 12),
        (0xFFFF_FFF0, 0xFFFF_FFFF),
        (0x1380, 1),
    ] {
        let mut d = image();
        set_directory(&mut d, dir::BASERELOC, rva, size);
        set_directory(&mut d, dir::EXCEPTION, rva, size);
        set_directory(&mut d, dir::LOAD_CONFIG, rva, size);
        let info = ImageInfo::parse(&d).unwrap();
        assert!(info.has_relocations, "{rva:#x}+{size:#x}");
        assert!(info.has_exception_table, "{rva:#x}+{size:#x}");
        assert!(info.has_load_config, "{rva:#x}+{size:#x}");
        // The rest of the parse is untouched by those entries.
        assert_eq!(info.import_count(), 2);
        assert_eq!(info.tls.unwrap().callback_rvas, [0x1010, 0x1020]);
    }
}

#[test]
fn relocations_need_both_a_directory_and_no_stripped_flag() {
    let fh = layout::file_header_offset(&image()).unwrap();
    let flags_at = fh + layout::FILE_HEADER_CHARACTERISTICS;
    let stripped = 0x0022 | layout::file_flags::RELOCS_STRIPPED;
    for (directory, flags, expected) in [
        (false, 0x0022, false),
        (false, stripped, false),
        (true, 0x0022, true),
        (true, stripped, false),
    ] {
        let mut d = image();
        if directory {
            set_directory(&mut d, dir::BASERELOC, 0x1380, 12);
        }
        put16(&mut d, flags_at, flags);
        let info = ImageInfo::parse(&d).unwrap();
        assert_eq!(info.has_relocations, expected, "{directory} {flags:#x}");
        // The flag word itself is reported as written either way.
        assert_eq!(info.characteristics, flags);
        // Stripping relocations says nothing about the other directories.
        assert!(!info.has_exception_table && !info.has_load_config);
    }
}

#[test]
fn nt_headers_after_a_dos_stub_are_found_through_e_lfanew() {
    // Linkers put a DOS stub between the MZ header and the NT headers. Moving
    // the NT headers 0x40 bytes further in changes nothing but where they are.
    let plain = ImageInfo::parse(&image()).unwrap();
    let mut d = image();
    let end = OPT + 240 + 40; // the end of the one section header
    let headers = d[0x40..end].to_vec();
    d[0x40..end].fill(0);
    d[0x80..end + 0x40].copy_from_slice(&headers);
    d[0x40..0x80].fill(0x90);
    put32(&mut d, layout::DOS_E_LFANEW, 0x80);
    assert!(end + 0x40 <= SECTION_FILE_OFFSET as usize);
    assert_eq!(layout::file_header_offset(&d), Some(0x84));
    assert_eq!(layout::optional_header_offset(&d), Some(OPT + 0x40));
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(format!("{info:?}"), format!("{plain:?}"));
}

#[test]
fn the_section_table_follows_the_declared_optional_header_size() {
    // SizeOfOptionalHeader, not the 240 bytes of a PE32+ header with 16
    // directories, says where the section table starts.
    let plain = ImageInfo::parse(&image()).unwrap();
    let mut d = image();
    let table = OPT + 240;
    let header = d[table..table + 40].to_vec();
    d[table..table + 40].fill(0);
    d[table + 16..table + 56].copy_from_slice(&header);
    put16(&mut d, 0x54, 256);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(format!("{info:?}"), format!("{plain:?}"));

    // Declaring the longer header without moving the table reads the
    // section header 16 bytes late: the section lands at RVA 0, and the
    // import directory at 0x1100 is no longer inside any section.
    let mut d = image();
    put16(&mut d, 0x54, 256);
    assert!(ImageInfo::parse(&d).is_err());
}

#[test]
fn stack_sizes_are_read_as_64_bit_values() {
    let mut d = image();
    put64(&mut d, OPT + 72, 0x1_0000_0000 | 0x20_0000);
    put64(&mut d, OPT + 80, 0x2_0000_0000 | 0x2000);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.stack_reserve, 0x1_0020_0000);
    assert_eq!(info.stack_commit, 0x2_0000_2000);
}

/// The synthetic image with its NT headers moved 0x40 bytes further in, so
/// that header offsets only come out right when they follow `e_lfanew`.
fn image_after_a_stub() -> Vec<u8> {
    let mut d = image();
    let end = OPT + 240 + 40;
    let headers = d[0x40..end].to_vec();
    d[0x40..end].fill(0);
    d[0x80..end + 0x40].copy_from_slice(&headers);
    put32(&mut d, layout::DOS_E_LFANEW, 0x80);
    d
}

#[test]
fn layout_offsets_patch_the_fields_image_info_reads() {
    // Patching code writes through `layout`; `ImageInfo` reads through
    // `object`. Both must name the same bytes, wherever the headers are.
    let mut d = image_after_a_stub();
    let fh = layout::file_header_offset(&d).unwrap();
    let opt = layout::optional_header_offset(&d).unwrap();
    assert_eq!((fh, opt), (0x84, OPT + 0x40));

    put16(
        &mut d,
        fh + layout::FILE_HEADER_CHARACTERISTICS,
        0x0022 | layout::file_flags::DLL,
    );
    put32(
        &mut d,
        opt + layout::OPT_ADDRESS_OF_ENTRY_POINT,
        ENTRY + 0x10,
    );
    let flags = layout::dll_flags::HIGH_ENTROPY_VA
        | layout::dll_flags::DYNAMIC_BASE
        | layout::dll_flags::NX_COMPAT;
    put16(&mut d, opt + layout::OPT_DLL_CHARACTERISTICS, flags);
    let info = ImageInfo::parse(&d).unwrap();
    assert!(info.is_dll());
    assert_eq!(info.entry_rva, ENTRY + 0x10);
    assert_eq!(info.dll_characteristics, flags);

    // The TLS directory slot found through `layout` is the one parsed.
    let tls = layout::data_directory_offset(&d, DIR_TLS).unwrap();
    assert_eq!(layout::read_u32(&d, tls), Some(TLS_DIRECTORY));
    assert_eq!(info.tls.unwrap().directory_rva, TLS_DIRECTORY);
}

#[test]
fn directories_set_through_layout_are_the_ones_reported() {
    let plain = ImageInfo::parse(&image_after_a_stub()).unwrap();
    assert!(!plain.has_relocations && !plain.has_exception_table && !plain.has_load_config);
    for (index, check) in [
        (
            dir::BASERELOC,
            (|i: &ImageInfo| i.has_relocations) as fn(&ImageInfo) -> bool,
        ),
        (dir::EXCEPTION, |i| i.has_exception_table),
        (dir::LOAD_CONFIG, |i| i.has_load_config),
    ] {
        let mut d = image_after_a_stub();
        let slot = layout::data_directory_offset(&d, index).unwrap();
        put32(&mut d, slot, 0x1300);
        put32(&mut d, slot + 4, 8);
        let info = ImageInfo::parse(&d).unwrap();
        assert!(check(&info), "directory {index}");
    }
}

#[test]
fn the_checksum_bound_import_and_iat_directories_change_nothing_read() {
    // Prepare rewrites the checksum and may clear bound imports; neither is
    // part of what `ImageInfo` reports, nor is the IAT directory, which only
    // says where the loader may write.
    let plain = ImageInfo::parse(&image()).unwrap();
    let mut d = image();
    let opt = layout::optional_header_offset(&d).unwrap();
    put32(&mut d, opt + layout::OPT_CHECKSUM, 0xDEAD_BEEF);
    for index in [dir::BOUND_IMPORT, dir::IAT] {
        let slot = layout::data_directory_offset(&d, index).unwrap();
        put32(&mut d, slot, 0x1380);
        put32(&mut d, slot + 4, 0x20);
    }
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(format!("{info:?}"), format!("{plain:?}"));
}

#[test]
fn dll_characteristic_flags_are_distinct_single_bits() {
    use layout::dll_flags::*;
    let all = [
        HIGH_ENTROPY_VA,
        DYNAMIC_BASE,
        FORCE_INTEGRITY,
        NX_COMPAT,
        APPCONTAINER,
        GUARD_CF,
    ];
    let mut seen = 0u16;
    for flag in all {
        assert_eq!(flag.count_ones(), 1, "{flag:#x}");
        assert_eq!(seen & flag, 0, "{flag:#x}");
        seen |= flag;
    }
    let mut d = image();
    put16(&mut d, OPT + layout::OPT_DLL_CHARACTERISTICS, seen);
    assert_eq!(ImageInfo::parse(&d).unwrap().dll_characteristics, seen);
}
