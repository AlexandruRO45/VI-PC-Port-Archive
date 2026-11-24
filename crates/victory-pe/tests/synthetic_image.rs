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

#[test]
fn each_flag_directory_sets_only_its_own_flag() {
    // The three presence flags come from three separate directory slots;
    // filling one slot does not report the other two.
    let flags = |info: &ImageInfo| {
        (
            info.has_relocations,
            info.has_exception_table,
            info.has_load_config,
        )
    };
    for (index, expected) in [
        (dir::BASERELOC, (true, false, false)),
        (dir::EXCEPTION, (false, true, false)),
        (dir::LOAD_CONFIG, (false, false, true)),
    ] {
        let mut d = image();
        set_directory(&mut d, index, 0x1380, 12);
        let info = ImageInfo::parse(&d).unwrap();
        assert_eq!(flags(&info), expected, "directory {index}");
    }
    // The neighbouring slots that are not flags (bound imports, the IAT)
    // report none of them.
    for index in [dir::BOUND_IMPORT, dir::IAT] {
        let mut d = image();
        set_directory(&mut d, index, 0x1380, 12);
        let info = ImageInfo::parse(&d).unwrap();
        assert_eq!(flags(&info), (false, false, false), "directory {index}");
    }
}

#[test]
fn overlapping_sections_map_through_the_first_one_in_the_table() {
    // Section headers are untrusted and may overlap. The lookup takes the
    // first section in table order that holds the RVA, so reordering the
    // table changes the answer but never makes it ambiguous.
    use victory_pe::SectionInfo;
    let mut info = ImageInfo::parse(&image()).unwrap();
    let shadow = SectionInfo {
        name: ".shadow".into(),
        rva: SECTION_RVA + 0x100,
        virtual_size: 0x100,
        file_offset: 0x8000,
        file_size: 0x100,
        characteristics: 0,
    };
    info.sections.push(shadow.clone());
    let inside = SECTION_RVA + 0x180;
    assert_eq!(info.rva_to_file_offset(inside), Some(at(inside) as u32));

    info.sections.reverse();
    assert_eq!(info.rva_to_file_offset(inside), Some(0x8080));
    // Outside the shadow, the original section still answers.
    assert_eq!(
        info.rva_to_file_offset(SECTION_RVA),
        Some(SECTION_FILE_OFFSET)
    );
    assert_eq!(
        info.rva_to_file_offset(SECTION_RVA + 0x200),
        Some(at(SECTION_RVA + 0x200) as u32)
    );
}

#[test]
fn a_section_below_the_rva_does_not_wrap_around() {
    // An RVA below a section's start is not inside it, even when the
    // subtraction would wrap to a small offset in unsigned arithmetic.
    let info = ImageInfo::parse(&image()).unwrap();
    for rva in [0, 1, SECTION_RVA - 1, SECTION_FILE_OFFSET] {
        assert_eq!(info.rva_to_file_offset(rva), None, "{rva:#x}");
    }
}

#[test]
fn only_exact_subsystem_values_have_names() {
    use layout::subsystem::*;
    // Values next to the named ones, and the known ones with high bits set,
    // are all "other": the name is a lookup, not a mask.
    for value in [
        0,
        WINDOWS_GUI - 1,
        WINDOWS_CUI + 1,
        XBOX - 1,
        XBOX + 1,
        XBOX_CODE_CATALOG + 1,
        WINDOWS_GUI | 0x100,
        XBOX | 0x8000,
        u16::MAX,
    ] {
        let mut d = image();
        put16(&mut d, OPT + 68, value);
        let info = ImageInfo::parse(&d).unwrap();
        assert_eq!(info.subsystem, value);
        assert_eq!(info.subsystem_name(), "other", "{value:#x}");
    }
}

#[test]
fn the_dll_bit_alone_decides_is_dll() {
    use layout::file_flags::DLL;
    let mut info = ImageInfo::parse(&image()).unwrap();
    for (characteristics, dll) in [(DLL, true), (u16::MAX, true), (!DLL, false), (0, false)] {
        info.characteristics = characteristics;
        assert_eq!(info.is_dll(), dll, "{characteristics:#x}");
    }
}

#[test]
fn the_name_table_not_the_iat_decides_how_many_entries_there_are() {
    // The name table ends after its first entry, but the IAT still holds a
    // second (stale) thunk: only the named slot is an import.
    let mut d = image();
    put64(&mut d, at(NAME_TABLE) + 8, 0);
    let info = ImageInfo::parse(&d).unwrap();
    let entries: Vec<_> = info.imports[0]
        .entries
        .iter()
        .map(|e| (e.symbol.to_string(), e.iat_slot_rva))
        .collect();
    assert_eq!(entries, [("ExitProcess".to_string(), IAT)]);
    assert_eq!(info.import_count(), 1);
}

#[test]
fn a_dll_imported_both_directly_and_delay_loaded_is_listed_in_each_table() {
    let mut d = image();
    add_delay_import(&mut d);
    put_bytes(&mut d, DELAY_DLL_NAME, b"KERNEL32.dll\0");
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.imports.len(), 1);
    assert_eq!(info.delay_imports.len(), 1);
    assert_eq!(info.imports[0].dll, "KERNEL32.dll");
    assert_eq!(info.delay_imports[0].dll, "KERNEL32.dll");
    // Each table keeps its own IAT slots; nothing is merged.
    assert_eq!(info.imports[0].entries[0].iat_slot_rva, IAT);
    assert_eq!(info.delay_imports[0].entries[0].iat_slot_rva, DELAY_IAT);
    assert_eq!(info.import_count(), 2);
}

#[test]
fn import_symbols_compare_on_every_field_but_display_without_the_hint() {
    use std::collections::BTreeSet;
    let name = |hint: u16, name: &str| ImportSymbol::Name {
        hint,
        name: name.into(),
    };
    // The hint is part of a symbol's identity even though it is not shown.
    assert_ne!(name(1, "Sleep"), name(2, "Sleep"));
    assert_eq!(name(1, "Sleep").to_string(), name(2, "Sleep").to_string());
    // Names order before ordinals, then by hint, then by name.
    let set: BTreeSet<_> = [
        ImportSymbol::Ordinal(3),
        name(2, "A"),
        ImportSymbol::Ordinal(1),
        name(1, "Z"),
        name(1, "Z"),
    ]
    .into_iter()
    .collect();
    let shown: Vec<_> = set.iter().map(ToString::to_string).collect();
    assert_eq!(shown, ["Z", "A", "#1", "#3"]);
}

#[test]
fn errors_name_their_kind_and_what_went_wrong() {
    let message = |d: &[u8]| ImageInfo::parse(d).unwrap_err().to_string();

    let mut d = image();
    put16(&mut d, 0x44, 0xAA64);
    assert_eq!(
        message(&d),
        "unsupported image: machine 0xaa64 is not AMD64"
    );

    let mut d = image();
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, SIZE_OF_IMAGE - 8);
    assert_eq!(
        message(&d),
        format!(
            "malformed image: IAT at {:#x} runs outside the image",
            SIZE_OF_IMAGE - 8
        )
    );

    let mut d = image();
    put64(&mut d, at(TLS_CALLBACKS) + 8, IMAGE_BASE - 0x1000);
    assert_eq!(
        message(&d),
        format!(
            "malformed image: VA 0x{:x} is outside the image",
            IMAGE_BASE - 0x1000
        )
    );

    let mut d = image();
    set_directory(&mut d, DIR_TLS, SECTION_RVA + SECTION_SIZE + 0x100, 40);
    assert_eq!(
        message(&d),
        "malformed image: TLS directory outside of sections"
    );

    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(SECTION_RVA + SECTION_SIZE + 0x100),
    );
    assert_eq!(
        message(&d),
        "malformed image: TLS callback array outside of sections"
    );

    // Errors from `object` keep its own text behind the same prefix.
    let mut d = image();
    d[0x40] = b'X';
    assert!(
        message(&d).starts_with("malformed image: "),
        "{}",
        message(&d)
    );
    // Too short to tell what kind of file it is: malformed, not unsupported.
    assert!(
        message(b"").starts_with("malformed image: "),
        "{}",
        message(b"")
    );
    // A 32-bit optional header is named by the kind found.
    let mut d = image();
    put16(&mut d, OPT, 0x10B);
    assert_eq!(message(&d), "unsupported image: expected PE32+, found Pe32");
}

#[test]
fn the_callback_pointer_read_through_layout_is_the_one_reported() {
    let d = image();
    let info = ImageInfo::parse(&d).unwrap();
    let tls = info.tls.as_ref().unwrap();
    let entry = layout::data_directory_offset(&d, DIR_TLS).unwrap();
    let directory_rva = layout::read_u32(&d, entry).unwrap();
    assert_eq!(directory_rva, tls.directory_rva);
    let directory = info.rva_to_file_offset(directory_rva).unwrap() as usize;
    let field = directory + layout::TLS64_ADDRESS_OF_CALLBACKS;
    assert_eq!(layout::read_u64(&d, field), Some(tls.callbacks_va));
    // Each callback slot is a 64-bit VA at the preferred base.
    let array = info
        .rva_to_file_offset(u32::try_from(tls.callbacks_va - info.image_base).unwrap())
        .unwrap() as usize;
    for (i, &rva) in tls.callback_rvas.iter().enumerate() {
        assert_eq!(
            layout::read_u64(&d, array + i * 8),
            Some(info.image_base + u64::from(rva))
        );
    }
    assert_eq!(
        layout::read_u64(&d, array + tls.callback_rvas.len() * 8),
        Some(0)
    );
}

#[test]
fn forwarders_by_ordinal_cover_the_whole_16_bit_range() {
    for (forward, kept) in [
        (&b"NTDLL.#0\0"[..], "NTDLL.#0"),
        (b"NTDLL.#1\0", "NTDLL.#1"),
        (b"NTDLL.#65535\0", "NTDLL.#65535"),
        // A forwarder by ordinal is re-rendered, so leading zeros go.
        (b"NTDLL.#007\0", "NTDLL.#7"),
    ] {
        let info = ImageInfo::parse(&with_forwarder(forward)).unwrap();
        let fwd = info.exports.iter().find(|e| e.ordinal == 2).unwrap();
        assert_eq!(
            (fwd.rva, fwd.forward.as_deref()),
            (None, Some(kept)),
            "{}",
            String::from_utf8_lossy(forward)
        );
    }
    assert!(matches!(
        ImageInfo::parse(&with_forwarder(b"NTDLL.#65536\0")),
        Err(PeError::Malformed(_))
    ));
}

/// Shrinks the one section's raw data to `raw` bytes; the rest of its
/// virtual size becomes zero fill that the file does not hold.
fn with_raw_size(raw: u32) -> Vec<u8> {
    let mut d = image();
    put32(&mut d, OPT + 240 + 16, raw);
    d
}

#[test]
fn an_iat_in_the_zero_filled_tail_of_a_section_is_accepted() {
    // The loader writes the IAT in memory, so it may sit where the file has
    // no bytes; only the name table it is paired with must be file data.
    let mut d = with_raw_size(0x280);
    let tail_iat = SECTION_RVA + 0x300;
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, tail_iat);
    let info = ImageInfo::parse(&d).unwrap();
    let slots: Vec<u32> = info.imports[0]
        .entries
        .iter()
        .map(|e| e.iat_slot_rva)
        .collect();
    assert_eq!(slots, [tail_iat, tail_iat + 8]);
    assert_eq!(info.rva_to_file_offset(tail_iat), None);
    assert_eq!(
        info.rva_to_file_offset(NAME_TABLE),
        Some(at(NAME_TABLE) as u32)
    );
}

#[test]
fn a_name_table_in_the_zero_filled_tail_is_malformed() {
    // The name table is read from the file; zero fill past the raw data is
    // not there to read, even though it is inside the section in memory.
    let mut d = with_raw_size(0x280);
    put32(&mut d, at(IMPORT_DESCRIPTORS), SECTION_RVA + 0x300);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn tls_data_in_the_zero_filled_tail_is_outside_the_sections() {
    // Raw data ends just before the callback array: the directory is read,
    // the array is not.
    let raw = TLS_CALLBACKS - SECTION_RVA;
    match ImageInfo::parse(&with_raw_size(raw)) {
        Err(PeError::Malformed(m)) => assert!(m.contains("callback array"), "{m}"),
        other => panic!("expected a malformed callback array, got {other:?}"),
    }
    // Raw data ends inside the directory, before its callback field.
    let raw = TLS_DIRECTORY - SECTION_RVA + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
    match ImageInfo::parse(&with_raw_size(raw)) {
        Err(PeError::Malformed(m)) => assert!(m.contains("truncated TLS"), "{m}"),
        other => panic!("expected a truncated TLS directory, got {other:?}"),
    }
    // Raw data ends before the directory starts.
    let raw = TLS_DIRECTORY - SECTION_RVA;
    match ImageInfo::parse(&with_raw_size(raw)) {
        Err(PeError::Malformed(m)) => assert!(m.contains("TLS directory outside"), "{m}"),
        other => panic!("expected a TLS directory outside the sections, got {other:?}"),
    }
    // One whole callback slot past the directory is enough to read it.
    let raw = TLS_CALLBACKS - SECTION_RVA + 8;
    let tls = ImageInfo::parse(&with_raw_size(raw)).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, [0x1010]);
}

/// The second section, `.data`: RVA 0x2000, file offset 0x600, 0x200 bytes.
/// Its file offset is not the `.text` shift (RVA 0x2000 would be file offset
/// 0x1200 there), and the file holds 0xEE bytes at 0x1200, so a read that
/// maps every RVA through the first section finds garbage.
const DATA_RVA: u32 = 0x2000;
const DATA_FILE_OFFSET: u32 = SECTION_FILE_OFFSET + SECTION_SIZE;
const DATA_SIZE: u32 = 0x200;

/// File offset of an RVA inside `.data`.
fn at_data(rva: u32) -> usize {
    (rva - DATA_RVA + DATA_FILE_OFFSET) as usize
}

fn put_data_bytes(d: &mut [u8], rva: u32, bytes: &[u8]) {
    d[at_data(rva)..at_data(rva) + bytes.len()].copy_from_slice(bytes);
}

/// `image()` with `.data` added after `.text` in the section table, or before
/// it when `data_first`.
fn with_data_section(data_first: bool) -> Vec<u8> {
    let mut d = image();
    let wrong = at(DATA_RVA);
    d.resize(DATA_FILE_OFFSET as usize + DATA_SIZE as usize, 0);
    d.resize(wrong, 0xEE);
    d.resize(wrong + DATA_SIZE as usize, 0xEE);
    put16(&mut d, 0x46, 2);
    put32(&mut d, OPT + 56, DATA_RVA + 0x1000);
    let text = OPT + 240;
    let data = text + 40;
    d.copy_within(text..text + 40, data);
    let (text, data) = if data_first {
        (data, text)
    } else {
        (text, data)
    };
    d[data..data + 40].fill(0);
    d[data..data + 5].copy_from_slice(b".data");
    put32(&mut d, data + 8, DATA_SIZE);
    put32(&mut d, data + 12, DATA_RVA);
    put32(&mut d, data + 16, DATA_SIZE);
    put32(&mut d, data + 20, DATA_FILE_OFFSET);
    put32(&mut d, data + 36, 0xC000_0040); // initialised data, read, write
    assert_eq!(&d[text..text + 5], b".text");
    d
}

#[test]
fn imports_in_a_second_section_are_read_at_its_file_offset() {
    // Two descriptors in `.data`, with their names and name tables there
    // too. The second one's IAT is back in `.text`: nothing reads the IAT, so
    // it may be in any section.
    const DESCRIPTORS: u32 = DATA_RVA;
    const NAMES: u32 = DATA_RVA + 0x80;
    const SLOTS: u32 = DATA_RVA + 0xA0;
    const DLL: u32 = DATA_RVA + 0xC0;
    const FUNC: u32 = DATA_RVA + 0xD0;
    const NAMES2: u32 = DATA_RVA + 0x100;
    const DLL2: u32 = DATA_RVA + 0x120;
    let mut d = with_data_section(false);
    set_directory(&mut d, dir::IMPORT, DESCRIPTORS, 60);
    let desc = at_data(DESCRIPTORS);
    put32(&mut d, desc, NAMES);
    put32(&mut d, desc + 12, DLL);
    put32(&mut d, desc + 16, SLOTS);
    put32(&mut d, desc + 20, NAMES2);
    put32(&mut d, desc + 20 + 12, DLL2);
    put32(&mut d, desc + 20 + 16, IAT);
    put64(&mut d, at_data(NAMES), u64::from(FUNC));
    put64(&mut d, at_data(NAMES) + 8, (1 << 63) | 9);
    put64(&mut d, at_data(NAMES2), (1 << 63) | 5);
    put_data_bytes(&mut d, DLL, b"USER32.dll\0");
    put_data_bytes(&mut d, DLL2, b"KERNEL32.dll\0");
    put16(&mut d, at_data(FUNC), 0x2B);
    put_data_bytes(&mut d, FUNC + 2, b"MessageBoxW\0");

    let info = ImageInfo::parse(&d).unwrap();
    let modules: Vec<_> = info
        .imports
        .iter()
        .map(|m| {
            let entries: Vec<_> = m
                .entries
                .iter()
                .map(|e| (e.symbol.to_string(), e.iat_slot_rva))
                .collect();
            (m.dll.as_str(), entries)
        })
        .collect();
    assert_eq!(
        modules,
        [
            (
                "USER32.dll",
                vec![("MessageBoxW".into(), SLOTS), ("#9".into(), SLOTS + 8)]
            ),
            ("KERNEL32.dll", vec![("#5".into(), IAT)]),
        ]
    );
    assert_eq!(
        info.imports[0].entries[0].symbol,
        ImportSymbol::Name {
            hint: 0x2B,
            name: "MessageBoxW".into()
        }
    );
    assert_eq!(info.import_count(), 3);
    assert_eq!(info.rva_to_file_offset(FUNC), Some(at_data(FUNC) as u32));

    // Names and name tables are read only from the section that holds the
    // import directory: pointing either one back into `.text`, where the
    // same strings and table are, makes the image malformed.
    for (field, rva) in [(12, DLL_NAME), (0, NAME_TABLE)] {
        let mut d = d.clone();
        put32(&mut d, desc + 20 + field, rva);
        assert!(
            matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))),
            "field {field}"
        );
    }
    // So is a hint/name entry in `.text`.
    put64(&mut d, at_data(NAMES), u64::from(HINT_NAME));
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn delay_imports_in_a_second_section_are_read_at_its_file_offset() {
    // The delay descriptor, its name table, DLL name and hint/name entry are
    // all in `.data`; its IAT is in `.text`.
    const DESCRIPTORS: u32 = DATA_RVA + 0x100;
    const NAMES: u32 = DATA_RVA + 0x140;
    const DLL: u32 = DATA_RVA + 0x160;
    const FUNC: u32 = DATA_RVA + 0x170;
    let mut d = with_data_section(false);
    set_directory(&mut d, DIR_DELAY_IMPORT, DESCRIPTORS, 64);
    let desc = at_data(DESCRIPTORS);
    put32(&mut d, desc, 1); // attributes: RVA based
    put32(&mut d, desc + 4, DLL);
    put32(&mut d, desc + 12, DELAY_IAT);
    put32(&mut d, desc + 16, NAMES);
    put64(&mut d, at_data(NAMES), u64::from(FUNC));
    put64(&mut d, at_data(NAMES) + 8, (1 << 63) | 0x1234);
    put_data_bytes(&mut d, DLL, b"USER32.dll\0");
    put16(&mut d, at_data(FUNC), 7);
    put_data_bytes(&mut d, FUNC + 2, b"MessageBoxW\0");

    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.delay_imports.len(), 1);
    let user32 = &info.delay_imports[0];
    assert_eq!(user32.dll, "USER32.dll");
    let entries: Vec<_> = user32
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
                DELAY_IAT
            ),
            (ImportSymbol::Ordinal(0x1234), DELAY_IAT + 8),
        ]
    );
    // The direct imports in `.text` are unchanged.
    assert_eq!(info.import_count(), 2);

    // As with direct imports, the DLL name must be in the descriptor's
    // section: the same string in `.text` is not read.
    add_delay_import(&mut d);
    set_directory(&mut d, DIR_DELAY_IMPORT, DESCRIPTORS, 64);
    put32(&mut d, desc + 4, DELAY_DLL_NAME);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn a_tls_callback_array_in_another_section_is_read_there() {
    // The directory stays in `.text`; its callbacks pointer names an array
    // in `.data`, which ends at the first null slot.
    const CALLBACKS: u32 = DATA_RVA + 0x1F0;
    let mut d = with_data_section(false);
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(CALLBACKS),
    );
    put64(&mut d, at_data(CALLBACKS), IMAGE_BASE + 0x1030);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.directory_rva, TLS_DIRECTORY);
    assert_eq!(tls.callbacks_va, IMAGE_BASE + u64::from(CALLBACKS));
    // Two slots fit before the end of `.data`, and the second is zero.
    assert_eq!(tls.callback_rvas, [0x1030]);

    // With the second slot set too, both are read. The file holds 0xEE bytes
    // right after `.data`'s raw data, and they are not read as a third.
    put64(&mut d, at_data(CALLBACKS) + 8, IMAGE_BASE + 0x1040);
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.callback_rvas, [0x1030, 0x1040]);
}

#[test]
fn a_tls_directory_in_another_section_than_its_callbacks_is_read_there() {
    // The reverse: the directory moves into `.data`, the array stays in
    // `.text`.
    const DIRECTORY: u32 = DATA_RVA + 0x40;
    let mut d = with_data_section(false);
    let old = at(TLS_DIRECTORY);
    d[old..old + 40].fill(0);
    set_directory(&mut d, DIR_TLS, DIRECTORY, 40);
    put64(
        &mut d,
        at_data(DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(TLS_CALLBACKS),
    );
    let tls = ImageInfo::parse(&d).unwrap().tls.unwrap();
    assert_eq!(tls.directory_rva, DIRECTORY);
    assert_eq!(tls.callback_rvas, [0x10, 0x20].map(|o| SECTION_RVA + o));
}

#[test]
fn the_section_table_order_does_not_change_what_is_read() {
    // `.data` listed before `.text`: sections are reported in table order,
    // and imports and TLS that cross between them are read the same way.
    const CALLBACKS: u32 = DATA_RVA + 0x10;
    let parse = |data_first: bool| {
        let mut d = with_data_section(data_first);
        put64(
            &mut d,
            at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
            IMAGE_BASE + u64::from(CALLBACKS),
        );
        put64(&mut d, at_data(CALLBACKS), IMAGE_BASE + 0x1010);
        put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, DATA_RVA + 0x100);
        ImageInfo::parse(&d).unwrap()
    };
    let (after, before) = (parse(false), parse(true));
    let names = |info: &ImageInfo| -> Vec<String> {
        info.sections
            .iter()
            .map(|s| s.name.trim_end_matches('\0').to_owned())
            .collect()
    };
    assert_eq!(names(&after), [".text", ".data"]);
    assert_eq!(names(&before), [".data", ".text"]);
    for info in [&after, &before] {
        assert_eq!(info.tls.as_ref().unwrap().callback_rvas, [0x1010]);
        let slots: Vec<u32> = info.imports[0]
            .entries
            .iter()
            .map(|e| e.iat_slot_rva)
            .collect();
        assert_eq!(slots, [DATA_RVA + 0x100, DATA_RVA + 0x108]);
        for rva in [ENTRY, HINT_NAME, DATA_RVA, CALLBACKS] {
            let expected = if rva < DATA_RVA {
                at(rva)
            } else {
                at_data(rva)
            };
            assert_eq!(info.rva_to_file_offset(rva), Some(expected as u32));
        }
        // Between `.text` and `.data`, and past `.data`, nothing is mapped.
        assert_eq!(info.rva_to_file_offset(SECTION_RVA + SECTION_SIZE), None);
        assert_eq!(info.rva_to_file_offset(DATA_RVA + DATA_SIZE), None);
    }
}

#[test]
fn exports_in_a_second_section_are_read_at_its_file_offset() {
    // The export directory and every table it points at are in `.data`;
    // the exported address is the entry point in `.text`, and a forwarder
    // string inside the directory's range is in `.data` too.
    const DIRECTORY: u32 = DATA_RVA + 0x20;
    const ADDRESSES: u32 = DATA_RVA + 0x50;
    const NAMES: u32 = DATA_RVA + 0x58;
    const ORDINALS: u32 = DATA_RVA + 0x60;
    const LOCAL: u32 = DATA_RVA + 0x68;
    const FWD: u32 = DATA_RVA + 0x70;
    const TO: u32 = DATA_RVA + 0x78;
    const END: u32 = DATA_RVA + 0x100;
    let mut d = with_data_section(false);
    set_directory(&mut d, DIR_EXPORT, DIRECTORY, END - DIRECTORY);
    let ed = at_data(DIRECTORY);
    put32(&mut d, ed + 16, 4); // ordinal base
    put32(&mut d, ed + 20, 2); // functions
    put32(&mut d, ed + 24, 2); // names
    put32(&mut d, ed + 28, ADDRESSES);
    put32(&mut d, ed + 32, NAMES);
    put32(&mut d, ed + 36, ORDINALS);
    put32(&mut d, at_data(ADDRESSES), ENTRY);
    put32(&mut d, at_data(ADDRESSES) + 4, TO);
    // Sorted names: "fwd" (index 1) before "local" (index 0).
    put32(&mut d, at_data(NAMES), FWD);
    put32(&mut d, at_data(NAMES) + 4, LOCAL);
    put16(&mut d, at_data(ORDINALS), 1);
    put16(&mut d, at_data(ORDINALS) + 2, 0);
    put_data_bytes(&mut d, LOCAL, b"local\0");
    put_data_bytes(&mut d, FWD, b"fwd\0");
    put_data_bytes(&mut d, TO, b"NTDLL.RtlBar\0");

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
            (4, Some("local"), Some(ENTRY), None),
            (5, Some("fwd"), None, Some("NTDLL.RtlBar")),
        ]
    );
    // The `.text` imports and TLS are read as before.
    assert_eq!(info.import_count(), 2);
    assert_eq!(info.tls.unwrap().callback_rvas, [0x1010, 0x1020]);

    // An address in `.text` is not a forwarder, even when it holds a string
    // that looks like one: only the directory's own range is.
    let mut text_fwd = d.clone();
    put_bytes(&mut text_fwd, 0x1080, b"NTDLL.RtlBar\0");
    put32(&mut text_fwd, at_data(ADDRESSES) + 4, 0x1080);
    let info = ImageInfo::parse(&text_fwd).unwrap();
    let fwd = info.exports.iter().find(|e| e.ordinal == 5).unwrap();
    assert_eq!((fwd.rva, fwd.forward.as_deref()), (Some(0x1080), None));

    // Names are read only from the directory's section: the same name in
    // `.text` makes the image malformed.
    put_bytes(&mut d, 0x1080, b"local\0");
    put32(&mut d, at_data(NAMES) + 4, 0x1080);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn a_debug_directory_in_another_section_than_its_record_is_read_there() {
    // The debug directory entry is in `.data`; its CodeView record stays in
    // `.text`, found through its file offset.
    const DEBUG: u32 = DATA_RVA + 0x180;
    let mut d = with_data_section(false);
    add_codeview(&mut d, b"RSDS");
    let old = at(DEBUG_DIRECTORY);
    d.copy_within(old..old + 28, at_data(DEBUG));
    d[old..old + 28].fill(0);
    set_directory(&mut d, DIR_DEBUG, DEBUG, 28);
    let cv = ImageInfo::parse(&d).unwrap().codeview.unwrap();
    assert_eq!((cv.pdb_path.as_str(), cv.age), ("t.pdb", 0x2A));

    // The record moved into `.data` is read there through its new file
    // offset, whatever RVA the entry gives.
    let record = at(CODEVIEW);
    d.copy_within(record..record + 30, at_data(DATA_RVA + 0x1C0));
    d[record..record + 30].fill(0);
    put32(
        &mut d,
        at_data(DEBUG) + 24,
        at_data(DATA_RVA + 0x1C0) as u32,
    );
    let cv = ImageInfo::parse(&d).unwrap().codeview.unwrap();
    assert_eq!((cv.pdb_path.as_str(), cv.age), ("t.pdb", 0x2A));
    assert_eq!(cv.symbol_key(), "030201000504070608090A0B0C0D0E0F2A");
}

#[test]
fn an_unsorted_export_name_table_still_names_every_export() {
    // The Windows loader binary-searches the name pointer table, but reading
    // the exports walks it in order: names out of order are all reported,
    // each against the ordinal its own name-ordinal entry gives.
    let mut d = image();
    add_exports(&mut d);
    let (names, ordinals) = (0x1384u32, 0x1390u32);
    let (local, fwdname, fwdord) = (0x13A0u32, 0x13A8u32, 0x13B0u32);
    for (i, (name, index)) in [(local, 0u16), (fwdord, 2), (fwdname, 1)]
        .into_iter()
        .enumerate()
    {
        put32(&mut d, at(names) + i * 4, name);
        put16(&mut d, at(ordinals) + i * 2, index);
    }
    let mut exports: Vec<_> = ImageInfo::parse(&d)
        .unwrap()
        .exports
        .into_iter()
        .map(|e| (e.ordinal, e.name, e.rva, e.forward))
        .collect();
    exports.sort();
    assert_eq!(
        exports,
        [
            (1, Some("local".into()), Some(ENTRY), None),
            (2, Some("fwdname".into()), None, Some("NTDLL.RtlFoo".into())),
            (3, Some("fwdord".into()), None, Some("NTDLL.#7".into())),
        ]
    );
}

#[test]
fn a_debug_directory_with_an_rva_but_no_size_has_no_codeview() {
    let mut d = image();
    add_codeview(&mut d, b"RSDS");
    set_directory(&mut d, DIR_DEBUG, DEBUG_DIRECTORY, 0);
    assert!(ImageInfo::parse(&d).unwrap().codeview.is_none());

    // With no RVA the entry is absent whatever its size says.
    set_directory(&mut d, DIR_DEBUG, 0, 28);
    assert!(ImageInfo::parse(&d).unwrap().codeview.is_none());
}

#[test]
fn an_import_with_an_empty_name_is_malformed() {
    // A hint/name entry whose name is just its NUL has no name to bind by,
    // so the image is rejected rather than importing "".
    let mut d = image();
    put_bytes(&mut d, HINT_NAME + 2, b"\0");
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    let mut d = image();
    add_delay_import(&mut d);
    put_bytes(&mut d, DELAY_HINT_NAME + 2, b"\0");
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn an_unaligned_iat_is_taken_as_written() {
    // Slot RVAs are not rounded: the loader gets the bytes the name table
    // pairs with, eight apart from wherever the IAT starts.
    for iat in [IAT + 1, IAT + 4, IAT + 7] {
        let mut d = image();
        put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, iat);
        let info = ImageInfo::parse(&d).unwrap();
        let slots: Vec<_> = info.imports[0]
            .entries
            .iter()
            .map(|e| e.iat_slot_rva)
            .collect();
        assert_eq!(slots, [iat, iat + 8], "IAT at {iat:#x}");
    }
}

#[test]
fn a_tls_callback_array_in_the_headers_is_outside_the_sections() {
    // The headers are mapped but belong to no section, so a callback array
    // pointed at them cannot be read.
    let mut d = image();
    put64(
        &mut d,
        at(TLS_DIRECTORY) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + 0x100,
    );
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    // The same holds for the TLS directory itself.
    let mut d = image();
    set_directory(&mut d, DIR_TLS, 0x100, 40);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
}

#[test]
fn an_iat_whose_last_slot_crosses_the_size_of_image_rejects_the_whole_image() {
    // The first slot fits; the second ends four bytes past SizeOfImage.
    // Nothing is returned for the module, not even the slot that fit.
    let mut d = image();
    let iat = SIZE_OF_IMAGE - 12;
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, iat);
    let err = ImageInfo::parse(&d).unwrap_err();
    assert!(
        matches!(&err, PeError::Malformed(m) if m.contains(&format!("{iat:#x}"))),
        "{err:?}"
    );
}

#[test]
fn a_module_without_entries_never_has_its_iat_checked() {
    // The IAT is only checked slot by slot, so a module whose name table is
    // empty keeps any IAT RVA, even one far outside the image.
    let mut d = image();
    d[at(NAME_TABLE)..at(NAME_TABLE) + 16].fill(0);
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 16, 0xFFFF_FFF8);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.imports[0].dll, "KERNEL32.dll");
    assert!(info.imports[0].entries.is_empty());

    let mut d = image();
    add_delay_import(&mut d);
    put64(&mut d, at(DELAY_NAME_TABLE), 0);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 12, 0xFFFF_FFF8);
    let info = ImageInfo::parse(&d).unwrap();
    assert_eq!(info.delay_imports[0].dll, "USER32.dll");
    assert!(info.delay_imports[0].entries.is_empty());
}

#[test]
fn names_that_run_to_the_end_of_the_section_without_a_nul_are_malformed() {
    // The last bytes of the section spell a name with no terminator after
    // them; the read stops at the section's end instead of running into the
    // file bytes that follow.
    let end = SECTION_RVA + SECTION_SIZE;
    let tail = end - 4;
    let unterminated = |d: &mut Vec<u8>| {
        put_bytes(d, tail, b"ABCD");
        d.extend_from_slice(b"EF\0");
    };

    let mut d = image();
    unterminated(&mut d);
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 12, tail);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    let mut d = image();
    unterminated(&mut d);
    put64(&mut d, at(NAME_TABLE), u64::from(tail - 2));
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    let mut d = image();
    add_delay_import(&mut d);
    unterminated(&mut d);
    put32(&mut d, at(DELAY_DESCRIPTORS) + 4, tail);
    assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));

    // The same name with its NUL as the section's last byte is read.
    let mut d = image();
    put_bytes(&mut d, end - 4, b"ABC\0");
    put32(&mut d, at(IMPORT_DESCRIPTORS) + 12, end - 4);
    assert_eq!(ImageInfo::parse(&d).unwrap().imports[0].dll, "ABC");
}
