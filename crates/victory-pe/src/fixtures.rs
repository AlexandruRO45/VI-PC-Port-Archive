//! A hand-built PE32+ image for tests.
//!
//! The tests used to parse the test binary itself
//! (`std::env::current_exe()`), which is a real MSVC-linked PE only when
//! the tests are built on Windows. Off Windows that file is an ELF and
//! every such test fails, so none of them could run on another host.
//!
//! Building the image here also pins down what the tests assert — the
//! import count, the TLS directory, the entry point — instead of trusting
//! whatever the linker of the day emitted.

/// RVA of `.text`.
const TEXT_RVA: u32 = 0x1000;
/// RVA of `.rdata`, and its file offset.
const RDATA_RVA: u32 = 0x2000;
const RDATA_OFF: u32 = 0x600;
/// Where the headers end and `.text` begins in the file.
const TEXT_OFF: u32 = 0x400;
const IMAGE_BASE: u64 = 0x1_4000_0000;
/// `IMAGE_DIRECTORY_ENTRY_TLS`.
const DIR_TLS: usize = 9;
const DIR_IMPORT: usize = 1;
const DIR_EXCEPTION: usize = 3;
/// Total file size: headers, `.text` and `.rdata`.
const FILE_SIZE: usize = 0x800;

fn put16(data: &mut [u8], at: usize, value: u16) {
    data[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn put32(data: &mut [u8], at: usize, value: u32) {
    data[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put64(data: &mut [u8], at: usize, value: u64) {
    data[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// A PE32+ executable: two sections, one KERNEL32 import, a TLS directory
/// whose callback array is empty (as a statically linked image has), and an
/// exception table, so `has_exception_table` holds as it does for any x64
/// image.
pub fn executable() -> Vec<u8> {
    let mut data = vec![0u8; FILE_SIZE];

    // DOS header: the signature and the offset of the NT headers.
    data[0] = b'M';
    data[1] = b'Z';
    let e_lfanew = 0x80usize;
    put32(&mut data, 0x3C, e_lfanew as u32);

    // NT signature, then the COFF file header.
    data[e_lfanew..e_lfanew + 4].copy_from_slice(b"PE\0\0");
    let fh = e_lfanew + 4;
    put16(&mut data, fh, 0x8664); // Machine: AMD64
    put16(&mut data, fh + 2, 2); // NumberOfSections
    put16(&mut data, fh + 16, 240); // SizeOfOptionalHeader (PE32+, 16 dirs)
    put16(&mut data, fh + 18, 0x0022); // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE

    // Optional header.
    let opt = fh + 20;
    put16(&mut data, opt, 0x20B); // PE32+
    put32(&mut data, opt + 4, 0x200); // SizeOfCode
    put32(&mut data, opt + 8, 0x200); // SizeOfInitializedData
    put32(&mut data, opt + 16, TEXT_RVA); // AddressOfEntryPoint
    put32(&mut data, opt + 20, TEXT_RVA); // BaseOfCode
    put64(&mut data, opt + 24, IMAGE_BASE);
    put32(&mut data, opt + 32, 0x1000); // SectionAlignment
    put32(&mut data, opt + 36, 0x200); // FileAlignment
    put16(&mut data, opt + 48, 6); // MajorSubsystemVersion
    put32(&mut data, opt + 56, 0x3000); // SizeOfImage
    put32(&mut data, opt + 60, TEXT_OFF); // SizeOfHeaders
    put16(&mut data, opt + 68, 3); // Subsystem: WINDOWS_CUI
    put64(&mut data, opt + 72, 0x10_0000); // SizeOfStackReserve
    put64(&mut data, opt + 80, 0x1000); // SizeOfStackCommit
    put64(&mut data, opt + 88, 0x10_0000); // SizeOfHeapReserve
    put64(&mut data, opt + 96, 0x1000); // SizeOfHeapCommit
    put32(&mut data, opt + 108, 16); // NumberOfRvaAndSizes

    // `.rdata` holds the import tables, the TLS directory and a runtime
    // function, at these RVAs.
    let import_rva = RDATA_RVA;
    let ilt_rva = RDATA_RVA + 0x28;
    let iat_rva = RDATA_RVA + 0x38;
    let dll_name_rva = RDATA_RVA + 0x48;
    let hint_rva = RDATA_RVA + 0x60;
    let tls_rva = RDATA_RVA + 0x68;
    let callbacks_rva = RDATA_RVA + 0x90;
    let pdata_rva = RDATA_RVA + 0x98;

    // Data directories.
    let dd = opt + 112;
    put32(&mut data, dd + DIR_IMPORT * 8, import_rva);
    put32(&mut data, dd + DIR_IMPORT * 8 + 4, 40); // two descriptors
    put32(&mut data, dd + DIR_EXCEPTION * 8, pdata_rva);
    put32(&mut data, dd + DIR_EXCEPTION * 8 + 4, 12); // one RUNTIME_FUNCTION
    put32(&mut data, dd + DIR_TLS * 8, tls_rva);
    put32(&mut data, dd + DIR_TLS * 8 + 4, 40); // IMAGE_TLS_DIRECTORY64

    // Section table.
    let sections = opt + 240;
    let text = sections;
    data[text..text + 5].copy_from_slice(b".text");
    put32(&mut data, text + 8, 0x200); // VirtualSize
    put32(&mut data, text + 12, TEXT_RVA); // VirtualAddress
    put32(&mut data, text + 16, 0x200); // SizeOfRawData
    put32(&mut data, text + 20, TEXT_OFF); // PointerToRawData
    put32(&mut data, text + 36, 0x6000_0020); // CODE | EXECUTE | READ

    let rdata = sections + 40;
    data[rdata..rdata + 6].copy_from_slice(b".rdata");
    put32(&mut data, rdata + 8, 0x200);
    put32(&mut data, rdata + 12, RDATA_RVA);
    put32(&mut data, rdata + 16, 0x200);
    put32(&mut data, rdata + 20, RDATA_OFF);
    put32(&mut data, rdata + 36, 0x4000_0040); // INITIALIZED_DATA | READ

    // A `ret` stands in for the entry point.
    data[TEXT_OFF as usize] = 0xC3;

    // Import descriptor, then the all-zero one that ends the table.
    let at = |rva: u32| (RDATA_OFF + (rva - RDATA_RVA)) as usize;
    let desc = at(import_rva);
    put32(&mut data, desc, ilt_rva); // OriginalFirstThunk
    put32(&mut data, desc + 12, dll_name_rva); // Name
    put32(&mut data, desc + 16, iat_rva); // FirstThunk

    // One name in both the lookup table and the address table, each ended
    // by a null entry.
    put64(&mut data, at(ilt_rva), u64::from(hint_rva));
    put64(&mut data, at(iat_rva), u64::from(hint_rva));

    data[at(dll_name_rva)..at(dll_name_rva) + 12].copy_from_slice(b"kernel32.dll");
    // Hint/name entry: a zero hint, then the name.
    data[at(hint_rva) + 2..at(hint_rva) + 7].copy_from_slice(b"Sleep");

    // IMAGE_TLS_DIRECTORY64. The callback array holds only its terminator,
    // which is what a statically linked image carries.
    let tls = at(tls_rva);
    put64(&mut data, tls, IMAGE_BASE + u64::from(pdata_rva));
    put64(&mut data, tls + 8, IMAGE_BASE + u64::from(pdata_rva));
    put64(&mut data, tls + 16, IMAGE_BASE + u64::from(pdata_rva));
    put64(&mut data, tls + 24, IMAGE_BASE + u64::from(callbacks_rva));

    // One RUNTIME_FUNCTION covering `.text`, so the exception table is real.
    let pdata = at(pdata_rva);
    put32(&mut data, pdata, TEXT_RVA);
    put32(&mut data, pdata + 4, TEXT_RVA + 1);

    data
}
