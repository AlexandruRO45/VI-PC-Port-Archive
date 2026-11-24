//! Raw byte offsets of PE32+ header fields, for code that patches images.
//!
//! All offsets are relative to the start of the structure named in the
//! constant's prefix. They follow the published PE/COFF specification.

/// Offset of `e_lfanew` (u32) in the DOS header.
pub const DOS_E_LFANEW: usize = 0x3C;

/// Size of the `PE\0\0` signature that precedes the file header.
pub const NT_SIGNATURE_SIZE: usize = 4;
/// Size of `IMAGE_FILE_HEADER`.
pub const FILE_HEADER_SIZE: usize = 20;

/// `IMAGE_FILE_HEADER::Characteristics` (u16).
pub const FILE_HEADER_CHARACTERISTICS: usize = 18;

/// `IMAGE_OPTIONAL_HEADER64::AddressOfEntryPoint` (u32).
pub const OPT_ADDRESS_OF_ENTRY_POINT: usize = 16;
/// `IMAGE_OPTIONAL_HEADER64::SizeOfHeaders` (u32).
pub const OPT_SIZE_OF_HEADERS: usize = 60;
/// `IMAGE_OPTIONAL_HEADER64::CheckSum` (u32).
pub const OPT_CHECKSUM: usize = 64;
/// `IMAGE_OPTIONAL_HEADER64::DllCharacteristics` (u16).
pub const OPT_DLL_CHARACTERISTICS: usize = 70;
/// `IMAGE_OPTIONAL_HEADER64::NumberOfRvaAndSizes` (u32).
pub const OPT_NUMBER_OF_RVA_AND_SIZES: usize = 108;
/// First `IMAGE_DATA_DIRECTORY` entry (each is `{ rva: u32, size: u32 }`).
pub const OPT_DATA_DIRECTORY: usize = 112;
/// Size of one `IMAGE_DATA_DIRECTORY`.
pub const DATA_DIRECTORY_SIZE: usize = 8;

/// `IMAGE_TLS_DIRECTORY64::AddressOfCallBacks` (u64 VA).
pub const TLS64_ADDRESS_OF_CALLBACKS: usize = 24;

/// Data directory indices.
pub mod dir {
    pub const IMPORT: usize = 1;
    pub const EXCEPTION: usize = 3;
    pub const BASERELOC: usize = 5;
    pub const LOAD_CONFIG: usize = 10;
    pub const BOUND_IMPORT: usize = 11;
    pub const IAT: usize = 12;
}

/// `IMAGE_FILE_HEADER::Characteristics` flags.
pub mod file_flags {
    pub const RELOCS_STRIPPED: u16 = 0x0001;
    pub const DLL: u16 = 0x2000;
}

/// `IMAGE_OPTIONAL_HEADER64::DllCharacteristics` flags.
pub mod dll_flags {
    pub const HIGH_ENTROPY_VA: u16 = 0x0020;
    pub const DYNAMIC_BASE: u16 = 0x0040;
    pub const FORCE_INTEGRITY: u16 = 0x0080;
    pub const NX_COMPAT: u16 = 0x0100;
    pub const APPCONTAINER: u16 = 0x1000;
    pub const GUARD_CF: u16 = 0x4000;
}

/// `IMAGE_OPTIONAL_HEADER64::Subsystem` values.
pub mod subsystem {
    pub const WINDOWS_GUI: u16 = 2;
    pub const WINDOWS_CUI: u16 = 3;
    pub const XBOX: u16 = 14;
    pub const XBOX_CODE_CATALOG: u16 = 17;
}

/// Offset of the optional header from the start of the file.
pub fn optional_header_offset(data: &[u8]) -> Option<usize> {
    let lfanew = read_u32(data, DOS_E_LFANEW)? as usize;
    lfanew.checked_add(NT_SIGNATURE_SIZE + FILE_HEADER_SIZE)
}

/// Offset of the file header from the start of the file.
pub fn file_header_offset(data: &[u8]) -> Option<usize> {
    let lfanew = read_u32(data, DOS_E_LFANEW)? as usize;
    lfanew.checked_add(NT_SIGNATURE_SIZE)
}

/// Offset of data directory `index` from the start of the file.
pub fn data_directory_offset(data: &[u8], index: usize) -> Option<usize> {
    let opt = optional_header_offset(data)?;
    let count = read_u32(data, opt.checked_add(OPT_NUMBER_OF_RVA_AND_SIZES)?)? as usize;
    if index >= count {
        return None;
    }
    opt.checked_add(OPT_DATA_DIRECTORY)?
        .checked_add(index.checked_mul(DATA_DIRECTORY_SIZE)?)
}

pub fn read_u32(data: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    Some(u32::from_le_bytes(data.get(at..end)?.try_into().ok()?))
}

pub fn read_u64(data: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    Some(u64::from_le_bytes(data.get(at..end)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_reject_overflowing_offsets() {
        let data = [0u8; 16];
        assert_eq!(read_u32(&data, usize::MAX - 1), None);
        assert_eq!(read_u64(&data, usize::MAX - 2), None);
    }

    /// Just enough of a PE32+ header for the offset helpers: `e_lfanew` and
    /// `NumberOfRvaAndSizes`.
    fn headers(lfanew: u32, directories: u32) -> Vec<u8> {
        let mut data = vec![0u8; 0x200];
        data[DOS_E_LFANEW..DOS_E_LFANEW + 4].copy_from_slice(&lfanew.to_le_bytes());
        let count =
            lfanew as usize + NT_SIGNATURE_SIZE + FILE_HEADER_SIZE + OPT_NUMBER_OF_RVA_AND_SIZES;
        data[count..count + 4].copy_from_slice(&directories.to_le_bytes());
        data
    }

    #[test]
    fn header_offsets_follow_e_lfanew() {
        let data = headers(0x80, 16);
        assert_eq!(file_header_offset(&data), Some(0x84));
        assert_eq!(optional_header_offset(&data), Some(0x98));
        assert_eq!(data_directory_offset(&data, 0), Some(0x98 + 112));
        assert_eq!(data_directory_offset(&data, 15), Some(0x98 + 112 + 15 * 8));
    }

    #[test]
    fn offsets_need_the_bytes_they_are_read_from() {
        // No `e_lfanew`: no header offsets at all.
        assert_eq!(file_header_offset(&[0; DOS_E_LFANEW + 3]), None);
        assert_eq!(optional_header_offset(&[0; DOS_E_LFANEW + 3]), None);
        // `e_lfanew` is read, but `NumberOfRvaAndSizes` is past the end.
        let mut data = vec![0u8; DOS_E_LFANEW + 4];
        data[DOS_E_LFANEW..].copy_from_slice(&0x40u32.to_le_bytes());
        assert_eq!(file_header_offset(&data), Some(0x44));
        assert_eq!(data_directory_offset(&data, 0), None);
        // A huge `e_lfanew` is an offset past the file, never a panic.
        let data = {
            let mut d = vec![0u8; 0x40];
            d[DOS_E_LFANEW..DOS_E_LFANEW + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            d
        };
        assert_eq!(
            optional_header_offset(&data),
            Some(u32::MAX as usize + NT_SIGNATURE_SIZE + FILE_HEADER_SIZE)
        );
        assert_eq!(data_directory_offset(&data, 0), None);
    }

    #[test]
    fn directories_past_number_of_rva_and_sizes_are_absent() {
        let data = headers(0x40, 10);
        assert!(data_directory_offset(&data, 9).is_some());
        assert_eq!(data_directory_offset(&data, 10), None);
        assert_eq!(data_directory_offset(&data, dir::BOUND_IMPORT), None);
        assert_eq!(data_directory_offset(&headers(0x40, 0), 0), None);
        // A count far beyond the 16 defined slots is taken as written; the
        // offset it allows may lie past the file, for the read to refuse.
        let data = headers(0x40, u32::MAX);
        let far = data_directory_offset(&data, 100).unwrap();
        assert_eq!(far, 0x40 + 24 + OPT_DATA_DIRECTORY + 800);
        assert_eq!(read_u32(&data, far), None);
        assert_eq!(data_directory_offset(&data, u32::MAX as usize), None);
    }

    #[test]
    fn reads_are_little_endian_and_bounded() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(read_u32(&data, 0), Some(0x0403_0201));
        assert_eq!(read_u32(&data, 4), Some(0x0807_0605));
        assert_eq!(read_u32(&data, 5), None);
        assert_eq!(read_u64(&data, 0), Some(0x0807_0605_0403_0201));
        assert_eq!(read_u64(&data, 1), None);
    }
}
