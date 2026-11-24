//! COFF import libraries (`.lib`).
//!
//! An umbrella import library tells us which DLL provides each API that a
//! console title may link against. That is the map the runtime needs to
//! decide which of our implementations answers which import.

use object::read::archive::ArchiveFile;
use object::read::coff::{ImportFile, ImportName, ImportType};

use crate::{ImportSymbol, PeError};

/// One `dll!export` pair described by an import library.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibImport {
    pub dll: String,
    pub export: ImportSymbol,
    pub kind: LibImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibImportKind {
    Code,
    Data,
    Const,
}

/// Result of reading an import library.
#[derive(Debug, Default)]
pub struct ImportLibrary {
    pub imports: Vec<LibImport>,
    /// Members that are not short-format import objects (regular objects,
    /// long-format import descriptors). Reported so nothing is silently lost.
    pub other_members: usize,
}

impl ImportLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let archive = ArchiveFile::parse(data)?;
        let mut lib = ImportLibrary::default();
        for member in archive.members() {
            let member = member?;
            let bytes = member.data(data)?;
            match ImportFile::parse(bytes) {
                Ok(import) => lib.imports.push(LibImport {
                    dll: String::from_utf8_lossy(import.dll()).into_owned(),
                    export: match import.import() {
                        ImportName::Name(name) => ImportSymbol::Name {
                            hint: 0,
                            name: String::from_utf8_lossy(name).into_owned(),
                        },
                        ImportName::Ordinal(o) => ImportSymbol::Ordinal(o),
                    },
                    kind: match import.import_type() {
                        ImportType::Code => LibImportKind::Code,
                        ImportType::Data => LibImportKind::Data,
                        ImportType::Const => LibImportKind::Const,
                    },
                }),
                Err(_) => lib.other_members += 1,
            }
        }
        lib.imports.sort();
        lib.imports.dedup();
        Ok(lib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short-format import object: `IMPORT_OBJECT_HEADER`, then the
    /// symbol and the DLL name, each NUL-terminated.
    fn short_import(
        dll: &str,
        symbol: &str,
        ordinal_or_hint: u16,
        ty: u16,
        name_ty: u16,
    ) -> Vec<u8> {
        let data_len = symbol.len() + 1 + dll.len() + 1;
        let mut out = Vec::new();
        out.extend_from_slice(&0u16.to_le_bytes()); // Sig1
        out.extend_from_slice(&0xFFFFu16.to_le_bytes()); // Sig2
        out.extend_from_slice(&0u16.to_le_bytes()); // Version
        out.extend_from_slice(&crate::MACHINE_AMD64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        out.extend_from_slice(&ordinal_or_hint.to_le_bytes());
        out.extend_from_slice(&(ty | name_ty << 2).to_le_bytes());
        for s in [symbol, dll] {
            out.extend_from_slice(s.as_bytes());
            out.push(0);
        }
        out
    }

    /// A `!<arch>` archive of `members`, each under a short GNU-style name.
    fn archive(members: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"!<arch>\n".to_vec();
        for (i, data) in members.iter().enumerate() {
            let header = format!(
                "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                format!("m{i}.obj/"),
                0,
                0,
                0,
                644,
                data.len()
            );
            assert_eq!(header.len(), 60);
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(data);
            if data.len() % 2 == 1 {
                out.push(b'\n');
            }
        }
        out
    }

    const CODE: u16 = 0;
    const DATA: u16 = 1;
    const CONST: u16 = 2;
    const BY_ORDINAL: u16 = 0;
    const BY_NAME: u16 = 1;
    const NO_PREFIX: u16 = 2;
    const UNDECORATE: u16 = 3;

    fn name(dll: &str, name: &str, kind: LibImportKind) -> LibImport {
        LibImport {
            dll: dll.into(),
            export: ImportSymbol::Name {
                hint: 0,
                name: name.into(),
            },
            kind,
        }
    }

    #[test]
    fn reads_short_imports_sorted_and_deduplicated() {
        let lib = ImportLibrary::parse(&archive(&[
            short_import("b.dll", "_Stdcall@8", 7, CODE, UNDECORATE),
            short_import("a.dll", "Plain", 3, CODE, BY_NAME),
            short_import("a.dll", "_Prefixed", 0, DATA, NO_PREFIX),
            short_import("a.dll", "Ignored", 42, CONST, BY_ORDINAL),
            // The same export again, under another hint: one entry.
            short_import("a.dll", "Plain", 9, CODE, BY_NAME),
            // Not a short import object: counted, not dropped silently.
            b"not an import".to_vec(),
        ]))
        .unwrap();

        assert_eq!(
            lib.imports,
            [
                name("a.dll", "Plain", LibImportKind::Code),
                name("a.dll", "Prefixed", LibImportKind::Data),
                LibImport {
                    dll: "a.dll".into(),
                    export: ImportSymbol::Ordinal(42),
                    kind: LibImportKind::Const,
                },
                name("b.dll", "Stdcall", LibImportKind::Code),
            ]
        );
        assert_eq!(lib.other_members, 1);
    }

    #[test]
    fn rejects_what_is_not_an_archive() {
        assert!(ImportLibrary::parse(b"MZ not an archive").is_err());
        assert!(
            ImportLibrary::parse(&archive(&[]))
                .unwrap()
                .imports
                .is_empty()
        );
    }
}
