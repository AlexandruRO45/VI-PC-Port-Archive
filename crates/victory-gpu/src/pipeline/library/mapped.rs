//! File-backed pipeline-library storage. Never rewrite a published file in
//! place: the driver can fault in its original blob long after `open`.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{CloseHandle, ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION};
use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_DELETE, FILE_SHARE_READ};
use windows_sys::Win32::System::Memory::*;

pub(super) struct View {
    address: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    // Keep the original file object alive even after its name was replaced.
    file: File,
}

// SAFETY: the view and file are owned by `View`; access and destruction
// follow the owning library's mutex. Mutable views never leave their writer.
unsafe impl Send for View {}

impl View {
    fn map(file: File, writable: bool) -> io::Result<Self> {
        let len = usize::try_from(file.metadata()?.len())
            .ok()
            .filter(|&n| n > 0 && n <= isize::MAX as usize)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid cache blob length")
            })?;
        // SAFETY: a live file; default security and explicit view length.
        let address = unsafe {
            let protection = if writable {
                PAGE_READWRITE
            } else {
                PAGE_READONLY
            };
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                protection,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return Err(io::Error::last_os_error());
            }
            let access = if writable {
                FILE_MAP_WRITE
            } else {
                FILE_MAP_READ
            };
            let address = MapViewOfFile(mapping, access, 0, 0, len);
            let error = address.Value.is_null().then(io::Error::last_os_error);
            CloseHandle(mapping);
            if let Some(error) = error {
                return Err(error);
            }
            address
        };
        Ok(Self { address, len, file })
    }

    pub(super) fn open(path: &Path) -> io::Result<Self> {
        // FILE_SHARE_READ | FILE_SHARE_DELETE: swap the name by rename, but
        // do not allow concurrent writes to the generation the driver reads.
        let file = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
            .open(path)?;
        Self::map(file, false)
    }

    /// Prefer the sidecar if publication could not replace the old name.
    pub(super) fn open_latest(path: &Path) -> io::Result<Self> {
        let next = next_path(path);
        // Promote a pending generation before mapping it, so a later save
        // can fall back to an unused sidecar name again on older Windows.
        if std::fs::rename(&next, path).is_ok() {
            return Self::open(path);
        }
        match Self::open(&next) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => Self::open(path),
            result => result,
        }
    }

    pub(super) fn bytes(&self) -> &[u8] {
        // SAFETY: a readable mapped view of this exact length.
        unsafe { std::slice::from_raw_parts(self.address.Value.cast(), self.len) }
    }

    /// Suggest dropping only these file-backed pages from the working set.
    /// VirtualUnlock removes pages from the working set even when it reports
    /// ERROR_NOT_LOCKED. It does not unmap or invalidate the file contents.
    pub(super) fn trim(&self) -> io::Result<()> {
        // SAFETY: a live mapping owned by this library.
        if unsafe { VirtualUnlock(self.address.Value, self.len) } != 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_NOT_LOCKED as i32) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: our view. Keep `file` open until this call has returned.
        unsafe { UnmapViewOfFile(self.address) };
    }
}

fn next_path(path: &Path) -> PathBuf {
    path.with_extension("next.bin")
}

struct Temp(PathBuf);

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(super) fn publish(
    destination: &Path,
    len: usize,
    serialize: impl FnOnce(&mut [u8]) -> io::Result<()>,
) -> io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (file, temp) = loop {
        let path = destination.with_extension(format!(
            "{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => break (file, Temp(path)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    file.set_len(len as u64)?;
    let writable = View::map(file, true)?;
    // SAFETY: a fresh, writable mapping of a private temporary file with no
    // aliases exposed to other code. The slice does not escape `serialize`.
    unsafe {
        serialize(std::slice::from_raw_parts_mut(
            writable.address.Value.cast(),
            len,
        ))?
    };
    // SAFETY: the view is still live; flush its bytes before publication.
    if unsafe { FlushViewOfFile(writable.address.Value, len) } == 0 {
        return Err(io::Error::last_os_error());
    }
    writable.file.sync_all()?;
    drop(writable);
    // Same directory and volume. Once a sidecar exists it takes priority;
    // update it rather than publishing behind it to the original name.
    let next = next_path(destination);
    if next.try_exists()? {
        return std::fs::rename(&temp.0, next);
    }
    match std::fs::rename(&temp.0, destination) {
        Err(e)
            if e.raw_os_error().is_some_and(|code| {
                code == ERROR_SHARING_VIOLATION as i32 || code == ERROR_ACCESS_DENIED as i32
            }) =>
        {
            std::fs::rename(&temp.0, next)
        }
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident_bytes(view: &View) -> usize {
        use windows_sys::Win32::System::ProcessStatus::{
            K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
        };
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut info = SYSTEM_INFO::default();
        // SAFETY: valid writable output for the system settings.
        unsafe { GetSystemInfo(&mut info) };
        let page = info.dwPageSize as usize;
        let mut pages = (0..view.len)
            .step_by(page)
            .map(|at| PSAPI_WORKING_SET_EX_INFORMATION {
                VirtualAddress: view.address.Value.wrapping_byte_add(at),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        // SAFETY: the byte count covers the array, and the query concerns
        // the live mapping in our own process. Bit zero is PSAPI's Valid bit.
        unsafe {
            assert_ne!(
                K32QueryWorkingSetEx(
                    GetCurrentProcess(),
                    pages.as_mut_ptr().cast(),
                    std::mem::size_of_val(&pages[..]) as u32
                ),
                0
            );
            pages
                .iter()
                .filter(|p| p.VirtualAttributes.Flags & 1 != 0)
                .count()
                * page
        }
    }

    #[test]
    fn replacement_uses_sidecar_when_the_original_disallows_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library.bin");
        std::fs::write(&path, [1]).unwrap();
        let locked = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&path)
            .unwrap();
        publish(&path, 1, |buf| {
            buf[0] = 2;
            Ok(())
        })
        .unwrap();
        assert_eq!(View::open_latest(&path).unwrap().bytes(), &[2]);
        drop(locked);
        let promoted = View::open_latest(&path).unwrap();
        assert_eq!(promoted.bytes(), &[2]);
        assert!(!next_path(&path).exists());
        drop(promoted);
        publish(&path, 1, |buf| {
            buf[0] = 3;
            Ok(())
        })
        .unwrap();
        assert_eq!(View::open_latest(&path).unwrap().bytes(), &[3]);
        assert_eq!(View::open(&path).unwrap().bytes(), &[3]);
        assert!(!next_path(&path).exists());
    }

    #[test]
    fn mapped_generation_survives_publication_and_trim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library.bin");
        let len = 16 << 20;
        publish(&path, len, |out| {
            out.fill(0xa5);
            Ok(())
        })
        .unwrap();
        let old = View::open(&path).unwrap();
        assert!(OpenOptions::new().write(true).open(&path).is_err());
        let rejected = publish(&path, len, |out| {
            out[0] = 0xff;
            Err(io::Error::other("synthetic serialization failure"))
        });
        assert!(rejected.is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        assert_eq!(View::open(&path).unwrap().bytes()[0], 0xa5);
        publish(&path, len, |out| {
            out.fill(0x5a);
            Ok(())
        })
        .unwrap();
        let new = View::open_latest(&path).unwrap();
        assert!(new.bytes().iter().step_by(4096).all(|&x| x == 0x5a));
        assert!(old.bytes().iter().step_by(4096).all(|&x| x == 0xa5));
        let before_trim = resident_bytes(&old);
        old.trim().unwrap();
        let after_trim = resident_bytes(&old);
        eprintln!("mapped working set: {before_trim} -> {after_trim} bytes on trim");
        assert!(old.bytes().iter().step_by(4096).all(|&x| x == 0xa5));
        assert_eq!(old.bytes()[len - 1], 0xa5);
        eprintln!(
            "same {len} bytes per blob: previous read/serialize used {len} private Rust-heap bytes each; mapped path uses 0 payload bytes in Rust heap (not a process-wide commit measurement)"
        );
    }
}
