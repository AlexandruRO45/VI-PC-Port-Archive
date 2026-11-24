//! One title session at a time, held by the operating system rather than by a
//! rule (CREW.md, "One GPU session at a time — never two").
//!
//! The title shares one GPU, one save folder and one shader cache, so two
//! sessions at once is the one failure this project cannot recover from: it
//! blacked the machine once already. A named mutex makes that a fact instead
//! of a hope — the second process finds the object taken and leaves before it
//! touches the GPU.
//!
//! The handle is created inheritable so every child this session starts
//! inherits it, which is what keeps the exclusion true when a child outlives
//! its parent: the object lives while any handle to it does, so an orphaned
//! child still blocks the next launch. Such a child finds
//! `VICTORY_SESSION_INHERITED` in its environment (set once, here, before any
//! thread of ours exists) and holds its inherited handle instead of refusing
//! itself.
//!
//! The name is `Global\` where the process may create a global object and
//! `Local\` otherwise, since `SeCreateGlobalPrivilege` is not granted to
//! ordinary interactive users; either way, one session per machine, or one per
//! session where that privilege is missing.

use anyhow::Result;

/// The session object's name, before its `Global\` or `Local\` scope.
const NAME: &str = "VIctory-title-session";

/// Set by the process that took the session; inherited by every child it
/// starts, which then hold the inherited object instead of taking it again.
pub(crate) const INHERITED: &str = "VICTORY_SESSION_INHERITED";

/// A session held for as long as this value lives: dropping it releases the
/// object, and so does the process ending by any means, including a panic or
/// Ctrl-C, because the system closes a process's handles.
pub(crate) struct Session {
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
    /// The mapping that publishes this session's pid, kept for as long as the
    /// session is held.
    #[cfg(windows)]
    owner: windows_sys::Win32::Foundation::HANDLE,
}

/// Takes the session, or fails saying who has it. Called once, before the
/// title runs and before any other thread of ours exists.
pub(crate) fn take() -> Result<Session> {
    // A child of a session: its inherited handle is the exclusion, and taking
    // the name again would refuse the child against its own parent.
    if std::env::var_os(INHERITED).is_some() {
        return Ok(Session {
            #[cfg(windows)]
            handle: std::ptr::null_mut(),
            #[cfg(windows)]
            owner: std::ptr::null_mut(),
        });
    }
    imp::take()
}

impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(windows)]
        for handle in [self.handle, self.owner] {
            if !handle.is_null() {
                // SAFETY: our own handles, closed once.
                unsafe { windows_sys::Win32::Foundation::CloseHandle(handle) };
            }
        }
    }
}

/// Set by whoever holds the session, in a mapping beside it, so the refusal
/// can name a pid instead of guessing one from a process list. Session-local:
/// a global mapping needs `SeCreateGlobalPrivilege`, which a global *mutex* may
/// be made without, and every process of a session can open this one.
const OWNER: &str = "Local\\VIctory-title-owner";

/// The refusal a second session prints. Pure, so the wording is testable
/// without a second process.
///
/// Only a `run` takes this object — translating a corpus, inspecting an
/// executable and every other subcommand are offline and never touch it — so
/// the holder is a title session by construction. It also cannot be a corpse:
/// the system releases a named object when the last handle to it closes, which
/// is the whole reason the guard is a mutex rather than a lock file.
fn refusal(owner: Option<u32>) -> String {
    let who = match owner {
        Some(pid) => format!(" It is held by victory.exe (pid {pid})."),
        // A holder that did not publish its pid, which an older build does not.
        None => String::new(),
    };
    format!(
        "another title session is running, so this one stops here without touching the GPU: two sessions share one GPU, one save folder and one shader cache. Wait for it to end, or ask whoever started it to stop it — a session nobody started but you is not yours to end.{who}"
    )
}

/// A failure to make the object at all, as opposed to finding it taken.
fn unavailable(error: std::io::Error) -> String {
    format!(
        "the session object could not be made ({error}): refusing to run without the guard that \
         keeps two sessions apart"
    )
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::System::Threading::{CreateMutexW, GetCurrentProcessId};

    use super::{INHERITED, NAME, OWNER, Session, refusal, unavailable};

    /// `ERROR_ACCESS_DENIED`: the global scope needs a privilege this process
    /// does not have.
    const ACCESS_DENIED: i32 = 5;

    pub(super) fn take() -> anyhow::Result<Session> {
        let (name, handle) = create()?;
        // SAFETY: a plain query of this thread's last error, which is the one
        // `CreateMutexW` set.
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            // SAFETY: the handle is ours, and is not the one we keep.
            unsafe { CloseHandle(handle) };
            anyhow::bail!(refusal(owner()));
        }
        // Every child from here on inherits the exclusion. Set before any
        // thread of ours exists, and only once, so nothing reads it while it
        // is being written.
        // SAFETY: single-threaded here (main, before the title starts).
        unsafe { std::env::set_var(INHERITED, &name) };
        let owner = publish();
        Ok(Session { handle, owner })
    }

    /// The inheritable object, in the widest scope this process may use.
    fn create() -> anyhow::Result<(String, HANDLE)> {
        match make("Global\\") {
            Ok(made) => Ok(made),
            Err(error) if error.raw_os_error() == Some(ACCESS_DENIED) => {
                make("Local\\").map_err(|error| anyhow::anyhow!(unavailable(error)))
            }
            Err(error) => Err(anyhow::anyhow!(unavailable(error))),
        }
    }

    fn make(scope: &str) -> std::io::Result<(String, HANDLE)> {
        let name = format!("{scope}{NAME}");
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        // An inheritable handle is what keeps the exclusion alive in a child
        // that outlives this process.
        let attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: std::ptr::null_mut(),
            bInheritHandle: 1,
        };
        // SAFETY: a null-terminated name, a security descriptor that only asks
        // for inheritance, and no initial ownership.
        let handle = unsafe { CreateMutexW(&raw const attributes, 0, wide.as_ptr()) };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        Ok((name, handle))
    }

    /// Publishes this session's pid beside the object, for the refusal's
    /// message, and keeps the mapping alive with the session.
    fn publish() -> HANDLE {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE,
        };
        let name = OWNER.to_owned();
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        // SAFETY: a null-terminated name and a page-file backed mapping of one
        // pid; every handle created here is inheritable, as the object's is.
        let mapping = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                size_of::<u32>() as u32,
                wide.as_ptr(),
            )
        };
        if mapping.is_null() || mapping == INVALID_HANDLE_VALUE {
            // Not fatal: the mutex alone is the exclusion, this only names the
            // holder in the refusal.
            eprintln!(
                "session: the owner pid could not be published ({})",
                std::io::Error::last_os_error()
            );
            return std::ptr::null_mut();
        }
        // SAFETY: our mapping, and a view of its whole size.
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size_of::<u32>()) };
        if view.Value.is_null() {
            eprintln!(
                "session: the owner pid could not be published ({})",
                std::io::Error::last_os_error()
            );
        } else {
            // SAFETY: a writable view of one pid.
            unsafe { view.Value.cast::<u32>().write(GetCurrentProcessId()) };
        }
        mapping
    }

    /// The pid the holder published, if one did.
    fn owner() -> Option<u32> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_READ, MapViewOfFile, OpenFileMappingW};
        let name = OWNER.to_owned();
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        // SAFETY: a null-terminated name; the mapping is opened read-only.
        let mapping = unsafe { OpenFileMappingW(FILE_MAP_READ, 0, wide.as_ptr()) };
        if mapping.is_null() {
            return None;
        }
        // SAFETY: the mapping is live and holds a pid.
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size_of::<u32>()) };
        let pid = if view.Value.is_null() {
            None
        } else {
            // SAFETY: a readable view of one pid, copied before the unmap.
            let pid = unsafe { view.Value.cast::<u32>().read() };
            Some(pid)
        };
        // SAFETY: the view and the mapping we opened.
        unsafe {
            if !view.Value.is_null() {
                windows_sys::Win32::System::Memory::UnmapViewOfFile(view);
            }
            CloseHandle(mapping);
        }
        pid.filter(|&pid| pid != 0)
    }
}

#[cfg(not(windows))]
mod imp {
    use super::Session;

    pub(super) fn take() -> anyhow::Result<Session> {
        Ok(Session {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refusal_names_the_holder_and_why_it_cannot_be_a_corpse() {
        let text = refusal(Some(4312));
        assert!(text.contains("victory.exe (pid 4312)"), "{text}");
        assert!(text.contains("without touching the GPU"), "{text}");
        // The holder is a live session by construction, so the refusal must not
        // invite the reader to end it: only whoever started it may stop it.
        assert!(!text.contains("or end it"), "{text}");
        assert!(text.contains("not yours to end"), "{text}");
        // A holder that published no pid still refuses, without inventing one.
        let text = refusal(None);
        assert!(text.contains("another title session is running"), "{text}");
        assert!(!text.contains("pid"), "{text}");
    }

    #[test]
    fn a_missing_object_is_not_a_taken_one() {
        let text = unavailable(std::io::Error::from_raw_os_error(5));
        assert!(text.contains("could not be made"), "{text}");
        assert!(!text.contains("another title session"), "{text}");
    }
}
