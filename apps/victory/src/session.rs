//! One title session at a time, held by the operating system rather than by a
//! rule (CREW.md, "One GPU session at a time — never two").
//!
//! The title shares one GPU, one save folder and one shader cache, so two
//! sessions at once is the one failure this project cannot recover from: it
//! blacked the machine once already. A named mutex makes that a fact instead
//! of a hope — the second process finds the object taken and leaves before it
//! touches the GPU.
//!
//! A child this session starts finds `VICTORY_SESSION_INHERITED` in its
//! environment (set once, here, before any thread of ours exists), opens the
//! name for itself and shares the parent's session instead of refusing itself.
//! The handle is deliberately not inheritable: when it was, every process the
//! title started inherited it — a console host among them — and a child that
//! outlived the title kept the name taken with no session behind it.
//!
//! The name is in the `Global\` scope, which every session on the machine
//! shares, and a mutex needs no privilege for it (`SeCreateGlobalPrivilege`
//! guards file mappings only — that is why the owner pid is published in a
//! `Local\` mapping). There is deliberately no `Local\` fallback: an
//! `ACCESS_DENIED` from the create means the name is held by a session this
//! process may not open, so it is a refusal and not a reason to try a name
//! that cannot collide with the holder.

use anyhow::Result;

/// The session object's name, in the machine-wide `Global\` scope.
///
/// The suffix is not decoration. A handle that survives its owner keeps a
/// mutex object alive while nobody holds it, and the guard refuses on
/// `ERROR_ALREADY_EXISTS`, so one dead handle to the old name refused every
/// launch on this machine with no session to show for it (2026-11-24: three
/// launches in five seconds, no `victory.exe` alive, the object still
/// openable). A new name is a new object and the corpse is irrelevant.
///
/// It also splits the fleet: a binary built before the rename speaks
/// `VIctory-title-session` and will not refuse a session under this name, so
/// every build on the machine has to be rebuilt before mixed binaries are
/// trusted together.
const NAME: &str = "VIctory-title-session-2";

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
    // the name again would refuse the child against its own parent. The
    // handle is what is inherited, not this variable, so the variable is a
    // claim to verify: a stray `set`, or a spawn that passed no handles, would
    // otherwise run a session with no guard at all.
    if let Some(name) = std::env::var_os(INHERITED) {
        return imp::inherited(&name.to_string_lossy());
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
const OWNER: &str = "Local\\VIctory-title-owner-2";

/// What both refusals say first: the fact, and who may end it.
fn stopping() -> &'static str {
    concat!(
        "another title session is running, so this one stops here without touching the GPU: two ",
        "sessions share one GPU, one save folder and one shader cache. Wait for it to end, or ask ",
        "whoever started it to stop it — a session nobody started but you is not yours to end."
    )
}

/// The refusal a second session prints. Pure, so the wording is testable
/// without a second process.
///
/// Only a `run` takes this object - translating a corpus, inspecting an
/// executable and every other subcommand are offline and never touch it - so
/// the holder is a title session by construction. It also cannot be a corpse:
/// the system releases a named object when the last handle to it closes, which
/// is the whole reason the guard is a mutex rather than a lock file.
fn refusal(owner: Option<u32>) -> String {
    // What this process knows is the pid the taker published, not the one
    // holding the object now: a child the taker started holds the same object
    // and outlives it, and an older build publishes nothing at all.
    let who = match owner {
        Some(pid) => format!(
            " The session that took it was victory.exe (pid {pid}); a child it started may be holding it too."
        ),
        None => " The session holding it published no pid, so it cannot be named here.".to_owned(),
    };
    format!("{}{who}", stopping())
}

/// The refusal when the name is taken and opening it is denied, which is the
/// one case where who holds it cannot be read at all.
#[cfg_attr(not(windows), allow(dead_code))]
fn held() -> String {
    format!(
        "{} The name is taken and opening it is denied, so this process cannot read who holds it.",
        stopping()
    )
}

/// A failure to make the object at all, as opposed to finding it taken.
fn unavailable(error: std::io::Error) -> String {
    format!(
        "the session object could not be made ({error}): refusing to run without the guard that keeps two sessions apart"
    )
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_ACCESS_DENIED, ERROR_ALREADY_EXISTS, GetLastError, HANDLE,
        INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::System::Threading::{
        CreateMutexW, GetCurrentProcessId, MUTEX_MODIFY_STATE, OpenMutexW,
    };

    use super::{INHERITED, NAME, OWNER, Session, held, refusal, unavailable};

    pub(super) fn take() -> anyhow::Result<Session> {
        take_named(&format!("Global\\{NAME}"))
    }

    /// Takes a session under `name`, refusing when it is already held.
    pub(super) fn take_named(name: &str) -> anyhow::Result<Session> {
        let handle = make(name)?;
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
        unsafe { std::env::set_var(INHERITED, name) };
        let owner = publish();
        Ok(Session { handle, owner })
    }

    /// A child of a session: the object it inherited has to be open, because
    /// that is what having inherited a handle means. A process carrying the
    /// variable without one refuses here rather than running unguarded.
    pub(super) fn inherited(name: &str) -> anyhow::Result<Session> {
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        // SAFETY: a null-terminated name; the object is opened only to prove
        // it exists, and the inherited handle is what holds it.
        let handle = unsafe { OpenMutexW(MUTEX_MODIFY_STATE, 0, wide.as_ptr()) };
        if handle.is_null() {
            anyhow::bail!(unavailable(std::io::Error::last_os_error()));
        }
        // SAFETY: a handle we opened and do not keep.
        unsafe { CloseHandle(handle) };
        Ok(Session {
            handle: std::ptr::null_mut(),
            owner: std::ptr::null_mut(),
        })
    }

    /// The inheritable object. One name and no fallback: `ACCESS_DENIED` means
    /// a session holds the name and this process may not open it, so a `Local\`
    /// object made instead cannot collide with that holder and would let a
    /// second session start beside it — silently, and only in that direction.
    fn make(name: &str) -> anyhow::Result<HANDLE> {
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        // Not inheritable, deliberately. A child of this session opens the
        // name for itself (`inherited`), so the exclusion survives a child that
        // outlives this process without the handle being handed to every
        // process the title starts — a console host that inherits it keeps the
        // session name taken after the title is gone, and then every launch on
        // the machine refuses with no session to show for it. Measured on
        // 2026-11-24: three launches refused, no `victory.exe` alive, and
        // `Global\VIctory-title-session` still openable.
        let attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: std::ptr::null_mut(),
            bInheritHandle: 0,
        };
        // SAFETY: a null-terminated name, a security descriptor that only asks
        // for inheritance, and no initial ownership.
        let handle = unsafe { CreateMutexW(&raw const attributes, 0, wide.as_ptr()) };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            let error = std::io::Error::last_os_error();
            return Err(match error.raw_os_error() {
                Some(code) if code == ERROR_ACCESS_DENIED as i32 => anyhow::anyhow!(held()),
                _ => anyhow::anyhow!(unavailable(error)),
            });
        }
        Ok(handle)
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

    pub(super) fn inherited(_: &str) -> anyhow::Result<Session> {
        Ok(Session {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refusal_names_the_publisher_and_why_it_cannot_be_a_corpse() {
        let text = refusal(Some(4312));
        assert!(text.contains("victory.exe (pid 4312)"), "{text}");
        assert!(text.contains("without touching the GPU"), "{text}");
        // The pid is the taker's, and a child of it can hold the object.
        assert!(text.contains("a child it started may be holding it too"), "{text}");
        // The holder is a live session by construction, so the refusal must not
        // invite the reader to end it: only whoever started it may stop it.
        assert!(!text.contains("or end it"), "{text}");
        assert!(text.contains("not yours to end"), "{text}");
        // A holder that published no pid still refuses, without inventing one.
        let text = refusal(None);
        assert!(text.contains("another title session is running"), "{text}");
        assert!(text.contains("published no pid"), "{text}");
        assert!(!text.contains("(pid"), "{text}");
    }

    #[test]
    fn a_denied_name_is_a_refusal_and_not_a_second_object() {
        let text = held();
        assert!(text.contains("another title session is running"), "{text}");
        assert!(text.contains("opening it is denied"), "{text}");
    }

    #[test]
    fn a_missing_object_is_not_a_taken_one() {
        let text = unavailable(std::io::Error::from_raw_os_error(5));
        assert!(text.contains("could not be made"), "{text}");
        assert!(!text.contains("another title session"), "{text}");
    }

    /// The mechanism, offline: a name taken twice refuses, and a name released
    /// by its holder is free again. Its own name, so a title session running on
    /// this machine cannot decide whether the test passes.
    #[cfg(windows)]
    #[test]
    fn a_second_session_under_one_name_is_refused() {
        let name = format!("Global\\VIctory-title-session-test-{}", std::process::id());
        let first = imp::take_named(&name).expect("a free name");
        let text = imp::take_named(&name)
            .err()
            .expect("the same name twice")
            .to_string();
        assert!(text.contains("another title session is running"), "{text}");
        assert!(text.contains("pid"), "{text}");
        drop(first);
        imp::take_named(&name).expect("a name its holder released");
    }

    /// A child that carries the marker without an object in reach is not
    /// admitted: the variable is a claim, and the object is what proves it.
    #[cfg(windows)]
    #[test]
    fn a_child_without_an_inherited_object_is_not_admitted() {
        let name = format!("Global\\VIctory-no-session-{}", std::process::id());
        let error = imp::inherited(&name).err().expect("nothing to inherit");
        let text = error.to_string();
        assert!(text.contains("could not be made"), "{text}");
    }
}
