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
        });
    }
    imp::take()
}

impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(windows)]
        if !self.handle.is_null() {
            // SAFETY: our own handle, closed once.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.handle) };
        }
    }
}

/// A session someone else holds: its pid, image, and whether it looks alive or
/// left over (its parent is gone, which is what a crash or a restart leaves).
struct Other {
    pid: u32,
    name: String,
    alive: bool,
}

/// The refusal a second session prints, with the sessions to act on: the
/// object itself names no process. Pure, so the wording is testable without a
/// second process.
///
/// An alive session and a corpse need opposite reactions — wait for the first,
/// end the second — so the line says which each looks like.
fn refusal(others: &[Other]) -> String {
    let mut running = Vec::new();
    let mut corpses = Vec::new();
    for other in others {
        let entry = format!("{} (pid {})", other.name, other.pid);
        if other.alive {
            running.push(entry);
        } else {
            corpses.push(entry);
        }
    }
    let mut who = String::new();
    if !running.is_empty() {
        who.push_str(&format!(" Running title sessions: {}.", running.join(", ")));
    }
    if !corpses.is_empty() {
        who.push_str(&format!(
            " Left over with no parent, which is what a crash or a restart leaves: {}. Ending one \
             is safe, it is not running a title.",
            corpses.join(", ")
        ));
    }
    format!(
        "another title session is running, so this one stops here without touching the GPU: two sessions share one GPU, one save folder and one shader cache.{who}"
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
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::Threading::{CreateMutexW, GetCurrentProcessId};

    use super::{INHERITED, NAME, Session, refusal, unavailable};

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
            anyhow::bail!(refusal(&others()));
        }
        // Every child from here on inherits the exclusion. Set before any
        // thread of ours exists, and only once, so nothing reads it while it
        // is being written.
        // SAFETY: single-threaded here (main, before the title starts).
        unsafe { std::env::set_var(INHERITED, &name) };
        Ok(Session { handle })
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

    /// Other processes whose executable looks like ours, without this one, each
    /// with whether its parent is still alive.
    fn others() -> Vec<super::Other> {
        // SAFETY: a plain query of our own pid.
        let me = unsafe { GetCurrentProcessId() };
        let mut found = Vec::new();
        // SAFETY: a snapshot of the process list, closed below.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if snapshot.is_null() || snapshot == INVALID_HANDLE_VALUE {
            return found;
        }
        // Everything alive in one pass, to tell a session that is running from
        // one whose parent is gone.
        let mut live = std::collections::HashSet::new();
        let mut ours = Vec::new();
        let mut entry: PROCESSENTRY32W = unsafe { std::mem::zeroed() };
        entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
        // SAFETY: a live snapshot and a structure of the size it was told.
        let mut more = unsafe { Process32FirstW(snapshot, &mut entry) } != 0;
        while more {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            let name = String::from_utf16_lossy(&entry.szExeFile[..len]);
            live.insert(entry.th32ProcessID);
            // Our own image only: a session may start renamed children (the
            // shader workers) that are not sessions themselves.
            if entry.th32ProcessID != me && name.eq_ignore_ascii_case("victory.exe") {
                ours.push((entry.th32ProcessID, entry.th32ParentProcessID, name));
            }
            // SAFETY: as above.
            more = unsafe { Process32NextW(snapshot, &mut entry) } != 0;
        }
        // SAFETY: our snapshot.
        unsafe { CloseHandle(snapshot) };
        for (pid, parent, name) in ours {
            // A launcher that is still running means a session someone drives;
            // a parent that is gone is what a crash or a restart leaves behind.
            let alive = parent != 0 && live.contains(&parent);
            found.push(super::Other { pid, name, alive });
        }
        found
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
    fn the_refusal_tells_a_running_session_from_a_corpse() {
        let running = [Other {
            pid: 4312,
            name: "victory.exe".to_owned(),
            alive: true,
        }];
        let text = refusal(&running);
        assert!(
            text.contains("Running title sessions: victory.exe (pid 4312)"),
            "{text}"
        );
        assert!(text.contains("without touching the GPU"), "{text}");
        assert!(!text.contains("No parent"), "{text}");

        // A corpse says it is safe to end, which is the reaction it needs.
        let text = refusal(&[Other {
            pid: 77,
            name: "victory.exe".to_owned(),
            alive: false,
        }]);
        assert!(text.contains("no parent"), "{text}");
        assert!(text.contains("Ending one is safe"), "{text}");
        assert!(!text.contains("Running title sessions"), "{text}");

        // Both at once, each named for what it is.
        let text = refusal(&[
            Other {
                pid: 1,
                name: "victory.exe".to_owned(),
                alive: true,
            },
            Other {
                pid: 2,
                name: "victory.exe".to_owned(),
                alive: false,
            },
        ]);
        assert!(text.contains("Running title sessions: victory.exe (pid 1)"), "{text}");
        assert!(text.contains("victory.exe (pid 2)"), "{text}");

        // Without a visible holder it still says the session is taken.
        let text = refusal(&[]);
        assert!(text.contains("another title session is running"), "{text}");
        assert!(!text.contains("Running title sessions"), "{text}");
    }

    #[test]
    fn a_missing_object_is_not_a_taken_one() {
        let text = unavailable(std::io::Error::from_raw_os_error(5));
        assert!(text.contains("could not be made"), "{text}");
        assert!(!text.contains("another title session"), "{text}");
    }
}
