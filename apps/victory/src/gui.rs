//! Windows file picker for starting a user-supplied local executable.

#[cfg(not(windows))]
pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("the executable picker is available on Windows")
}

#[cfg(windows)]
pub fn run() -> anyhow::Result<()> {
    use std::mem::size_of;
    use std::os::windows::ffi::OsStringExt;
    use std::path::PathBuf;
    use std::process::Command;

    use anyhow::Context;
    use windows_sys::Win32::UI::Controls::Dialogs::{
        GetOpenFileNameW, OFN_FILEMUSTEXIST, OFN_NOCHANGEDIR, OFN_PATHMUSTEXIST, OPENFILENAMEW,
    };

    let filter: Vec<u16> = "Windows executable\0*.exe\0All files\0*.*\0\0"
        .encode_utf16()
        .collect();
    let title: Vec<u16> = "Select a local executable\0".encode_utf16().collect();
    let mut selected = vec![0u16; 32_768];
    let mut dialog = OPENFILENAMEW {
        lStructSize: size_of::<OPENFILENAMEW>() as u32,
        lpstrFilter: filter.as_ptr(),
        lpstrFile: selected.as_mut_ptr(),
        nMaxFile: selected.len() as u32,
        lpstrTitle: title.as_ptr(),
        Flags: OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR,
        ..Default::default()
    };

    // SAFETY: filter, title, and selected remain alive for the call;
    // nMaxFile gives Windows the size of the writable path buffer.
    if unsafe { GetOpenFileNameW(&mut dialog) } == 0 {
        return Ok(());
    }

    let length = selected
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(selected.len());
    let executable = PathBuf::from(std::ffi::OsString::from_wide(&selected[..length]));
    anyhow::ensure!(
        executable
            .extension()
            .is_some_and(|extension| extension.to_string_lossy().eq_ignore_ascii_case("exe")),
        "choose a Windows executable"
    );
    victory_core::ensure_outside_repo(&executable)?;

    let runner = std::env::current_exe().context("finding the VIctory executable")?;
    let status = Command::new(runner)
        .arg("run")
        .arg(&executable)
        .arg("--execute")
        .status()
        .with_context(|| format!("starting {}", executable.display()))?;
    anyhow::ensure!(status.success(), "VIctory exited with {status}");
    Ok(())
}
