//! Windows desktop launcher. Python 3.13 and PySide6 host the Qt Quick UI;
//! the existing Rust runner remains responsible for starting game sessions.

#[cfg(not(windows))]
pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("the VIctory desktop launcher is available on Windows")
}

#[cfg(windows)]
mod native {
    use std::ffi::OsString;
    use std::fs::{self, OpenOptions};
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    use anyhow::{Context, Result};
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MessageBoxW};

    const MAIN_PY: &str = include_str!("../launcher/main.py");
    const MAIN_QML: &str = include_str!("../launcher/qml/Main.qml");
    const HERO_SVG: &str = include_str!("../launcher/qml/hero.svg");

    struct Runtime {
        command: OsString,
        prefix: Vec<OsString>,
        packaged: bool,
    }

    pub fn run() -> Result<()> {
        let launcher_dir = launcher_dir()?;
        let qml_dir = launcher_dir.join("qml");
        fs::create_dir_all(&qml_dir).with_context(|| format!("creating {}", qml_dir.display()))?;
        write_if_changed(&launcher_dir.join("main.py"), MAIN_PY)?;
        write_if_changed(&qml_dir.join("Main.qml"), MAIN_QML)?;
        write_if_changed(&qml_dir.join("hero.svg"), HERO_SVG)?;

        let current_exe = std::env::current_exe().context("finding victory.exe")?;
        let runtime = find_runtime(&current_exe, &launcher_dir);
        let script = launcher_dir.join("main.py");
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .context("finding the VIctory repository root")?;

        let mut command = Command::new(&runtime.command);
        command.args(&runtime.prefix);
        if !runtime.packaged {
            command.arg(&script);
        }
        let bootstrap_log = launcher_dir.join("launcher-runtime.log");
        let log_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&bootstrap_log)
            .with_context(|| format!("opening {}", bootstrap_log.display()))?;
        let log_copy = log_file
            .try_clone()
            .context("opening launcher log stream")?;
        command
            .arg("--victory-exe")
            .arg(&current_exe)
            .arg("--repo-root")
            .arg(repo_root)
            .current_dir(&launcher_dir)
            .env("VICTORY_HOME", &launcher_dir)
            .env("PYTHONIOENCODING", "utf-8")
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_copy))
            .stderr(Stdio::from(log_file));

        let status = match command.status() {
            Ok(status) => status,
            Err(error) => {
                let message = format!(
                    "VIctory could not start the Python 3.13 / PySide6 launcher runtime.\n\n\
                     Install the launcher requirements or place the packaged runtime beside \
                     victory.exe.\n\n{error}"
                );
                show_error(&message);
                return Err(error).context("starting the VIctory Qt Quick launcher");
            }
        };
        if !status.success() {
            let details = fs::read_to_string(&bootstrap_log).unwrap_or_default();
            let details = details.trim();
            let details = if details.chars().count() > 2_400 {
                let tail: String = details.chars().rev().take(2_400).collect();
                tail.chars().rev().collect::<String>()
            } else {
                details.to_owned()
            };
            let message = if details.is_empty() {
                format!("The Qt Quick launcher exited before opening ({status}).")
            } else {
                format!("The Qt Quick launcher exited before opening ({status}):\n\n{details}")
            };
            show_error(&message);
            anyhow::bail!("{message}");
        }
        Ok(())
    }

    fn launcher_dir() -> Result<PathBuf> {
        let local = std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set")?;
        let path = PathBuf::from(local).join("victory").join("launcher");
        victory_core::ensure_outside_repo(&path)?;
        fs::create_dir_all(&path).with_context(|| format!("creating {}", path.display()))?;
        Ok(path)
    }

    fn find_runtime(current_exe: &Path, launcher_dir: &Path) -> Runtime {
        if let Some(path) = std::env::var_os("VICTORY_PYTHON") {
            return Runtime {
                command: path,
                prefix: Vec::new(),
                packaged: false,
            };
        }

        let exe_dir = current_exe.parent().unwrap_or_else(|| Path::new("."));
        let candidates = [
            (
                exe_dir
                    .join("launcher-runtime")
                    .join("VictoryQmlRuntime")
                    .join("VictoryQmlRuntime.exe"),
                true,
            ),
            (exe_dir.join("launcher-runtime").join("python.exe"), false),
            (
                launcher_dir
                    .join("runtime")
                    .join("VictoryQmlRuntime")
                    .join("VictoryQmlRuntime.exe"),
                true,
            ),
            (launcher_dir.join("runtime").join("python.exe"), false),
            (launcher_dir.join("python313").join("python.exe"), false),
            (
                launcher_dir.join("venv").join("Scripts").join("python.exe"),
                false,
            ),
        ];
        if let Some((path, packaged)) = candidates.iter().find(|(path, _)| path.is_file()) {
            return Runtime {
                command: path.as_os_str().to_owned(),
                prefix: Vec::new(),
                packaged: *packaged,
            };
        }

        Runtime {
            command: OsString::from("py.exe"),
            prefix: vec![OsString::from("-3.13")],
            packaged: false,
        }
    }

    fn write_if_changed(path: &Path, content: &str) -> Result<()> {
        if let Ok(existing) = fs::read_to_string(path) {
            if existing == content {
                return Ok(());
            }
        }
        fs::write(path, content).with_context(|| format!("writing {}", path.display()))
    }

    fn show_error(message: &str) {
        let title = wide("VIctory Launcher");
        let message = wide(message);
        // SAFETY: Both strings are NUL-terminated UTF-16 buffers that stay alive
        // for the duration of the synchronous Windows message box call.
        unsafe {
            MessageBoxW(
                std::ptr::null_mut(),
                message.as_ptr(),
                title.as_ptr(),
                MB_OK | MB_ICONERROR,
            );
        }
    }

    fn wide(value: &str) -> Vec<u16> {
        std::ffi::OsStr::new(value)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }
}

#[cfg(windows)]
pub use native::run;
