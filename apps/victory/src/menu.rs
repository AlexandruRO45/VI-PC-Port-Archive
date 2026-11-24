//! What opening `victory.exe` from Explorer shows: a small menu, where the
//! command-line help used to flash in a window that closed at once (#527).
//!
//! Run a game (its executable dragged onto the window, or typed), run the
//! last one again, check what this PC has for VIctory (Windows 10's shader
//! model, AVX2, the GPU's shader model, mesh shaders and wave size, the
//! configured paths), or open the data folder (saves, `crashes.log`). The
//! window stays open after a run until Enter, so its last lines can be
//! read.
//!
//! The colours are a sunset gradient of VIctory's own (violet, pink,
//! peach), nothing of the title's. `NO_COLOR` turns them off, and so does a
//! console without virtual-terminal sequences.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::{Console, Unimplemented};

/// Whether `victory` was started with no arguments in a console of its own
/// (a double-click), or `VICTORY_MENU=1` asks for the menu anyway.
pub fn wanted() -> bool {
    if std::env::args_os().len() != 1 {
        return false;
    }
    std::env::var("VICTORY_MENU").is_ok_and(|v| v == "1") || own_console()
}

/// The console has no other process attached: Explorer made it for us.
#[cfg(windows)]
fn own_console() -> bool {
    let mut ids = [0u32; 2];
    // SAFETY: the buffer holds `ids.len()` process ids.
    let n = unsafe {
        windows_sys::Win32::System::Console::GetConsoleProcessList(
            ids.as_mut_ptr(),
            ids.len() as u32,
        )
    };
    n == 1
}

#[cfg(not(windows))]
fn own_console() -> bool {
    false
}

/// A 24-bit colour.
type Rgb = (u8, u8, u8);

/// The sunset, top to bottom: dusk blue, violet, orchid, pink, peach, amber.
const SUNSET: [Rgb; 6] = [
    (86, 98, 196),
    (134, 96, 204),
    (198, 104, 184),
    (236, 128, 164),
    (246, 160, 124),
    (250, 190, 110),
];
const CREAM: Rgb = (255, 244, 230);
const DIM: Rgb = (176, 150, 190);
const GOOD: Rgb = (120, 220, 200);
const WARN: Rgb = (250, 190, 110);
const BAD: Rgb = (240, 100, 140);

/// Writes text in colour, or plain.
#[derive(Clone, Copy)]
struct Paint {
    color: bool,
}

impl Paint {
    fn rgb(self, text: &str, (r, g, b): Rgb) -> String {
        if self.color {
            format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    fn bold(self, text: &str, c: Rgb) -> String {
        if self.color {
            format!("\x1b[1m{}", self.rgb(text, c))
        } else {
            text.to_owned()
        }
    }

    /// Each character of `text` at its place along `stops`.
    fn gradient(self, text: &str, stops: &[Rgb]) -> String {
        if !self.color {
            return text.to_owned();
        }
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len().max(2) - 1;
        let mut out = String::from("\x1b[1m");
        for (i, ch) in chars.iter().enumerate() {
            let (r, g, b) = along(stops, i as f32 / n as f32);
            out.push_str(&format!("\x1b[38;2;{r};{g};{b}m{ch}"));
        }
        out.push_str("\x1b[0m");
        out
    }
}

/// The colour at `t` (0..=1) along evenly spaced `stops`.
fn along(stops: &[Rgb], t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
    let i = (t.floor() as usize).min(stops.len() - 2);
    let f = t - i as f32;
    let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * f).round() as u8;
    let (a, b) = (stops[i], stops[i + 1]);
    (mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

/// Turns on virtual-terminal sequences; false where the console has none.
/// `NO_COLOR` turns colour off, `CLICOLOR_FORCE=1` on (output to a file).
#[cfg(windows)]
fn enable_color() -> bool {
    use windows_sys::Win32::System::Console::{
        ENABLE_VIRTUAL_TERMINAL_PROCESSING, GetConsoleMode, GetStdHandle, STD_OUTPUT_HANDLE,
        SetConsoleMode,
    };
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("CLICOLOR_FORCE").is_ok_and(|v| v == "1") {
        return true;
    }
    // SAFETY: plain console calls on our own output handle.
    unsafe {
        let out = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0;
        GetConsoleMode(out, &mut mode) != 0
            && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
    }
}

#[cfg(not(windows))]
fn enable_color() -> bool {
    std::env::var_os("NO_COLOR").is_none()
}

/// What a menu answer asks for.
#[derive(Debug, PartialEq, Eq)]
enum Choice {
    Run,
    Again,
    Check,
    Data,
    Quit,
    Unknown,
}

fn choice(input: &str) -> Choice {
    // (A byte-order mark leads piped input from some shells.)
    match input
        .trim_start_matches('\u{feff}')
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "1" => Choice::Run,
        "2" | "r" => Choice::Again,
        "3" | "c" => Choice::Check,
        "4" | "d" => Choice::Data,
        "q" | "0" | "x" | "exit" | "quit" => Choice::Quit,
        _ => Choice::Unknown,
    }
}

/// A path as a console gives it when a file is dragged onto the window or
/// pasted: spaces around it, quotes around it, PowerShell's `& '...'`.
fn clean_path(input: &str) -> PathBuf {
    let mut s = input.trim_start_matches('\u{feff}').trim();
    if let Some(rest) = s.strip_prefix('&') {
        s = rest.trim();
    }
    for q in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
            s = &s[1..s.len() - 1];
        }
    }
    PathBuf::from(s)
}

/// Where the last game run from the menu is remembered.
fn last_file(data_dir: &Path) -> PathBuf {
    data_dir.join("menu-last-game.txt")
}

fn load_last(data_dir: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(last_file(data_dir)).ok()?;
    let path = PathBuf::from(text.trim());
    path.is_file().then_some(path)
}

fn save_last(data_dir: &Path, exe: &Path) {
    let _ = std::fs::create_dir_all(data_dir);
    let _ = std::fs::write(last_file(data_dir), exe.display().to_string());
}

/// One line of the setup check.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    Good,
    Warn,
    Bad,
}

/// What [`check`] found, as facts.
#[derive(Debug, Clone, Default)]
struct Pc {
    /// Windows build number, if read.
    windows_build: Option<u32>,
    avx2: bool,
    /// The GPU device made with shader model 6.6, and if so its mesh
    /// shader support and widest wave.
    sm66: Option<bool>,
    mesh_shaders: Option<bool>,
    max_lanes: Option<u32>,
    /// The Agility runtime folder beside this executable.
    agility_beside: bool,
}

/// The setup check's lines: status, what, and what to do.
fn verdicts(pc: &Pc) -> Vec<(Status, String, Option<String>)> {
    let mut out = Vec::new();
    match pc.windows_build {
        Some(b) if b >= 22000 => out.push((Status::Good, format!("Windows 11 (build {b})"), None)),
        Some(b) => out.push((
            Status::Warn,
            format!("Windows 10 (build {b})"),
            Some(if pc.agility_beside {
                "Its built-in D3D12 stops at shader model 6.5. A D3D12\\ folder is beside victory.exe; this build must also be made to load it (#175).".to_owned()
            } else {
                "Its built-in D3D12 stops at shader model 6.5. VIctory needs 6.6 from the D3D12 Agility SDK runtime (#84, #175).".to_owned()
            }),
        )),
        None => out.push((Status::Warn, "Windows version unknown".to_owned(), None)),
    }
    out.push(if pc.avx2 {
        (Status::Good, "CPU has AVX2".to_owned(), None)
    } else {
        (
            Status::Bad,
            "CPU has no AVX2".to_owned(),
            Some("The title needs it.".to_owned()),
        )
    });
    match pc.sm66 {
        Some(true) => out.push((Status::Good, "GPU: shader model 6.6".to_owned(), None)),
        Some(false) => out.push((
            Status::Bad,
            "GPU: no shader model 6.6 device".to_owned(),
            Some("Translated shaders need it. On Windows 10, see above.".to_owned()),
        )),
        None => out.push((Status::Warn, "GPU: not checked".to_owned(), None)),
    }
    if pc.sm66 == Some(true) {
        match pc.mesh_shaders {
            Some(true) => out.push((Status::Good, "GPU: mesh shaders".to_owned(), None)),
            Some(false) => out.push((
                Status::Warn,
                "GPU: no mesh shaders".to_owned(),
                Some("Menus can run; the world's mesh draws are refused (#3).".to_owned()),
            )),
            None => {}
        }
        if let Some(lanes) = pc.max_lanes {
            out.push(if lanes >= 64 {
                (Status::Good, format!("GPU: waves up to {lanes} lanes"), None)
            } else {
                (
                    Status::Warn,
                    format!("GPU: waves of {lanes} lanes"),
                    Some("The title's programs are written for 64; some run in halves, some are refused (#578, #778).".to_owned()),
                )
            });
        }
    }
    out
}

/// The configured paths' lines, as [`verdicts`] gives them.
fn path_verdicts(cfg: &victory_core::Config) -> Vec<(Status, String, Option<String>)> {
    let mut out = Vec::new();
    match &cfg.game_dir {
        Some(d) if d.is_dir() => {
            out.push((Status::Good, format!("Game folder: {}", d.display()), None));
        }
        Some(d) => out.push((
            Status::Warn,
            format!("Game folder: {} (not there)", d.display()),
            Some(
                "Fix game_dir in victory.toml. Running from this menu does not need it.".to_owned(),
            ),
        )),
        None => out.push((
            Status::Warn,
            "Game folder: not set".to_owned(),
            Some(
                "Only the other commands need it (game_dir in victory.toml); 1 asks for the .exe."
                    .to_owned(),
            ),
        )),
    }
    for (what, dir) in [
        ("Data folder", &cfg.data_dir),
        ("Cache folder", &cfg.cache_dir),
    ] {
        match victory_core::ensure_outside_repo(dir) {
            Ok(()) => out.push((Status::Good, format!("{what}: {}", dir.display()), None)),
            Err(e) => out.push((
                Status::Bad,
                format!("{what}: {}", dir.display()),
                Some(e.to_string()),
            )),
        }
    }
    out
}

/// The facts of this PC.
fn check() -> Pc {
    let mut pc = Pc {
        windows_build: windows_build(),
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        agility_beside: std::env::current_exe()
            .ok()
            .and_then(|e| {
                e.parent()
                    .map(|d| d.join("D3D12").join("D3D12Core.dll").is_file())
            })
            .unwrap_or(false),
        ..Pc::default()
    };
    gpu(&mut pc);
    pc
}

#[cfg(windows)]
fn windows_build() -> Option<u32> {
    use windows_sys::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW};
    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let (key, name) = (
        wide("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion"),
        wide("CurrentBuildNumber"),
    );
    let mut buf = [0u16; 32];
    let mut len = std::mem::size_of_val(&buf) as u32;
    // SAFETY: NUL-terminated names and a buffer of `len` bytes.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            buf.as_mut_ptr().cast(),
            &mut len,
        )
    };
    if status != 0 {
        return None;
    }
    String::from_utf16_lossy(&buf[..(len as usize / 2).saturating_sub(1)])
        .trim()
        .parse()
        .ok()
}

#[cfg(not(windows))]
fn windows_build() -> Option<u32> {
    None
}

/// The default adapter's shader model, mesh shaders and wave size, asked
/// in this process: what a run from this executable would get.
#[cfg(windows)]
fn gpu(pc: &mut Pc) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_D3D12_OPTIONS7,
        D3D12_FEATURE_DATA_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
        D3D12_MESH_SHADER_TIER_NOT_SUPPORTED,
    };
    let Ok(host) = victory_shader::emit::host::Host::new(false) else {
        pc.sm66 = Some(false);
        return;
    };
    pc.sm66 = Some(true);
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: each query writes its own structure of the given size.
    unsafe {
        if host
            .device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&raw mut o7).cast(),
                size_of_val(&o7) as u32,
            )
            .is_ok()
        {
            pc.mesh_shaders = Some(o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED);
        }
        if host
            .device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut o1).cast(),
                size_of_val(&o1) as u32,
            )
            .is_ok()
        {
            pc.max_lanes = Some(o1.WaveLaneCountMax);
        }
    }
}

#[cfg(not(windows))]
fn gpu(_: &mut Pc) {}

/// The title line and the horizon under it.
fn banner(p: Paint) -> String {
    let title = p.gradient("V I C T O R Y", &SUNSET);
    let horizon = if p.color {
        p.gradient(&"\u{2500}".repeat(40), &SUNSET)
    } else {
        "-".repeat(40)
    };
    format!(
        "\n  {title}\n  {}\n  {horizon}\n",
        p.rgb("your own copy of the title, on this PC", DIM)
    )
}

fn menu_text(p: Paint, last: Option<&Path>) -> String {
    let key = |k: &str| p.bold(k, SUNSET[3]);
    let mut out = banner(p);
    out.push('\n');
    out.push_str(&format!("  {}  Run a game\n", key("1")));
    match last {
        Some(path) => out.push_str(&format!(
            "  {}  Run again   {}\n",
            key("2"),
            p.rgb(&path.display().to_string(), DIM)
        )),
        None => out.push_str(&p.rgb("  2  Run again   (nothing run yet)\n", DIM)),
    }
    out.push_str(&format!("  {}  Check this PC\n", key("3")));
    out.push_str(&format!(
        "  {}  Open the data folder   {}\n",
        key("4"),
        p.rgb("saves, crashes.log", DIM)
    ));
    out.push_str(&format!("  {}  Quit\n\n", key("Q")));
    out.push_str(&format!("  {} ", p.rgb("Choose:", CREAM)));
    out
}

fn check_text(p: Paint, pc: &Pc, cfg: Option<&victory_core::Config>) -> String {
    let mut out = String::new();
    let mut section = |title: &str, lines: Vec<(Status, String, Option<String>)>| {
        out.push_str(&format!("\n  {}\n\n", p.bold(title, SUNSET[2])));
        for (status, what, hint) in lines {
            let mark = match status {
                Status::Good => p.rgb("\u{2713}", GOOD),
                Status::Warn => p.rgb("!", WARN),
                Status::Bad => p.rgb("\u{2717}", BAD),
            };
            out.push_str(&format!("  {mark}  {what}\n"));
            if let Some(h) = hint {
                out.push_str(&format!("     {}\n", p.rgb(&h, DIM)));
            }
        }
    };
    section("This PC", verdicts(pc));
    if let Some(cfg) = cfg {
        section("Paths", path_verdicts(cfg));
    }
    out
}

fn read_line(input: &mut impl BufRead) -> Option<String> {
    let mut line = String::new();
    match input.read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
    }
}

fn pause(p: Paint, input: &mut impl BufRead) {
    print!("\n  {} ", p.rgb("Press Enter to go back to the menu.", DIM));
    let _ = std::io::stdout().flush();
    let _ = read_line(input);
}

/// Runs `exe` as `victory run <exe> --execute` would, then waits for Enter.
fn run_game(p: Paint, exe: &Path, data_dir: &Path, input: &mut impl BufRead) {
    if !exe.is_file() {
        println!(
            "\n  {} {}",
            p.rgb("\u{2717}", BAD),
            p.rgb("No file there.", CREAM)
        );
        pause(p, input);
        return;
    }
    save_last(data_dir, exe);
    println!("\n  {} {}\n", p.rgb("\u{25b6}", SUNSET[4]), exe.display());
    match crate::run::run(exe, Unimplemented::Trap, false, true, Console::SeriesX) {
        Ok(()) => println!(
            "\n  {} {}",
            p.rgb("\u{2713}", GOOD),
            p.rgb("The run ended.", CREAM)
        ),
        Err(e) => println!(
            "\n  {} {}\n     {}",
            p.rgb("\u{2717}", BAD),
            p.rgb("The run stopped:", CREAM),
            p.rgb(&format!("{e:#}"), DIM)
        ),
    }
    pause(p, input);
}

pub fn run() -> Result<()> {
    let p = Paint {
        color: enable_color(),
    };
    let data_dir = victory_core::Config::load()
        .map(|c| c.data_dir)
        .unwrap_or_else(|_| std::env::temp_dir().join("victory"));
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        let last = load_last(&data_dir);
        print!("{}", menu_text(p, last.as_deref()));
        let _ = std::io::stdout().flush();
        let Some(line) = read_line(&mut input) else {
            return Ok(());
        };
        match choice(&line) {
            Choice::Run => {
                print!(
                    "\n  {}\n  {} ",
                    p.rgb("Drag the game's .exe onto this window, or type its path, then press Enter.", CREAM),
                    p.rgb(">", SUNSET[3])
                );
                let _ = std::io::stdout().flush();
                let Some(path) = read_line(&mut input) else {
                    return Ok(());
                };
                if !path.trim().is_empty() {
                    run_game(p, &clean_path(&path), &data_dir, &mut input);
                }
            }
            Choice::Again => match last {
                Some(exe) => run_game(p, &exe, &data_dir, &mut input),
                None => {
                    println!("\n  {}", p.rgb("Nothing run yet: choose 1 first.", DIM));
                    pause(p, &mut input);
                }
            },
            Choice::Check => {
                let cfg = victory_core::Config::load().ok();
                print!("{}", check_text(p, &check(), cfg.as_ref()));
                pause(p, &mut input);
            }
            Choice::Data => {
                let _ = std::fs::create_dir_all(&data_dir);
                let _ = std::process::Command::new("explorer")
                    .arg(&data_dir)
                    .spawn();
            }
            Choice::Quit => return Ok(()),
            Choice::Unknown if line.trim().is_empty() => {}
            Choice::Unknown => {
                println!(
                    "\n  {}",
                    p.rgb("Type 1, 2, 3 or 4 and press Enter (Q to quit).", WARN)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragged_and_pasted_paths_come_clean() {
        let want = PathBuf::from(r"D:\Games\Some Title\title.exe");
        for input in [
            "D:\\Games\\Some Title\\title.exe\r\n",
            "  \"D:\\Games\\Some Title\\title.exe\"  \n",
            "& 'D:\\Games\\Some Title\\title.exe'\n",
            "'D:\\Games\\Some Title\\title.exe'",
        ] {
            assert_eq!(clean_path(input), want, "{input:?}");
        }
    }

    #[test]
    fn answers_map_to_choices() {
        assert_eq!(choice("1\n"), Choice::Run);
        assert_eq!(choice(" R "), Choice::Again);
        assert_eq!(choice("3"), Choice::Check);
        assert_eq!(choice("d"), Choice::Data);
        assert_eq!(choice("Q"), Choice::Quit);
        assert_eq!(choice("exit"), Choice::Quit);
        assert_eq!(choice("9"), Choice::Unknown);
        assert_eq!(choice("\u{feff}3\r\n"), Choice::Check);
    }

    #[test]
    fn the_last_game_is_remembered_while_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_last(dir.path()), None);
        let exe = dir.path().join("title.exe");
        std::fs::write(&exe, b"").unwrap();
        save_last(dir.path(), &exe);
        assert_eq!(load_last(dir.path()), Some(exe.clone()));
        std::fs::remove_file(&exe).unwrap();
        assert_eq!(load_last(dir.path()), None, "a moved game is not offered");
    }

    #[test]
    fn the_check_says_what_matters_on_windows_10_and_nvidia() {
        let pc = Pc {
            windows_build: Some(19045),
            avx2: true,
            sm66: Some(true),
            mesh_shaders: Some(false),
            max_lanes: Some(32),
            agility_beside: false,
        };
        let v = verdicts(&pc);
        let find = |what: &str| v.iter().find(|l| l.1.starts_with(what)).unwrap();
        assert_eq!(find("Windows 10").0, Status::Warn);
        assert!(find("Windows 10").2.as_deref().unwrap().contains("Agility"));
        assert_eq!(find("CPU has AVX2").0, Status::Good);
        assert_eq!(find("GPU: no mesh shaders").0, Status::Warn);
        assert_eq!(find("GPU: waves of 32").0, Status::Warn);
        // Windows 11, no SM 6.6: nothing about mesh or waves (not asked).
        let pc = Pc {
            windows_build: Some(26100),
            sm66: Some(false),
            ..pc
        };
        let v = verdicts(&pc);
        assert_eq!(v[0].0, Status::Good);
        assert!(v.iter().any(|l| l.0 == Status::Bad && l.1.contains("6.6")));
        assert!(
            !v.iter()
                .any(|l| l.1.contains("mesh") || l.1.contains("waves"))
        );
    }

    #[test]
    fn colours_follow_the_sunset_and_stay_out_when_off() {
        assert_eq!(along(&SUNSET, 0.0), SUNSET[0]);
        assert_eq!(along(&SUNSET, 1.0), SUNSET[5]);
        let mid = along(&[(0, 0, 0), (200, 100, 50)], 0.5);
        assert_eq!(mid, (100, 50, 25));
        let plain = Paint { color: false };
        assert_eq!(plain.gradient("VICTORY", &SUNSET), "VICTORY");
        assert!(!menu_text(plain, None).contains('\x1b'));
        let colored = Paint { color: true };
        assert!(
            colored
                .gradient("VI", &SUNSET)
                .contains("\x1b[38;2;86;98;196mV")
        );
    }

    #[test]
    fn the_menu_offers_run_again_only_with_a_last_game() {
        let p = Paint { color: false };
        assert!(menu_text(p, None).contains("(nothing run yet)"));
        let text = menu_text(p, Some(Path::new(r"D:\Games\title.exe")));
        assert!(text.contains(r"Run again   D:\Games\title.exe"));
    }
}
