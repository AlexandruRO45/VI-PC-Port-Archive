//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! A variable that is set but empty counts as unset. Relative paths in the
//! config file are relative to the file, not to the working directory;
//! relative paths in variables are relative to the working directory. Both
//! are made absolute when loaded. A
//! leading `~` in a path (from the file or a variable) is the home directory,
//! and an error when there is none. A near miss of one of the variables
//! (`VICTORY_GAMEDIR`, `VICTORY_DATA_PATH`) is an error while the real one
//! is unset.
//! See `victory.example.toml` at the repository root.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::ensure_outside_repo;

#[derive(Debug, Default, Clone)]
pub struct Config {
    /// The user's dump of Grand Theft Auto VI.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
    /// Where each value above came from.
    pub origins: Origins,
}

/// Where a configured value came from, so `victory config` and errors can
/// name the setting to change instead of leaving the user to guess which of
/// the variable, the file and the default won.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Not set anywhere: the built-in default (or nothing, for `game_dir`).
    #[default]
    Default,
    /// The `[paths]` table of the config file.
    File,
    /// The named environment variable.
    Env(&'static str),
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Origin::Default => f.write_str("default"),
            Origin::File => f.write_str("config file"),
            Origin::Env(name) => f.write_str(name),
        }
    }
}

/// The [`Origin`] of each [`Config`] value.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Origins {
    pub game_dir: Origin,
    pub cache_dir: Origin,
    pub data_dir: Origin,
    pub test_titles: Origin,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
}

impl FileConfig {
    /// The `[paths]` table with every relative path made relative to `dir`,
    /// the directory holding the config file. With `VICTORY_CONFIG` pointing
    /// elsewhere, `cache_dir = "cache"` then means the folder next to that
    /// file, not one under whatever directory the tool was started from
    /// (often the repository, which the asset policy refuses). A path that
    /// starts with `~` is under `home` instead, when there is one. A Windows
    /// drive path on another host is refused (see [`ensure_native`]) rather
    /// than anchored here as a relative path, and so is a path holding a
    /// control character (see [`ensure_no_control`]).
    ///
    /// An empty value (`game_dir = ""`, a blank a template leaves to fill
    /// in, or `test_titles = [""]`) counts as unset, like an empty variable.
    /// Anchored, it would be the config file's own folder (often the
    /// repository), or with `./victory.toml` the empty path, which reads
    /// and writes the working directory.
    fn anchored_at(self, dir: &Path, home: Option<&Path>) -> Result<FilePaths> {
        let anchor = |key: &str, p: PathBuf| -> Result<PathBuf> {
            check_setting(&format!("[paths].{key}"), &p)?;
            let p = expand_home(&format!("[paths].{key}"), p, home)?;
            Ok(if p.is_relative() { dir.join(p) } else { p })
        };
        let set = |p: &PathBuf| !p.as_os_str().is_empty();
        let paths = self.paths;
        Ok(FilePaths {
            game_dir: paths
                .game_dir
                .filter(set)
                .map(|p| anchor("game_dir", p))
                .transpose()?,
            cache_dir: paths
                .cache_dir
                .filter(set)
                .map(|p| anchor("cache_dir", p))
                .transpose()?,
            data_dir: paths
                .data_dir
                .filter(set)
                .map(|p| anchor("data_dir", p))
                .transpose()?,
            test_titles: paths
                .test_titles
                .into_iter()
                .filter(set)
                .map(|p| anchor("test_titles", p))
                .collect::<Result<_>>()?,
        })
    }
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let env = |k: &str| std::env::var_os(k);
        ensure_no_misspelled_variables(std::env::vars_os().map(|(k, _)| k), env)?;
        let home = home_dir(env);
        let file = match config_file(explicit_config(env, home.as_deref())?)? {
            Some(path) => {
                let text = read_config(&path)?;
                Some((path, text))
            }
            None => None,
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed =
                    parse_file(&text).with_context(|| format!("parsing {}", path.display()))?;
                let dir = path.parent().unwrap_or(Path::new("")).to_path_buf();
                let paths = parsed
                    .anchored_at(&dir, home.as_deref())
                    .with_context(|| format!("in {}", path.display()))?;
                (Some(path), paths)
            }
            None => (None, FilePaths::default()),
        };
        let cwd = std::env::current_dir().context("reading the working directory")?;
        Self::resolve_in(parsed, source, env, Some(&cwd))
    }

    /// [`Config::resolve_in`] with relative paths kept as written.
    #[cfg(test)]
    fn resolve(
        file: FilePaths,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        Self::resolve_in(file, source, env, None)
    }

    /// Resolves every value. A path still relative at the end (from a
    /// variable such as `VICTORY_DATA_DIR=data`, or from `./victory.toml`,
    /// whose folder is the empty path) is relative to the working directory,
    /// so it is joined to `cwd` when one is given. Kept relative, it would
    /// slip past the folder checks below, which compare paths as written:
    /// `VICTORY_CACHE_DIR=/home/me/victory` with `VICTORY_DATA_DIR=data`
    /// started from `/home/me/victory` puts the saves inside the cache, and
    /// `victory config` would show `data` rather than the folder used. It
    /// would also move if the working directory later changed.
    fn resolve_in(
        file: FilePaths,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
        cwd: Option<&Path>,
    ) -> Result<Self> {
        let absolute = |p: PathBuf| match cwd {
            Some(cwd) if p.is_relative() => cwd.join(p),
            _ => p,
        };
        // A quoted `VICTORY_GAME_DIR="~/games"` reaches us unexpanded.
        let home = home_dir(&env);
        let expand = |k: &str, p: PathBuf| expand_home(k, p, home.as_deref());
        // The variable, else the file, with where the value came from.
        let pick = |k: &'static str, from_file: Option<PathBuf>| -> Result<_> {
            Ok(match env(k).filter(|v| !v.is_empty()) {
                Some(v) => {
                    let p = PathBuf::from(v);
                    check_setting(k, &p)?;
                    (Some(expand(k, p)?), Origin::Env(k))
                }
                None if from_file.is_some() => (from_file, Origin::File),
                None => (None, Origin::Default),
            })
        };
        let (test_titles, test_titles_origin) =
            match env("VICTORY_TEST_TITLES").filter(|v| !v.is_empty()) {
                Some(v) => {
                    // Checked before splitting: on this host `:` would cut
                    // `D:\a` in two.
                    check_setting("VICTORY_TEST_TITLES", Path::new(&v))?;
                    let titles: Vec<PathBuf> = std::env::split_paths(&v)
                        .filter(|p| !p.as_os_str().is_empty())
                        .collect();
                    // `/a: /b` passes the check above as a whole.
                    for title in &titles {
                        ensure_trimmed("VICTORY_TEST_TITLES", title)?;
                        ensure_one_title(title, cfg!(windows))?;
                    }
                    (
                        titles
                            .into_iter()
                            .map(|p| expand("VICTORY_TEST_TITLES", p))
                            .collect::<Result<_>>()?,
                        Origin::Env("VICTORY_TEST_TITLES"),
                    )
                }
                None if file.test_titles.is_empty() => (file.test_titles, Origin::Default),
                None => (file.test_titles, Origin::File),
            };
        let test_titles: Vec<PathBuf> = test_titles.into_iter().map(absolute).collect();
        let (game_dir, game_origin) = pick("VICTORY_GAME_DIR", file.game_dir)?;
        let game_dir = game_dir.map(absolute);
        let (cache_dir, cache_origin) = pick("VICTORY_CACHE_DIR", file.cache_dir)?;
        let cache_dir = absolute(cache_dir.map_or_else(|| default_dir(Kind::Cache, &env), Ok)?);
        let (data_dir, data_origin) = pick("VICTORY_DATA_DIR", file.data_dir)?;
        let data_dir = absolute(data_dir.map_or_else(|| default_dir(Kind::Data, &env), Ok)?);
        let origins = Origins {
            game_dir: game_origin,
            cache_dir: cache_origin,
            data_dir: data_origin,
            test_titles: test_titles_origin,
        };
        for (name, dir, origin, setting) in [
            (
                "cache",
                &cache_dir,
                origins.cache_dir,
                "VICTORY_CACHE_DIR or [paths].cache_dir",
            ),
            (
                "data",
                &data_dir,
                origins.data_dir,
                "VICTORY_DATA_DIR or [paths].data_dir",
            ),
        ] {
            ensure_own_folder(name, dir, home.as_deref(), origin, setting)?;
        }
        ensure_data_outside_cache(&cache_dir, &data_dir, &origins)?;
        ensure_data_outside_user_cache(&data_dir, origins.data_dir, &env)?;
        for (name, dir, origin, setting) in [
            (
                "cache",
                &cache_dir,
                origins.cache_dir,
                "VICTORY_CACHE_DIR or [paths].cache_dir",
            ),
            (
                "data",
                &data_dir,
                origins.data_dir,
                "VICTORY_DATA_DIR or [paths].data_dir",
            ),
        ] {
            ensure_not_user_base(name, dir, origin, setting, &env)?;
        }
        if let Some(game) = &game_dir {
            ensure_game_apart(game, &cache_dir, &data_dir, &origins)?;
        }
        ensure_titles_apart(&test_titles, &cache_dir, &data_dir, &origins)?;
        ensure_titles_distinct(game_dir.as_deref(), &test_titles, &origins)?;
        for (name, dir, origin, setting) in [
            (
                "game",
                game_dir.as_deref(),
                origins.game_dir,
                "VICTORY_GAME_DIR or [paths].game_dir",
            ),
            (
                "cache",
                Some(&*cache_dir),
                origins.cache_dir,
                "VICTORY_CACHE_DIR or [paths].cache_dir",
            ),
            (
                "data",
                Some(&*data_dir),
                origins.data_dir,
                "VICTORY_DATA_DIR or [paths].data_dir",
            ),
        ] {
            if let Some(dir) = dir {
                ensure_directory_path(&format!("{name} folder"), dir, origin, setting)?;
            }
        }
        for title in &test_titles {
            ensure_directory_path(
                "test title",
                title,
                origins.test_titles,
                "VICTORY_TEST_TITLES or [paths].test_titles",
            )?;
        }
        Ok(Config {
            game_dir,
            cache_dir,
            data_dir,
            test_titles,
            source,
            origins,
        })
    }

    /// The game directory, or a helpful error if it is not configured.
    pub fn require_game_dir(&self) -> Result<&Path> {
        self.game_dir.as_deref().context(
            "no game directory configured: set VICTORY_GAME_DIR or [paths].game_dir in \
             victory.toml (see victory.example.toml)",
        )
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.data_dir, sub)
    }
}

/// The keys of the `[paths]` table.
const PATH_KEYS: [&str; 4] = ["game_dir", "cache_dir", "data_dir", "test_titles"];

/// Parses a config file. A `[paths]` key written at the top level, above
/// the `[paths]` header or in a file without one (`game_dir = '...'` on its
/// own is an easy first file to write), is otherwise reported only as
/// "unknown field `game_dir`, expected `paths`", which does not say what
/// to change; the error names the key and the table it belongs in.
/// The text of the config file at `path`, found by [`config_file`]. A file
/// that is there but cannot be read (no read permission, or a folder
/// swapped in after the check) and text that is not UTF-8 are both reported
/// with the path, so it is clear which file to fix.
fn read_config(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    config_text(bytes).with_context(|| format!("reading {}", path.display()))
}

fn parse_file(text: &str) -> Result<FileConfig> {
    let err = match toml::from_str(text) {
        Ok(parsed) => return Ok(parsed),
        Err(e) => e,
    };
    let misplaced: Vec<&str> = text
        .parse::<toml::Table>()
        .map(|top| {
            PATH_KEYS
                .into_iter()
                .filter(|k| top.contains_key(*k))
                .collect()
        })
        .unwrap_or_default();
    if misplaced.is_empty() {
        return Err(err.into());
    }
    anyhow::bail!(
        "{} must be under the [paths] table: put a `[paths]` line above {} \
         (see victory.example.toml)",
        misplaced.join(", "),
        if misplaced.len() == 1 { "it" } else { "them" }
    );
}

/// The text of a config file. Windows editors and shells save text in ways
/// `read_to_string` and the TOML parser reject with messages that do not
/// say what to change: Notepad's "UTF-8 with BOM" puts U+FEFF before the
/// first key, and Windows PowerShell 5's `>` / `Out-File` (or Notepad's
/// "Unicode") writes UTF-16, reported only as "stream did not contain valid
/// UTF-8". A UTF-8 byte order mark is dropped (it is not part of the
/// text); UTF-16, with or without a mark, and other non-UTF-8 bytes are
/// refused with the fix.
fn config_text(bytes: Vec<u8>) -> Result<String> {
    const SAVE_AS: &str = "save it as UTF-8 (in Notepad: Save As, Encoding: UTF-8; in \
                           PowerShell: Set-Content -Encoding utf8)";
    let utf16 = match bytes.as_slice() {
        [0xFF, 0xFE, ..] | [0xFE, 0xFF, ..] => true,
        // ASCII text in UTF-16 without a mark: every other byte is zero.
        [a, 0, ..] | [0, a, ..] if a.is_ascii() && *a != 0 => true,
        _ => false,
    };
    if utf16 {
        anyhow::bail!("the file is UTF-16 text, not UTF-8: {SAVE_AS}");
    }
    let bytes = match bytes.strip_prefix(b"\xEF\xBB\xBF") {
        Some(rest) => rest.to_vec(),
        None => bytes,
    };
    String::from_utf8(bytes).map_err(|e| {
        let at = e.utf8_error().valid_up_to();
        let line = 1 + e.as_bytes()[..at].iter().filter(|&&b| b == b'\n').count();
        anyhow::anyhow!(
            "line {line} is not UTF-8 text (byte {at}, perhaps a path typed in another \
             encoding): {SAVE_AS}"
        )
    })
}

/// The variables [`Config::load`] reads, by the name a misspelling of them
/// is compared with.
const SETTINGS: [&str; 5] = [
    "VICTORY_GAME_DIR",
    "VICTORY_CACHE_DIR",
    "VICTORY_DATA_DIR",
    "VICTORY_TEST_TITLES",
    "VICTORY_CONFIG",
];

/// Refuses a near miss of one of the [`SETTINGS`] variables while the real
/// one is unset or empty. `VICTORY_GAMEDIR`, `VICTORY_DATA_PATH` or a
/// lowercase `victory_cache_dir` is otherwise ignored without a word, and the
/// default is used: saves land in `~/.local/share/victory` rather than the
/// folder the user named, and the game folder is reported as not set.
/// `names` are the names of the variables in the environment.
fn ensure_no_misspelled_variables(
    names: impl IntoIterator<Item = std::ffi::OsString>,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<()> {
    for name in names {
        // A name that is not Unicode cannot be one of ours.
        let Some(name) = name.to_str() else { continue };
        let Some(meant) = misspelling_of(name) else {
            continue;
        };
        if env(meant).is_some_and(|v| !v.is_empty()) {
            // The real one wins anyway; the other is someone else's.
            continue;
        }
        anyhow::bail!(
            "{name} is set, but Victory reads {meant}, which is not set; rename {name} to {meant}"
        );
    }
    Ok(())
}

/// The [`SETTINGS`] variable `name` looks like a misspelling of: the same
/// letters ignoring case, `_` and `-` (`VICTORY_GAMEDIR`,
/// `victory_cache_dir`), or a `_DIR` variable with the suffix dropped or
/// spelt `_PATH`, `_DIRECTORY` or `_FOLDER` (`VICTORY_DATA`,
/// `VICTORY_GAME_PATH`), `VICTORY_TEST_TITLE`, `VICTORY_TITLES`, or
/// `VICTORY_CONFIG` with `_FILE` or `_PATH` (`VICTORY_CONFIG_FILE`). `None`
/// for the real names (which also match case-insensitively on Windows,
/// where variable names are) and for everything else.
fn misspelling_of(name: &str) -> Option<&'static str> {
    let squash = |s: &str| -> String {
        s.chars()
            .filter(|c| !matches!(c, '_' | '-'))
            .map(|c| c.to_ascii_uppercase())
            .collect()
    };
    let got = squash(name);
    SETTINGS.into_iter().find(|&real| {
        if name == real || (cfg!(windows) && name.eq_ignore_ascii_case(real)) {
            return false;
        }
        let near: Vec<String> = match real.strip_suffix("_DIR") {
            Some(stem) => ["", "_PATH", "_DIRECTORY", "_FOLDER"]
                .iter()
                .map(|suffix| format!("{stem}{suffix}"))
                .collect(),
            None if real == "VICTORY_TEST_TITLES" => {
                vec!["VICTORY_TEST_TITLE".into(), "VICTORY_TITLES".into()]
            }
            None => vec![format!("{real}_FILE"), format!("{real}_PATH")],
        };
        std::iter::once(real.to_owned())
            .chain(near)
            .any(|candidate| squash(&candidate) == got)
    })
}

/// `VICTORY_CONFIG`, empty counting as unset, with a leading `~` as the
/// home directory like every other path variable: a quoted
/// `VICTORY_CONFIG="~/victory.toml"` reaches us unexpanded and would
/// otherwise be reported as not existing. Checked like the other path
/// variables (see [`check_setting`]).
fn explicit_config(
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
    home: Option<&Path>,
) -> Result<Option<PathBuf>> {
    env("VICTORY_CONFIG")
        .filter(|v| !v.is_empty())
        .map(|v| {
            let p = PathBuf::from(v);
            check_setting("VICTORY_CONFIG", &p)?;
            expand_home("VICTORY_CONFIG", p, home)
        })
        .transpose()
}

/// The config file to read: `explicit` (`VICTORY_CONFIG`), which must be a
/// file, else `./victory.toml` when that is a file.
fn config_file(explicit: Option<PathBuf>) -> Result<Option<PathBuf>> {
    config_file_or(explicit, Path::new("victory.toml"))
}

/// [`config_file`] with `default` in place of `./victory.toml`. A missing
/// `default` is no config file, but one that is there and cannot be read
/// as one is refused rather than skipped: a `victory.toml` folder, a
/// `victory.toml` link whose target was moved or deleted, or one that
/// cannot be checked (no permission) would otherwise be passed over without
/// a word, and every setting in it silently replaced by its default.
fn config_file_or(explicit: Option<PathBuf>, default: &Path) -> Result<Option<PathBuf>> {
    let (path, what, checking) = match explicit {
        Some(path) => {
            let what = format!("VICTORY_CONFIG points at {}", path.display());
            let checking = format!("checking VICTORY_CONFIG at {}", path.display());
            (path, what, checking)
        }
        None => {
            let path = default.to_path_buf();
            if std::fs::symlink_metadata(&path)
                .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
            {
                return Ok(None);
            }
            let what = format!("the config file {}", path.display());
            let checking = format!("checking the config file {}", path.display());
            (path, what, checking)
        }
    };
    // The path with a trailing separator and interior `.` dropped, as in
    // [`ensure_directory_path`]: the OS resolves `victory.toml/` as a
    // folder, so a file named that way is only "Not a directory" and a link
    // that leads nowhere is not seen as a link at all.
    let plain: PathBuf = path.components().collect();
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        // `path` itself failed, so `plain` is only a file when the spelling
        // differs (`Path` equality ignores a trailing slash).
        Err(_) if std::fs::metadata(&plain).is_ok_and(|m| m.is_file()) => {
            anyhow::bail!(
                "{what}, a file named with a trailing slash or `/.`, which makes it a folder; \
                 drop it: {}",
                plain.display()
            );
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            // A link to a file that was moved or deleted: the link itself
            // is there, so say where it leads.
            if let Ok(target) = std::fs::read_link(&plain) {
                anyhow::bail!(
                    "{what}, a link to {}, which does not exist",
                    target.display()
                );
            }
            anyhow::bail!("{what}, which does not exist");
        }
        Err(err) => {
            return Err(err).context(checking);
        }
    };
    if metadata.is_file() {
        return Ok(Some(path));
    }
    if metadata.is_dir() {
        // An easy slip: the folder holding victory.toml instead of the file.
        let inside = path.join("victory.toml");
        let hint = if inside.is_file() {
            format!("; did you mean {}?", inside.display())
        } else {
            String::new()
        };
        anyhow::bail!("{what}, which is a folder, not a config file{hint}");
    }
    anyhow::bail!("{what}, which is not a regular file");
}

/// Refuses a cache or data folder that is the root of a drive, the home
/// directory, or a folder holding it (`/home`). The cache may be deleted at
/// any time, so clearing a cache of `~` would take the whole home folder
/// with it, and a data folder of `~` mixes saves, settings and crashes.log
/// into it. It is an easy slip with a variable that expanded to nothing:
/// `VICTORY_CACHE_DIR=$HOME/$APP` with `APP` unset is `~/`, and
/// `VICTORY_DATA_DIR=$XDG_DATA_HOME/` with `XDG_DATA_HOME` unset is `/`.
/// Compared lexically like [`ensure_data_outside_cache`]. `name` is the
/// folder ("cache", "data") and `setting` the variable and key that set it.
fn ensure_own_folder(
    name: &str,
    dir: &Path,
    home: Option<&Path>,
    origin: Origin,
    setting: &str,
) -> Result<()> {
    let folded = lexical(dir);
    let what = if folded.parent().is_none() {
        "the root of the drive"
    } else if home.is_some_and(|h| is_within(h, &folded, FOLD_CASE)) {
        if home.is_some_and(|h| is_within(&folded, h, FOLD_CASE)) {
            "your home folder"
        } else {
            "a folder holding your home folder"
        }
    } else {
        return Ok(());
    };
    anyhow::bail!(
        "the {name} folder {} (from {origin}) is {what}; set {setting} to a folder used \
         only by Victory, or leave it unset for the default",
        dir.display()
    );
}

/// The data folder holds what titles must keep and the cache may be deleted
/// at any time, so data inside (or equal to) the cache would be lost with it.
/// Easy to set up by accident, e.g. `VICTORY_CACHE_DIR=~/victory` with
/// `VICTORY_DATA_DIR=~/victory/data`. Compared without touching the disk
/// (no symlinks followed; the folders need not exist yet), but with `.` and
/// `..` folded away (see [`lexical`]): otherwise `~/.cache/victory/../data`
/// would be refused although it is a sibling of the cache, and a cache of
/// `~/victory/tmp/..` would not be seen to hold `~/victory/data`. The error
/// says where each folder came from, so it is clear which setting caused it.
fn ensure_data_outside_cache(cache_dir: &Path, data_dir: &Path, origins: &Origins) -> Result<()> {
    if is_within(data_dir, cache_dir, FOLD_CASE) {
        if origins.cache_dir == Origin::Default && origins.data_dir == Origin::Default {
            // Both defaults: XDG_CACHE_HOME and XDG_DATA_HOME (or one under
            // the other) are the cause, and the generic advice would not
            // say so.
            anyhow::bail!(
                "the default data folder {} is inside the default cache folder {}, which \
                 may be deleted at any time, because XDG_DATA_HOME is inside (or the same \
                 as) XDG_CACHE_HOME; point them at different folders, or set \
                 VICTORY_DATA_DIR or [paths].data_dir to a folder outside the cache",
                data_dir.display(),
                cache_dir.display()
            );
        }
        anyhow::bail!(
            "the data folder {} (from {}) is inside the cache folder {} (from {}), which may \
             be deleted at any time; set VICTORY_DATA_DIR or [paths].data_dir to a folder \
             outside it",
            data_dir.display(),
            origins.data_dir,
            cache_dir.display(),
            origins.cache_dir
        );
    }
    Ok(())
}

/// [`ensure_data_outside_cache`] for the user's whole cache, not only
/// Victory's: `$XDG_CACHE_HOME` (else `~/.cache`) is what desktop cleaners
/// and users empty, so data under it is lost with it even when it is not
/// inside Victory's cache folder. The easy slips are
/// `VICTORY_DATA_DIR=~/.cache/victory-saves`, and `XDG_DATA_HOME` set
/// under `XDG_CACHE_HOME` (`~/.cache/share`), which moves the default data
/// folder there although the cache folder is a sibling. Skipped when that
/// base is the home folder or holds it (`XDG_CACHE_HOME=$HOME`): then it is
/// not a folder anyone clears. Compared lexically, without the disk.
fn ensure_data_outside_user_cache(
    data_dir: &Path,
    origin: Origin,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<()> {
    let absolute = |k: &str| env(k).map(PathBuf::from).filter(|p| p.is_absolute());
    let home = absolute("HOME");
    let (base, named) = match absolute("XDG_CACHE_HOME") {
        Some(base) => (base, "XDG_CACHE_HOME"),
        None => match &home {
            Some(home) => (home.join(".cache"), "~/.cache"),
            None => return Ok(()),
        },
    };
    if !is_within(data_dir, &base, FOLD_CASE)
        || home.is_some_and(|h| is_within(&h, &base, FOLD_CASE))
    {
        return Ok(());
    }
    if origin == Origin::Default {
        anyhow::bail!(
            "the default data folder {} is inside your cache folder {} ({named}), which \
             may be deleted at any time, because XDG_DATA_HOME is inside it; point \
             XDG_DATA_HOME elsewhere, or set VICTORY_DATA_DIR or [paths].data_dir to a \
             folder outside the cache",
            data_dir.display(),
            base.display()
        );
    }
    anyhow::bail!(
        "the data folder {} (from {origin}) is inside your cache folder {} ({named}), which \
         may be deleted at any time; set VICTORY_DATA_DIR or [paths].data_dir to a folder \
         outside it, such as one under ~/.local/share",
        data_dir.display(),
        base.display()
    );
}

/// The folders every program shares: the XDG base directories
/// (`$XDG_CACHE_HOME`, `$XDG_DATA_HOME`, `$XDG_CONFIG_HOME`,
/// `$XDG_STATE_HOME`), their defaults under the home folder (`~/.cache`,
/// `~/.local/share`, `~/.local/state`, `~/.config`, and `~/.local` holding
/// two of them), and on Windows `%LOCALAPPDATA%` and `%APPDATA%`. Each is
/// named by the variable or path the user would recognise. Values that are
/// unset, empty or relative are skipped, as [`default_dir`] skips them.
fn user_bases(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Vec<(PathBuf, String)> {
    let absolute = |k: &str| env(k).map(PathBuf::from).filter(|p| p.is_absolute());
    let mut bases: Vec<(PathBuf, String)> = [
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "LOCALAPPDATA",
        "APPDATA",
    ]
    .into_iter()
    .filter_map(|k| absolute(k).map(|p| (p, k.to_owned())))
    .collect();
    if let Some(home) = absolute("HOME") {
        for rel in [
            ".cache",
            ".local/share",
            ".local/state",
            ".config",
            ".local",
        ] {
            bases.push((home.join(rel), format!("~/{rel}")));
        }
    }
    bases
}

/// Refuses a cache or data folder that is one of the [`user_bases`], or a
/// folder holding one, rather than a folder of Victory's own inside it.
/// The easy slip is leaving off the last part: `VICTORY_CACHE_DIR=~/.cache`
/// or `VICTORY_DATA_DIR=$XDG_DATA_HOME`. Victory's cache may be deleted at
/// any time, so clearing a cache of `~/.cache` would empty every program's
/// cache, and a data folder of `~/.local/share` mixes saves and crashes.log
/// in with every other program's files. Run after [`ensure_own_folder`],
/// which refuses the home folder and the root, and compared like it:
/// lexically, without the disk.
fn ensure_not_user_base(
    name: &str,
    dir: &Path,
    origin: Origin,
    setting: &str,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<()> {
    for (base, named) in user_bases(env) {
        if !is_within(&base, dir, FOLD_CASE) {
            continue;
        }
        let what = if is_within(dir, &base, FOLD_CASE) {
            "is"
        } else {
            "holds"
        };
        anyhow::bail!(
            "the {name} folder {} (from {origin}) {what} your shared {named} folder {}, \
             used by every program; set {setting} to a folder used only by Victory, such \
             as {}, or leave it unset for the default",
            dir.display(),
            base.display(),
            base.join("victory").display()
        );
    }
    Ok(())
}

/// The game folder is the user's own dump: Victory reads it and never
/// writes there, and it must survive the cache being cleared. So it may not
/// be inside (or equal to) the cache folder, where clearing the cache would
/// delete it, and neither the cache nor the data folder may be inside it,
/// where derived files and saves would be mixed into the dump. Each is an
/// easy slip with one parent folder for everything, e.g.
/// `VICTORY_CACHE_DIR=~/gta` with `VICTORY_GAME_DIR=~/gta/game`, or
/// `[paths].cache_dir = "cache"` in a `victory.toml` kept inside the dump.
/// Compared like [`ensure_data_outside_cache`]: lexically, without the disk.
fn ensure_game_apart(game: &Path, cache: &Path, data: &Path, origins: &Origins) -> Result<()> {
    if is_within(game, cache, FOLD_CASE) {
        anyhow::bail!(
            "the game folder {} (from {}) is inside the cache folder {} (from {}), which may \
             be deleted at any time; set VICTORY_CACHE_DIR or [paths].cache_dir to a folder \
             outside the game folder",
            game.display(),
            origins.game_dir,
            cache.display(),
            origins.cache_dir
        );
    }
    for (name, dir, origin, setting) in [
        (
            "cache",
            cache,
            origins.cache_dir,
            "VICTORY_CACHE_DIR or [paths].cache_dir",
        ),
        (
            "data",
            data,
            origins.data_dir,
            "VICTORY_DATA_DIR or [paths].data_dir",
        ),
    ] {
        if is_within(dir, game, FOLD_CASE) {
            anyhow::bail!(
                "the {name} folder {} (from {origin}) is inside the game folder {} (from {}), \
                 which holds your dump and is only read; set {setting} to a folder outside it",
                dir.display(),
                game.display(),
                origins.game_dir
            );
        }
    }
    Ok(())
}

/// Each of the test titles is, like the game folder, a dump that Victory
/// only reads, so [`ensure_game_apart`] applies to them too: a title inside
/// (or equal to) the cache folder would be deleted with the cache, and a
/// cache or data folder inside a title would mix derived files and saves
/// into that dump. The easy slip is one parent folder for everything, e.g.
/// `VICTORY_CACHE_DIR=~/xbox` with `VICTORY_TEST_TITLES=~/xbox/title-a`.
/// The error names the title, so it is clear which entry of the list is
/// meant. Compared lexically, without the disk.
fn ensure_titles_apart(
    titles: &[PathBuf],
    cache: &Path,
    data: &Path,
    origins: &Origins,
) -> Result<()> {
    for title in titles {
        if is_within(title, cache, FOLD_CASE) {
            anyhow::bail!(
                "the test title {} (from {}) is inside the cache folder {} (from {}), which \
                 may be deleted at any time; set VICTORY_CACHE_DIR or [paths].cache_dir to a \
                 folder outside the test titles",
                title.display(),
                origins.test_titles,
                cache.display(),
                origins.cache_dir
            );
        }
        for (name, dir, origin, setting) in [
            (
                "cache",
                cache,
                origins.cache_dir,
                "VICTORY_CACHE_DIR or [paths].cache_dir",
            ),
            (
                "data",
                data,
                origins.data_dir,
                "VICTORY_DATA_DIR or [paths].data_dir",
            ),
        ] {
            if is_within(dir, title, FOLD_CASE) {
                anyhow::bail!(
                    "the {name} folder {} (from {origin}) is inside the test title {} (from \
                     {}), which is a dump that is only read; set {setting} to a folder \
                     outside it",
                    dir.display(),
                    title.display(),
                    origins.test_titles
                );
            }
        }
    }
    Ok(())
}

/// Refuses a test title listed twice, or one that is the game folder.
/// `victory census`, `translate` and `shader` take the game folder followed
/// by the test titles when given no paths, so a repeated dump was scanned
/// and translated twice, doubling the time and the report for nothing. The
/// easy slips are a title added both to `[paths].test_titles` and, later,
/// to a copied `VICTORY_TEST_TITLES` line, the same folder written two ways
/// (`/x/a` and `/x/a/`, or `C:\Xbox\A` and `c:\xbox\a` on Windows), or
/// the game folder pasted into the list of other titles. Compared like the
/// other folder checks: lexically, without the disk. A title inside another
/// title or inside the game folder is left alone: it is a different folder.
fn ensure_titles_distinct(
    game: Option<&Path>,
    titles: &[PathBuf],
    origins: &Origins,
) -> Result<()> {
    let same = |a: &Path, b: &Path| is_within(a, b, FOLD_CASE) && is_within(b, a, FOLD_CASE);
    for (i, title) in titles.iter().enumerate() {
        if let Some(game) = game.filter(|game| same(title, game)) {
            anyhow::bail!(
                "the test title {} (from {}) is the game folder {} (from {}); the test titles \
                 are other dumps, and the game folder is already used, so remove it from \
                 VICTORY_TEST_TITLES or [paths].test_titles",
                title.display(),
                origins.test_titles,
                game.display(),
                origins.game_dir
            );
        }
        if let Some(first) = titles[..i].iter().find(|earlier| same(title, earlier)) {
            anyhow::bail!(
                "the test title {} (from {}) is listed twice (also as {}); remove one from \
                 VICTORY_TEST_TITLES or [paths].test_titles",
                title.display(),
                origins.test_titles,
                first.display()
            );
        }
    }
    Ok(())
}

/// Refuses a folder setting that names or descends from a non-directory. The easy slip is
/// the game's executable or a file in the dump rather than its folder
/// (`VICTORY_GAME_DIR=~/games/gta/default.xex`, picked in a file dialog or
/// pasted from a launcher's target); it was accepted and only failed later,
/// per file, with "Not a directory". The same goes for each entry of
/// `VICTORY_TEST_TITLES` / `[paths].test_titles`, which is also a dump.
/// For the cache or data folder `create_dir_all` would fail on first use
/// with "File exists", naming neither the setting nor the fix; the same goes
/// for a link (the folder or one above it) whose target was moved or deleted,
/// or that loops, so it is refused too. A path that
/// does not exist yet is fine: the cache and data folders are created when
/// needed. `what` is the folder ("game folder", "cache folder", "data
/// folder", "test title") and `setting` the variable and key that set it.
fn ensure_directory_path(what: &str, dir: &Path, origin: Origin, setting: &str) -> Result<()> {
    // A missing leaf can be created later, but only if its first existing
    // ancestor is a directory. Follow symlinks just as create_dir_all does,
    // but stop at a link that leads nowhere (its target moved or deleted, or
    // a loop): it is there, so create_dir_all cannot make a folder in its
    // place and would fail with "File exists" on first use.
    //
    // Walk the path with trailing separators and interior `.` dropped
    // (`Path::components` does both, and keeps `..`): the OS resolves a
    // link named with a trailing slash (`lstat("link/")` follows it), so
    // `dangling/` or `loop/` would never be seen as a link, and
    // `Path::ancestors` goes straight from `default.xex/` to its parent, so
    // a file named with a trailing slash would pass as a folder to create.
    let walk: PathBuf = dir.components().collect();
    let existing = walk.ancestors().find_map(|p| match std::fs::metadata(p) {
        Ok(m) => Some(Ok((p, m))),
        Err(_) => std::fs::symlink_metadata(p)
            .is_ok_and(|m| m.file_type().is_symlink())
            .then_some(Err(p)),
    });
    let (path, metadata) = match existing {
        None => return Ok(()),
        Some(Ok(found)) => found,
        Some(Err(link)) => {
            let target = std::fs::read_link(link)
                .map(|t| format!(" to {}", t.display()))
                .unwrap_or_default();
            let place = if link == walk {
                String::new()
            } else {
                format!(" cannot be a folder because {}", link.display())
            };
            anyhow::bail!(
                "the {what} {} (from {origin}){place} is a link{target} that leads nowhere \
                 (its target was moved or deleted, or the link loops); \
                 fix or remove the link, or set {setting} elsewhere",
                dir.display()
            );
        }
    };
    if metadata.is_dir() {
        return Ok(());
    }
    if path != walk {
        anyhow::bail!(
            "the {what} {} (from {origin}) cannot be a folder because {} is not a directory; \
             set {setting} to a path under a directory",
            dir.display(),
            path.display()
        );
    }
    if !metadata.is_file() {
        anyhow::bail!(
            "the {what} {} (from {origin}) is not a directory; set {setting} to a folder",
            dir.display()
        );
    }
    let parent = walk.parent().filter(|p| !p.as_os_str().is_empty());
    let hint = match (what, parent) {
        ("game folder" | "test title", Some(parent)) => format!(
            "set {setting} to the folder holding the dump, perhaps {}",
            parent.display()
        ),
        _ => format!("set {setting} to a folder, or move the file away"),
    };
    anyhow::bail!(
        "the {what} {} (from {origin}) is a file, not a folder; {hint}",
        dir.display()
    );
}

/// Whether folder names differ only by case on the usual file systems of
/// this host: NTFS (Windows) and APFS (macOS) ignore case by default, so
/// there `C:\Victory\cache` holds `c:\victory\cache\data`.
const FOLD_CASE: bool = cfg!(any(windows, target_os = "macos"));

/// Whether `path` is `parent` or inside it, compared like
/// [`ensure_data_outside_cache`] says: lexically (see [`lexical`]), and
/// ignoring ASCII case when `fold_case` (a parameter so both behaviours can
/// be tested on any host). Without folding, a cache of `C:\Victory` and a
/// data folder of `c:\victory\data` would be taken as unrelated on Windows
/// although they are one folder and its child.
fn is_within(path: &Path, parent: &Path, fold_case: bool) -> bool {
    let path = lexical(path);
    let parent = lexical(parent);
    if !fold_case {
        return path.starts_with(&parent);
    }
    let mut inner = path.components();
    parent.components().all(|p| {
        inner.next().is_some_and(|c| {
            c.as_os_str()
                .as_encoded_bytes()
                .eq_ignore_ascii_case(p.as_os_str().as_encoded_bytes())
        })
    })
}

/// `path` with `.` dropped and each `..` removing the component before it,
/// without reading the disk. A `..` with nothing left to remove is kept
/// (it cannot climb above the root, which [`Path::components`] already
/// keeps first).
fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => out.push(".."),
            },
            other => out.push(other),
        }
    }
    out
}

/// The checks every configured path goes through before it is used: no
/// control characters ([`ensure_no_control`]), no surrounding quotes
/// ([`ensure_unquoted`]), no surrounding whitespace ([`ensure_trimmed`]),
/// no `file:` URI ([`ensure_not_uri`]), no unexpanded variable
/// ([`ensure_no_variable`]) and no Windows drive, network or backslash
/// path off Windows ([`ensure_native`]). `what` names the setting for the error.
fn check_setting(what: &str, path: &Path) -> Result<()> {
    ensure_no_control(what, path)?;
    ensure_unquoted(what, path)?;
    ensure_trimmed(what, path)?;
    ensure_not_uri(what, path)?;
    ensure_no_variable(what, path)?;
    ensure_native(what, path, cfg!(windows))
}

/// Refuses a `file:` URI (`file:///home/me/Games/gta`) instead of a path.
/// Copying a folder in a Linux file manager (Nautilus, Dolphin's "Copy
/// Location" in some setups), dragging one into a browser or launcher, or
/// taking it from a `.desktop` file or a Steam/Lutris/Heroic config gives
/// the URI, often percent-encoded (`My%20Games`). Nothing here decodes it:
/// the value is relative, so it would be taken as a folder named `file:`
/// under the config file's folder or the working directory (often the
/// repository), and `%20` would stay three characters. The error suggests
/// the decoded path when the URI names a local file. A folder that merely
/// starts with `file` (`/files`, `filed:`) is left alone; the scheme is
/// matched case-insensitively, as URI schemes are.
fn ensure_not_uri(what: &str, path: &Path) -> Result<()> {
    let text = path.to_string_lossy();
    let is_uri = text
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("file:"));
    if !is_uri {
        return Ok(());
    }
    let hint = match uri_path(&text[5..]) {
        Some(p) => format!("; write the path instead, perhaps {p}"),
        None => "; write the path instead".to_owned(),
    };
    anyhow::bail!("{what} is {text}, a file: URI rather than a path{hint}");
}

/// The local path a `file:` URI names, from what follows `file:`:
/// `///home/me` or `//localhost/home/me` (the empty or `localhost` host),
/// with `%XX` decoded; `/C:/games` becomes `C:/games`. `None` for another
/// host (`//server/share`, which is a network share, not a local path), a
/// bad escape, or a result that is not UTF-8.
fn uri_path(rest: &str) -> Option<String> {
    let path = match rest.strip_prefix("//") {
        Some(after) => {
            let slash = after.find('/')?;
            let host = &after[..slash];
            if !(host.is_empty() || host.eq_ignore_ascii_case("localhost")) {
                return None;
            }
            &after[slash..]
        }
        None if rest.starts_with('/') => rest,
        None => return None,
    };
    let mut out = Vec::with_capacity(path.len());
    let mut bytes = path.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let hex = [bytes.next()?, bytes.next()?];
            let hex = std::str::from_utf8(&hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
        } else {
            out.push(b);
        }
    }
    let decoded = String::from_utf8(out).ok()?;
    // `/C:/games`: a Windows drive keeps its own root.
    Some(match decoded.strip_prefix('/') {
        Some(drive) if is_drive_path(Path::new(drive)) => drive.to_owned(),
        _ => decoded,
    })
}

/// Refuses a path that names an environment variable instead of its value:
/// `$HOME/Games/gta`, `${XDG_DATA_HOME}/victory` or
/// `%LOCALAPPDATA%\victory` (the form `victory.example.toml` uses for the
/// defaults, so an easy one to copy). Only a shell (or `cmd` for `%...%`)
/// expands these; TOML, an env file read by `docker run --env-file` or
/// systemd, and a launcher or IDE run configuration keep them literally.
/// The path is then relative, so it would be taken as a folder named
/// `$HOME` under the config file's folder or the working directory (often
/// the repository) and reported later only as "does not exist" or "inside
/// the repository". Real folders like `$Recycle.Bin` or `100%` are left
/// alone: a `$` part must be a whole variable name (`$HOME`, not
/// `$HOME.old`), and `%` must enclose one.
fn ensure_no_variable(what: &str, path: &Path) -> Result<()> {
    let text = path.to_string_lossy();
    let Some(var) = text.split(['/', '\\']).find_map(variable_in) else {
        return Ok(());
    };
    anyhow::bail!(
        "{what} is {}, which names the variable {var}; variables are not expanded in \
         the config file or when the value comes from an env file or a launcher rather \
         than a shell, so write the path out in full (a leading ~ is the home directory)",
        path.display()
    );
}

/// The variable reference in one path component, as written: the whole
/// component `$NAME`, or `${NAME}` or `%NAME%` anywhere in it.
fn variable_in(part: &str) -> Option<&str> {
    let is_name = |s: &str| {
        let mut chars = s.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    if part.strip_prefix('$').is_some_and(is_name) {
        return Some(part);
    }
    if let Some(start) = part.find("${") {
        let rest = &part[start + 2..];
        if let Some(end) = rest.find('}')
            && is_name(&rest[..end])
        {
            return Some(&part[start..start + 2 + end + 1]);
        }
    }
    // `%ProgramFiles(x86)%` is a real Windows variable.
    let mut pieces = part.split('%');
    pieces.next();
    let mut offset = part.find('%')?;
    for piece in pieces {
        let name = piece.replace("(x86)", "");
        let closed = offset + 1 + piece.len() < part.len();
        if closed && is_name(&name) {
            return Some(&part[offset..offset + piece.len() + 2]);
        }
        offset += piece.len() + 1;
    }
    None
}

/// Refuses a path that starts or ends with whitespace (a space, or a
/// no-break space pasted from a web page). An env file or launcher keeps
/// what a shell would split off: `VICTORY_GAME_DIR= /games/gta` or a value
/// with a trailing space. A leading space makes the path relative, so it
/// would be taken as a folder named ` ` under the config file's folder or
/// the working directory (often the repository); a trailing one is reported
/// later only as "does not exist", which looks right on screen. Windows
/// strips trailing spaces from folder names, and no real dump or cache
/// folder starts or ends with one; spaces inside the name are left alone.
/// The error quotes the value so the space shows.
fn ensure_trimmed(what: &str, path: &Path) -> Result<()> {
    let text = path.to_string_lossy();
    let (start, end) = match (text.chars().next(), text.chars().next_back()) {
        (Some(s), Some(e)) => (s.is_whitespace(), e.is_whitespace()),
        _ => return Ok(()),
    };
    if !start && !end {
        return Ok(());
    }
    let place = match (start, end) {
        (true, true) => "starts and ends",
        (true, false) => "starts",
        _ => "ends",
    };
    anyhow::bail!("{what} is {text:?}, which {place} with whitespace; remove it");
}

/// Refuses a path wrapped in a matching pair of quotes (`"/games/gta"` or
/// `'/games/gta'`, quotes included). Only a shell removes them: an env file
/// read by `docker run --env-file`, systemd's `EnvironmentFile=` (single
/// quotes on older versions), an IDE run configuration or a Windows
/// `set VICTORY_GAME_DIR="D:\games"` keeps them as part of the value. The
/// path then starts with a quote, so it is not absolute and would be taken
/// as a folder named `"` under the working directory (often the
/// repository), reported later as "does not exist" or "inside the
/// repository". No real folder name starts and ends with the same quote,
/// and `"` cannot appear in a Windows path at all; a quote elsewhere in
/// the name (`/games/Rockstar's`) is left alone.
fn ensure_unquoted(what: &str, path: &Path) -> Result<()> {
    let bytes = path.as_os_str().as_encoded_bytes();
    let quoted = match bytes {
        [first, .., last] => first == last && matches!(first, b'"' | b'\''),
        _ => false,
    };
    if !quoted {
        return Ok(());
    }
    anyhow::bail!(
        "{what} is {}, quotes included; the quotes are not removed when the value comes \
         from an env file or a launcher rather than a shell, so remove them",
        path.display()
    );
}

/// Refuses a path holding a control character (a carriage return, newline,
/// tab, ...). No real folder name has one, but a variable read from a
/// `.env` file with Windows line endings (`set -a; . ./victory.env`, an
/// env file for a container or a Proton/Wine wrapper) ends in an invisible
/// `\r`, and a value pasted from a document can carry a tab or newline.
/// Without this the path is reported later only as "does not exist", which
/// looks right on screen; the error quotes the value so the character shows.
fn ensure_no_control(what: &str, path: &Path) -> Result<()> {
    let text = path.to_string_lossy();
    let Some(c) = text.chars().find(|c| c.is_control()) else {
        return Ok(());
    };
    let hint = if c == '\r' {
        " (a Windows line ending: was it read from a file saved with CRLF line endings?)"
    } else {
        ""
    };
    anyhow::bail!(
        "{what} is {text:?}, which contains the control character {c:?}{hint}; remove it"
    );
}

/// Refuses a Windows drive path (`D:\games`, `D:/games`, `D:`) when the
/// host is not Windows (`on_windows` is `cfg!(windows)`, a parameter so the
/// check can be tested anywhere). There it is not absolute, so it would be
/// taken as a relative path: under the config file's folder, or the working
/// directory for a variable, which is often the repository. The example
/// config ships such a path, so copying it unedited on Linux is an easy way
/// to get here; the error names the setting instead of a confusing
/// "inside the repository" or "does not exist" later on.
///
/// A Windows network path (`\\nas\games\gta`, or the `\\?\` and
/// `\\.\` device forms) is refused the same way: off Windows the
/// backslashes are ordinary characters, so it would be one relative folder
/// named `\\nas\games\gta`. The error says to mount the share and use the
/// mount point instead.
///
/// So is a path that starts the Windows way with a backslash (`\Games\gta`,
/// `.\gta`, `..\gta`, `~\Games\gta`), as copied from a Windows guide or a
/// `.bat` file. Off Windows it would be one relative folder named
/// `\Games\gta` (and `~\Games` would be read as another user's home); the
/// error suggests the same path with `/`. A backslash later in a name
/// (`games\gta`) is left alone, since a Unix folder name may hold one.
fn ensure_native(what: &str, path: &Path, on_windows: bool) -> Result<()> {
    if on_windows {
        return Ok(());
    }
    if is_drive_path(path) {
        anyhow::bail!(
            "{what} is {}, a Windows path, but this host is not Windows; set it to a \
             path on this machine",
            path.display()
        );
    }
    if is_unc_path(path) {
        anyhow::bail!(
            "{what} is {}, a Windows network path, but this host is not Windows; mount \
             the share and set it to the folder under the mount point",
            path.display()
        );
    }
    if is_backslash_relative(path) {
        let text = path.to_string_lossy();
        anyhow::bail!(
            "{what} is {text}, which uses Windows backslashes, but this host is not \
             Windows; write it with `/`, perhaps {}",
            text.replace('\\', "/")
        );
    }
    Ok(())
}

/// A path that starts with a Windows separator: a single leading `\`
/// (`\Games`, root of the current drive), or `.`, `..` or `~` followed by
/// `\`. A UNC path (`\\nas`) is [`is_unc_path`]'s business.
fn is_backslash_relative(path: &Path) -> bool {
    match path.as_os_str().as_encoded_bytes() {
        [b'\\'] => true,
        [b'\\', next, ..] => *next != b'\\',
        [b'.' | b'~', b'\\', ..] | [b'.', b'.', b'\\', ..] => true,
        _ => false,
    }
}

/// Refuses a `VICTORY_TEST_TITLES` entry that is really several titles
/// joined with the wrong separator: `/t/one;/t/two` or `/t/one,/t/two` off
/// Windows (where the list is split on `:`), or `C:\t\one:D:\t\two` or
/// `C:\t\one,D:\t\two` on Windows (split on `;`). A list copied from a
/// Windows `.bat` or from a guide for the other host, or written the way
/// most lists are, stays one entry, so it would be taken as a single title
/// named `one;` under `/t`, reported later only as "does not exist", with
/// both titles silently dropped. Only a separator directly followed (after
/// any spaces) by the start of another absolute path counts: off Windows
/// `/` or `~`, on Windows a drive path. A folder whose name merely holds a
/// `;` or `,` (`/games/a,b`) is left alone. `on_windows` is `cfg!(windows)`,
/// a parameter so the check can be tested anywhere.
fn ensure_one_title(title: &Path, on_windows: bool) -> Result<()> {
    let text = title.to_string_lossy();
    let (right, wrong): (char, &[char]) = if on_windows {
        (';', &[':', ','])
    } else {
        (':', &[';', ','])
    };
    for (i, c) in text.char_indices() {
        if i == 0 || !wrong.contains(&c) {
            continue;
        }
        let rest = text[i + c.len_utf8()..].trim_start();
        let another = if on_windows {
            is_drive_path(Path::new(rest))
        } else {
            rest.starts_with('/') || rest.starts_with('~')
        };
        if another {
            anyhow::bail!(
                "VICTORY_TEST_TITLES holds {text:?}, which looks like several titles joined \
                 with {c:?}; on this host they are separated by {right:?}"
            );
        }
    }
    Ok(())
}

/// `\\` followed by a server name, `?` or `.` (`\\nas\games`, `\\?\C:\x`,
/// `\\.\pipe`). Only backslashes count: `//nas/games` is an absolute path
/// on a Unix host.
fn is_unc_path(path: &Path) -> bool {
    match path.as_os_str().as_encoded_bytes() {
        [b'\\', b'\\', next, ..] => !matches!(next, b'\\' | b'/'),
        _ => false,
    }
}

/// `X:` followed by nothing, `\` or `/`.
fn is_drive_path(path: &Path) -> bool {
    match path.as_os_str().as_encoded_bytes() {
        [drive, b':'] => drive.is_ascii_alphabetic(),
        [drive, b':', sep, ..] => drive.is_ascii_alphabetic() && matches!(sep, b'\\' | b'/'),
        _ => false,
    }
}

/// `<root>/<sub>`, created. `sub` must be a plain relative path (names
/// only: no `..`, `.`, root or drive): [`Path::join`] with an absolute `sub`
/// would replace `root` outright, and a `..` would climb out of it, so a
/// title id or other computed part that came out wrong would write outside
/// the cache or data folder, possibly into the dump, without an error.
fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    ensure_plain_subdir(sub)?;
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// Refuses a `sub` for [`subdir`] that is empty or is not made only of
/// ordinary names.
fn ensure_plain_subdir(sub: &str) -> Result<()> {
    let path = Path::new(sub);
    let plain = !sub.is_empty()
        && !is_drive_path(path)
        && path.components().all(|c| matches!(c, Component::Normal(_)))
        // `Path::components` drops interior `.`; refuse it all the same.
        && !sub.split(['/', '\\']).any(|part| part == "." || part == "..");
    if !plain {
        anyhow::bail!(
            "internal error: subfolder {sub:?} is not a plain relative path; it would \
             leave the cache or data folder"
        );
    }
    Ok(())
}

/// The user's home directory: `HOME`, else `USERPROFILE` (Windows), when
/// set to an absolute path.
fn home_dir(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .filter_map(|k| env(k).map(PathBuf::from))
        .find(|p| p.is_absolute())
}

/// `~` and `~/rest` (`~\rest` on Windows) under `home`. Neither TOML nor a
/// quoted shell value expands `~`, so without this `game_dir = "~/games"`
/// would name a folder called `~`.
///
/// `~user` (another user's home, as a shell would read it) is refused, not
/// kept: kept, it is a relative folder literally named `~user` under the
/// config file's folder or the working directory, which a quoted value or
/// a TOML string reaches us as. Looking up another user's home is not
/// worth it here; the error asks for the full path, or `./~name` for a
/// folder that really is named that way.
///
/// With no home directory (`HOME` and `USERPROFILE` unset or relative, as
/// under `env -i`, a bare systemd service or some containers) a leading `~`
/// is refused rather than kept: kept, it is a relative folder named `~`
/// under the config file's folder or the working directory (often the
/// repository), created on first use and easy to delete by mistake with a
/// `rm -rf ~`. `what` names the setting for the error.
fn expand_home(what: &str, path: PathBuf, home: Option<&Path>) -> Result<PathBuf> {
    let mut parts = path.components();
    match parts.next() {
        Some(Component::Normal(first)) if first == "~" => {
            let Some(home) = home else {
                anyhow::bail!(
                    "{what} is {}, but there is no home directory to put in place of `~` \
                     (HOME is unset or not absolute); set HOME or write the full path",
                    path.display()
                );
            };
            let rest = parts.as_path();
            Ok(if rest.as_os_str().is_empty() {
                home.to_path_buf()
            } else {
                home.join(rest)
            })
        }
        Some(Component::Normal(first)) if first.to_string_lossy().starts_with('~') => {
            anyhow::bail!(
                "{what} is {}, but only a bare `~` is expanded, not `{}` (another user's \
                 home); write the full path, or `./{}` for a folder with that name",
                path.display(),
                first.to_string_lossy(),
                path.display()
            );
        }
        _ => Ok(path),
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Cache,
    Data,
}

/// `%LOCALAPPDATA%/victory/{cache,data}`. Without `LOCALAPPDATA` (a native
/// non-Windows host), the XDG base directories: the cache under
/// `$XDG_CACHE_HOME` (`~/.cache`), the data under `$XDG_DATA_HOME`
/// (`~/.local/share`), so clearing the user's cache does not take what titles
/// must keep with it.
///
/// A variable that is set but empty or relative is skipped, as the XDG base
/// directory spec asks: otherwise `XDG_CACHE_HOME=` (or a relative value)
/// would put the directory under the working directory, which is often the
/// repository itself and so refused by the asset policy.
fn default_dir(kind: Kind, env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<PathBuf> {
    let absolute = |k: &str| env(k).map(PathBuf::from).filter(|p| p.is_absolute());
    if let Some(local) = absolute("LOCALAPPDATA") {
        let name = match kind {
            Kind::Cache => "cache",
            Kind::Data => "data",
        };
        return Ok(local.join("victory").join(name));
    }
    let (xdg, fallback) = match kind {
        Kind::Cache => ("XDG_CACHE_HOME", Path::new(".cache")),
        Kind::Data => ("XDG_DATA_HOME", Path::new(".local/share")),
    };
    let base = absolute(xdg)
        .or_else(|| absolute("HOME").map(|h| h.join(fallback)))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn environment_overrides_file() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            cache_dir: Some("cache-from-file".into()),
            data_dir: Some("data-from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_GAME_DIR" => Some(OsString::from("from-env")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_GAME_DIR" => Some(OsString::new()),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    fn defaults(vars: &[(&str, OsString)]) -> Config {
        let env = |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
        Config::resolve(FilePaths::default(), None, env).unwrap()
    }

    #[test]
    fn local_app_data_holds_cache_and_data() {
        let cfg = defaults(&[("LOCALAPPDATA", abs("l")), ("XDG_CACHE_HOME", abs("x"))]);
        let base = PathBuf::from(abs("l")).join("victory");
        assert_eq!(cfg.cache_dir, base.join("cache"));
        assert_eq!(cfg.data_dir, base.join("data"));
    }

    #[test]
    fn data_is_not_under_the_xdg_cache() {
        let cfg = defaults(&[("XDG_CACHE_HOME", abs("c")), ("XDG_DATA_HOME", abs("d"))]);
        assert_eq!(cfg.cache_dir, PathBuf::from(abs("c")).join("victory"));
        assert_eq!(cfg.data_dir, PathBuf::from(abs("d")).join("victory"));

        let cfg = defaults(&[("HOME", abs("h"))]);
        let home = PathBuf::from(abs("h"));
        assert_eq!(cfg.cache_dir, home.join(".cache").join("victory"));
        assert_eq!(cfg.data_dir, home.join(".local/share").join("victory"));
        assert!(!cfg.data_dir.starts_with(&cfg.cache_dir));
    }

    #[test]
    fn empty_or_relative_values_are_ignored() {
        let cfg = defaults(&[
            ("LOCALAPPDATA", OsString::new()),
            ("XDG_CACHE_HOME", OsString::new()),
            ("XDG_DATA_HOME", OsString::from("relative")),
            ("HOME", abs("h")),
        ]);
        let home = PathBuf::from(abs("h"));
        assert_eq!(cfg.cache_dir, home.join(".cache").join("victory"));
        assert_eq!(cfg.data_dir, home.join(".local/share").join("victory"));

        // A relative LOCALAPPDATA or XDG_CACHE_HOME falls through as well.
        let cfg = defaults(&[
            ("LOCALAPPDATA", OsString::from("relative/lad")),
            ("XDG_CACHE_HOME", OsString::from("relative/cache")),
            ("HOME", abs("h")),
        ]);
        assert_eq!(cfg.cache_dir, home.join(".cache").join("victory"));
    }

    #[test]
    fn default_dir_prefers_localappdata_then_xdg_then_home() {
        let all = |k: &str| match k {
            "LOCALAPPDATA" => Some(abs("lad")),
            "XDG_CACHE_HOME" => Some(abs("xdg")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        assert_eq!(
            default_dir(Kind::Cache, all).unwrap(),
            PathBuf::from(abs("lad")).join("victory").join("cache")
        );
        let no_lad = |k: &str| if k == "LOCALAPPDATA" { None } else { all(k) };
        assert_eq!(
            default_dir(Kind::Cache, no_lad).unwrap(),
            PathBuf::from(abs("xdg")).join("victory")
        );
        let home_only = |k: &str| (k == "HOME").then(|| abs("home"));
        assert_eq!(
            default_dir(Kind::Cache, home_only).unwrap(),
            PathBuf::from(abs("home")).join(".cache").join("victory")
        );
    }

    #[test]
    fn default_dir_skips_relative_bases_for_both_kinds() {
        // A relative `LOCALAPPDATA` or XDG value is skipped tier by tier, not
        // joined to the working directory, for the data folder as for the
        // cache.
        let env = |k: &str| match k {
            "LOCALAPPDATA" => Some(OsString::from("AppData/Local")),
            "XDG_CACHE_HOME" => Some(OsString::from(".cache")),
            "XDG_DATA_HOME" => Some(OsString::from("share")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let home = PathBuf::from(abs("home"));
        assert_eq!(
            default_dir(Kind::Cache, env).unwrap(),
            home.join(".cache").join("victory")
        );
        assert_eq!(
            default_dir(Kind::Data, env).unwrap(),
            home.join(".local/share").join("victory")
        );
        // `LOCALAPPDATA` keeps cache and data apart under one `victory`.
        let lad = |k: &str| (k == "LOCALAPPDATA").then(|| abs("lad"));
        assert_eq!(
            default_dir(Kind::Data, lad).unwrap(),
            PathBuf::from(abs("lad")).join("victory").join("data")
        );
    }

    /// A literal `~` in a base variable (a quoted `Environment=` line, a
    /// `.env` file or a Docker `ENV`, none of which expand it) is relative,
    /// so it is skipped like any relative value: the defaults never become a
    /// folder named `~` under the working directory. A trailing slash on a
    /// base is harmless.
    #[test]
    fn a_literal_tilde_in_a_base_variable_is_skipped() {
        let home = PathBuf::from(abs("home"));
        let cfg = defaults(&[
            ("LOCALAPPDATA", OsString::from("~/AppData/Local")),
            ("XDG_CACHE_HOME", OsString::from("~/.cache")),
            ("XDG_DATA_HOME", OsString::from("~")),
            ("HOME", home.clone().into()),
        ]);
        assert_eq!(cfg.cache_dir, home.join(".cache").join("victory"));
        assert_eq!(cfg.data_dir, home.join(".local/share").join("victory"));
        for dir in [&cfg.cache_dir, &cfg.data_dir] {
            assert!(dir.is_absolute(), "{}", dir.display());
            assert!(
                !dir.components().any(|c| c.as_os_str() == "~"),
                "{}",
                dir.display()
            );
        }

        // `HOME=~` is no home either: with nothing else usable it is the
        // usual error, not `./~/.cache/victory`.
        let tilde_home = |k: &str| (k == "HOME").then(|| OsString::from("~"));
        let err = default_dir(Kind::Cache, tilde_home)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("set VICTORY_CACHE_DIR and VICTORY_DATA_DIR"),
            "{err}"
        );
        assert!(default_dir(Kind::Data, tilde_home).is_err());

        let mut slash = abs("xc");
        slash.push(std::path::MAIN_SEPARATOR_STR);
        let trailing = |k: &str| (k == "XDG_CACHE_HOME").then(|| slash.clone());
        assert_eq!(
            default_dir(Kind::Cache, trailing).unwrap(),
            PathBuf::from(abs("xc")).join("victory")
        );
    }

    #[test]
    fn a_tilde_with_extra_slashes_is_still_home() {
        // `~/` and `~//x` are what a trailing slash or a doubled separator
        // in a copied path give; they are home and `home/x`, not a folder
        // named `~`.
        let home = PathBuf::from(abs("home"));
        let ex = |p: &str| expand_home("VICTORY_GAME_DIR", p.into(), Some(&home)).unwrap();
        assert_eq!(ex("~/"), home);
        assert_eq!(ex("~//games"), home.join("games"));
        assert_eq!(ex("~/./games"), home.join("games"));
    }

    #[test]
    fn no_user_directory_is_an_error() {
        let env = |_: &str| None;
        assert!(Config::resolve(FilePaths::default(), None, env).is_err());
        let nothing_usable = |k: &str| (k == "HOME").then(OsString::new);
        assert!(default_dir(Kind::Cache, nothing_usable).is_err());
        assert!(default_dir(Kind::Data, nothing_usable).is_err());
    }

    #[test]
    fn empty_test_titles_variable_falls_back_to_the_file() {
        let file = FilePaths {
            test_titles: vec!["from-file".into()],
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(OsString::new()),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn relative_file_paths_are_relative_to_the_file() {
        let text = format!(
            "[paths]\ngame_dir = 'game'\ncache_dir = {:?}\ntest_titles = ['t/one']\n",
            PathBuf::from(abs("elsewhere")).display().to_string()
        );
        let parsed: FileConfig = toml::from_str(&text).unwrap();
        let dir = PathBuf::from(abs("configs"));
        let paths = parsed.anchored_at(&dir, None).unwrap();
        assert_eq!(paths.game_dir.unwrap(), dir.join("game"));
        // Absolute paths are kept as written.
        assert_eq!(paths.cache_dir.unwrap(), PathBuf::from(abs("elsewhere")));
        assert_eq!(paths.data_dir, None);
        assert_eq!(paths.test_titles, vec![dir.join("t/one")]);
    }

    #[test]
    fn empty_file_values_count_as_unset() {
        let text = "[paths]\ngame_dir = ''\ncache_dir = \"\"\ndata_dir = ''\n\
                    test_titles = ['', 't/one', '']\n";
        let parsed: FileConfig = toml::from_str(text).unwrap();
        let dir = PathBuf::from(abs("configs"));
        let paths = parsed.anchored_at(&dir, None).unwrap();
        // Not the config file's folder.
        assert_eq!(paths.game_dir, None);
        assert_eq!(paths.cache_dir, None);
        assert_eq!(paths.data_dir, None);
        assert_eq!(paths.test_titles, vec![dir.join("t/one")]);
        // With `./victory.toml` they are not the empty path either, and the
        // defaults apply with nothing said to come from the file.
        let parsed: FileConfig = toml::from_str(text).unwrap();
        let paths = parsed.anchored_at(Path::new(""), None).unwrap();
        let env = |k: &str| match k {
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(paths, None, env).unwrap();
        assert_eq!(cfg.game_dir, None);
        assert!(cfg.cache_dir.is_absolute(), "{}", cfg.cache_dir.display());
        assert!(cfg.data_dir.is_absolute(), "{}", cfg.data_dir.display());
        assert_eq!(cfg.origins.game_dir, Origin::Default);
        assert_eq!(cfg.origins.cache_dir, Origin::Default);
        assert_eq!(cfg.origins.data_dir, Origin::Default);
        assert_eq!(cfg.test_titles, vec![PathBuf::from("t/one")]);
        assert_eq!(cfg.origins.test_titles, Origin::File);
    }

    #[test]
    fn a_relative_victory_config_anchors_at_its_folder_then_the_working_directory() {
        // `VICTORY_CONFIG=conf/victory.toml`: its folder is `conf`, itself
        // relative, so the file's relative settings become `conf/...` and
        // only then are joined to the working directory, not dropped there.
        let text = "[paths]\ngame_dir = './~gta'\ncache_dir = '../shared/cache'\n\
                    data_dir = '~/saves'\ntest_titles = ['t/one']\n";
        let parsed: FileConfig = toml::from_str(text).unwrap();
        let home = PathBuf::from(abs("home"));
        let config = Path::new("conf").join("victory.toml");
        let dir = config.parent().unwrap();
        let paths = parsed.anchored_at(dir, Some(&home)).unwrap();
        // `./~gta` is a folder named `~gta` beside the file, not a home.
        assert_eq!(paths.game_dir.as_deref(), Some(&*dir.join("./~gta")));
        // `..` is kept for the lexical checks, not resolved against disk.
        assert_eq!(
            paths.cache_dir.as_deref(),
            Some(&*dir.join("../shared/cache"))
        );
        // `~` in the file is home even when the file's folder is relative.
        assert_eq!(paths.data_dir.as_deref(), Some(&*home.join("saves")));

        let cwd = PathBuf::from(abs("work"));
        let env = |k: &str| match k {
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve_in(paths, Some(config.clone()), env, Some(&cwd)).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), cwd.join("conf/./~gta"));
        assert_eq!(cfg.cache_dir, cwd.join("conf/../shared/cache"));
        assert_eq!(cfg.data_dir, home.join("saves"));
        assert_eq!(cfg.test_titles, vec![cwd.join("conf/t/one")]);
        assert_eq!(cfg.source, Some(config));
        assert_eq!(cfg.origins.game_dir, Origin::File);
        assert_eq!(cfg.origins.cache_dir, Origin::File);
        assert_eq!(cfg.origins.data_dir, Origin::File);
        assert_eq!(cfg.origins.test_titles, Origin::File);
    }

    #[test]
    fn a_config_file_at_the_root_anchors_at_the_root() {
        // `VICTORY_CONFIG=/victory.toml` has `/` as its folder, not the
        // empty path, so relative settings land under the root and are not
        // left relative to the working directory.
        let root = PathBuf::from(abs(""));
        let config = root.join("victory.toml");
        let dir = config.parent().unwrap();
        assert_eq!(dir, root);
        let parsed: FileConfig =
            toml::from_str("[paths]\ngame_dir = 'gta'\ncache_dir = 'srv/cache'\n").unwrap();
        let paths = parsed.anchored_at(dir, None).unwrap();
        assert_eq!(paths.game_dir.unwrap(), root.join("gta"));
        assert_eq!(paths.cache_dir.unwrap(), root.join("srv/cache"));
        // Only a relative path is anchored: `cache_dir = '.'` is the root
        // itself, which the folder checks then refuse as a cache.
        let parsed: FileConfig = toml::from_str("[paths]\ncache_dir = '.'\n").unwrap();
        let paths = parsed.anchored_at(dir, None).unwrap();
        let env = |k: &str| match k {
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let err = Config::resolve(paths, Some(config), env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].cache_dir"), "{err}");
    }

    #[test]
    fn the_default_config_file_keeps_paths_as_written() {
        // `./victory.toml` has an empty parent, so nothing changes.
        let parsed: FileConfig = toml::from_str("[paths]\ngame_dir = 'game'\n").unwrap();
        let paths = parsed.anchored_at(Path::new(""), None).unwrap();
        assert_eq!(paths.game_dir.unwrap(), PathBuf::from("game"));
    }

    #[test]
    fn a_leading_tilde_is_the_home_directory() {
        let home = PathBuf::from(abs("home"));
        let h = Some(home.as_path());
        let ex = |p: &str, h| expand_home("VICTORY_GAME_DIR", p.into(), h).unwrap();
        assert_eq!(ex("~", h), home);
        assert_eq!(ex("~/games/gta", h), home.join("games/gta"));
        // Only a leading `~` counts; `./~user` names a folder called `~user`.
        assert_eq!(ex("a/~/b", h), PathBuf::from("a/~/b"));
        assert_eq!(ex("./~user/x", h), PathBuf::from("./~user/x"));
        // Without a home directory, paths without a leading `~` are kept.
        assert_eq!(ex("/x/~", None), PathBuf::from("/x/~"));
    }

    #[test]
    fn a_leading_tilde_user_is_refused() {
        // `~user` is not expanded; kept, it would be a relative folder named
        // `~user` under the working directory or the config file's folder.
        let home = PathBuf::from(abs("home"));
        for h in [Some(home.as_path()), None] {
            for p in ["~user", "~user/games/gta", "~root"] {
                let err = expand_home("VICTORY_GAME_DIR", p.into(), h)
                    .unwrap_err()
                    .to_string();
                assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
                assert!(err.contains("another user's"), "{err}");
                assert!(err.contains(&format!("./{p}")), "{err}");
            }
        }
        let parsed: FileConfig = toml::from_str("[paths]\ncache_dir = '~steam/c'\n").unwrap();
        let err = parsed
            .anchored_at(&PathBuf::from(abs("configs")), Some(&home))
            .unwrap_err();
        assert!(err.to_string().contains("[paths].cache_dir"), "{err}");
        let env = |k: &str| match k {
            "VICTORY_CONFIG" => Some(OsString::from("~other/victory.toml")),
            _ => None,
        };
        let err = explicit_config(env, Some(&home)).unwrap_err();
        assert!(err.to_string().contains("VICTORY_CONFIG"), "{err}");
    }

    #[test]
    fn a_leading_tilde_without_a_home_is_refused() {
        // `env -i`, a bare systemd unit: no HOME, so `~` cannot be expanded
        // and would otherwise be a folder named `~` under the working
        // directory.
        for p in ["~", "~/games/gta"] {
            let err = expand_home("VICTORY_GAME_DIR", p.into(), None)
                .unwrap_err()
                .to_string();
            assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
            assert!(err.contains("no home directory"), "{err}");
        }
        let no_home = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(OsString::from("~/cache")),
            "HOME" => Some(OsString::from("relative")),
            "XDG_DATA_HOME" => Some(abs("data")),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, no_home).unwrap_err();
        assert!(err.to_string().contains("VICTORY_CACHE_DIR"), "{err}");
        let sep = if cfg!(windows) { ";" } else { ":" };
        let titles = format!("{}{sep}~/t", abs("one").to_string_lossy());
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(OsString::from(&titles)),
            "XDG_CACHE_HOME" => Some(abs("cache")),
            "XDG_DATA_HOME" => Some(abs("data")),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, env).unwrap_err();
        assert!(err.to_string().contains("VICTORY_TEST_TITLES"), "{err}");
        let parsed: FileConfig = toml::from_str("[paths]\ndata_dir = '~/d'\n").unwrap();
        let err = parsed
            .anchored_at(&PathBuf::from(abs("configs")), None)
            .unwrap_err();
        assert!(err.to_string().contains("[paths].data_dir"), "{err}");
    }

    #[test]
    fn tilde_in_the_file_is_home_not_next_to_the_file() {
        let parsed: FileConfig =
            toml::from_str("[paths]\ngame_dir = '~/games/gta'\ntest_titles = ['~/t']\n").unwrap();
        let home = PathBuf::from(abs("home"));
        let paths = parsed
            .anchored_at(&PathBuf::from(abs("configs")), Some(&home))
            .unwrap();
        assert_eq!(paths.game_dir.unwrap(), home.join("games/gta"));
        assert_eq!(paths.test_titles, vec![home.join("t")]);
    }

    #[test]
    fn tilde_in_variables_is_home() {
        let sep = if cfg!(windows) { ";" } else { ":" };
        let titles = format!("~/one{sep}~/two");
        let env = |k: &str| match k {
            "VICTORY_GAME_DIR" => Some(OsString::from("~/games/gta")),
            "VICTORY_CACHE_DIR" => Some(OsString::from("~/cache")),
            "VICTORY_TEST_TITLES" => Some(OsString::from(&titles)),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(FilePaths::default(), None, env).unwrap();
        let home = PathBuf::from(abs("home"));
        assert_eq!(cfg.game_dir.unwrap(), home.join("games/gta"));
        assert_eq!(cfg.cache_dir, home.join("cache"));
        assert_eq!(cfg.test_titles, vec![home.join("one"), home.join("two")]);
    }

    #[test]
    fn home_dir_falls_back_to_userprofile_and_skips_relative() {
        let env = |k: &str| match k {
            "HOME" => Some(OsString::from("relative")),
            "USERPROFILE" => Some(abs("profile")),
            _ => None,
        };
        assert_eq!(home_dir(env), Some(PathBuf::from(abs("profile"))));
        assert_eq!(home_dir(|_: &str| None), None);
    }

    #[test]
    fn victory_config_expands_tilde_and_ignores_empty() {
        let home = PathBuf::from(abs("home"));
        let with = |v: &'static str| move |k: &str| (k == "VICTORY_CONFIG").then(|| v.into());
        assert_eq!(
            explicit_config(with("~/victory.toml"), Some(&home)).unwrap(),
            Some(home.join("victory.toml"))
        );
        assert_eq!(explicit_config(with(""), Some(&home)).unwrap(), None);
        assert_eq!(explicit_config(|_: &str| None, Some(&home)).unwrap(), None);
        // Without a home directory a leading `~` cannot be expanded.
        let err = explicit_config(with("~/victory.toml"), None).unwrap_err();
        assert!(err.to_string().contains("VICTORY_CONFIG"), "{err}");
    }

    #[test]
    fn control_characters_in_paths_are_refused() {
        // A CRLF `.env` file leaves a `\r` on the end of every value.
        let err = ensure_no_control("VICTORY_GAME_DIR", Path::new("/games/gta\r"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
        assert!(err.contains("\"/games/gta\\r\""), "{err}");
        assert!(err.contains("CRLF"), "{err}");
        let err = ensure_no_control("[paths].data_dir", Path::new("/a\tb"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("'\\t'"), "{err}");
        assert!(!err.contains("CRLF"), "{err}");
        for ok in ["/games/gta", "games with spaces", "~/jeux/été", ""] {
            assert!(ensure_no_control("x", Path::new(ok)).is_ok(), "{ok}");
        }

        // Every way in: each path variable, VICTORY_CONFIG and the file.
        for var in [
            "VICTORY_GAME_DIR",
            "VICTORY_CACHE_DIR",
            "VICTORY_DATA_DIR",
            "VICTORY_TEST_TITLES",
        ] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("/victory/x\r")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(var), "{err}");
        }
        let env = |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("victory.toml\r"));
        let err = explicit_config(env, None).unwrap_err().to_string();
        assert!(err.contains("VICTORY_CONFIG"), "{err}");
        let parsed: FileConfig = toml::from_str("[paths]\ncache_dir = \"cache\\n\"\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].cache_dir"), "{err}");
    }

    #[test]
    fn quoted_paths_are_refused() {
        // `docker run --env-file` keeps the quotes of `VICTORY_GAME_DIR="/games/gta"`.
        let err = ensure_unquoted("VICTORY_GAME_DIR", Path::new("\"/games/gta\""))
            .unwrap_err()
            .to_string();
        assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
        assert!(err.contains("\"/games/gta\""), "{err}");
        assert!(err.contains("quotes included"), "{err}");
        assert!(ensure_unquoted("x", Path::new("'~/gta'")).is_err());
        for ok in [
            "/games/gta",
            "/games/Rockstar's",
            "'/games/gta\"",
            "\"",
            "'",
            "",
            "games with spaces",
        ] {
            assert!(ensure_unquoted("x", Path::new(ok)).is_ok(), "{ok}");
        }

        // Every way in: each path variable, VICTORY_CONFIG and the file.
        for var in [
            "VICTORY_GAME_DIR",
            "VICTORY_CACHE_DIR",
            "VICTORY_DATA_DIR",
            "VICTORY_TEST_TITLES",
        ] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("\"/victory/x\"")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(var), "{err}");
            assert!(err.contains("quotes included"), "{err}");
        }
        let env = |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("'victory.toml'"));
        let err = explicit_config(env, None).unwrap_err().to_string();
        assert!(err.contains("VICTORY_CONFIG"), "{err}");
        let parsed: FileConfig = toml::from_str("[paths]\ngame_dir = '\"/games/gta\"'\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].game_dir"), "{err}");
    }

    #[test]
    fn surrounding_whitespace_in_paths_is_refused() {
        let err = ensure_trimmed("VICTORY_GAME_DIR", Path::new(" /games/gta"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
        assert!(err.contains("\" /games/gta\""), "{err}");
        assert!(err.contains("starts with whitespace"), "{err}");
        let err = ensure_trimmed("x", Path::new("/games/gta "))
            .unwrap_err()
            .to_string();
        assert!(err.contains("ends with whitespace"), "{err}");
        let err = ensure_trimmed("x", Path::new(" a "))
            .unwrap_err()
            .to_string();
        assert!(err.contains("starts and ends"), "{err}");
        // A no-break space pasted from a web page.
        assert!(ensure_trimmed("x", Path::new("/games/gta\u{a0}")).is_err());
        for ok in [
            "/games/gta",
            "games with spaces",
            "/Program Files/x",
            "~/é",
            "",
        ] {
            assert!(ensure_trimmed("x", Path::new(ok)).is_ok(), "{ok}");
        }

        // Every way in: each path variable, VICTORY_CONFIG and the file.
        for var in [
            "VICTORY_GAME_DIR",
            "VICTORY_CACHE_DIR",
            "VICTORY_DATA_DIR",
            "VICTORY_TEST_TITLES",
        ] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from(" /victory/x")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(var), "{err}");
            assert!(err.contains("whitespace"), "{err}");
        }
        // The second of several test titles, after the separator.
        let sep = if cfg!(windows) { ";" } else { ":" };
        let titles = OsString::from(format!("/victory/a{sep} /victory/b"));
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(titles.clone()),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("\" /victory/b\""), "{err}");
        let env = |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("victory.toml "));
        let err = explicit_config(env, None).unwrap_err().to_string();
        assert!(err.contains("VICTORY_CONFIG"), "{err}");
        let parsed: FileConfig = toml::from_str("[paths]\ndata_dir = ' data'\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].data_dir"), "{err}");
    }

    #[test]
    fn file_uris_are_refused() {
        // A folder copied from a file manager, percent-encoded.
        let err = ensure_not_uri(
            "VICTORY_GAME_DIR",
            Path::new("file:///games/My%20Games/gta"),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
        assert!(err.contains("file: URI"), "{err}");
        assert!(err.contains("perhaps /games/My Games/gta"), "{err}");
        assert_eq!(uri_path("//localhost/a/b").as_deref(), Some("/a/b"));
        assert_eq!(uri_path("/a/%C3%A9").as_deref(), Some("/a/é"));
        assert_eq!(uri_path("///C:/games").as_deref(), Some("C:/games"));
        // Another host, a bad escape or no path: no suggestion.
        for bad in [
            "//server/share",
            "///a%2",
            "///a%zz",
            "///a%FF",
            "games",
            "//host",
        ] {
            assert_eq!(uri_path(bad), None, "{bad}");
        }
        let err = ensure_not_uri("x", Path::new("FILE://server/share"))
            .unwrap_err()
            .to_string();
        assert!(err.ends_with("write the path instead"), "{err}");
        for ok in ["/files/gta", "files", "filed", "file", "/games/file:x", ""] {
            assert!(ensure_not_uri("x", Path::new(ok)).is_ok(), "{ok}");
        }

        // Every way in: each path variable, VICTORY_CONFIG and the file.
        for var in [
            "VICTORY_GAME_DIR",
            "VICTORY_CACHE_DIR",
            "VICTORY_DATA_DIR",
            "VICTORY_TEST_TITLES",
        ] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("file:///victory/x")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(var), "{err}");
            assert!(err.contains("file: URI"), "{err}");
        }
        let env =
            |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("file:///c/victory.toml"));
        let err = explicit_config(env, None).unwrap_err().to_string();
        assert!(err.contains("VICTORY_CONFIG"), "{err}");
        let parsed: FileConfig =
            toml::from_str("[paths]\ngame_dir = 'file:///games/gta'\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].game_dir"), "{err}");
    }

    #[test]
    fn unexpanded_variables_in_paths_are_refused() {
        // An env file or the config file keeps `$HOME` literally.
        let err = ensure_no_variable("VICTORY_GAME_DIR", Path::new("$HOME/Games/gta"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
        assert!(err.contains("the variable $HOME;"), "{err}");
        assert!(err.contains("~ is the home directory"), "{err}");
        for (path, var) in [
            ("${XDG_DATA_HOME}/victory", "${XDG_DATA_HOME}"),
            ("/srv/${USER}-victory/cache", "${USER}"),
            ("/home/$USER/gta", "$USER"),
            ("%LOCALAPPDATA%\\victory\\cache", "%LOCALAPPDATA%"),
            ("%LocalAppData%/victory", "%LocalAppData%"),
            ("%ProgramFiles(x86)%\\gta", "%ProgramFiles(x86)%"),
            ("~/%USERNAME%-gta", "%USERNAME%"),
        ] {
            let err = ensure_no_variable("x", Path::new(path))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("the variable {var};")),
                "{path}: {err}"
            );
        }
        // Real folder names with `$` or `%` in them.
        for ok in [
            "/games/gta",
            "C:\\$Recycle.Bin\\x",
            "/mnt/share$/gta",
            "/games/$HOME.old",
            "/games/100%",
            "/games/50% off/x",
            "/a/${}/b",
            "/a/$/b",
            "/a/%%/b",
            "~/games",
            "",
        ] {
            assert!(ensure_no_variable("x", Path::new(ok)).is_ok(), "{ok}");
        }

        // Every way in: each path variable, VICTORY_CONFIG and the file.
        for var in [
            "VICTORY_GAME_DIR",
            "VICTORY_CACHE_DIR",
            "VICTORY_DATA_DIR",
            "VICTORY_TEST_TITLES",
        ] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("$HOME/victory/x")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(var), "{err}");
            assert!(err.contains("$HOME"), "{err}");
        }
        let env = |k: &str| {
            (k == "VICTORY_CONFIG").then(|| OsString::from("${XDG_CONFIG_HOME}/victory.toml"))
        };
        let err = explicit_config(env, None).unwrap_err().to_string();
        assert!(err.contains("VICTORY_CONFIG"), "{err}");
        let parsed: FileConfig =
            toml::from_str("[paths]\ndata_dir = '%LOCALAPPDATA%\\victory\\data'\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].data_dir"), "{err}");
        assert!(err.contains("%LOCALAPPDATA%"), "{err}");
    }

    #[test]
    fn misspelled_variables_are_refused() {
        for (name, meant) in [
            ("VICTORY_GAMEDIR", "VICTORY_GAME_DIR"),
            ("VICTORY_GAME_PATH", "VICTORY_GAME_DIR"),
            ("VICTORY_GAME", "VICTORY_GAME_DIR"),
            ("VICTORY_CACHE_DIRECTORY", "VICTORY_CACHE_DIR"),
            ("VICTORY_DATA", "VICTORY_DATA_DIR"),
            ("VICTORY_DATA_FOLDER", "VICTORY_DATA_DIR"),
            ("VICTORY-DATA-DIR", "VICTORY_DATA_DIR"),
            ("VICTORY_TEST_TITLE", "VICTORY_TEST_TITLES"),
            ("VICTORY_TITLES", "VICTORY_TEST_TITLES"),
            ("VICTORY_CONFIG_FILE", "VICTORY_CONFIG"),
            ("VICTORYCONFIG", "VICTORY_CONFIG"),
        ] {
            assert_eq!(misspelling_of(name), Some(meant), "{name}");
        }
        // Not on Windows, where variable names ignore case.
        if !cfg!(windows) {
            assert_eq!(
                misspelling_of("victory_cache_dir"),
                Some("VICTORY_CACHE_DIR")
            );
        }
        for name in [
            "VICTORY_GAME_DIR",
            "VICTORY_CONFIG",
            "VICTORY_LOG",
            "VICTORY_WARP",
            "VICTORY_CONFIG_DIR",
            "VICTORY_GAME_DIRS",
            "HOME",
            "XDG_DATA_HOME",
        ] {
            assert_eq!(misspelling_of(name), None, "{name}");
        }

        let names =
            |list: &[&str]| -> Vec<std::ffi::OsString> { list.iter().map(Into::into).collect() };
        let err = ensure_no_misspelled_variables(names(&["HOME", "VICTORY_DATA_PATH"]), |_| None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("VICTORY_DATA_PATH is set, but Victory reads VICTORY_DATA_DIR"),
            "{err}"
        );
        // The real variable set: the near miss belongs to something else.
        ensure_no_misspelled_variables(names(&["VICTORY_DATA_PATH"]), |k| {
            (k == "VICTORY_DATA_DIR").then(|| "/d".into())
        })
        .unwrap();
        // Set but empty counts as unset, like everywhere else.
        assert!(
            ensure_no_misspelled_variables(names(&["VICTORY_GAMEDIR"]), |k| {
                (k == "VICTORY_GAME_DIR").then(Default::default)
            })
            .is_err()
        );
        ensure_no_misspelled_variables(names(&["VICTORY_LOG", "PATH"]), |_| None).unwrap();
    }

    #[test]
    fn victory_config_must_name_a_file() {
        let dir = std::env::temp_dir().join(format!("victory-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("victory.toml");
        let _ = std::fs::remove_file(&file);

        let err = config_file(Some(dir.clone())).unwrap_err().to_string();
        assert!(err.contains("is a folder"), "{err}");
        assert!(!err.contains("did you mean"), "{err}");

        std::fs::write(&file, "").unwrap();
        let err = config_file(Some(dir.clone())).unwrap_err().to_string();
        assert!(err.contains("did you mean"), "{err}");
        assert!(err.contains(&file.display().to_string()), "{err}");
        assert_eq!(config_file(Some(file.clone())).unwrap(), Some(file.clone()));

        let missing = dir.join("missing.toml");
        let err = config_file(Some(missing)).unwrap_err().to_string();
        assert!(err.contains("does not exist"), "{err}");
        #[cfg(unix)]
        {
            let blocked = file.join("child");
            let err = config_file(Some(blocked.clone())).unwrap_err();
            assert!(err.to_string().contains(&blocked.display().to_string()));
            assert_eq!(
                err.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::NotADirectory
            );

            let device = PathBuf::from("/dev/null");
            let err = config_file(Some(device)).unwrap_err().to_string();
            assert!(
                err.contains("/dev/null, which is not a regular file"),
                "{err}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_default_config_that_is_not_a_file_is_refused() {
        // A scratch folder under the system temp folder, never the
        // repository or the working directory; removed at the end.
        let dir = std::env::temp_dir().join(format!(
            "victory-default-config-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let default = dir.join("victory.toml");

        // Missing: no config file, as before.
        assert_eq!(config_file_or(None, &default).unwrap(), None);

        // A regular file is read.
        std::fs::write(&default, "").unwrap();
        assert_eq!(
            config_file_or(None, &default).unwrap(),
            Some(default.clone())
        );
        // VICTORY_CONFIG still wins over the default.
        let other = dir.join("other.toml");
        std::fs::write(&other, "").unwrap();
        assert_eq!(
            config_file_or(Some(other.clone()), &default).unwrap(),
            Some(other.clone())
        );
        std::fs::remove_file(&default).unwrap();

        // A folder named victory.toml is reported, not skipped.
        std::fs::create_dir(&default).unwrap();
        let err = config_file_or(None, &default).unwrap_err().to_string();
        assert!(err.contains("is a folder, not a config file"), "{err}");
        assert!(err.contains(&default.display().to_string()), "{err}");
        assert!(!err.contains("VICTORY_CONFIG"), "{err}");
        std::fs::remove_dir(&default).unwrap();

        #[cfg(unix)]
        {
            // A link whose target is gone names the target, for the default
            // file and for VICTORY_CONFIG.
            let gone = dir.join("moved-away.toml");
            std::os::unix::fs::symlink(&gone, &default).unwrap();
            let err = config_file_or(None, &default).unwrap_err().to_string();
            assert!(
                err.contains(&format!(
                    "a link to {}, which does not exist",
                    gone.display()
                )),
                "{err}"
            );
            let err = config_file(Some(default.clone())).unwrap_err().to_string();
            assert!(err.starts_with("VICTORY_CONFIG points at"), "{err}");
            assert!(err.contains(&gone.display().to_string()), "{err}");

            // Once the target is back, the link is read.
            std::fs::write(&gone, "").unwrap();
            assert_eq!(
                config_file_or(None, &default).unwrap(),
                Some(default.clone())
            );
            std::fs::remove_file(&default).unwrap();

            // A path that cannot be checked is an error, not "no file".
            std::fs::write(&default, "").unwrap();
            let blocked = default.join("victory.toml");
            let err = config_file_or(None, &blocked).unwrap_err();
            assert_eq!(
                err.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::NotADirectory
            );
            assert!(
                err.to_string().contains("checking the config file"),
                "{err}"
            );
            std::fs::remove_file(&default).unwrap();

            // Something that is neither a file nor a folder.
            std::os::unix::fs::symlink("/dev/null", &default).unwrap();
            let err = config_file_or(None, &default).unwrap_err().to_string();
            assert!(err.contains("which is not a regular file"), "{err}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn victory_config_is_read_without_looking_at_a_broken_default() {
        // A scratch folder under the system temp folder, never the
        // repository or the working directory; removed at the end.
        let dir = std::env::temp_dir().join(format!(
            "victory-explicit-over-default-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let default = dir.join("victory.toml");
        let explicit = dir.join("explicit.toml");
        std::fs::write(&explicit, "").unwrap();

        // A ./victory.toml folder would be refused on its own, but
        // VICTORY_CONFIG names the file to read, so it is never checked.
        std::fs::create_dir(&default).unwrap();
        assert!(config_file_or(None, &default).is_err());
        assert_eq!(
            config_file_or(Some(explicit.clone()), &default).unwrap(),
            Some(explicit.clone())
        );
        // A folder holding victory.toml earns the hint, for the default too.
        std::fs::write(default.join("victory.toml"), "").unwrap();
        let err = config_file_or(None, &default).unwrap_err().to_string();
        assert!(
            err.contains(&format!(
                "did you mean {}?",
                default.join("victory.toml").display()
            )),
            "{err}"
        );
        std::fs::remove_dir_all(&default).unwrap();

        #[cfg(unix)]
        {
            // Nor is a ./victory.toml link whose target is gone.
            std::os::unix::fs::symlink(dir.join("gone.toml"), &default).unwrap();
            assert!(config_file_or(None, &default).is_err());
            assert_eq!(
                config_file_or(Some(explicit.clone()), &default).unwrap(),
                Some(explicit.clone())
            );
            std::fs::remove_file(&default).unwrap();

            // A link to a folder is a folder, not "not a regular file".
            let folder = dir.join("folder");
            std::fs::create_dir(&folder).unwrap();
            std::os::unix::fs::symlink(&folder, &default).unwrap();
            let err = config_file_or(None, &default).unwrap_err().to_string();
            assert!(err.contains("is a folder, not a config file"), "{err}");
            std::fs::remove_file(&default).unwrap();

            // A chain of links ending at a file is read, keeping the path
            // as given so relative settings stay anchored beside it.
            let real = dir.join("real.toml");
            std::fs::write(&real, "").unwrap();
            let middle = dir.join("middle.toml");
            std::os::unix::fs::symlink(&real, &middle).unwrap();
            std::os::unix::fs::symlink(&middle, &default).unwrap();
            assert_eq!(
                config_file_or(None, &default).unwrap(),
                Some(default.clone())
            );
            // Breaking the far end of the chain names the next link's
            // target, not the final file.
            std::fs::remove_file(&real).unwrap();
            let err = config_file_or(None, &default).unwrap_err().to_string();
            assert!(
                err.contains(&format!("a link to {}", middle.display())),
                "{err}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn victory_config_with_a_trailing_slash_names_the_file_or_the_link() {
        // A scratch folder under the system temp folder, never the
        // repository or the working directory; removed at the end.
        let dir = std::env::temp_dir().join(format!(
            "victory-config-trailing-slash-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("victory.toml");
        std::fs::write(&file, "").unwrap();
        let gone = dir.join("moved-away.toml");
        let dangling = dir.join("dangling.toml");
        std::os::unix::fs::symlink(&gone, &dangling).unwrap();
        let good = dir.join("good.toml");
        std::os::unix::fs::symlink(&file, &good).unwrap();
        let slash = |p: &Path| PathBuf::from(format!("{}/", p.display()));
        let dot = |p: &Path| PathBuf::from(format!("{}/.", p.display()));

        // A file (or a good link to one) spelled as a folder says what to
        // drop, rather than "Not a directory (os error 20)".
        for (named, spelled) in [
            (&file, slash(&file)),
            (&file, dot(&file)),
            (&good, slash(&good)),
        ] {
            let err = config_file(Some(spelled.clone())).unwrap_err().to_string();
            assert!(
                err.starts_with(&format!("VICTORY_CONFIG points at {}", spelled.display())),
                "{err}"
            );
            assert!(
                err.contains(&format!("drop it: {}", named.display())),
                "{err}"
            );
            assert!(!err.contains("os error"), "{err}");
        }

        // A link that leads nowhere, spelled with a trailing slash, still
        // names its target.
        for spelled in [slash(&dangling), dot(&dangling)] {
            let err = config_file(Some(spelled)).unwrap_err().to_string();
            assert!(
                err.contains(&format!(
                    "a link to {}, which does not exist",
                    gone.display()
                )),
                "{err}"
            );
        }

        // A folder spelled with a trailing slash is still a folder, with
        // the hint; a file below a file is still the OS error, as before.
        let err = config_file(Some(slash(&dir))).unwrap_err().to_string();
        assert!(err.contains("is a folder, not a config file"), "{err}");
        assert!(err.contains("did you mean"), "{err}");
        let blocked = file.join("child.toml");
        let err = config_file(Some(blocked)).unwrap_err();
        assert_eq!(
            err.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::NotADirectory
        );
        // A path without the slash is read as before.
        assert_eq!(config_file(Some(file.clone())).unwrap(), Some(file));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_config_link_that_loops_is_refused_not_skipped() {
        // A scratch folder under the system temp folder, never the
        // repository or the working directory; removed at the end.
        let dir =
            std::env::temp_dir().join(format!("victory-config-loop-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let default = dir.join("victory.toml");

        // A link to itself and a two-link cycle: the links are there, so
        // this is not "no config file", and the loop is not "does not
        // exist" either (that would send the user hunting for a file).
        let other = dir.join("other.toml");
        for (first, second) in [(&default, &default), (&default, &other)] {
            let _ = std::fs::remove_file(&default);
            let _ = std::fs::remove_file(&other);
            std::os::unix::fs::symlink(second, first).unwrap();
            if first != second {
                std::os::unix::fs::symlink(first, second).unwrap();
            }
            let err = config_file_or(None, &default).unwrap_err();
            let io = err.downcast_ref::<std::io::Error>().unwrap();
            assert_ne!(io.kind(), std::io::ErrorKind::NotFound, "{err:#}");
            let text = err.to_string();
            assert!(
                text.starts_with(&format!("checking the config file {}", default.display())),
                "{text}"
            );
            assert!(!text.contains("does not exist"), "{text}");

            // The same through VICTORY_CONFIG, naming the variable.
            let err = config_file(Some(default.clone())).unwrap_err().to_string();
            assert!(
                err.starts_with(&format!("checking VICTORY_CONFIG at {}", default.display())),
                "{err}"
            );

            // VICTORY_CONFIG naming a real file is read without touching
            // the looping default.
            let real = dir.join("real.toml");
            std::fs::write(&real, "").unwrap();
            assert_eq!(
                config_file_or(Some(real.clone()), &default).unwrap(),
                Some(real)
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_config_file_that_cannot_be_checked_is_refused_not_skipped() {
        use std::os::unix::fs::PermissionsExt;
        // A scratch folder under the system temp folder, never the
        // repository or the working directory; removed at the end.
        let dir =
            std::env::temp_dir().join(format!("victory-config-locked-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let locked = dir.join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        let default = locked.join("victory.toml");
        std::fs::write(&default, "").unwrap();
        let real = dir.join("real.toml");
        std::fs::write(&real, "").unwrap();

        // A folder without search permission: whether victory.toml is in
        // it cannot be told, so it is neither "no config file" nor "does
        // not exist", but the permission error with what was being checked.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        // Root (or a CAP_DAC_OVERRIDE container) reads through the mode
        // bits, leaving nothing to check.
        let enforced = std::fs::symlink_metadata(&default).is_err();
        if enforced {
            let err = config_file_or(None, &default).unwrap_err();
            let io = err.downcast_ref::<std::io::Error>().unwrap();
            assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied, "{err:#}");
            let text = err.to_string();
            assert_eq!(
                text,
                format!("checking the config file {}", default.display())
            );

            // The same through VICTORY_CONFIG, naming the variable.
            let err = config_file(Some(default.clone())).unwrap_err();
            assert_eq!(
                err.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::PermissionDenied
            );
            assert_eq!(
                err.to_string(),
                format!("checking VICTORY_CONFIG at {}", default.display())
            );

            // A trailing slash changes nothing: still unreadable, never
            // taken for a file named with a slash.
            let slashed = PathBuf::from(format!("{}/", default.display()));
            let err = config_file(Some(slashed)).unwrap_err();
            assert_eq!(
                err.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::PermissionDenied
            );

            // VICTORY_CONFIG naming a readable file is used without
            // looking at the default that cannot be checked.
            assert_eq!(
                config_file_or(Some(real.clone()), &default).unwrap(),
                Some(real.clone())
            );
        }
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        // Search permission back: the same file is found as before.
        assert_eq!(
            config_file_or(None, &default).unwrap(),
            Some(default.clone())
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_config_file_that_is_found_but_cannot_be_read_names_the_file() {
        use std::os::unix::fs::PermissionsExt;
        // A scratch folder under the system temp folder, never the
        // repository or the working directory; removed at the end.
        let dir = std::env::temp_dir().join(format!(
            "victory-config-unreadable-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("victory.toml");
        std::fs::write(&file, "[paths]\n").unwrap();

        // Readable: the text, and a UTF-8 mark written by Notepad is dropped
        // on the way from the disk too.
        assert_eq!(read_config(&file).unwrap(), "[paths]\n");
        std::fs::write(&file, b"\xEF\xBB\xBF[paths]\n").unwrap();
        assert_eq!(read_config(&file).unwrap(), "[paths]\n");

        // UTF-16 or stray bytes on disk: the fix, under the file's name.
        for bytes in [
            &b"\xFF\xFE[\0p\0"[..],
            &b"[paths]\ngame_dir = \"\xE9\"\n"[..],
        ] {
            std::fs::write(&file, bytes).unwrap();
            let err = read_config(&file).unwrap_err();
            assert_eq!(err.to_string(), format!("reading {}", file.display()));
            assert!(format!("{err:#}").contains("save it as UTF-8"), "{err:#}");
        }

        // No read permission: config_file still finds the file (only the
        // folder's search permission is needed), so the read reports the
        // permission error with the path rather than a bare "Permission
        // denied (os error 13)", and never falls back to the defaults.
        std::fs::write(&file, "").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o000)).unwrap();
        assert_eq!(config_file_or(None, &file).unwrap(), Some(file.clone()));
        // Root (or a CAP_DAC_OVERRIDE container) reads through the mode
        // bits, leaving nothing to check.
        if std::fs::read(&file).is_err() {
            let err = read_config(&file).unwrap_err();
            assert_eq!(
                err.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::PermissionDenied
            );
            assert_eq!(err.to_string(), format!("reading {}", file.display()));
        }
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(read_config(&file).unwrap(), "");

        // A folder put where the file was after it was found: the OS error
        // under the file's name, not a parse of nothing.
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir(&file).unwrap();
        let err = read_config(&file).unwrap_err();
        assert_eq!(err.to_string(), format!("reading {}", file.display()));
        assert_eq!(
            err.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::IsADirectory
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_folder_setting_blocked_by_a_file_is_refused() {
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end.
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-file-not-folder-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(scratch.join("gta")).unwrap();
        let exe = scratch.join("gta").join("default.xex");
        std::fs::write(&exe, b"").unwrap();
        let with = |k: &'static str, v: PathBuf| {
            let cache = scratch.join("cache");
            let data = scratch.join("data");
            move |key: &str| match key {
                _ if key == k => Some(v.clone().into_os_string()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };

        let err = Config::resolve(
            FilePaths::default(),
            None,
            with("VICTORY_GAME_DIR", exe.clone()),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("game folder"), "{err}");
        assert!(err.contains("(from VICTORY_GAME_DIR) is a file"), "{err}");
        let parent = scratch.join("gta");
        assert!(err.contains(&parent.display().to_string()), "{err}");

        let blocked = exe.join("missing").join("saves");
        let err = Config::resolve(
            FilePaths::default(),
            None,
            with("VICTORY_DATA_DIR", blocked.clone()),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains(&format!("{} (from VICTORY_DATA_DIR)", blocked.display())),
            "{err}"
        );
        assert!(
            err.contains(&format!("{} is not a directory", exe.display())),
            "{err}"
        );

        // The same file as the cache or data folder, from the file.
        for (key, file) in [
            (
                "cache",
                FilePaths {
                    cache_dir: Some(exe.clone()),
                    ..FilePaths::default()
                },
            ),
            (
                "data",
                FilePaths {
                    data_dir: Some(exe.clone()),
                    ..FilePaths::default()
                },
            ),
        ] {
            let env = |k: &str| match k {
                "VICTORY_CACHE_DIR" if key == "data" => {
                    Some(scratch.join("cache").into_os_string())
                }
                "VICTORY_DATA_DIR" if key == "cache" => Some(scratch.join("data").into_os_string()),
                _ => None,
            };
            let err = Config::resolve(file, None, env).unwrap_err().to_string();
            assert!(
                err.contains(&format!("the {key} folder")) && err.contains("(from config file)"),
                "{err}"
            );
            assert!(err.contains(&format!("[paths].{key}_dir")), "{err}");
        }

        // A test title naming a file, from the variable or the file.
        let err = Config::resolve(
            FilePaths::default(),
            None,
            with("VICTORY_TEST_TITLES", exe.clone()),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("the test title") && err.contains("(from VICTORY_TEST_TITLES) is a file"),
            "{err}"
        );
        assert!(
            err.contains(&format!("perhaps {}", parent.display())),
            "{err}"
        );
        let titles = FilePaths {
            test_titles: vec![scratch.join("other"), exe.clone()],
            ..FilePaths::default()
        };
        let err = Config::resolve(titles, None, with("VICTORY_GAME_DIR", parent.clone()))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("(from config file) is a file") && err.contains("[paths].test_titles"),
            "{err}"
        );

        // The folder itself, and a folder that does not exist yet, are fine.
        assert!(
            Config::resolve(
                FilePaths::default(),
                None,
                with("VICTORY_TEST_TITLES", parent.clone())
            )
            .is_ok()
        );
        assert!(
            Config::resolve(FilePaths::default(), None, with("VICTORY_GAME_DIR", parent)).is_ok()
        );
        assert!(
            Config::resolve(
                FilePaths::default(),
                None,
                with("VICTORY_GAME_DIR", scratch.join("missing"))
            )
            .is_ok()
        );

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn a_folder_setting_naming_a_device_is_refused() {
        let env = |key: &str| match key {
            "VICTORY_DATA_DIR" => Some(OsString::from("/dev/null")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, env)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("/dev/null (from VICTORY_DATA_DIR) is not a directory"),
            "{err}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_folder_setting_at_or_under_a_socket_device_or_file_dot_dot_is_refused() {
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end. Kept short: a socket path is
        // limited to about a hundred bytes.
        let scratch = std::env::temp_dir().join(format!("vc-sock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(scratch.join("gta")).unwrap();
        let exe = scratch.join("gta").join("default.xex");
        std::fs::write(&exe, b"").unwrap();
        let sock = scratch.join("s");
        let _listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        let env = |k: &'static str, v: PathBuf| {
            let cache = scratch.join("cache");
            let data = scratch.join("data");
            move |key: &str| match key {
                _ if key == k => Some(v.clone().into_os_string()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let refused = |k: &'static str, v: PathBuf| {
            Config::resolve(FilePaths::default(), None, env(k, v))
                .unwrap_err()
                .to_string()
        };

        // A socket is neither a folder nor a file: it gets the plain "not a
        // directory" error, not the "is a file" hint pointing at its parent.
        let err = refused("VICTORY_GAME_DIR", sock.clone());
        assert!(
            err.contains(&format!(
                "the game folder {} (from VICTORY_GAME_DIR) is not a directory; set \
                 VICTORY_GAME_DIR",
                sock.display()
            )),
            "{err}"
        );
        assert!(
            !err.contains("is a file") && !err.contains("perhaps"),
            "{err}"
        );

        // A folder still to create below a socket or a device names the
        // blocking path, as it does for a file.
        let below = sock.join("victory");
        let err = refused("VICTORY_CACHE_DIR", below.clone());
        assert!(
            err.contains(&format!(
                "the cache folder {} (from VICTORY_CACHE_DIR) cannot be a folder because {} is \
                 not a directory",
                below.display(),
                sock.display()
            )),
            "{err}"
        );
        let err = refused("VICTORY_DATA_DIR", PathBuf::from("/dev/null/victory"));
        assert!(
            err.contains("cannot be a folder because /dev/null is not a directory"),
            "{err}"
        );

        // `..` after a file does not climb back out of it: the OS refuses
        // `default.xex/..` just as it refuses `default.xex/saves`, so the
        // file is named as the blocker and nothing is created next to it.
        let climbed = exe.join("..").join("saves");
        let err = refused("VICTORY_DATA_DIR", climbed.clone());
        assert!(
            err.contains(&format!(
                "the data folder {} (from VICTORY_DATA_DIR) cannot be a folder because {} is \
                 not a directory",
                climbed.display(),
                exe.display()
            )),
            "{err}"
        );
        assert!(!scratch.join("gta").join("saves").exists());

        // The same `..` after a real folder is fine.
        assert!(
            Config::resolve(
                FilePaths::default(),
                None,
                env(
                    "VICTORY_GAME_DIR",
                    scratch.join("gta").join("..").join("gta")
                )
            )
            .is_ok()
        );

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn a_folder_setting_at_under_or_linked_to_a_named_pipe_is_refused() {
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end. The pipe is only looked at, never
        // opened: opening a pipe with no writer would hang the check.
        let scratch = std::env::temp_dir().join(format!("vc-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        let fifo = scratch.join("p");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .is_ok_and(|s| s.success());
        if !made {
            // No `mkfifo` on this host; nothing to check.
            std::fs::remove_dir_all(&scratch).unwrap();
            return;
        }
        let link = scratch.join("to-pipe");
        std::os::unix::fs::symlink("p", &link).unwrap();
        let env = |k: &'static str, v: PathBuf| {
            let cache = scratch.join("cache");
            let data = scratch.join("data");
            move |key: &str| match key {
                _ if key == k => Some(v.clone().into_os_string()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let refused = |k: &'static str, v: PathBuf| {
            Config::resolve(FilePaths::default(), None, env(k, v))
                .unwrap_err()
                .to_string()
        };

        // The pipe itself, named plainly, with a trailing slash, or through a
        // link to it, is neither a folder nor a file: the plain "not a
        // directory" error, without the "is a file" hint.
        for (k, v) in [
            ("VICTORY_GAME_DIR", fifo.clone()),
            ("VICTORY_CACHE_DIR", scratch.join("p/")),
            ("VICTORY_DATA_DIR", link.clone()),
        ] {
            let err = refused(k, v.clone());
            assert!(
                err.contains(&format!("{} (from {k}) is not a directory", v.display())),
                "{err}"
            );
            assert!(
                !err.contains("is a file") && !err.contains("leads nowhere"),
                "{err}"
            );
        }

        // A folder still to create below the pipe, or below the link to it,
        // names the path as written as the blocker, and nothing is created.
        for blocker in [&fifo, &link] {
            let below = blocker.join("victory");
            let err = refused("VICTORY_CACHE_DIR", below.clone());
            assert!(
                err.contains(&format!(
                    "the cache folder {} (from VICTORY_CACHE_DIR) cannot be a folder because \
                     {} is not a directory",
                    below.display(),
                    blocker.display()
                )),
                "{err}"
            );
        }
        assert!(!scratch.join("cache").exists());

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn a_folder_setting_through_a_link_that_leads_nowhere_is_refused() {
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end.
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-dangling-folder-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(scratch.join("real")).unwrap();
        let gone = scratch.join("moved-away");
        let dangling = scratch.join("dangling");
        std::os::unix::fs::symlink(&gone, &dangling).unwrap();
        let looped = scratch.join("loop");
        std::os::unix::fs::symlink(&looped, &looped).unwrap();
        let good = scratch.join("good");
        std::os::unix::fs::symlink(scratch.join("real"), &good).unwrap();
        let env = |cache: PathBuf, data: PathBuf| {
            move |key: &str| match key {
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let resolve = |cache: PathBuf, data: PathBuf| {
            Config::resolve(FilePaths::default(), None, env(cache, data))
        };

        // A link to a folder is followed, as create_dir_all would, and a
        // missing folder under it is fine.
        let cfg = resolve(good.join("cache"), good.join("data")).unwrap();
        assert_eq!(cfg.cache_dir, good.join("cache"));
        assert!(resolve(good.clone(), scratch.join("data")).is_ok());

        // The folder itself is a link whose target is gone: named with its
        // target and setting, not passed over as a folder still to create.
        let err = resolve(dangling.clone(), scratch.join("data"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&format!(
                "the cache folder {} (from VICTORY_CACHE_DIR) is a link to {} that leads nowhere",
                dangling.display(),
                gone.display()
            )),
            "{err}"
        );
        assert!(err.contains("set VICTORY_CACHE_DIR or"), "{err}");

        // A dangling link further up blocks the folder below it.
        let below = dangling.join("victory").join("data");
        let err = resolve(scratch.join("cache"), below.clone())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&format!(
                "the data folder {} (from VICTORY_DATA_DIR) cannot be a folder because {} is a link",
                below.display(),
                dangling.display()
            )),
            "{err}"
        );

        // A link to itself is refused the same way, not skipped.
        let err = resolve(looped.clone(), scratch.join("data"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("leads nowhere"), "{err}");
        assert!(err.contains("loops"), "{err}");

        // Nothing was created in place of the links.
        assert!(!gone.exists());
        assert!(
            dangling
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_symlink()
        );
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn a_trailing_slash_or_dot_does_not_hide_a_file_or_a_link_that_leads_nowhere() {
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end.
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-trailing-slash-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(scratch.join("gta")).unwrap();
        std::fs::create_dir_all(scratch.join("real")).unwrap();
        let exe = scratch.join("gta").join("default.xex");
        std::fs::write(&exe, b"").unwrap();
        let gone = scratch.join("moved-away");
        let dangling = scratch.join("dangling");
        std::os::unix::fs::symlink(&gone, &dangling).unwrap();
        let looped = scratch.join("loop");
        std::os::unix::fs::symlink(&looped, &looped).unwrap();
        let good = scratch.join("good");
        std::os::unix::fs::symlink(scratch.join("real"), &good).unwrap();
        // `<path>/` and `<path>/.`, spelled as a user would type them.
        let slash = |p: &Path| PathBuf::from(format!("{}/", p.display()));
        let dot = |p: &Path| PathBuf::from(format!("{}/.", p.display()));
        let env = |k: &'static str, v: PathBuf| {
            let cache = scratch.join("cache");
            let data = scratch.join("data");
            move |key: &str| match key {
                _ if key == k => Some(v.clone().into_os_string()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let resolve =
            |k: &'static str, v: PathBuf| Config::resolve(FilePaths::default(), None, env(k, v));
        let refused = |k: &'static str, v: PathBuf| resolve(k, v).unwrap_err().to_string();

        // The game's executable with a trailing slash or `/.` is still a
        // file, not a folder still to create next to it.
        for spelled in [slash(&exe), dot(&exe)] {
            let err = refused("VICTORY_GAME_DIR", spelled.clone());
            assert!(
                err.contains(&format!(
                    "the game folder {} (from VICTORY_GAME_DIR) is a file, not a folder",
                    spelled.display()
                )),
                "{err}"
            );
            assert!(
                err.contains(&format!("perhaps {}", exe.parent().unwrap().display())),
                "{err}"
            );
        }

        // A link whose target is gone, named with a trailing slash: the OS
        // follows the link for `<link>/`, so it must be looked at without
        // the slash, or it passes as a folder to create and the first use
        // fails with "File exists".
        for spelled in [slash(&dangling), dot(&dangling)] {
            let err = refused("VICTORY_CACHE_DIR", spelled.clone());
            assert!(
                err.contains(&format!(
                    "the cache folder {} (from VICTORY_CACHE_DIR) is a link to {} that leads \
                     nowhere",
                    spelled.display(),
                    gone.display()
                )),
                "{err}"
            );
        }
        let err = refused("VICTORY_DATA_DIR", slash(&looped));
        assert!(err.contains("leads nowhere"), "{err}");
        assert!(err.contains("loops"), "{err}");
        let title = slash(&dangling);
        let err = refused("VICTORY_TEST_TITLES", title.clone());
        assert!(
            err.contains(&format!(
                "the test title {} (from VICTORY_TEST_TITLES) is a link",
                title.display()
            )),
            "{err}"
        );

        // A folder below them, spelled with a trailing slash, names the
        // blocking path as before.
        let below = slash(&dangling.join("victory"));
        let err = refused("VICTORY_DATA_DIR", below.clone());
        assert!(
            err.contains(&format!(
                "the data folder {} (from VICTORY_DATA_DIR) cannot be a folder because {} is a link",
                below.display(),
                dangling.display()
            )),
            "{err}"
        );

        // A trailing slash on a real folder, or on a good link to one, is
        // still fine, and the setting is kept as written.
        let cfg = resolve("VICTORY_CACHE_DIR", slash(&good)).unwrap();
        assert_eq!(cfg.cache_dir, slash(&good));
        assert!(resolve("VICTORY_GAME_DIR", slash(&scratch.join("gta"))).is_ok());
        assert!(resolve("VICTORY_DATA_DIR", slash(&scratch.join("new"))).is_ok());

        // Nothing was created in place of the links.
        assert!(!gone.exists());
        assert!(!scratch.join("new").exists());
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn a_folder_setting_through_a_link_to_a_file_is_refused() {
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end.
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-link-to-file-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(scratch.join("gta")).unwrap();
        let exe = scratch.join("gta").join("default.xex");
        std::fs::write(&exe, b"").unwrap();
        let to_file = scratch.join("to-file");
        std::os::unix::fs::symlink(&exe, &to_file).unwrap();
        // A link to a link that leads nowhere is followed to its end.
        let dangling = scratch.join("dangling");
        std::os::unix::fs::symlink(scratch.join("moved-away"), &dangling).unwrap();
        let chained = scratch.join("chained");
        std::os::unix::fs::symlink(&dangling, &chained).unwrap();
        let env = |k: &'static str, v: PathBuf| {
            let cache = scratch.join("cache");
            let data = scratch.join("data");
            move |key: &str| match key {
                _ if key == k => Some(v.clone().into_os_string()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let resolve = |k: &'static str, v: PathBuf| {
            Config::resolve(FilePaths::default(), None, env(k, v))
                .unwrap_err()
                .to_string()
        };

        // The folder itself is a link to a file: a file, not a link that
        // leads nowhere, and the hint names the folder holding the link.
        let err = resolve("VICTORY_CACHE_DIR", to_file.clone());
        assert!(
            err.contains(&format!(
                "the cache folder {} (from VICTORY_CACHE_DIR) is a file, not a folder",
                to_file.display()
            )),
            "{err}"
        );
        assert!(!err.contains("leads nowhere"), "{err}");
        let err = resolve("VICTORY_GAME_DIR", to_file.clone());
        assert!(
            err.contains(&format!("perhaps {}", scratch.display())),
            "{err}"
        );

        // A folder under a link to a file names the link, not its target.
        let below = to_file.join("saves");
        let err = resolve("VICTORY_DATA_DIR", below.clone());
        assert!(
            err.contains(&format!(
                "the data folder {} (from VICTORY_DATA_DIR) cannot be a folder because {} \
                 is not a directory",
                below.display(),
                to_file.display()
            )),
            "{err}"
        );

        // A chain ending in a link that leads nowhere is refused at the
        // first link, with that link's own target.
        let err = resolve("VICTORY_DATA_DIR", chained.clone());
        assert!(
            err.contains(&format!(
                "{} (from VICTORY_DATA_DIR) is a link to {} that leads nowhere",
                chained.display(),
                dangling.display()
            )),
            "{err}"
        );

        // Nothing was created or replaced.
        assert!(exe.is_file());
        assert!(!scratch.join("moved-away").exists());
        assert!(!scratch.join("cache").exists() && !scratch.join("data").exists());
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn a_subfolder_blocked_by_a_link_that_leads_nowhere_is_an_error() {
        // The settings are checked when resolved; a link made inside the
        // cache afterwards is met by `create_dir_all`, which must fail
        // naming the path rather than succeed or replace the link.
        let scratch =
            std::env::temp_dir().join(format!("victory-config-subdir-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        let cfg = Config {
            cache_dir: scratch.join("cache"),
            data_dir: scratch.join("data"),
            ..Default::default()
        };
        if ensure_outside_repo(&scratch).is_err() {
            return;
        }
        std::fs::create_dir_all(&cfg.cache_dir).unwrap();
        let link = cfg.cache_dir.join("shaders");
        std::os::unix::fs::symlink(scratch.join("moved-away"), &link).unwrap();
        for sub in ["shaders", "shaders/inner"] {
            let err = format!("{:#}", cfg.cache_subdir(sub).unwrap_err());
            assert!(err.contains("creating"), "{sub}: {err}");
            assert!(err.contains(&link.display().to_string()), "{sub}: {err}");
        }
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
        assert!(!scratch.join("moved-away").exists());
        // A link to a real folder is followed.
        std::fs::create_dir_all(scratch.join("real")).unwrap();
        let good = cfg.cache_dir.join("good");
        std::os::unix::fs::symlink(scratch.join("real"), &good).unwrap();
        let dir = cfg.cache_subdir("good/inner").unwrap();
        assert_eq!(dir, good.join("inner"));
        assert!(scratch.join("real").join("inner").is_dir());
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn folder_setting_links_with_relative_targets_resolve_from_the_link() {
        // `ln -s ../real nested/up` and friends: a relative target is read
        // from the link's own folder, never the working directory, and a
        // link that leads nowhere is reported with its target as written.
        // A scratch folder under the system temp folder, never the
        // repository; removed at the end.
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-relative-link-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(scratch.join("real")).unwrap();
        std::fs::create_dir_all(scratch.join("nested")).unwrap();
        std::fs::write(scratch.join("default.xex"), b"").unwrap();
        let good = scratch.join("good");
        std::os::unix::fs::symlink("real", &good).unwrap();
        // `../real` from the working directory is not there; from `nested`
        // it is the real folder.
        let up = scratch.join("nested").join("up");
        std::os::unix::fs::symlink("../real", &up).unwrap();
        let dangling = scratch.join("dangling");
        std::os::unix::fs::symlink("moved-away", &dangling).unwrap();
        // A relative chain: `chained` -> `dangling` -> `moved-away`.
        let chained = scratch.join("chained");
        std::os::unix::fs::symlink("dangling", &chained).unwrap();
        let to_file = scratch.join("nested").join("to-file");
        std::os::unix::fs::symlink("../default.xex", &to_file).unwrap();
        let resolve = |cache: PathBuf| {
            let data = scratch.join("data");
            Config::resolve(FilePaths::default(), None, move |key: &str| match key {
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            })
        };

        // Links to the real folder, at the leaf or above a missing folder.
        for cache in [
            good.clone(),
            good.join("cache"),
            up.clone(),
            up.join("cache"),
        ] {
            let cfg = resolve(cache.clone()).unwrap();
            assert_eq!(cfg.cache_dir, cache);
        }

        // The target is named as the link holds it, not made absolute.
        let err = resolve(dangling.clone()).unwrap_err().to_string();
        assert!(
            err.contains(&format!(
                "the cache folder {} (from VICTORY_CACHE_DIR) is a link to moved-away that leads \
                 nowhere",
                dangling.display()
            )),
            "{err}"
        );
        let below = chained.join("cache");
        let err = resolve(below.clone()).unwrap_err().to_string();
        assert!(
            err.contains(&format!(
                "the cache folder {} (from VICTORY_CACHE_DIR) cannot be a folder because {} is a \
                 link to dangling that leads nowhere",
                below.display(),
                chained.display()
            )),
            "{err}"
        );

        // A relative link to a file is a file, at the leaf or above.
        let err = resolve(to_file.clone()).unwrap_err().to_string();
        assert!(err.contains("is a file, not a folder"), "{err}");
        let err = resolve(to_file.join("cache")).unwrap_err().to_string();
        assert!(
            err.contains(&format!("because {} is not a directory", to_file.display())),
            "{err}"
        );

        // Nothing was created in place of the links or through them.
        assert!(!scratch.join("moved-away").exists());
        assert!(!scratch.join("real").join("cache").exists());
        assert!(!scratch.join("data").exists());
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    fn cache_or_data_at_the_root_or_home_is_refused() {
        let home = PathBuf::from(abs("home")).join("me");
        let root = PathBuf::from(abs("x")).parent().unwrap().to_path_buf();
        for (key, value, what) in [
            ("VICTORY_CACHE_DIR", home.clone(), "your home folder"),
            // `$HOME/$APP` with `APP` unset.
            ("VICTORY_DATA_DIR", home.join(""), "your home folder"),
            (
                "VICTORY_CACHE_DIR",
                home.join("games").join(".."),
                "your home folder",
            ),
            (
                "VICTORY_DATA_DIR",
                PathBuf::from(abs("home")),
                "a folder holding",
            ),
            ("VICTORY_CACHE_DIR", root.clone(), "the root of the drive"),
            (
                "VICTORY_DATA_DIR",
                root.join("tmp").join(".."),
                "the root of the drive",
            ),
        ] {
            let env = |k: &str| match k {
                "HOME" => Some(home.clone().into()),
                k if k == key => Some(value.clone().into()),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(what), "{key}={}: {err}", value.display());
            assert!(err.contains(&format!("(from {key})")), "{err}");
            assert!(err.contains(&format!("set {key}")), "{err}");
        }

        // From the config file too, and without a home only the root counts.
        let file = FilePaths {
            cache_dir: Some(root.clone()),
            ..FilePaths::default()
        };
        let no_home = |k: &str| (k == "XDG_DATA_HOME").then(|| abs("share"));
        let err = Config::resolve(file, None, no_home)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("the cache folder") && err.contains("(from config file)"),
            "{err}"
        );

        // A folder inside home, a sibling of it and a folder whose name
        // merely starts with home's are fine.
        for value in [
            home.join("victory"),
            PathBuf::from(abs("home")).join("other"),
            PathBuf::from(abs("home")).join("me2"),
        ] {
            let env = |k: &str| match k {
                "HOME" => Some(home.clone().into()),
                "VICTORY_CACHE_DIR" => Some(value.clone().into()),
                "VICTORY_DATA_DIR" => Some(home.join("data").into()),
                _ => None,
            };
            assert!(
                Config::resolve(FilePaths::default(), None, env).is_ok(),
                "{}",
                value.display()
            );
        }
    }

    #[test]
    fn a_tilde_spelling_of_home_or_above_is_refused_as_a_cache_or_data_folder() {
        // `~` is expanded before the folder checks, so every spelling that
        // lands on home (or a folder holding it) is caught, from a variable
        // or the config file, and whether home came from HOME (with or
        // without a trailing slash) or the USERPROFILE fallback.
        let home = PathBuf::from(abs("home")).join("me");
        let homes: [(&str, OsString); 3] = [
            ("HOME", home.clone().into()),
            ("HOME", home.join("").into()),
            ("USERPROFILE", home.clone().into()),
        ];
        for (home_key, home_value) in &homes {
            for (value, what) in [
                ("~", "your home folder"),
                ("~/", "your home folder"),
                ("~/.", "your home folder"),
                ("~/games/..", "your home folder"),
                ("~/..", "a folder holding"),
            ] {
                for key in ["VICTORY_CACHE_DIR", "VICTORY_DATA_DIR"] {
                    let env = |k: &str| match k {
                        k if k == *home_key => Some(home_value.clone()),
                        k if k == key => Some(OsString::from(value)),
                        // USERPROFILE alone gives no default base.
                        "XDG_CACHE_HOME" => Some(abs("xc")),
                        "XDG_DATA_HOME" => Some(abs("xd")),
                        _ => None,
                    };
                    let err = Config::resolve(FilePaths::default(), None, env)
                        .unwrap_err()
                        .to_string();
                    assert!(err.contains(what), "{home_key} {key}={value}: {err}");
                    assert!(err.contains(&format!("(from {key})")), "{err}");
                }
                let text = format!("[paths]\ncache_dir = '{value}'\n");
                let parsed: FileConfig = toml::from_str(&text).unwrap();
                let file = parsed
                    .anchored_at(&PathBuf::from(abs("configs")), Some(&home))
                    .unwrap();
                let env = |k: &str| match k {
                    k if k == *home_key => Some(home_value.clone()),
                    "VICTORY_DATA_DIR" => Some(home.join("victory-data").into()),
                    "XDG_CACHE_HOME" => Some(abs("xc")),
                    _ => None,
                };
                let err = Config::resolve(file, None, env).unwrap_err().to_string();
                assert!(err.contains(what), "{home_key} file {value}: {err}");
                assert!(err.contains("(from config file)"), "{err}");
            }
        }

        // A folder under home spelled with `~` stays fine.
        let env = |k: &str| match k {
            "HOME" => Some(home.join("").into()),
            "VICTORY_CACHE_DIR" => Some("~/victory/../victory-cache".into()),
            "VICTORY_DATA_DIR" => Some("~/./victory-data".into()),
            _ => None,
        };
        let cfg = Config::resolve(FilePaths::default(), None, env).unwrap();
        assert!(
            cfg.cache_dir.starts_with(&home),
            "{}",
            cfg.cache_dir.display()
        );
        assert!(
            cfg.data_dir.starts_with(&home),
            "{}",
            cfg.data_dir.display()
        );
    }

    #[test]
    fn cache_or_data_at_a_shared_user_folder_is_refused() {
        let home = PathBuf::from(abs("home")).join("me");
        let xdg_cache = PathBuf::from(abs("xc"));
        let xdg_config = PathBuf::from(abs("cfg"));
        for (key, value, what) in [
            // `$XDG_CACHE_HOME` with `/victory` left off.
            (
                "VICTORY_CACHE_DIR",
                xdg_cache.clone(),
                "is your shared XDG_CACHE_HOME",
            ),
            (
                "VICTORY_CACHE_DIR",
                xdg_cache.join(""),
                "is your shared XDG_CACHE_HOME",
            ),
            (
                "VICTORY_DATA_DIR",
                xdg_config.clone(),
                "is your shared XDG_CONFIG_HOME",
            ),
            (
                "VICTORY_DATA_DIR",
                home.join(".local/share"),
                "is your shared ~/.local/share",
            ),
            (
                "VICTORY_CACHE_DIR",
                home.join(".config"),
                "is your shared ~/.config",
            ),
            (
                "VICTORY_DATA_DIR",
                home.join(".local"),
                "holds your shared ~/.local/share",
            ),
            (
                "VICTORY_CACHE_DIR",
                home.join(".cache").join("x").join(".."),
                "is your shared ~/.cache",
            ),
        ] {
            let env = |k: &str| match k {
                "HOME" => Some(home.clone().into()),
                "XDG_CACHE_HOME" => Some(xdg_cache.clone().into()),
                "XDG_CONFIG_HOME" => Some(xdg_config.clone().into()),
                k if k == key => Some(value.clone().into()),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(what), "{key}={}: {err}", value.display());
            assert!(err.contains(&format!("(from {key})")), "{err}");
            assert!(err.contains(&format!("set {key}")), "{err}");
        }

        // A folder holding one, from the config file, says which and how.
        let file = FilePaths {
            data_dir: Some(PathBuf::from(abs("nest"))),
            ..FilePaths::default()
        };
        let nest = PathBuf::from(abs("nest"));
        let env = |k: &str| match k {
            "HOME" => Some(home.clone().into()),
            "XDG_STATE_HOME" => Some(nest.join("state").into()),
            _ => None,
        };
        let err = Config::resolve(file, None, env).unwrap_err().to_string();
        assert!(
            err.contains("the data folder")
                && err.contains("(from config file)")
                && err.contains("holds your shared XDG_STATE_HOME")
                && err.contains(&nest.join("state").join("victory").display().to_string()),
            "{err}"
        );

        // A folder of Victory's own inside each, and a sibling whose name
        // merely starts with one, are fine; so are the defaults.
        for value in [
            xdg_cache.join("victory"),
            home.join(".config/victory"),
            home.join(".config2"),
            PathBuf::from(abs("xcache")),
        ] {
            let env = |k: &str| match k {
                "HOME" => Some(home.clone().into()),
                "XDG_CACHE_HOME" => Some(xdg_cache.clone().into()),
                "VICTORY_CACHE_DIR" => Some(value.clone().into()),
                _ => None,
            };
            assert!(
                Config::resolve(FilePaths::default(), None, env).is_ok(),
                "{}",
                value.display()
            );
        }
        let env = |k: &str| match k {
            "HOME" => Some(home.clone().into()),
            "XDG_CACHE_HOME" => Some(xdg_cache.clone().into()),
            "XDG_CONFIG_HOME" => Some(xdg_config.clone().into()),
            _ => None,
        };
        assert!(Config::resolve(FilePaths::default(), None, env).is_ok());
    }

    #[test]
    fn data_inside_the_cache_is_refused() {
        let env = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(abs("v")),
            "VICTORY_DATA_DIR" => Some(PathBuf::from(abs("v")).join("data").into()),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("inside the cache folder"), "{err}");

        // The same folder for both counts as inside.
        let same = |k: &str| match k {
            "VICTORY_CACHE_DIR" | "VICTORY_DATA_DIR" => Some(abs("v")),
            _ => None,
        };
        assert!(Config::resolve(FilePaths::default(), None, same).is_err());

        // A sibling whose name merely starts with the cache's is fine, and so
        // is the cache inside the data folder.
        let sibling = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(abs("v")),
            "VICTORY_DATA_DIR" => Some(abs("v-data")),
            _ => None,
        };
        assert!(Config::resolve(FilePaths::default(), None, sibling).is_ok());
        let cache_in_data = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(PathBuf::from(abs("v")).join("cache").into()),
            "VICTORY_DATA_DIR" => Some(abs("v")),
            _ => None,
        };
        assert!(Config::resolve(FilePaths::default(), None, cache_in_data).is_ok());
    }

    #[test]
    fn data_inside_the_user_cache_is_refused() {
        let home = PathBuf::from(abs("home"));
        let run = |vars: Vec<(&'static str, OsString)>| {
            let env = move |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
            Config::resolve(FilePaths::default(), None, env)
        };
        // An explicit data folder under ~/.cache, apart from Victory's cache.
        let err = run(vec![
            ("HOME", home.clone().into()),
            ("VICTORY_CACHE_DIR", abs("c")),
            ("VICTORY_DATA_DIR", home.join(".cache").join("saves").into()),
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("inside your cache folder"), "{err}");
        assert!(err.contains("(from VICTORY_DATA_DIR)"), "{err}");
        assert!(err.contains("~/.cache"), "{err}");

        // XDG_CACHE_HOME wins over ~/.cache, through `..` too.
        let err = run(vec![
            ("HOME", home.clone().into()),
            ("XDG_CACHE_HOME", abs("xc")),
            (
                "VICTORY_DATA_DIR",
                PathBuf::from(abs("v"))
                    .join("..")
                    .join("xc")
                    .join("d")
                    .into(),
            ),
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("(XDG_CACHE_HOME)"), "{err}");
        // ...and then ~/.cache is not the user cache.
        assert!(
            run(vec![
                ("HOME", home.clone().into()),
                ("XDG_CACHE_HOME", abs("xc")),
                ("VICTORY_DATA_DIR", home.join(".cache").join("d").into()),
            ])
            .is_ok()
        );

        // XDG_DATA_HOME under XDG_CACHE_HOME moves the default data there.
        let err = run(vec![
            ("XDG_CACHE_HOME", abs("xc")),
            (
                "XDG_DATA_HOME",
                PathBuf::from(abs("xc")).join("share").into(),
            ),
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("the default data folder"), "{err}");
        assert!(err.contains("XDG_DATA_HOME is inside it"), "{err}");

        // A sibling such as ~/.cache-old, the usual defaults, and an
        // XDG_CACHE_HOME that is the home folder itself are all fine.
        assert!(
            run(vec![
                ("HOME", home.clone().into()),
                ("VICTORY_DATA_DIR", home.join(".cache-old").into()),
            ])
            .is_ok()
        );
        assert!(run(vec![("HOME", home.clone().into())]).is_ok());
        assert!(
            run(vec![
                ("HOME", home.clone().into()),
                ("XDG_CACHE_HOME", home.clone().into()),
                ("VICTORY_CACHE_DIR", abs("c")),
            ])
            .is_ok()
        );
    }

    #[test]
    fn relative_paths_are_joined_to_the_working_directory() {
        let cwd = PathBuf::from(abs("v"));
        // Started from the cache folder, `VICTORY_DATA_DIR=data` is inside it.
        let env = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(abs("v")),
            "VICTORY_DATA_DIR" => Some(OsString::from("data")),
            _ => None,
        };
        let err = Config::resolve_in(FilePaths::default(), None, env, Some(&cwd))
            .unwrap_err()
            .to_string();
        assert!(err.contains("inside the cache folder"), "{err}");

        // Elsewhere, every relative value (variable or `./victory.toml`) is
        // reported as the folder it names.
        let file = FilePaths {
            game_dir: Some("game".into()),
            test_titles: vec!["t".into()],
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(OsString::from("cache")),
            "VICTORY_DATA_DIR" => Some(OsString::from("data")),
            _ => None,
        };
        let cfg = Config::resolve_in(file, None, env, Some(&cwd)).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), cwd.join("game"));
        assert_eq!(cfg.cache_dir, cwd.join("cache"));
        assert_eq!(cfg.data_dir, cwd.join("data"));
        assert_eq!(cfg.test_titles, vec![cwd.join("t")]);
        assert_eq!(cfg.origins.cache_dir, Origin::Env("VICTORY_CACHE_DIR"));
    }

    #[test]
    fn data_inside_the_cache_ignores_dot_dot() {
        let with = |cache: PathBuf, data: PathBuf| {
            move |k: &str| match k {
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let v = PathBuf::from(abs("v"));
        // `v/cache/../data` is `v/data`, a sibling of the cache: allowed.
        let sibling = with(v.join("cache"), v.join("cache").join("..").join("data"));
        assert!(Config::resolve(FilePaths::default(), None, sibling).is_ok());
        // A cache of `v/tmp/..` is `v`, which holds `v/./data`: refused.
        let inside = with(v.join("tmp").join(".."), v.join(".").join("data"));
        let err = Config::resolve(FilePaths::default(), None, inside)
            .unwrap_err()
            .to_string();
        assert!(err.contains("inside the cache folder"), "{err}");
    }

    #[test]
    fn game_folder_is_kept_apart_from_cache_and_data() {
        let with = |game: PathBuf, cache: PathBuf, data: PathBuf| {
            move |k: &str| match k {
                "VICTORY_GAME_DIR" => Some(game.clone().into_os_string()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let v = PathBuf::from(abs("v"));
        let err = |env| {
            Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string()
        };

        // The game inside (or equal to) the cache would go with it.
        let e = err(with(v.join("game"), v.clone(), v.join("..").join("data")));
        assert!(e.contains("game folder"), "{e}");
        assert!(e.contains("inside the cache folder"), "{e}");
        assert!(e.contains("(from VICTORY_GAME_DIR)"), "{e}");
        let e = err(with(v.clone(), v.clone(), PathBuf::from(abs("d"))));
        assert!(e.contains("inside the cache folder"), "{e}");

        // The cache or the data inside the game would write into the dump.
        let e = err(with(v.clone(), v.join("cache"), PathBuf::from(abs("d"))));
        assert!(e.contains("cache folder"), "{e}");
        assert!(e.contains("inside the game folder"), "{e}");
        let e = err(with(v.clone(), PathBuf::from(abs("c")), v.join("saves")));
        assert!(e.contains("data folder"), "{e}");
        assert!(e.contains("(from VICTORY_DATA_DIR)"), "{e}");
        // `.` and `..` are folded first: `v/../c` is outside `v`.
        let e = err(with(
            v.join("."),
            v.join("x").join("..").join("c"),
            v.join("..").join("d"),
        ));
        assert!(e.contains("inside the game folder"), "{e}");

        // Siblings are fine, including one whose name starts with the game's.
        let ok = with(
            v.clone(),
            PathBuf::from(abs("v-cache")),
            PathBuf::from(abs("v-data")),
        );
        assert!(Config::resolve(FilePaths::default(), None, ok).is_ok());
        let ok = with(
            v.join("game"),
            v.join("cache"),
            v.join("game").join("..").join("data"),
        );
        assert!(Config::resolve(FilePaths::default(), None, ok).is_ok());

        // Without a game folder there is nothing to check.
        let none = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(abs("c")),
            "VICTORY_DATA_DIR" => Some(abs("d")),
            _ => None,
        };
        assert!(Config::resolve(FilePaths::default(), None, none).is_ok());
    }

    #[test]
    fn test_titles_are_kept_apart_from_cache_and_data() {
        let with = |titles: Vec<PathBuf>, cache: PathBuf, data: PathBuf| {
            move |k: &str| match k {
                "VICTORY_TEST_TITLES" => Some(std::env::join_paths(&titles).unwrap()),
                "VICTORY_CACHE_DIR" => Some(cache.clone().into_os_string()),
                "VICTORY_DATA_DIR" => Some(data.clone().into_os_string()),
                _ => None,
            }
        };
        let x = PathBuf::from(abs("x"));
        let err = |env| {
            Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string()
        };
        let other = PathBuf::from(abs("t"));

        // A title inside (or equal to) the cache would be deleted with it;
        // the second entry is the one named.
        let e = err(with(
            vec![other.clone(), x.join("title-a")],
            x.clone(),
            PathBuf::from(abs("d")),
        ));
        assert!(e.contains("test title"), "{e}");
        assert!(e.contains("title-a"), "{e}");
        assert!(e.contains("inside the cache folder"), "{e}");
        assert!(e.contains("(from VICTORY_TEST_TITLES)"), "{e}");
        let e = err(with(vec![x.clone()], x.clone(), PathBuf::from(abs("d"))));
        assert!(e.contains("inside the cache folder"), "{e}");

        // The cache or data inside a title would write into that dump.
        let e = err(with(
            vec![x.clone()],
            x.join("cache"),
            PathBuf::from(abs("d")),
        ));
        assert!(e.contains("cache folder"), "{e}");
        assert!(e.contains("inside the test title"), "{e}");
        let e = err(with(
            vec![other.clone(), x.clone()],
            PathBuf::from(abs("c")),
            x.join("saves"),
        ));
        assert!(e.contains("data folder"), "{e}");
        assert!(e.contains("(from VICTORY_DATA_DIR)"), "{e}");
        // `.` and `..` are folded first.
        let e = err(with(
            vec![x.join(".")],
            x.join("y").join("..").join("c"),
            PathBuf::from(abs("d")),
        ));
        assert!(e.contains("inside the test title"), "{e}");

        // Siblings are fine, including names that start with the title's.
        let ok = with(
            vec![x.clone(), other],
            PathBuf::from(abs("x-cache")),
            x.join("..").join("x-data"),
        );
        assert!(Config::resolve(FilePaths::default(), None, ok).is_ok());

        // Titles from the file are named as such.
        let file = FilePaths {
            test_titles: vec![x.join("t")],
            ..FilePaths::default()
        };
        let env = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(abs("x")),
            "VICTORY_DATA_DIR" => Some(abs("d")),
            _ => None,
        };
        let e = Config::resolve(file, None, env).unwrap_err().to_string();
        assert!(e.contains("(from config file)"), "{e}");
    }

    #[test]
    fn test_titles_must_be_distinct_and_not_the_game_folder() {
        let x = PathBuf::from(abs("x"));
        let with = |titles: Vec<PathBuf>, game: Option<PathBuf>| {
            move |k: &str| match k {
                "VICTORY_TEST_TITLES" => Some(std::env::join_paths(&titles).unwrap()),
                "VICTORY_GAME_DIR" => game.clone().map(PathBuf::into_os_string),
                "VICTORY_CACHE_DIR" => Some(abs("c")),
                "VICTORY_DATA_DIR" => Some(abs("d")),
                _ => None,
            }
        };
        let err = |env| {
            Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string()
        };

        // The same title twice, also written with `.`/`..`; the later
        // entry is named, with the earlier spelling.
        let a = x.join("a");
        let e = err(with(vec![a.clone(), x.join("b"), a.clone()], None));
        assert!(e.contains("is listed twice"), "{e}");
        assert!(e.contains("(from VICTORY_TEST_TITLES)"), "{e}");
        let e = err(with(
            vec![a.clone(), x.join("b").join("..").join("a")],
            None,
        ));
        assert!(e.contains("is listed twice"), "{e}");
        assert!(e.contains(&format!("(also as {})", a.display())), "{e}");

        // The game folder pasted into the list.
        let e = err(with(vec![x.join("b"), a.join(".")], Some(a.clone())));
        assert!(e.contains("is the game folder"), "{e}");
        assert!(e.contains("(from VICTORY_GAME_DIR)"), "{e}");

        // Different folders are fine, nested ones included.
        let ok = with(vec![a.join("dlc"), x.join("b"), x.join("ab")], Some(a));
        assert!(Config::resolve(FilePaths::default(), None, ok).is_ok());

        // From the file, named as such; case counts only where it is folded.
        let file = FilePaths {
            test_titles: vec![x.join("Title"), x.join("title")],
            ..FilePaths::default()
        };
        let env = |k: &str| match k {
            "VICTORY_CACHE_DIR" => Some(abs("c")),
            "VICTORY_DATA_DIR" => Some(abs("d")),
            _ => None,
        };
        let got = Config::resolve(file, None, env);
        if FOLD_CASE {
            let e = got.unwrap_err().to_string();
            assert!(e.contains("is listed twice"), "{e}");
            assert!(e.contains("(from config file)"), "{e}");
        } else {
            assert!(got.is_ok());
        }
    }

    #[test]
    fn a_config_file_inside_the_dump_cannot_put_the_cache_there() {
        // `victory.toml` kept in the game folder with `cache_dir = "cache"`:
        // anchored at the file, the cache lands inside the dump.
        let dump = PathBuf::from(abs("gta"));
        let parsed = parse_file("[paths]\ngame_dir = '.'\ncache_dir = 'cache'\n").unwrap();
        let paths = parsed.anchored_at(&dump, None).unwrap();
        let env = |k: &str| (k == "VICTORY_DATA_DIR").then(|| abs("d"));
        let err = Config::resolve(paths, None, env).unwrap_err().to_string();
        assert!(err.contains("inside the game folder"), "{err}");
        assert!(err.contains("(from config file)"), "{err}");
    }

    #[test]
    fn lexical_folds_dot_and_dot_dot() {
        let l = |p: &str| lexical(Path::new(p));
        assert_eq!(l("a/./b/../c"), PathBuf::from("a/c"));
        assert_eq!(l("../a/.."), PathBuf::from(".."));
        assert_eq!(l("a/../../b"), PathBuf::from("../b"));
        #[cfg(unix)]
        {
            assert_eq!(l("/../a"), PathBuf::from("/a"));
            assert_eq!(l("/x/y/../../z"), PathBuf::from("/z"));
        }
    }

    #[test]
    fn lexical_keeps_leading_dot_dots_and_drops_trailing_slashes() {
        let l = |p: &str| lexical(Path::new(p));
        // Each `..` with nothing to remove is kept, in order.
        assert_eq!(l("../../a"), PathBuf::from("../../a"));
        assert_eq!(l("./../a/./"), PathBuf::from("../a"));
        // A trailing slash or a doubled separator is not a component.
        assert_eq!(l("a//b/"), PathBuf::from("a/b"));
        #[cfg(unix)]
        {
            assert_eq!(l("/.."), PathBuf::from("/"));
            assert_eq!(l("/a/../../.."), PathBuf::from("/"));
        }
    }

    #[test]
    fn folder_overlap_ignores_trailing_slashes_and_dot_dot_in_the_parent() {
        for fold in [false, true] {
            // Trailing slashes and doubled separators do not split a match.
            assert!(is_within(
                Path::new("/v/cache//data/"),
                Path::new("/v/cache/"),
                fold
            ));
            // `..` is folded in the parent as in the path.
            assert!(is_within(
                Path::new("/v/data"),
                Path::new("/v/cache/.."),
                fold
            ));
            assert!(!is_within(
                Path::new("/v/cache"),
                Path::new("/v/cache/../data"),
                fold
            ));
            // A parent longer than the path never holds it.
            assert!(!is_within(Path::new("/v"), Path::new("/v/cache"), fold));
        }
    }

    #[test]
    fn home_dir_prefers_home_and_skips_empty_values() {
        let both = |k: &str| match k {
            "HOME" => Some(abs("home")),
            "USERPROFILE" => Some(abs("profile")),
            _ => None,
        };
        assert_eq!(home_dir(both), Some(PathBuf::from(abs("home"))));
        // An empty HOME (`HOME= cmd`) is not absolute, so it is skipped.
        let empty = |k: &str| match k {
            "HOME" => Some(OsString::new()),
            "USERPROFILE" => Some(abs("profile")),
            _ => None,
        };
        assert_eq!(home_dir(empty), Some(PathBuf::from(abs("profile"))));
        let neither = |k: &str| matches!(k, "HOME" | "USERPROFILE").then(OsString::new);
        assert_eq!(home_dir(neither), None);
    }

    #[test]
    fn origins_name_the_setting_that_won() {
        let file = FilePaths {
            game_dir: Some(abs("file-game").into()),
            cache_dir: Some(abs("file-cache").into()),
            test_titles: vec![abs("file-title").into()],
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_GAME_DIR" => Some(abs("env-game")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(
            cfg.origins,
            Origins {
                game_dir: Origin::Env("VICTORY_GAME_DIR"),
                cache_dir: Origin::File,
                data_dir: Origin::Default,
                test_titles: Origin::File,
            }
        );

        // Empty variables are unset, so they are not the origin either.
        let env = |k: &str| match k {
            "VICTORY_CACHE_DIR" | "VICTORY_TEST_TITLES" => Some(OsString::new()),
            "VICTORY_DATA_DIR" => Some(abs("env-data")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(FilePaths::default(), None, env).unwrap();
        assert_eq!(
            cfg.origins,
            Origins {
                game_dir: Origin::Default,
                cache_dir: Origin::Default,
                data_dir: Origin::Env("VICTORY_DATA_DIR"),
                test_titles: Origin::Default,
            }
        );
        assert_eq!(
            Origin::Env("VICTORY_DATA_DIR").to_string(),
            "VICTORY_DATA_DIR"
        );
        assert_eq!(Origin::File.to_string(), "config file");
        assert_eq!(Origin::Default.to_string(), "default");
    }

    #[test]
    fn data_inside_the_cache_names_both_origins() {
        // The cache from the file holds the default data folder: the error
        // must say which setting is which.
        let file = FilePaths {
            cache_dir: Some(abs("v").into()),
            ..Default::default()
        };
        let env =
            |k: &str| (k == "XDG_DATA_HOME").then(|| PathBuf::from(abs("v")).join("share").into());
        let err = Config::resolve(file, None, env).unwrap_err().to_string();
        assert!(err.contains("(from default)"), "{err}");
        assert!(err.contains("(from config file)"), "{err}");
    }

    #[test]
    fn windows_drive_paths_are_refused_off_windows() {
        for p in ["D:\\victory-data\\game", "d:/games", "D:"] {
            assert!(is_drive_path(Path::new(p)), "{p}");
            let err = ensure_native("[paths].game_dir", Path::new(p), false)
                .unwrap_err()
                .to_string();
            assert!(err.contains("[paths].game_dir"), "{err}");
            assert!(err.contains("Windows path"), "{err}");
            assert!(ensure_native("[paths].game_dir", Path::new(p), true).is_ok());
        }
        for p in [
            "/home/me/games",
            "games",
            "~/games",
            "DD:/x",
            "1:/x",
            "D:x",
            "",
        ] {
            assert!(!is_drive_path(Path::new(p)), "{p}");
            assert!(ensure_native("x", Path::new(p), false).is_ok());
        }
        if cfg!(windows) {
            return;
        }
        // Through the file and the variables, naming the setting.
        let parsed: FileConfig =
            toml::from_str("[paths]\ntest_titles = ['ok', 'E:\\\\t']\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].test_titles"), "{err}");
        for var in [
            "VICTORY_GAME_DIR",
            "VICTORY_DATA_DIR",
            "VICTORY_TEST_TITLES",
        ] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("D:\\victory")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = Config::resolve(FilePaths::default(), None, env)
                .unwrap_err()
                .to_string();
            assert!(err.contains(var), "{err}");
        }
    }

    #[test]
    fn test_titles_joined_with_the_wrong_separator_are_refused() {
        for (p, on_windows) in [
            ("/t/one;/t/two", false),
            ("/t/one; /t/two", false),
            ("/t/one,~/t/two", false),
            ("t;/x", false),
            ("C:\\t\\one:D:\\t\\two", true),
            ("C:\\t\\one,D:/t/two", true),
        ] {
            let err = ensure_one_title(Path::new(p), on_windows)
                .unwrap_err()
                .to_string();
            assert!(err.contains("VICTORY_TEST_TITLES"), "{err}");
            assert!(err.contains("several titles"), "{err}");
            let right = if on_windows { "';'" } else { "':'" };
            assert!(err.contains(right), "{err}");
        }
        for (p, on_windows) in [
            ("/games/a,b", false),
            ("/games/a;b", false),
            ("/games/a, b", false),
            ("C:\\t\\one", true),
            ("C:\\games\\a,b", true),
            ("/t/one;/t/two", true),
            ("C:\\t\\one:D:\\t\\two", false),
            ("", false),
        ] {
            assert!(ensure_one_title(Path::new(p), on_windows).is_ok(), "{p}");
        }
        // Through the variable, on this host.
        let joined = if cfg!(windows) {
            "C:\\victory\\a:C:\\victory\\b"
        } else {
            "/victory/a;/victory/b"
        };
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(OsString::from(joined)),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("several titles"), "{err}");
        // The right separator still gives two titles.
        let sep = if cfg!(windows) { ";" } else { ":" };
        let titles = OsString::from(format!(
            "{}{sep}{}",
            abs("a").to_string_lossy(),
            abs("b").to_string_lossy()
        ));
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(titles.clone()),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(FilePaths::default(), None, env).unwrap();
        assert_eq!(cfg.test_titles.len(), 2);
    }

    #[test]
    fn joined_title_check_needs_a_title_on_both_sides() {
        // A separator at the very start, at the end, before only spaces or
        // before a relative name does not split off another absolute title.
        for (p, on_windows) in [
            (";/t/one", false),
            (",~/t/one", false),
            ("/t/one;", false),
            ("/t/one,  ", false),
            ("/t/one;two", false),
            ("/t/one,./two", false),
            (":C:\\t", true),
            ("C:\\t\\one:", true),
            ("C:\\t\\one,two", true),
        ] {
            assert!(ensure_one_title(Path::new(p), on_windows).is_ok(), "{p}");
        }
        // A bare drive after the separator is still another title.
        assert!(ensure_one_title(Path::new("C:\\t\\one,D:"), true).is_err());
        // Names that are not ASCII before the separator are walked by
        // character, not byte, and the error quotes the whole entry.
        let err = ensure_one_title(Path::new("/t/caf\u{e9};/t/\u{fc}ber"), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("/t/caf\u{e9};/t/\u{fc}ber"), "{err}");
        assert!(err.contains("';'"), "{err}");
        // Only the first wrong separator that starts a title is reported.
        let err = ensure_one_title(Path::new("/t/a,b;/t/c"), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("joined with ';'"), "{err}");
    }

    #[test]
    fn windows_network_paths_are_refused_off_windows() {
        for p in [
            "\\\\nas\\games\\gta",
            "\\\\nas",
            "\\\\?\\C:\\games",
            "\\\\.\\pipe\\x",
            "\\\\nas/games",
        ] {
            assert!(is_unc_path(Path::new(p)), "{p}");
            let err = ensure_native("VICTORY_GAME_DIR", Path::new(p), false)
                .unwrap_err()
                .to_string();
            assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
            assert!(err.contains("network path"), "{err}");
            assert!(err.contains("mount"), "{err}");
            assert!(ensure_native("VICTORY_GAME_DIR", Path::new(p), true).is_ok());
        }
        // A Unix `//host` path, a lone `\\` and a backslash later in the
        // name are not network paths.
        for p in [
            "//nas/games",
            "\\\\",
            "\\\\\\x",
            "\\\\/x",
            "games\\\\gta",
            "",
        ] {
            assert!(!is_unc_path(Path::new(p)), "{p}");
            assert!(ensure_native("x", Path::new(p), false).is_ok(), "{p}");
        }
        if cfg!(windows) {
            return;
        }
        // Through the file and the variables, naming the setting.
        let parsed: FileConfig = toml::from_str("[paths]\ngame_dir = '\\\\nas\\games'\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].game_dir"), "{err}");
        assert!(err.contains("network path"), "{err}");
        for var in ["VICTORY_CACHE_DIR", "VICTORY_TEST_TITLES", "VICTORY_CONFIG"] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("\\\\nas\\victory")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = if var == "VICTORY_CONFIG" {
                explicit_config(env, None).unwrap_err()
            } else {
                Config::resolve(FilePaths::default(), None, env).unwrap_err()
            }
            .to_string();
            assert!(err.contains(var), "{err}");
            assert!(err.contains("network path"), "{err}");
        }
    }

    #[test]
    fn windows_backslash_paths_are_refused_off_windows() {
        for (p, suggested) in [
            ("\\Games\\gta", "/Games/gta"),
            ("\\", "/"),
            (".\\gta", "./gta"),
            ("..\\games\\gta", "../games/gta"),
            ("~\\Games\\gta", "~/Games/gta"),
        ] {
            let err = ensure_native("VICTORY_GAME_DIR", Path::new(p), false)
                .unwrap_err()
                .to_string();
            assert!(err.contains("VICTORY_GAME_DIR"), "{err}");
            assert!(err.contains("backslashes"), "{err}");
            assert!(err.ends_with(&format!("perhaps {suggested}")), "{err}");
            assert!(ensure_native("VICTORY_GAME_DIR", Path::new(p), true).is_ok());
        }
        // A backslash later in a name, a Unix path, a dotted name and a
        // leading `~` or `.` without a backslash are left alone.
        for p in [
            "games\\gta",
            "/games/a\\b",
            "./gta",
            "~/gta",
            "..gta",
            ".x\\y",
            "~user",
            "",
        ] {
            assert!(!is_backslash_relative(Path::new(p)), "{p}");
            assert!(ensure_native("x", Path::new(p), false).is_ok(), "{p}");
        }
        if cfg!(windows) {
            return;
        }
        // Through the file and the variables, naming the setting, and ahead
        // of the `~user` error a `~\` would otherwise get.
        let parsed: FileConfig = toml::from_str("[paths]\ndata_dir = '~\\saves'\n").unwrap();
        let err = parsed
            .anchored_at(Path::new("/configs"), Some(Path::new("/home/me")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("[paths].data_dir"), "{err}");
        assert!(err.contains("backslashes"), "{err}");
        for var in ["VICTORY_CACHE_DIR", "VICTORY_TEST_TITLES", "VICTORY_CONFIG"] {
            let env = |k: &str| match k {
                k if k == var => Some(OsString::from("..\\victory")),
                "HOME" => Some(abs("home")),
                _ => None,
            };
            let err = if var == "VICTORY_CONFIG" {
                explicit_config(env, None).unwrap_err()
            } else {
                Config::resolve(FilePaths::default(), None, env).unwrap_err()
            }
            .to_string();
            assert!(err.contains(var), "{err}");
            assert!(err.contains("backslashes"), "{err}");
        }
    }

    #[test]
    fn config_text_drops_a_utf8_byte_order_mark() {
        let text = config_text(b"\xEF\xBB\xBF[paths]\ngame_dir = 'g'\n".to_vec()).unwrap();
        assert_eq!(text, "[paths]\ngame_dir = 'g'\n");
        let parsed = parse_file(&text).unwrap();
        assert_eq!(parsed.paths.game_dir, Some(PathBuf::from("g")));
        // Only a leading mark: one later in the file is the TOML's business.
        assert_eq!(config_text(b"a\xEF\xBB\xBF".to_vec()).unwrap(), "a\u{FEFF}");
        assert_eq!(config_text(Vec::new()).unwrap(), "");
        assert_eq!(config_text(b"\xEF\xBB\xBF".to_vec()).unwrap(), "");
    }

    #[test]
    fn config_text_refuses_utf16() {
        let encode = |le: bool, mark: bool| -> Vec<u8> {
            let units = mark
                .then_some(0xFEFF)
                .into_iter()
                .chain("[paths]\n".encode_utf16());
            units
                .flat_map(|u| if le { u.to_le_bytes() } else { u.to_be_bytes() })
                .collect()
        };
        for le in [true, false] {
            for mark in [true, false] {
                let err = format!("{:#}", config_text(encode(le, mark)).unwrap_err());
                assert!(err.contains("UTF-16"), "le={le} mark={mark}: {err}");
                assert!(err.contains("save it as UTF-8"), "{err}");
            }
        }
    }

    #[test]
    fn config_text_names_the_line_that_is_not_utf8() {
        // `é` in Latin-1, as an older editor would save a folder name.
        let err = format!(
            "{:#}",
            config_text(b"[paths]\ngame_dir = '/jeux/\xE9t\xE9'\n".to_vec()).unwrap_err()
        );
        assert!(err.contains("line 2"), "{err}");
        assert!(err.contains("save it as UTF-8"), "{err}");
        assert!(!err.contains("UTF-16"), "{err}");
    }

    #[test]
    fn test_titles_env_splits_paths() {
        // The empty entry between them (`first::second`) is skipped, not
        // read as the working directory.
        let value = std::env::join_paths(["first", "", "second"]).unwrap();
        let file = FilePaths::default();
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(value.clone()),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn config_text_counts_lines_after_the_mark_and_at_the_end() {
        // The byte offset and line are of the text after a dropped mark.
        let err = format!(
            "{:#}",
            config_text(b"\xEF\xBB\xBF[paths]\n\xFF".to_vec()).unwrap_err()
        );
        assert!(err.contains("line 2 "), "{err}");
        assert!(err.contains("byte 8,"), "{err}");
        // A lone stray byte is not UTF-16, just not UTF-8 on line 1.
        let err = format!("{:#}", config_text(b"\xFF".to_vec()).unwrap_err());
        assert!(err.contains("line 1 "), "{err}");
        assert!(!err.contains("UTF-16"), "{err}");
        // A character cut off at the end of the file is on the last line.
        let err = format!("{:#}", config_text(b"a\nb\n\xC3".to_vec()).unwrap_err());
        assert!(err.contains("line 3 "), "{err}");
        // A non-ASCII first byte before a zero is not taken for UTF-16.
        let err = format!("{:#}", config_text(b"\xE9\0".to_vec()).unwrap_err());
        assert!(!err.contains("UTF-16"), "{err}");
    }

    #[test]
    fn misplaced_path_keys_in_broken_toml_keep_the_parser_message() {
        // The file does not parse as TOML at all: the hint would be a guess,
        // so the parser's own message (naming the broken header) stays.
        let err = format!(
            "{:#}",
            parse_file("game_dir = '/games/gta'\n[paths\n").expect_err("refused")
        );
        assert!(!err.contains("must be under"), "{err}");
        // A top-level key that is not a path key is the parser's business.
        let err = format!("{:#}", parse_file("log = 'info'\n").expect_err("refused"));
        assert!(!err.contains("must be under"), "{err}");
        assert!(err.contains("log"), "{err}");
        // A path key nested in another table is not top level either.
        let err = format!(
            "{:#}",
            parse_file("[game]\ngame_dir = '/games/gta'\n").expect_err("refused")
        );
        assert!(!err.contains("must be under"), "{err}");
    }

    #[test]
    fn each_misspelling_is_checked_against_its_own_variable() {
        let names =
            |list: &[&str]| -> Vec<std::ffi::OsString> { list.iter().map(Into::into).collect() };
        // Another real variable being set does not excuse a near miss.
        let err = ensure_no_misspelled_variables(names(&["VICTORY_GAMEDIR"]), |k| {
            (k == "VICTORY_DATA_DIR").then(|| "/d".into())
        })
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("VICTORY_GAMEDIR is set, but Victory reads VICTORY_GAME_DIR"),
            "{err}"
        );
        // An excused near miss does not hide a later one.
        let err = ensure_no_misspelled_variables(
            names(&["VICTORY_DATA_PATH", "PATH", "VICTORY_CONFIG_FILE"]),
            |k| (k == "VICTORY_DATA_DIR").then(|| "/d".into()),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("rename VICTORY_CONFIG_FILE to VICTORY_CONFIG"),
            "{err}"
        );
        assert!(!err.contains("VICTORY_DATA_PATH"), "{err}");
        // With several unexcused, the first in the environment is named.
        let err =
            ensure_no_misspelled_variables(names(&["VICTORY_TITLES", "VICTORY_CACHE"]), |_| None)
                .unwrap_err()
                .to_string();
        assert!(err.starts_with("VICTORY_TITLES is set"), "{err}");
        // Every real variable set: nothing to report.
        ensure_no_misspelled_variables(
            names(&["VICTORY_GAMEDIR", "VICTORY_TITLES", "VICTORY_CONFIG_PATH"]),
            |k| SETTINGS.contains(&k).then(|| "/x".into()),
        )
        .unwrap();
        // Mixed case and hyphens are near misses too (not on Windows,
        // where the real names already match case-insensitively).
        if !cfg!(windows) {
            for (name, meant) in [
                ("Victory_Data_Dir", "VICTORY_DATA_DIR"),
                ("victory_config", "VICTORY_CONFIG"),
                ("victory-test-title", "VICTORY_TEST_TITLES"),
                ("VICTORY_Game_Folder", "VICTORY_GAME_DIR"),
            ] {
                assert_eq!(misspelling_of(name), Some(meant), "{name}");
            }
        }
    }

    #[test]
    fn misspellings_ignore_doubled_separators_and_other_suffixes() {
        for (name, meant) in [
            ("VICTORY__GAME__DIR", "VICTORY_GAME_DIR"),
            ("VICTORY_CACHE_PATH", "VICTORY_CACHE_DIR"),
            ("VICTORY_CONFIG_PATH", "VICTORY_CONFIG"),
            ("VICTORY_TESTTITLES", "VICTORY_TEST_TITLES"),
            ("-VICTORY_DATA_DIR-", "VICTORY_DATA_DIR"),
        ] {
            assert_eq!(misspelling_of(name), Some(meant), "{name}");
        }
        for name in [
            "",
            "_",
            "VICTORY",
            "VICTORY_",
            "VICTORY_TEST_TITLES_DIR",
            "VICTORY_CONFIG_FOLDER",
            "VICTORY_GAME_DIR_PATH",
            "VICTORY_CACHEDIR2",
        ] {
            assert_eq!(misspelling_of(name), None, "{name}");
        }
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed = parse_file(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
    }

    #[test]
    fn top_level_path_keys_say_where_they_belong() {
        let err = format!(
            "{:#}",
            parse_file("game_dir = '/games/gta'\n").expect_err("refused")
        );
        assert!(
            err.contains("game_dir must be under the [paths] table"),
            "{err}"
        );

        // Above the header, two keys, and a [paths] table that is fine.
        let text = "cache_dir = 'c'\ntest_titles = []\n[paths]\ndata_dir = 'd'\n";
        let err = format!("{:#}", parse_file(text).expect_err("refused"));
        assert!(
            err.contains("cache_dir, test_titles must be under"),
            "{err}"
        );
        assert!(err.contains("above them"), "{err}");

        // Other mistakes keep the parser's own message.
        let err = format!("{:#}", parse_file("[path]\n").expect_err("refused"));
        assert!(err.contains("path"), "{err}");
        assert!(!err.contains("must be under"), "{err}");
        let err = format!(
            "{:#}",
            parse_file("[paths]\ngamedir = 'x'\n").expect_err("refused")
        );
        assert!(err.contains("gamedir"), "{err}");

        // A correct file still parses.
        let parsed = parse_file("[paths]\ngame_dir = '/games/gta'\n").expect("parses");
        assert_eq!(parsed.paths.game_dir, Some(PathBuf::from("/games/gta")));
    }

    #[test]
    fn subfolders_must_stay_inside_their_root() {
        for ok in ["images", "0000ABCD/storage", "a/b/c"] {
            assert!(ensure_plain_subdir(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "/etc",
            "../escape",
            "a/../../b",
            "a/./b",
            ".",
            "..",
            "C:\\x",
            "C:/x",
            "a\\..\\b",
        ] {
            let err = ensure_plain_subdir(bad).unwrap_err().to_string();
            assert!(err.contains("not a plain relative path"), "{bad}: {err}");
        }
        // Refused before anything is created: the root need not exist.
        let cfg = Config {
            cache_dir: PathBuf::from("/nonexistent-victory-test-root"),
            data_dir: PathBuf::from("/nonexistent-victory-test-data"),
            ..Default::default()
        };
        assert!(cfg.cache_subdir("/tmp").is_err());
        assert!(cfg.data_subdir("../x").is_err());
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            data_dir: inside,
            ..Default::default()
        };
        assert!(cfg.data_subdir("x").is_err());
    }

    #[test]
    fn subfolders_are_created_nested_and_again_without_error() {
        // A scratch folder under the system temp folder, never the
        // repository. `TMPDIR` may point inside the checkout (a worktree's
        // own scratch folder); the asset policy then refuses every subfolder,
        // which is right, so check that instead of creating anything.
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-subdir-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let cfg = Config {
            cache_dir: scratch.join("cache"),
            data_dir: scratch.join("data"),
            ..Default::default()
        };
        if ensure_outside_repo(&scratch).is_err() {
            assert!(cfg.cache_subdir("images").is_err());
            assert!(!scratch.join("cache").exists());
            return;
        }
        // Nested parts are created in one go, and asking again is not an error.
        for _ in 0..2 {
            let dir = cfg.cache_subdir("0000ABCD/shaders").unwrap();
            assert_eq!(dir, scratch.join("cache").join("0000ABCD/shaders"));
            assert!(dir.is_dir());
            let dir = cfg.data_subdir("0000ABCD/storage").unwrap();
            assert_eq!(dir, scratch.join("data").join("0000ABCD/storage"));
            assert!(dir.is_dir());
        }
        // A file where a subfolder goes is an error naming that path, and is
        // left as it was.
        let blocker = scratch.join("cache").join("blocked");
        std::fs::write(&blocker, b"x").unwrap();
        let err = format!("{:#}", cfg.cache_subdir("blocked/inner").unwrap_err());
        assert!(err.contains("creating"), "{err}");
        assert!(err.contains("blocked"), "{err}");
        assert!(blocker.is_file());
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// A path that is absolute on the host (`/x` is not absolute on Windows).
    #[test]
    fn folder_overlap_folds_case_only_when_asked() {
        let cache = Path::new("/Victory/Cache");
        assert!(is_within(Path::new("/victory/cache/data"), cache, true));
        assert!(is_within(Path::new("/VICTORY/CACHE"), cache, true));
        assert!(!is_within(Path::new("/victory/cache/data"), cache, false));
        assert!(is_within(Path::new("/Victory/Cache/data"), cache, false));
        // Still whole components, and still with `..` folded first.
        assert!(!is_within(Path::new("/victory/cachex"), cache, true));
        assert!(!is_within(Path::new("/victory/cache/../data"), cache, true));
        assert!(!is_within(Path::new("/victory"), cache, true));
    }

    #[test]
    fn folder_overlap_folds_only_ascii_case_and_keeps_relative_apart() {
        for fold in [false, true] {
            // A relative path is never inside an absolute folder, or the
            // reverse, even when the names match.
            assert!(!is_within(
                Path::new("v/cache/data"),
                Path::new("/v/cache"),
                fold
            ));
            assert!(!is_within(
                Path::new("/v/cache/data"),
                Path::new("v/cache"),
                fold
            ));
            // Leading `..` are kept and compared like names.
            assert!(is_within(Path::new("../v/data"), Path::new("../v"), fold));
            assert!(!is_within(
                Path::new("../v/data"),
                Path::new("../../v"),
                fold
            ));
        }
        // Only ASCII letters fold: `É` and `é` stay different folders.
        assert!(!is_within(
            Path::new("/Jeux/É/data"),
            Path::new("/jeux/é"),
            true
        ));
        assert!(is_within(
            Path::new("/Jeux/É/data"),
            Path::new("/jeux/É"),
            true
        ));
    }

    /// A subfolder is checked and created through any link already inside
    /// the cache or data folder, as the OS will follow it: a link that leads
    /// into a repository (a folder holding the root marker) is refused
    /// before anything is created there, a link to another folder is
    /// followed, and a link that leads nowhere is an error naming the path,
    /// left as it was.
    #[cfg(unix)]
    #[test]
    fn subfolders_follow_links_inside_their_root() {
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-subdir-links-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        if ensure_outside_repo(&scratch).is_err() {
            // `TMPDIR` is inside the checkout; every subfolder is refused,
            // which `subfolders_are_created_nested_and_again_without_error`
            // already checks.
            return;
        }
        let _ = std::fs::remove_dir_all(&scratch);
        let cache = scratch.join("cache");
        let repo = scratch.join("repo");
        let elsewhere = scratch.join("elsewhere");
        for dir in [&cache, &repo, &elsewhere] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(repo.join(crate::policy::ROOT_MARKER), b"").unwrap();
        std::os::unix::fs::symlink(&repo, cache.join("into-repo")).unwrap();
        std::os::unix::fs::symlink(&elsewhere, cache.join("elsewhere")).unwrap();
        std::os::unix::fs::symlink(scratch.join("moved-away"), cache.join("dangling")).unwrap();
        let cfg = Config {
            cache_dir: cache.clone(),
            data_dir: scratch.join("data"),
            ..Default::default()
        };

        // Into a repository, directly or a few folders down: refused, and
        // nothing is made inside it.
        for sub in [
            "into-repo",
            "into-repo/shaders",
            "into-repo/0000ABCD/shaders",
        ] {
            assert!(cfg.cache_subdir(sub).is_err(), "{sub}");
        }
        assert_eq!(std::fs::read_dir(&repo).unwrap().count(), 1);

        // To an ordinary folder: followed, and the subfolder made there.
        let dir = cfg.cache_subdir("elsewhere/0000ABCD").unwrap();
        assert_eq!(dir, cache.join("elsewhere/0000ABCD"));
        assert!(elsewhere.join("0000ABCD").is_dir());

        // Leading nowhere: an error naming the path; the link is kept and
        // its target not created.
        let err = format!("{:#}", cfg.cache_subdir("dangling/inner").unwrap_err());
        assert!(err.contains("creating"), "{err}");
        assert!(err.contains("dangling"), "{err}");
        assert!(
            cache
                .join("dangling")
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(!scratch.join("moved-away").exists());

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_subfolder_under_a_looping_link_a_file_root_or_a_locked_parent_is_an_error() {
        use std::os::unix::fs::PermissionsExt;
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-subdir-odd-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        if ensure_outside_repo(&scratch).is_err() {
            // `TMPDIR` is inside the checkout; every subfolder is refused,
            // which `subfolders_are_created_nested_and_again_without_error`
            // already checks.
            return;
        }
        let _ = std::fs::remove_dir_all(&scratch);
        let cache = scratch.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let cfg = Config {
            cache_dir: cache.clone(),
            data_dir: scratch.join("data"),
            ..Default::default()
        };

        // A link inside the cache that loops: an error naming the path for
        // the link itself and for a folder under it, and the link is kept.
        let looped = cache.join("loop");
        std::os::unix::fs::symlink(&looped, &looped).unwrap();
        for sub in ["loop", "loop/inner", "loop/a/b"] {
            let err = format!("{:#}", cfg.cache_subdir(sub).unwrap_err());
            assert!(err.contains("creating"), "{sub}: {err}");
            assert!(err.contains(&looped.display().to_string()), "{sub}: {err}");
        }
        assert!(looped.symlink_metadata().unwrap().file_type().is_symlink());

        // A link to a folder a few levels inside a repository: refused by
        // the marker above its target, and nothing is made there.
        let deep = scratch.join("repo").join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(scratch.join("repo").join(crate::policy::ROOT_MARKER), b"").unwrap();
        std::os::unix::fs::symlink(&deep, cache.join("deep")).unwrap();
        for sub in ["deep", "deep/shaders"] {
            assert!(cfg.cache_subdir(sub).is_err(), "{sub}");
        }
        assert_eq!(std::fs::read_dir(&deep).unwrap().count(), 0);

        // The data folder itself a file (swapped in after the settings were
        // checked): an error naming the subfolder, and the file kept as is.
        std::fs::write(&cfg.data_dir, b"saves").unwrap();
        let err = format!("{:#}", cfg.data_subdir("0000ABCD").unwrap_err());
        assert!(err.contains("creating"), "{err}");
        assert!(
            err.contains(&cfg.data_dir.join("0000ABCD").display().to_string()),
            "{err}"
        );
        assert_eq!(std::fs::read(&cfg.data_dir).unwrap(), b"saves");
        std::fs::remove_file(&cfg.data_dir).unwrap();

        // A cache folder not made yet, under a folder without write
        // permission: the permission error with the subfolder it was
        // creating, and nothing made.
        let locked = scratch.join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
        let shut = Config {
            cache_dir: locked.join("cache"),
            data_dir: scratch.join("data"),
            ..Default::default()
        };
        // Root (or a CAP_DAC_OVERRIDE container) writes through the mode
        // bits, leaving nothing to check.
        let enforced = std::fs::create_dir(locked.join("probe")).is_err();
        if enforced {
            let err = shut.cache_subdir("images").unwrap_err();
            let io = err.downcast_ref::<std::io::Error>().unwrap();
            assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied, "{err:#}");
            assert_eq!(
                err.to_string(),
                format!("creating {}", locked.join("cache/images").display())
            );
            assert!(!locked.join("cache").exists());
        }
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        // Write permission back: the same call succeeds.
        assert!(shut.cache_subdir("images").unwrap().is_dir());

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// The subfolder itself, not a folder above it, already taken: a file,
    /// a link to a file or a link that leads nowhere is an error naming the
    /// path, with or without a trailing slash, and is left as it was; a link
    /// to a folder is used as the subfolder, with nothing made in its place.
    #[cfg(unix)]
    #[test]
    fn a_subfolder_that_is_itself_a_file_or_a_link_is_checked_where_it_leads() {
        let scratch = std::env::temp_dir().join(format!(
            "victory-config-subdir-last-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        if ensure_outside_repo(&scratch).is_err() {
            // `TMPDIR` is inside the checkout; every subfolder is refused,
            // which `subfolders_are_created_nested_and_again_without_error`
            // already checks.
            return;
        }
        let _ = std::fs::remove_dir_all(&scratch);
        let data = scratch.join("data");
        let elsewhere = scratch.join("elsewhere");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        let note = scratch.join("note.txt");
        std::fs::write(&note, b"keep").unwrap();
        std::fs::write(data.join("saves"), b"slot1").unwrap();
        std::os::unix::fs::symlink(&note, data.join("to-file")).unwrap();
        std::os::unix::fs::symlink(scratch.join("moved-away"), data.join("dangling")).unwrap();
        std::os::unix::fs::symlink(&elsewhere, data.join("linked")).unwrap();
        let cfg = Config {
            cache_dir: scratch.join("cache"),
            data_dir: data.clone(),
            ..Default::default()
        };

        for sub in [
            "saves",
            "saves/",
            "to-file",
            "to-file/",
            "dangling",
            "dangling/",
        ] {
            let err = format!("{:#}", cfg.data_subdir(sub).unwrap_err());
            assert!(err.contains("creating"), "{sub}: {err}");
            let name = sub.trim_end_matches('/');
            assert!(
                err.contains(&data.join(name).display().to_string()),
                "{sub}: {err}"
            );
        }
        assert_eq!(std::fs::read(data.join("saves")).unwrap(), b"slot1");
        assert_eq!(std::fs::read(&note).unwrap(), b"keep");
        for link in ["to-file", "dangling"] {
            let kind = data.join(link).symlink_metadata().unwrap().file_type();
            assert!(kind.is_symlink(), "{link}");
        }
        assert!(!scratch.join("moved-away").exists());

        // A link to a folder: the subfolder path is the link, it is not
        // replaced, and asking again is not an error.
        for _ in 0..2 {
            let dir = cfg.data_subdir("linked").unwrap();
            assert_eq!(dir, data.join("linked"));
            assert!(dir.is_dir());
            let kind = dir.symlink_metadata().unwrap().file_type();
            assert!(kind.is_symlink());
        }
        assert_eq!(std::fs::read_dir(&elsewhere).unwrap().count(), 0);

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    fn subfolders_refuse_a_leading_dot_and_allow_a_trailing_slash() {
        // A leading `./` is a `.` component; a trailing slash or a doubled
        // separator adds none, and a `~` is a plain name, not the home.
        for bad in ["./a", "a/.", "./"] {
            let err = ensure_plain_subdir(bad).unwrap_err().to_string();
            assert!(err.contains("not a plain relative path"), "{bad}: {err}");
        }
        for ok in ["a/", "a//b", "~", "~/x", "..a", "a.."] {
            assert!(ensure_plain_subdir(ok).is_ok(), "{ok}");
        }
    }

    #[test]
    fn shared_xdg_base_names_the_xdg_variables() {
        let base = abs("shared");
        let env = |k: &str| match k {
            "XDG_CACHE_HOME" | "XDG_DATA_HOME" => Some(base.clone()),
            _ => None,
        };
        let err = Config::resolve(FilePaths::default(), None, env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("XDG_DATA_HOME is inside"), "{err}");
        assert!(err.contains("VICTORY_DATA_DIR"), "{err}");
        // Overriding either folder is the way out.
        let data = abs("elsewhere");
        let env = |k: &str| match k {
            "XDG_CACHE_HOME" | "XDG_DATA_HOME" => Some(base.clone()),
            "VICTORY_DATA_DIR" => Some(data.clone()),
            _ => None,
        };
        let cfg = Config::resolve(FilePaths::default(), None, env).unwrap();
        assert_eq!(cfg.data_dir, PathBuf::from(&data));
    }

    #[test]
    fn uri_and_variable_checks_handle_odd_spellings() {
        // Hex escapes and the `localhost` host are matched in any case, and
        // an encoded `/` is decoded like any other byte.
        assert_eq!(uri_path("/a/%c3%a9").as_deref(), Some("/a/é"));
        assert_eq!(uri_path("//LocalHost/a").as_deref(), Some("/a"));
        assert_eq!(uri_path("///a%2Fb").as_deref(), Some("/a/b"));
        assert_eq!(uri_path("///c:/games").as_deref(), Some("c:/games"));
        // A drive without a separator after it is not a drive path.
        assert_eq!(uri_path("///C:games").as_deref(), Some("/C:games"));
        // A name must start with a letter or `_`: `$1`, `${1}` and `%1%`
        // are not variables, but `$_X` and `${_}` are.
        for ok in ["/a/$1", "/a/${1}", "/a/%1%", "/a/$-x", "/a/%a b%"] {
            assert!(ensure_no_variable("x", Path::new(ok)).is_ok(), "{ok}");
        }
        for (bad, var) in [("/a/$_X", "$_X"), ("/a/${_}x", "${_}")] {
            let err = ensure_no_variable("x", Path::new(bad))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("the variable {var};")),
                "{bad}: {err}"
            );
        }
    }

    /// A variable name that is not Unicode cannot be one of ours and is
    /// skipped, not refused or mistaken for a near miss.
    #[cfg(unix)]
    #[test]
    fn misspelled_variables_skip_names_that_are_not_unicode() {
        use std::os::unix::ffi::OsStringExt;
        let odd = OsString::from_vec(b"VICTORY_GAME\xFFDIR".to_vec());
        ensure_no_misspelled_variables([odd.clone()], |_| None).unwrap();
        // It does not hide a real near miss listed after it.
        let err = ensure_no_misspelled_variables([odd, "VICTORY_GAMEDIR".into()], |_| None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VICTORY_GAMEDIR is set"), "{err}");
    }

    /// Base folders set empty or relative are skipped, as [`default_dir`]
    /// skips them, and each kept one is named by its variable or `~/` path.
    #[test]
    fn user_bases_skip_empty_and_relative_values() {
        let home = PathBuf::from(abs("home"));
        let roaming = PathBuf::from(abs("roaming"));
        let env = |k: &str| match k {
            "HOME" => Some(home.clone().into()),
            "XDG_CACHE_HOME" => Some(OsString::new()),
            "XDG_DATA_HOME" => Some("share".into()),
            "XDG_CONFIG_HOME" => Some("./cfg".into()),
            "APPDATA" => Some(roaming.clone().into()),
            _ => None,
        };
        let bases = user_bases(env);
        let names: Vec<&str> = bases.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "APPDATA",
                "~/.cache",
                "~/.local/share",
                "~/.local/state",
                "~/.config",
                "~/.local"
            ]
        );
        assert_eq!(bases[0].0, roaming);
        assert_eq!(bases[1].0, home.join(".cache"));
        // No absolute home: only the variables that are absolute themselves.
        let env = |k: &str| match k {
            "HOME" => Some("me".into()),
            "XDG_STATE_HOME" => Some(abs("st")),
            _ => None,
        };
        let bases = user_bases(env);
        assert_eq!(bases, [(PathBuf::from(abs("st")), "XDG_STATE_HOME".into())]);
        assert!(user_bases(|_| None).is_empty());
    }

    /// `%APPDATA%` is shared like `%LOCALAPPDATA%`, on every host, and a
    /// relative `XDG_CONFIG_HOME` does not hide `~/.config`.
    #[test]
    fn cache_or_data_at_appdata_or_a_relative_xdg_fallback_is_refused() {
        let home = PathBuf::from(abs("home"));
        let roaming = PathBuf::from(abs("roaming"));
        let run = |key: &'static str, value: PathBuf| {
            let (home, roaming) = (home.clone(), roaming.clone());
            let env = move |k: &str| match k {
                "HOME" => Some(home.clone().into()),
                "APPDATA" => Some(roaming.clone().into()),
                "XDG_CONFIG_HOME" => Some("cfg".into()),
                k if k == key => Some(value.clone().into()),
                _ => None,
            };
            Config::resolve(FilePaths::default(), None, env)
        };
        let err = run("VICTORY_DATA_DIR", roaming.clone())
            .unwrap_err()
            .to_string();
        assert!(err.contains("is your shared APPDATA folder"), "{err}");
        assert!(
            err.contains(&roaming.join("victory").display().to_string()),
            "{err}"
        );
        let err = run("VICTORY_CACHE_DIR", home.join(".config"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("is your shared ~/.config folder"), "{err}");
        // Victory's own folder inside either is fine.
        run("VICTORY_DATA_DIR", roaming.join("victory")).unwrap();
        run("VICTORY_CACHE_DIR", home.join(".config").join("victory")).unwrap();
    }

    /// A relative `XDG_CACHE_HOME` is skipped like an unset one, so data
    /// under `~/.cache` is still refused and named that way; with no
    /// absolute home and no absolute `XDG_CACHE_HOME` there is no user cache
    /// to compare with, so nothing is refused.
    #[test]
    fn user_cache_check_skips_a_relative_xdg_cache_home() {
        let home = PathBuf::from(abs("home"));
        let data = home.join(".cache").join("saves");
        let env = |k: &str| match k {
            "HOME" => Some(home.clone().into()),
            "XDG_CACHE_HOME" => Some("rel-cache".into()),
            _ => None,
        };
        let err = ensure_data_outside_user_cache(&data, Origin::Env("VICTORY_DATA_DIR"), env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("(~/.cache)"), "{err}");
        assert!(err.contains("(from VICTORY_DATA_DIR)"), "{err}");
        let env = |k: &str| match k {
            "HOME" => Some("home".into()),
            "XDG_CACHE_HOME" => Some("rel-cache".into()),
            _ => None,
        };
        ensure_data_outside_user_cache(&data, Origin::Env("VICTORY_DATA_DIR"), env).unwrap();
        ensure_data_outside_user_cache(&data, Origin::Default, |_| None).unwrap();
    }

    fn abs(p: &str) -> OsString {
        if cfg!(windows) {
            format!("C:\\{p}").into()
        } else {
            format!("/{p}").into()
        }
    }
}
