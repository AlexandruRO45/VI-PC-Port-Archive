//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::path::{Path, PathBuf};

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

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = std::env::var_os("VICTORY_CONFIG").map(PathBuf::from);
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed.paths, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FilePaths,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_path = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            None => file.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(file.cache_dir)
            .map_or_else(|| default_dir("cache", &env), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.data_dir)
            .map_or_else(|| default_dir("data", &env), Ok)?;
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(file.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            source,
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

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `%LOCALAPPDATA%/victory/<name>` (elsewhere the XDG or home cache).
///
/// A variable that is set but empty or relative is skipped, as the XDG base
/// directory spec asks: otherwise `XDG_CACHE_HOME=` (or a relative value)
/// would put the cache under the working directory, which is often the
/// repository itself and so refused by the asset policy.
fn default_dir(name: &str, env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<PathBuf> {
    let absolute = |k: &str| env(k).map(PathBuf::from).filter(|p| p.is_absolute());
    let base = absolute("LOCALAPPDATA")
        .or_else(|| absolute("XDG_CACHE_HOME"))
        .or_else(|| absolute("HOME").map(|h| h.join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join(name))
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

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
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

    /// A path that is absolute on the host (`/x` is not absolute on Windows).
    fn abs(p: &str) -> OsString {
        if cfg!(windows) {
            format!("C:\\{p}").into()
        } else {
            format!("/{p}").into()
        }
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
            default_dir("cache", all).unwrap(),
            PathBuf::from(abs("lad")).join("victory").join("cache")
        );
        let no_lad = |k: &str| if k == "LOCALAPPDATA" { None } else { all(k) };
        assert_eq!(
            default_dir("data", no_lad).unwrap(),
            PathBuf::from(abs("xdg")).join("victory").join("data")
        );
        let home_only = |k: &str| (k == "HOME").then(|| abs("home"));
        assert_eq!(
            default_dir("cache", home_only).unwrap(),
            PathBuf::from(abs("home"))
                .join(".cache")
                .join("victory")
                .join("cache")
        );
    }

    #[test]
    fn default_dir_skips_empty_and_relative_values() {
        let env = |k: &str| match k {
            "LOCALAPPDATA" => Some(OsString::new()),
            "XDG_CACHE_HOME" => Some(OsString::from("relative/cache")),
            "HOME" => Some(abs("home")),
            _ => None,
        };
        assert_eq!(
            default_dir("cache", env).unwrap(),
            PathBuf::from(abs("home"))
                .join(".cache")
                .join("victory")
                .join("cache")
        );
        let nothing_usable = |k: &str| (k == "HOME").then(OsString::new);
        assert!(default_dir("cache", nothing_usable).is_err());
    }

    #[test]
    fn resolve_uses_the_given_environment_for_defaults() {
        let env = |k: &str| (k == "HOME").then(|| abs("home"));
        let cfg = Config::resolve(FilePaths::default(), None, env).unwrap();
        let base = PathBuf::from(abs("home")).join(".cache").join("victory");
        assert_eq!(cfg.cache_dir, base.join("cache"));
        assert_eq!(cfg.data_dir, base.join("data"));
    }
}
