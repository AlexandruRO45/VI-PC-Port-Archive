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
            .map_or_else(|| default_dir(Kind::Cache, &env), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.data_dir)
            .map_or_else(|| default_dir(Kind::Data, &env), Ok)?;
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
fn default_dir(kind: Kind, env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<PathBuf> {
    let var = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(local) = var("LOCALAPPDATA") {
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
    // The XDG spec ignores a relative value.
    let base = var(xdg)
        .filter(|p| p.is_absolute())
        .or_else(|| var("HOME").map(|h| h.join(fallback)))
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
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(|| OsString::from("from-env"));
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
            "HOME" => Some(OsString::from("/h")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    fn defaults(vars: &[(&str, &str)]) -> Config {
        let vars: Vec<(String, OsString)> = vars
            .iter()
            .map(|&(k, v)| (k.to_owned(), OsString::from(v)))
            .collect();
        let env = |k: &str| vars.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        Config::resolve(FilePaths::default(), None, env).unwrap()
    }

    #[test]
    fn local_app_data_holds_cache_and_data() {
        let cfg = defaults(&[("LOCALAPPDATA", "/l"), ("XDG_CACHE_HOME", "/x")]);
        assert_eq!(cfg.cache_dir, Path::new("/l/victory/cache"));
        assert_eq!(cfg.data_dir, Path::new("/l/victory/data"));
    }

    #[test]
    fn data_is_not_under_the_xdg_cache() {
        let cfg = defaults(&[("XDG_CACHE_HOME", "/c"), ("XDG_DATA_HOME", "/d")]);
        assert_eq!(cfg.cache_dir, Path::new("/c/victory"));
        assert_eq!(cfg.data_dir, Path::new("/d/victory"));

        let cfg = defaults(&[("HOME", "/h")]);
        assert_eq!(cfg.cache_dir, Path::new("/h/.cache/victory"));
        assert_eq!(cfg.data_dir, Path::new("/h/.local/share/victory"));
        assert!(!cfg.data_dir.starts_with(&cfg.cache_dir));
    }

    #[test]
    fn empty_or_relative_xdg_values_are_ignored() {
        let cfg = defaults(&[
            ("LOCALAPPDATA", ""),
            ("XDG_CACHE_HOME", ""),
            ("XDG_DATA_HOME", "relative"),
            ("HOME", "/h"),
        ]);
        assert_eq!(cfg.cache_dir, Path::new("/h/.cache/victory"));
        assert_eq!(cfg.data_dir, Path::new("/h/.local/share/victory"));
    }

    #[test]
    fn no_user_directory_is_an_error() {
        let env = |_: &str| None;
        assert!(Config::resolve(FilePaths::default(), None, env).is_err());
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
}
