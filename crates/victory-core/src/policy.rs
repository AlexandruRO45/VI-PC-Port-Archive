//! Rule #1, enforced at runtime: game-derived data never lands in the repo.
//!
//! Everything VIctory produces from the user's files (converted assets,
//! prepared images, shader caches, crash dumps) is as much "from the game"
//! as the files themselves. Every tool that writes such output must call
//! [`ensure_outside_repo`] on the destination first. The static half of the
//! rule is `cargo xtask policy`, which checks what git tracks.

use std::path::{Component, Path, PathBuf};

/// Marker file at the root of the source tree.
pub const ROOT_MARKER: &str = ".victory-root";

/// The source tree this binary was compiled from.
fn compiled_workspace_root() -> PathBuf {
    // crates/victory-core -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error(
        "refusing to use {path} for game-derived data: it is inside the VIctory source tree \
         ({root}). Keep game files, caches and outputs outside the repository \
         (see docs/asset-policy.md)"
    )]
    InsideRepo { path: PathBuf, root: PathBuf },
    #[error("cannot resolve {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Returns `Ok` only if `path` is outside every VIctory source tree.
///
/// `path` does not have to exist yet. Two checks are made:
/// 1. it is not below the source tree this binary was built from;
/// 2. none of its ancestors contains the [`ROOT_MARKER`] file (catches
///    other clones/copies of the repository).
pub fn ensure_outside_repo(path: &Path) -> Result<(), PolicyError> {
    let resolved = resolve(path)?;

    if let Ok(root) = std::fs::canonicalize(compiled_workspace_root())
        && root.join(ROOT_MARKER).is_file()
        && resolved.starts_with(&root)
    {
        return Err(PolicyError::InsideRepo {
            path: path.to_path_buf(),
            root,
        });
    }

    for ancestor in resolved.ancestors() {
        if ancestor.join(ROOT_MARKER).is_file() {
            return Err(PolicyError::InsideRepo {
                path: path.to_path_buf(),
                root: ancestor.to_path_buf(),
            });
        }
    }
    Ok(())
}

/// Resolves `path` component by component, canonicalizing each existing prefix
/// before applying `..` so symlinks exposed by or followed by `..` are
/// resolved against their real target rather than collapsed lexically.
fn resolve(path: &Path) -> Result<PathBuf, PolicyError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| PolicyError::Io {
                path: path.to_path_buf(),
                source,
            })?
            .join(path)
    };

    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if let Ok(canon) = std::fs::canonicalize(&out) {
                    out = canon;
                }
                out.pop();
            }
            Component::Normal(part) => {
                out.push(part);
                if let Ok(canon) = std::fs::canonicalize(&out) {
                    out = canon;
                }
            }
        }
    }

    let mut existing = out.as_path();
    let mut rest = Vec::new();
    loop {
        match std::fs::canonicalize(existing) {
            Ok(mut canon) => {
                for component in rest.iter().rev() {
                    canon.push(component);
                }
                return Ok(canon);
            }
            Err(source) => match (existing.parent(), existing.file_name()) {
                (Some(parent), Some(name)) => {
                    rest.push(name.to_os_string());
                    existing = parent;
                }
                _ => {
                    return Err(PolicyError::Io {
                        path: path.to_path_buf(),
                        source,
                    });
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_inside_this_repo() {
        let inside = compiled_workspace_root().join("cache").join("textures");
        assert!(matches!(
            ensure_outside_repo(&inside),
            Err(PolicyError::InsideRepo { .. })
        ));
        let sneaky = std::env::temp_dir().join("..").join("x"); // sanity: no panic
        let _ = ensure_outside_repo(&sneaky);
    }

    #[test]
    fn rejects_dotdot_escapes_back_into_repo() {
        let root = compiled_workspace_root();
        let path = root
            .join("crates")
            .join("..")
            .join("not-yet-created")
            .join("out");
        assert!(ensure_outside_repo(&path).is_err());
    }

    #[test]
    fn rejects_other_clones_via_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(ROOT_MARKER), "").unwrap();
        assert!(ensure_outside_repo(&tmp.path().join("cache")).is_err());
    }

    #[test]
    fn accepts_unrelated_directories() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_outside_repo(&tmp.path().join("does").join("not").join("exist")).unwrap();
    }

    #[test]
    fn resolves_dot_dot_in_a_path_that_does_not_exist_yet() {
        let base = std::env::temp_dir().join("victory-policy-no-such");
        let path = base.join("a").join("..").join("b");
        let resolved = resolve(&path).expect("a `..` in the missing tail must resolve lexically");
        assert_eq!(resolved, resolve(&base.join("b")).unwrap());
    }

    #[test]
    fn rejects_symlink_into_repo_across_dotdot_components() {
        let tmp = tempfile::tempdir().unwrap();
        let fake_repo = tmp.path().join("repo");
        let repo_sub = fake_repo.join("sub");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&repo_sub).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(fake_repo.join(ROOT_MARKER), "").unwrap();

        let link_root = tmp.path().join("link-root");
        let link_sub = tmp.path().join("link-sub");
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&fake_repo, &link_root)
            .and_then(|()| std::os::unix::fs::symlink(&repo_sub, &link_sub));
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_dir(&fake_repo, &link_root)
            .and_then(|()| std::os::windows::fs::symlink_dir(&repo_sub, &link_sub));
        if linked.is_err() {
            return; // environment lacks symlink creation privilege
        }

        // Case 1: `<outside>/missing/../../<link-root>/out` enters `<repo>/out`.
        let into_root = outside
            .join("missing")
            .join("..")
            .join("..")
            .join("link-root")
            .join("out");
        assert!(matches!(
            ensure_outside_repo(&into_root),
            Err(PolicyError::InsideRepo { .. })
        ));

        // Case 2: `<outside>/missing/../../<link-sub>/../out` steps out of
        // `<repo>/sub` into `<repo>/out` rather than lexically cancelling `link-sub`.
        let via_sub_parent = outside
            .join("missing")
            .join("..")
            .join("..")
            .join("link-sub")
            .join("..")
            .join("out");
        assert!(matches!(
            ensure_outside_repo(&via_sub_parent),
            Err(PolicyError::InsideRepo { .. })
        ));
    }
}
