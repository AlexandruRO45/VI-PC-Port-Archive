//! Rule #1, statically: the repository only ever contains small text files
//! that we wrote. See docs/asset-policy.md.
//!
//! This is an allowlist, not a blocklist: a file passes only if its name is
//! on the list, it is not too big, it is text, and it does not smuggle data
//! inside text (long hex/base64 runs).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

const MAX_FILE_SIZE: usize = 512 * 1024;
const MAX_ENCODED_RUN: usize = 512;

const ALLOWED_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "txt", "lock", "yml", "yaml", "json", "hlsl", "hlsli", "ps1", "cmd", "sh",
];
const ALLOWED_NAMES: &[&str] = &[
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".victory-root",
    "pre-commit",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE",
];

/// Paths explicitly exempted, with a reason. Keep this empty if at all possible.
const EXCEPTIONS: &[(&str, &str)] = &[(
    "crates/victory-formats/src/bcpack-tables.bin",
    "the BCPack format's constant tables: nothing decodes without them and they cannot be derived (owner's decision, 2026-11-23; docs/asset-policy.md)",
)];

/// Signatures of formats that must never be committed.
const FORBIDDEN_MAGIC: &[(&[u8], &str)] = &[
    (b"MZ", "PE executable"),
    (b"DXBC", "shader container"),
    (b"DDS ", "DDS texture"),
    (b"RIFF", "RIFF (audio/video)"),
    (b"BC\xC0\xDE", "LLVM bitcode"),
    (b"Microsoft C/C++ MSF", "PDB"),
    (b"!<arch>", "library archive"),
    (b"PK\x03\x04", "zip"),
    (b"7z\xBC\xAF", "7-zip"),
    (b"MSCF", "cabinet"),
    (b"\xD0\xCF\x11\xE0", "MSI/OLE"),
];

pub fn run(root: &Path, staged: bool) -> Result<()> {
    let files = list_files(root, staged)?;
    let mut violations = Vec::new();
    for file in &files {
        let path = &file.name;
        if EXCEPTIONS.iter().any(|(p, _)| p == path) {
            continue;
        }
        if !file.utf8_name {
            // Lossily decoded, so it could not be matched or read by name.
            violations.push(format!("{path}: file name is not valid UTF-8"));
        }
        let content = match read(root, file) {
            Ok(Some(content)) => content,
            Ok(None) => continue,
            Err(problem) => {
                violations.push(format!("{path}: {problem}"));
                continue;
            }
        };
        for problem in check(path, &content) {
            violations.push(format!("{path}: {problem}"));
        }
    }
    if violations.is_empty() {
        println!("asset policy: {} files ok", files.len());
        return Ok(());
    }
    for v in &violations {
        eprintln!("POLICY: {v}");
    }
    bail!(
        "{} asset-policy violation(s). Nothing from the game may be committed; \
         see docs/asset-policy.md",
        violations.len()
    )
}

/// One path to check.
struct File {
    /// Repository-relative, `/`-separated, lossily decoded if need be.
    name: String,
    utf8_name: bool,
    source: Source,
}

enum Source {
    /// On disk, at this path.
    Worktree(PathBuf),
    /// Staged: the index entry's mode and object id.
    Index { mode: String, oid: String },
}

/// The content to check; `Ok(None)` if there is nothing to check (a tracked
/// file deleted from the working tree), `Err` with the problem if a path the
/// check cannot read must not pass unread.
fn read(root: &Path, file: &File) -> Result<Option<Vec<u8>>, String> {
    match &file.source {
        Source::Index { mode, .. } if mode == "160000" => {
            Err("a submodule: content from another repository bypasses this check".to_owned())
        }
        Source::Index { oid, .. } => {
            // By object id, not `:<path>`, so no file name can make it fail.
            let out = Command::new("git")
                .args(["cat-file", "blob", oid])
                .current_dir(root)
                .output()
                .map_err(|e| format!("cannot run git: {e}"))?;
            if out.status.success() {
                Ok(Some(out.stdout))
            } else {
                Err("cannot read the staged content".to_owned())
            }
        }
        Source::Worktree(path) => match std::fs::read(path) {
            Ok(content) => Ok(Some(content)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) if path.is_dir() => {
                Err("a directory tracked as a file (a submodule?)".to_owned())
            }
            Err(e) => Err(format!("cannot read: {e}")),
        },
    }
}

/// Splits NUL-terminated git output; each field is returned raw.
fn git_z(root: &Path, args: &[&str]) -> Result<Option<Vec<Vec<u8>>>> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .context("running git")?;
    if !out.status.success() {
        return Ok(None);
    }
    Ok(Some(
        out.stdout
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(<[u8]>::to_vec)
            .collect(),
    ))
}

/// Decodes a path git printed, keeping the exact bytes for reading it.
fn decode(root: &Path, raw: &[u8]) -> (String, bool, PathBuf) {
    let name = String::from_utf8_lossy(raw).replace('\\', "/");
    let utf8 = std::str::from_utf8(raw).is_ok();
    #[cfg(unix)]
    let on_disk = root.join(<std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(raw));
    // Git for Windows prints paths as UTF-8, and Windows names are Unicode.
    #[cfg(not(unix))]
    let on_disk = root.join(&name);
    (name, utf8, on_disk)
}

fn list_files(root: &Path, staged: bool) -> Result<Vec<File>> {
    if staged {
        return staged_files(root);
    }
    if let Some(fields) = git_z(
        root,
        &[
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
    )? {
        // A submodule is a directory on disk, or nothing if not checked out;
        // only the index says what it is.
        let submodules: Vec<Vec<u8>> = git_z(root, &["ls-files", "-z", "--stage"])?
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entry| {
                let tab = entry.iter().position(|&b| b == b'\t')?;
                entry
                    .starts_with(b"160000 ")
                    .then(|| entry[tab + 1..].to_vec())
            })
            .collect();
        let mut files: Vec<File> = fields
            .iter()
            .map(|raw| {
                let (name, utf8_name, on_disk) = decode(root, raw);
                let source = if submodules.contains(raw) {
                    Source::Index {
                        mode: "160000".to_owned(),
                        oid: String::new(),
                    }
                } else {
                    Source::Worktree(on_disk)
                };
                File {
                    name,
                    utf8_name,
                    source,
                }
            })
            .collect();
        files.sort_by(|a, b| a.name.cmp(&b.name));
        files.dedup_by(|a, b| a.name == b.name);
        return Ok(files);
    }
    // Not a git repository: walk everything except build output.
    Ok(walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !matches!(e.file_name().to_str(), Some("target" | ".git")))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let relative = e.path().strip_prefix(root).ok()?;
            Some(File {
                name: relative.to_string_lossy().replace('\\', "/"),
                utf8_name: relative.to_str().is_some(),
                source: Source::Worktree(e.path().to_owned()),
            })
        })
        .collect())
}

/// What the index holds that the next commit adds or changes.
fn staged_files(root: &Path) -> Result<Vec<File>> {
    // --raw gives each entry's new mode and object id, so content is read by
    // id and submodules are seen as such. T: a path whose type changed (a
    // symlink replaced by a regular file, or the reverse) carries new content
    // just as a modification does. --no-renames: a rename is checked as the
    // addition it is, with one path per record.
    let fields = git_z(
        root,
        &[
            "diff",
            "--cached",
            "--raw",
            "-z",
            "--no-abbrev",
            "--no-renames",
            "--diff-filter=ACMT",
        ],
    )?
    .context("--staged needs a git repository")?;
    let mut files = Vec::new();
    let mut fields = fields.into_iter();
    while let Some(header) = fields.next() {
        // :<old mode> <new mode> <old id> <new id> <status>, then the path.
        let header = String::from_utf8_lossy(&header);
        let parts: Vec<&str> = header.trim_start_matches(':').split(' ').collect();
        let [_, mode, _, oid, _] = parts[..] else {
            bail!("unexpected `git diff --raw` record: {header:?}");
        };
        let raw = fields
            .next()
            .with_context(|| format!("`git diff --raw` record without a path: {header:?}"))?;
        let (name, utf8_name, _) = decode(root, &raw);
        files.push(File {
            name,
            utf8_name,
            source: Source::Index {
                mode: mode.to_owned(),
                oid: oid.to_owned(),
            },
        });
    }
    Ok(files)
}

/// All problems with one file. Pure, for testing.
pub fn check(path: &str, content: &[u8]) -> Vec<String> {
    let mut problems = Vec::new();
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let allowed = ALLOWED_NAMES.contains(&name)
        || ext
            .as_deref()
            .is_some_and(|e| ALLOWED_EXTENSIONS.contains(&e));
    if !allowed {
        problems.push("file type not on the allowlist".to_owned());
    }
    if content.len() > MAX_FILE_SIZE {
        problems.push(format!(
            "{} bytes exceeds the {MAX_FILE_SIZE}-byte limit",
            content.len()
        ));
    }
    for (magic, what) in FORBIDDEN_MAGIC {
        if content.starts_with(magic) {
            problems.push(format!("looks like a {what}"));
        }
    }
    if content.contains(&0) {
        problems.push("binary content (NUL bytes)".to_owned());
    } else if let Some(run) = longest_encoded_run(content)
        && run > MAX_ENCODED_RUN
    {
        problems.push(format!("{run}-character hex/base64 run (embedded data?)"));
    }
    problems
}

/// Longest run of characters that could be hex or base64 payload.
fn longest_encoded_run(content: &[u8]) -> Option<usize> {
    let is_payload = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=');
    content.split(|b| !is_payload(b)).map(<[u8]>::len).max()
}

#[cfg(test)]
mod tests {
    use super::check;

    /// Enforces the policy wherever tests run, even in clones without the hook.
    #[test]
    fn repository_passes_policy() {
        super::run(&crate::workspace_root(), false).unwrap();
    }

    #[test]
    fn accepts_source() {
        assert!(check("crates/x/src/lib.rs", b"fn main() {}\n").is_empty());
        assert!(check(".gitignore", b"/target/\n").is_empty());
        assert!(check(".githooks/pre-commit", b"#!/bin/sh\n").is_empty());
    }

    #[test]
    fn rejects_game_like_files() {
        assert!(!check("assets/tex.dds", b"DDS \x7c\0\0\0").is_empty());
        assert!(
            !check("notes.txt", b"MZ\x90\0").is_empty(),
            "magic wins over extension"
        );
        assert!(
            !check("data.bin", b"hello").is_empty(),
            "extension not allowed"
        );
        assert!(
            !check("a.rs", &vec![b'a'; 600 * 1024]).is_empty(),
            "too big"
        );
    }

    /// Runs git in `root` with a throwaway identity, asserting success.
    #[cfg(unix)]
    fn git(root: &std::path::Path, args: &[&std::ffi::OsStr]) -> Vec<u8> {
        let out = std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}");
        out.stdout
    }

    #[cfg(unix)]
    fn git_str(root: &std::path::Path, args: &[&str]) -> Vec<u8> {
        let args: Vec<&std::ffi::OsStr> = args.iter().map(std::ffi::OsStr::new).collect();
        git(root, &args)
    }

    #[cfg(unix)]
    fn names(root: &std::path::Path, staged: bool) -> Vec<String> {
        super::list_files(root, staged)
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect()
    }

    /// A symlink turned into a regular file is staged as a type change (`T`),
    /// not a modification; the pre-commit check must still read it.
    #[cfg(unix)]
    #[test]
    fn staged_check_reads_type_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::os::unix::fs::symlink("elsewhere.rs", root.join("data.rs")).unwrap();
        git_str(root, &["add", "data.rs"]);
        git_str(root, &["commit", "-q", "--no-verify", "-m", "link"]);
        std::fs::remove_file(root.join("data.rs")).unwrap();
        std::fs::write(root.join("data.rs"), b"MZ\x90\0binary").unwrap();
        git_str(root, &["add", "data.rs"]);

        assert_eq!(names(root, true), ["data.rs"]);
        assert!(super::run(root, true).is_err());
    }

    /// A renamed file is checked under its new name, with its staged content.
    #[cfg(unix)]
    #[test]
    fn staged_check_reads_renames() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::fs::write(root.join("a.rs"), b"fn a() {}\n").unwrap();
        git_str(root, &["add", "a.rs"]);
        git_str(root, &["commit", "-q", "--no-verify", "-m", "a"]);
        git_str(root, &["mv", "a.rs", "b.rs"]);
        assert_eq!(names(root, true), ["b.rs"]);
        super::run(root, true).unwrap();
    }

    /// A file whose name is not UTF-8 used to be read back under its lossily
    /// decoded name, which does not exist, and so passed unread.
    #[cfg(unix)]
    #[test]
    fn non_utf8_names_are_violations() {
        use std::io::Write;
        use std::os::unix::ffi::OsStrExt;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let mut hash = std::process::Command::new("git")
            .args(["hash-object", "-w", "--stdin"])
            .current_dir(root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        hash.stdin.take().unwrap().write_all(b"MZ\x90\0").unwrap();
        let out = hash.wait_with_output().unwrap();
        let oid = String::from_utf8(out.stdout).unwrap();
        // Staged straight into the index: macOS refuses such names on disk.
        let cacheinfo = format!("100644,{},", oid.trim()).into_bytes();
        let cacheinfo = [cacheinfo.as_slice(), b"data\xff.rs"].concat();
        git(
            root,
            &[
                "update-index".as_ref(),
                "--add".as_ref(),
                "--cacheinfo".as_ref(),
                std::ffi::OsStr::from_bytes(&cacheinfo),
            ],
        );

        let staged = super::list_files(root, true).unwrap();
        assert_eq!(staged.len(), 1);
        assert!(!staged[0].utf8_name);
        assert_eq!(
            super::read(root, &staged[0]).unwrap().as_deref(),
            Some(&b"MZ\x90\0"[..]),
            "staged content is read by object id"
        );
        assert!(super::run(root, true).is_err());
    }

    /// The same, for a file on disk (Linux file systems allow such names).
    #[cfg(target_os = "linux")]
    #[test]
    fn non_utf8_names_on_disk_are_read() {
        use std::os::unix::ffi::OsStrExt;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let name = std::ffi::OsStr::from_bytes(b"data\xff.rs");
        std::fs::write(root.join(name), b"MZ\x90\0").unwrap();

        let files = super::list_files(root, false).unwrap();
        assert_eq!(files.len(), 1);
        assert!(!files[0].utf8_name);
        assert_eq!(
            super::read(root, &files[0]).unwrap().as_deref(),
            Some(&b"MZ\x90\0"[..])
        );
        assert!(super::run(root, false).is_err());
    }

    /// A submodule pulls in another repository's content, which no check
    /// here reads; it is refused whether staged or merely tracked.
    #[cfg(unix)]
    #[test]
    fn submodules_are_violations() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let oid = "0123456789abcdef0123456789abcdef01234567";
        git_str(
            root,
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                &format!("160000,{oid},vendor"),
            ],
        );

        assert_eq!(names(root, true), ["vendor"]);
        assert!(super::run(root, true).is_err());
        // Not checked out, so nothing is on disk.
        assert_eq!(names(root, false), ["vendor"]);
        assert!(super::run(root, false).is_err());
    }

    #[test]
    fn rejects_data_smuggled_into_text() {
        let blob = format!("const DATA: &str = \"{}\";", "A1".repeat(400));
        assert!(!check("src/blob.rs", blob.as_bytes()).is_empty());
    }
}
