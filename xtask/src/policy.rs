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
/// Longest array of numeric literals, or string of `\xHH` escapes, a file may
/// hold. The largest table in the tree (XMA's, 272 entries) is well under.
const MAX_LITERAL_RUN: usize = 512;

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

/// Paths explicitly exempted, each pinned to the BLAKE3 of the one content
/// it is exempted for, with a reason. Keep this empty if at all possible.
const EXCEPTIONS: &[(&str, &str, &str)] = &[(
    "crates/victory-formats/src/bcpack-tables.bin",
    "f0f032aa310e7cfd431c075d2816decdd39d21b1927cbffc38386599c0de73c0",
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

/// First line of a Git LFS pointer, current and pre-1.0 spec. The file
/// itself is small text; what it stands for is stored outside the
/// repository, where this check never sees it.
const LFS_POINTER_PREFIXES: &[&[u8]] = &[
    b"version https://git-lfs.github.com/spec/",
    b"version https://hawser.github.com/spec/",
];

pub fn run(root: &Path, staged: bool) -> Result<()> {
    let files = list_files(root, staged)?;
    let indexed = indexed_blobs(root, &files)?;
    let mut violations = Vec::new();
    for file in &files {
        let path = &file.name;
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
        check_content(path, path, &content, &mut violations);
        // The working copy is not what the repository holds: a tracked file
        // edited after a `--no-verify` commit, or hidden from `git status`
        // by `--assume-unchanged` or `--skip-worktree`, still has its
        // committed content in the index. Checked as well when it differs.
        // A conflicted path has an entry per stage (base, ours, theirs),
        // and each is checked: `git checkout --ours` or `--theirs` puts any
        // one of them in the next commit.
        if let Source::Worktree { index, .. } = &file.source {
            for oid in index {
                match indexed.get(oid) {
                    Some(Some(blob)) if *blob != content => {
                        check_content(path, &format!("{path} (as indexed)"), blob, &mut violations);
                    }
                    Some(Some(_)) => {}
                    _ => violations.push(format!("{path}: cannot read the indexed content")),
                }
            }
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

/// Adds the problems with one file's content, reported as `label`.
fn check_content(path: &str, label: &str, content: &Content, violations: &mut Vec<String>) {
    if content.size > content.head.len() as u64 {
        // Only the start was read: no exception covers it, since the one
        // exempted content is within the limit, and the start shows the rest.
        violations.push(format!(
            "{label}: {} bytes exceeds the {MAX_FILE_SIZE}-byte limit",
            content.size
        ));
        for problem in check(path, content) {
            violations.push(format!("{label}: {problem}"));
        }
        return;
    }
    if let Some(problem) = exempt(path, content) {
        if let Err(problem) = problem {
            violations.push(format!("{label}: {problem}"));
        }
        return;
    }
    for problem in check(path, content) {
        violations.push(format!("{label}: {problem}"));
    }
}

/// Content as read: at most the first `MAX_FILE_SIZE` bytes, and the size of
/// the whole.
///
/// A file over the limit is refused whatever it holds, and is read no
/// further: a game dump or disc image of several gigabytes left in the
/// checkout (untracked, not ignored) or staged by mistake was read into
/// memory whole before being refused as too big, and could exhaust it.
/// The rest is still streamed through, without being kept, to count it.
#[derive(Debug, PartialEq)]
struct Content {
    head: Vec<u8>,
    size: u64,
}

impl std::ops::Deref for Content {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.head
    }
}

impl Content {
    fn read(reader: impl std::io::Read) -> std::io::Result<Self> {
        use std::io::Read;
        let mut reader = reader;
        let mut head = Vec::new();
        (&mut reader)
            .take(MAX_FILE_SIZE as u64)
            .read_to_end(&mut head)?;
        let rest = std::io::copy(&mut reader, &mut std::io::sink())?;
        Ok(Self {
            size: head.len() as u64 + rest,
            head,
        })
    }
}

/// The index's blob for every tracked file of the full check, by object id,
/// read with one `git cat-file --batch` rather than a process per file;
/// `None` for an object git does not have.
fn indexed_blobs(
    root: &Path,
    files: &[File],
) -> Result<std::collections::HashMap<String, Option<Content>>> {
    use std::io::{BufRead, Read, Write};
    let mut oids: Vec<&str> = files
        .iter()
        .filter_map(|file| match &file.source {
            Source::Worktree { index, .. } => Some(index.iter().map(String::as_str)),
            Source::Index { .. } => None,
        })
        .flatten()
        .collect();
    oids.sort_unstable();
    oids.dedup();
    let mut blobs = std::collections::HashMap::new();
    if oids.is_empty() {
        return Ok(blobs);
    }
    let mut child = git_command(root)
        .args(["cat-file", "--batch"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .context("running git cat-file")?;
    let mut stdin = child.stdin.take().context("git cat-file stdin")?;
    let request: String = oids.iter().map(|oid| format!("{oid}\n")).collect();
    // Written from another thread: git answers while it reads, and a full
    // stdout pipe would otherwise stop both.
    let writer = std::thread::spawn(move || stdin.write_all(request.as_bytes()));
    let mut out = std::io::BufReader::new(child.stdout.take().context("git cat-file stdout")?);
    for oid in &oids {
        // `<oid> <type> <size>\n<content>\n`, or `<oid> missing\n`.
        let mut header = String::new();
        out.read_line(&mut header).context("reading git cat-file")?;
        let size = match header.split_whitespace().collect::<Vec<_>>()[..] {
            [_, _, size] => size.parse::<u64>().ok(),
            _ => None,
        };
        let Some(size) = size else {
            if header.is_empty() {
                bail!("git cat-file ended early");
            }
            blobs.insert((*oid).to_owned(), None);
            continue;
        };
        let content = Content::read((&mut out).take(size)).context("reading git cat-file")?;
        if content.size != size {
            bail!("git cat-file ended early");
        }
        out.read_exact(&mut [0]).context("reading git cat-file")?;
        blobs.insert((*oid).to_owned(), Some(content));
    }
    writer
        .join()
        .map_err(|_| anyhow::anyhow!("git cat-file writer panicked"))?
        .context("writing to git cat-file")?;
    child.wait().context("waiting for git cat-file")?;
    Ok(blobs)
}

/// One path to check.
struct File {
    /// Repository-relative, `/`-separated, lossily decoded if need be.
    name: String,
    utf8_name: bool,
    source: Source,
}

enum Source {
    /// On disk, at `path`. `index` is the object ids of the blobs the index
    /// holds for it, one per stage, none if untracked: the content that
    /// stays in the repository, and in the next commit, while the file is
    /// missing from disk.
    Worktree { path: PathBuf, index: Vec<String> },
    /// Staged: the index entry's mode and object id.
    Index { mode: String, oid: String },
}

/// The content to check; `Ok(None)` if there is nothing to check (a file
/// gone from both the working tree and the index), `Err` with the problem if
/// a path the check cannot read must not pass unread.
fn read(root: &Path, file: &File) -> Result<Option<Content>, String> {
    match &file.source {
        Source::Index { mode, .. } if mode == "160000" => {
            Err("a submodule: content from another repository bypasses this check".to_owned())
        }
        Source::Index { oid, .. } => read_blob(root, oid).map(Some),
        Source::Worktree { path, index } => {
            // Not `metadata`: that follows a symlink, and git stores a
            // symlink's target path, not what it points to.
            let kind = match std::fs::symlink_metadata(path) {
                Ok(meta) => meta.file_type(),
                // Deleted but not `git rm`ed, or left out of a sparse
                // checkout: the index still holds the file, and a plain
                // `git commit` keeps it. Committed with `--no-verify` and
                // then removed, it used to pass the full check unread.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // `run` checks the other stages against this one.
                    return index.first().map(|oid| read_blob(root, oid)).transpose();
                }
                Err(e) => return Err(format!("cannot read: {e}")),
            };
            if kind.is_symlink() {
                let target = std::fs::read_link(path).map_err(|e| format!("cannot read: {e}"))?;
                let head = link_bytes(&target);
                let size = head.len() as u64;
                return Ok(Some(Content { head, size }));
            }
            if kind.is_dir() {
                // `git ls-files --others` lists an untracked repository
                // inside this one as its directory, with a trailing `/`.
                return Err(if file.name.ends_with('/') {
                    "a git repository nested in this one: its files bypass this check"
                } else {
                    "a directory tracked as a file (a submodule?)"
                }
                .to_owned());
            }
            if !kind.is_file() {
                // A FIFO or device: reading it could block or never end.
                return Err("not a regular file".to_owned());
            }
            std::fs::File::open(path)
                .and_then(Content::read)
                .map(Some)
                .map_err(|e| format!("cannot read: {e}"))
        }
    }
}

/// A blob's content. By object id, not `:<path>`, so no file name can make
/// it fail.
/// Streamed, so a blob over the limit is not held in memory whole.
fn read_blob(root: &Path, oid: &str) -> Result<Content, String> {
    let mut child = git_command(root)
        .args(["cat-file", "blob", oid])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot run git: {e}"))?;
    let content = child.stdout.take().map(Content::read);
    let status = child.wait().map_err(|e| format!("cannot run git: {e}"))?;
    match content {
        Some(Ok(content)) if status.success() => Ok(content),
        _ => Err("cannot read the indexed content".to_owned()),
    }
}

/// A symlink's content as git stores it: the target path.
fn link_bytes(target: &Path) -> Vec<u8> {
    #[cfg(unix)]
    let bytes =
        <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::as_bytes(target.as_os_str()).to_vec();
    // Git for Windows stores the target as UTF-8 with `/` separators.
    #[cfg(not(unix))]
    let bytes = target.to_string_lossy().replace('\\', "/").into_bytes();
    bytes
}

/// Variables through which the environment, not `root`, picks the repository
/// git works on (`git rev-parse --local-env-vars`).
#[cfg(test)]
const REPOSITORY_ENV: &[&str] = &[
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
];

/// Variables that change how git reads a pathspec: literal, glob or
/// case-insensitive. Tests pass pathspecs (`check-ignore -- <path>`, the
/// hooks' `ls-files -- .githooks`), and `check-ignore` refuses to run at all
/// under `GIT_LITERAL_PATHSPECS`, so `tempdir` would take an ignored
/// directory for a visible one.
#[cfg(test)]
const PATHSPEC_ENV: &[&str] = &[
    "GIT_LITERAL_PATHSPECS",
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
];

/// `git`, run in `root`.
///
/// The hook needs the environment it inherits: `git commit -a` and
/// `git commit <path>` stage into a temporary index that only
/// `GIT_INDEX_FILE` names. Tests point `root` at temporary repositories
/// instead, and when they run from a hook those variables would send every
/// one of them to the same, outside index, so there they are dropped.
///
/// Temporary repositories also read no global or system configuration: a
/// developer's `core.excludesFile` would hide the files a test stages, and
/// their `core.hooksPath` would run their hooks inside it. Nor do they read
/// the ignore and attributes files git looks for without any configuration
/// (`$XDG_CONFIG_HOME/git/ignore`, else `~/.config/git/ignore`, the same for
/// `attributes`, and the system gitattributes): a `*.rs` there hides what a
/// test stages just as an excludes file would, and a `filter=` there changes
/// what it stages. The checkout itself keeps all of that, since a global
/// excludes file is how files that will never be added (`.DS_Store`) stay
/// out of the full check.
///
/// Nor are they made from the environment's choices: `GIT_TEMPLATE_DIR`
/// copies its `info/exclude` and hooks into every `git init`, and
/// `GIT_DEFAULT_HASH=sha256` makes repositories that the SHA-1 object ids
/// tests stage are refused by. `GIT_ATTR_SOURCE` reads attributes from a
/// tree-ish that a new repository does not have, and every command fails.
/// Nor do they look for a repository above themselves.
///
/// Nor does a hook's quarantine reach them: a `pre-receive` hook runs with
/// `GIT_QUARANTINE_PATH` set beside the object directories dropped above,
/// and with it git refuses every ref update, `git init` included. And since
/// they read no global configuration, the `safe.directory` a developer
/// keeps there for a TMPDIR owned by someone else (a shared one, or one an
/// elevated Windows shell made, which belongs to Administrators) cannot
/// reach them either; they trust themselves, which a test made them, as
/// command-line configuration that git honours for it.
pub(crate) fn git_command(root: &Path) -> Command {
    let mut command = Command::new("git");
    // `git replace` refs are local and never pushed, yet git reads through
    // them by default: replacing a staged blob shows `cat-file` other
    // content than the commit will hold, and replacing HEAD with a commit
    // whose tree already has a staged file drops it from `diff --cached`.
    // The check reads the objects as they will be committed.
    command.current_dir(root).arg("--no-replace-objects");
    unredirect_output(&mut command, std::env::vars_os().map(|(key, _)| key));
    #[cfg(test)]
    {
        for var in REPOSITORY_ENV.iter().chain(PATHSPEC_ENV) {
            command.env_remove(var);
        }
        if root != crate::workspace_root() {
            // Git for Windows opens "/dev/null" as NUL.
            command
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_ATTR_NOSYSTEM", "1")
                .env("XDG_CONFIG_HOME", NO_XDG_CONFIG)
                // Empty: `git init` copies no template at all, silently.
                .env("GIT_TEMPLATE_DIR", "")
                .env_remove("GIT_DEFAULT_HASH")
                // Reads `.gitattributes` from that tree-ish instead of the
                // working tree; a fresh repository has no HEAD to read.
                .env_remove("GIT_ATTR_SOURCE")
                .env_remove("GIT_QUARANTINE_PATH")
                .args(["-c", "safe.directory=*"]);
            // Git looks no further up than `root` itself: a directory that is
            // not a repository is walked, not taken for part of whatever
            // repository encloses it (the checkout, when the temporary
            // directory is made beside the test binary). Git stops before
            // entering a ceiling, so the ceiling is the parent.
            #[cfg(unix)]
            if let Some(parent) = root.is_absolute().then(|| root.parent()).flatten() {
                command.env("GIT_CEILING_DIRECTORIES", parent);
            }
        }
    }
    command
}

/// Keeps git's stdout to what the command prints, since the check parses it.
///
/// `GIT_TRACE=/dev/stdout` (or `GIT_TRACE_SETUP`, `GIT_TRACE2_EVENT`, any
/// of the trace variables pointed at stdout) puts trace lines in front of
/// the first NUL-separated field. In the full check that field is the first
/// file listed: its name no longer exists, so it is never read, and that
/// file passes whatever it holds. `--staged` stops at a record it cannot
/// parse, and every test that reads git's output failed. Git for Windows'
/// `GIT_REDIRECT_STDOUT` sends the output to a file instead, and the check
/// sees an empty repository. Trace2 can also be turned on from global or
/// system configuration, which its variables override, so those are set
/// off rather than only dropped.
fn unredirect_output(command: &mut Command, keys: impl IntoIterator<Item = std::ffi::OsString>) {
    for key in keys {
        // Variable names are case-insensitive on Windows.
        let upper = key.to_string_lossy().to_ascii_uppercase();
        if upper.starts_with("GIT_TRACE") || upper.starts_with("GIT_REDIRECT_") {
            command.env_remove(key);
        }
    }
    for var in ["GIT_TRACE2", "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF"] {
        command.env(var, "0");
    }
}

/// A configuration directory that never exists, so git finds no `git/ignore`
/// or `git/attributes` in it. Relative paths would resolve against each
/// temporary repository, where a test could create them.
#[cfg(test)]
const NO_XDG_CONFIG: &str = if cfg!(windows) {
    "NUL:/victory-policy-tests"
} else {
    "/dev/null/victory-policy-tests"
};

/// Splits NUL-terminated git output; each field is returned raw.
fn git_z(root: &Path, args: &[&str]) -> Result<Option<Vec<Vec<u8>>>> {
    let out = git_command(root)
        .args(args)
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
        // only the index says what it is. Each entry is
        // `<mode> <oid> <stage>\t<path>`; a conflicted path has several,
        // and any one of them is content the check must see.
        let mut index: std::collections::HashMap<Vec<u8>, Vec<(String, String)>> =
            std::collections::HashMap::new();
        for entry in git_z(root, &["ls-files", "-z", "--stage"])?.unwrap_or_default() {
            let Some(tab) = entry.iter().position(|&b| b == b'\t') else {
                continue;
            };
            let Ok(header) = std::str::from_utf8(&entry[..tab]) else {
                continue;
            };
            let mut parts = header.split(' ');
            if let (Some(mode), Some(oid)) = (parts.next(), parts.next()) {
                index
                    .entry(entry[tab + 1..].to_vec())
                    .or_default()
                    .push((mode.to_owned(), oid.to_owned()));
            }
        }
        let mut files: Vec<File> = fields
            .iter()
            .map(|raw| {
                let (name, utf8_name, on_disk) = decode(root, raw);
                let stages = index.get(raw).map(Vec::as_slice).unwrap_or_default();
                let source = if stages.iter().any(|(mode, _)| mode == "160000") {
                    Source::Index {
                        mode: "160000".to_owned(),
                        oid: String::new(),
                    }
                } else {
                    Source::Worktree {
                        path: on_disk,
                        index: stages.iter().map(|(_, oid)| oid.clone()).collect(),
                    }
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
                source: Source::Worktree {
                    path: e.path().to_owned(),
                    index: Vec::new(),
                },
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
    // addition it is, with one path per record. --ignore-submodules=none:
    // `diff.ignoreSubmodules` or an `ignore = all` line in .gitmodules (read
    // from the working tree, even untracked) would otherwise leave a staged
    // submodule out of the list.
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
            "--ignore-submodules=none",
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

/// `None` if `path` has no exception; otherwise whether `content` is the
/// content it was granted for. The exception covers that content only, so
/// nothing else (an asset, an executable, a file of any size) can be
/// committed under the exempted name.
fn exempt(path: &str, content: &[u8]) -> Option<Result<(), String>> {
    let (_, pinned, _) = EXCEPTIONS.iter().find(|(p, ..)| *p == path)?;
    let actual = blake3::hash(content).to_hex();
    Some(if actual.as_str() == *pinned {
        Ok(())
    } else {
        Err(format!(
            "exempted only with BLAKE3 {pinned}, but its content hashes to {actual}; \
             a new version needs the owner's approval and its hash in xtask/src/policy.rs"
        ))
    })
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
    if LFS_POINTER_PREFIXES
        .iter()
        .any(|prefix| content.starts_with(prefix))
    {
        problems.push(
            "a Git LFS pointer: the content it stands for is stored outside the repository, \
             unchecked"
                .to_owned(),
        );
    }
    if content.contains(&0) {
        problems.push("binary content (NUL bytes)".to_owned());
    } else {
        // Binary data need not hold a NUL: byte stuffing such as COBS
        // removes every one, and what is left passed as text, since bytes
        // over 0x7F break every run below. Text is UTF-8 here, without
        // control characters other than tab, line breaks and form feed.
        match std::str::from_utf8(content) {
            Err(e) => problems.push(format!(
                "not UTF-8 text (binary content?): invalid byte at offset {}",
                e.valid_up_to()
            )),
            Ok(text) => {
                if let Some((at, c)) = text
                    .char_indices()
                    .find(|&(_, c)| c.is_control() && !matches!(c, '\t' | '\n' | '\r' | '\x0c'))
                {
                    problems.push(format!(
                        "control character U+{:04X} at offset {at} (binary content?)",
                        c as u32
                    ));
                }
            }
        }
        let run = longest_encoded_run(content);
        if run > MAX_ENCODED_RUN {
            problems.push(format!("{run}-character hex/base64 run (embedded data?)"));
        } else {
            let run = longest_printable_run(content);
            if run > MAX_ENCODED_RUN {
                problems.push(format!(
                    "{run}-character run without whitespace (ascii85, Z85, uuencoded or base65536 data?)"
                ));
            }
        }
        let run = longest_literal_run(content);
        if run > MAX_LITERAL_RUN {
            problems.push(format!(
                "{run}-element numeric array or byte-escape string (embedded data?)"
            ));
        }
    }
    problems
}

/// Longest run of characters that could be hex or base64 payload.
///
/// A single line break (LF or CRLF) between two payload characters does not
/// end a run: `base64` wraps its output at 76 columns, PEM and
/// `openssl base64` at 64, `xxd -p` at 60, so data encoded by any of them
/// would otherwise pass as many short runs. The break itself is not counted.
///
/// Both base64 alphabets count, each on its own: the standard one (`+`, `/`)
/// and the URL- and filename-safe one (`-`, `_`; RFC 4648 section 5) that
/// JWTs, `basenc --base64url` and most web tooling write. Measured with the
/// standard alphabet alone, base64url breaks at every `-` and `_`, one
/// character in 32 on average, and a file of it passes as short runs.
fn longest_encoded_run(content: &[u8]) -> usize {
    longest_run(content, b"+/=").max(longest_run(content, b"-_="))
}

/// Longest run of alphanumerics and `extra` bytes, joined across single line
/// breaks as `longest_encoded_run` describes.
fn longest_run(content: &[u8], extra: &[u8]) -> usize {
    let is_payload = |b: &u8| b.is_ascii_alphanumeric() || extra.contains(b);
    let (mut longest, mut run, mut i) = (0, 0, 0);
    while let Some(b) = content.get(i) {
        if is_payload(b) {
            run += 1;
            longest = longest.max(run);
            i += 1;
            continue;
        }
        let line_break = match &content[i..] {
            [b'\r', b'\n', ..] => 2,
            [b'\n', ..] => 1,
            _ => 0,
        };
        if line_break > 0 && run > 0 && content.get(i + line_break).is_some_and(is_payload) {
            i += line_break;
            continue;
        }
        run = 0;
        i += 1;
    }
    longest
}

/// Shortest line after which a line break continues a printable run.
const MIN_WRAPPED_LINE: usize = 40;

/// Longest run of characters that are neither whitespace nor control
/// characters, counted in characters.
///
/// ascii85 (`!` to `u`: Python's `a85encode`, PostScript, PDF), Z85
/// (`basenc --z85`, ZeroMQ) and uuencode (`` ` `` to `_`) put punctuation in
/// almost every few characters, so their output never makes a long hex or
/// base64 run and a file of it passed. Text has whitespace far more often:
/// the longest such run in the tree is a few dozen characters.
///
/// Characters beyond ASCII count as well: base32768 and base65536 carry 15
/// or 16 bits in each CJK or other letter they write, with no whitespace,
/// and passed as text that no ASCII run saw at all.
///
/// A single line break continues the run as for base64, but only after a
/// line of at least `MIN_WRAPPED_LINE` characters: encoders wrap at 60 to 80
/// columns (`basenc` 76, `uuencode` 61), while a list of short entries
/// without spaces, such as `.gitignore` patterns, is not one long run.
fn longest_printable_run(content: &[u8]) -> usize {
    let text = String::from_utf8_lossy(content);
    let content: Vec<char> = text.chars().collect();
    let is_payload = |c: &char| !c.is_whitespace() && !c.is_control();
    let (mut longest, mut run, mut line, mut i) = (0, 0, 0, 0);
    while let Some(c) = content.get(i) {
        if is_payload(c) {
            run += 1;
            line += 1;
            longest = longest.max(run);
            i += 1;
            continue;
        }
        let line_break = match &content[i..] {
            ['\r', '\n', ..] => 2,
            ['\n', ..] => 1,
            _ => 0,
        };
        if line_break > 0
            && line >= MIN_WRAPPED_LINE
            && content.get(i + line_break).is_some_and(is_payload)
        {
            line = 0;
            i += line_break;
            continue;
        }
        (run, line) = (0, 0);
        i += 1;
    }
    longest
}

/// Longest list of numeric literals separated by commas, or run of `\xHH`
/// or `\u{..}` escapes (across string boundaries, as `string_join_len`
/// describes), in `content`.
///
/// Neither is a hex/base64 run: `xxd -i` and `bin2c` write a file as
/// `0x4d, 0x5a, 0x90, ...`, and a byte string as `"\x4d\x5a\x90..."`,
/// each token two to four characters long. So an executable pasted into a
/// `const DATA: [u8; N]` passed as long as the file stayed under the size
/// limit. Whitespace and line breaks around the commas do not end a list;
/// anything else (a bracket, a comment, an identifier) does, so the rows of
/// a two-dimensional table count one by one.
fn longest_literal_run(content: &[u8]) -> usize {
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let (mut longest, mut i) = (0, 0);
    // Comma-separated literals: `0x4d`, `77u8`, `1_000`, `0.5`, `-3`.
    let mut run = 0;
    while i < content.len() {
        let starts_literal = match content[i] {
            b'0'..=b'9' => true,
            b'-' => content.get(i + 1).is_some_and(u8::is_ascii_digit),
            _ => false,
        } && (i == 0 || !ident(content[i - 1]));
        if !starts_literal {
            run = 0;
            i += 1;
            continue;
        }
        i += 1;
        while content.get(i).is_some_and(|&b| ident(b) || b == b'.') {
            i += 1;
        }
        run += 1;
        longest = longest.max(run);
        let mut j = i;
        while content.get(j).is_some_and(u8::is_ascii_whitespace) {
            j += 1;
        }
        if content.get(j) == Some(&b',') {
            j += 1;
            while content.get(j).is_some_and(u8::is_ascii_whitespace) {
                j += 1;
            }
            i = j;
        } else {
            run = 0;
        }
    }
    // Escapes back to back, joined across string boundaries.
    let (mut run, mut i) = (0, 0);
    while i < content.len() {
        if let Some(len) = escape_len(&content[i..]) {
            run += 1;
            longest = longest.max(run);
            i += len;
            let join = string_join_len(&content[i..]);
            if join > 0 && escape_len(&content[i + join..]).is_some() {
                i += join;
            }
            continue;
        }
        run = 0;
        i += 1;
    }
    longest
}

/// Length of the byte escape at the start of `rest`: `\xHH` (Rust, C,
/// HLSL) or `\u{H}` to `\u{HHHHHH}` (Rust), which `escape_default` and
/// `{:?}` write for any character outside printable ASCII.
fn escape_len(rest: &[u8]) -> Option<usize> {
    match rest {
        [b'\\', b'x', h, l, ..] if h.is_ascii_hexdigit() && l.is_ascii_hexdigit() => Some(4),
        [b'\\', b'u', b'{', digits @ ..] => {
            let n = digits.iter().take_while(|b| b.is_ascii_hexdigit()).count();
            ((1..=6).contains(&n) && digits.get(n) == Some(&b'}')).then_some(n + 4)
        }
        _ => None,
    }
}

/// Length of what closes one string of escapes and opens the next, so that
/// the strings count as one run: a Rust string continuation (`\` and a line
/// break, then indentation), adjacent literals (`"\x4d" "\x5a"`, which C and
/// HLSL concatenate), or literals separated by commas (`concat!("\x4d",
/// "\x5a")`, `[b"\x4d", b"\x5a"]`). Without it, a string of escapes wrapped
/// every hundred escapes passed whatever its length. Zero if `rest` starts
/// with anything else, or with whitespace alone: inside one string that is
/// text, as in `"\x1b[0m \x1b[1m"`.
fn string_join_len(rest: &[u8]) -> usize {
    let (mut i, mut boundary) = (0, false);
    loop {
        let len = match &rest[i..] {
            [b'\\', b'\r', b'\n', ..] => 3,
            [b'\\', b'\n', ..] => 2,
            [b'b', b'"', ..] => 2,
            [b'"', ..] => 1,
            [b, ..] if b.is_ascii_whitespace() || *b == b',' => {
                i += 1;
                continue;
            }
            _ => return if boundary { i } else { 0 },
        };
        (i, boundary) = (i + len, true);
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_ENCODED_RUN, MAX_LITERAL_RUN, check};

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
    ///
    /// The identity and dates are set as variables, not `user.*` config:
    /// the `GIT_AUTHOR_*` and `GIT_COMMITTER_*` an environment sets
    /// (`git commit --date` and `git rebase` export them to hooks) win over
    /// config, and an empty name or a date git cannot parse fails the commit.
    #[cfg(unix)]
    fn git(root: &std::path::Path, args: &[&std::ffi::OsStr]) -> Vec<u8> {
        let out = super::git_command(root)
            .envs(IDENTITY)
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}");
        out.stdout
    }

    /// A fixed author and committer, so commits in temporary repositories
    /// neither take the environment's nor fail on it.
    #[cfg(unix)]
    const IDENTITY: [(&str, &str); 6] = [
        ("GIT_AUTHOR_NAME", "t"),
        ("GIT_AUTHOR_EMAIL", "t@t"),
        ("GIT_AUTHOR_DATE", "@0 +0000"),
        ("GIT_COMMITTER_NAME", "t"),
        ("GIT_COMMITTER_EMAIL", "t@t"),
        ("GIT_COMMITTER_DATE", "@0 +0000"),
    ];

    #[cfg(unix)]
    fn git_str(root: &std::path::Path, args: &[&str]) -> Vec<u8> {
        let args: Vec<&std::ffi::OsStr> = args.iter().map(std::ffi::OsStr::new).collect();
        git(root, &args)
    }

    /// A temporary directory the checkout's own full check cannot see.
    ///
    /// Under a TMPDIR inside the checkout, a test's temporary repository is
    /// an untracked directory there, and `repository_passes_policy`, running
    /// at the same time, fails on it and on the files the test stages. The
    /// directory is then made beside the test binary instead, which is
    /// ignored as build output; if that is not ignored either, the test says
    /// so rather than flaking.
    fn tempdir() -> tempfile::TempDir {
        let checkout = crate::workspace_root();
        let checkout = checkout.canonicalize().unwrap_or(checkout);
        let candidates = [
            Some(std::env::temp_dir()),
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf)),
        ];
        for dir in candidates.into_iter().flatten() {
            let Ok(dir) = dir.canonicalize() else {
                continue;
            };
            if unseen_by_full_check(&checkout, &dir) {
                return tempfile::tempdir_in(dir).unwrap();
            }
        }
        panic!(
            "no temporary directory outside {} or ignored by it: set TMPDIR outside the checkout",
            checkout.display()
        );
    }

    /// Whether nothing under `dir` is listed by the full check of `checkout`:
    /// it is elsewhere, or ignored there.
    fn unseen_by_full_check(checkout: &std::path::Path, dir: &std::path::Path) -> bool {
        let Ok(relative) = dir.strip_prefix(checkout) else {
            return true;
        };
        let status = super::git_command(checkout)
            .args(["check-ignore", "-q", "--"])
            .arg(relative.join("probe"))
            .status();
        match status.ok().and_then(|s| s.code()) {
            Some(0) => true,
            Some(1) => false,
            // Not a git checkout: the walk skips only `target` and `.git`.
            _ => relative
                .components()
                .any(|c| matches!(c.as_os_str().to_str(), Some("target" | ".git"))),
        }
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
        let tmp = tempdir();
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

    /// Replace refs stay local, so the hook must not read through them: a
    /// staged blob replaced by harmless text, or a HEAD replaced by a commit
    /// whose tree already holds the staged file, used to pass.
    #[cfg(unix)]
    #[test]
    fn staged_check_ignores_replace_refs() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::fs::write(root.join("a.rs"), b"fn a() {}\n").unwrap();
        git_str(root, &["add", "a.rs"]);
        git_str(root, &["commit", "-q", "--no-verify", "-m", "a"]);
        std::fs::write(root.join("b.rs"), b"MZ\x90\0binary").unwrap();
        git_str(root, &["add", "b.rs"]);

        let staged = git_str(root, &["rev-parse", ":b.rs"]);
        let staged = String::from_utf8(staged).unwrap();
        std::fs::write(root.join("ok.rs"), b"fn ok() {}\n").unwrap();
        let harmless = git_str(root, &["hash-object", "-w", "ok.rs"]);
        let harmless = String::from_utf8(harmless).unwrap();
        std::fs::remove_file(root.join("ok.rs")).unwrap();
        git_str(root, &["replace", staged.trim(), harmless.trim()]);
        assert!(super::run(root, true).is_err(), "replaced blob");
        git_str(root, &["replace", "-d", staged.trim()]);

        let tree = git_str(root, &["write-tree"]);
        let tree = String::from_utf8(tree).unwrap();
        let fake = git_str(root, &["commit-tree", tree.trim(), "-p", "HEAD", "-m", "x"]);
        let fake = String::from_utf8(fake).unwrap();
        git_str(root, &["replace", "HEAD", fake.trim()]);
        assert_eq!(names(root, true), ["b.rs"]);
        assert!(super::run(root, true).is_err(), "replaced HEAD");
    }

    /// Git stores a symlink as its target path. The full check used to
    /// follow the link instead: it refused a link to a binary outside the
    /// repository that the hook had accepted, passed a dangling link whose
    /// path the hook refuses, and would block on a link to a FIFO.
    #[cfg(unix)]
    #[test]
    fn symlinks_are_checked_as_git_stores_them() {
        let outside = tempdir();
        let binary = outside.path().join("game.bin");
        std::fs::write(&binary, b"MZ\x90\0binary").unwrap();
        let fifo = outside.path().join("fifo");
        let made = std::process::Command::new("mkfifo").arg(&fifo).status();
        assert!(made.unwrap().success(), "mkfifo");

        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::os::unix::fs::symlink(&binary, root.join("binary.rs")).unwrap();
        std::os::unix::fs::symlink(&fifo, root.join("fifo.rs")).unwrap();
        git_str(root, &["add", "binary.rs", "fifo.rs"]);
        super::run(root, true).unwrap();
        super::run(root, false).unwrap();

        // A path that is itself a long base64-like run, pointing nowhere.
        let target = format!("{}/x", "a".repeat(super::MAX_ENCODED_RUN));
        std::os::unix::fs::symlink(&target, root.join("dangling.rs")).unwrap();
        let files = super::list_files(root, false).unwrap();
        let dangling = files.iter().find(|f| f.name == "dangling.rs").unwrap();
        assert_eq!(
            super::read(root, dangling).unwrap().as_deref(),
            Some(target.as_bytes())
        );
        assert!(super::run(root, false).is_err());
        git_str(root, &["add", "dangling.rs"]);
        assert!(super::run(root, true).is_err());
    }

    /// An exception covers the content it was granted for, not whatever is
    /// later committed under its name.
    #[test]
    fn exceptions_are_pinned_to_their_content() {
        assert!(super::exempt("src/lib.rs", b"fn main() {}\n").is_none());
        for (path, pinned, _) in super::EXCEPTIONS {
            assert_eq!(pinned.len(), 64, "{path}: pin a full BLAKE3 hex digest");
            let err = super::exempt(path, b"MZ\x90\0anything").unwrap();
            assert!(err.is_err(), "{path}: other content passed");
            assert!(super::exempt(path, b"").unwrap().is_err());
        }
    }

    /// The full and staged checks both refuse new content at an exempted
    /// path, which used to pass whatever it held.
    #[cfg(unix)]
    #[test]
    fn exempted_paths_refuse_other_content() {
        let (path, ..) = super::EXCEPTIONS[0];
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let file = root.join(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"MZ\x90\0a game executable").unwrap();
        git_str(root, &["add", path]);
        assert!(super::run(root, false).is_err());
        assert!(super::run(root, true).is_err());

        let real = crate::workspace_root().join(path);
        if let Ok(content) = std::fs::read(real) {
            std::fs::write(&file, content).unwrap();
            git_str(root, &["add", path]);
            super::run(root, false).unwrap();
            super::run(root, true).unwrap();
        }
    }

    /// A renamed file is checked under its new name, with its staged content.
    #[cfg(unix)]
    #[test]
    fn staged_check_reads_renames() {
        let tmp = tempdir();
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

        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let mut hash = super::git_command(root)
            .args(["hash-object", "-w", "--stdin"])
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

        let tmp = tempdir();
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

    /// A tracked file missing from disk is still in the index, and a plain
    /// `git commit` keeps it; the full check reads it from there. Committed
    /// with `--no-verify` and then deleted, or hidden from a sparse checkout,
    /// it used to pass unread.
    #[cfg(unix)]
    #[test]
    fn tracked_files_missing_from_disk_are_read_from_the_index() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::fs::write(root.join("a.rs"), b"fn a() {}\n").unwrap();
        std::fs::write(root.join("game.rs"), b"MZ\x90\0binary").unwrap();
        git_str(root, &["add", "a.rs", "game.rs"]);
        git_str(root, &["commit", "-q", "--no-verify", "-m", "x"]);

        std::fs::remove_file(root.join("game.rs")).unwrap();
        assert_eq!(names(root, false), ["a.rs", "game.rs"]);
        assert!(super::run(root, false).is_err(), "deleted, not staged");

        git_str(root, &["update-index", "--skip-worktree", "game.rs"]);
        assert!(
            super::run(root, false).is_err(),
            "outside the sparse checkout"
        );
        git_str(root, &["update-index", "--no-skip-worktree", "game.rs"]);

        // Removed from the index too: nothing of it is left to check.
        git_str(root, &["rm", "-q", "--cached", "game.rs"]);
        assert_eq!(names(root, false), ["a.rs"]);
        super::run(root, false).unwrap();

        // A deleted symlink is read as the target path git stores.
        let target = format!("{}/x", "a".repeat(super::MAX_ENCODED_RUN));
        std::os::unix::fs::symlink(&target, root.join("link.rs")).unwrap();
        git_str(root, &["add", "link.rs"]);
        std::fs::remove_file(root.join("link.rs")).unwrap();
        let files = super::list_files(root, false).unwrap();
        let link = files.iter().find(|f| f.name == "link.rs").unwrap();
        assert_eq!(
            super::read(root, link).unwrap().as_deref(),
            Some(target.as_bytes())
        );
    }

    /// The full check reads what the index holds as well as the working
    /// copy: a binary committed with `--no-verify` and then overwritten on
    /// disk with text stays in the repository, and `--assume-unchanged` or
    /// `--skip-worktree` hide it from `git status` too.
    #[cfg(unix)]
    #[test]
    fn tracked_files_are_read_as_indexed_as_well() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::fs::write(root.join("a.rs"), b"fn a() {}\n").unwrap();
        std::fs::write(root.join("game.rs"), b"MZ\x90\0binary").unwrap();
        git_str(root, &["add", "a.rs", "game.rs"]);
        git_str(root, &["commit", "-q", "--no-verify", "-m", "x"]);

        std::fs::write(root.join("game.rs"), b"fn game() {}\n").unwrap();
        assert!(super::run(root, false).is_err(), "edited on disk");
        for flag in ["--assume-unchanged", "--skip-worktree"] {
            git_str(root, &["update-index", flag, "game.rs"]);
            assert!(super::run(root, false).is_err(), "{flag}");
            git_str(
                root,
                &["update-index", &flag.replace("--", "--no-"), "game.rs"],
            );
        }
        let indexed = super::indexed_blobs(root, &super::list_files(root, false).unwrap()).unwrap();
        assert_eq!(indexed.len(), 2);
        assert!(
            indexed
                .values()
                .any(|b| b.as_deref() == Some(b"MZ\x90\0binary"))
        );

        // Staged, the index holds the text too, and both checks pass.
        git_str(root, &["add", "game.rs"]);
        super::run(root, false).unwrap();
        super::run(root, true).unwrap();

        // Text on disk over text in the index is read once.
        std::fs::write(root.join("a.rs"), b"fn a() { 1 }\n").unwrap();
        super::run(root, false).unwrap();
    }

    /// A file over the limit is read no further than the limit, on disk, in
    /// the index and when staged, and is refused with its whole size.
    #[cfg(unix)]
    #[test]
    fn files_over_the_limit_are_not_read_whole() {
        const BIG: u64 = 8 << 20;
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        // Sparse, so the test writes almost nothing to disk.
        std::fs::File::create(root.join("game.rs"))
            .unwrap()
            .set_len(BIG)
            .unwrap();
        let read = |staged: bool| {
            let files = super::list_files(root, staged).unwrap();
            assert_eq!(files.len(), 1);
            let content = super::read(root, &files[0]).unwrap().unwrap();
            assert_eq!(content.size, BIG);
            assert_eq!(content.head.len(), super::MAX_FILE_SIZE);
        };
        read(false);
        assert!(super::run(root, false).is_err(), "untracked");

        git_str(root, &["add", "game.rs"]);
        read(true);
        assert!(super::run(root, true).is_err(), "staged");
        std::fs::write(root.join("game.rs"), b"fn game() {}\n").unwrap();
        let files = super::list_files(root, false).unwrap();
        let indexed = super::indexed_blobs(root, &files).unwrap();
        let blob = indexed.values().next().unwrap().as_ref().unwrap();
        assert_eq!((blob.size, blob.head.len()), (BIG, super::MAX_FILE_SIZE));
        assert!(super::run(root, false).is_err(), "as indexed");

        // At the limit exactly, the whole file is read and nothing is cut.
        let edge = vec![b'\n'; super::MAX_FILE_SIZE];
        let content = super::Content::read(&edge[..]).unwrap();
        assert_eq!(
            (content.size, content.head.len()),
            (edge.len() as u64, edge.len())
        );
        std::fs::write(root.join("game.rs"), &edge).unwrap();
        git_str(root, &["add", "game.rs"]);
        super::run(root, false).unwrap();
        super::run(root, true).unwrap();
    }

    /// A conflicted path has one index entry per stage, and the full check
    /// reads every one of them, not only the last: `git checkout --ours`
    /// puts stage 2 back in the next commit whatever the working copy and
    /// stage 3 hold.
    #[cfg(unix)]
    #[test]
    fn every_stage_of_a_conflicted_path_is_read() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let blob = |content: &[u8]| {
            std::fs::write(root.join("blob"), content).unwrap();
            let oid = git_str(root, &["hash-object", "-w", "blob"]);
            std::fs::remove_file(root.join("blob")).unwrap();
            String::from_utf8(oid).unwrap().trim().to_owned()
        };
        let clean = blob(b"fn game() {}\n");
        let game = blob(b"MZ\x90\0binary");
        let conflict = |stages: [&str; 3]| {
            let info: String = stages
                .iter()
                .enumerate()
                .map(|(n, oid)| format!("100644 {oid} {}\tgame.rs\n", n + 1))
                .collect();
            std::fs::write(root.join("info"), info).unwrap();
            let out = super::git_command(root)
                .args(["update-index", "--index-info"])
                .stdin(std::fs::File::open(root.join("info")).unwrap())
                .output()
                .unwrap();
            std::fs::remove_file(root.join("info")).unwrap();
            assert!(out.status.success());
        };
        std::fs::write(root.join("game.rs"), b"fn game() {}\n").unwrap();

        for stage in 0..3 {
            let mut stages = [clean.as_str(); 3];
            stages[stage] = &game;
            conflict(stages);
            assert_eq!(names(root, false), ["game.rs"]);
            assert!(super::run(root, false).is_err(), "stage {}", stage + 1);

            // Gone from disk, the other stages are read as well.
            std::fs::rename(root.join("game.rs"), root.join("moved")).unwrap();
            assert!(
                super::run(root, false).is_err(),
                "stage {}, deleted",
                stage + 1
            );
            std::fs::rename(root.join("moved"), root.join("game.rs")).unwrap();
        }

        conflict([clean.as_str(); 3]);
        super::run(root, false).unwrap();
    }

    /// A submodule pulls in another repository's content, which no check
    /// here reads; it is refused whether staged or merely tracked.
    #[cfg(unix)]
    #[test]
    fn submodules_are_violations() {
        let tmp = tempdir();
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

    /// An untracked repository inside the checkout is listed as one
    /// directory; it is refused as what it is, not as a tracked directory.
    #[cfg(unix)]
    #[test]
    fn nested_repositories_are_violations() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::fs::create_dir(root.join("inner")).unwrap();
        git_str(&root.join("inner"), &["init", "-q"]);
        std::fs::write(root.join("inner/a.rs"), b"fn a() {}\n").unwrap();

        assert_eq!(names(root, false), ["inner/"]);
        let files = super::list_files(root, false).unwrap();
        let problem = super::read(root, &files[0]).unwrap_err();
        assert!(problem.contains("nested"), "{problem}");
        assert!(super::run(root, false).is_err());
    }

    /// Git's own settings can hide a submodule from `git diff`; the staged
    /// check must see it anyway.
    #[cfg(unix)]
    #[test]
    fn staged_submodules_are_seen_whatever_git_ignores() {
        let tmp = tempdir();
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
        git_str(root, &["config", "diff.ignoreSubmodules", "all"]);
        // Untracked, so it is not itself staged, yet `git diff` honours it.
        std::fs::write(
            root.join(".gitmodules"),
            "[submodule \"v\"]\n\tpath = vendor\n\turl = x\n\tignore = all\n",
        )
        .unwrap();
        let hidden = git_str(root, &["diff", "--cached", "--name-only"]);
        assert!(hidden.is_empty(), "git itself hides the submodule");

        assert_eq!(names(root, true), ["vendor"]);
        assert!(super::run(root, true).is_err());
    }

    /// Tests run git in temporary repositories; from a hook (`git commit -a`
    /// sets an absolute GIT_INDEX_FILE) every one of them used to write into
    /// that one outside index and fail.
    #[test]
    fn test_git_ignores_the_repository_environment() {
        let command = super::git_command(std::path::Path::new("."));
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key.to_str().unwrap().to_owned())
            .collect();
        for var in super::REPOSITORY_ENV {
            assert!(removed.iter().any(|r| r == var), "{var} is inherited");
        }
        // The list is git's own; a git that adds a variable fails here.
        let mut git = std::process::Command::new("git");
        super::unredirect_output(&mut git, std::env::vars_os().map(|(key, _)| key));
        let Some(out) = git
            .args(["rev-parse", "--local-env-vars"])
            .output()
            .ok()
            .filter(|o| o.status.success())
        else {
            eprintln!("skipped: no git");
            return;
        };
        for var in String::from_utf8(out.stdout).unwrap().lines() {
            assert!(super::REPOSITORY_ENV.contains(&var), "{var} is missing");
        }
    }

    /// A developer's global `core.excludesFile` used to hide the files the
    /// temporary repositories stage (`git add` refused them), and their
    /// `core.hooksPath` ran their hooks there. Only the repository's own
    /// configuration may reach those tests; the checkout keeps the rest.
    #[test]
    fn temporary_repositories_read_only_their_own_config() {
        let tmp = tempdir();
        let root = tmp.path();
        let out = super::git_command(root).args(["init", "-q"]).output();
        let Some(_) = out.ok().filter(|o| o.status.success()) else {
            eprintln!("skipped: no git");
            return;
        };
        let out = super::git_command(root)
            .args(["config", "--list", "--show-scope"])
            .output()
            .unwrap();
        assert!(out.status.success());
        for line in String::from_utf8(out.stdout).unwrap().lines() {
            assert!(
                line.starts_with("local\t") || line == "command\tsafe.directory=*",
                "outside config: {line}"
            );
        }

        let checkout = super::git_command(&crate::workspace_root());
        assert!(
            checkout
                .get_envs()
                .all(|(key, _)| key != "GIT_CONFIG_GLOBAL"),
            "the checkout's own check must keep the developer's excludes"
        );
    }

    /// Git reads `~/.config/git/ignore` and `~/.config/git/attributes` with
    /// no configuration naming them, so emptying the global config left them
    /// in force: a developer's `*.rs` there made `git add` refuse every file
    /// the temporary repositories stage, and a `filter=` would rewrite them.
    #[cfg(unix)]
    #[test]
    fn temporary_repositories_read_no_default_ignore_or_attributes() {
        let home = tempdir();
        let config = home.path().join(".config/git");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(config.join("ignore"), "*.rs\n").unwrap();
        std::fs::write(config.join("attributes"), "*.rs filter=lfs\n").unwrap();
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let probe = |command: &mut std::process::Command, args: &[&str]| {
            let out = command
                .args(args)
                .env("HOME", home.path())
                .output()
                .unwrap();
            String::from_utf8(out.stdout).unwrap()
        };

        // Plain git in the same place, as the tests ran before: the files
        // under that HOME apply.
        let mut plain = std::process::Command::new("git");
        plain.current_dir(root).env_remove("XDG_CONFIG_HOME");
        super::unredirect_output(&mut plain, std::env::vars_os().map(|(key, _)| key));
        for var in super::REPOSITORY_ENV.iter().chain(super::PATHSPEC_ENV) {
            plain.env_remove(var);
        }
        plain
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .args(["-c", "safe.directory=*"]);
        assert_eq!(probe(&mut plain, &["check-ignore", "a.rs"]), "a.rs\n");

        let ignored = probe(&mut super::git_command(root), &["check-ignore", "a.rs"]);
        assert_eq!(ignored, "", "the default ignore file is read");
        let filter = probe(
            &mut super::git_command(root),
            &["check-attr", "filter", "--", "a.rs"],
        );
        assert_eq!(filter, "a.rs: filter: unspecified\n");

        // The checkout keeps the developer's own files.
        let checkout = super::git_command(&crate::workspace_root());
        assert!(
            checkout
                .get_envs()
                .all(|(key, _)| key != "XDG_CONFIG_HOME" && key != "GIT_ATTR_NOSYSTEM"),
        );
    }

    /// `GIT_TEMPLATE_DIR` and `GIT_DEFAULT_HASH` reach `git init` from the
    /// environment: a template's `info/exclude` hid the files the temporary
    /// repositories stage, its hooks ran in them (`--no-verify` skips only
    /// some), and a SHA-256 repository refused the SHA-1 ids tests stage.
    #[cfg(unix)]
    #[test]
    fn temporary_repositories_are_made_without_template_or_hash_choice() {
        let template = tempdir();
        std::fs::create_dir_all(template.path().join("info")).unwrap();
        std::fs::write(template.path().join("info/exclude"), "*.rs\n").unwrap();
        let tmp = tempdir();
        let root = tmp.path();
        let made = super::git_command(root)
            .args(["init", "-q"])
            .env("GIT_TEMPLATE_DIR", template.path())
            .env("GIT_DEFAULT_HASH", "sha256")
            .output()
            .unwrap();
        assert!(made.status.success());
        // Handed both choices, as the environment used to: both apply.
        let format = git_str(root, &["rev-parse", "--show-object-format"]);
        assert_eq!(format, b"sha256\n");
        assert!(root.join(".git/info/exclude").exists());

        let command = super::git_command(root);
        let envs: Vec<_> = command.get_envs().collect();
        assert!(envs.contains(&("GIT_TEMPLATE_DIR".as_ref(), Some("".as_ref()))));
        assert!(envs.contains(&("GIT_DEFAULT_HASH".as_ref(), None)));

        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let format = git_str(root, &["rev-parse", "--show-object-format"]);
        assert_eq!(format, b"sha1\n");
        assert!(!root.join(".git/info/exclude").exists());
        assert!(!root.join(".git/hooks").exists());

        // The checkout is never initialised here and keeps its environment.
        let checkout = super::git_command(&crate::workspace_root());
        assert!(
            checkout
                .get_envs()
                .all(|(key, _)| key != "GIT_TEMPLATE_DIR" && key != "GIT_DEFAULT_HASH"),
        );
    }

    /// Under `GIT_ATTR_SOURCE=HEAD` every git command that reads attributes
    /// (`add`, `check-attr`) failed in a fresh temporary repository; under a
    /// `GIT_AUTHOR_DATE` git cannot parse, or an empty `GIT_AUTHOR_NAME`,
    /// every test that commits did; and under `GIT_LITERAL_PATHSPECS`
    /// `check-ignore` refused to run.
    #[cfg(unix)]
    #[test]
    fn temporary_repositories_ignore_attr_source_identity_and_pathspec_env() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);

        // Handed those variables, as the environment used to: git fails.
        let refused = |var: &str, value: &str, args: &[&str]| {
            let out = super::git_command(root)
                .env(var, value)
                .args(args)
                .output()
                .unwrap();
            assert!(!out.status.success(), "{var}={value:?} git {args:?}");
        };
        refused(
            "GIT_ATTR_SOURCE",
            "HEAD",
            &["check-attr", "-a", "--", "a.rs"],
        );
        refused(
            "GIT_LITERAL_PATHSPECS",
            "1",
            &["check-ignore", "-q", "--", "a.rs"],
        );
        let empty_commit = [
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "c",
        ];
        let mut commit = super::git_command(root);
        commit.envs(IDENTITY).env("GIT_AUTHOR_DATE", "garbage");
        assert!(!commit.args(empty_commit).status().unwrap().success());

        let command = super::git_command(root);
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key.to_str().unwrap().to_owned())
            .collect();
        for var in super::PATHSPEC_ENV.iter().chain(&["GIT_ATTR_SOURCE"]) {
            assert!(removed.iter().any(|r| r == var), "{var} is inherited");
        }
        // The helper's identity is its own, whatever the environment holds.
        git_str(root, &["commit", "-q", "--allow-empty", "-m", "c"]);
        let author = git_str(root, &["log", "-1", "--format=%an <%ae> %at|%cn <%ce> %ct"]);
        assert_eq!(author, b"t <t@t> 0|t <t@t> 0\n");

        // Pathspec variables change no full check (it passes none); the
        // attribute source is the checkout's own business.
        let checkout = super::git_command(&crate::workspace_root());
        assert!(checkout.get_envs().all(|(key, _)| key != "GIT_ATTR_SOURCE"));
    }

    /// A `pre-receive` hook's `GIT_QUARANTINE_PATH` made every temporary
    /// repository's `git init` fail ("ref updates forbidden inside
    /// quarantine environment"), and a TMPDIR owned by another user made
    /// git refuse each one as a dubious repository, with no global config
    /// to hold the `safe.directory` that would let it in. Git's own
    /// `GIT_TEST_ASSUME_DIFFERENT_OWNER` stands in for that owner.
    #[cfg(unix)]
    #[test]
    fn temporary_repositories_ignore_quarantine_and_ownership() {
        let tmp = tempdir();
        let root = tmp.path();
        let control = tempdir();

        // Plain git, as the tests used to run it: refused.
        let init = super::git_command(control.path())
            .env("GIT_QUARANTINE_PATH", root)
            .args(["init", "-q"])
            .output()
            .unwrap();
        assert!(!init.status.success(), "quarantine allowed git init");
        git_str(control.path(), &["init", "-q"]);
        // As test git, less its trust (and with no config that could hold one).
        let mut plain = std::process::Command::new("git");
        for var in super::REPOSITORY_ENV {
            plain.env_remove(var);
        }
        let dubious = plain
            .current_dir(control.path())
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
            .args(["rev-parse", "--git-dir"])
            .output()
            .unwrap();
        assert!(!dubious.status.success(), "ownership was not checked");

        // Test git drops the quarantine, and trusts a repository another
        // user owns, so the check runs.
        let command = super::git_command(root);
        assert!(
            command
                .get_envs()
                .any(|(key, value)| key == "GIT_QUARANTINE_PATH" && value.is_none()),
            "GIT_QUARANTINE_PATH is inherited"
        );
        git_str(root, &["init", "-q"]);
        std::fs::write(root.join("game.bin"), b"MZ\x90\0").unwrap();
        let add = super::git_command(root)
            .env("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
            .args(["add", "game.bin"])
            .output()
            .unwrap();
        assert!(add.status.success(), "ownership refused the test");
        assert_eq!(names(root, true), ["game.bin"]);

        // The checkout keeps both checks.
        let checkout = super::git_command(&crate::workspace_root());
        assert!(
            checkout
                .get_envs()
                .all(|(key, _)| key != "GIT_QUARANTINE_PATH")
        );
        assert!(checkout.get_args().all(|arg| arg != "safe.directory=*"));
    }

    /// Trace output pointed at stdout used to be parsed as git's output: the
    /// first file listed was taken for a name that does not exist and went
    /// unread, and `--staged` failed on the record. The variables are
    /// dropped (trace2's set off) for every git the check runs.
    #[cfg(unix)]
    #[test]
    fn trace_output_is_not_read_as_git_output() {
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        std::fs::write(root.join("a.bin"), b"MZ\x90\0").unwrap();
        git_str(root, &["add", "a.bin"]);
        const TRACES: [&str; 4] = [
            "GIT_TRACE",
            "GIT_TRACE_SETUP",
            "GIT_TRACE_PERFORMANCE",
            "GIT_TRACE2_EVENT",
        ];
        let ls_files = |unredirect: bool| {
            let mut command = super::git_command(root);
            for var in TRACES {
                command.env(var, "/dev/stdout");
            }
            command.env("GIT_TRACE2", "/dev/stdout");
            if unredirect {
                let keys = TRACES.iter().chain(&["GIT_TRACE2", "GIT_REDIRECT_STDOUT"]);
                super::unredirect_output(&mut command, keys.map(std::ffi::OsString::from));
            }
            command.args(["ls-files", "-z"]).output().unwrap()
        };

        // As git was run before: trace lines lead the first name.
        let traced = ls_files(false);
        assert!(traced.status.success());
        assert_ne!(traced.stdout, b"a.bin\0", "git wrote no trace to stdout");
        // As it is run now: only the names.
        let plain = ls_files(true);
        assert!(plain.status.success());
        assert_eq!(plain.stdout, b"a.bin\0");

        // And whatever this process inherited, the check reads the file.
        assert_eq!(names(root, false), ["a.bin"]);
        assert!(super::run(root, false).is_err());
        assert!(super::run(root, true).is_err());
    }

    /// A temporary directory that is not a repository is walked. Made beside
    /// the test binary, it used to be taken for an ignored part of the
    /// checkout: `git ls-files` there succeeded, listed nothing, and the
    /// check passed a directory it never read.
    #[cfg(unix)]
    #[test]
    fn directories_outside_any_repository_are_walked() {
        let outer = tempdir();
        git_str(outer.path(), &["init", "-q"]);
        std::fs::write(outer.path().join(".gitignore"), "plain/\n").unwrap();
        let root = outer.path().join("plain");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("src/a.rs"), b"fn a() {}\n").unwrap();
        std::fs::write(root.join("target/game.bin"), b"MZ\x90\0build output").unwrap();
        std::fs::write(root.join("game.bin"), b"MZ\x90\0a game executable").unwrap();

        // Without the ceiling, as before: git finds the enclosing repository,
        // where the directory is ignored and lists as empty.
        let found = super::git_command(&root)
            .env_remove("GIT_CEILING_DIRECTORIES")
            .args([
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ])
            .output()
            .unwrap();
        assert!(found.status.success());
        assert!(found.stdout.is_empty());

        let outside = super::git_command(&root)
            .args(["rev-parse", "--git-dir"])
            .output()
            .unwrap();
        assert!(!outside.status.success(), "a repository above was found");
        // The walk skips build output and lists in no fixed order.
        let mut walked = names(&root, false);
        walked.sort();
        assert_eq!(walked, ["game.bin", "src/a.rs"]);
        assert!(super::run(&root, false).is_err());
        std::fs::remove_file(root.join("game.bin")).unwrap();
        super::run(&root, false).unwrap();

        // The checkout keeps git's own discovery.
        let checkout = super::git_command(&crate::workspace_root());
        assert!(
            checkout
                .get_envs()
                .all(|(key, _)| key != "GIT_CEILING_DIRECTORIES")
        );
    }

    const LFS_POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1\n\
        oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n\
        size 12345\n";

    /// A Git LFS pointer is small, allowlisted text, but the file it stands
    /// for goes to LFS storage, which no check reads.
    #[test]
    fn rejects_lfs_pointers() {
        assert!(!check("assets/model.txt", LFS_POINTER).is_empty());
        let legacy = b"version https://hawser.github.com/spec/v1\noid sha256:00\nsize 1\n";
        assert!(!check("data.json", legacy).is_empty());
        assert!(
            check("notes.md", b"See https://git-lfs.github.com/spec/v1.\n").is_empty(),
            "only a pointer, not a mention"
        );
    }

    /// With `filter=lfs`, the index holds the pointer while the working tree
    /// holds the real file: the staged check reads only the pointer, and a
    /// checkout without git-lfs sees only the pointer too. Both checks must
    /// refuse it. The clean filter here stands in for git-lfs, which the
    /// test does not need installed.
    #[cfg(unix)]
    #[test]
    fn lfs_tracked_files_are_violations() {
        let outside = tempdir();
        let pointer = outside.path().join("pointer");
        std::fs::write(&pointer, LFS_POINTER).unwrap();
        let tmp = tempdir();
        let root = tmp.path();
        git_str(root, &["init", "-q"]);
        let clean = format!("cat '{}'", pointer.display());
        git_str(root, &["config", "filter.lfs.clean", &clean]);
        git_str(root, &["config", "filter.lfs.smudge", "cat"]);
        git_str(root, &["config", "filter.lfs.required", "true"]);
        std::fs::write(root.join(".gitattributes"), "*.txt filter=lfs -text\n").unwrap();
        std::fs::write(root.join("strings.txt"), "plain text on disk\n").unwrap();
        git_str(root, &["add", ".gitattributes", "strings.txt"]);
        let staged = git_str(root, &["cat-file", "blob", ":strings.txt"]);
        assert_eq!(staged, LFS_POINTER, "the index holds the pointer");
        assert!(super::run(root, true).is_err());

        // A checkout without the filter: the pointer is the file on disk.
        std::fs::write(root.join("strings.txt"), LFS_POINTER).unwrap();
        assert!(super::run(root, false).is_err());
    }

    #[test]
    fn rejects_data_smuggled_into_text() {
        let blob = format!("const DATA: &str = \"{}\";", "A1".repeat(400));
        assert!(!check("src/blob.rs", blob.as_bytes()).is_empty());
    }

    /// `base64` output wrapped at 76 columns, PEM-style 64-column lines and
    /// `xxd -p` hex at 60, with LF or CRLF: each line is short, the run is not.
    #[test]
    fn rejects_encoded_data_wrapped_across_lines() {
        for (width, alphabet, eol) in [
            (
                76,
                &b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[..],
                "\n",
            ),
            (
                64,
                &b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[..],
                "\r\n",
            ),
            (60, &b"0123456789abcdef"[..], "\n"),
        ] {
            let line: String = (0..width)
                .map(|i| alphabet[i * 7 % alphabet.len()] as char)
                .collect();
            let lines = MAX_ENCODED_RUN / width + 1;
            let blob = format!("{line}{eol}").repeat(lines) + "QUJD==" + eol;
            assert_eq!(
                super::longest_encoded_run(blob.as_bytes()),
                lines * width + 6,
                "width {width}, eol {eol:?}"
            );
            assert!(
                check("strings.txt", blob.as_bytes())
                    .iter()
                    .any(|p| p.contains("hex/base64 run")),
                "width {width}, eol {eol:?}"
            );
        }
    }

    /// base64url (`-` and `_` for `+` and `/`) is as much a payload as base64,
    /// wrapped or not; the two alphabets are not mixed into one run.
    #[test]
    fn rejects_base64url_data() {
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let encoded: String = (0..MAX_ENCODED_RUN * 2)
            .map(|i| alphabet[i * 7 % alphabet.len()] as char)
            .collect();
        assert!(
            encoded.contains('-') && encoded.contains('_'),
            "the test data uses the URL-safe characters"
        );
        assert!(
            super::longest_run(encoded.as_bytes(), b"+/=") < 64,
            "the standard alphabet alone sees only short runs"
        );
        for blob in [
            format!("const TOKEN: &str = \"{encoded}\";\n"),
            encoded
                .as_bytes()
                .chunks(76)
                .map(|line| String::from_utf8_lossy(line) + "\n")
                .collect(),
        ] {
            assert!(
                check("strings.txt", blob.as_bytes())
                    .iter()
                    .any(|p| p.contains("hex/base64 run")),
                "{blob:?}"
            );
        }
        let run = super::longest_encoded_run;
        assert_eq!(run(b"ab-cd_ef"), 8);
        assert_eq!(run(b"ab+cd/ef"), 8);
        assert_eq!(run(b"ab+cd-ef"), 5, "the alphabets do not mix");
        assert_eq!(run(b"ab-cd\nef_gh"), 10, "wrapped like base64");
    }

    /// Only a single line break joins: prose, code, a blank line or
    /// indentation between payload characters still ends the run.
    #[test]
    fn line_breaks_join_only_payload_lines() {
        let run = super::longest_encoded_run;
        assert_eq!(run(b"ab\ncd"), 4);
        assert_eq!(run(b"ab\r\ncd"), 4);
        assert_eq!(run(b"ab\n\ncd"), 2, "a blank line ends the run");
        assert_eq!(run(b"ab\n cd"), 2, "indentation ends the run");
        assert_eq!(run(b"ab\r\rcd"), 2);
        assert_eq!(
            run(b"\nab\n"),
            2,
            "leading and trailing breaks count nothing"
        );
        assert_eq!(run(b""), 0);
        let paragraph = "The quick brown fox\njumps over the lazy dog\n".repeat(200);
        assert!(run(paragraph.as_bytes()) < 20);
    }

    /// ascii85, Z85 and uuencode alphabets, unwrapped and wrapped: punctuation
    /// breaks every hex/base64 run, the data is there all the same.
    #[test]
    fn rejects_ascii85_z85_and_uuencoded_data() {
        let ascii85: Vec<u8> = (b'!'..=b'u').collect();
        let uuencode: Vec<u8> = (b'!'..=b'`').collect();
        let z85 = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
        for (what, alphabet, width) in [
            ("ascii85", &ascii85[..], 76),
            ("Z85", &z85[..], 76),
            ("uuencode", &uuencode[..], 61),
        ] {
            let encoded: String = (0..MAX_ENCODED_RUN * 2)
                .map(|i| alphabet[i * 7 % alphabet.len()] as char)
                .collect();
            assert!(
                super::longest_encoded_run(encoded.as_bytes()) < 32,
                "{what}: the hex/base64 check alone sees only short runs"
            );
            let wrapped: String = encoded
                .as_bytes()
                .chunks(width)
                .map(|line| String::from_utf8_lossy(line) + "\n")
                .collect();
            for blob in [
                encoded.clone(),
                wrapped.clone(),
                wrapped.replace('\n', "\r\n"),
            ] {
                assert_eq!(
                    super::longest_printable_run(blob.as_bytes()),
                    encoded.len(),
                    "{what}"
                );
                assert!(
                    check("strings.txt", blob.as_bytes())
                        .iter()
                        .any(|p| p.contains("without whitespace")),
                    "{what}: {blob:?}"
                );
            }
        }
    }

    /// Whitespace ends a printable run, and a line break continues one only
    /// after a line as long as an encoder writes.
    #[test]
    fn printable_runs_join_only_long_lines() {
        let run = super::longest_printable_run;
        assert_eq!(run(b"a+b c"), 3);
        assert_eq!(run(b"ab\n\ncd"), 2);
        let long = "x!".repeat(super::MIN_WRAPPED_LINE / 2);
        assert_eq!(run(format!("{long}\n{long}").as_bytes()), long.len() * 2);
        assert_eq!(run(format!("{long}\n {long}").as_bytes()), long.len());
        assert_eq!(run(format!("x!\n{long}").as_bytes()), long.len());
        let gitignore = "/target/\n*.exe\n*.dll\n**/*.bin\n".repeat(100);
        assert!(check(".gitignore", gitignore.as_bytes()).is_empty());
        let url = format!("<https://example.com/{}?a=1&b=2>\n", "p/".repeat(150));
        assert!(check("README.md", url.as_bytes()).is_empty());
    }

    /// Binary data with every NUL removed, by COBS byte stuffing here, is
    /// still binary: it is not UTF-8, or holds control characters.
    #[test]
    fn rejects_binary_without_nul_bytes() {
        fn cobs(data: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            for block in data.split(|&b| b == 0) {
                for chunk in block.chunks(254) {
                    out.push(chunk.len() as u8 + 1);
                    out.extend_from_slice(chunk);
                }
                if block.is_empty() {
                    out.push(1);
                }
            }
            out
        }
        let mut state = 0x9e37_79b9u32;
        let bytes: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect();
        let stuffed = cobs(&bytes);
        assert!(!stuffed.contains(&0));
        assert!(super::longest_encoded_run(&stuffed) <= MAX_ENCODED_RUN);
        assert!(super::longest_printable_run(&stuffed) <= MAX_ENCODED_RUN);
        let problems = check("strings.txt", &stuffed);
        assert!(
            problems.iter().any(|p| p.contains("not UTF-8")),
            "{problems:?}"
        );
        // Bytes under 0x80 only (7-bit stuffing) are UTF-8, but not text.
        let seven: Vec<u8> = stuffed.iter().map(|b| (b & 0x7f).max(1)).collect();
        let problems = check("strings.txt", &seven);
        assert!(
            problems.iter().any(|p| p.contains("control character")),
            "{problems:?}"
        );
        for ok in [&b"a\tb\r\n\x0cc\n"[..], "naïve café — → ✓\n".as_bytes()] {
            assert!(check("README.md", ok).is_empty(), "{ok:?}");
        }
    }

    /// base32768 and base65536 write 15 or 16 bits per letter, CJK and the
    /// like, without whitespace: a run of them is a printable run too.
    #[test]
    fn rejects_base65536_like_data() {
        let encoded: String = (0..3000u32)
            .map(|i| char::from_u32(0x4e00 + (i * 7919) % 0x5000).unwrap())
            .collect();
        assert_eq!(super::longest_encoded_run(encoded.as_bytes()), 0);
        for blob in [
            format!("const DATA: &str = \"{encoded}\";\n"),
            encoded
                .chars()
                .collect::<Vec<_>>()
                .chunks(64)
                .map(|line| line.iter().collect::<String>() + "\n")
                .collect(),
        ] {
            assert!(
                check("src/data.rs", blob.as_bytes())
                    .iter()
                    .any(|p| p.contains("without whitespace")),
                "{blob:.80}"
            );
        }
        // Prose in any script, with its spaces and line breaks, passes.
        let prose = "日本語の文章。短い行です。\n".repeat(200);
        assert!(check("docs/notes.md", prose.as_bytes()).is_empty());
    }

    /// `xxd -i` output, a Rust byte array and a byte-escape string: every
    /// token is short, but the file is an executable all the same.
    #[test]
    fn rejects_data_as_numeric_arrays_and_escapes() {
        let bytes: Vec<u8> = (0..MAX_LITERAL_RUN * 2).map(|i| (i * 37) as u8).collect();
        let xxd: String = bytes
            .chunks(12)
            .map(|row| {
                let row: Vec<String> = row.iter().map(|b| format!("0x{b:02x}")).collect();
                format!("  {},\n", row.join(", "))
            })
            .collect();
        let decimal: Vec<String> = bytes.iter().map(|b| format!("{b}u8")).collect();
        let escaped: String = bytes.iter().map(|b| format!("\\x{b:02x}")).collect();
        for blob in [
            format!("unsigned char data[] = {{\n{xxd}}};\n"),
            format!(
                "const DATA: [u8; {}] = [{}];\n",
                bytes.len(),
                decimal.join(",")
            ),
            format!("const DATA: &[u8] = b\"{escaped}\";\n"),
        ] {
            assert_eq!(
                super::longest_literal_run(blob.as_bytes()),
                bytes.len(),
                "{blob:.80}"
            );
            assert!(
                check("src/data.rs", blob.as_bytes())
                    .iter()
                    .any(|p| p.contains("numeric array")),
                "{blob:.80}"
            );
        }
    }

    /// The same escapes split into many short strings, each well under the
    /// limit: a Rust string continuation, C/HLSL adjacent literals,
    /// `concat!` arguments, an array of byte strings, and `\u{..}` escapes.
    #[test]
    fn rejects_escape_strings_split_across_literals() {
        let bytes: Vec<u8> = (0..MAX_LITERAL_RUN * 2).map(|i| (i * 37) as u8).collect();
        let rows = |escape: &dyn Fn(u8) -> String, join: &str| -> String {
            bytes
                .chunks(64)
                .map(|row| row.iter().map(|&b| escape(b)).collect::<String>())
                .collect::<Vec<_>>()
                .join(join)
        };
        let x = |b: u8| format!("\\x{b:02x}");
        let u = |b: u8| format!("\\u{{{b:x}}}");
        for blob in [
            format!("const DATA: &[u8] = b\"{}\";\n", rows(&x, "\\\n    ")),
            format!("const DATA: &[u8] = b\"{}\";\n", rows(&x, "\\\r\n\t")),
            format!("const char data[] =\n    \"{}\";\n", rows(&x, "\"\n    \"")),
            format!("const DATA: &str = concat!(\"{}\");\n", rows(&x, "\", \"")),
            format!(
                "const DATA: &[&[u8]] = &[\n    b\"{}\",\n];\n",
                rows(&x, "\",\n    b\"")
            ),
            format!("const DATA: &str = \"{}\";\n", rows(&u, "\\\n    ")),
        ] {
            assert_eq!(
                super::longest_literal_run(blob.as_bytes()),
                bytes.len(),
                "{blob:.80}"
            );
            assert!(
                check("src/data.rs", blob.as_bytes())
                    .iter()
                    .any(|p| p.contains("byte-escape string")),
                "{blob:.80}"
            );
        }
    }

    /// Strings of escapes join only when nothing but a string boundary lies
    /// between them.
    #[test]
    fn escape_runs_join_only_across_string_boundaries() {
        let run = super::longest_literal_run;
        assert_eq!(
            run(br#""\x1b[0m" "\x1b[1m""#),
            1,
            "text between escapes ends a run"
        );
        assert_eq!(run(br#"f("\x00", 1, "\x01")"#), 1, "an argument ends a run");
        assert_eq!(run(br#""\x00\x01", // two"#), 2);
        assert_eq!(run(br#""\x00\x01", "\x02""#), 3);
        assert_eq!(run(b"\"\\x00\\\n   \\x01\""), 2);
        assert_eq!(
            run(br"\x00, \x01"),
            1,
            "a comma alone is text inside one string"
        );
        assert_eq!(
            run(br#""\u{1F600}\u{1}\u{}\u{1234567}""#),
            2,
            "one to six digits"
        );
        assert_eq!(run(br#""caf\u{e9}", "na\u{ef}ve""#), 1);
    }

    /// Tables split into rows, numbers inside identifiers, prose and the odd
    /// escape stay short.
    #[test]
    fn literal_runs_end_at_anything_but_commas() {
        let run = super::longest_literal_run;
        assert_eq!(run(b"[1, 2,\n 3,\r\n\t-4, 0x5_u8, 6.5]"), 6);
        assert_eq!(run(b"[[1, 2], [3, 4]]"), 2, "rows count one by one");
        assert_eq!(run(b"1, 2, // three\n4, 5"), 2, "a comment ends the list");
        assert_eq!(
            run(b"f(x86_64, u8, 2)"),
            1,
            "digits inside identifiers are not literals"
        );
        assert_eq!(run(b"a - 1, b"), 1);
        assert_eq!(run(br"\x4d\x5a\x90\x00 \x01"), 4);
        assert_eq!(run(br"\x4\xzz\\x"), 0);
        assert_eq!(run(b""), 0);
        let table: String = (0..64)
            .map(|r| {
                format!(
                    "    [{}],\n",
                    (0..16)
                        .map(|c| (r * c % 7).to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();
        assert_eq!(run(table.as_bytes()), 16);
        assert!(check("src/table.rs", table.as_bytes()).is_empty());
    }
}
