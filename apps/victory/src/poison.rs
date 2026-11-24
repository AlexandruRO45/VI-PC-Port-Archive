//! `victory poison`: which modules this build's driver compiler faults on.
//!
//! A create sometimes ends the process inside `amdxc64.dll+0x85fd0f`, and which
//! module caused it decides whether it happens: one dump line naming
//! `compute:a1266cc7391ae0d6` faults on the first create of a fresh process,
//! while thousands of other modules create in the same process without trouble.
//! Since a fault cannot be caught in-process, each module is created here in a
//! **child** — this program never creates anything — and the ones whose child
//! died are written to the list the runtime consults before it creates.
//!
//! The list is written for this build and this adapter: an entry is a statement
//! about the DXIL we emit at one commit, so another build's list is ignored.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

/// One dump line per module, and one child per module.
pub fn run(dump: PathBuf, shaders: Option<PathBuf>, out: Option<PathBuf>, modules: Option<PathBuf>) -> Result<()> {
    let shaders = shaders
        .or_else(|| dump.parent().map(|dir| dir.to_path_buf()))
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("."));
    let text = std::fs::read_to_string(&dump).with_context(|| format!("reading {}", dump.display()))?;
    // The first line that names each module: a module is what faults, and one
    // line is all a child needs to ask the compiler for it.
    let mut first_line: BTreeMap<(String, String), &str> = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(stages) = line.split_once("stages=").map(|(_, rest)| rest) else { continue };
        let stages = stages.split(' ').next().unwrap_or_default();
        for pair in stages.split(',') {
            let Some((stage, hash)) = pair.split_once(':') else { continue };
            first_line.entry((stage.to_owned(), hash.to_owned())).or_insert(line);
        }
    }
    let wanted: Option<std::collections::HashSet<(String, String)>> = match &modules {
        Some(path) => {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            Some(
                text.lines()
                    .filter_map(|line| line.split_once(char::is_whitespace))
                    .map(|(stage, hash)| (stage.trim().to_owned(), hash.trim().to_owned()))
                    .collect(),
            )
        }
        None => None,
    };
    let out = out.unwrap_or_else(|| shaders.join("poisoned-modules.txt"));
    let exe = std::env::current_exe().context("current exe")?;
    let key = victory_gpu::pipeline::poison_key().unwrap_or_else(|| "unknown".to_owned());
    // A list is a statement about one build's DXIL, and the runtime obeys only
    // the list whose `build` matches its own bytes. Writing this build's list
    // over another's would leave a boot that reads a list and refuses nothing,
    // so the older list is kept and the run stops here instead.
    if let Ok(old) = std::fs::read_to_string(&out)
        && let Some(old) = old.lines().find_map(|line| line.trim().strip_prefix("# build="))
        && old.trim() != key
    {
        anyhow::bail!(
            "{} holds a list written for build {} and this is {}: remove it, or write this one \
             with --out to another path",
            out.display(),
            old.trim(),
            key
        );
    }
    let mut probe = std::fs::File::create(&out).with_context(|| format!("writing {}", out.display()))?;
    use std::io::Write;
    writeln!(probe, "# build={key}")?;
    writeln!(
        probe,
        "# adapter={}",
        if std::env::var_os("VICTORY_D3D12_WARP").is_some() { "warp" } else { "card" }
    )?;
    let (mut probed, mut poisoned) = (0usize, 0usize);
    for ((stage, hash), line) in &first_line {
        if let Some(wanted) = &wanted
            && !wanted.contains(&(stage.clone(), hash.clone()))
        {
            continue;
        }
        probed += 1;
        let tag = format!("{stage}-{hash}");
        let dir = std::env::temp_dir().join(format!("victory-poison-{tag}"));
        let _ = std::fs::create_dir_all(&dir);
        let line_file = dir.join("line.txt");
        std::fs::write(&line_file, format!("{line}\n"))?;
        // A fresh cache per child: a pipeline the library already holds is never
        // created, and would read as a module that does not fault.
        let child_cache = dir.join("cache");
        let _ = std::fs::remove_dir_all(&child_cache);
        std::fs::create_dir_all(&child_cache)?;
        let status = std::process::Command::new(&exe)
            .arg("precompile")
            .arg(&line_file)
            .arg("--shaders")
            .arg(&shaders)
            .arg("--cache")
            .arg(&child_cache)
            .arg("--jobs")
            .arg("1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .with_context(|| format!("probing {stage}:{hash}"))?;
        if !status.success() {
            poisoned += 1;
            writeln!(probe, "{stage} {hash}")?;
            probe.flush()?;
        }
        if probed % 25 == 0 {
            probe.flush()?;
            println!("probed {probed}, poisoned {poisoned}");
        }
    }
    probe.flush()?;
    println!("probed {probed}, poisoned {poisoned} of {}", first_line.len());
    println!("written {}", out.display());
    Ok(())
}
