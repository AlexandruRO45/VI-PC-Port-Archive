//! The pipeline pre-build, the warm-up's second half: every pipeline of the
//! title's shader pack made ahead of a run into the host pipeline library
//! the run reads (`<cache>/pipelines`), so that a first run, and a first
//! visit to any place, compiles nothing the pack lists.
//!
//! Each pack record becomes the creation stream the engine would hand the
//! device for that pipeline (`victory_game::pipelines`), which goes through
//! the run's own code (`victory_gpu::pipeline::Prebuild`): a pipeline is
//! only ever used if its host description is the run's, byte for byte.
//! Records whose states cannot be named, or whose shaders are refused, are
//! counted by cause and skipped. The library is saved as it grows: an
//! interrupted pre-build keeps its work, and the next one finds those
//! pipelines there.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use victory_game::pipelines::{self, Named, Unnamed};
use victory_game::{sgd5, shaderpack};
use victory_gpu::pipeline::{self, Prebuild};
use victory_shader::{Container, FourCC, HwStage, native_programs};

use crate::shader::{MAX_FILE, Source, sources};

/// Where a native program's container is: file, source, and its stage.
type Found = (usize, usize, HwStage);

/// One pipeline to make: a graphics or compute record of the pack.
#[derive(Clone, Copy, Debug)]
enum Job {
    Graphics(usize),
    Compute(usize),
}

pub fn run(paths: &[PathBuf], threads: usize) -> Result<()> {
    let started = Instant::now();
    pipeline::use_pipeline_cache(victory_core::Config::load()?.cache_subdir("pipelines")?);
    let inputs = collect_inputs(paths);
    let Some((pack_path, pack)) = find_pack(&inputs) else {
        println!("pipelines    no shader pack among the files: nothing to pre-build");
        return Ok(());
    };
    let pack = shaderpack::Pack::parse(&pack).map_err(anyhow::Error::msg)?;
    let graphics: Vec<shaderpack::Graphics> = pack.graphics().collect();
    let compute: Vec<shaderpack::Compute> = pack.compute().collect();
    let files = &inputs.shader_files;

    // Every container by the hash instructions its native program starts
    // with, which the pack's programs start with too.
    let sources: Vec<Vec<Source>> = files.iter().map(|(_, d)| sources(d)).collect();
    let programs: Mutex<HashMap<[u8; 24], Found>> = Mutex::new(HashMap::new());
    for (f, list) in sources.iter().enumerate() {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads.min(list.len()) {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(source) = list.get(i) else { break };
                        let Some(bytes) = source.bytes() else {
                            continue;
                        };
                        for (id, stage) in program_ids(&bytes) {
                            lock(&programs).entry(id).or_insert((f, i, stage));
                        }
                    }
                });
            }
        });
    }
    let programs = programs.into_inner().unwrap_or_else(|e| e.into_inner());
    let databases: Vec<sgd5::Database> =
        files.iter().flat_map(|(_, d)| sgd5::databases(d)).collect();
    let named = Named::find(&pack, &databases);
    eprintln!(
        "{}: shader pack, {} graphics and {} compute pipelines; states named: {} depth-stencil, {} rasteriser, {} blend",
        pack_path.display(),
        graphics.len(),
        compute.len(),
        named.depth.len(),
        named.raster.len(),
        named.blend.len(),
    );

    let container = |code: &[u8]| -> Option<(Vec<u8>, HwStage)> {
        let &(f, i, stage) = programs.get(&shaderpack::program_id(code)?)?;
        let bytes = sources[f][i].bytes()?;
        // As the title passes it: the container's own size.
        let size = u32::from_le_bytes(bytes.get(0x18..0x1c)?.try_into().ok()?) as usize;
        Some((bytes.get(..size)?.to_vec(), stage))
    };
    // A job's pipeline, or why it has none.
    let prepare = |job: Job| -> Result<Prebuild, String> {
        let g = match job {
            Job::Graphics(i) => &graphics[i],
            Job::Compute(i) => {
                let (cs, _) = container(compute[i].stage.program).ok_or("shader not found")?;
                return Ok(Prebuild::compute(&cs));
            }
        };
        let [ps, vs, gs, ..] = g.stages;
        if g.stages[3..].iter().any(Option::is_some) {
            return Err("stages other than vertex, mesh and pixel".into());
        }
        let (first, stage) = vs
            .or(gs)
            .and_then(|s| container(s.program))
            .ok_or("shader not found")?;
        let pixel = match ps {
            Some(ps) => Some(container(ps.program).ok_or("shader not found")?.0),
            None => None,
        };
        let mesh = match stage {
            HwStage::Ms => true,
            HwStage::Vs => false,
            _ => return Err("first stage not a vertex or mesh shader".into()),
        };
        let stream = pipelines::stream(g, &named, &first, mesh, pixel.as_deref()).map_err(|u| {
            match u {
                Unnamed::DepthStencil => "depth-stencil state not named",
                Unnamed::Rasterizer => "rasteriser state not named",
                Unnamed::Blend => "blend state not named",
                Unnamed::Metadata => "metadata not read",
            }
            .to_owned()
        })?;
        // SAFETY: the stream points at `first`, `pixel` and its own input
        // elements, all alive here.
        Ok(unsafe { Prebuild::graphics(stream.bytes()) })
    };

    // A job's shaders as a run names them (diagnostics).
    let shaders_of = |job: Job| {
        let stages: Vec<(&str, &[u8])> = match job {
            Job::Compute(i) => vec![("compute", compute[i].stage.program)],
            Job::Graphics(i) => {
                let [ps, vs, gs, ..] = graphics[i].stages;
                let first = vs
                    .map(|s| ("vertex", s.program))
                    .or(gs.map(|s| ("mesh", s.program)));
                first
                    .into_iter()
                    .chain(ps.map(|s| ("pixel", s.program)))
                    .collect()
            }
        };
        let names: Vec<String> = (stages.iter())
            .map(|(stage, code)| match container(code) {
                Some((c, _)) => format!("{stage} {}", &blake3::hash(&c).to_hex()[..16]),
                None => format!("{stage} ?"),
            })
            .collect();
        format!("[{}]", names.join(", "))
    };
    let mut jobs: Vec<Job> = (0..compute.len()).map(Job::Compute).collect();
    jobs.extend((0..graphics.len()).map(Job::Graphics));
    // Diagnostics, `VICTORY_PREBUILD_KEYS=<file>`: make nothing, list each
    // record's key (what a run lists pipelines by) with its shaders and
    // state as a run's `VICTORY_PIPELINE_LOG` has them, or why it is
    // skipped; and the records each source named states for.
    let keys_only = std::env::var_os("VICTORY_PREBUILD_KEYS");
    if keys_only.is_some() {
        let mut by: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for g in &graphics {
            for (kind, h) in [
                ("depth-stencil", g.depth_stencil_hash),
                ("rasteriser", g.rasterizer_hash),
                ("blend", g.blend_hash),
            ] {
                *by.entry((kind, named.source.get(&h).copied().unwrap_or("not named")))
                    .or_default() += 1;
            }
        }
        for ((kind, source), n) in by {
            println!("  {kind:<14} {source:<12} {n:>6} records");
        }
    }
    let skipped: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
    let skip = |why: String| *lock(&skipped).entry(why).or_default() += 1;
    // First every record's key, which needs no translation: what a
    // pre-build put in the library is skipped, what an earlier run drew
    // with is made first.
    let lines: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let todo: Mutex<Vec<(bool, Job)>> = Mutex::new(Vec::new());
    let listed = AtomicUsize::new(0);
    each(&jobs, threads, |job| match prepare(job) {
        Ok(p) if keys_only.is_some() => {
            lock(&lines).push(format!("{}\t{job:?}\t{}", p.key(), p.describe()))
        }
        Ok(p) if pipeline::prebuilt(&p.key()) => drop(listed.fetch_add(1, Ordering::Relaxed)),
        Ok(p) => lock(&todo).push((!pipeline::wanted_early(&p.key()), job)),
        Err(why) => {
            if keys_only.is_some() {
                lock(&lines).push(format!("-\t{job:?}\t{}\t{why}", shaders_of(job)));
            }
            skip(why);
        }
    });
    let listed = listed.into_inner();
    if let Some(file) = keys_only {
        std::fs::write(file, lock(&lines).join("\n"))?;
    }
    let mut todo = todo.into_inner().unwrap_or_else(|e| e.into_inner());
    todo.sort_by_key(|&(later, _)| later);
    let todo: Vec<Job> = todo.into_iter().map(|(_, job)| job).collect();

    let next = AtomicUsize::new(0);
    let (made0, loaded0) = pipeline::pipeline_counts();
    let counts = || {
        let (made, loaded) = pipeline::pipeline_counts();
        (made - made0, loaded - loaded0)
    };
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.min(todo.len()))
            .map(|_| {
                s.spawn(|| {
                    while let Some(&job) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if let Err(why) = prepare(job).and_then(|p| p.make()) {
                            skip(why);
                        }
                    }
                })
            })
            .collect();
        let (n, what) = (todo.len(), "pipelines to make");
        crate::translate::report(&pack_path, n, what, &next, started, &workers, || {
            let (made, loaded) = counts();
            format!("; {made} made, {loaded} found in the library")
        });
    });
    pipeline::save_pipeline_cache();
    let (made, loaded) = counts();
    println!(
        "pipelines    {} in the pack, {:.1} s: {made} made, {} already in the library",
        jobs.len(),
        started.elapsed().as_secs_f64(),
        listed + loaded as usize,
    );
    for (why, n) in lock(&skipped).iter() {
        println!("  {n:>7}  skipped: {why}");
    }
    Ok(())
}

/// Runs `f` on every job, on `threads` threads.
fn each(jobs: &[Job], threads: usize, f: impl Fn(Job) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(jobs.len()) {
            s.spawn(|| {
                while let Some(&job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    f(job);
                }
            });
        }
    });
}

/// The hash instructions each native program of a container starts with.
fn program_ids(bytes: &[u8]) -> Vec<([u8; 24], HwStage)> {
    let Ok(c) = Container::parse(bytes) else {
        return Vec::new();
    };
    let Some(part) = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) else {
        return Vec::new();
    };
    native_programs(part.data)
        .map(|programs| {
            (programs.iter())
                .filter_map(|p| Some((shaderpack::program_id(&p.code)?, p.stage)))
                .collect()
        })
        .unwrap_or_default()
}

struct Inputs {
    /// First standalone shader-pack file (`PS\0\x09`) without shader containers.
    direct_pack: Option<(PathBuf, Vec<u8>)>,
    /// Files holding shader containers (`SGD5` or `DXBC`), kept for program/database lookup.
    shader_files: Vec<(PathBuf, Vec<u8>)>,
    /// Remaining files within `MAX_FILE`, checked for a compressed pack only if the fast path misses.
    other_paths: Vec<PathBuf>,
}

fn collect_inputs(paths: &[PathBuf]) -> Inputs {
    let mut inputs = Inputs {
        direct_pack: None,
        shader_files: Vec::new(),
        other_paths: Vec::new(),
    };
    for entry in paths
        .iter()
        .flat_map(|root| {
            walkdir::WalkDir::new(root)
                .into_iter()
                .filter_map(Result::ok)
        })
        .filter(|e| e.file_type().is_file() && e.metadata().is_ok_and(|m| m.len() <= MAX_FILE))
    {
        let path = entry.path();
        let Ok(data) = std::fs::read(path) else {
            continue;
        };
        if !sources(&data).is_empty() {
            inputs.shader_files.push((path.to_owned(), data));
        } else if data.starts_with(shaderpack::MAGIC) {
            if inputs.direct_pack.is_none() {
                inputs.direct_pack = Some((path.to_owned(), data));
            }
        } else {
            inputs.other_paths.push(path.to_owned());
        }
    }
    inputs
}

/// Locates the shader pack across the scanned inputs: first checking direct
/// `PS\0\x09` and `SGD5` fast paths, then falling back to `shaderpack::find`
/// on remaining shader files and other candidate paths.
fn find_pack(inputs: &Inputs) -> Option<(PathBuf, Cow<'_, [u8]>)> {
    if let Some((path, data)) = &inputs.direct_pack
        && let Some(pack) = shaderpack::find(data)
    {
        return Some((path.clone(), pack));
    }
    let find_in = |fast: bool| {
        inputs
            .shader_files
            .iter()
            .filter(|(_, d)| (d.starts_with(shaderpack::MAGIC) || sgd5::is_sgd5(d)) == fast)
            .find_map(|(p, d)| Some((p.clone(), shaderpack::find(d)?)))
    };
    find_in(true).or_else(|| find_in(false)).or_else(|| {
        inputs.other_paths.iter().find_map(|path| {
            let data = std::fs::read(path).ok()?;
            let pack = shaderpack::find(&data)?.into_owned();
            Some((path.clone(), Cow::Owned(pack)))
        })
    })
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "victory-prebuild-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn minimal_pack(fingerprint: u64) -> Vec<u8> {
        let mut out = vec![0; 0x80];
        out[..4].copy_from_slice(shaderpack::MAGIC);
        out[0x48..0x50].copy_from_slice(&fingerprint.to_le_bytes());
        out
    }

    fn minimal_dxbc() -> Vec<u8> {
        let mut out = vec![0; 0x20];
        out[..4].copy_from_slice(b"DXBC");
        out[0x18..0x1C].copy_from_slice(&0x20_u32.to_le_bytes());
        out
    }

    fn zlib_stored(data: &[u8]) -> Vec<u8> {
        assert!(data.len() <= u16::MAX as usize);
        let len = data.len() as u16;
        let mut out = vec![0x78, 0x01, 0x01];
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(data);
        let (mut s1, mut s2) = (1u32, 0u32);
        for &b in data {
            s1 = (s1 + u32::from(b)) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        out.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());
        out
    }

    fn archive_with_pack(prefix: &[u8], pack: &[u8]) -> Vec<u8> {
        let mut out = prefix.to_vec();
        if out.is_empty() {
            out.extend_from_slice(b"ARCHIVE_HEADER_0");
        }
        out.resize(out.len().next_multiple_of(16), 0);
        out.extend_from_slice(&zlib_stored(pack));
        out.extend_from_slice(&[0; 16]);
        out
    }

    #[test]
    fn finds_compressed_pack_only_archive_without_polluting_shader_files() {
        let dir = TempDir::new();
        let junk = dir.write("junk.bin", b"plain non-shader non-pack data");
        let dxbc_path = dir.write("shaders.dxbc", &minimal_dxbc());
        let expected = minimal_pack(0x1122_3344_5566_7788);
        let archive = archive_with_pack(b"RPF_NO_MAGIC_HDR", &expected);
        assert!(!archive.starts_with(shaderpack::MAGIC));
        assert!(!sgd5::is_sgd5(&archive));
        assert!(sources(&archive).is_empty());
        let pack_only = dir.write("pack-only.rpf", &archive);

        let inputs = collect_inputs(&[junk, dxbc_path.clone(), pack_only.clone()]);
        assert_eq!(inputs.shader_files.len(), 1);
        assert_eq!(inputs.shader_files[0].0, dxbc_path);

        let (found_path, found_pack) =
            find_pack(&inputs).expect("compressed pack-only archive must be discovered");
        assert_eq!(found_path, pack_only);
        let parsed = shaderpack::Pack::parse(&found_pack).unwrap();
        assert_eq!(parsed.fingerprint(), 0x1122_3344_5566_7788);
    }

    #[test]
    fn finds_compressed_pack_in_dxbc_archive_without_sgd5_magic() {
        let dir = TempDir::new();
        let expected = minimal_pack(0xCAFE_BABE);
        let mut mixed = minimal_dxbc();
        mixed = archive_with_pack(&mixed, &expected);
        assert!(!mixed.starts_with(shaderpack::MAGIC));
        assert!(!sgd5::is_sgd5(&mixed));
        assert!(!sources(&mixed).is_empty());
        let mixed_path = dir.write("mixed.rpf", &mixed);

        let inputs = collect_inputs(std::slice::from_ref(&mixed_path));
        assert_eq!(inputs.shader_files.len(), 1);
        assert!(inputs.other_paths.is_empty());

        let (found_path, found_pack) =
            find_pack(&inputs).expect("compressed pack in dxbc archive must be discovered");
        assert_eq!(found_path, mixed_path);
        assert_eq!(
            shaderpack::Pack::parse(&found_pack).unwrap().fingerprint(),
            0xCAFE_BABE
        );
    }

    #[test]
    fn prefers_direct_magic_and_sgd5_fast_paths_and_returns_none_when_absent() {
        // Direct PS\0\x09 file wins and stays out of shader_files.
        let dir = TempDir::new();
        let direct = dir.write("direct.pack", &minimal_pack(0xAAAA));
        let fallback = dir.write(
            "fallback.rpf",
            &archive_with_pack(b"RPF_FALLBACK", &minimal_pack(0xBBBB)),
        );
        let dxbc = dir.write("shaders.dxbc", &minimal_dxbc());
        let inputs = collect_inputs(&[fallback, dxbc.clone(), direct.clone()]);
        assert_eq!(inputs.shader_files.len(), 1);
        assert_eq!(inputs.shader_files[0].0, dxbc);
        let (found_path, found_pack) = find_pack(&inputs).unwrap();
        assert_eq!(found_path, direct);
        assert_eq!(
            shaderpack::Pack::parse(&found_pack).unwrap().fingerprint(),
            0xAAAA
        );

        // SGD5 archive wins over pack-only archive via the fast path.
        let dir2 = TempDir::new();
        let mut sgd5_blob = b"SGD5PADDING_0000".to_vec();
        sgd5_blob.extend_from_slice(&minimal_dxbc());
        let sgd5_archive = archive_with_pack(&sgd5_blob, &minimal_pack(0xCCCC));
        let sgd5_path = dir2.write("shaders.sgd5", &sgd5_archive);
        let fallback2 = dir2.write(
            "pack-only.rpf",
            &archive_with_pack(b"RPF_FALLBACK", &minimal_pack(0xDDDD)),
        );
        let inputs2 = collect_inputs(&[fallback2.clone(), sgd5_path.clone()]);
        // Fast path must short-circuit without touching `other_paths`.
        std::fs::remove_file(&fallback2).unwrap();
        let (found_path2, found_pack2) = find_pack(&inputs2).unwrap();
        assert_eq!(found_path2, sgd5_path);
        assert_eq!(
            shaderpack::Pack::parse(&found_pack2).unwrap().fingerprint(),
            0xCCCC
        );

        // No pack anywhere -> None, and non-shader files still do not pollute shader_files.
        let dir3 = TempDir::new();
        let junk = dir3.write("notes.txt", b"not a shader or pack");
        let only_dxbc = dir3.write("only.dxbc", &minimal_dxbc());
        let inputs3 = collect_inputs(&[junk, only_dxbc.clone()]);
        assert_eq!(inputs3.shader_files.len(), 1);
        assert_eq!(inputs3.shader_files[0].0, only_dxbc);
        assert!(find_pack(&inputs3).is_none());
    }
}
