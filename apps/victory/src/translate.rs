//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied (decision 0003).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, validator::Validator};
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{
    Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir, native_programs,
};
use victory_shader::{io, resource};

use crate::shader::{MAX_FILE, Source, sources};

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    programs: usize,
    translated: usize,
    instructions: usize,
    structured: usize,
    vector: usize,
    guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    opaque: usize,
    lane_branches: usize,
    uniform_branches: usize,
    tracked: usize,
    loaded: usize,
    built: usize,
    constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    dynamic: usize,
    scalar: usize,
    used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    io_programs: usize,
    io_raised: usize,
    interpolated: usize,
    vertex_values: usize,
    /// Exports by target kind.
    exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    emit_programs: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    /// The same for compute programs.
    cs_programs: usize,
    cs_emitted: usize,
    cs_validated: usize,
    cs_pipelines: usize,
    /// The same for mesh programs.
    ms_programs: usize,
    ms_emitted: usize,
    ms_validated: usize,
    ms_pipelines: usize,
    /// Reason → programs stopped by it, and the first one.
    failed: BTreeMap<String, (usize, String)>,
    /// With `--list`: one line per program and outcome.
    list: bool,
    lines: Vec<String>,
}

impl Tally {
    fn fail(&mut self, reason: String, example: impl FnOnce() -> String) {
        if self.list {
            let e = example();
            self.lines.push(format!("{e}  {reason}"));
            self.failed.entry(reason).or_insert((0, e)).0 += 1;
            return;
        }
        self.failed
            .entry(reason)
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.programs += other.programs;
        self.translated += other.translated;
        self.instructions += other.instructions;
        self.structured += other.structured;
        self.vector += other.vector;
        self.guarded += other.guarded;
        self.opaque += other.opaque;
        self.lane_branches += other.lane_branches;
        self.uniform_branches += other.uniform_branches;
        self.tracked += other.tracked;
        self.loaded += other.loaded;
        self.built += other.built;
        self.constant += other.constant;
        self.dynamic += other.dynamic;
        self.scalar += other.scalar;
        self.used += other.used;
        for (k, n) in other.untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += other.unlisted;
        self.io_programs += other.io_programs;
        self.io_raised += other.io_raised;
        self.interpolated += other.interpolated;
        self.vertex_values += other.vertex_values;
        self.compressed += other.compressed;
        self.emit_programs += other.emit_programs;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.cs_programs += other.cs_programs;
        self.cs_emitted += other.cs_emitted;
        self.cs_validated += other.cs_validated;
        self.cs_pipelines += other.cs_pipelines;
        self.ms_programs += other.ms_programs;
        self.ms_emitted += other.ms_emitted;
        self.ms_validated += other.ms_validated;
        self.ms_pipelines += other.ms_pipelines;
        for (k, n) in other.exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.lines.extend(other.lines);
        for (reason, (n, example)) in other.failed {
            self.failed.entry(reason).or_insert((0, example)).0 += n;
        }
    }
}

/// Emitted shaders by label, shared between threads.
type Kept = std::sync::Arc<Mutex<Vec<(String, emit::HostShader)>>>;

/// The optional checks after emission, one set per thread.
struct Checks {
    /// Shaders named in `VICTORY_PAIRS` (`vs+ps,...`, label substrings),
    /// kept to build those pipelines after the walk.
    keep: Option<Kept>,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    list: bool,
    jobs: Option<usize>,
) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let dll = validate.then(Validator::default_path).flatten();
    if validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    if let Some(manifest) = std::env::var_os("VICTORY_LOCAL_SHADER_MANIFEST") {
        return local_shader_baseline(&paths, &manifest, dll.as_deref(), validate, jobs);
    }
    if pipelines {
        Host::new(false).map_err(anyhow::Error::msg)?;
    }
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| p.split_once('+'))
        .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Vec::new()));
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        let mut t = Tally {
                            list,
                            ..Tally::default()
                        };
                        let mut checks = Checks {
                            keep: (!pairs.is_empty()).then(|| kept.clone()),
                            validator: validate
                                .then(|| Validator::load(dll.as_deref()).ok())
                                .flatten(),
                            host: pipelines
                                .then(|| Host::new(false).ok().map(|h| (h, host::stub_ps())))
                                .flatten(),
                        };
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            translate(job, entry.path(), &seen, &mut t, &mut checks);
                        }
                        lock(&total).merge(t);
                    });
                }
            });
        }
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    println!(
        "containers   {} ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "programs     {} translated of {} ({} IR instructions)",
        t.translated, t.programs, t.instructions
    );
    println!(
        "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
         branches {} per lane, {} uniform)",
        t.structured,
        t.translated,
        t.vector,
        t.guarded,
        t.opaque,
        t.lane_branches,
        t.uniform_branches
    );
    println!(
        "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
         {} scalar accesses; {} with a computed term)",
        t.tracked, t.translated, t.loaded, t.built, t.constant, t.scalar, t.dynamic
    );
    println!(
        "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
        t.used,
        t.untraced_used.values().sum::<usize>(),
        t.untraced_used,
        t.unlisted
    );
    println!(
        "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
         {} vertex values; exports {:?}, {} still f16 pairs)",
        t.io_raised, t.io_programs, t.interpolated, t.vertex_values, t.exports, t.compressed
    );
    println!(
        "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
        t.emitted, t.emit_programs, t.validated, t.pipelines
    );
    println!(
        "             {} emitted of {} compute programs; {} validated, {} pipelines",
        t.cs_emitted, t.cs_programs, t.cs_validated, t.cs_pipelines
    );
    println!(
        "             {} emitted of {} mesh programs; {} validated, {} pipelines",
        t.ms_emitted, t.ms_programs, t.ms_validated, t.ms_pipelines
    );
    if !pairs.is_empty() {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        let kept = lock(&kept);
        let find = |name: &str| kept.iter().find(|(l, _)| l.contains(name));
        for (v, p) in &pairs {
            let made = match (find(v), find(p)) {
                (Some((_, vs)), Some((_, ps))) => if vs.stage == HwStage::Ms {
                    host.mesh_pipeline(vs, ps)
                } else {
                    host.pipeline(vs, ps)
                }
                .map(|_| "pipeline created".to_owned())
                .unwrap_or_else(|e| e),
                _ => "not emitted".to_owned(),
            };
            println!("pair {v} + {p}: {made}");
        }
    }
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in failed {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
    Ok(())
}

fn local_shader_baseline(
    paths: &[PathBuf],
    manifest: &std::ffi::OsStr,
    dll: Option<&Path>,
    validate: bool,
    jobs: Option<usize>,
) -> Result<()> {
    let ids: HashSet<String> = std::fs::read_to_string(manifest)?
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(id, _)| id.into())
        .collect();
    let found = Mutex::new(HashSet::new());
    let threads = jobs.unwrap_or(4);
    for root in paths {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let sources = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..threads.min(sources.len()) {
                    scope.spawn(|| {
                        let mut validator = validate.then(|| Validator::load(dll).unwrap());
                        while let Some(source) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = source.bytes() else {
                                continue;
                            };
                            let end = bytes
                                .get(24..28)
                                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
                            let entire = matches!(source, Source::Packed(_)).then_some(bytes.as_ref());
                            let trimmed = end.and_then(|n| bytes.get(..n));
                            for input in entire.into_iter().chain(trimmed) {
                                let hash = blake3::hash(input).to_hex();
                                let id = &hash[..16];
                                if !ids.contains(id) || !lock(&found).insert(id.to_owned()) {
                                    continue;
                                }
                                if std::env::var_os("VICTORY_LOCAL_HANG_DIAG").is_some() {
                                    local_control_diag(input);
                                }
                                let result = match emit::translate(input) {
                                    Ok(mut shader) => match validator.as_mut() {
                                        Some(v) => match v.validate(&mut shader.dxil) {
                                            Ok(()) => {
                                                if std::env::var_os("VICTORY_LOCAL_AUDIT").is_some() {
                                                    format!("validated\t{}", local_wave_audit(input, &shader))
                                                } else {
                                                    "validated".into()
                                                }
                                            },
                                            Err(e) => {
                                                let why = e.lines().filter(|l| l.contains("error:")).take(6).collect::<Vec<_>>().join(" | ");
                                                if std::env::var_os("VICTORY_LOCAL_STAGE_DIAG").is_some()
                                                    && let Some(stage) = local_stage_diag(input)
                                                {
                                                    format!("validator: {why}\t{stage}")
                                                } else {
                                                    format!("validator: {why}")
                                                }
                                            },
                                        },
                                        None => "translated".into(),
                                    },
                                    Err(e) => {
                                        let mut reason = e.to_string();
                                        if std::env::var_os("VICTORY_LOCAL_SHADER_DIAG").is_some()
                                            && let ShaderError::Unsupported { offset, what } = e
                                            && let Some(diag) = local_shader_diag(input, offset, what)
                                        {
                                            reason.push('\t');
                                            reason.push_str(&diag);
                                        }
                                        reason
                                    }
                                };
                                println!("{id}\t{result}");
                            }
                        }
                    });
                }
            });
        }
    }
    println!("matched {} of {}", lock(&found).len(), ids.len());
    Ok(())
}

fn local_wave_audit(input: &[u8], shader: &emit::HostShader) -> String {
    let Ok(c) = Container::parse(&shader.dxil) else { return "unknown".into() };
    let Some(dxil) = c.part(FourCC::DXIL).and_then(|p| victory_shader::DxilProgram::parse(p.data).ok()) else { return "unknown".into() };
    let wave = c.part(FourCC::SFI0).is_some_and(|p| p.data.get(1).is_some_and(|b| b & 0x40 != 0));
    if shader.stage != HwStage::Ps || !wave {
        return format!("{:?} sm={}.{} wave={wave}", shader.stage, dxil.shader_model.0, dxil.shader_model.1);
    }
    let fallback = format!("Ps sm={}.{} wave=true", dxil.shader_model.0, dxil.shader_model.1);
    let n = Container::parse(input).ok().and_then(|c| c.part(FourCC::DXIL).and_then(|p| native_programs(p.data).ok()))
        .and_then(|ns| ns.into_iter().find(|n| n.stage == HwStage::Ps));
    let Some(n) = n else { return fallback };
    let Ok(p) = ir::build(&n.patched_code(), n.wave64, n.vgprs) else { return fallback };
    let Ok(st) = structure::structure(&p) else { return fallback };
    let loops = p.blocks.iter().enumerate().filter(|(b, blk)| blk.exit.successors().into_iter().any(|t| t <= *b)).count();
    let mut first = 0;
    let mut guarded = 0;
    for (b, blk) in p.blocks.iter().enumerate() {
        for (i, inst) in blk.insts.iter().enumerate() {
            if inst.name() == "v_readfirstlane_b32" {
                first += 1;
                guarded += usize::from(st.guards[b][i].is_some());
            }
        }
    }
    format!("{fallback} loops:{loops} firstlane:{first} guarded:{guarded}")
}

fn local_control_diag(input: &[u8]) {
    let Ok(c) = Container::parse(input) else { return };
    let Some(dxil) = c.part(FourCC::DXIL) else { return };
    let Ok(programs) = native_programs(dxil.data) else { return };
    let Some(n) = programs.into_iter().find(|p| matches!(p.stage, HwStage::Ps | HwStage::Cs | HwStage::Ms | HwStage::Vs)) else { return };
    let Ok(p) = ir::build(&n.patched_code(), n.wave64, n.vgprs) else { return };
    let Ok(st) = structure::structure(&p) else { return };
    let Ok(res) = resource::track(&p, &n.layout, n.stage, Some(&st)) else { return };
    let mut defs = std::collections::HashMap::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        for i in &blk.insts {
            for &(_, v) in &i.writes {
                defs.insert(v, (b, i.name()));
            }
        }
        for q in &blk.phis {
            defs.insert(q.value, (b, "phi"));
        }
    }
    let show = |v: ir::Value| {
        match defs.get(&v) {
            Some(&(b, op)) => format!("%{}={op}@{:x}", v.0, p.blocks[b].offset),
            None => format!("%{}=input", v.0),
        }
    };
    let formula = |m| {
        if m == structure::Bool::TRUE || m == structure::Bool::FALSE {
            return format!("{}", m == structure::Bool::TRUE);
        }
        st.bools.support(m).into_iter().map(|a| match a {
            Atom::Lane(v) => format!("lane({})", show(v)),
            Atom::Uniform(v) => format!("uniform({})", show(v)),
            Atom::Phi(v) => format!("join({})", show(v)),
            Atom::Quad(v) => format!("quad({})", show(v)),
            Atom::Opaque(v) => format!("opaque({})", show(v)),
            Atom::Constant(_) => "constant".into(),
        }).collect::<Vec<_>>().join(", ")
    };
    eprintln!("native: {:?} wave{} blocks={} waterfall reads={} loads={}", n.stage, if n.wave64 {64}else{32}, p.blocks.len(), res.waterfall.reads.len(), res.waterfall.loads.len());
    for (b, blk) in p.blocks.iter().enumerate() {
        if !blk.exit.successors().iter().any(|&t| t <= b) {
            continue;
        }
        let header = blk.exit.successors().into_iter().min().unwrap();
        let ops = p.blocks[header..=b].iter().flat_map(|b| &b.insts);
        let mut wave_ops = std::collections::BTreeSet::new();
        for op in ops {
            if op.name().contains("lane") || op.name().contains("wqm") || op.name().starts_with("s_bcnt") {
                wave_ops.insert(op.name());
            }
        }
        let guard = match st.exits[b] {
            LaneExit::Branch { when, .. } => formula(when),
            _ => "unconditional".into(),
        };
        eprintln!("loop {:x} <- {:x} {} [{}]; ops {:?}", p.blocks[header].offset, blk.offset, blk.insts.last().map_or("", |i| i.name()), guard, wave_ops);
    }
    for (b, blk) in p.blocks.iter().enumerate() {
        if !(0x29b8..=0x2d38).contains(&blk.offset) && !(0x5318..=0x5638).contains(&blk.offset) {
            continue;
        }
        let condition = match st.exits[b] { LaneExit::Branch {when,..} => formula(when), _ => String::new() };
        eprintln!("block {:x}, exit {:?} [{}]", blk.offset, st.exits[b], condition);
        for phi in blk.phis.iter().filter(|q| matches!(q.reg, ir::Reg::S(_))) {
            if let Some(&m) = st.masks.get(&phi.value) {
                let vals = blk.preds.iter().zip(&phi.args).map(|(&q, &a)| format!("{:x}: {}", p.blocks[q].offset, show(a))).collect::<Vec<_>>();
                eprintln!("  {} {:?} [{}] <- {}", show(phi.value), phi.reg, formula(m), vals.join("; "));
            }
        }
        for (j,i) in blk.insts.iter().enumerate().filter(|(_,i)| {
            i.name().contains("exec") || i.name().contains("lane") || i.name().contains("wqm") || i.name().starts_with("s_cbranch") || i.name().starts_with("v_cmp") || i.writes.iter().any(|(r,_)| matches!(r, ir::Reg::S(ir::EXEC | ir::VCC) | ir::Reg::Scc))
        }) {
            let inputs = i.reads.iter().filter(|(r, _)| !matches!(r, ir::Reg::S(ir::EXEC))).map(|(_, v)| show(*v)).collect::<Vec<_>>();
            let outs = i.writes.iter().filter_map(|(_, v)| st.masks.get(v).map(|m| format!("{} [{}]", show(*v), formula(*m)))).collect::<Vec<_>>();
            let guard = st.guards[b][j].map_or("none".into(), formula);
            eprintln!("  {:x} {}: {} -> {} (guard {guard})", i.machine.offset, i.name(), inputs.join(", "), outs.join(", "));
        }
    }
}

fn local_stage_diag(input: &[u8]) -> Option<String> {
    let c = Container::parse(input).ok()?;
    let n = native_programs(c.part(FourCC::DXIL)?.data).ok()?.into_iter().find(|p| {
        matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms)
    })?;
    let victory_shader::native::StageIo::Cs(cs) = n.io else { return None };
    let p = ir::build(&n.patched_code(), n.wave64, n.vgprs).ok()?;
    let mut names = std::collections::BTreeSet::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        let name = i.name();
        if name.starts_with("image_") || name.contains("swizzle") || name.contains("permute") {
            names.insert(name);
        }
    }
    Some(format!("threads:{:?}; {}", cs.threads, names.into_iter().collect::<Vec<_>>().join(",")))
}

fn local_shader_diag(input: &[u8], offset: usize, what: &str) -> Option<String> {
    use victory_shader::{gfx10::{Fields, Operand}, ir::{Reg, Value}};
    let c = Container::parse(input).ok()?;
    let n = native_programs(c.part(FourCC::DXIL)?.data).ok()?.into_iter().find(|p| {
        matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms)
    })?;
    let p = ir::build(&n.patched_code(), n.wave64, n.vgprs).ok()?;
    let st = structure::structure(&p).ok()?;
    let definitions: std::collections::HashMap<_, _> = p.blocks.iter().flat_map(|b| &b.insts)
        .flat_map(|i| i.writes.iter().map(move |&(_, v)| (v, i))).collect();
    let phis: std::collections::HashMap<_, _> = p.blocks.iter().flat_map(|b| &b.phis)
        .map(|q| (q.value, q)).collect();
    let describe = |v: Value| {
        let origin = |v| definitions.get(&v).map_or("input", |i| i.name());
        if let Some(q) = phis.get(&v) {
            let mut names: Vec<_> = q.args.iter().map(|&a| if phis.contains_key(&a) { "phi" } else { origin(a) }).collect();
            names.sort_unstable();
            names.dedup();
            format!("phi({})", names.join(","))
        } else if let Some(d) = definitions.get(&v) {
            let mut names: Vec<_> = d.reads.iter().filter(|(r, _)| matches!(r, Reg::S(x) if *x < 126)).map(|&(_, a)| {
                if phis.contains_key(&a) { "phi" } else { origin(a) }
            }).collect();
            names.sort_unstable();
            names.dedup();
            format!("{}({})", d.name(), names.join(","))
        } else {
            origin(v).into()
        }
    };
    let stage = format!("{:?}/{}", n.stage, if n.wave64 { 64 } else { 32 });
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    if what == "a structured buffer of unknown stride" || what == "a scalar memory read as data" {
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).ok()?;
        let mut forms = std::collections::BTreeSet::new();
        let mut uses = std::collections::HashMap::<Value, Vec<&str>>::new();
        for i in insts().filter(|i| matches!(i.fields, Fields::Sop { .. } | Fields::Valu(_))) {
            for &(_, v) in &i.reads {
                uses.entry(v).or_default().push(i.name());
            }
        }
        let address = |site| res.scalar.iter().find(|(s, _)| *s == site).map(|(_, a)| a);
        for d in &res.descriptors {
            let i = &p.blocks[d.site.0].insts[d.site.1];
            let mut name = i.name();
            if what.starts_with("a structured") {
                if !matches!(i.fields, Fields::Buffer { idxen: true, format: None, .. }) || name.contains("format") {
                    continue;
                }
            } else if !name.starts_with("s_buffer_load") {
                continue;
            } else {
                name = "s_buffer_load";
            }
            for source in d.source.leaves() {
                match *source {
                    resource::Source::Loaded { load, first } | resource::Source::Fetched { load, first } => {
                        let Some(a) = address(load) else { continue };
                        let Some(r) = n.layout.get(a.param) else { continue };
                        let dword = a.offset / 4 + u32::from(first);
                        let u = n.used.iter().filter(|u| u.root == r.root && u.user_data == r.user_data && u.table_offset_dwords <= dword)
                            .max_by_key(|u| u.table_offset_dwords);
                        if what.starts_with("a structured") {
                            forms.insert(format!("{name}/{:?}/{}", r.root, u.map_or(0, |u| u.stride)));
                        } else {
                            let load = &p.blocks[load.0].insts[load.1];
                            for (k, &(_, v)) in load.writes.iter().enumerate() {
                                for usage in uses.get(&v).into_iter().flatten() {
                                    if !matches!(*usage, "s_mov_b32" | "s_mov_b64" | "v_mov_b32" | "v_readlane_b32" | "v_writelane_b32") {
                                        forms.insert(format!("{:?}/{:?}/{k}/{usage}", r.root, u.map(|u| u.kind)));
                                    }
                                }
                            }
                        }
                    }
                    resource::Source::Built { address: a, bits, .. } if what.starts_with("a structured") => {
                        forms.insert(format!("{name}/Built/{:?}/{}", n.layout.get(a.param).map(|r| r.root), bits >> 16 & 0x3fff));
                    }
                    _ => {}
                }
            }
        }
        return Some(format!("{stage} {}", forms.into_iter().collect::<Vec<_>>().join("; ")));
    }
    if matches!(what, "an address not from a root parameter" | "an address from a phi") {
        let i = insts().find(|i| i.machine.offset == offset)?;
        let mut forms = std::collections::BTreeSet::new();
        for &(reg, v) in &i.reads {
            if matches!(reg, Reg::S(c) if c < 126) || matches!(reg, Reg::V(_)) {
                forms.insert(describe(v));
            }
        }
        return Some(format!("{stage} {} {}", i.name(), forms.into_iter().collect::<Vec<_>>().join("; ")));
    }
    if what == "a descriptor read only as data" {
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).ok()?;
        let associated: HashSet<_> = res.descriptors.iter().flat_map(|d| d.source.leaves())
            .filter_map(|s| match *s { resource::Source::Loaded { load, .. } => Some(load), _ => None })
            .collect();
        let mut words = std::collections::HashMap::new();
        for &(site, a) in &res.scalar {
            let i = &p.blocks[site.0].insts[site.1];
            if i.name().starts_with("s_load_") && !associated.contains(&site) {
                if let Some(r) = n.layout.get(a.param) {
                    for (k, &(_, v)) in i.writes.iter().enumerate() {
                        words.insert(v, (r.root, i.name(), k, a.dynamic.map(&describe)));
                    }
                }
            }
        }
        let mut forms = std::collections::BTreeSet::new();
        for i in insts() {
            let ops = match &i.fields {
                Fields::Sop { src, .. } if i.name().contains("bfe") => src,
                Fields::Valu(v) if i.name() == "v_alignbit_b32" => &v.src,
                _ => continue,
            };
            for &op in ops {
                if let Operand::Scalar { code, .. } = op
                    && let Some(w) = i.read(Reg::S(code)).and_then(|v| words.get(&v))
                {
                    let pattern = ops.last().map(|op| match *op {
                        Operand::Literal(x) => format!("offset:{},width:{}", x & 63, x >> 16 & 127),
                        Operand::Const(c @ 128..=192) => format!("offset:{},width:0", c - 128),
                        _ => "dynamic".to_owned(),
                    }).unwrap_or_default();
                    forms.insert(format!("{:?}/{}/{} {} {pattern} addr:{:?}", w.0, w.1, w.2, i.name(), w.3));
                }
            }
        }
        let bitcode = victory_shader::DxilProgram::parse(c.part(FourCC::DXIL)?.data).map_or(0, |d| d.bitcode.len());
        return Some(format!("{stage} bitcode:{} {}", bitcode != 0, forms.into_iter().collect::<Vec<_>>().join("; ")));
    }
    if std::env::var_os("VICTORY_LOCAL_RESOURCE_DIAG").is_some()
        && matches!(what, "a scalar memory read as data" | "a structured buffer of unknown stride" | "a descriptor's address or format read as data") {
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).ok()?;
        let leaves: Vec<_> = res.descriptors.iter().flat_map(|d| {
            d.source.leaves().into_iter().map(move |s| (d, s))
        }).collect();
        let mut readers = std::collections::HashMap::<Value, Vec<&str>>::new();
        for i in insts().filter(|i| !matches!(i.name(), "s_mov_b32" | "s_mov_b64" | "v_mov_b32" | "v_readlane_b32" | "v_writelane_b32")) {
            for &(_, v) in &i.reads {
                readers.entry(v).or_default().push(i.name());
            }
        }
        let mut forms = Vec::new();
        let address = |site| res.scalar.iter().find(|(s, _)| *s == site).map(|(_, a)| a);
        for &(site, a) in &res.scalar {
            let i = &p.blocks[site.0].insts[site.1];
            if !i.name().starts_with("s_load_") && !i.name().starts_with("global_load_") {
                continue;
            }
            let mut uses = Vec::new();
            for &(d, s) in &leaves {
                if let resource::Source::Loaded { load, first } = *s
                    && let Some(at) = address(load)
                    && a.param == at.param && a.dynamic == at.dynamic
                {
                    let start = at.offset + 4 * u32::from(first);
                    let delta = a.offset.wrapping_sub(start) as i32;
                    if (-64..64).contains(&delta) {
                        uses.push(format!("{}:{delta}:{}", if site == load {"same"} else {"other"}, p.blocks[d.site.0].insts[d.site.1].name()));
                    }
                }
            }
            uses.sort();
            uses.dedup();
            let Some(r) = n.layout.get(a.param) else { continue };
            let mut types: Vec<_> = n.used.iter().filter(|u| r.user_data == u.user_data && r.root == u.root)
                .map(|u| (format!("{:?}", u.kind), u.dwords, u.stride)).collect();
            types.sort();
            types.dedup();
            let mut consumers: Vec<_> = i.writes.iter().flat_map(|(_, w)| readers.get(w).into_iter().flatten().copied()).collect();
            consumers.sort_unstable();
            consumers.dedup();
            if !consumers.is_empty() && (!what.contains("only as data") || !uses.iter().any(|s| s.starts_with("same:"))) {
                forms.push(format!("{:?}/{}/{}  {:?}  {}  {}", r.root, i.name(), if a.dynamic.is_some() {"dynamic"} else {"const"}, types, uses.join(","), consumers.join(",")));
            }
        }
        forms.sort();
        forms.dedup();
        return Some(format!("{stage} {}", forms.join("; ")));
    }
    if what == "a select on an untracked mask" {
        let mut forms = Vec::new();
        for i in insts().filter(|i| i.name() == "v_cndmask_b32") {
            let Fields::Valu(v) = &i.fields else { continue };
            let form = match v.src.get(2) {
                Some(&Operand::Scalar { code, .. }) => {
                    match i.read(Reg::S(code)) {
                        Some(v) if st.masks.contains_key(&v) => continue,
                        Some(v) => describe(v),
                        None => "unread".into(),
                    }
                }
                Some(Operand::Const(128)) => "zero".into(),
                Some(Operand::Const(193)) => "minus one".into(),
                Some(Operand::Const(_)) => "constant".into(),
                Some(Operand::Literal(_)) => "literal".into(),
                _ => "other".into(),
            };
            forms.push(form);
        }
        forms.sort();
        forms.dedup();
        return Some(format!("{stage} {}", forms.join("; ")));
    }
    if what == "a lane mask holding lanes elsewhere read as data" {
        let mut forms = Vec::new();
        for i in insts() {
            for &(r, v) in &i.reads {
                if !matches!(r, Reg::S(c) if c < 126) {
                    continue;
                }
                let lo = st.highs.get(&v).copied().unwrap_or(v);
                if st.masks.contains_key(&lo) && !st.active.contains(&lo)
                    && !i.writes.iter().any(|&(_, w)| st.masks.contains_key(&w))
                {
                    forms.push(format!("{} <- {}", i.name(), describe(lo)));
                }
            }
        }
        forms.sort();
        forms.dedup();
        return Some(format!("{stage} {}", forms.join("; ")));
    }
    let i = insts().find(|i| i.machine.offset == offset)?;
    let reads = if let Fields::Valu(v) = &i.fields
        && i.name() == "v_cndmask_b32"
    {
        match v.src.get(2) {
            Some(&Operand::Scalar { code, .. }) => (code..code + if n.wave64 { 2 } else { 1 })
                .map(|r| i.read(Reg::S(r)).map_or("unread".into(), describe))
                .collect::<Vec<_>>().join(","),
            Some(Operand::Const(_)) => "constant".into(),
            Some(Operand::Literal(_)) => "literal".into(),
            _ => "other".into(),
        }
    } else {
        let mut masks: Vec<_> = i.reads.iter().filter(|(r, v)| {
            matches!(r, Reg::S(c) if *c < 126) && (st.masks.contains_key(v) || st.highs.contains_key(v))
        }).map(|&(_, v)| describe(v)).collect();
        masks.sort();
        masks.dedup();
        masks.join(",")
    };
    Some(format!("{stage} {} {reads}", i.name()))
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail("corrupt record".into(), example);
    };
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    let Some(dxil) = container.part(FourCC::DXIL).map(|p| p.data) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    if !lock(seen).insert(blake3::hash(dxil)) {
        t.duplicates += 1;
        return;
    }
    let programs = match native_programs(dxil) {
        Ok(p) => p,
        Err(e) => return t.fail(e.to_string(), example),
    };
    for p in programs {
        t.programs += 1;
        match ir::build(&p.patched_code(), p.wave64, p.vgprs) {
            Ok(program) => {
                t.translated += 1;
                t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
                // Per-lane descriptors are traced on the per-lane form; its
                // failure is listed after the resources' and I/O's.
                let structured = structure::structure(&program);
                let lanes = structured.as_ref().ok();
                let res = match resource::track(&program, &p.layout, p.stage, lanes) {
                    Ok(r) => {
                        tally_resources(&p, &r, t);
                        Some(r)
                    }
                    Err(ShaderError::Unsupported { offset, what }) => {
                        t.fail(format!("resources: {what}"), || {
                            format!("{} {:?} at {offset:#x}", example(), p.stage)
                        });
                        None
                    }
                    Err(e) => {
                        t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage));
                        None
                    }
                };
                let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
                let compute = p.stage == HwStage::Cs;
                let mut raised = None;
                if graphics || compute || p.stage == HwStage::Ms {
                    t.io_programs += usize::from(graphics);
                    match io::raise(&program, &p) {
                        Ok(r) => {
                            if graphics {
                                tally_io(&r, t);
                            }
                            raised = Some(r);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => t
                            .fail(format!("I/O: {what}"), || {
                                format!("{} {:?} at {offset:#x}", example(), p.stage)
                            }),
                        Err(e) => t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage)),
                    }
                }
                let structured = match structured {
                    Ok(s) => {
                        tally_structured(&program, &s, t);
                        Some(s)
                    }
                    Err(e) => {
                        let reason = match e {
                            ShaderError::Unsupported { what, .. } => format!("structure: {what}"),
                            ShaderError::Isa { reason, .. } => format!("structure: {reason}"),
                            e => e.to_string(),
                        };
                        t.fail(reason, || format!("{} {:?}", example(), p.stage));
                        None
                    }
                };
                if let (Some(r), Some(i), Some(s)) = (res, raised, structured) {
                    let label = || format!("{} {:?}", example(), p.stage);
                    *counter(t, p.stage, 0) += 1;
                    match emit::emit(&container, &p, &program, &s, &i, &r) {
                        Ok(shader) => {
                            *counter(t, p.stage, 1) += 1;
                            check(shader, t, checks, label);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => t
                            .fail(format!("emit: {what}"), || {
                                format!("{} at {offset:#x}", label())
                            }),
                        Err(e) => t.fail(format!("emit: {e}"), label),
                    }
                }
            }
            Err(e) => {
                let reason = match e {
                    ShaderError::Unsupported { what, .. } => format!("unsupported: {what}"),
                    e => e.to_string(),
                };
                t.fail(reason, || format!("{} {:?}", example(), p.stage));
            }
        }
    }
}

/// Validates an emitted shader if asked to.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) {
    if let Some(keep) = &checks.keep {
        lock(keep).push((label(), shader.clone()));
    }
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if let Some((h, stub_ps)) = &checks.host {
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop)),
        };
        match made {
            Ok(()) => {
                *counter(t, shader.stage, 3) += 1;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => t.fail(format!("pipeline: {e}"), &label),
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return;
    };
    match v.validate(&mut shader.dxil) {
        Ok(()) => {
            *counter(t, shader.stage, 2) += 1;
            if t.list {
                t.lines.push(format!("{}  validated", label()));
            }
        }
        Err(text) => {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let n = t
                    .failed
                    .get(&format!("validator: {first}"))
                    .map_or(0, |f| f.0);
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    let _ = std::fs::write(format!("{name}.dxil"), &shader.dxil);
                    let _ = std::fs::write(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.fail(format!("validator: {first}"), label);
        }
    }
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines.
fn counter(t: &mut Tally, stage: HwStage, k: usize) -> &mut usize {
    let [a, b, c, d] = match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        _ => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    };
    [a, b, c, d].into_iter().nth(k).expect("0 to 3")
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } => t.interpolated += 1,
            io::Interp::Vertex { .. } => t.vertex_values += 1,
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } | resource::Source::Fetched { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. }
            | resource::Source::Select { .. }
            | resource::Source::Lanes { .. }
            | resource::Source::LanePhi { .. } => {
                unreachable!("leaves are no phis, selects or per-lane choices")
            }
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables are no root parameter.
        if let Some(a) = a
            && a.param != resource::CODE
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(program: &ir::Program, s: &structure::Structured, t: &mut Tally) {
    t.structured += 1;
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
