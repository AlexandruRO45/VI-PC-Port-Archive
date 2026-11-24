//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, MeshEmu, MeshMode, MeshOutput, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies this build: the BLAKE3 of our executable file and of
/// the translation settings it runs with (the loop limit, which a
/// diagnostics build can change at run time).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let exe = std::env::current_exe().and_then(std::fs::read);
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        match exe {
            Ok(mut bytes) => {
                bytes.extend_from_slice(&limit.to_le_bytes());
                *blake3::hash(&bytes).as_bytes()
            }
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                *blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes()
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8], mesh: MeshMode) -> Result<HostShader, String> {
    let id = match mesh {
        MeshMode::Native => "shader.dxil",
        MeshMode::Emulate => "shader.dxil.mesh-emu",
    };
    let translate = |c: &[u8]| victory_shader::emit::translate_with(c, mesh).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => cached(cache, build(), id, container, translate),
        None => translate(container),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    id: &'static str,
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        self.id
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    id: &'static str,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        id,
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH2";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    let encode_outputs = |out: &mut Vec<u8>, items: &[MeshOutput]| {
        word(out, items.len() as u32);
        for o in items {
            let name = o.name.as_bytes();
            word(out, name.len() as u32);
            out.extend(name);
            for v in [
                o.index,
                u32::from(o.kind),
                u32::from(o.comp),
                u32::from(o.interp),
                o.row.unwrap_or(u32::MAX),
                u32::from(o.col),
                u32::from(o.cols),
                o.dword_offset,
            ] {
                word(out, v);
            }
        }
    };
    match &s.mesh_emu {
        None => word(&mut out, 0),
        Some(m) => {
            word(&mut out, 1);
            for v in [
                m.threads,
                m.max_vertices,
                m.max_primitives,
                m.vertex_dwords,
                m.prim_dwords,
                u32::from(m.reads_group_id),
            ] {
                word(&mut out, v);
            }
            encode_outputs(&mut out, &m.vertex_outputs);
            encode_outputs(&mut out, &m.prim_outputs);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
        fn outputs(&mut self) -> Option<Vec<MeshOutput>> {
            let n = self.word()? as usize;
            let mut out = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                let name = String::from_utf8(self.list()?).ok()?;
                let index = self.word()?;
                let kind = u8::try_from(self.word()?).ok()?;
                let comp = u8::try_from(self.word()?).ok()?;
                let interp = u8::try_from(self.word()?).ok()?;
                let row = match self.word()? {
                    u32::MAX => None,
                    r => Some(r),
                };
                let col = u8::try_from(self.word()?).ok()?;
                let cols = u8::try_from(self.word()?).ok()?;
                let dword_offset = self.word()?;
                out.push(MeshOutput {
                    name,
                    index,
                    kind,
                    comp,
                    interp,
                    row,
                    col,
                    cols,
                    dword_offset,
                });
            }
            Some(out)
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let mesh_emu = match r.word()? {
        0 => None,
        1 => Some(MeshEmu {
            threads: r.word()?,
            max_vertices: r.word()?,
            max_primitives: r.word()?,
            vertex_dwords: r.word()?,
            prim_dwords: r.word()?,
            reads_group_id: match r.word()? {
                0 => false,
                1 => true,
                _ => return None,
            },
            vertex_outputs: r.outputs()?,
            prim_outputs: r.outputs()?,
        }),
        _ => return None,
    };
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        mesh_emu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            mesh_emu: None,
        }
    }

    fn mesh_emu_shader() -> HostShader {
        HostShader {
            stage: HwStage::Ms,
            dxil: b"DXBC emulated mesh cs".to_vec(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            mesh_emu: Some(MeshEmu {
                threads: 64,
                max_vertices: 32,
                max_primitives: 16,
                vertex_dwords: 8,
                prim_dwords: 4,
                reads_group_id: true,
                vertex_outputs: vec![
                    MeshOutput {
                        name: "SV_Position".into(),
                        index: 0,
                        kind: 3,
                        comp: 9,
                        interp: 4,
                        row: None,
                        col: 0,
                        cols: 4,
                        dword_offset: 0,
                    },
                    MeshOutput {
                        name: "COLOR".into(),
                        index: 0,
                        kind: 0,
                        comp: 9,
                        interp: 2,
                        row: Some(0),
                        col: 0,
                        cols: 4,
                        dword_offset: 4,
                    },
                ],
                prim_outputs: vec![MeshOutput {
                    name: "PRIMID".into(),
                    index: 0,
                    kind: 0,
                    comp: 5,
                    interp: 1,
                    row: Some(1),
                    col: 0,
                    cols: 1,
                    dword_offset: 0,
                }],
            }),
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, "shader.dxil", container, translate),
            Ok(shader())
        );
        assert_eq!(
            cached(&cache, &build, "shader.dxil", container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another converter id caches separately.
        let emu_runs = AtomicU64::new(0);
        let emu_translate = |_: &[u8]| {
            emu_runs.fetch_add(1, Ordering::Relaxed);
            Ok(mesh_emu_shader())
        };
        assert_eq!(
            cached(&cache, &build, "shader.dxil.mesh-emu", container, emu_translate),
            Ok(mesh_emu_shader())
        );
        assert_eq!(emu_runs.load(Ordering::Relaxed), 1);

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], "shader.dxil", container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH2 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, "shader.dxil", container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, "shader.dxil", container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, "shader.dxil", b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    /// A local debugging aid: dumps the bindings of the mesh (emulated)
    /// entries in a real run's cache, so the draw path's root parameters can
    /// be checked against what the title's shaders actually read. Run with
    /// `cargo test -p victory-gpu dump_cached_mesh -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a local run's shader cache"]
    fn dump_cached_mesh_entries() {
        let Some(local) = std::env::var_os("LOCALAPPDATA") else {
            return;
        };
        let dir = std::path::Path::new(&local).join("victory").join("cache").join("shaders");
        let entries = walk(&dir);
        let mut mesh = 0;
        for e in &entries {
            let Ok(bytes) = std::fs::read(e) else {
                continue;
            };
            let Some(s) = decode(&bytes) else {
                continue;
            };
            if s.stage != HwStage::Ms {
                continue;
            }
            mesh += 1;
            println!("=== {} ({}) mesh_emu: {}", e.display(), s.dxil.len(), s.mesh_emu.is_some());
            println!(
                "user_data ({}): {:?}",
                s.user_data.len(),
                s.user_data
                    .iter()
                    .map(|u| format!("dword {} slot {} {:?}+{}", u.index, u.slot, u.root, u.dword))
                    .collect::<Vec<_>>()
            );
            println!("root_cbvs {:?} root_srvs {:?}", s.root_cbvs, s.root_srvs);
            println!(
                "samplers {} static {}",
                s.samplers.len(),
                s.static_samplers.len()
            );
            if let Some(m) = &s.mesh_emu {
                println!(
                    "mesh_emu: threads {} verts {} prims {} vert_dwords {} prim_dwords {} \
                     reads_group_id {} group_bytes {}",
                    m.threads,
                    m.max_vertices,
                    m.max_primitives,
                    m.vertex_dwords,
                    m.prim_dwords,
                    m.reads_group_id,
                    m.group_bytes()
                );
                println!(
                    "vertex outputs: {:?}",
                    m.vertex_outputs
                        .iter()
                        .map(|o| format!("{}@{}", o.name, o.dword_offset))
                        .collect::<Vec<_>>()
                );
                println!(
                    "prim outputs: {:?}",
                    m.prim_outputs
                        .iter()
                        .map(|o| format!("{}@{}", o.name, o.dword_offset))
                        .collect::<Vec<_>>()
                );
            }
        }
        println!("mesh entries: {mesh} of {}", entries.len());
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        for s in [shader(), mesh_emu_shader()] {
            let bytes = encode(&s);
            assert_eq!(decode(&bytes), Some(s));
            assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
            assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        }
        assert_eq!(decode(b"VSH1"), None);
    }
}
