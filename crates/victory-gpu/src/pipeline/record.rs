//! Recording the title's pipelines and making them before it asks.
//!
//! A run writes what each of its pipelines is made from (the shader
//! containers and the graphics state) beside the shader cache; the next
//! launch reads them and starts them on the workers at once, so the world's
//! first frames find compiled pipelines instead of a burst of compiles.
//! Diagnostics: `VICTORY_PSO_RECORD=1`.

use std::sync::Arc;

use super::Shader;
use super::host::GraphicsState;

/// Where the recordings live, once the cache dir is known.
static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

pub(crate) fn use_dir(dir: std::path::PathBuf) {
    let _ = DIR.set(dir);
}

fn enabled() -> bool {
    cfg!(feature = "diagnostics") && std::env::var("VICTORY_PSO_RECORD").as_deref() == Ok("1")
}

/// The cache dir the loader set, or `VICTORY_PSO_DIR` for a harness run.
fn dir() -> Option<std::path::PathBuf> {
    DIR.get()
        .cloned()
        .or_else(|| std::env::var("VICTORY_PSO_DIR").ok().map(Into::into))
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}

struct Reader<'a> {
    at: usize,
    b: &'a [u8],
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Option<u32> {
        let v = self.b.get(self.at..self.at + 4)?.try_into().ok()?;
        self.at += 4;
        Some(u32::from_le_bytes(v))
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        let v = self.b.get(self.at..self.at + n)?;
        self.at += n;
        Some(v)
    }
}

/// Writes one pipeline: its shaders' containers and its graphics state.
pub(crate) fn record(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) {
    tracing::info!(
        enabled = enabled(),
        shaders = shaders.len(),
        "pso record entry"
    );
    if !enabled() {
        return;
    }
    let Some(dir) = dir() else {
        return;
    };
    let mut out = Vec::new();
    put_u32(&mut out, shaders.len() as u32);
    for s in shaders {
        put_bytes(&mut out, s.stage.as_bytes());
        put_bytes(&mut out, &s.container);
    }
    match graphics {
        Some(g) => {
            out.push(1);
            put_bytes(&mut out, &g.encode());
        }
        None => out.push(0),
    }
    let mut key = blake3::Hasher::new();
    key.update(&out);
    let name = dir.join(format!("{}.pso", key.finalize().to_hex()));
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(name, out);
}

/// Makes every recorded pipeline on the workers, before the title asks.
pub fn prebuild() {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    prebuild_all();
}

fn prebuild_all() {
    if !enabled() {
        return;
    }
    let Some(dir) = dir() else {
        return;
    };
    let Ok(list) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut made = 0u32;
    for entry in list.flatten() {
        if entry.path().extension().and_then(|e| e.to_str()) != Some("pso") {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        let Some((shaders, graphics)) = decode(&bytes) else {
            continue;
        };
        let pending = Arc::new(super::make::Pending::new(shaders, graphics));
        // Only what an earlier run actually drew: the recordings hold every
        // pipeline the title made, most of which it never uses.
        if !super::library::wanted_early(pending.key()) {
            continue;
        }
        super::make::start(&pending);
        made += 1;
    }
    tracing::info!(
        pipelines = made,
        "recorded pipelines started before the title"
    );
    if cfg!(feature = "diagnostics") {
        tracing::info!(
            pipelines = made,
            "recorded pipelines started (stderr mirror)"
        );
    }
}

fn decode(bytes: &[u8]) -> Option<(Vec<Arc<Shader>>, Option<Box<GraphicsState>>)> {
    let mut r = Reader { at: 0, b: bytes };
    let n = r.u32()?;
    let mut shaders = Vec::new();
    for _ in 0..n {
        let stage = std::str::from_utf8(r.bytes()?).ok()?;
        let stage: &'static str = Box::leak(stage.to_owned().into_boxed_str());
        let container = r.bytes()?;
        let code = super::Bytecode {
            code: container.as_ptr(),
            size: container.len(),
        };
        // SAFETY: the container bytes live until copy returns its own Vec.
        shaders.extend(unsafe { super::copy(stage, &code) });
    }
    let graphics = match r.b.get(r.at)? {
        0 => None,
        1 => {
            r.at += 1;
            Some(Box::new(GraphicsState::decode(r.bytes()?)?))
        }
        _ => return None,
    };
    Some((shaders, graphics))
}
