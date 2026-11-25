//! Counts of a module's instructions by kind (a diagnostic): what a
//! translated program costs before the driver's compiler sees it.
//!
//! `VICTORY_SHADER_CENSUS` (any value) turns the counting on; the counts of
//! the module a thread wrote last are read with [`take`]. The `translate`
//! command writes them as a table (see `apps/victory/src/translate.rs`).

use super::{Cast, Module, Op, Tag, Term, V};
use std::cell::RefCell;

/// The kinds, in the order of [`Census::counts`].
pub const KINDS: [&str; 26] = [
    "total",
    "arith",
    "math_call",
    "bitcast",
    "cast",
    "select",
    "guard_select",
    "phi",
    "extract",
    "buffer_load",
    "cbuffer_load",
    "texture_read",
    "lds_load",
    "buffer_store",
    "lds_store",
    "atomic",
    "handle",
    "annotate",
    "dimensions",
    "barrier",
    "wave_op",
    "io_other",
    "gep",
    "clamp",
    "loop_counter",
    "blocks",
];

/// Instruction counts of one written module, by [`KINDS`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Census {
    pub counts: [u32; KINDS.len()],
}

thread_local! {
    /// The modules this thread wrote since the last [`take`]: the length and
    /// a hash of each one's bitcode, with its counts (a program may be
    /// emitted in several forms before one is kept).
    static WRITTEN: RefCell<Vec<(usize, u64, Census)>> = const { RefCell::new(Vec::new()) };
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn parse_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some()
}

/// Whether modules are counted (`VICTORY_SHADER_CENSUS` set).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_enabled(std::env::var_os("VICTORY_SHADER_CENSUS").as_deref()))
}

/// The counts of the module whose bitcode the container `dxil` holds, if
/// this thread wrote it since the last call (which forgets the others).
pub fn take(dxil: &[u8]) -> Option<Census> {
    let written = WRITTEN.with(|l| std::mem::take(&mut *l.borrow_mut()));
    let at = dxil.windows(4).position(|w| w == b"BC\xC0\xDE")?;
    written
        .into_iter()
        .rev()
        .find(|(len, hash, _)| dxil.get(at..at + len).is_some_and(|b| fnv(b) == *hash))
        .map(|w| w.2)
}

/// Counts `m`, whose written form is `bitcode`, and keeps the result for
/// [`take`].
pub(crate) fn note(m: &Module, bitcode: &[u8]) {
    note_if(enabled(), m, bitcode);
}

fn note_if(on: bool, m: &Module, bitcode: &[u8]) {
    if !on {
        return;
    }
    let c = m.census();
    WRITTEN.with(|l| l.borrow_mut().push((bitcode.len(), fnv(bitcode), c)));
}

fn kind(name: &str) -> usize {
    let n = name.strip_prefix("dx.op.").unwrap_or(name);
    let n = n.split('.').next().unwrap_or(n);
    let k = |s: &str| KINDS.iter().position(|k| *k == s).expect("kind");
    match n {
        "unary" | "binary" | "tertiary" | "unaryBits" | "legacyF16ToF32" | "legacyF32ToF16"
        | "isSpecialFloat" | "dot2" | "dot3" | "dot4" | "quaternary" => k("math_call"),
        "bufferLoad" | "rawBufferLoad" => k("buffer_load"),
        "cbufferLoadLegacy" | "cbufferLoad" => k("cbuffer_load"),
        "bufferStore" | "rawBufferStore" | "textureStore" => k("buffer_store"),
        "atomicBinOp" | "atomicCompareExchange" => k("atomic"),
        "createHandle" | "createHandleFromBinding" | "createHandleFromHeap" => k("handle"),
        "annotateHandle" => k("annotate"),
        "getDimensions" => k("dimensions"),
        "barrier" => k("barrier"),
        _ if n.starts_with("wave") || n.starts_with("quad") => k("wave_op"),
        _ if n.starts_with("textureLoad")
            || n.starts_with("sample")
            || n.starts_with("textureGather")
            || n.starts_with("calculateLOD") =>
        {
            k("texture_read")
        }
        _ => k("io_other"),
    }
}

impl Module {
    /// The instructions [`Module::write`] writes, by kind.
    pub(crate) fn census(&self) -> Census {
        let k = |s: &str| KINDS.iter().position(|k| *k == s).expect("kind");
        let call_kind: Vec<usize> = self.funcs.iter().map(|f| kind(&f.name)).collect();
        // The blocks `write` lays out: those reached from the entry.
        let (seen, _) = self.reachable();
        let live = self.live(&seen);
        let mut c = Census::default();
        for (b, blk) in self.blocks.iter().enumerate() {
            if !seen[b] {
                continue;
            }
            c.counts[k("blocks")] += 1;
            let _: Option<Term> = blk.term;
            for &i in blk
                .phis
                .iter()
                .filter(|&&i| live[i as usize])
                .chain(&blk.body)
            {
                c.counts[0] += 1;
                let at = match &self.insts[i as usize].op {
                    Op::Bin(..) | Op::Cmp(..) => k("arith"),
                    Op::Cast(Cast::Bitcast, _) => k("bitcast"),
                    Op::Cast(..) => k("cast"),
                    Op::Select(..) => k("select"),
                    Op::Extract(..) => k("extract"),
                    Op::Phi(_) => k("phi"),
                    Op::Gep(..) => k("gep"),
                    Op::Load(_) => k("lds_load"),
                    Op::Store(..) => k("lds_store"),
                    Op::AtomicRmw(..) | Op::CmpXchg(..) => k("atomic"),
                    Op::Alloca(..) => k("io_other"),
                    Op::Call(f, _) => call_kind[*f as usize],
                };
                c.counts[at] += 1;
                match self.tags.get(&i) {
                    Some(Tag::Guard) => c.counts[k("guard_select")] += 1,
                    Some(Tag::Clamp) => c.counts[k("clamp")] += 1,
                    Some(Tag::Counter) => c.counts[k("loop_counter")] += 1,
                    None => {}
                }
                let _ = V::Inst(i);
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::bitcode::{Attr, Bin};
    use std::ffi::OsStr;

    fn count_of(c: &Census, name: &str) -> u32 {
        c.counts[KINDS.iter().position(|k| *k == name).expect("kind")]
    }

    #[test]
    fn fnv_and_enable_switch_work() {
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);
        assert!(!parse_enabled(None));
        assert!(parse_enabled(Some(OsStr::new("1"))));
    }

    #[test]
    fn enabled_and_note_follow_env_switch() {
        let mut m = Module::default();
        let entry = m.new_block();
        let i32t = m.int(32);
        let void = m.void();
        let one = m.i32(1);
        let sum = m.bin(entry, Bin::Add, one, one);
        let fty = m.func_ty(void, vec![i32t, i32t]);
        let st = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        m.call(entry, st, vec![one, sum]);
        m.tag(sum, Tag::Clamp);
        m.terminate(entry, Term::Ret);
        let bc = b"BC\xC0\xDEtest";
        let mut container = b"DXBC".to_vec();
        container.extend_from_slice(bc);
        if std::env::var_os("VICTORY_SHADER_CENSUS").is_some() {
            assert!(enabled());
            note(&m, bc);
            let c = take(&container).expect("census recorded");
            assert_eq!(count_of(&c, "clamp"), 1);
            return;
        }
        assert!(!enabled());
        note(&m, bc);
        assert!(take(&container).is_none());
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .env("VICTORY_SHADER_CENSUS", "1")
            .args([
                "--exact",
                "emit::bitcode::census::tests::enabled_and_note_follow_env_switch",
                "--nocapture",
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn call_kinds_cover_every_category() {
        for (name, want) in [
            ("dx.op.unary.f32", "math_call"),
            ("dx.op.bufferLoad.i32", "buffer_load"),
            ("dx.op.cbufferLoadLegacy.f32", "cbuffer_load"),
            ("dx.op.textureStore.f32", "buffer_store"),
            ("dx.op.atomicBinOp.i32", "atomic"),
            ("dx.op.createHandleFromHeap", "handle"),
            ("dx.op.annotateHandle", "annotate"),
            ("dx.op.getDimensions", "dimensions"),
            ("dx.op.barrier", "barrier"),
            ("dx.op.waveReadLaneFirst.i32", "wave_op"),
            ("dx.op.quadOp.f32", "wave_op"),
            ("dx.op.textureLoad.f32", "texture_read"),
            ("dx.op.sampleLevel.f32", "texture_read"),
            ("dx.op.textureGather.f32", "texture_read"),
            ("dx.op.calculateLOD", "texture_read"),
            ("dx.op.loadInput.f32", "io_other"),
            ("custom_fn", "io_other"),
        ] {
            assert_eq!(KINDS[kind(name)], want, "{name}");
        }
    }

    #[test]
    fn module_census_and_take_count_live_reachable_instructions() {
        let mut m = Module::default();
        let i32t = m.int(32);
        let f32t = m.float();
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t]);
        let r_fn = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let s_fn = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let mty = m.func_ty(void, vec![]);
        m.function("main", mty, None, true);

        let entry = m.new_block();
        let join = m.new_block();
        let dead_blk = m.new_block();
        let zero = m.i32(0);
        let one = m.i32(1);
        m.terminate(entry, Term::Br(join));
        m.terminate(dead_blk, Term::Ret);
        let _dead_only = m.bin(dead_blk, Bin::Add, one, one);

        let live_phi = m.phi(join, i32t);
        let dead_phi = m.phi(join, i32t);
        m.set_phi(live_phi, vec![(one, entry)]);
        m.set_phi(dead_phi, vec![(one, entry)]);

        let arr_t = m.array(2, i32t);
        let alloca = m.alloca(join, arr_t);
        let gep = m.element(join, alloca, zero);
        m.store(join, gep, one);
        let ld = m.load(join, gep);
        let old = m.cmpxchg(join, gep, zero, one);
        m.tag_when(one, Tag::Counter, true);
        m.tag_when(old, Tag::Counter, false);
        assert!(m.tags.is_empty());
        m.tag_when(old, Tag::Counter, true);
        let sum = m.bin(join, Bin::Add, live_phi, old);
        m.tag_when(sum, Tag::Clamp, true);
        let fval = m.cast(join, Cast::Bitcast, sum, f32t);
        let _back = m.cast(join, Cast::UiToFp, sum, f32t);
        let cmp = m.cmp(join, crate::emit::bitcode::pred::EQ, sum, ld);
        let sel = m.select(join, cmp, fval, fval);
        m.tag_when(sel, Tag::Guard, true);
        let buf = m.call(join, r_fn, vec![sum]);
        m.call(join, s_fn, vec![buf]);
        m.terminate(join, Term::Ret);

        let c = m.census();
        assert_eq!(count_of(&c, "blocks"), 2);
        assert_eq!(count_of(&c, "phi"), 1);
        assert_eq!(count_of(&c, "arith"), 2);
        assert_eq!(count_of(&c, "bitcast"), 1);
        assert_eq!(count_of(&c, "cast"), 1);
        assert_eq!(count_of(&c, "select"), 1);
        assert_eq!(count_of(&c, "extract"), 1);
        assert_eq!(count_of(&c, "gep"), 1);
        assert_eq!(count_of(&c, "lds_load"), 1);
        assert_eq!(count_of(&c, "lds_store"), 1);
        assert_eq!(count_of(&c, "atomic"), 1);
        assert_eq!(count_of(&c, "io_other"), 1);
        assert_eq!(count_of(&c, "buffer_load"), 1);
        assert_eq!(count_of(&c, "buffer_store"), 1);
        assert_eq!(count_of(&c, "clamp"), 1);
        assert_eq!(count_of(&c, "guard_select"), 1);
        assert_eq!(count_of(&c, "loop_counter"), 1);
        assert_eq!(count_of(&c, "total"), 14);

        // `note_if(false, ...)` does not record; `note_if(true, ...)` records and `take` matches last writer.
        let _ = take(&[]);
        note_if(false, &m, b"BC\xC0\xDEaaaa");
        assert!(take(b"BC\xC0\xDEaaaa").is_none());

        let bc1 = b"BC\xC0\xDEone!".to_vec();
        let bc2 = b"BC\xC0\xDEtwo!!".to_vec();
        note_if(true, &m, &bc1);
        note_if(true, &m, &bc2);
        // Truncated slice fails cleanly.
        assert!(take(b"BC\xC0\xDE").is_none());
        // Empty after `take()` clears thread-local buffer.
        note_if(true, &m, &bc1);
        let mut container = b"DXBC____".to_vec();
        container.extend_from_slice(&bc1);
        assert_eq!(take(&container), Some(c.clone()));
        assert!(take(&container).is_none());
        let _ = count_of(&c, "handle");
    }
}
