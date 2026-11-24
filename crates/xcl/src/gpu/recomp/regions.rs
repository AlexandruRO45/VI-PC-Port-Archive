//! Outline repeated, single-entry arms of the structured compute CFG. The input is the
//! emitter's own HLSL. A closed set of expressions makes the register interface explicit:
//! other operations, local entry-point state and nonlocal jumps stay in their original scope.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use super::{block_end, insts_in, Ctx};
use crate::gpu::rdna2;

pub(super) fn worthwhile(c: &Ctx, bytes: &[u8]) -> bool {
    // Scalar copy-in/copy-out also costs DXIL. Two measured trees of 1,066 and
    // 6,466 instructions were smaller inline. Look for extensive whole-program
    // repetition before introducing call boundaries.
    if c.insts.len() < 4096 || c.edges.len() < 128 {
        return false;
    }
    let mut copies = HashMap::<&[u8], usize>::new();
    let mut repeated = 0;
    for (i, &start) in c.leaders.iter().enumerate() {
        let Some(last) = insts_in(c, start, block_end(c, i)).last() else { continue };
        let end = if rdna2::branch_target(last).is_some() { last.pc as usize } else { last.pc as usize + last.len };
        let block = &bytes[start as usize..end];
        let count = copies.entry(block).or_default();
        *count += 1;
        if *count >= 16 {
            repeated += block.len();
        }
    }
    repeated > bytes.len() / 3
}

#[derive(Default)]
pub(super) struct Regions {
    // Full string equality is required. The insertion order supplies stable function names.
    by_body: HashMap<String, usize>,
    functions: Vec<Function>,
}

struct Function {
    body: String,
    writes: Vec<bool>,
    uses: usize,
}

fn numbered(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix).is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn register(name: &str) -> bool {
    numbered(name, "s") || numbered(name, "v") || numbered(name, "ttmp") ||
        matches!(name, "scc" | "m0" | "vcc_lo" | "vcc_hi" | "exec_lo" | "exec_hi")
}

// These emitter operations take their registers by value. An operation with an output
// argument needs an explicit write rule before it can be added here.
fn known(name: &str) -> bool {
    matches!(name,
        "if" | "else" | "true" | "false" | "uint" | "uint2" | "uint3" | "uint4" | "uint64_t" |
        "int" | "int2" | "int3" | "int4" | "int64_t" | "float" | "float2" | "float3" | "float4" |
        "bool" | "bool2" | "bool3" | "bool4" | "asfloat" | "asuint" | "asint" | "min" | "max" |
        "select" | "or" | "and" | "abs" | "floor" | "mad" | "rcp" | "rsqrt" | "saturate" |
        "isnan" | "reversebits" | "f16tof32" | "f32tof16" | "cvt_u32" | "bfe_u" | "mul_legacy" |
        "lane_id" | "lanebit" | "ballot" | "xcl_x_ballot" | "xcl_x_ballot_local" |
        "mem" | "Load" | "Load2" | "Load3" | "Load4" | "Store" | "Store2" | "Store3" | "Store4" |
        "xcl_w" | "xcl_uma_base" | "xcl_loadx4" | "xcl_buf_in_range" | "FMT_INFO" |
        "fmt_bytes" | "ld_raw" | "fmt_comp" | "fmt_load_selected" | "dst_sel" | "img_cls" |
        "ResourceDescriptorHeap" | "SamplerDescriptorHeap" | "NonUniformResourceIndex" |
        "Texture2D" | "SamplerState" | "GatherRed" | "SampleLevel"
    ) || numbered(name, "xcl_ss") || (name.len() <= 4 && name.bytes().all(|b| b"xyzw".contains(&b)))
}

fn assigning(tail: &str) -> bool {
    let tail = tail.trim_start();
    (tail.starts_with('=') && !tail.starts_with("==")) ||
        ["+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "++", "--"].iter().any(|op| tail.starts_with(op))
}

impl Regions {
    pub fn outline(&mut self, body: &mut String, start: usize, indent: usize) {
        let mut regs = Vec::<&str>::new();
        let mut outputs = HashSet::new();
        let mut temps = HashMap::new();
        let mut canonical = String::new();
        for raw in body[start..].lines() {
            let line = raw.split("//").next().unwrap().trim();
            if line.is_empty() { continue; }
            // Child calls occupy one whole line and pass each register exactly once.
            // Propagate their copy-out parameters when forming the enclosing function.
            if let Some(call) = line.strip_prefix("xcl_region_") {
                let Some((id, args)) = call.split_once('(') else { return };
                let Some(f) = id.parse::<usize>().ok().and_then(|i| self.functions.get(i)) else { return };
                let Some(args) = args.strip_suffix(");") else { return };
                for (arg, &written) in args.split(", ").zip(&f.writes) {
                    if written { outputs.insert(arg); }
                }
            }
            let mut rest = line;
            while let Some(i) = rest.find(|c: char| c.is_ascii_alphanumeric() || c == '_') {
                canonical.push_str(&rest[..i]);
                rest = &rest[i..];
                let end = rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(rest.len());
                let token = &rest[..end];
                rest = &rest[end..];
                if register(token) {
                    let index = regs.iter().position(|&r| r == token).unwrap_or_else(|| { regs.push(token); regs.len() - 1 });
                    if assigning(rest) { outputs.insert(token); }
                    let _ = write!(canonical, "p{index}");
                } else if let Some(temp) = token.strip_prefix('t').filter(|s| s.starts_with(|c: char| c.is_ascii_digit())) {
                    let digits = temp.find(|c: char| !c.is_ascii_digit()).unwrap_or(temp.len());
                    let suffix = &temp[digits..];
                    if !suffix.is_empty() && !suffix.starts_with('_') { return; }
                    let next = temps.len();
                    let index = temps.entry(&temp[..digits]).or_insert(next);
                    let _ = write!(canonical, "t{index}{suffix}");
                } else if token.as_bytes()[0].is_ascii_digit() || known(token) || numbered(token, "xcl_region_") {
                    canonical.push_str(token);
                } else {
                    return;
                }
            }
            canonical.push_str(rest);
            canonical.push('\n');
        }
        if canonical.len() < 100 { return; }
        let id = *self.by_body.entry(canonical.clone()).or_insert_with(|| {
            let writes = regs.iter().map(|r| outputs.contains(r)).collect();
            self.functions.push(Function { body: canonical, writes, uses: 0 });
            self.functions.len() - 1
        });
        self.functions[id].uses += 1;
        let args = regs.join(", ");
        body.truncate(start);
        let _ = writeln!(body, "{}xcl_region_{id}({args});", "    ".repeat(indent));
    }

    pub fn emit(&self, h: &mut String) {
        for (id, f) in self.functions.iter().enumerate() {
            if f.uses > 1 { h.push_str("[noinline] "); }
            let _ = write!(h, "void xcl_region_{id}(");
            for (i, &write) in f.writes.iter().enumerate() {
                if i != 0 { h.push_str(", "); }
                if write { h.push_str("inout "); }
                let _ = write!(h, "uint p{i}");
            }
            h.push_str(") {\n");
            h.push_str(&f.body);
            h.push_str("}\n");
        }
    }
}
