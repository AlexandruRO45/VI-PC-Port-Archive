//! I/O raising (decision 0003): what a program's input registers, its
//! interpolation and its exports mean.
//!
//! Pixel shaders get the VGPRs `SPI_PS_INPUT_ADDR` enables, in a fixed
//! order (barycentrics, then position, front face...). `v_interp_p1` and
//! `v_interp_p2` (and their f16 forms) evaluate an attribute from a pair of
//! barycentrics; `v_interp_mov` reads one vertex's value. After the 32-bit
//! export patch ([`NativeProgram::patched_code`]) exports carry plain
//! 32-bit values: `mrt<n>` is render target n, `mrtz` depth, stencil and
//! sample mask, `null` nothing. A lane whose EXEC bit is clear at the
//! export that ends the program (`done`) is discarded: that export's guard
//! after structurisation.
//!
//! Vertex shaders that run as the hardware VS get the vertex id in v0, the
//! primitive id in v2 and the instance id in v3; the fetch shader loads
//! channel k of each attribute into its VGPR + k. `exp pos0` is the
//! position, `exp param<n>` the output with parameter n.
//!
//! Attributes and outputs are named by their row: a register of the
//! container's signature part.
//!
//! Compute programs get the thread id in the group in v0-v2 and the group
//! id in the SGPRs after the user data, as `COMPUTE_PGM_RSRC2` enables them.
//!
//! Mesh programs run as one NGG stage. The program computes its thread
//! index from the wave index (s3 bits 24-27) and the lane (`v_mbcnt`), and
//! its group id from v5 of the first lane (`v_readfirstlane`: the group
//! times the group size). Lane n exports vertex n (`exp pos`, `exp param`)
//! and primitive n (`exp prim`: 10-bit vertex indices at bits 0, 10, 20,
//! bit 31 culls); `param` entries with bit 7 are per-primitive outputs. One
//! lane sends the output counts (`s_sendmsg` GS_ALLOC_REQ, m0 = vertices |
//! primitives << 12).

use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::resource::Site;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    VertexId,
    PrimitiveId,
    InstanceId,
    /// A vertex attribute the fetch shader loads.
    Attribute {
        row: u8,
        channel: u8,
    },
    /// `index` 0 is i, 1 is j (2 the pull model's w).
    Barycentric {
        kind: Barycentric,
        index: u8,
    },
    /// Pixel position channel (0-3: x, y, z, w).
    Position(u8),
    FrontFace,
    /// Sample id and other bits.
    Ancillary,
    SampleCoverage,
    FixedPosition,
    LineStipple,
    /// Compute: thread-group id component (an SGPR).
    GroupId(u8),
    /// Compute: thread id within the group, component (a VGPR).
    ThreadIdInGroup(u8),
    /// Compute: the thread-group info SGPR (wave index and count).
    GroupInfo,
    /// Mesh: v5, the group id times the group size (read as uniform).
    MeshGroup,
    /// Mesh: s3, the wave's index in the group in bits 24-27.
    MergedWaveInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barycentric {
    PerspectiveSample,
    PerspectiveCenter,
    PerspectiveCentroid,
    PerspectivePullModel,
    LinearSample,
    LinearCenter,
    LinearCentroid,
}

/// The pixel input VGPRs in hardware order: one entry per
/// `SPI_PS_INPUT_ADDR` bit.
const PS_VGPRS: [(u8, Option<Barycentric>); 16] = [
    (2, Some(Barycentric::PerspectiveSample)),
    (2, Some(Barycentric::PerspectiveCenter)),
    (2, Some(Barycentric::PerspectiveCentroid)),
    (3, Some(Barycentric::PerspectivePullModel)),
    (2, Some(Barycentric::LinearSample)),
    (2, Some(Barycentric::LinearCenter)),
    (2, Some(Barycentric::LinearCentroid)),
    (1, None), // line stipple
    (1, None), // position x
    (1, None), // y
    (1, None), // z
    (1, None), // w
    (1, None), // front face
    (1, None), // ancillary
    (1, None), // sample coverage
    (1, None), // fixed-point position
];

fn ps_system(bit: usize) -> Input {
    match bit {
        7 => Input::LineStipple,
        8..=11 => Input::Position(bit as u8 - 8),
        12 => Input::FrontFace,
        13 => Input::Ancillary,
        14 => Input::SampleCoverage,
        _ => Input::FixedPosition,
    }
}

/// An attribute channel of the input signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribute {
    pub usage: Usage,
    pub row: u8,
    pub channel: u8,
    /// The upper f16 half of the channel (packed fp16 inputs).
    pub high: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interp {
    /// This site's result (`v_interp_p2*`) is the attribute interpolated
    /// with `barycentric`; `first` are the `v_interp_p1*` halves it
    /// completes (one per path where the compiler shares a first half).
    Interpolated {
        attribute: Attribute,
        barycentric: Barycentric,
        first: Vec<Site>,
    },
    /// One vertex's value (`v_interp_mov_f32`), as encoded: 0 for P10,
    /// 1 for P20, 2 for P0 (the provoking vertex).
    Vertex { attribute: Attribute, vertex: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Color(u8),
    /// Depth, stencil, sample mask (channels as `DB_SHADER_CONTROL` enables).
    Depth,
    Null,
    Position(u8),
    Param {
        usage: Usage,
        row: u8,
    },
    /// `param0` of a vertex shader without outputs: the hardware wants one
    /// parameter export regardless.
    Unread,
    /// The primitive export of a merged or mesh stage.
    Primitive,
    /// A mesh program's per-primitive output (a row of `PSG1`).
    PrimitiveParam {
        usage: Usage,
        row: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub site: Site,
    pub target: Target,
    /// Per channel, as the instruction enables them.
    pub values: [Option<Value>; 4],
    /// Two f16 values per dword.
    pub compressed: bool,
    /// The program's last export (for pixels: kills the lanes not in EXEC).
    pub done: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Io {
    /// Input values the hardware loads, with their meaning; other VGPRs
    /// read before written hold undefined values.
    pub inputs: Vec<(Value, Input)>,
    pub interps: Vec<(Site, Interp)>,
    pub exports: Vec<Export>,
}

/// Raises the I/O of `p`, built from `native`'s patched code. Fails for
/// stages whose stage data is not read and on constructs not covered yet.
pub fn raise(p: &Program, native: &NativeProgram) -> Result<Io, ShaderError> {
    let unsupported = |offset, what| ShaderError::Unsupported { offset, what };
    let (loaded, ps_inputs, params) = match (&native.io, native.stage) {
        (
            StageIo::Ps {
                inputs, input_addr, ..
            },
            HwStage::Ps,
        ) => {
            // A pixel shader exports no parameters; `exp param` is refused below.
            let loaded = ps_vgprs(*input_addr)
                .into_iter()
                .map(|(r, i)| (Reg::V(r), i))
                .collect();
            (loaded, &inputs[..], &[][..])
        }
        (StageIo::Vs { inputs, outputs }, HwStage::Vs) => {
            let mut loaded = vec![
                (Reg::V(0), Input::VertexId),
                (Reg::V(2), Input::PrimitiveId),
                (Reg::V(3), Input::InstanceId),
            ];
            for i in inputs {
                for channel in (0..4).filter(|c| i.mask >> c & 1 != 0) {
                    let input = Input::Attribute {
                        row: i.row,
                        channel,
                    };
                    // The stage data names the first VGPR with a byte; a
                    // channel past v255 would wrap round onto v0-v2.
                    let reg = i
                        .vgpr
                        .checked_add(channel)
                        .ok_or(ShaderError::Malformed("vertex attribute past v255"))?;
                    // The fetch shader writes each VGPR once; a second
                    // meaning for it would leave one of the two unread.
                    if loaded.iter().any(|l| l.0 == Reg::V(reg)) {
                        return Err(ShaderError::Malformed("two vertex inputs in one VGPR"));
                    }
                    loaded.push((Reg::V(reg), input));
                }
            }
            (loaded, &[][..], &outputs[..])
        }
        // Compute: thread ids in v0-v2, group ids and the group info in the
        // SGPRs after the user data.
        (StageIo::Cs(cs), HwStage::Cs) => {
            let (ids, info) = cs.system_sgprs();
            // `TIDIG_COMP_CNT` is two bits of the title's stage data; 3 would
            // name a fourth component (v3), which no thread id has.
            if cs.thread_id_vgprs() > 3 {
                return Err(ShaderError::Malformed("a thread id component past z"));
            }
            let mut loaded: Vec<(Reg, Input)> = (0..cs.thread_id_vgprs())
                .map(|c| (Reg::V(c), Input::ThreadIdInGroup(c)))
                .collect();
            for (c, s) in ids.into_iter().enumerate() {
                if let Some(s) = s {
                    loaded.push((Reg::S(s), Input::GroupId(c as u8)));
                }
            }
            loaded.extend(info.map(|s| (Reg::S(s), Input::GroupInfo)));
            (loaded, &[][..], &[][..])
        }
        (StageIo::Ms(ms), HwStage::Ms) => {
            let group = p.inputs.iter().find(|i| i.0 == Reg::V(5)).map(|i| i.1);
            // The group input holds the same value in every lane only as the
            // first lane reads it; a VGPR write reads its old value too.
            let per_lane = p.blocks.iter().flat_map(|b| &b.insts).find(|i| {
                group.is_some_and(|g| i.reads.contains(&(Reg::V(5), g)))
                    && i.name() != "v_readfirstlane_b32"
                    && i.write(Reg::V(5)).is_none()
            });
            if let Some(i) = per_lane {
                return Err(unsupported(
                    i.machine.offset,
                    "the mesh group input read per lane",
                ));
            }
            let loaded = vec![
                (Reg::V(5), Input::MeshGroup),
                (Reg::S(3), Input::MergedWaveInfo),
            ];
            (loaded, &[][..], &ms.outputs[..])
        }
        _ => return Err(unsupported(0, "I/O of this stage")),
    };
    let inputs: Vec<(Value, Input)> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| loaded.iter().find(|l| l.0 == reg).map(|l| (v, l.1)))
        .collect();
    let input_of = |v: Value| inputs.iter().find(|i| i.0 == v).map(|i| i.1);
    let attribute = |inst: &Inst, attr: &Operand| match *attr {
        Operand::Attr { index, chan, high } => {
            let mut listed = ps_inputs.iter().filter(|i: &&PsInput| i.attr == index);
            let i = listed.next().ok_or(unsupported(
                inst.machine.offset,
                "an interpolated attribute the stage data does not list",
            ))?;
            // Two rows for one attribute: which one this reads is unknown.
            if listed.next().is_some() {
                return Err(ShaderError::Malformed(
                    "an attribute the stage data lists twice",
                ));
            }
            Ok(Attribute {
                usage: i.usage,
                row: i.row,
                channel: chan,
                high,
            })
        }
        _ => Err(unsupported(inst.machine.offset, "an interpolation operand")),
    };
    // A VGPR value's barycentric kind and index, if it is one as loaded.
    let barycentric = |v: Option<Value>| match v.and_then(input_of) {
        Some(Input::Barycentric { kind, index }) => Some((kind, index)),
        _ => None,
    };

    let mut io = Io {
        inputs: inputs.clone(),
        ..Io::default()
    };
    let defs: std::collections::HashMap<Value, Site> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
        })
        .collect();
    let phis: std::collections::HashMap<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .phis
                .iter()
                .enumerate()
                .map(move |(i, phi)| (phi.value, (b, i)))
        })
        .collect();
    let halves = Halves {
        p,
        defs: &defs,
        phis: &phis,
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let at = inst.machine.offset;
            let name = inst.name();
            match &inst.fields {
                Fields::Vintrp { dst, src, attr } => {
                    let attribute = attribute(inst, attr)?;
                    match (name, *src) {
                        ("v_interp_mov_f32", Operand::Const(vertex)) => {
                            // VSRC is eight bits, but only P10, P20 and P0
                            // name a vertex; 3 and up read no vertex at all.
                            if vertex > 2 {
                                return Err(ShaderError::Malformed(
                                    "a vertex interpolation naming no vertex",
                                ));
                            }
                            io.interps
                                .push((site, Interp::Vertex { attribute, vertex }));
                        }
                        ("v_interp_p1_f32", _) => {}
                        ("v_interp_p2_f32", Operand::Vector { reg, .. }) => {
                            let Operand::Vector { reg: d, .. } = *dst else {
                                return Err(unsupported(at, "an interpolation operand"));
                            };
                            // The first half's result is this one's old dst.
                            let interp = halves
                                .pair(
                                    attr,
                                    inst.read(Reg::V(d)),
                                    inst.read(Reg::V(reg)),
                                    &barycentric,
                                )
                                .map_err(|what| unsupported(at, what))?;
                            io.interps.push((site, interp.with(attribute)));
                        }
                        _ => return Err(unsupported(at, "an interpolation form")),
                    }
                }
                Fields::Valu(v) if name.starts_with("v_interp_") => match name {
                    "v_interp_p1ll_f16" => {}
                    "v_interp_p2_f16" => {
                        let attribute = attribute(inst, &v.src[0])?;
                        let read = |k: usize| match v.src.get(k) {
                            Some(&Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                            _ => None,
                        };
                        let interp = halves
                            .pair(&v.src[0], read(2), read(1), &barycentric)
                            .map_err(|what| unsupported(at, what))?;
                        io.interps.push((site, interp.with(attribute)));
                    }
                    _ => return Err(unsupported(at, "an interpolation form")),
                },
                Fields::Exp {
                    target,
                    src,
                    compr,
                    done,
                    ..
                } => {
                    let target = match *target {
                        t @ 0..=7 => Target::Color(t),
                        8 => Target::Depth,
                        9 => Target::Null,
                        t @ 12..=16 => Target::Position(t - 12),
                        20 => Target::Primitive,
                        // Parameters feed the pixel shader; one has none to write.
                        32..=63 if native.stage == HwStage::Ps => {
                            return Err(ShaderError::Malformed(
                                "a pixel shader exporting a parameter",
                            ));
                        }
                        32 if params.is_empty() => Target::Unread,
                        t @ 32..=63 => {
                            // Bit 7: a mesh program's per-primitive output.
                            let mut listed = params.iter().filter(|o| o.param & 0x7F == t - 32);
                            let o = listed.next().ok_or(unsupported(
                                at,
                                "a parameter export the stage data does not list",
                            ))?;
                            // Two outputs (or a per-vertex and a per-primitive
                            // one) in one parameter: which this writes is unknown.
                            if listed.next().is_some() {
                                return Err(ShaderError::Malformed(
                                    "a parameter the stage data lists twice",
                                ));
                            }
                            let (usage, row) = (o.usage, o.row);
                            if o.param & 0x80 != 0 {
                                Target::PrimitiveParam { usage, row }
                            } else {
                                Target::Param { usage, row }
                            }
                        }
                        _ => return Err(unsupported(at, "an export target")),
                    };
                    let values = src.map(|s| match s {
                        Some(Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                        _ => None,
                    });
                    io.exports.push(Export {
                        site,
                        target,
                        values,
                        compressed: *compr,
                        done: *done,
                    });
                }
                _ => {}
            }
        }
    }
    Ok(io)
}

/// The pixel input VGPRs that `addr` enables, with their meaning.
fn ps_vgprs(addr: u32) -> Vec<(u8, Input)> {
    let mut out = Vec::new();
    let mut next = 0u8;
    for (bit, &(count, kind)) in PS_VGPRS.iter().enumerate() {
        if addr >> bit & 1 == 0 {
            continue;
        }
        for index in 0..count {
            let input = match kind {
                Some(kind) => Input::Barycentric { kind, index },
                None => ps_system(bit),
            };
            out.push((next + index, input));
        }
        next += count;
    }
    out
}

/// An interpolation before its attribute is known.
struct Pair {
    barycentric: Barycentric,
    first: Vec<Site>,
}

impl Pair {
    fn with(self, attribute: Attribute) -> Interp {
        Interp::Interpolated {
            attribute,
            barycentric: self.barycentric,
            first: self.first,
        }
    }
}

/// Finds the first halves of interpolations.
struct Halves<'a> {
    p: &'a Program,
    defs: &'a std::collections::HashMap<Value, Site>,
    phis: &'a std::collections::HashMap<Value, (usize, usize)>,
}

impl Halves<'_> {
    /// The interpolation whose second half reads the first half's result
    /// `acc`, the barycentric `j`, and attribute `attr`: every first half
    /// must read the same attribute channel and the same kind's i.
    fn pair(
        &self,
        attr: &Operand,
        acc: Option<Value>,
        j: Option<Value>,
        barycentric: &impl Fn(Option<Value>) -> Option<(Barycentric, u8)>,
    ) -> Result<Pair, &'static str> {
        let mut first = Vec::new();
        self.sites(
            acc.ok_or("an interpolation without its first half")?,
            &mut Vec::new(),
            &mut first,
        )?;
        let Some((kind, 1)) = barycentric(j) else {
            return Err("an interpolation with computed barycentrics");
        };
        for &(b, i) in &first {
            let inst = &self.p.blocks[b].insts[i];
            let (a, i) = match &inst.fields {
                Fields::Vintrp {
                    src: Operand::Vector { reg, .. },
                    attr,
                    ..
                } if inst.name() == "v_interp_p1_f32" => (*attr, inst.read(Reg::V(*reg))),
                Fields::Valu(v) if inst.name() == "v_interp_p1ll_f16" => match v.src.get(1) {
                    Some(&Operand::Vector { reg, .. }) => (v.src[0], inst.read(Reg::V(reg))),
                    _ => return Err("an interpolation without its first half"),
                },
                _ => return Err("an interpolation without its first half"),
            };
            if a != *attr {
                return Err("interpolation halves of different attributes");
            }
            if barycentric(i) != Some((kind, 0)) {
                return Err("an interpolation with computed barycentrics");
            }
        }
        Ok(Pair {
            barycentric: kind,
            first,
        })
    }

    /// The instructions `v` comes from, through phis (`seen`: phis on the
    /// way; a loop back to one adds nothing).
    fn sites(
        &self,
        v: Value,
        seen: &mut Vec<Value>,
        out: &mut Vec<Site>,
    ) -> Result<(), &'static str> {
        if let Some(&site) = self.defs.get(&v) {
            if !out.contains(&site) {
                out.push(site);
            }
            return Ok(());
        }
        let &(b, i) = self
            .phis
            .get(&v)
            .ok_or("an interpolation without its first half")?;
        if !seen.contains(&v) {
            seen.push(v);
            for &a in &self.p.blocks[b].phis[i].args {
                self.sites(a, seen, out)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ExportPatch;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn program(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::ir::build(&code, false, 256).unwrap()
    }

    fn pixel(inputs: Vec<PsInput>, input_addr: u32) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs,
                input_addr,
                db_shader_control: 0,
                patches: Vec::<ExportPatch>::new(),
            },
            code: vec![],
        }
    }

    const TEXCOORD: PsInput = PsInput {
        usage: Usage::Generic,
        row: 1,
        mask: 3,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 1,
    };

    #[test]
    fn interpolation_pairs_and_color_exports() {
        // Blit's pixel shader: TEXCOORD.xy from the perspective-center i/j
        // in v0/v1, then exported unconverted to MRT0.
        let p = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0302_0102, // exp mrt0, v2, v1, v2, v3 done vm
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        let input = |reg| p.inputs.iter().find(|i| i.0 == Reg::V(reg)).unwrap().1;
        let center = |index| Input::Barycentric {
            kind: Barycentric::PerspectiveCenter,
            index,
        };
        assert!(io.inputs.contains(&(input(0), center(0))));
        assert!(io.inputs.contains(&(input(1), center(1))));
        // v2 and v3 are read before written but hold nothing.
        assert!(!io.inputs.iter().any(|i| i.0 == input(3)));
        assert_eq!(
            io.interps,
            [(
                (0, 1),
                Interp::Interpolated {
                    attribute: Attribute {
                        usage: Usage::Generic,
                        row: 1,
                        channel: 0,
                        high: false
                    },
                    barycentric: Barycentric::PerspectiveCenter,
                    first: vec![(0, 0)],
                }
            )]
        );
        let [e] = &io.exports[..] else {
            panic!("{:?}", io.exports)
        };
        assert_eq!(
            (e.target, e.done, e.compressed),
            (Target::Color(0), true, false)
        );
        assert_eq!(e.values[0], p.blocks[0].insts[1].write(Reg::V(2)));
    }

    // Both branches feed the accumulator in v4 to one interpolation at the join.
    fn joined_interpolation(taken: u32) -> Result<Io, ShaderError> {
        let p = program(&[
            0xBF85_0002, // s_cbranch_scc1 -> 12
            0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
            0xBF82_0001, // s_branch -> 16
            taken,
            0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
            S_ENDPGM,
        ]);
        // Perspective center in v0/v1, linear center in v2/v3.
        raise(&p, &pixel(vec![TEXCOORD], (1 << 1) | (1 << 5)))
    }

    #[test]
    fn interpolation_at_a_join_records_both_first_halves() {
        let io = joined_interpolation(0xC810_0000).unwrap();
        let [
            (
                (3, 0),
                Interp::Interpolated {
                    attribute,
                    barycentric,
                    first,
                },
            ),
        ] = io.interps.as_slice()
        else {
            panic!("unexpected interpolation sites: {:?}", io.interps);
        };
        assert_eq!(*barycentric, Barycentric::PerspectiveCenter);
        assert_eq!((attribute.row, attribute.channel), (TEXCOORD.row, 0));
        let mut first = first.clone();
        first.sort_unstable();
        assert_eq!(first, [(1, 0), (2, 0)]);
    }

    #[test]
    fn interpolation_at_a_join_checks_each_incoming_value() {
        for (taken, what) in [
            (0xC810_0100, "interpolation halves of different attributes"), // attr0.y
            (0xC810_0002, "an interpolation with computed barycentrics"),  // linear i
            (0x7E08_0300, "an interpolation without its first half"),      // v_mov_b32 v4, v0
        ] {
            assert_eq!(
                joined_interpolation(taken),
                Err(ShaderError::Unsupported { offset: 16, what }),
                "taken instruction {taken:#x}"
            );
        }
    }

    #[test]
    fn unpaired_or_unknown_interpolation_fails() {
        // p2 alone: its old dst is no p1's result.
        let lone = program(&[0xC809_0001, S_ENDPGM]); // v_interp_p2_f32 v2, v1, attr0.x
        assert!(raise(&lone, &pixel(vec![TEXCOORD], 2)).is_err());
        // attr1 is not in the stage data.
        let other = program(&[0xC808_0400, S_ENDPGM]); // v_interp_p1_f32 v2, v0, attr1.x
        assert!(raise(&other, &pixel(vec![TEXCOORD], 2)).is_err());
    }

    #[test]
    fn a_vertex_interpolation_names_p10_p20_or_p0() {
        // v_interp_mov_f32 v2, <vsrc>, attr0.x
        let mov = |vsrc: u32| program(&[0xC80A_0000 | vsrc, S_ENDPGM]);
        for vertex in 0..=2u8 {
            let io = raise(&mov(vertex.into()), &pixel(vec![TEXCOORD], 2)).unwrap();
            assert_eq!(
                io.interps,
                [(
                    (0, 0),
                    Interp::Vertex {
                        attribute: Attribute {
                            usage: Usage::Generic,
                            row: 1,
                            channel: 0,
                            high: false
                        },
                        vertex
                    }
                )]
            );
        }
        for vsrc in [3, 4, 0x80, 0xFF] {
            assert_eq!(
                raise(&mov(vsrc), &pixel(vec![TEXCOORD], 2)).unwrap_err(),
                ShaderError::Malformed("a vertex interpolation naming no vertex"),
                "vsrc {vsrc:#x}"
            );
        }
    }

    #[test]
    fn mesh_inputs_and_primitive_export() {
        let mesh = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, true, 256).unwrap();
            let native = NativeProgram {
                stage: HwStage::Ms,
                wave64: true,
                vgprs: 256,
                layout: vec![],
                used: vec![],
                io: StageIo::Ms(crate::native::MeshIo {
                    outputs: vec![],
                    clip_cntl: 0,
                    threads: 128,
                    max_vertices: 128,
                    max_primitives: 128,
                    topology: 2,
                    lds_bytes: 0,
                }),
                code,
            };
            raise(&p, &native)
        };
        let io = mesh(&[
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xF800_0941,
            0x0000_0009, // exp prim, v9, off, off, off done
            S_ENDPGM,
        ])
        .unwrap();
        assert!(io.inputs.iter().any(|i| i.1 == Input::MeshGroup));
        assert_eq!(io.exports[0].target, Target::Primitive);
        // The group input is uniform only as the first lane reads it.
        assert!(mesh(&[0x7E02_0305, S_ENDPGM]).is_err()); // v_mov_b32 v1, v5
    }

    fn vertex(inputs: Vec<crate::native::VsInput>) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Vs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Vs {
                inputs,
                outputs: vec![],
            },
            code: vec![],
        }
    }

    /// An attribute's channels go to consecutive VGPRs from a byte-sized
    /// base; one that would pass v255 is refused instead of wrapping onto
    /// the vertex id in v0.
    #[test]
    fn vertex_attributes_stay_below_v256() {
        let p = program(&[S_ENDPGM]);
        let input = |vgpr, mask| crate::native::VsInput {
            usage: Usage::Generic,
            row: 0,
            mask,
            vgpr,
        };
        assert!(raise(&p, &vertex(vec![input(252, 0xF)])).is_ok());
        assert!(raise(&p, &vertex(vec![input(255, 0x1)])).is_ok());
        for (vgpr, mask) in [(253, 0xF), (255, 0x2), (254, 0x8)] {
            assert_eq!(
                raise(&p, &vertex(vec![input(vgpr, mask)])),
                Err(ShaderError::Malformed("vertex attribute past v255")),
                "v{vgpr} mask {mask:#x}"
            );
        }
    }

    /// The fetch shader loads each VGPR once: an attribute channel landing
    /// on another one (or on the vertex, primitive or instance id) is
    /// refused rather than resolved to whichever input is listed first.
    #[test]
    fn vertex_inputs_do_not_share_a_vgpr() {
        let p = program(&[S_ENDPGM]);
        let input = |row, vgpr, mask| crate::native::VsInput {
            usage: Usage::Generic,
            row,
            mask,
            vgpr,
        };
        assert!(raise(&p, &vertex(vec![input(0, 4, 0xF), input(1, 8, 0xF)])).is_ok());
        // A channel the mask leaves off loads nothing.
        assert!(raise(&p, &vertex(vec![input(0, 4, 0x7), input(1, 7, 0x1)])).is_ok());
        for inputs in [
            vec![input(0, 4, 0xF), input(1, 7, 0x1)],
            vec![input(0, 4, 0x1), input(1, 4, 0x1)],
            vec![input(0, 3, 0x1)], // the instance id
            vec![input(0, 0, 0x1)], // the vertex id
        ] {
            assert_eq!(
                raise(&p, &vertex(inputs.clone())),
                Err(ShaderError::Malformed("two vertex inputs in one VGPR")),
                "{inputs:?}"
            );
        }
    }

    /// An interpolated attribute and a parameter export each name one row
    /// of the signature; stage data listing either twice is refused.
    #[test]
    fn attributes_and_parameters_are_listed_once() {
        let interp = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            S_ENDPGM,
        ]);
        let again = PsInput { row: 2, ..TEXCOORD };
        assert!(raise(&interp, &pixel(vec![TEXCOORD], 2)).is_ok());
        // Another attribute's entry is no ambiguity.
        let other = PsInput { attr: 1, ..again };
        assert!(raise(&interp, &pixel(vec![TEXCOORD, other], 2)).is_ok());
        assert_eq!(
            raise(&interp, &pixel(vec![TEXCOORD, again], 2)),
            Err(ShaderError::Malformed(
                "an attribute the stage data lists twice"
            ))
        );

        let export = program(&[
            0xF800_0A01,
            0x0000_0000, // exp param0, v0, off, off, off done
            S_ENDPGM,
        ]);
        let out = |row, param| crate::native::VsOutput {
            usage: Usage::Generic,
            row,
            param,
            fp16: false,
            pack2: false,
        };
        let vs = |outputs| NativeProgram {
            io: StageIo::Vs {
                inputs: vec![],
                outputs,
            },
            ..vertex(vec![])
        };
        let io = raise(&export, &vs(vec![out(1, 0), out(2, 1)])).unwrap();
        assert_eq!(
            io.exports[0].target,
            Target::Param {
                usage: Usage::Generic,
                row: 1
            }
        );
        // Bit 7 marks a per-primitive output in the same parameter space.
        for outputs in [vec![out(1, 0), out(2, 0)], vec![out(1, 0), out(2, 0x80)]] {
            assert_eq!(
                raise(&export, &vs(outputs.clone())),
                Err(ShaderError::Malformed(
                    "a parameter the stage data lists twice"
                )),
                "{outputs:?}"
            );
        }
    }

    /// `COMPUTE_PGM_RSRC2` enables one to three thread id VGPRs; the
    /// field's fourth value would load a component past z into v3.
    #[test]
    fn compute_thread_ids_stop_at_z() {
        let compute = |count: u32| NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(crate::native::ComputeIo {
                rsrc2: count << 11,
                threads: [64, 1, 1],
            }),
            code: vec![],
        };
        let reads = |v: u32| program(&[0x7E08_0300 | v, S_ENDPGM]); // v_mov_b32 v4, v<v>
        let id = |count, v| {
            raise(&reads(v), &compute(count))
                .unwrap()
                .inputs
                .iter()
                .map(|i| i.1)
                .collect::<Vec<_>>()
        };
        assert_eq!(id(0, 0), [Input::ThreadIdInGroup(0)]);
        assert_eq!(id(2, 2), [Input::ThreadIdInGroup(2)]);
        // v2 without a z component is not a thread id.
        assert_eq!(id(1, 2), []);
        for v in [0, 3] {
            assert_eq!(
                raise(&reads(v), &compute(3)),
                Err(ShaderError::Malformed("a thread id component past z")),
                "v{v}"
            );
        }
    }

    #[test]
    fn pixel_vgprs_follow_the_input_address() {
        // Linear center (bit 5), then position x and y (bits 8, 9).
        let v = ps_vgprs(1 << 5 | 1 << 8 | 1 << 9);
        assert_eq!(
            v,
            [
                (
                    0,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 0
                    }
                ),
                (
                    1,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 1
                    }
                ),
                (2, Input::Position(0)),
                (3, Input::Position(1)),
            ]
        );
    }

    /// Every `SPI_PS_INPUT_ADDR` bit packs its VGPRs straight after the
    /// last enabled one: all sixteen fill v0-v23 without a gap, and the
    /// register's upper bits, which name no input, load nothing.
    #[test]
    fn every_pixel_input_bit_packs_v0_to_v23() {
        assert!(ps_vgprs(0).is_empty());
        assert!(ps_vgprs(0xFFFF_0000).is_empty());
        let all = ps_vgprs(u32::MAX);
        assert_eq!(all, ps_vgprs(0xFFFF));
        let regs: Vec<u8> = all.iter().map(|v| v.0).collect();
        assert_eq!(regs, (0..24).collect::<Vec<u8>>());
        // The pull model's three VGPRs sit between the centroid pair and
        // the linear sample pair.
        assert_eq!(
            all[6..9].iter().map(|v| v.1).collect::<Vec<_>>(),
            (0..3)
                .map(|index| Input::Barycentric {
                    kind: Barycentric::PerspectivePullModel,
                    index
                })
                .collect::<Vec<_>>()
        );
        // Then the system values, one VGPR each, in bit order.
        assert_eq!(
            all[15..].iter().map(|v| v.1).collect::<Vec<_>>(),
            [
                Input::LineStipple,
                Input::Position(0),
                Input::Position(1),
                Input::Position(2),
                Input::Position(3),
                Input::FrontFace,
                Input::Ancillary,
                Input::SampleCoverage,
                Input::FixedPosition,
            ]
        );
        // Only the last bit: fixed-point position alone, in v0.
        assert_eq!(ps_vgprs(1 << 15), [(0, Input::FixedPosition)]);
    }

    /// The six-bit export target: colours, depth, null, positions, the
    /// primitive and parameters each name a target; the numbers between
    /// them are refused, and a pixel shader has no parameter to export.
    #[test]
    fn export_targets_map_or_are_refused() {
        // exp <t>, v0, off, off, off done
        let export = |t: u32| program(&[0xF800_0801 | t << 4, 0x0000_0000, S_ENDPGM]);
        let target =
            |t, native: &NativeProgram| raise(&export(t), native).map(|io| io.exports[0].target);
        let ps = pixel(vec![], 0);
        for t in 0..=7u8 {
            assert_eq!(target(t.into(), &ps), Ok(Target::Color(t)));
        }
        assert_eq!(target(8, &ps), Ok(Target::Depth));
        assert_eq!(target(9, &ps), Ok(Target::Null));
        let vs = vertex(vec![]);
        for t in 12..=16u8 {
            assert_eq!(target(t.into(), &vs), Ok(Target::Position(t - 12)));
        }
        assert_eq!(target(20, &vs), Ok(Target::Primitive));
        // A vertex shader without outputs still exports param0.
        assert_eq!(target(32, &vs), Ok(Target::Unread));
        assert_eq!(
            target(33, &vs),
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "a parameter export the stage data does not list"
            })
        );
        for t in [10, 11, 17, 18, 19, 21, 31] {
            assert_eq!(
                target(t, &vs),
                Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "an export target"
                }),
                "target {t}"
            );
        }
        for t in [32, 33, 63] {
            assert_eq!(
                target(t, &ps),
                Err(ShaderError::Malformed(
                    "a pixel shader exporting a parameter"
                )),
                "target {t}"
            );
        }
    }
}
