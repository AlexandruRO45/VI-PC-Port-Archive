impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>, numeric_selects: bool) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let dead = crate::sgpr_spill::dead(p);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let wide_lds =
            insts().any(|i| ds_integer_atomic(i.name()).is_some_and(|(_, wide, _)| wide));
        let mut console_lanes = false;
        let mut split = false;
        let mut wants_console_flow = false;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            // Instructions that run (not the spills forwarding leaves dead).
            let live = || {
                p.blocks.iter().enumerate().flat_map(|(b, block)| {
                    let dead = &dead;
                    block
                        .insts
                        .iter()
                        .enumerate()
                        .filter(move |&(i, _)| !dead.contains(&(b, i)))
                        .map(|(_, inst)| inst)
                })
            };
            // Swizzles, DPP and quad derivatives read lanes of the same 32
            // (rows of 16, groups of 8, 32 or 4): the rest may read the
            // other half of a wave64.
            let in_half = |i: &Inst| {
                i.name().starts_with("ds_swizzle")
                    || implicit_derivatives(i.name())
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
            };
            let whole = |i: &Inst| {
                let n = i.name();
                n.contains("permute")
                    || n.starts_with("v_permlane")
                    || n.starts_with("v_mbcnt")
                    // Its derivatives are the quad's.
                    || n == "image_sample"
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || reads_mask_as_data(x.st, i)
                    || numeric_selects && numeric_select(x.st, i)
            };
            let reads_whole = live().any(whole);
            let masks_held = {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                })
            };
            console_lanes = reads_whole || masks_held || live().any(in_half);
            let wave = if x.native.wave64 { 64 } else { 32 };
            let one_wave = cs.threads.iter().product::<u32>() <= wave;
            // On a host without 64-lane waves, a wave64 program whose lanes
            // read only their own half runs as two 32-lane waves: what each
            // lane reads is in its host wave, at its host lane's place in the
            // row. Not with LDS (the console compiler may leave out barriers
            // between the halves) nor whole-wave sections (their reductions
            // become wave operations over the host's wave).
            split = console_lanes
                && x.native.wave64
                && !reads_whole
                && !masks_held
                && s.lds_dwords == 0
                && !x.target.runs(64)
                && x.target.runs(32)
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
            // The rest of such programs need the halves to exchange values
            // (masks read whole, lanes of the other half): the console flow,
            // for a group of one console wave without LDS.
            wants_console_flow = console_lanes
                && !split
                && x.native.wave64
                && !x.target.runs(64)
                && x.target.runs(32)
                && s.lds_dwords == 0
                && cs.threads.iter().product::<u32>() <= 64
                && !insts().any(|i| i.name().contains("permute"))
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
            // On a host that runs the console's wave size: lane reads and
            // writes in a program whose lanes branch differently (see
            // [`CONSOLE_FLOW`]).
            let uniform = |when: Bool| {
                x.st.bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(_)))
            };
            wants_console_flow |= x.target.runs(wave)
                && live().any(|i| matches!(i.name(), "v_readlane_b32" | "v_writelane_b32"))
                && x.st
                    .exits
                    .iter()
                    .any(|e| matches!(*e, LaneExit::Branch { when, .. } if !uniform(when)))
                && !insts().any(|i| i.name().contains("permute"))
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
            if x.console_cf {
                console_lanes = true;
                // In halves only where the host cannot run the console wave.
                split = x.native.wave64 && !x.target.runs(64);
            }
            if split {
                s.wave_size = Some(32);
            } else if console_lanes || one_wave && s.lds_dwords > 0 {
                s.wave_size = Some(wave);
            }
        }
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            wide_lds,
            src0: None,
            console_lanes,
            numeric_selects,
            split,
            console_cf: x.console_cf,
            wants_console_flow,
            console_flat: None,
            xchg: None,
            flat: None,
            no_barrier,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            half_inputs: {
                let hi = half_inputs(&x.native.io, &x.inputs);
                hi
            },
            half_input_spills: half_input_spills(&half_inputs(&x.native.io, &x.inputs), &x.inputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs | HwStage::Ms => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            mask_scc: mask_sccs(x.st, p),
            mask_numbers: mask_numbers(x.st, p),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            one_lane: Default::default(),
            coherent: None,
            ordered: None,
            made: Default::default(),
            code_table: None,
            scratch: None,
            counters: HashMap::new(),
            loop_exits: HashMap::new(),
            loop_bodies: HashMap::new(),
            finals: HashMap::new(),
            abort: None,
            lane_paths: Vec::new(),
            pending_paths: Vec::new(),
            dead,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| if a.includes_base { 0 } else { a.offset & !3 })
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// Private (scratch) memory, `scratch_load_*` and `scratch_store_*`.
    /// Each lane has its own: the address (VGPR, SGPR or none, plus the
    /// offset) is a byte offset into the lane's memory, and the hardware
    /// interleaves the lanes' dwords itself (RDNA 2 ISA, "Scratch
    /// (Private)"; LLVM AMDGPUUsage, "Private Segment"). On the host it is
    /// an array of dwords per lane (`alloca`, as DXC makes local arrays).
    // shortcut: a dword access is taken as aligned, a byte or short one
    // not to cross a dword (console compilers keep private data aligned).
    fn scratch(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds {
            return Err("a scratch load into LDS");
        }
        let name = inst.name();
        let (load, rest) = match name.strip_prefix("scratch_load_") {
            Some(rest) => (true, rest),
            None => (false, name.strip_prefix("scratch_store_").ok_or(name)?),
        };
        let width = scratch_width(rest).ok_or(name)?;
        let (signed, hi) = (rest.starts_with('s'), rest.ends_with("_hi"));
        let guard = self.guard(b, i)?;
        let base = match (addr, saddr) {
            (Some(o), _) => self.vgprs(inst, o, 1)?[0],
            (None, Some(o)) => self.operand(inst, o)?,
            (None, None) => self.s.m.i32(0),
        };
        let (arr, dwords) = match self.scratch {
            Some(a) => a,
            None => {
                let dwords = scratch_bytes(self.x.p).div_ceil(4);
                let ty = self.s.m.array(dwords, self.s.t.i32);
                let a = (self.s.m.alloca(self.s.entry, ty), dwords);
                self.scratch = Some(a);
                a
            }
        };
        let c = self.cur;
        let off = self.s.m.i32(i32::from(*offset) as u32);
        let at = self.s.m.bin(c, Bin::Add, base, off);
        let two = self.s.m.i32(2);
        let first = self.s.m.bin(c, Bin::LShr, at, two);
        // Dword `k` of the access: its element (the first when out of
        // range) and whether it is in range.
        let element = |l: &mut Self, k: u32| -> (V, V) {
            let c = l.cur;
            let k = l.s.m.i32(k);
            let index = l.s.m.bin(c, Bin::Add, first, k);
            let n = l.s.m.i32(dwords);
            let inside = l.s.m.cmp(c, pred::IULT, index, n);
            let zero = l.s.m.i32(0);
            let index = l.s.m.select(c, inside, index, zero);
            (l.s.m.element(c, arr, index), inside)
        };
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        let shift = self.s.m.bin(c, Bin::Shl, byte, three);
        if load {
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a scratch load result");
            };
            let zero = self.s.m.i32(0);
            if width >= 4 {
                for k in 0..width / 4 {
                    let (p, inside) = element(self, k);
                    let x = self.s.m.load(c, p);
                    let x = self.s.m.select(c, inside, x, zero);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let (p, inside) = element(self, 0);
            let w = self.s.m.load(c, p);
            let w = self.s.m.select(c, inside, w, zero);
            let x = self.s.m.bin(c, Bin::LShr, w, shift);
            let left = self.s.m.i32(32 - 8 * width);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let mut x = self
                .s
                .m
                .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            if rest.contains("d16") {
                // Into one half of the VGPR (extended to 16 bits); the
                // other half keeps its bits.
                let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    _ => self.s.m.i32(0),
                };
                let low = self.s.m.i32(0xFFFF);
                let h = self.s.m.bin(c, Bin::And, x, low);
                let (h, keep) = if hi {
                    let sixteen = self.s.m.i32(16);
                    (self.s.m.bin(c, Bin::Shl, h, sixteen), low)
                } else {
                    (h, self.s.m.i32(0xFFFF_0000))
                };
                let o = self.s.m.bin(c, Bin::And, old, keep);
                x = self.s.m.bin(c, Bin::Or, o, h);
            }
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let data = data.as_ref().ok_or("scratch store data")?;
        let vals = self.vgprs(inst, data, width.div_ceil(4) as u8)?;
        for (k, mut x) in vals.into_iter().enumerate() {
            let (p, inside) = element(self, k as u32);
            let g = match guard {
                Some(g) => self.s.m.bin(self.cur, Bin::And, g, inside),
                None => inside,
            };
            let at = self.begin_guarded(Some(g));
            let c = self.cur;
            if width < 4 {
                // The byte or short within its dword (`_d16_hi`: from the
                // high half of the VGPR).
                if hi {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let mask = self.s.m.i32((1 << (8 * width)) - 1);
                let mask = self.s.m.bin(c, Bin::Shl, mask, shift);
                let x = self.s.m.bin(c, Bin::Shl, x, shift);
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, mask, ones);
                let old = self.s.m.load(c, p);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let new = self.s.m.bin(c, Bin::Or, old, x);
                self.s.m.store(c, p, new);
            } else {
                self.s.m.store(c, p, x);
            }
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, lds, .. } = &inst.fields else {
            unreachable!()
        };
        if *lds {
            return Err("a global load directly into LDS");
        }
        let name = inst.name();
        if self.x.res.addresses && !self.scalar.contains_key(&(b, i)) {
            return self.flat_memory(inst, b, i);
        }
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global store or atomic without root UAV address support")?;
        let sub = subword_load(rest);
        let n = match rest {
            "dword" => 1,
            "dwordx2" => 2,
            "dwordx3" => 3,
            "dwordx4" => 4,
            "dword_addtid" => return Err("a global load with a console thread ID offset"),
            _ if sub.is_some() => 1,
            _ => return Err(name),
        };
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let Some(&a) = self.scalar.get(&(b, i)) else {
            if sub.is_none() && crate::resource::traces_rays(self.x.p) {
                // shortcut: out of the acceleration structure.
                let z = self.s.m.i32(0);
                for k in 0..n {
                    self.write(inst, Reg::V(reg + k as u8), z, guard);
                }
                return Ok(());
            }
            return Err("an untracked global address");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = self.dynamic_term(&a)? {
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let u = l.s.m.undef(l.s.t.i32);
                    Ok(l.raw_dword(h, [byte, u]))
                }
            }
        };
        if let Some((width, signed, half)) = sub {
            let x = self.read_subword(off, width, signed, dword)?;
            self.write_subword_result(inst, reg, x, half, guard)?;
        } else {
            for k in 0..n {
                let four = self.s.m.i32(4 * k);
                let at = self.s.m.bin(c, Bin::Add, off, four);
                let x = dword(self, at)?;
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if self.dead.contains(&(b, i)) {
                    continue;
                }
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// The instructions of the whole-wave section from `start` to `end`:
    /// the values that leave it are made from what the simulation found
    /// them to be. A value it found no shape for, written by a step that
    /// does not look at other lanes (scalar code the compiler scheduled
    /// into the section, or lane-local vector code), is lowered as outside
    /// a section: the lanes active on entry compute it as they would under
    /// their own EXEC, and only they run on the host.
    fn section(&mut self, b: usize, start: usize, end: usize) -> Result<(), Stop> {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let insts = &self.x.p.blocks[b].insts;
        let inside: HashSet<Value> = insts[start + 1..end]
            .iter()
            .flat_map(|inst| inst.writes.iter().map(|w| w.1))
            .collect();
        let mut need: HashSet<Value> = inside
            .iter()
            .copied()
            .filter(|w| self.section_out.contains(w))
            .collect();
        let mut plain = vec![false; end - start];
        for j in (start + 1..end).rev() {
            let inst = &insts[j];
            let shapes = &self.sections[&b][&start].shapes;
            let wanted = inst
                .writes
                .iter()
                .any(|&(_, w)| need.contains(&w) && !shapes.contains_key(&w));
            if wanted && lane_local(inst) {
                plain[j - start] = true;
                need.extend(
                    inst.reads
                        .iter()
                        .map(|r| r.1)
                        .filter(|r| inside.contains(r)),
                );
            }
        }
        for j in start + 1..end {
            if plain[j - start] {
                self.inst(b, j)?;
            }
            for &(_, w) in &insts[j].writes {
                let shapes = &self.sections[&b][&start].shapes;
                let mut shape = need.contains(&w).then(|| shapes.get(&w).copied()).flatten();
                // Lanes that do not agree, of which one is read after.
                if shape.is_none()
                    && need.contains(&w)
                    && let Some(lane) = self.lane_read(w, b, start, end)
                    && let Some(sh) = self.sections[&b][&start].lane_shape(w, lane)
                {
                    shape = Some(sh);
                    self.one_lane.insert(w);
                }
                let val = match (shape.map(|sh| self.shape(sh)), plain[j - start]) {
                    (Some(Ok(x)), _) => Val::I32(x),
                    (_, true) => continue,
                    (Some(Err(why)), false) => Val::Missing(why),
                    (None, false) => Val::Missing(UNKNOWN),
                };
                self.vals.insert(w, val);
            }
        }
        Ok(())
    }

    /// The one lane of `w` read outside the section from `start` to `end` of
    /// block `b`, if nothing else reads it there.
    fn lane_read(&self, w: Value, b: usize, start: usize, end: usize) -> Option<u32> {
        let mut found = None;
        for (ob, block) in self.x.p.blocks.iter().enumerate() {
            if block.phis.iter().any(|phi| phi.args.contains(&w)) {
                return None;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if ob == b && i > start && i < end || !inst.reads.iter().any(|r| r.1 == w) {
                    continue;
                }
                let Fields::Valu(v) = &inst.fields else {
                    return None;
                };
                let lane = match (inst.name(), v.src.get(1)) {
                    ("v_readlane_b32", Some(&Operand::Const(c))) => crate::resource::inline(c)?,
                    _ => return None,
                };
                if found.is_some_and(|f| f != lane) {
                    return None;
                }
                found = Some(lane);
            }
        }
        found
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::ActiveF64(lo, hi, mask, is_hi) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let lo_raw = self.i32_of(lo)?;
                let hi_raw = self.i32_of(hi)?;
                let z = self.s.m.i32(0);
                let lo_v = self.s.m.select(c, on, lo_raw, z);
                let hi_v = self.s.m.select(c, on, hi_raw, z);
                self.s.flags |= flags::WAVE_OPS | flags::DOUBLES;
                let f64t = self.s.m.double();
                let d = self.s.op(
                    c,
                    "makeDouble.f64",
                    Attr::ReadNone,
                    f64t,
                    101,
                    &[lo_v, hi_v],
                );
                let (k0, u0) = (self.s.m.i8(0), self.s.m.i8(0));
                let sum = self.s.op(
                    c,
                    "waveActiveOp.f64",
                    Attr::NoUnwind,
                    f64t,
                    119,
                    &[d, k0, u0],
                );
                let i32t = self.s.t.i32;
                let pair = self
                    .s
                    .m
                    .named_struct("dx.types.splitdouble", vec![i32t, i32t]);
                let split = self
                    .s
                    .op(c, "splitDouble.f64", Attr::ReadNone, pair, 102, &[sum]);
                return Ok(self.s.m.extract(c, split, u32::from(is_hi)));
            }
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        if self.split {
            return Err("a wave reduction in a wave64 program split in halves");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let f16 = op == Op::AddF16;
        let float = matches!(op, Op::AddF | Op::AddF16 | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if f16 {
            self.s
                .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
        } else if float {
            self.as_float(x)
        } else {
            x
        };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF | Op::AddF16, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF | Op::AddF16 => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        Ok(if f16 {
            self.s
                .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[r])
        } else if float {
            self.as_int(r)
        } else {
            r
        })
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        let user_data = self.lower_ir()?;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let stride_table = self.s.stride_table.is_some();
        let cmp_samplers = self.s.cmp_samplers;
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            annotations: std::mem::take(&mut self.s.annotations),
            stride_table,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
        })
    }

    fn lower_ir(&mut self) -> Result<Vec<UserData>, Stop> {
        if self.wants_console_flow && !self.x.console_cf && !self.x.per_lane_only {
            return Err(unsupported(0, CONSOLE_FLOW));
        }
        if let Some(n) = self.s.wave_size
            && !self.x.target.runs(n)
        {
            // shortcut: no form of these lanes on the host's wave size yet
            // (two 32-lane waves standing for one console wave64 need the
            // halves to exchange masks and lane values).
            return Err(unsupported(
                0,
                match (n, self.console_lanes) {
                    (64, true) => "console wave64 lanes on a host without 64-lane waves",
                    (64, false) => {
                        "a wave64 group sharing LDS without barriers on a host without 64-lane waves"
                    }
                    _ => "a console wave size the host does not run",
                },
            ));
        }
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.find_loop_exits();
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        // Dispatches count threads too: the last group's others do not run.
        if self.x.res.addresses
            && self.s.mesh.is_none()
            && let Some([x, ..]) = self.s.threads
        {
            let e = self.s.entry;
            let zero = self.s.m.i32(0);
            let i32t = self.s.t.i32;
            let id = self
                .s
                .op(e, "threadId.i32", Attr::ReadNone, i32t, 93, &[zero]);
            let start = self.host_data(1, RootType::GROUPS, 0);
            let group = self.s.m.i32(x);
            let start = self.s.m.bin(e, Bin::Mul, start, group);
            let id = self.s.m.bin(e, Bin::Add, id, start);
            let most = self.host_data(2, RootType::THREADS, x);
            let runs = self.s.m.cmp(e, pred::IULT, id, most);
            let stop = self.abort_block();
            let go = self.s.m.new_block();
            self.s.m.terminate(e, Term::CondBr(runs, go, stop));
            self.cur = go;
            self.block[order[0]] = Some(go);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            // Compute programs only: their host waves are the console's, so a
            // wave's value is the console wave's.
            if self.mask_data == MaskData::Console {
                self.leave_loops(b);
                if !self.loop_exits.contains_key(&b) {
                    self.join_scalars(b);
                }
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                self.inst(b, i)?;
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    self.section(b, i, end)?;
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        if self.s.pixel {
            settle_position_mode(&mut self.s.inputs);
            settle_row_modes(&mut self.s.inputs, true).map_err(|w| unsupported(0, w))?;
        } else {
            settle_row_modes(&mut self.s.outputs, false).map_err(|w| unsupported(0, w))?;
        }
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        Ok(user_data)
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        // RDNA2 POPS (ROV) pixel shaders read a hardware wave-ordering SGPR
        // (bit 31 = overlap flag) and spin on `src_pops_exiting_wave_id` (239)
        // when bit 31 is set. Zeroing that SGPR takes the non-overlapping fast
        // path straight past the `s_sleep` spin loop into the critical section.
        let uses_pops = self.s.pixel
            && self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|inst| {
                matches!(
                    &inst.fields,
                    Fields::Sop { src, .. } if src.contains(&Operand::Const(239))
                )
            });
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // Compute: past the user SGPRs the hardware sets only
                    // the group's, which are known. Others no one set (a
                    // path that does not need them); they read as zero, as
                    // VGPRs do. In a pixel program the hardware's come
                    // within the first few after the 32 user SGPRs.
                    None if n <= 105
                        && (self.s.threads.is_some() && self.s.mesh.is_none()
                            || self.s.pixel && (n >= 40 || uses_pops)) =>
                    {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => {
                        if std::env::var_os("VICTORY_WHY").is_some() {
                            eprintln!("input {reg:?}");
                        }
                        Val::Missing("a hardware SGPR input")
                    }
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(0) if self.x.res.addresses => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let first = self.host_data(1, RootType::GROUPS, 0);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Add, g, first));
            }
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        if let Some(&(val, h)) = self.finals.get(&v)
            && !self.loop_bodies[&h].contains(&self.cur_ir)
        {
            if is_mask && !self.mask_ok {
                return Err(LANE_MASK);
            }
            return Ok(val);
        }
        if is_mask && !self.mask_ok && std::env::var_os("VICTORY_WHY").is_some() {
            let def = self.def_block.get(&v).map(|&b| {
                (
                    b,
                    self.x.p.blocks[b]
                        .insts
                        .iter()
                        .find(|i| i.writes.iter().any(|w| w.1 == v))
                        .map(|i| (format!("{:#x}", i.machine.offset), i.name().to_owned())),
                    self.x.p.blocks[b]
                        .phis
                        .iter()
                        .find(|p| p.value == v)
                        .map(|p| (p.reg, p.args.clone())),
                )
            });
            eprintln!(
                "mask data {v:?} {:?} def {def:?} cur {} high {:?} val {:?}",
                self.x.st.masks.get(&v),
                self.cur_ir,
                self.x.st.highs.get(&v),
                self.vals.get(&v)
            );
        }
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(_)) if self.mask_ok && self.mask_scc.contains_key(&v) => {
                let lo = self.mask_scc[&v];
                return self.mask_any(lo).map(Val::Bit);
            }
            // A mask's number (from a set operation) only for the uses
            // `MaskData` allows.
            Some(&Val::I32(_)) if is_mask && !self.mask_ok && !self.phi_at.contains_key(&v) => {
                return Err(LANE_MASK);
            }
            // shortcut: descriptors a program that traces rays reads as data
            // (a hit's material, say): no ray hits anything.
            Some(&Val::Missing(
                "a descriptor's address or format read as data"
                | "a descriptor read only as data"
                | "a constant buffer's descriptor read as data",
            )) if crate::resource::traces_rays(self.x.p) => {
                return Ok(Val::I32(self.s.m.i32(0)));
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    /// The computed byte offset `a` adds to its parameter's address: its
    /// computed term, less the parameter's address when that term is a
    /// pointer that holds it (see [`Address::includes_base`]).
    fn dynamic_term(&mut self, a: &Address) -> Result<Option<V>, Why> {
        let Some(d) = a.dynamic else { return Ok(None) };
        let d = self.i32_of(d)?;
        if !a.includes_base {
            return Ok(Some(d));
        }
        if a.param == CODE {
            return Ok(Some(d));
        }
        let base = match self.user(a.param, 0) {
            Val::I32(x) => x,
            _ => return Err("an advanced pointer into a root descriptor"),
        };
        Ok(Some(self.s.m.bin(self.cur, Bin::Sub, d, base)))
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            // Its address, for programs that use memory by address.
            RootType::Descriptor if self.x.res.addresses => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// The heap index of the host's first view of memory by address: the
    /// map, then the memory it names.
    fn memory_base(&mut self) -> V {
        self.host_data(0, RootType::MEMORY, 0)
    }

    /// User data the host adds behind the title's: the `nth` such dword.
    fn host_data(&mut self, nth: u32, root: RootType, dword: u32) -> V {
        let layout = &self.x.native.layout;
        let index = layout
            .iter()
            .map(|p| u32::from(p.user_data) + p.sgprs.max(1))
            .max()
            .unwrap_or(0)
            + nth;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: u8::MAX,
                root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// The host's view of the title memory at `hi:lo` (40 bits) and the
    /// byte in it. The map holds two dwords per MiB: which view, and the
    /// byte in it of that MiB's start.
    fn memory(&mut self, lo: V, hi: V, nonuniform: bool) -> (V, V) {
        let c = self.cur;
        let first = self.memory_base();
        let map = self.s.handle_at(c, first, false, false, (11, 0));
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (k255, k12, k20, k3, low) = (
            k(self, 255),
            k(self, 12),
            k(self, 20),
            k(self, 3),
            k(self, 0xF_FFFF),
        );
        let h = self.s.m.bin(c, Bin::And, hi, k255);
        let h = self.s.m.bin(c, Bin::Shl, h, k12);
        let l = self.s.m.bin(c, Bin::LShr, lo, k20);
        let mib = self.s.m.bin(c, Bin::Or, h, l);
        let at = self.s.m.bin(c, Bin::Shl, mib, k3);
        let (mask, align, u) = (
            self.s.m.i8(3),
            self.s.m.i32(4),
            self.s.m.undef(self.s.t.i32),
        );
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[map, at, u, mask, align],
        );
        let view = self.s.m.extract(c, r, 0);
        let view = self.as_int(view);
        let start = self.s.m.extract(c, r, 1);
        let start = self.as_int(start);
        // Never past the views of ours, whatever the map holds.
        let (i32t, most) = (
            self.s.t.i32,
            self.s.m.i32(super::VIEW_SLOTS - super::OWN_SLOTS - 1),
        );
        let view = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[view, most]);
        let index = self.s.m.bin(c, Bin::Add, first, view);
        // Builders read what other groups of the dispatch wrote.
        let p = self.x.p;
        let shared = *self.coherent.get_or_insert_with(|| {
            p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                let n = i.name();
                n.starts_with("global_store") || n.starts_with("global_atomic")
            })
        });
        let shared = if shared { 0x4000 } else { 0 };
        let handle = self
            .s
            .handle_at(c, index, false, nonuniform, (11 | 0x1000 | shared, 0));
        let within = self.s.m.bin(c, Bin::And, lo, low);
        (handle, self.s.m.bin(c, Bin::Add, start, within))
    }

    /// `hi:lo` plus `x` (sign-extended).
    fn add_address(&mut self, (lo, hi): (V, V), x: V, signed: bool) -> (V, V) {
        let c = self.cur;
        let (r, carry) = self.add_sub_carry(c, true, lo, x, None);
        let up = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let mut hi = self.s.m.bin(c, Bin::Add, hi, up);
        if signed {
            let k31 = self.s.m.i32(31);
            let s = self.s.m.bin(c, Bin::AShr, x, k31);
            hi = self.s.m.bin(c, Bin::Add, hi, s);
        }
        (r, hi)
    }

    /// `count` dwords of title memory at `hi:lo`.
    fn memory_dwords(&mut self, at: (V, V), count: u32, nonuniform: bool) -> Vec<V> {
        let (h, off) = self.memory(at.0, at.1, nonuniform);
        let mut out = Vec::new();
        let mut done = 0;
        while done < count {
            let n = (count - done).min(4);
            let c = self.cur;
            let d = self.s.m.i32(4 * done);
            let a = self.s.m.bin(c, Bin::Add, off, d);
            let (mask, align, u) = (
                self.s.m.i8((1u8 << n) - 1),
                self.s.m.i32(4),
                self.s.m.undef(self.s.t.i32),
            );
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[h, a, u, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, k);
                let x = self.as_int(x);
                out.push(x);
            }
            done += n;
        }
        out
    }

    /// The address a global memory instruction gives each lane.
    fn flat_at(&mut self, inst: &Inst) -> Result<(V, V), Why> {
        let Fields::Flat {
            addr,
            saddr,
            offset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = *addr else {
            return Err("a global address");
        };
        let lo = inst.read(Reg::V(reg)).ok_or("a global address")?;
        let lo = self.i32_of(lo)?;
        let mut at = match *saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let (a, b) = (
                    inst.read(Reg::S(code)).ok_or("a global address")?,
                    inst.read(Reg::S(code + 1)).ok_or("a global address")?,
                );
                let base = (self.i32_of(a)?, self.i32_of(b)?);
                self.add_address(base, lo, false)
            }
            _ => {
                let hi = inst.read(Reg::V(reg + 1)).ok_or("a global address")?;
                (lo, self.i32_of(hi)?)
            }
        };
        if *offset != 0 {
            let x = self.s.m.i32(i32::from(*offset) as u32);
            at = self.add_address(at, x, true);
        }
        Ok(at)
    }

    /// A global memory access by address.
    fn flat_memory(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, data, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let at = self.flat_at(inst)?;
        let dwords = |s: &str| match s {
            "dword" => Some(1u32),
            "dwordx2" => Some(2),
            "dwordx3" => Some(3),
            "dwordx4" => Some(4),
            _ => None,
        };
        if let Some(rest) = name.strip_prefix("global_load_") {
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a global load result");
            };
            if let Some((width, signed, half)) = subword_load(rest) {
                let (h, off) = self.memory(at.0, at.1, true);
                let x = self.read_subword(off, width, signed, |l, byte| {
                    let u = l.s.m.undef(l.s.t.i32);
                    Ok(l.raw_dword(h, [byte, u]))
                })?;
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            let n = dwords(rest).ok_or(name)?;
            let xs = self.memory_dwords(at, n, true);
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let data = data.as_ref().ok_or("global data")?;
        if let Some(n) = name.strip_prefix("global_store_").and_then(dwords) {
            let vals = self.vgprs(inst, data, n as u8)?;
            let place = self.begin_guarded(guard);
            let (h, off) = self.memory(at.0, at.1, true);
            let iu = self.s.m.undef(self.s.t.i32);
            let mut args = vec![h, off, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(self.s.m.i32(4));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                void,
                140,
                &args,
            );
            self.end_guarded(place, &[]);
            return Ok(());
        }
        let op = name.strip_prefix("global_atomic_").ok_or(name)?;
        let code = match op {
            "add" | "sub" => Some(0),
            "and" => Some(1),
            "or" => Some(2),
            "xor" => Some(3),
            "smin" => Some(4),
            "smax" => Some(5),
            "umin" => Some(6),
            "umax" => Some(7),
            "swap" => Some(8),
            "cmpswap" | "fmin" | "fmax" => None,
            _ => return Err(name),
        };
        let vals = self.vgprs(inst, data, if op == "cmpswap" { 2 } else { 1 })?;
        let zero = self.s.m.i32(0);
        let place = self.begin_guarded(guard);
        let (h, off) = self.memory(at.0, at.1, true);
        let iu = self.s.m.undef(self.s.t.i32);
        let coords = [off, iu, iu];
        let r = match (op, code) {
            ("cmpswap", _) => self.buffer_compare_exchange(h, coords, vals[1], vals[0], false),
            // shortcut: no denormal or NaN rules of the console's.
            ("fmin" | "fmax", _) => {
                // Not exchanged where nothing changes: a whole scene's lanes
                // meet at its bounds.
                let head = self.cur;
                let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
                let ret = self.s.t.res_ret;
                let first = self.s.op(
                    head,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    ret,
                    139,
                    &[h, off, iu, mask, align],
                );
                let first = self.s.m.extract(head, first, 0);
                let first = self.as_int(first);
                let retry = self.s.m.new_block();
                let change = self.s.m.new_block();
                let done = self.s.m.new_block();
                self.s.m.terminate(head, Term::Br(retry));
                self.cur = retry;
                let i32t = self.s.t.i32;
                let expected = self.s.m.phi(retry, i32t);
                let (a, b) = (self.as_float(expected), self.as_float(vals[0]));
                let x = self.binary(if op == "fmin" { 36 } else { 35 }, a, b);
                let x = self.as_int(x);
                let same = self.s.m.cmp(retry, pred::EQ, x, expected);
                self.s.m.terminate(retry, Term::CondBr(same, done, change));
                self.cur = change;
                let old = self.buffer_compare_exchange(h, coords, expected, x, false);
                let ok = self.s.m.cmp(change, pred::EQ, old, expected);
                self.s
                    .m
                    .set_phi(expected, vec![(first, head), (old, change)]);
                self.s.m.terminate(change, Term::CondBr(ok, done, retry));
                self.cur = done;
                expected
            }
            (_, Some(code)) => {
                let code = self.s.m.i32(code);
                let data = if op == "sub" {
                    self.s.m.bin(self.cur, Bin::Sub, zero, vals[0])
                } else {
                    vals[0]
                };
                let i32t = self.s.t.i32;
                self.s.op(
                    self.cur,
                    "atomicBinOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    78,
                    &[h, code, off, iu, iu, data],
                )
            }
            _ => return Err(name),
        };
        let r = self.end_guarded(place, &[(r, zero)])[0];
        if let Some(Operand::Vector { reg, .. }) = *dst {
            self.write(inst, Reg::V(reg), r, guard);
        }
        Ok(())
    }

    /// `image_bvh64_intersect_ray`: the ray `words` (node pointer, extent,
    /// origin, direction, inverse direction) against a node of the
    /// acceleration structure, as the console's ray accelerator answers.
    ///
    /// A node pointer is an address over 8 with the node's kind in its low
    /// bits: 0 to 3 a triangle of a node of 5 vertices, 4 a box of 4
    /// children with 16-bit floats, 5 with 32-bit ones (4 child pointers,
    /// then each child's minimum and maximum); the rest is the title's own.
    /// A box gives its children hit, nearest first, then -1; a triangle the
    /// distance and barycentrics i and j, each over the second value
    /// (negative from behind), as the title divides them.
    // shortcut: no watertight edges, no box growing.
    fn intersect(&mut self, words: &[V; 12], guard: Option<V>) -> [V; 4] {
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let kf = |l: &mut Self, x: f32| l.s.m.f32(x.to_bits());
        let [lo, hi, rest @ ..] = *words;
        let f: Vec<V> = rest.iter().map(|&x| self.as_float(x)).collect();
        let extent = f[0];
        let (o, d, inv) = ([f[1], f[2], f[3]], [f[4], f[5], f[6]], [f[7], f[8], f[9]]);
        let c = self.cur;
        let (k3, k7, k29, not7, ones) = (
            k(self, 3),
            k(self, 7),
            k(self, 29),
            k(self, !7),
            k(self, u32::MAX),
        );
        let kind = self.s.m.bin(c, Bin::And, lo, k7);
        let base = self.s.m.bin(c, Bin::And, lo, not7);
        let alo = self.s.m.bin(c, Bin::Shl, base, k3);
        let ahi = self.s.m.bin(c, Bin::Shl, hi, k3);
        let up = self.s.m.bin(c, Bin::LShr, lo, k29);
        let ahi = self.s.m.bin(c, Bin::Or, ahi, up);
        let (k4, k5, k6) = (k(self, 4), k(self, 5), k(self, 6));
        let and = |l: &mut Self, a: V, b: V| {
            let c = l.cur;
            l.s.m.bin(c, Bin::And, a, b)
        };
        let boxes = self.s.m.bin(c, Bin::And, kind, k6);
        let is_box = self.s.m.cmp(c, pred::EQ, boxes, k4);
        let is_tri = self.s.m.cmp(c, pred::IULT, kind, k4);
        let (is_box, is_tri) = match guard {
            Some(g) => (and(self, is_box, g), and(self, is_tri, g)),
            None => (is_box, is_tri),
        };

        // Boxes.
        let place = self.begin_guarded(Some(is_box));
        let node = self.memory_dwords((alo, ahi), 28, true);
        let c = self.cur;
        let wide = self.s.m.cmp(c, pred::EQ, kind, k5);
        let (k16, kffff) = (k(self, 16), k(self, 0xFFFF));
        let f32t = self.s.t.f32;
        let mut bounds = Vec::new();
        for n in 0..24usize {
            let full = self.as_float(node[4 + n]);
            let w = node[4 + n / 2];
            let h = if n % 2 == 1 {
                self.s.m.bin(c, Bin::LShr, w, k16)
            } else {
                w
            };
            let h = self.s.m.bin(c, Bin::And, h, kffff);
            let half = self
                .s
                .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
            bounds.push(self.s.m.select(c, wide, full, half));
        }
        let (zero, far) = (kf(self, 0.0), kf(self, f32::INFINITY));
        let mut hits: Vec<(V, V)> = Vec::new();
        for child in 0..4usize {
            let (mut near_all, mut far_all) = (zero, extent);
            for axis in 0..3usize {
                let side = |l: &mut Self, which: usize| {
                    let x = bounds[6 * child + 3 * which + axis];
                    let x = l.s.m.bin(c, Bin::Sub, x, o[axis]);
                    l.s.m.bin(c, Bin::Mul, x, inv[axis])
                };
                let (a, b) = (side(self, 0), side(self, 1));
                let (n, x) = (self.binary(36, a, b), self.binary(35, a, b));
                near_all = self.binary(35, near_all, n);
                far_all = self.binary(36, far_all, x);
            }
            let pointer = node[child];
            let there = self.s.m.cmp(c, pred::NE, pointer, ones);
            let meets = self.s.m.cmp(c, pred::OLE, near_all, far_all);
            let hit = self.s.m.bin(c, Bin::And, there, meets);
            let p = self.s.m.select(c, hit, pointer, ones);
            let t = self.s.m.select(c, hit, near_all, far);
            hits.push((p, t));
        }
        for (a, b) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
            let ((pa, ta), (pb, tb)) = (hits[a], hits[b]);
            // Misses last, also behind hits infinitely far.
            let later = self.s.m.cmp(c, pred::OGT, ta, tb);
            let (pn, pf) = (
                self.s.m.select(c, later, pb, pa),
                self.s.m.select(c, later, pa, pb),
            );
            let (tn, tf) = (
                self.s.m.select(c, later, tb, ta),
                self.s.m.select(c, later, ta, tb),
            );
            hits[a] = (pn, tn);
            hits[b] = (pf, tf);
        }
        let found: Vec<(V, V)> = hits.iter().map(|h| (h.0, ones)).collect();
        let r = self.end_guarded(place, &found);

        // Triangles.
        let place = self.begin_guarded(Some(is_tri));
        let node = self.memory_dwords((alo, ahi), 16, true);
        let c = self.cur;
        let vertex = |l: &mut Self, v: usize, axis: usize| l.as_float(node[3 * v + axis]);
        // The vertices of each of the node's triangles.
        const VERTICES: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];
        let is: Vec<V> = (0..3u32)
            .map(|n| {
                let n = k(self, n);
                self.s.m.cmp(c, pred::EQ, kind, n)
            })
            .collect();
        let mut v = [[zero; 3]; 3];
        for (corner, row) in v.iter_mut().enumerate() {
            for (axis, x) in row.iter_mut().enumerate() {
                let mut at = vertex(self, VERTICES[3][corner], axis);
                for n in (0..3).rev() {
                    let other = vertex(self, VERTICES[n][corner], axis);
                    at = self.s.m.select(c, is[n], other, at);
                }
                *x = at;
            }
        }
        let sub = |l: &mut Self, a: [V; 3], b: [V; 3]| {
            [0, 1, 2].map(|i| l.s.m.bin(c, Bin::Sub, a[i], b[i]))
        };
        let cross = |l: &mut Self, a: [V; 3], b: [V; 3]| {
            [(1, 2), (2, 0), (0, 1)].map(|(i, j)| {
                let x = l.s.m.bin(c, Bin::Mul, a[i], b[j]);
                let y = l.s.m.bin(c, Bin::Mul, a[j], b[i]);
                l.s.m.bin(c, Bin::Sub, x, y)
            })
        };
        let dot = |l: &mut Self, a: [V; 3], b: [V; 3]| {
            let x = l.s.m.bin(c, Bin::Mul, a[0], b[0]);
            let x = l.fmad(a[1], b[1], x);
            l.fmad(a[2], b[2], x)
        };
        let (e1, e2) = (sub(self, v[1], v[0]), sub(self, v[2], v[0]));
        let p = cross(self, d, e2);
        let det = dot(self, e1, p);
        let tv = sub(self, o, v[0]);
        let b1 = dot(self, tv, p);
        let q = cross(self, tv, e1);
        let b2 = dot(self, d, q);
        let t = dot(self, e2, q);
        let rest = self.s.m.bin(c, Bin::Sub, det, b1);
        let b0 = self.s.m.bin(c, Bin::Sub, rest, b2);
        // All on the side of the determinant.
        let back = self.s.m.cmp(c, pred::OLT, det, zero);
        let mut inside = self.s.m.cmp(c, pred::ONE, det, zero);
        for x in [b0, b1, b2, t] {
            let neg = self.s.m.bin(c, Bin::Sub, zero, x);
            let x = self.s.m.select(c, back, neg, x);
            let ok = self.s.m.cmp(c, pred::OGE, x, zero);
            inside = self.s.m.bin(c, Bin::And, inside, ok);
        }
        // Which barycentrics i and j are: 8 bits per triangle.
        let shift = self.s.m.bin(c, Bin::Shl, kind, k3);
        let id = self.s.m.bin(c, Bin::LShr, node[15], shift);
        let (one, two) = (k(self, 1), k(self, 2));
        let pick = |l: &mut Self, id: V| {
            let n = l.s.m.bin(c, Bin::And, id, k3);
            let first = l.s.m.cmp(c, pred::EQ, n, one);
            let second = l.s.m.cmp(c, pred::EQ, n, two);
            let x = l.s.m.select(c, second, b2, b0);
            l.s.m.select(c, first, b1, x)
        };
        let i = pick(self, id);
        let id = self.s.m.bin(c, Bin::LShr, id, two);
        let j = pick(self, id);
        let unit = kf(self, 1.0);
        let out = [(t, far), (det, unit), (i, zero), (j, zero)].map(|(x, miss)| {
            let x = self.s.m.select(c, inside, x, miss);
            self.as_int(x)
        });
        let found: Vec<(V, V)> = out
            .iter()
            .zip(&r)
            .map(|(&x, &before)| (x, before))
            .collect();
        let r = self.end_guarded(place, &found);
        [r[0], r[1], r[2], r[3]]
    }

    /// The host's view of a buffer whose descriptor the code makes, the byte
    /// its base is at, and its stride.
    fn made_buffer(&mut self, inst: &Inst, rsrc: &Operand) -> Result<(V, V, V), Why> {
        let Operand::Scalar { code, .. } = *rsrc else {
            return Err("a buffer descriptor");
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(code + k)).ok_or("a buffer descriptor")?;
            l.i32_of(v)
        };
        let (w0, w1) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let (k16, k3fff) = (self.s.m.i32(16), self.s.m.i32(0x3FFF));
        let s = self.s.m.bin(c, Bin::LShr, w1, k16);
        let stride = self.s.m.bin(c, Bin::And, s, k3fff);
        let (h, base) = self.memory(w0, w1, false);
        Ok((h, base, stride))
    }

    /// Both dwords `(lo, hi)` of a lane mask's ballot, taken in the block
    /// that writes the mask (see [`MaskData`]).
    fn mask_ballot(&mut self, lo: Value) -> Result<(V, V), Why> {
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if self.console_cf {
            // Every lane runs every block: the mask's bits are the lanes'
            // own, both halves' words exchanged.
            if let Some(&xy) = self.ballots.get(&lo) {
                return Ok(xy);
            }
            let at = match self.def_block.get(&lo) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
            };
            let bit = self.boolean(m, at)?;
            let xy = self.console_ballot(at, bit);
            self.ballots.insert(lo, xy);
            return Ok(xy);
        }
        if self.split {
            return Err("a lane mask read whole in a wave64 program split in halves");
        }
        if !self.x.st.active.contains(&lo) {
            if std::env::var_os("VICTORY_WHY").is_some() {
                let def = self.def_block.get(&lo).map(|&b| {
                    (
                        b,
                        self.x.p.blocks[b]
                            .insts
                            .iter()
                            .find(|i| i.writes.iter().any(|w| w.1 == lo))
                            .map(|i| (i.machine.offset, i.name().to_owned())),
                        self.x.p.blocks[b]
                            .phis
                            .iter()
                            .find(|p| p.value == lo)
                            .map(|p| (p.reg, p.args.clone())),
                    )
                });
                eprintln!("mask {lo:?} {m:?} def {def:?} cur {}", self.cur_ir);
            }
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        if let Some(&xy) = self.ballots.get(&lo) {
            return Ok(xy);
        }
        let at = match self.def_block.get(&lo) {
            // Program inputs (EXEC on entry).
            None => self.s.entry,
            Some(&b) if b == self.cur_ir => self.cur,
            Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let bit = self.boolean(m, at);
        self.balloting.remove(&lo);
        let bit = bit?;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
        self.ballots.insert(lo, xy);
        Ok(xy)
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let (x, y) = self.mask_ballot(lo)?;
        Ok(if high { y } else { x })
    }

    /// Whether some lane of the mask with low dword `lo` is set, as data
    /// (SCC of the operation writing it): its ballot is not zero.
    fn mask_any(&mut self, lo: Value) -> Result<V, Why> {
        // When `lo` is the new EXEC written in block `b` (e.g. `s_*_wrexec_*` or
        // `s_*_saveexec_*`) and `b` exits via `s_cbranch_execz`/`execnz` without
        // rewriting EXEC, a block entered solely through one branch edge knows
        // whether `lo != 0` without balloting lanes from elsewhere.
        if let Some(&b) = self.def_block.get(&lo) {
            let blk = &self.x.p.blocks[b];
            if let crate::ir::Exit::Branch { taken, not_taken } = blk.exit
                && taken != not_taken
                && self.x.p.blocks[self.cur_ir].preds.as_slice() == [b]
                && let Some((idx, def)) = blk
                    .insts
                    .iter()
                    .enumerate()
                    .find(|(_, inst)| inst.writes.iter().any(|&(_, w)| w == lo))
                && (def.write(Reg::S(EXEC)) == Some(lo) || def.name().contains("_wrexec_"))
                && !blk.insts[idx + 1..]
                    .iter()
                    .any(|inst| inst.write(Reg::S(EXEC)).is_some())
                && let Some(term) = blk.insts.last()
            {
                let nonzero = match term.name() {
                    "s_cbranch_execnz" => Some(self.cur_ir == taken),
                    "s_cbranch_execz" => Some(self.cur_ir == not_taken),
                    _ => None,
                };
                if let Some(bit) = nonzero {
                    return Ok(self.s.m.int_const(self.s.t.i1, i64::from(bit)));
                }
            }
        }
        let x = self.mask_dword(lo)?;
        let (_, y) = self.ballots[&lo];
        let c = self.cur;
        let any = if self.x.p.wave64 {
            self.s.m.bin(c, Bin::Or, x, y)
        } else {
            x
        };
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, any, z))
    }

    /// For a wave32 program under [`MaskData::Host`], a 32-bit scalar operand
    /// holding a tracked mask still represents a host-wave ballot, so its
    /// `(lo, hi)` pair must be read to preserve lanes 32-63 on wave64 hosts.
    fn host_mask_pair(&mut self, inst: &Inst, op: &Operand) -> Result<Option<(V, V)>, Why> {
        if self.mask_data != MaskData::Host || self.x.native.wave64 {
            return Ok(None);
        }
        let Operand::Scalar { code, .. } = *op else {
            return Ok(None);
        };
        if code == NULL {
            return Ok(None);
        }
        let Some(v) = inst.read(Reg::S(code)) else {
            return Ok(None);
        };
        if matches!(self.vals.get(&v), Some(Val::I32(_) | Val::Bit(_)))
            || !self.x.st.masks.contains_key(&v)
        {
            return Ok(None);
        }
        self.mask_ballot(v).map(Some)
    }

    fn is_zero_operand(op: &Operand) -> bool {
        matches!(op, Operand::Scalar { code: NULL, .. } | Operand::Literal(0))
            || matches!(op, Operand::Const(c) if crate::resource::inline(*c) == Some(0))
    }

    fn bcnt1_64(&mut self, c: usize, a: V, b: V) -> V {
        let i32t = self.s.t.i32;
        let na = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
        let nb = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
        self.s.m.bin(c, Bin::Add, na, nb)
    }

    fn ff1_64(&mut self, c: usize, a: V, b: V) -> V {
        // The lowest set bit; -1 for none (as DXIL's first bit low).
        let i32t = self.s.t.i32;
        let fa = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
        let fb = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
        let none = self.s.m.cmp(c, pred::EQ, b, z);
        let fb = self.s.m.select(c, none, fb, fb32);
        let low = self.s.m.cmp(c, pred::NE, a, z);
        self.s.m.select(c, low, fa, fb)
    }

    fn eq_64(&mut self, c: usize, (a0, a1): (V, V), (b0, b1): (V, V)) -> V {
        let e0 = self.s.m.cmp(c, pred::EQ, a0, b0);
        let e1 = self.s.m.cmp(c, pred::EQ, a1, b1);
        self.s.m.bin(c, Bin::And, e0, e1)
    }

    /// `(1 << below) - 1`, all ones from 32 on.
    fn below_mask(&mut self, c: usize, below: V) -> V {
        let (one, k31, k32, ones) = (
            self.s.m.i32(1),
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(u32::MAX),
        );
        let sh = self.s.m.bin(c, Bin::And, below, k31);
        let m = self.s.m.bin(c, Bin::Shl, one, sh);
        let m = self.s.m.bin(c, Bin::Sub, m, one);
        let full = self.s.m.cmp(c, pred::IUGE, below, k32);
        self.s.m.select(c, full, ones, m)
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            // RDNA2 `src_pops_exiting_wave_id`: POPS spin-wait counter (bypassed
            // on host by zeroing the POPS overlap SGPR at PS entry).
            Operand::Const(239) => Ok(self.s.m.i32(0)),
            // SCC as a number.
            Operand::Const(253) => {
                let bit = self.scc_in(inst)?;
                let c = self.cur;
                let (one, zero) = (self.s.m.i32(1), self.s.m.i32(0));
                Ok(self.s.m.select(c, bit, one, zero))
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// An operand read as f16 data: an inline float constant is its f16
    /// value in both halves (so op_sel picks either), not the f32 bits
    /// `operand` gives, whose low half is 0.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => Ok(self.s.m.i32(inline_f16(c))),
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    fn sign_extend(&mut self, x: V, bits: u32) -> V {
        let n = self.s.m.i32(32 - bits);
        let x = self.s.m.bin(self.cur, Bin::Shl, x, n);
        self.s.m.bin(self.cur, Bin::AShr, x, n)
    }

    fn pack_halves(&mut self, lo: V, hi: V) -> V {
        let mask = self.s.m.i32(0xFFFF);
        let k16 = self.s.m.i32(16);
        let lo = self.s.m.bin(self.cur, Bin::And, lo, mask);
        let hi = self.s.m.bin(self.cur, Bin::Shl, hi, k16);
        self.s.m.bin(self.cur, Bin::Or, lo, hi)
    }

    fn int_pick(&mut self, p: u8, a: V, b: V) -> V {
        let choose = self.s.m.cmp(self.cur, p, a, b);
        self.s.m.select(self.cur, choose, a, b)
    }

    fn clamp_int(&mut self, x: V, bits: u32, min: i64, max: i64, signed: bool) -> V {
        let t = self.s.m.int(bits);
        let (min, max) = (self.s.m.int_const(t, min), self.s.m.int_const(t, max));
        let (ge, le) = if signed {
            (pred::SGE, pred::SLE)
        } else {
            (pred::IUGE, pred::IULE)
        };
        let x = self.int_pick(ge, x, min);
        self.int_pick(le, x, max)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        self.bitfield(a, off, width, signed, 32)
    }

    fn bitfield(&mut self, a: V, off: V, width: V, signed: bool, bits: u32) -> V {
        let c = self.cur;
        let ty = self.s.m.int(bits);
        let last = self.s.m.i32(bits - 1);
        let size = self.s.m.i32(bits);
        let zero = self.s.m.i32(0);
        let one = self.s.m.int_const(ty, 1);
        let ones = self.s.m.int_const(ty, -1);
        let off = self.s.m.bin(c, Bin::And, off, last);
        let off = if bits == 64 {
            self.s.m.cast(c, Cast::ZExt, off, ty)
        } else {
            off
        };
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, size);
        let w = self.s.m.bin(c, Bin::And, width, last);
        let count = if bits == 64 {
            self.s.m.cast(c, Cast::ZExt, w, ty)
        } else {
            w
        };
        let mask = self.s.m.bin(c, Bin::Shl, one, count);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than the register (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, size, w);
        let left = self.s.m.bin(c, Bin::And, left, last);
        let left = if bits == 64 {
            self.s.m.cast(c, Cast::ZExt, left, ty)
        } else {
            left
        };
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    ///
    /// Uses `dx.op.isSpecialFloat.f32` (`IsNaN`) instead of `fcmp uno`: some
    /// D3D12 drivers (observed on Intel Iris Xe) fold `fcmp uno x, x` to false
    /// under default shader fast-math, returning the wrong operand when the
    /// third input is NaN.
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let is_nan = |l: &mut Self, x: V| {
            l.s.op(c, "isSpecialFloat.f32", Attr::ReadNone, l.s.t.i1, 8, &[x])
        };
        let nan_a = is_nan(self, a);
        let nan_b = is_nan(self, b);
        let nan_c = is_nan(self, third);
        let ab = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan_c);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    /// `a × b + addend` fused, with `0 × anything = 0` as in D3D9
    /// (`v_fma_legacy_f32`).
    fn fma_legacy(&mut self, a: V, b: V, addend: V) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        let product_zero = self.s.m.bin(c, Bin::Add, zero, addend);
        // shortcut: FMad lets the driver fuse, as for `v_fma_f32`.
        let fused = self.fmad(a, b, addend);
        self.s.m.select(c, z, product_zero, fused)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64; host lane = console
    /// lane, so not in a program split in halves).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        debug_assert!(!self.split || self.console_cf);
        if self.console_cf {
            // The console lane's bit: its half's word, its host lane's bit.
            let (lane, half) = self.console_lane();
            let (k31, one, z) = (self.s.m.i32(31), self.s.m.i32(1), self.s.m.i32(0));
            let word = match hi {
                Some(hi) => {
                    let low = self.s.m.cmp(at, pred::EQ, half, z);
                    self.s.m.select(at, low, lo, hi)
                }
                None => lo,
            };
            let sh = self.s.m.bin(at, Bin::And, lane, k31);
            let t = self.s.m.bin(at, Bin::LShr, word, sh);
            let t = self.s.m.bin(at, Bin::And, t, one);
            return self.s.m.cmp(at, pred::NE, t, z);
        }
        let lane = self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    fn select_condition(&mut self, inst: &Inst, mask: Option<&Operand>) -> Result<V, Why> {
        if let Some(&m) = inst
            .writes
            .first()
            .and_then(|&(_, v)| self.x.st.select_masks.get(&v))
        {
            return self.boolean(m, self.cur);
        }
        let Some(&Operand::Scalar { code, .. }) = mask else {
            return Err(UNTRACKED_SELECT);
        };
        if code == NULL {
            return Ok(self.s.m.bool(false));
        }
        if !self.numeric_selects || !self.console_lanes {
            return Err(UNTRACKED_SELECT);
        }
        // A scalar operation can define either dword independently. The SSA
        // operands at this use give the actual pair, even across a join.
        let lo = self.i32_of(inst.read(Reg::S(code)).ok_or(UNTRACKED_SELECT)?)?;
        let hi = if self.x.p.wave64 {
            Some(self.i32_of(inst.read(Reg::S(code + 1)).ok_or(UNTRACKED_SELECT)?)?)
        } else {
            None
        };
        self.s.flags |= flags::WAVE_OPS;
        Ok(self.lane_bit(self.cur, lo, hi))
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        if self.console_cf {
            // Every lane is here: EXEC is the guard. Lane reads and writes
            // ignore EXEC on the console.
            let inst = &self.x.p.blocks[b].insts[i];
            let ignores = matches!(
                inst.name(),
                "v_readlane_b32" | "v_writelane_b32" | "v_readfirstlane_b32"
            ) || matches!(
                inst.fields,
                Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
            );
            return match self.x.st.execs[b].get(i).copied().flatten() {
                Some(e) if !ignores && e != Bool::TRUE => Ok(Some(self.boolean(e, self.cur)?)),
                _ => Ok(None),
            };
        }
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Atomically updates one location. The first compare also obtains the
    /// initial value, so every observation is atomic, even when it is i64.
    fn atomic_update(&mut self, zero: V, exchange: impl FnOnce(&mut Self, V) -> V) -> V {
        let head = self.cur;
        let retry = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s.m.terminate(head, Term::Br(retry));
        self.cur = retry;
        let ty = self.s.m.type_of(zero);
        let expected = self.s.m.phi(retry, ty);
        let old = exchange(self, expected);
        let success = self.s.m.cmp(retry, pred::EQ, expected, old);
        self.s.m.set_phi(expected, vec![(zero, head), (old, retry)]);
        self.s
            .m
            .terminate(retry, Term::CondBr(success, done, retry));
        self.cur = done;
        old
    }

    /// AMD RDNA 2 ISA sections 12.13–12.14 (including DS_WRAP).
    fn atomic_new_value(&mut self, op: AtomicUpdate, old: V, data: V, data2: V) -> V {
        use AtomicUpdate::*;
        let c = self.cur;
        let ty = self.s.m.type_of(old);
        let zero = self.s.m.int_const(ty, 0);
        let one = self.s.m.int_const(ty, 1);
        match op {
            Inc => {
                let wrap = self.s.m.cmp(c, pred::IUGE, old, data);
                let next = self.s.m.bin(c, Bin::Add, old, one);
                self.s.m.select(c, wrap, zero, next)
            }
            Dec => {
                let at_zero = self.s.m.cmp(c, pred::EQ, old, zero);
                let above = self.s.m.cmp(c, pred::IUGT, old, data);
                let wrap = self.s.m.bin(c, Bin::Or, at_zero, above);
                let next = self.s.m.bin(c, Bin::Sub, old, one);
                self.s.m.select(c, wrap, data, next)
            }
            ReverseSub => self.s.m.bin(c, Bin::Sub, data, old),
            MaskOr => {
                let ones = self.s.m.int_const(ty, -1);
                let keep = self.s.m.bin(c, Bin::Xor, data, ones);
                let kept = self.s.m.bin(c, Bin::And, old, keep);
                self.s.m.bin(c, Bin::Or, kept, data2)
            }
            Wrap => {
                let wrap = self.s.m.cmp(c, pred::IUGE, old, data);
                let low = self.s.m.bin(c, Bin::Sub, old, data);
                let high = self.s.m.bin(c, Bin::Add, old, data2);
                self.s.m.select(c, wrap, low, high)
            }
            ClampSub => {
                let underflow = self.s.m.cmp(c, pred::IULT, old, data);
                let diff = self.s.m.bin(c, Bin::Sub, old, data);
                self.s.m.select(c, underflow, zero, diff)
            }
        }
    }

    fn byte_shift(&mut self, address: V) -> V {
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(self.cur, Bin::And, address, three);
        self.s.m.bin(self.cur, Bin::Shl, byte, three)
    }

    fn write_subword_result(
        &mut self,
        inst: &Inst,
        reg: u8,
        x: V,
        half: Option<u32>,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let reg = Reg::V(reg);
        let x = if let Some(shift) = half {
            let old = self.i32_of(
                inst.read(reg)
                    .ok_or("a D16 load's previous register value")?,
            )?;
            let c = self.cur;
            let (mask, keep, at) = (
                self.s.m.i32(0xFFFF),
                self.s.m.i32(!(0xFFFF << shift)),
                self.s.m.i32(shift),
            );
            let field = self.s.m.bin(c, Bin::And, x, mask);
            let field = self.s.m.bin(c, Bin::Shl, field, at);
            let old = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, old, field)
        } else {
            x
        };
        self.write(inst, reg, x, guard);
        Ok(())
    }

    /// Loads a byte field via aligned dword accesses. A half starting at
    /// byte 3 reads the first byte of the next dword.
    fn read_subword(
        &mut self,
        address: V,
        width: u32,
        signed: bool,
        dword: impl Fn(&mut Self, V) -> Result<V, Why>,
    ) -> Result<V, Why> {
        let c = self.cur;
        let shift = self.byte_shift(address);
        let not3 = self.s.m.i32(!3);
        let aligned = self.s.m.bin(c, Bin::And, address, not3);
        let w = dword(self, aligned)?;
        let low = self.s.m.bin(c, Bin::LShr, w, shift);
        let x = if width == 16 {
            let k24 = self.s.m.i32(24);
            let split = self.s.m.cmp(c, pred::EQ, shift, k24);
            let at = self.begin_guarded(Some(split));
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let high = dword(self, next)?;
            let eight = self.s.m.i32(8);
            let high = self.s.m.bin(self.cur, Bin::Shl, high, eight);
            let both = self.s.m.bin(self.cur, Bin::Or, low, high);
            self.end_guarded(at, &[(both, low)])[0]
        } else {
            low
        };
        let left = self.s.m.i32(32 - width);
        let x = self.s.m.bin(self.cur, Bin::Shl, x, left);
        Ok(self.s.m.bin(
            self.cur,
            if signed { Bin::AShr } else { Bin::LShr },
            x,
            left,
        ))
    }

    /// Masks the affected bytes in each dword. A CAS keeps concurrent
    /// writes to other bytes in the same dword from being lost.
    fn write_subword(
        &mut self,
        address: V,
        width: u32,
        data: V,
        store: impl Fn(&mut Self, V, V, V),
    ) {
        let c = self.cur;
        let shift = self.byte_shift(address);
        let not3 = self.s.m.i32(!3);
        let aligned = self.s.m.bin(c, Bin::And, address, not3);
        let mask = self.s.m.i32((1 << width) - 1);
        let data = self.s.m.bin(c, Bin::And, data, mask);
        let bits = self.s.m.bin(c, Bin::Shl, data, shift);
        let mask = self.s.m.bin(c, Bin::Shl, mask, shift);
        store(self, aligned, mask, bits);
        if width == 16 {
            let c = self.cur;
            let k24 = self.s.m.i32(24);
            let split = self.s.m.cmp(c, pred::EQ, shift, k24);
            let at = self.begin_guarded(Some(split));
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let (eight, byte_mask) = (self.s.m.i32(8), self.s.m.i32(255));
            let bits = self.s.m.bin(self.cur, Bin::LShr, data, eight);
            store(self, next, byte_mask, bits);
            self.end_guarded(at, &[]);
        }
    }

    fn lds_ptr(&mut self, byte: V) -> V {
        let shift = self.s.m.i32(if self.wide_lds { 3 } else { 2 });
        let index = self.s.m.bin(self.cur, Bin::LShr, byte, shift);
        let lds = self.s.lds(self.wide_lds);
        self.s.m.element(self.cur, lds, index)
    }

    // DXIL forbids reinterpreting TGSM through a differently sized pointer.
    // When a shader needs i64 atomics, every access uses the same i64 array.
    fn lds_half_shift(&mut self, byte: V) -> V {
        let (four, eight) = (self.s.m.i32(4), self.s.m.i32(8));
        let half = self.s.m.bin(self.cur, Bin::And, byte, four);
        let shift = self.s.m.bin(self.cur, Bin::Mul, half, eight);
        let ty = self.s.m.int(64);
        self.s.m.cast(self.cur, Cast::ZExt, shift, ty)
    }

    fn lds_load_dword(&mut self, byte: V) -> V {
        if self.wide_lds {
            let end = self.s.m.i32(self.s.lds_dwords * 4);
            let in_bounds = self.s.m.cmp(self.cur, pred::IULT, byte, end);
            let at = self.begin_guarded(Some(in_bounds));
            let p = self.lds_ptr(byte);
            let ty = self.s.m.int(64);
            let zero = self.s.m.int_const(ty, 0);
            // Atomic access avoids racing with a lane using the other half.
            let pair = self.s.m.atomic(self.cur, Rmw::Or, p, zero);
            let shift = self.lds_half_shift(byte);
            let x = self.s.m.bin(self.cur, Bin::LShr, pair, shift);
            let x = self.s.m.cast(self.cur, Cast::Trunc, x, self.s.t.i32);
            let zero = self.s.m.i32(0);
            return self.end_guarded(at, &[(x, zero)])[0];
        }
        let p = self.lds_ptr(byte);
        self.s.m.load(self.cur, p)
    }

    fn lds_update_dword(&mut self, byte: V, update: impl FnOnce(&mut Self, V) -> V) -> V {
        let p = self.lds_ptr(byte);
        let zero = if self.wide_lds {
            let ty = self.s.m.int(64);
            self.s.m.int_const(ty, 0)
        } else {
            self.s.m.i32(0)
        };
        let shift = self.wide_lds.then(|| self.lds_half_shift(byte));
        let mut result = None;
        self.atomic_update(zero, |l, old| {
            let c = l.cur;
            let field = if let Some(shift) = shift {
                let field = l.s.m.bin(c, Bin::LShr, old, shift);
                l.s.m.cast(c, Cast::Trunc, field, l.s.t.i32)
            } else {
                old
            };
            result = Some(field);
            let next = update(l, field);
            let next = if let Some(shift) = shift {
                let ty = l.s.m.int(64);
                let next = l.s.m.cast(c, Cast::ZExt, next, ty);
                let next = l.s.m.bin(c, Bin::Shl, next, shift);
                let mask = l.s.m.int_const(ty, i64::from(u32::MAX));
                let mask = l.s.m.bin(c, Bin::Shl, mask, shift);
                l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, next)
            } else {
                next
            };
            l.s.m.cmpxchg(c, p, old, next)
        });
        result.expect("the compare-exchange body was emitted")
    }

    fn lds_store_dword(&mut self, byte: V, data: V) {
        if self.wide_lds {
            self.lds_update_dword(byte, |_, _| data);
        } else {
            let p = self.lds_ptr(byte);
            self.s.m.store(self.cur, p, data);
        }
    }

    fn atomic_data(&mut self, inst: &Inst, data: &Operand, wide: bool) -> Result<V, Why> {
        let words = self.vgprs(inst, data, if wide { 2 } else { 1 })?;
        Ok(if wide {
            self.join64(words[0], words[1])
        } else {
            words[0]
        })
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let f = self.console_flat();
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        // The console allows larger outputs than the host; the signatures
        // link to the title's pixel shader, so none are dropped.
        if mesh_output_bytes(&self.s.outputs, &self.s.prims, mesh) > MAX_MESH_OUTPUT_BYTES {
            return Err("mesh outputs beyond the host's 32 KiB");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// Console flow: `x` of console lane `lane` (uniform), from whichever
    /// half holds it.
    fn console_read_lane(&mut self, x: V, lane: V) -> V {
        if !self.split {
            // One host wave is the console wave, all of its lanes here.
            return self.read_lane(x, lane);
        }
        let c = self.cur;
        let (k31, k5, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(5),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let at = self.s.m.bin(c, Bin::And, lane, k31);
        let mine = self.read_lane(x, at);
        let (x0, x1) = self.exchange(c, mine);
        let h = self.s.m.bin(c, Bin::LShr, lane, k5);
        let h = self.s.m.bin(c, Bin::And, h, one);
        let low = self.s.m.cmp(c, pred::EQ, h, z);
        self.s.m.select(c, low, x0, x1)
    }

    /// Whether a read of another lane of this lane's quad must reach helper
    /// lanes: in a pixel shader, always. At a triangle's edge the quad's
    /// other pixels are helpers (and so are discarded ones), which only quad
    /// reads see: wave reads give 0 from them on AMD, so a derivative taken
    /// by hand (`quad_perm`) was the whole value there, the texture read
    /// that used it took its smallest level, and the triangles' edges showed
    /// as lines on car paint, glass and skin. Upstream does this after
    /// discards only (77053a0).
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead (upstream 0601955).
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out: once only helper lanes are
    /// left in such a loop, `WaveReadLaneFirst` reads no lane, no lane
    /// matches, and the loop runs to its limit in every such lane, with
    /// whatever descriptor the read gave. With its own value a helper lane
    /// matches in the first pass and leaves; what it computes only feeds
    /// derivatives (those of a texture read's dependent coordinates, at
    /// triangle edges), and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// Coarse derivatives of `coords` across this lane's quad, as DXIL
    /// gradients (x for up to three coordinates, then y), each times
    /// `scale` if given.
    fn quad_gradients(&mut self, coords: &[V], scale: Option<V>) -> Vec<V> {
        let c = self.cur;
        let lane = self.lane_index();
        let top = self.s.m.i32(!3);
        let base = self.s.m.bin(c, Bin::And, lane, top);
        let lanes: Vec<V> = (1..3u32)
            .map(|k| {
                let k = self.s.m.i32(k);
                self.s.m.bin(c, Bin::Or, base, k)
            })
            .collect();
        let fu = self.s.m.undef(self.s.t.f32);
        let mut g = vec![fu; 6];
        for (k, &x) in coords.iter().enumerate().take(3) {
            let xi = self.as_int(x);
            let pixel = self.s.pixel;
            let at = |l: &mut Self, lane: V| {
                // A pixel's quad through quad reads (see `helpers_read`).
                let r = if pixel {
                    l.quad_pick(xi, lane)
                } else {
                    l.read_lane(xi, lane)
                };
                l.as_float(r)
            };
            let origin = at(self, base);
            for (d, &lane) in lanes.iter().enumerate() {
                let there = at(self, lane);
                let mut diff = self.s.m.bin(c, Bin::Sub, there, origin);
                if let Some(s) = scale {
                    diff = self.s.m.bin(c, Bin::Mul, diff, s);
                }
                g[3 * d + k] = diff;
            }
        }
        g
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// A byte or halfword LDS read or write at byte address `a + offset`,
    /// through the dword holding it.
    #[allow(clippy::too_many_arguments)]
    fn ds_sub_dword(
        &mut self,
        inst: &Inst,
        sub: SubDword,
        a: V,
        offset: u32,
        dst_reg: Option<u8>,
        data: &[Operand],
        guard: Option<V>,
    ) -> Result<(), Why> {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        // shortcut: as for dwords, in bounds; a halfword is taken as
        // 2-aligned (the bytes never straddle two dwords).
        let o = k(self, offset);
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let not3 = k(self, !3);
        let aligned = self.s.m.bin(c, Bin::And, at, not3);
        let three = k(self, 3);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        let shift = self.s.m.bin(c, Bin::Shl, byte, three);
        let mask = k(self, (1 << sub.bits) - 1);
        if !sub.write {
            let reg = dst_reg.ok_or("an LDS read result")?;
            let x = self.lds_load_dword(aligned);
            let c = self.cur;
            let x = self.s.m.bin(c, Bin::LShr, x, shift);
            let x = if sub.signed {
                let up = k(self, 32 - sub.bits);
                let x = self.s.m.bin(c, Bin::Shl, x, up);
                self.s.m.bin(c, Bin::AShr, x, up)
            } else {
                self.s.m.bin(c, Bin::And, x, mask)
            };
            let x = match sub.d16 {
                None => x,
                Some(high) => {
                    let old = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                    let old = self.i32_of(old)?;
                    let low16 = k(self, 0xFFFF);
                    let half = self.s.m.bin(c, Bin::And, x, low16);
                    if high {
                        let keep = self.s.m.bin(c, Bin::And, old, low16);
                        let sixteen = k(self, 16);
                        let half = self.s.m.bin(c, Bin::Shl, half, sixteen);
                        self.s.m.bin(c, Bin::Or, keep, half)
                    } else {
                        let high16 = k(self, 0xFFFF_0000);
                        let keep = self.s.m.bin(c, Bin::And, old, high16);
                        self.s.m.bin(c, Bin::Or, keep, half)
                    }
                }
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let d = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
        let d = if sub.d16 == Some(true) {
            let sixteen = k(self, 16);
            self.s.m.bin(c, Bin::LShr, d, sixteen)
        } else {
            d
        };
        let d = self.s.m.bin(c, Bin::And, d, mask);
        let set = self.s.m.bin(c, Bin::Shl, d, shift);
        let bits = self.s.m.bin(c, Bin::Shl, mask, shift);
        let ones = k(self, !0);
        let clear = self.s.m.bin(c, Bin::Xor, bits, ones);
        // Lanes write other bytes of the same dword at the same time, so
        // the bytes go in atomically: cleared, then set. shortcut: lanes
        // writing the same bytes at once leave both values or-ed together
        // where the console keeps one of them.
        let at = self.begin_guarded(guard);
        if self.wide_lds {
            // The LDS is an i64 array (64-bit atomics): update the dword
            // in place through its compare-exchange loop.
            self.lds_update_dword(aligned, |l, old| {
                let c = l.cur;
                let kept = l.s.m.bin(c, Bin::And, old, clear);
                l.s.m.bin(c, Bin::Or, kept, set)
            });
        } else {
            let p = self.lds_ptr(aligned);
            self.s.m.atomic(self.cur, Rmw::And, p, clear);
            self.s.m.atomic(self.cur, Rmw::Or, p, set);
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        if name == "ds_nop" {
            return Ok(());
        }
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad || self.helpers_read(quad) {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if name == "ds_bpermute_b32" {
            // Each lane reads the data of the lane it names (times 4): 0 if
            // that one does not run this.
            self.cross_lane(false)?;
            if self.split {
                return Err("a lane permutation in a wave64 program split in halves");
            }
            let at = self.vgprs(inst, addr.as_ref().ok_or("a permutation address")?, 1)?[0];
            let x = self.vgprs(inst, data.first().ok_or("permutation data")?, 1)?[0];
            let (by, two, last) = (
                self.s.m.i32(offset),
                self.s.m.i32(2),
                self.s.m.i32(self.wave() - 1),
            );
            let at = self.s.m.bin(c, Bin::Add, at, by);
            let at = self.s.m.bin(c, Bin::LShr, at, two);
            let src = self.s.m.bin(c, Bin::And, at, last);
            let z = self.s.m.i32(0);
            let x = match guard {
                Some(g) => self.s.m.select(c, g, x, z),
                None => x,
            };
            let r = self.read_lane(x, src);
            let reg = dst_reg.ok_or("a permutation result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        if name.contains("addtid") {
            return Err("an LDS address based on the console lane ID and M0");
        }
        if name.contains("_src2_") {
            return Err("an LDS operation with a second memory source");
        }
        if name.starts_with("ds_gws")
            || matches!(name, "ds_append" | "ds_consume" | "ds_ordered_count")
        {
            return Err("an LDS or GDS queue or global wave synchronization operation");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        let byte = |l: &mut Self, bytes: u32| {
            let o = l.s.m.i32(bytes);
            l.s.m.bin(l.cur, Bin::Add, a, o)
        };
        // shortcut: whole-dword LDS addresses are taken as aligned and in bounds.
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        if let Some(sub) = SubDword::of(name) {
            return self.ds_sub_dword(inst, sub, a, offset, dst_reg, data, guard);
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let reg = dst_reg.ok_or("an LDS read result")?;
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let address = byte(self, off + 4 * k);
                    let x = self.lds_load_dword(address);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let address = byte(self, off);
                self.lds_store_dword(address, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        if name.ends_with("_f32") || name.ends_with("_f64") {
            // Mesa's aco_insert_fp_mode.cpp tracks the applicable MODE
            // denormal bits for these. NativeProgram does not carry them.
            return Err("an LDS floating-point atomic requiring the console's denormal mode");
        }
        if name.starts_with("ds_condxchg32") {
            return Err("an LDS conditional exchange with undocumented compare operands");
        }
        let (op, wide, kind) = ds_integer_atomic(name).ok_or("an LDS operation")?;
        if matches!(op, "wrxchg2" | "wrxchg2st64") {
            // AMD RDNA 2 ISA §12.13 specifies separate exchanges at the
            // two addresses. It does not specify their order if they alias.
            if offset0 == offset1 {
                return Err("an LDS paired exchange with coincident addresses");
            }
            let reg = dst_reg.ok_or("an LDS exchange result")?;
            let d0 = self.atomic_data(inst, data.first().ok_or("LDS exchange data")?, wide)?;
            let d1 = self.atomic_data(inst, data.get(1).ok_or("LDS exchange data")?, wide)?;
            let ty = self.s.m.type_of(d0);
            let zero = self.s.m.int_const(ty, 0);
            let at = self.begin_guarded(guard);
            let mut old = Vec::new();
            for ((off, _), data) in pair(name).into_iter().zip([d0, d1]) {
                let address = byte(self, off);
                let x = if !wide && self.wide_lds {
                    self.lds_update_dword(address, |_, _| data)
                } else {
                    let p = self.lds_ptr(address);
                    self.s.m.atomic(self.cur, Rmw::Xchg, p, data)
                };
                old.push((x, zero));
            }
            if self.wide_lds {
                self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
            }
            let words = if wide { 2 } else { 1 };
            for (k, x) in self.end_guarded(at, &old).into_iter().enumerate() {
                self.write_atomic_result(inst, reg + k as u8 * words, x, wide, guard);
            }
            return Ok(());
        }
        let rmw = match (op, kind) {
            ("add" | "sub", _) => Some(Rmw::Add),
            ("and", _) => Some(Rmw::And),
            ("or", _) => Some(Rmw::Or),
            ("xor", _) => Some(Rmw::Xor),
            ("min", b'i') => Some(Rmw::Min),
            ("max", b'i') => Some(Rmw::Max),
            ("min", _) => Some(Rmw::UMin),
            ("max", _) => Some(Rmw::UMax),
            ("wrxchg", _) => Some(Rmw::Xchg),
            _ => None,
        };
        let update = match op {
            "rsub" => Some(AtomicUpdate::ReverseSub),
            "inc" => Some(AtomicUpdate::Inc),
            "dec" => Some(AtomicUpdate::Dec),
            "mskor" => Some(AtomicUpdate::MaskOr),
            "wrap" => Some(AtomicUpdate::Wrap),
            _ if rmw.is_some() || op == "cmpst" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.atomic_data(inst, data.first().ok_or("LDS atomic data")?, wide)?;
        let ty = self.s.m.type_of(v0);
        let zero = self.s.m.int_const(ty, 0);
        let v1 = if matches!(op, "mskor" | "wrap" | "cmpst") {
            self.atomic_data(
                inst,
                data.get(1).ok_or("a second LDS atomic operand")?,
                wide,
            )?
        } else {
            zero
        };
        let address = byte(self, offset);
        let at = self.begin_guarded(guard);
        if wide || self.wide_lds {
            self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
        }
        let r = if !wide && self.wide_lds {
            self.lds_update_dword(address, |l, old| {
                if let Some(update) = update {
                    return l.atomic_new_value(update, old, v0, v1);
                }
                let c = l.cur;
                match op {
                    "add" => l.s.m.bin(c, Bin::Add, old, v0),
                    "sub" => l.s.m.bin(c, Bin::Sub, old, v0),
                    "and" => l.s.m.bin(c, Bin::And, old, v0),
                    "or" => l.s.m.bin(c, Bin::Or, old, v0),
                    "xor" => l.s.m.bin(c, Bin::Xor, old, v0),
                    "wrxchg" => v0,
                    "cmpst" => {
                        let same = l.s.m.cmp(c, pred::EQ, old, v0);
                        l.s.m.select(c, same, v1, old)
                    }
                    _ => {
                        let pred = match (op, kind) {
                            ("min", b'i') => pred::SLT,
                            ("max", b'i') => pred::SGT,
                            ("min", _) => pred::IULT,
                            ("max", _) => pred::IUGT,
                            _ => unreachable!(),
                        };
                        let take = l.s.m.cmp(c, pred, v0, old);
                        l.s.m.select(c, take, v0, old)
                    }
                }
            })
        } else if let Some(rmw) = rmw {
            let p = self.lds_ptr(address);
            // DXIL supports atomic add, with subtraction encoded as -DATA.
            let data = if op == "sub" {
                self.s.m.bin(self.cur, Bin::Sub, zero, v0)
            } else {
                v0
            };
            self.s.m.atomic(self.cur, rmw, p, data)
        } else if let Some(update) = update {
            let p = self.lds_ptr(address);
            self.atomic_update(zero, |l, old| {
                let x = l.atomic_new_value(update, old, v0, v1);
                l.s.m.cmpxchg(l.cur, p, old, x)
            })
        } else {
            // Compare with data0, store data1.
            let p = self.lds_ptr(address);
            self.s.m.cmpxchg(self.cur, p, v0, v1)
        };
        let r = self.end_guarded(at, &[(r, zero)])[0];
        if let Some(reg) = dst_reg {
            self.write_atomic_result(inst, reg, r, wide, guard);
        }
        Ok(())
    }

    fn write_atomic_result(&mut self, inst: &Inst, reg: u8, old: V, wide: bool, guard: Option<V>) {
        if wide {
            let (lo, hi) = self.split64(old);
            self.write(inst, Reg::V(reg), lo, guard);
            self.write(inst, Reg::V(reg + 1), hi, guard);
        } else {
            self.write(inst, Reg::V(reg), old, guard);
        }
    }

    /// See [`Lower::ordered`]: programs that reach memory by address or
    /// write through root views, which are the builders of ray tracing's
    /// structures.
    fn ordered(&mut self) -> bool {
        let x = self.x;
        *self.ordered.get_or_insert_with(|| {
            super::rays()
                && (x.res.addresses
                    || x.native
                        .used
                        .iter()
                        .any(|u| u.root == RootType::Descriptor && u.kind == ResourceType::Uav))
        })
    }

    /// Writes to buffers made before are seen before those made after, by
    /// every group.
    fn uav_fence(&mut self) {
        let (mode, void) = (self.s.m.i32(2), self.s.t.void);
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    // ---- instructions ----

}
