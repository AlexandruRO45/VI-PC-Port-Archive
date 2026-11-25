impl<'a> Lower<'a> {
    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err("an MTBUF store requiring a packed-format write");
        }
        if name.contains("_format_d16_hi_") {
            return Err("a typed buffer access to the high half with an untracked numeric format");
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        let atomic = name.strip_prefix("buffer_atomic_");
        let wide = atomic.is_some_and(|n| n.ends_with("_x2"));
        let atomic = atomic.map(|n| n.strip_suffix("_x2").unwrap_or(n));
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let native_atomic = match atomic {
            None => None,
            Some("add" | "sub") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap" | "inc" | "dec" | "csub") => None,
            Some("fcmpswap" | "fmin" | "fmax") => {
                // Mesa aco_insert_fp_mode.cpp, vmem_default_needs.
                return Err("a buffer floating-point atomic requiring the console's denormal mode");
            }
            Some(_) => return Err(name),
        };
        let sub_load = name.strip_prefix("buffer_load_").and_then(subword_load);
        let sub_store = match name {
            "buffer_store_byte" | "buffer_store_byte_d16_hi" => Some(8),
            "buffer_store_short" | "buffer_store_short_d16_hi" => Some(16),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || sub_load.is_some() || sub_store.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let stride = used.map_or(0, |u| u.stride);
        if self.by_address(src)
            && let Fields::Buffer { rsrc, .. } = &inst.fields
        {
            if typed && format.is_none() {
                return Err("a typed buffer access by address");
            }
            let (h, start, stride) = self.made_buffer(inst, rsrc)?;
            self.made.insert(site, h);
            off = self.s.m.bin(c, Bin::Add, off, start);
            if let Some(i) = index {
                let rows = self.s.m.bin(c, Bin::Mul, i, stride);
                off = self.s.m.bin(c, Bin::Add, off, rows);
            }
        }
        let index = if self.by_address(src) { None } else { index };
        let (base, extra) = if self.by_address(src) {
            (0, None)
        } else {
            self.built_offset(src)?
        };
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            if store || atomic.is_some() {
                return Err("a write or atomic operation on a constant buffer");
            }
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            if let Some((width, signed, half)) = sub_load {
                let x =
                    self.read_subword(byte, width, signed, |l, at| Ok(l.cbuffer_dword(h, at)))?;
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // A dword access by index through a descriptor built in code: the
        // stride is the code's own, so the element becomes a byte offset in
        // the raw root view.
        let (index, off) = match index {
            Some(i) if !typed && format.is_none() => match self.built_element(src, i, off)? {
                Some(byte) => (None, byte),
                // Structured buffers of 16-byte multiples are raw views of
                // their bytes on the host: D3D12 takes structured views only
                // on multiples of the stride, and AMD reads element 0 for all
                // of a raw view's. (Others find a start they fit, and raw
                // views are to start on 16 bytes.)
                None if stride >= 16 && stride.is_multiple_of(16) => {
                    let st = self.s.m.i32(stride);
                    let first = self.s.m.bin(c, Bin::Mul, i, st);
                    (None, self.s.m.bin(c, Bin::Add, first, off))
                }
                None => (Some(i), off),
            },
            x => (x, off),
        };
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if sub_load.is_some() || sub_store.is_some() {
            let data = data.as_ref().ok_or("buffer sub-dword data")?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("buffer sub-dword data");
            };
            let h = if index.is_some() {
                if stride < 4 || !stride.is_multiple_of(4) {
                    return Err(
                        "a structured sub-dword buffer access without a dword-aligned stride",
                    );
                }
                self.handle(site, src, false, (12 | u, stride))?
            } else {
                self.handle(site, src, false, (11 | u, 0))?
            };
            let coords = |byte| index.map_or([byte, iu], |i| [i, byte]);
            if let Some((width, signed, half)) = sub_load {
                // Root SRVs have no size in DXIL.
                let bounds = (!matches!(src, Source::Built { .. } | Source::Address))
                    .then(|| self.buffer_bounds(h, index, off, stride, width / 8));
                let at = self.begin_guarded(bounds);
                let x =
                    self.read_subword(off, width, signed, |l, at| Ok(l.raw_dword(h, coords(at))))?;
                let zero = self.s.m.i32(0);
                let x = self.end_guarded(at, &[(x, zero)])[0];
                self.write_subword_result(inst, reg, x, half, guard)?;
            } else if let Some(width) = sub_store {
                let mut x = self.vgprs(inst, data, 1)?[0];
                if name.ends_with("_hi") {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let bounds = self.buffer_bounds(h, index, off, stride, width / 8);
                let active = self.buffer_store_guard(bounds, guard);
                let at = self.begin_guarded(Some(active));
                self.write_subword(off, width, x, |l, byte, mask, bits| {
                    let [a, b] = coords(byte);
                    let zero = l.s.m.i32(0);
                    l.atomic_update(zero, |l, old| {
                        let x = l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, bits);
                        l.buffer_compare_exchange(h, [a, b, iu], old, x, false)
                    });
                });
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(op) = atomic {
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            if op == "csub" && !returns {
                // LLVM BUFInstructions.td only selects the returning form
                // on GFX10.3.
                return Err("buffer_atomic_csub without the required GLC bit");
            }
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let words = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, d, if op == "cmpswap" { 2 * words } else { words })?;
            let val = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let ty = self.s.m.type_of(val);
            let zero = self.s.m.int_const(ty, 0);
            let (h, coords) = match index {
                Some(index) => {
                    if stride >= 4 && stride.is_multiple_of(4) {
                        if wide && !stride.is_multiple_of(8) {
                            return Err(
                                "a 64-bit buffer atomic without an eight-byte-aligned stride",
                            );
                        }
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        (h, [index, off, iu])
                    } else {
                        if wide {
                            return Err(
                                "a 64-bit buffer atomic through a structured buffer of unknown stride",
                            );
                        }
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        (self.handle(site, src, false, (11 | u, 0))?, [byte, iu, iu])
                    }
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_HEAP64;
            }
            let active = if wide || matches!(op, "sub" | "inc" | "dec" | "csub") {
                let bounds = self.buffer_bounds(h, index, off, stride, 4 * u32::from(words));
                Some(self.buffer_store_guard(bounds, guard))
            } else {
                guard
            };
            let at = self.begin_guarded(active);
            let r = if op == "cmpswap" {
                // cmpswap: data is (new, compare).
                let compare = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                self.buffer_compare_exchange(h, coords, compare, val, wide)
            } else if let Some(code) = native_atomic {
                let name = if wide {
                    "atomicBinOp.i64"
                } else {
                    "atomicBinOp.i32"
                };
                let code = self.s.m.i32(code);
                let data = if op == "sub" {
                    self.s.m.bin(self.cur, Bin::Sub, zero, val)
                } else {
                    val
                };
                self.s.op(
                    self.cur,
                    name,
                    Attr::NoUnwind,
                    ty,
                    78,
                    &[h, code, coords[0], coords[1], coords[2], data],
                )
            } else {
                let update = match op {
                    "inc" => AtomicUpdate::Inc,
                    "dec" => AtomicUpdate::Dec,
                    "csub" => AtomicUpdate::ClampSub,
                    _ => unreachable!(),
                };
                self.atomic_update(zero, |l, old| {
                    let x = l.atomic_new_value(update, old, val, zero);
                    l.buffer_compare_exchange(h, coords, old, x, wide)
                })
            };
            if returns {
                let r = self.end_guarded(at, &[(r, zero)])[0];
                self.write_atomic_result(inst, reg, r, wide, guard);
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write a float format.
                // The channels the opcode leaves out are written as 0: the
                // validator has typed stores write all four.
                if *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                args.resize(3 + 4, self.s.m.f32(0));
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let byte = self.runtime_stride_offset(src, index, off)?;
                            args.push(self.handle(site, src, false, (11 | u, 0))?);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err(
                    "an indexed MTBUF load requiring a raw view of a structured descriptor",
                );
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        let h = self.handle(site, src, false, (11 | u, 0))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    /// The program's natural loops (from their reachable back edges), and
    /// for each that lanes leave through one block (every edge into it from
    /// the loop): the scalar values written in the loop that code after it
    /// reads (see [`Lower::leave_loops`]).
    fn find_loop_exits(&mut self) {
        let p = self.x.p;
        let reachable = |l: &Self, q: usize| l.block[q].is_some();
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); p.blocks.len()];
        for (b, block) in p.blocks.iter().enumerate() {
            if reachable(self, b) {
                for &q in block.preds.iter().filter(|&&q| reachable(self, q)) {
                    succs[q].push(b);
                }
            }
        }
        let scalar = |r: Reg| match r {
            Reg::S(c) => c != EXEC && c != EXEC + 1,
            Reg::Scc => true,
            _ => false,
        };
        let mut headers: Vec<usize> = self.counters.keys().copied().collect();
        headers.sort_unstable();
        for h in headers {
            let mut body: HashSet<usize> = HashSet::from([h]);
            let mut work: Vec<usize> = p.blocks[h]
                .preds
                .iter()
                .copied()
                .filter(|&q| reachable(self, q) && self.rpo[h] <= self.rpo[q])
                .collect();
            while let Some(q) = work.pop() {
                if body.insert(q) {
                    work.extend(
                        p.blocks[q]
                            .preds
                            .iter()
                            .copied()
                            .filter(|&r| reachable(self, r)),
                    );
                }
            }
            let mut exits: Vec<usize> = body
                .iter()
                .flat_map(|&b| succs[b].iter().copied())
                .filter(|t| !body.contains(t))
                .collect();
            exits.sort_unstable();
            exits.dedup();
            let single = match exits[..] {
                [e] if p.blocks[e]
                    .preds
                    .iter()
                    .all(|q| !reachable(self, *q) || body.contains(q)) =>
                {
                    Some(e)
                }
                _ => None,
            };
            // A waterfall loop runs once with each lane's own descriptor (its
            // compares hold in every lane), so its scalars stay each lane's.
            let waterfall = self
                .x
                .res
                .waterfall
                .compares
                .iter()
                .any(|site| body.contains(&site.0));
            if let (Some(e), true) = (single, waterfall) {
                // Listed without values: the exit joins nothing either.
                self.loop_exits.entry(e).or_default();
            } else if let Some(e) = single {
                let mut written = HashSet::new();
                for &b in &body {
                    let block = &p.blocks[b];
                    let phis = block.phis.iter().map(|phi| (phi.reg, phi.value));
                    let writes = block.insts.iter().flat_map(|i| i.writes.iter().copied());
                    written.extend(
                        phis.chain(writes)
                            .filter(|&(r, _)| scalar(r))
                            .map(|(_, v)| v),
                    );
                }
                let mut after = Vec::new();
                let mut seen = HashSet::new();
                for (b, block) in p.blocks.iter().enumerate() {
                    if body.contains(&b) || !reachable(self, b) {
                        continue;
                    }
                    let reads = block
                        .insts
                        .iter()
                        .flat_map(|i| i.reads.iter().map(|r| r.1))
                        .chain(block.phis.iter().flat_map(|phi| phi.args.iter().copied()));
                    for v in reads {
                        if written.contains(&v) && seen.insert(v) {
                            after.push(v);
                        }
                    }
                }
                self.loop_exits.entry(e).or_default().push((h, after));
            }
            self.loop_bodies.insert(h, body);
        }
    }

    /// Entering `b`, a join of branches (not a loop header): its scalar
    /// joins as the wave has them. The console runs each path some lane
    /// takes, in code order, with one set of scalar registers, so after the
    /// join every lane reads what the last path run left; here each lane
    /// reads what its own path left. Each lane's path is ranked by its code
    /// order, and the lanes of the highest rank hold the wave's values.
    // shortcut: a scalar both paths change is the later path's own result,
    // not that path run on the earlier one's.
    fn join_scalars(&mut self, b: usize) {
        let p = self.x.p;
        if self.counters.contains_key(&b) {
            return;
        }
        let scalar = |r: Reg| match r {
            Reg::S(c) => c != EXEC && c != EXEC + 1,
            Reg::Scc => true,
            _ => false,
        };
        let joins: Vec<usize> = (0..p.blocks[b].phis.len())
            .filter(|&i| {
                let phi = &p.blocks[b].phis[i];
                scalar(phi.reg)
                    && matches!(self.vals.get(&phi.value), Some(Val::I32(_) | Val::Bit(_)))
            })
            .collect();
        if joins.is_empty() {
            return;
        }
        // Reachable predecessors and the DXIL block each edge arrives from.
        let mut preds: Vec<(usize, usize)> = Vec::new();
        for &q in &p.blocks[b].preds {
            let Some(tail) = self.tail[q] else { continue };
            let via = self.edges.get(&(q, b)).copied().unwrap_or(tail);
            preds.push((q, via));
        }
        if preds.len() < 2 {
            return;
        }
        let mut order: Vec<usize> = preds.iter().map(|&(q, _)| q).collect();
        order.sort_by_key(|&q| (p.blocks[q].offset, self.rpo[q]));
        order.dedup();
        let c = self.cur;
        let i32t = self.s.t.i32;
        let rank = self.s.m.phi(c, i32t);
        let incoming: Vec<(V, usize)> = preds
            .iter()
            .map(|&(q, via)| {
                let r = order.iter().position(|&x| x == q).expect("a predecessor") as u32;
                (self.s.m.i32(r), via)
            })
            .collect();
        self.s.m.set_phi(rank, incoming);
        let (max, unsigned) = (self.s.m.i8(3), self.s.m.i8(1));
        let wave_max = |l: &mut Self, x: V| {
            l.s.op(
                c,
                "waveActiveOp.i32",
                Attr::NoUnwind,
                i32t,
                119,
                &[x, max, unsigned],
            )
        };
        let last = wave_max(self, rank);
        let mine = self.s.m.cmp(c, pred::EQ, rank, last);
        self.s.flags |= flags::WAVE_OPS;
        let zero = self.s.m.i32(0);
        for i in joins {
            let v = p.blocks[b].phis[i].value;
            let f = match self.vals[&v] {
                Val::I32(x) => {
                    let x = self.s.m.select(c, mine, x, zero);
                    Val::I32(wave_max(self, x))
                }
                Val::Bit(x) => {
                    let x = self.s.m.cast(c, Cast::ZExt, x, i32t);
                    let x = self.s.m.select(c, mine, x, zero);
                    let w = wave_max(self, x);
                    Val::Bit(self.s.m.cmp(c, pred::NE, w, zero))
                }
                Val::Missing(_) => continue,
            };
            self.vals.insert(v, f);
        }
    }

    /// Entering `b`, the block lanes leave loops through: the scalar values
    /// those loops wrote, as the wave's last iteration left them. The console
    /// runs a divergent loop until its last lane leaves, so afterwards every
    /// lane reads the last iteration's scalars; here each lane leaves with
    /// those of its own last iteration. The lanes that ran the most
    /// iterations hold the wave's (a scalar is the same in all of them).
    // shortcut: a value lowered after this block (a loop block later in
    // reverse postorder) keeps each lane's own.
    fn leave_loops(&mut self, b: usize) {
        let Some(loops) = self.loop_exits.get(&b).cloned() else {
            return;
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        for (h, values) in loops {
            let (count, _) = self.counters[&h];
            let (max, unsigned) = (self.s.m.i8(3), self.s.m.i8(1));
            let wave_max = |l: &mut Self, x: V| {
                l.s.op(
                    c,
                    "waveActiveOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    119,
                    &[x, max, unsigned],
                )
            };
            let last = wave_max(self, count);
            let mine = self.s.m.cmp(c, pred::EQ, count, last);
            self.s.flags |= flags::WAVE_OPS;
            let zero = self.s.m.i32(0);
            let final_of = |l: &mut Self, val: Val| -> Option<Val> {
                Some(match val {
                    Val::I32(x) => {
                        let x = l.s.m.select(c, mine, x, zero);
                        Val::I32(wave_max(l, x))
                    }
                    Val::Bit(x) => {
                        let x = l.s.m.cast(c, Cast::ZExt, x, i32t);
                        let x = l.s.m.select(c, mine, x, zero);
                        let w = wave_max(l, x);
                        Val::Bit(l.s.m.cmp(c, pred::NE, w, zero))
                    }
                    Val::Missing(_) => return None,
                })
            };
            for v in values {
                let Some(val) = self.vals.get(&v).copied() else {
                    continue;
                };
                if let Some(f) = final_of(self, val) {
                    self.finals.insert(v, (f, h));
                }
            }
            // This block's joins of the loop's scalars.
            for phi in &self.x.p.blocks[b].phis {
                let scalar = matches!(phi.reg, Reg::S(r) if r != EXEC && r != EXEC + 1)
                    || phi.reg == Reg::Scc;
                if !scalar {
                    continue;
                }
                if let Some(val) = self.vals.get(&phi.value).copied()
                    && let Some(f) = final_of(self, val)
                {
                    self.vals.insert(phi.value, f);
                }
            }
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } if self.console_cf => {
                let cond = self.console_branch(b).map_err(|w| unsupported(offset, w))?;
                self.branch_term(b, cond, taken, not_taken)
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                self.branch_term(b, cond, taken, not_taken)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// The terminator of block `b` branching to `taken` where `cond` holds.
    fn branch_term(&mut self, b: usize, cond: V, taken: usize, not_taken: usize) -> Term {
        // A back edge is taken only below the iteration limit.
        let back = |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
        // (No `xor` of a compare: a loop exit testing one hung the
        // AMD driver.)
        let cond = if back(self, taken) {
            let ok = self.count_iteration(taken, pred::IULT);
            self.s.m.bin(self.cur, Bin::And, cond, ok)
        } else if back(self, not_taken) {
            let stop = self.count_iteration(not_taken, pred::IUGE);
            self.s.m.bin(self.cur, Bin::Or, cond, stop)
        } else {
            cond
        };
        // Critical edges get a block of their own, so loops have
        // dedicated exits and preheaders as DXC emits them.
        let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
        for (to, via) in [(taken, t), (not_taken, f)] {
            if !back(self, to) {
                // From an edge block, or from here.
                let from = if Some(via) == self.block[to] {
                    self.cur
                } else {
                    via
                };
                self.enter_loop(b, to, from);
            }
        }
        Term::CondBr(cond, t, f)
    }

    /// Console flow: whether block `b`'s `s_cbranch_*` is taken, the same
    /// in both halves (wave-wide mask tests through [`Self::any_lane`]).
    fn console_branch(&mut self, b: usize) -> Result<V, Why> {
        let inst = self.x.p.blocks[b].insts.last().ok_or("a branch")?;
        let name = inst.name();
        let c = self.cur;
        let (any, set) = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let v = inst.read(Reg::Scc).ok_or("a branch's SCC")?;
                let scc = self.scc_bit(v)?;
                (scc, name == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") {
                    crate::ir::VCC
                } else {
                    EXEC
                };
                let lo = inst.read(Reg::S(reg)).ok_or("a branch's mask")?;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&lo)
                    .ok_or("a branch on an untracked mask")?;
                (self.any_lane(m)?, name.ends_with("nz"))
            }
            _ => return Err("a branch the console flow does not model"),
        };
        Ok(if set {
            any
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, any, t)
        })
    }

    /// Console flow: an SCC value as a bit. A mask operation's SCC (result
    /// not zero) is whether some lane of the console wave has its bit.
    fn scc_bit(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&Val::Bit(x)) = self.vals.get(&v) {
            return Ok(x);
        }
        let b = *self
            .def_block
            .get(&v)
            .ok_or("an SCC from the program's start")?;
        let inst = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
            .ok_or("an SCC without its instruction")?;
        let mask = inst
            .writes
            .iter()
            .filter(|w| matches!(w.0, Reg::S(_)))
            .find_map(|w| self.x.st.masks.get(&w.1).copied())
            .ok_or("an SCC the emitter did not compute")?;
        self.any_lane(mask)
    }

    /// Console flow: whether `m` holds in some lane of the console wave.
    fn any_lane(&mut self, m: Bool) -> Result<V, Why> {
        let c = self.cur;
        let bit = self.boolean(m, c)?;
        let (lo, hi) = self.console_ballot(c, bit);
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, or, z))
    }

    /// Console flow: the console wave's ballot of `bit`, low and high word.
    fn console_ballot(&mut self, at: usize, bit: V) -> (V, V) {
        if self.split {
            let word = self.ballot_word(at, bit);
            return self.exchange(at, word);
        }
        // One host wave is the console wave (the high word is 0 in waves
        // of 32).
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1))
    }

    /// The host wave's ballot of `bit` (32 lanes: the first word).
    fn ballot_word(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        self.s.m.extract(at, r, 0)
    }

    /// The console thread's flat id in the group (see [`Self::thread_id`]).
    fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.s.op(
            e,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.console_flat = Some(f);
        f
    }

    /// This lane's console lane (0..63, 0..31 in waves of 32) and its
    /// half of a wave64 (0, 1).
    fn console_lane(&mut self) -> (V, V) {
        let f = self.console_flat();
        let e = self.s.entry;
        let (mask, k5) = (self.s.m.i32(self.wave() - 1), self.s.m.i32(5));
        let lane = self.s.m.bin(e, Bin::And, f, mask);
        let half = self.s.m.bin(e, Bin::LShr, lane, k5);
        (lane, half)
    }

    /// Console flow: each half gives `x` (the same in all its lanes) and
    /// both get (the lower half's, the upper half's): through two group-
    /// shared dwords between group barriers, which every thread of the
    /// group reaches (see [`CONSOLE_FLOW`]). A group of 32 threads or
    /// fewer has no upper half: its lanes count as inactive there (0).
    fn exchange(&mut self, at: usize, x: V) -> (V, V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let z = self.s.m.i32(0);
        if threads <= 32 {
            return (x, z);
        }
        let (_, half) = self.console_lane();
        let slots = match self.xchg {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(2, self.s.t.i32);
                let g = self.s.m.global("xchg", arr, 3);
                self.xchg = Some(g);
                g
            }
        };
        let void = self.s.t.void;
        // Thread-group sync with a group-shared memory fence.
        let mode = self.s.m.i32(1 | 8);
        let mine = self.s.m.element(at, slots, half);
        self.s.m.store(at, mine, x);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let one = self.s.m.i32(1);
        let p0 = self.s.m.element(at, slots, z);
        let p1 = self.s.m.element(at, slots, one);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // Both read before the next exchange writes.
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        (lo, hi)
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = self.s.m.i32(super::loop_limit());
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// A dword a scalar load writes, where the load is a descriptor the
    /// program uses as a resource or sampler.
    fn descriptor_dword(&self, v: Value) -> bool {
        let Some(load) = self.x.p.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))
                .map(|i| (b, i))
        }) else {
            return false;
        };
        self.x.p.blocks[load.0].insts[load.1]
            .name()
            .starts_with("s_load_dword")
            && self.x.res.descriptors.iter().any(|d| {
                d.source
                    .leaves()
                    .iter()
                    .any(|l| matches!(l, Source::Loaded { load: at, .. } if *at == load))
            })
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        self.fill_paths().map_err(|w| unsupported(0, w))?;
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A register initialized only on one branch may carry a dead
            // descriptor dword left over from earlier register reuse on the
            // other branch (guarded by a separate predicate). Only treat a
            // descriptor-dword `Missing` as fatal when no incoming edge carries
            // a real data value.
            let has_data = !bit
                && p.blocks[b]
                    .preds
                    .iter()
                    .enumerate()
                    .filter(|&(_, &q)| self.tail[q].is_some())
                    .any(|(n, _)| matches!(self.vals.get(&phi.args[n]), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                }
                // A root parameter no instruction read.
                if !self.vals.contains_key(&arg)
                    && let Some(&(param, dword)) = self.params.get(&arg)
                    && self.read_phis.contains(&phi.value)
                {
                    let val = self.user(param, dword);
                    self.vals.insert(arg, val);
                }
                // An edge from outside a loop takes the loop's final value.
                let looked = match self.finals.get(&arg) {
                    Some(&(f, h)) if !self.loop_bodies[&h].contains(&q) => Some(f),
                    _ => self.vals.get(&arg).copied(),
                };
                let x = match (looked, bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && matches!(
                                why,
                                "a descriptor's address or format read as data"
                                    | "a descriptor read only as data"
                            ) =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    // A descriptor's dwords reaching a join that also takes
                    // data (the registers are reused after the last access).
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only after the data edge.
                    (Some(Val::Missing(_)), _) if self.descriptor_dword(arg) => {
                        if bit {
                            self.s.m.undef(self.s.t.i1)
                        } else {
                            self.s.m.undef(self.s.t.i32)
                        }
                    }
                    // SCC of a mask operation joined with a comparison's: the
                    // edge past a section no lane is in, where it is clear.
                    (Some(Val::Missing(why)), true)
                        if why == LANE_MASK
                            && phi
                                .args
                                .iter()
                                .any(|a| matches!(self.vals.get(a), Some(Val::Bit(_)))) =>
                    {
                        self.s.m.bool(false)
                    }
                    // As above, of a lane mask: the registers held one before.
                    // shortcut: as above, zero there.
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && (why == LANE_MASK
                                || self.x.st.masks.contains_key(&arg)
                                || self.x.st.highs.contains_key(&arg)) =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                // Console flow: every lane arrives by every edge, so a join
                // also takes masks that are constants or built from scalars.
                let arg = &phi.args[n];
                // shortcut: a plain scalar on the edge is the register in
                // another use, taken for a mask no lane is in; right as long
                // as the program does not read the join after that edge.
                let x = match self
                    .x
                    .st
                    .masks
                    .get(arg)
                    .or_else(|| self.x.st.others.get(arg))
                {
                    Some(&m) => self.boolean(m, qb).map_err(|w| unsupported(offset, w))?,
                    None => self.s.m.bool(false),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}
