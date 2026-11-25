impl<'a> Lower<'a> {
    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        if self.dead.contains(&(b, i)) {
            for &(_, v) in &inst.writes {
                self.vals
                    .insert(v, Val::Missing("a spilled SGPR's lane read as data"));
            }
            return Ok(());
        }
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. })
                && !(name.starts_with("global_load_") || name.starts_with("scratch_load_"))
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let fence = (name.starts_with("buffer_atomic") || name.starts_with("global_atomic"))
            && self.ordered();
        if fence {
            self.uav_fence();
        }
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } if name.starts_with("scratch_") => self.scratch(inst, b, i),
            // Generic addresses: private, shared or global memory by aperture.
            Fields::Flat { .. } => Err("flat memory"),
        };
        if fence && r.is_ok() {
            self.uav_fence();
        }
        if let Err(w) = &r
            && std::env::var_os("VICTORY_WHY").is_some()
        {
            eprintln!("missing {offset:#x} {name} {w}");
        }
        #[cfg(test)]
        if let Err(why) = r
            && AUDIT_OFFSET.get() == Some(offset)
        {
            return Err(unsupported(offset, why));
        }
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        // A lane mask written by a set operation also gets its value as a
        // number, from its sources' numbers. A ballot where the mask is
        // written misses the lanes it holds that do not run there: a loop's
        // exit mask gathers lanes that left in earlier iterations. Only the
        // masks something reads as numbers get one.
        let number = self.mask_ok
            && mask_set_op(name)
            && inst.writes.iter().any(|w| self.mask_numbers.contains(&w.1));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) && !number {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                Some(Operand::Literal(x)) if name == "s_bfe_i64" => {
                    // AMDGPUOperandSyntax: a literal in a signed i64 operand sign-extends.
                    Ok(l.s.m.i32(if (*x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        // See AMD's RDNA 2 ISA, sections 12.1–12.4 (SOP2/K/1/C).
        let old = |l: &mut Self, h: u8| -> Result<V, Why> {
            let reg = dst_code.ok_or("a missing destination")?;
            if reg == NULL {
                return Ok(l.s.m.i32(0));
            }
            let value = inst.read(Reg::S(reg + h)).ok_or("an unread destination")?;
            l.i32_of(value)
        };
        if name.starts_with("s_bitcmp") {
            let index = src32(self, 1)?;
            let a = half(self, 0, 0)?;
            let x = if name.ends_with("64") {
                let b = half(self, 0, 1)?;
                let k32 = self.s.m.i32(32);
                let hi = self.s.m.bin(c, Bin::And, index, k32);
                let zero = self.s.m.i32(0);
                let lo = self.s.m.cmp(c, pred::EQ, hi, zero);
                self.s.m.select(c, lo, a, b)
            } else {
                a
            };
            let last = self.s.m.i32(31);
            let index = self.s.m.bin(c, Bin::And, index, last);
            let bit = self.s.m.bin(c, Bin::LShr, x, index);
            let one = self.s.m.i32(1);
            let bit = self.s.m.bin(c, Bin::And, bit, one);
            let k = self.s.m.i32(u32::from(name.starts_with("s_bitcmp1")));
            let test = self.s.m.cmp(c, pred::EQ, bit, k);
            self.set(scc, Val::Bit(test));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            if matches!(name, "s_cmp_eq_u32" | "s_cmp_lg_u32") {
                let s0 = src.first().ok_or("a missing operand")?;
                let s1 = src.get(1).ok_or("a missing operand")?;
                let p0 = self.host_mask_pair(inst, s0)?;
                let p1 = self.host_mask_pair(inst, s1)?;
                let eq = match (p0, p1) {
                    (Some(a), Some(b)) => Some(self.eq_64(c, a, b)),
                    (Some(a), None) if Self::is_zero_operand(s1) => {
                        let z = self.s.m.i32(0);
                        Some(self.eq_64(c, a, (z, z)))
                    }
                    (None, Some(b)) if Self::is_zero_operand(s0) => {
                        let z = self.s.m.i32(0);
                        Some(self.eq_64(c, (z, z), b))
                    }
                    _ => None,
                };
                if let Some(mut r) = eq {
                    if op.starts_with("lg") {
                        let t = self.s.m.bool(true);
                        r = self.s.m.bin(c, Bin::Xor, r, t);
                    }
                    self.set(scc, Val::Bit(r));
                    return Ok(());
                }
            }
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64")
            || name.ends_with("_u64")
            || name.ends_with("_i64")
            || name == "s_bitreplicate_b64_b32";
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let r = self.bcnt1_64(c, a, b);
                    let r = if name == "s_bcnt0_i32_b64" {
                        let size = self.s.m.i32(64);
                        self.s.m.bin(c, Bin::Sub, size, r)
                    } else {
                        r
                    };
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64" => {
                    let (mut a, mut b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    if name == "s_ff0_i32_b64" {
                        let ones = self.s.m.i32(u32::MAX);
                        a = self.s.m.bin(c, Bin::Xor, a, ones);
                        b = self.s.m.bin(c, Bin::Xor, b, ones);
                    } else if name == "s_flbit_i32_i64" {
                        let k31 = self.s.m.i32(31);
                        let sign = self.s.m.bin(c, Bin::AShr, b, k31);
                        a = self.s.m.bin(c, Bin::Xor, a, sign);
                        b = self.s.m.bin(c, Bin::Xor, b, sign);
                    }
                    let op = if name.starts_with("s_flbit") {
                        (a, b) = (b, a);
                        33 // first bit high, counted from the MSB
                    } else {
                        32 // first bit low
                    };
                    let first =
                        self.s
                            .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                    let second =
                        self.s
                            .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let offset = self.s.m.bin(c, Bin::Add, second, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let second = self.s.m.select(c, none, second, offset);
                    let found = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, found, first, second);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_brev_b64" => {
                    for (h, dst) in [(0, hi), (1, lo)] {
                        let x = half(self, 0, h)?;
                        let x = self
                            .s
                            .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[x]);
                        self.set(dst, Val::I32(x));
                    }
                    Ok(())
                }
                "s_bitreplicate_b64_b32" => {
                    let x = src32(self, 0)?;
                    let k16 = self.s.m.i32(16);
                    let upper = self.s.m.bin(c, Bin::LShr, x, k16);
                    for (mut x, dst) in [(x, lo), (upper, hi)] {
                        let mask = self.s.m.i32(0xFFFF);
                        x = self.s.m.bin(c, Bin::And, x, mask);
                        // Spread the 16 source bits onto even positions.
                        for (shift, mask) in [
                            (8, 0x00FF_00FF),
                            (4, 0x0F0F_0F0F),
                            (2, 0x3333_3333),
                            (1, 0x5555_5555),
                        ] {
                            let sh = self.s.m.i32(shift);
                            let m = self.s.m.i32(mask);
                            let t = self.s.m.bin(c, Bin::Shl, x, sh);
                            let t = self.s.m.bin(c, Bin::Or, t, x);
                            x = self.s.m.bin(c, Bin::And, t, m);
                        }
                        let one = self.s.m.i32(1);
                        let other = self.s.m.bin(c, Bin::Shl, x, one);
                        let r = self.s.m.bin(c, Bin::Or, x, other);
                        self.set(dst, Val::I32(r));
                    }
                    Ok(())
                }
                "s_bfe_u64" | "s_bfe_i64" | "s_bfm_b64" | "s_bitset0_b64" | "s_bitset1_b64" => {
                    let ty = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let one = self.s.m.int_const(ty, 1);
                    let k63 = self.s.m.i32(63);
                    let count = |l: &mut Self, x: V| {
                        let x = l.s.m.bin(c, Bin::And, x, k63);
                        l.s.m.cast(c, Cast::ZExt, x, ty)
                    };
                    let r = match name {
                        "s_bfe_u64" | "s_bfe_i64" => {
                            let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                            let a = self.join64(a, b);
                            let off = src32(self, 1)?;
                            let k16 = self.s.m.i32(16);
                            let k127 = self.s.m.i32(127);
                            let w = self.s.m.bin(c, Bin::LShr, off, k16);
                            let w = self.s.m.bin(c, Bin::And, w, k127);
                            self.bitfield(a, off, w, name == "s_bfe_i64", 64)
                        }
                        "s_bfm_b64" => {
                            let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                            let (width, off) = (count(self, a), count(self, b));
                            let m = self.s.m.bin(c, Bin::Shl, one, width);
                            let m = self.s.m.bin(c, Bin::Sub, m, one);
                            self.s.m.bin(c, Bin::Shl, m, off)
                        }
                        _ => {
                            let (a, b) = (old(self, 0)?, old(self, 1)?);
                            let x = self.join64(a, b);
                            let bit = src32(self, 0)?;
                            let bit = count(self, bit);
                            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
                            if name == "s_bitset0_b64" {
                                let ones = self.s.m.int_const(ty, -1);
                                let mask = self.s.m.bin(c, Bin::Xor, bit, ones);
                                self.s.m.bin(c, Bin::And, x, mask)
                            } else {
                                self.s.m.bin(c, Bin::Or, x, bit)
                            }
                        }
                    };
                    let (a, b) = self.split64(r);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    let nz = nonzero(self, a, b);
                    self.set(scc, Val::Bit(nz));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_getpc_b64" => {
                    // Addresses in the code are offsets in it.
                    let next = self.s.m.i32(inst.machine.offset as u32 + 4);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(next));
                    self.set(hi, Val::I32(zero));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                "s_cmov_b64" => {
                    // Keeps the old destination when SCC is clear.
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let a = half(self, 0, h)?;
                        let old = dst_code
                            .and_then(|d| inst.read(Reg::S(d + h)))
                            .ok_or("an unread destination")?;
                        let old = self.i32_of(old)?;
                        let r = self.s.m.select(c, cond, a, old);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cmov_b32" | "s_cmovk_i32" => {
                // Keeps the old destination when SCC is clear.
                let cond = self.scc_in(inst)?;
                let a = if name == "s_cmovk_i32" {
                    sext16(self)
                } else {
                    src32(self, 0)?
                };
                let d = dst.as_ref().ok_or("a missing operand")?;
                let old = self.operand(inst, d)?;
                (self.s.m.select(c, cond, a, old), None)
            }
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff0_i32_b32" | "s_ff1_i32_b32" | "s_bcnt0_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let op0 = src.first().ok_or("a missing operand")?;
                let pair = if matches!(name, "s_ff1_i32_b32" | "s_bcnt1_i32_b32") {
                    self.host_mask_pair(inst, op0)?
                } else {
                    None
                };
                let r = if let Some((a, b)) = pair {
                    if name == "s_ff1_i32_b32" {
                        self.ff1_64(c, a, b)
                    } else {
                        self.bcnt1_64(c, a, b)
                    }
                } else {
                    let mut a = src32(self, 0)?;
                    if matches!(name, "s_ff0_i32_b32" | "s_bcnt0_i32_b32") {
                        let ones = self.s.m.i32(u32::MAX);
                        a = self.s.m.bin(c, Bin::Xor, a, ones);
                    }
                    let op = if name.starts_with("s_ff") { 32 } else { 31 };
                    self.s
                        .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a])
                };
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                // SCC: the sum overflows 32 bits, through bits shifted out
                // of src0 or the add's carry.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let n = u32::from(name.as_bytes()[6] - b'0');
                let k = self.s.m.i32(n);
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                let r = self.s.m.bin(c, Bin::Add, t, b);
                let (rest, z) = (self.s.m.i32(32 - n), self.s.m.i32(0));
                let out = self.s.m.bin(c, Bin::LShr, a, rest);
                let shifted_out = self.s.m.cmp(c, pred::NE, out, z);
                let carry = self.s.m.cmp(c, pred::IULT, r, t);
                (r, Some(self.s.m.bin(c, Bin::Or, shifted_out, carry)))
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                // SCC: src0 is the one chosen.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), Some(t))
            }
            "s_abs_i32" | "s_absdiff_i32" => {
                let mut a = src32(self, 0)?;
                if name == "s_absdiff_i32" {
                    let b = src32(self, 1)?;
                    a = self.s.m.bin(c, Bin::Sub, a, b);
                }
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_sext_i32_i8" | "s_sext_i32_i16" => {
                let a = src32(self, 0)?;
                (
                    self.sign_extend(a, if name.ends_with("i8") { 8 } else { 16 }),
                    None,
                )
            }
            "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hh_b32_b16" => {
                let (mut a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let k16 = self.s.m.i32(16);
                if name == "s_pack_hh_b32_b16" {
                    a = self.s.m.bin(c, Bin::LShr, a, k16);
                }
                if name != "s_pack_ll_b32_b16" {
                    b = self.s.m.bin(c, Bin::LShr, b, k16);
                }
                (self.pack_halves(a, b), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cmov_b32"
                        | "s_cmovk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name.starts_with("s_scratch_") {
            return Err("scalar access to the console scratch aperture");
        }
        if name.contains("_store_") {
            return Err("a scalar store without root UAV address support");
        }
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            if self.scalar.get(&site).is_none()
                && self.x.res.addresses
                && let Some(count) = name.strip_prefix("s_load_dword").and_then(smem_dwords)
                && let Some(Operand::Scalar { code, .. }) = *data
            {
                let Fields::Smem {
                    base: Operand::Scalar { code: at, .. },
                    ..
                } = inst.fields
                else {
                    return Err("a scalar load's address");
                };
                let (lo, hi) = (
                    inst.read(Reg::S(at)).ok_or("a scalar load's address")?,
                    inst.read(Reg::S(at + 1)).ok_or("a scalar load's address")?,
                );
                let mut address = (self.i32_of(lo)?, self.i32_of(hi)?);
                if *offset != 0 {
                    let x = self.s.m.i32(*offset as u32);
                    address = self.add_address(address, x, true);
                }
                if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
                    let so = self.operand(inst, soffset)?;
                    address = self.add_address(address, so, false);
                }
                let xs = self.memory_dwords(address, count, false);
                for (k, x) in xs.into_iter().enumerate() {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            }
            if self.scalar.get(&site).is_none()
                && crate::resource::traces_rays(self.x.p)
                && let Some(count) = name.strip_prefix("s_load_dword").and_then(smem_dwords)
                && let Some(Operand::Scalar { code, .. }) = *data
            {
                // shortcut: out of the acceleration structure.
                let z = self.s.m.i32(0);
                for k in 0..count {
                    self.write(inst, Reg::S(code + k as u8), z, None);
                }
                return Ok(());
            }
            if let Some(a) = self.scalar.get(&site).copied()
                && self
                    .x
                    .native
                    .layout
                    .get(a.param)
                    .is_some_and(|p| p.root == RootType::Descriptor)
            {
                return self.root_smem(inst, site, a);
            }
            return self.descriptor_data(inst, site);
        };
        let count = smem_dwords(rest).ok_or("a scalar buffer load")?;
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        if self.by_address(src)
            && let Fields::Smem { base, .. } = &inst.fields
        {
            let (h, start, _) = self.made_buffer(inst, base)?;
            let mut addr = self.s.m.i32(*offset as u32);
            if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
                let so = self.operand(inst, soffset)?;
                addr = self.s.m.bin(c, Bin::Add, addr, so);
            }
            let addr = self.s.m.bin(c, Bin::Add, addr, start);
            let words = self.view_dwords(h, Some(0), addr, None, count);
            for (k, x) in words.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // A raw or structured buffer view (an SRV, or a UAV the code also
        // reads through scalar memory), else a constant buffer view.
        let raw = kind
            .filter(|u| matches!(u.kind, ResourceType::Srv | ResourceType::Uav))
            .map(|u| {
                let u_bit = if u.kind == ResourceType::Uav {
                    0x1000
                } else {
                    0
                };
                let stride = if u.stride >= 4
                    && u.stride.is_multiple_of(4)
                    && !u.stride.is_multiple_of(16)
                {
                    u.stride
                } else {
                    0
                };
                (u_bit, stride)
            });
        let props = match raw {
            Some((u_bit, 0)) => (11 | u_bit, 0),
            Some((u_bit, stride)) => (12 | u_bit, stride),
            None => (13, 0x1_0000),
        };
        let h = self.handle(site, src, false, props)?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let words = self.view_dwords(h, raw.map(|(_, s)| s), addr, constant, count);
        for (k, x) in words.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// `count` dwords from byte `addr` (`constant` when known) of a raw
    /// buffer (`raw` = `Some(0)`), a structured buffer of that stride
    /// (`raw` = `Some(stride)`) or a constant buffer view (`None`) `h`. The
    /// address's two low bits are ignored, as scalar memory does.
    fn view_dwords(
        &mut self,
        h: V,
        raw: Option<u32>,
        addr: V,
        constant: Option<u32>,
        count: u32,
    ) -> Vec<V> {
        let c = self.cur;
        let mut out = Vec::new();
        if let Some(stride) = raw {
            let not3 = self.s.m.i32(!3);
            let addr = self.s.m.bin(c, Bin::And, addr, not3);
            let st = (stride != 0).then(|| self.s.m.i32(stride));
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let (idx, elem_off) = match st {
                    Some(st) => (
                        self.s.m.bin(c, Bin::UDiv, a, st),
                        self.s.m.bin(c, Bin::URem, a, st),
                    ),
                    None => (a, self.s.m.undef(self.s.t.i32)),
                };
                let mask = self.s.m.i8(1);
                let al = self.s.m.i32(4);
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, idx, elem_off, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                out.push(self.as_int(x));
            }
            return out;
        }
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base.wrapping_add(4 * k);
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            out.push(x);
        }
        out
    }

    /// `s_load_dword*` from the memory of a root CBV or SRV: constants the
    /// code reads through scalar memory at an address it computes from the
    /// root descriptor's, read through the root view (a raw buffer for an
    /// SRV, a constant buffer otherwise).
    // shortcut: like the console's scalar loads, the host's root views are
    // not bounds-checked, so a read past the view's data is undefined on
    // both.
    fn root_smem(&mut self, inst: &Inst, site: Site, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = inst
            .name()
            .strip_prefix("s_load_dword")
            .and_then(smem_dwords)
            .ok_or("a scalar memory read as data")?;
        let src = Source::Built {
            address: a,
            bits: 0,
            words: [0, 0],
        };
        let srv = self
            .used_kind(&src)
            .is_some_and(|u| u.kind == ResourceType::Srv);
        let props = if srv { (11, 0) } else { (13, 0x1_0000) };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = self.dynamic_term(&a)? {
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        let constant = a.dynamic.is_none().then_some(a.offset);
        let words = self.view_dwords(h, srv.then_some(0), addr, constant, count);
        for (k, x) in words.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// Two scalar loads of the same dwords: same opcode and immediate
    /// offset, base read from the same values, and the same or an
    /// equivalent register offset.
    fn twin_loads(&self, a: Site, b: Site) -> bool {
        let (ia, ib): (&Inst, &Inst) = (
            &self.x.p.blocks[a.0].insts[a.1],
            &self.x.p.blocks[b.0].insts[b.1],
        );
        let (
            Fields::Smem {
                base: Operand::Scalar { code: ba, .. },
                offset: oa,
                soffset: sa,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: bb, .. },
                offset: ob,
                soffset: sb,
                ..
            },
        ) = (&ia.fields, &ib.fields)
        else {
            return false;
        };
        let same_base = (0..2).all(|k| {
            let va = ia.read(Reg::S(ba + k));
            va.is_some() && va == ib.read(Reg::S(bb + k))
        });
        let same_soffset = match (sa, sb) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: ca, .. }, Operand::Scalar { code: cb, .. }) => {
                match (ia.read(Reg::S(*ca)), ib.read(Reg::S(*cb))) {
                    (Some(x), Some(y)) => self.same_scalar(x, y),
                    _ => false,
                }
            }
            _ => false,
        };
        ia.name() == ib.name() && oa == ob && same_base && same_soffset
    }

    /// The same value, or two `v_readfirstlane_b32` of the same VGPR value
    /// (each waterfall loop reads the index anew).
    fn same_scalar(&self, x: Value, y: Value) -> bool {
        if x == y {
            return true;
        }
        let first_lane_of = |v: Value| {
            self.x
                .p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.writes.iter().any(|w| w.1 == v))
                .filter(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| {
                    i.reads
                        .iter()
                        .find(|r| matches!(r.0, Reg::V(_)))
                        .map(|r| r.1)
                })
        };
        matches!((first_lane_of(x), first_lane_of(y)), (Some(a), Some(b)) if a == b)
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program. A cube reads as its six faces (not a cube array).
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = inst
            .name()
            .strip_prefix("s_load_dword")
            .and_then(smem_dwords)
            .ok_or("a scalar memory read as data")?;
        // How the program uses the descriptor tells the view's kind: this
        // load's dwords, or those of another load of the same descriptor
        // (the code often loads the whole descriptor for the resource and a
        // dword or two of it again for its size). `skew` is the dword of
        // the descriptor this load's first dword is; another load's
        // descriptor counts only if this load starts inside it. Prefer an
        // image or vector buffer use when a descriptor has both scalar and
        // vector uses, since those carry format and indexing cues.
        let find = |own: bool| {
            let mut best = None;
            for d in self
                .x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
            {
                for l in d.source.leaves() {
                    let Source::Loaded { load, first } = *l else {
                        continue;
                    };
                    if (load == site) != own {
                        continue;
                    }
                    // A twin load of the same dwords (a non-uniform index
                    // reloads the descriptor in each waterfall loop, one for
                    // the size bits, another for the access) is at skew 0.
                    let Some(skew) = self
                        .load_skew(inst, site, load)
                        .or_else(|| (!own && self.twin_loads(site, load)).then_some(0))
                        .map(|s| s - i32::from(first))
                    else {
                        continue;
                    };
                    if !own && !(0..8).contains(&skew) {
                        continue;
                    }
                    let ui = &self.x.p.blocks[d.site.0].insts[d.site.1];
                    let found = (d.site, load, first, skew);
                    if matches!(ui.fields, Fields::Image { .. } | Fields::Buffer { .. }) {
                        return Some(found);
                    }
                    best.get_or_insert(found);
                }
            }
            best
        };
        let use_at = find(true).or_else(|| find(false));
        let (load, skew, src, mut props, image, uav) = if let Some((at, load, first, skew)) = use_at
        {
            let src = Source::Loaded { load, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        3 => 5,
                        4 => 6,
                        5 => 7,
                        6 => 3,
                        7 => 8,
                        _ => return Err("an image dimension"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        (10 | u, 0x409)
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4)
                    {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                Fields::Smem { .. } if un.starts_with("s_buffer_") => {
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if stride >= 4 && stride.is_multiple_of(4) {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    (props, None)
                }
                _ => return Err("a scalar memory read as data"),
            };
            (load, skew, src, props, image, uav)
        } else {
            // An image the program only measures (it reads the size bits and
            // samples the image in other programs): eight dwords are an image
            // descriptor, taken here as a 2D texture.
            // shortcut: a 3D or array image would report depth or layers 1.
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let used = self.used_kind(&src);
            if count == 8 && !used.is_some_and(|u| u.kind != ResourceType::Srv) {
                (site, 0, src, (2, 0x409), Some(2), false)
            } else {
                // A buffer the program only counts the records of (#329):
                // an exact statically addressed four-dword binding of a
                // structured buffer the container declares, never the
                // nearest earlier entry or a dynamically indexed range.
                let uav = used.is_some_and(|u| u.kind == ResourceType::Uav);
                let address = self
                    .scalar
                    .get(&site)
                    .ok_or("a descriptor read only as data")?;
                let Some(binding) = used.filter(|b| {
                    count == 4
                        && address.dynamic.is_none()
                        && b.table_offset_dwords.checked_mul(4) == Some(address.offset)
                        && b.dwords == 4
                        && matches!(b.kind, ResourceType::Srv | ResourceType::Uav)
                        && (4..=2048).contains(&b.stride)
                        && b.stride.is_multiple_of(4)
                        && self.x.psv.is_some_and(|part| {
                            super::declares_structured_buffer(part, b.lower_bound, uav)
                        })
                }) else {
                    return Err("a descriptor read only as data");
                };
                let u = if uav { 0x1000 } else { 0 };
                (site, 0, src, (12 | u, binding.stride), None, uav)
            }
        };
        // A structured buffer's records, of a raw view's bytes.
        let records = (props.0 & 0xfff == 12 && props.1.is_multiple_of(16)).then_some(props.1);
        if records.is_some() {
            props = (11 | props.0 & 0x1000, 0);
        }
        // A typed or structured buffer's element count comes from the host's
        // table of them by slot: the AMD driver's compiler crashes on a size
        // query of a structured buffer from the heap (stride 60, nine
        // compute programs of the world) and takes minutes over hundreds of
        // a typed one's; the table holds the title's own count.
        // (A structured buffer the host views raw keeps the query, of bytes.)
        if image.is_none() && records.is_none() && matches!(props.0 & 0xfff, 10 | 12) {
            let slot = if load == site {
                self.heap_slot(&src, false)?
            } else {
                self.heap_index_at(site, skew.wrapping_neg(), false)?
            };
            let n = self.slot_dword(slot, super::OWN_SLOTS);
            for k in 0..count {
                if (k as i32).wrapping_add(skew) == 2 {
                    self.write(inst, Reg::S(code + k as u8), n, None);
                } else if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                    self.vals.insert(v, Val::Missing(OTHER));
                }
            }
            return Ok(());
        }
        // The handle is built from this load's own address (the other
        // load's registers need not hold it here): the descriptor starts
        // `skew` dwords before what this load reads.
        let h = if load == site {
            self.handle(site, &src, false, props)?
        } else {
            let index = self.heap_index_at(site, skew.wrapping_neg(), false)?;
            self.s.heap_handle(self.cur, index, false, false, props)
        };
        // The size words are asked of the host only when the program reads
        // one as data: most loads take the descriptor whole for an image or
        // buffer instruction, and hundreds of size queries of heap handles
        // in one program stop the AMD driver's compiler (minutes, then
        // E_FAIL: nine compute programs of the world).
        let sized = if image.is_some() { 1..=4 } else { 2..=2 };
        let wanted = (0..count).any(|k| {
            sized.contains(&(k as i32).wrapping_add(skew))
                && inst
                    .write(Reg::S(code + k as u8))
                    .is_some_and(|v| self.read_as_data(v))
        });
        if !wanted {
            for k in 0..count {
                if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                    self.vals.insert(v, Val::Missing(OTHER));
                }
            }
            return Ok(());
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some_and(|k| !matches!(k, 3 | 8)) && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 | 3 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    // A cube is six faces (DXIL's dimensions have no depth).
                    5 => (field(self, 1), self.s.m.i32(6)),
                    _ => (field(self, 1), field(self, 2)),
                };
                // The level count only feeds word 3. A level query of a 3D
                // image inside a loop crashes NVIDIA's Vulkan compiler (vkd3d-
                // proton turns it into OpImageQueryLevels), so query it only
                // when the program reads that word as data.
                let reads_word3 = (0..count)
                    .filter(|&k| (k as i32).wrapping_add(skew) == 3)
                    .filter_map(|k| inst.write(Reg::S(code + k as u8)))
                    .any(|v| self.read_as_data(v));
                let levels = if uav || matches!(kind, 3 | 8) || !reads_word3 {
                    one
                } else {
                    field(self, 3)
                };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 14, 10, 11, 12, 13, 15][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => {
                let mut n = field(self, 0);
                if let Some(stride) = records {
                    let st = self.s.m.i32(stride);
                    n = self.s.m.bin(c, Bin::UDiv, n, st);
                }
                vec![(2, n)]
            }
        };
        for k in 0..count {
            let rel = (k as i32).wrapping_add(skew);
            match words.iter().find(|w| w.0 as i32 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// The dword of what the scalar load at `other` reads that `inst` (at
    /// `site`) reads first: the same base and offset registers, immediates
    /// a whole number of dwords apart, or else traced addresses that are
    /// (the same parameter and computed term: the code often builds the
    /// table pointer `{table, 0}` again before the second load); none for
    /// loads that differ otherwise.
    fn load_skew(&self, inst: &Inst, site: Site, other: Site) -> Option<i32> {
        if other == site {
            return Some(0);
        }
        if let (Some(a), Some(b)) = (self.scalar.get(&site), self.scalar.get(&other))
            && (a.param, a.dynamic, a.includes_base) == (b.param, b.dynamic, b.includes_base)
            && a.offset.wrapping_sub(b.offset) % 4 == 0
        {
            return Some(a.offset.wrapping_sub(b.offset) as i32 / 4);
        }
        let o: &Inst = &self.x.p.blocks[other.0].insts[other.1];
        let (
            Fields::Smem {
                base: Operand::Scalar { code: b0, .. },
                offset: f0,
                soffset: s0,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: b1, .. },
                offset: f1,
                soffset: s1,
                ..
            },
        ) = (&inst.fields, &o.fields)
        else {
            return None;
        };
        let reads = |i: &Inst, code: u8| (i.read(Reg::S(code)), i.read(Reg::S(code + 1)));
        let same_soffset = match (s0, s1) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: c0, .. }, Operand::Scalar { code: c1, .. }) => {
                inst.read(Reg::S(*c0)).is_some() && inst.read(Reg::S(*c0)) == o.read(Reg::S(*c1))
            }
            _ => false,
        };
        let delta = f0.wrapping_sub(*f1);
        (reads(inst, *b0).0.is_some()
            && reads(inst, *b0) == reads(o, *b1)
            && same_soffset
            && delta % 4 == 0)
            .then_some(delta / 4)
    }

    /// Whether an instruction or phi reads `v` other than as a resource
    /// operand of a memory instruction (which takes the descriptor whole).
    fn read_as_data(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| {
                    !matches!(
                        i.fields,
                        Fields::Image { .. } | Fields::Buffer { .. } | Fields::Smem { .. }
                    ) && i.reads.iter().any(|r| r.1 == v)
                })
        })
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = inst
            .name()
            .strip_prefix("s_load_dword")
            .and_then(smem_dwords)
            .ok_or("a scalar load from the code")?;
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = self.dynamic_term(&a)? {
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    // DXIL's division may be off by an ulp (65535 / 65535 =
                    // 0.99999994 on AMD); the format gives exactly 1.0 for
                    // the largest value, which an f16 conversion keeps.
                    let top = self.s.m.i32(((1u64 << w) - 1) as u32);
                    let full = self.s.m.cmp(c, pred::EQ, field, top);
                    let one = self.s.m.f32(1.0f32.to_bits());
                    let q = self.s.m.select(c, full, one, q);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// Channels of buffer format `format`, as [`Self::format_load`] gives
    /// them, as d16 data: 16 bits each, two per dword, low half first (an
    /// odd last high half 0). Integer formats keep their low 16 bits; the
    /// others become f16.
    fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (_, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::new();
        for x in channels {
            halves.push(if matches!(numeric, Numeric::Uint | Numeric::Sint) {
                let m = self.s.m.i32(0xFFFF);
                self.s.m.bin(c, Bin::And, x, m)
            } else {
                let (f, i32t) = (self.as_float(x), self.s.t.i32);
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            });
        }
        let mut out = Vec::new();
        for pair in halves.chunks(2) {
            out.push(match *pair {
                [lo, hi] => {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                }
                [lo] => lo,
                _ => unreachable!("chunks of one or two"),
            });
        }
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = self.dynamic_term(address)?;
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The byte offset of element `index` at `off` in a buffer whose
    /// descriptor the code builds around a root descriptor; none for other
    /// descriptors. GFX10 addresses base + offset + stride * index with
    /// swizzling off, the stride in dword 1 bits 16-29 (the constant bits the
    /// code sets there), swizzling in dword 1 bit 31 and the lane id added to
    /// the index in dword 3 bit 23.
    fn built_element(&mut self, src: &Source, index: V, off: V) -> Result<Option<V>, Why> {
        let Source::Built { bits, words, .. } = src else {
            return Ok(None);
        };
        if bits >> 31 != 0 {
            return Err("a swizzled buffer");
        }
        if words[1] >> 23 & 1 != 0 {
            return Err("a buffer indexed by lane");
        }
        let stride = bits >> 16 & 0x3FFF;
        let c = self.cur;
        let s = self.s.m.i32(stride);
        let rows = self.s.m.bin(c, Bin::Mul, index, s);
        Ok(Some(self.s.m.bin(c, Bin::Add, rows, off)))
    }

    /// Whether a buffer is accessed by its address (see
    /// [`crate::resource::track_with`]): no root view leads to it, or one
    /// that is written through.
    fn by_address(&self, src: &Source) -> bool {
        match src {
            Source::Address => true,
            Source::Built { address, .. } => {
                self.x.res.addresses
                    && address.param != CODE
                    && self
                        .x
                        .native
                        .layout
                        .get(address.param)
                        .is_some_and(|rp| rp.root == RootType::Descriptor)
                    && !self
                        .used_kind(src)
                        .is_some_and(|u| u.kind == ResourceType::Cbv)
            }
            _ => false,
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. }
        | Source::Select { args, .. }
        | Source::Lanes { args, .. }
        | Source::Paths { args, .. } = src
        {
            return args.iter().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        if let Some(&h) = self.made.get(&site).filter(|_| self.by_address(src)) {
            return Ok(h);
        }
        match src {
            Source::Address => Err("a buffer descriptor made in code"),
            Source::Loaded { load, first } => {
                // The element type is part of the key: one UAV read as float
                // and used atomically needs two handles (an atomic on a float
                // handle is invalid and crashed the NVIDIA compiler).
                let key = (self.cur, *load, *first, sampler, props);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                let h = self.s.heap_handle(self.cur, index, sampler, false, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if props.0 & 0x8000 != 0 {
                    // shortcut: one sampler used with and without comparison
                    // would need two static samplers; it becomes a comparison one.
                    self.s.cmp_samplers |= 1 << n;
                }
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        let h = self.s.root_srv_handle(rp.slot);
                        // A draw with wild vertex numbers must not read past
                        // the memory behind the view (a page fault).
                        if super::limits_indices()
                            && (self.s.threads.is_none() || self.s.mesh.is_some())
                            && !self.s.limits.iter().any(|l| l.0 == h)
                        {
                            let nth = 1 + self.s.limits.len() as u32;
                            let limit = self.host_data(nth, RootType::LIMIT, rp.slot.into());
                            self.s.limits.push((h, limit));
                        }
                        Ok(h)
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, 0x100B)
                    | (RootType::Descriptor, Some(ResourceType::Uav), 11) => {
                        // Other groups write what a builder reads.
                        let shared = self.ordered();
                        Ok(self.s.root_uav_handle(rp.slot, shared))
                    }
                    _ => Err("a root UAV or typed root view"),
                }
            }
            // shortcut: of samplers given in the code that a branch picks one
            // of (how they treat coordinates off the texture differs), the
            // first is taken.
            Source::Phi { args, .. }
                if sampler
                    && let Some(arg) = args.iter().find(|a| matches!(a, Source::Constant(_))) =>
            {
                self.handle(site, arg, sampler, props)
            }
            Source::Phi { block, args } => {
                // Each predecessor computes its heap index at its end; the
                // join picks one with a phi.
                let _ = site;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let Some(tail) = self.tail[q] else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = match arg {
                        Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
                        Source::Phi { .. } | Source::Select { .. } => self.heap_slot(arg, sampler),
                        Source::Unset => Ok(self.s.m.i32(0)),
                        _ => {
                            if std::env::var_os("VICTORY_WHY").is_some() {
                                eprintln!("phi of {arg:?}");
                            }
                            Err("a descriptor chosen by control flow")
                        }
                    };
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(self.s.heap_handle(self.cur, phi, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::Paths { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let io = &self.x.io.inputs;
                if self
                    .x
                    .res
                    .waterfall
                    .unset
                    .iter()
                    .any(|u| io.iter().any(|i| i.0 == *u))
                {
                    return Err("a per-lane descriptor joined with a VGPR the hardware sets");
                }
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
            Source::Carried { .. } | Source::Unset => Err("a per-lane descriptor out of a join"),
        }
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block.
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::Paths { block, args } => self.paths_index(*block, args, sampler),
            Source::Carried { block } => self
                .lane_paths
                .iter()
                .rev()
                .find(|p| p.0 == *block)
                .map(|p| p.1)
                .ok_or("a per-lane descriptor carried out of its loop"),
            // Undefined: any index will do.
            Source::Unset => Ok(self.s.m.i32(0)),
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of a descriptor joined per lane at `block`: a phi of
    /// each predecessor's index, computed at its end (later for those not
    /// lowered yet, as a loop's back edge).
    fn paths_index(&mut self, block: usize, args: &[Source], sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor joined by control flow";
        let join = self.block[block].ok_or(WHY)?;
        let phi = self.s.m.phi(join, self.s.t.i32);
        self.lane_paths.push((block, phi));
        let here = self.cur;
        let mut incoming = Vec::new();
        let mut result = Ok(());
        for (&q, arg) in self.x.p.blocks[block].preds.iter().zip(args) {
            if self.block[q].is_none() {
                continue;
            }
            let Some(tail) = self.tail[q] else {
                incoming.push((q, Err(arg.clone())));
                continue;
            };
            self.cur = self.edges.get(&(q, block)).copied().unwrap_or(tail);
            let index = self.lane_heap_index(arg, sampler);
            match index {
                Ok(x) => incoming.push((q, Ok(x))),
                Err(why) => {
                    result = Err(why);
                    break;
                }
            }
        }
        self.cur = here;
        let outer = self.lane_paths.clone();
        self.lane_paths.pop();
        result?;
        if incoming.iter().all(|i| i.1.is_ok()) {
            let done = incoming
                .into_iter()
                .map(|(q, x)| (x.unwrap(), self.edge_block(q, block)));
            let done = done.collect();
            self.s.m.set_phi(phi, done);
        } else {
            self.pending_paths.push(PendingPath {
                phi,
                block,
                sampler,
                outer,
                incoming,
            });
        }
        Ok(phi)
    }

    /// The DXIL block that ends the edge from IR block `q` (lowered) to `to`.
    fn edge_block(&self, q: usize, to: usize) -> usize {
        let tail = self.tail[q].expect("a lowered predecessor");
        self.edges.get(&(q, to)).copied().unwrap_or(tail)
    }

    /// Fills the heap index phis [`Self::paths_index`] left for later.
    fn fill_paths(&mut self) -> Result<(), Why> {
        while let Some(pending) = self.pending_paths.pop() {
            let saved = std::mem::replace(&mut self.lane_paths, pending.outer);
            let mut incoming = Vec::new();
            for (q, x) in pending.incoming {
                let at = self.edge_block(q, pending.block);
                let x = match x {
                    Ok(x) => x,
                    Err(arg) => {
                        self.cur = at;
                        self.lane_heap_index(&arg, pending.sampler)?
                    }
                };
                incoming.push((x, at));
            }
            self.lane_paths = saved;
            self.s.m.set_phi(pending.phi, incoming);
        }
        Ok(())
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        if self.x.st.guards[load.0][load.1].is_some() {
            // Lanes outside EXEC there keep their old dwords.
            return Err("a per-lane descriptor loaded under a partial EXEC");
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        if matches!(li.fields, Fields::Buffer { .. }) {
            // A buffer over the heap or a table: its address (the table's
            // byte offset, or 0 for a heap) plus each lane's offset.
            let c = self.cur;
            let mut at = match self.user(a.param, 0) {
                Val::I32(x) => x,
                _ => return Err(WHY),
            };
            let k = self.s.m.i32(a.offset.wrapping_add(4 * u32::from(first)));
            at = self.s.m.bin(c, Bin::Add, at, k);
            if let Some(d) = self.dynamic_term(&a)? {
                at = self.s.m.bin(c, Bin::Add, at, d);
            }
            let shift = self.s.m.i32(if sampler { 4 } else { 5 });
            return Ok(self.s.m.bin(c, Bin::LShr, at, shift));
        }
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The view-heap slot of `src` when it comes from a descriptor table or
    /// heap (`Loaded` or `Phi` of `Loaded`).
    fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let Some(tail) = self.tail[q] else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = match arg {
                        Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
                        Source::Phi { .. } | Source::Select { .. } => self.heap_slot(arg, sampler),
                        Source::Unset => Ok(self.s.m.i32(0)),
                        _ => {
                            if std::env::var_os("VICTORY_WHY").is_some() {
                                eprintln!("phi of {arg:?}");
                            }
                            Err("a descriptor chosen by control flow")
                        }
                    };
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            // A descriptor chosen per lane or by SCC: its index in each lane.
            _ => self.lane_heap_index(src, sampler),
        }
    }

    /// Byte offset `index * stride + off` for a structured buffer whose
    /// stride was not recovered statically, reading `stride` from the host's
    /// runtime stride table (`t0`, [`super::STRIDE_SPACE`]).
    fn runtime_stride_offset(&mut self, src: &Source, index: V, off: V) -> Result<V, Why> {
        let slot = self
            .heap_slot(src, false)
            .map_err(|_| "a structured buffer of unknown stride")?;
        let c = self.cur;
        let stride = self.slot_dword(slot, 0);
        let elem_off = self.s.m.bin(c, Bin::Mul, index, stride);
        Ok(self.s.m.bin(c, Bin::Add, elem_off, off))
    }

    /// What the host's table by heap slot (`t0`, [`super::STRIDE_SPACE`])
    /// holds for `slot`, `from` dwords on: the element stride of its buffer
    /// view from 0, the element count from [`super::OWN_SLOTS`].
    fn slot_dword(&mut self, slot: V, from: u32) -> V {
        let c = self.cur;
        let table = self.s.stride_table_handle();
        // A root view has no bounds: a wild index would read past the table.
        let slot = if super::limits_indices() || from != 0 {
            let (i32t, last) = (self.s.t.i32, self.s.m.i32(super::OWN_SLOTS - 1));
            self.s
                .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last])
        } else {
            slot
        };
        let slot = if from != 0 {
            let from = self.s.m.i32(from);
            self.s.m.bin(c, Bin::Add, slot, from)
        } else {
            slot
        };
        let four = self.s.m.i32(4);
        let addr = self.s.m.bin(c, Bin::Mul, slot, four);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let res = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, addr, iu, mask, align],
        );
        let x = self.s.m.extract(c, res, 0);
        self.as_int(x)
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        self.heap_index_at(load, i32::from(first), sampler)
    }

    /// The heap index of the descriptor `dwords` dwords from the first one
    /// the scalar load at `load` reads (before it if negative), computed
    /// in the current block.
    fn heap_index_at(&mut self, load: Site, dwords: i32, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4u32.wrapping_mul(dwords as u32)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }

}
