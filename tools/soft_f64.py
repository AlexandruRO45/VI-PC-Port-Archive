"""--f64-soft: binary64 arithmetic on (lo, hi) pairs of 32-bit words, for devices without shaderFloat64.

Metal has no double type, so MoltenVK reports shaderFloat64 false and a module that declares Float64 cannot run there
(nor on Intel Arc, which has neither shaderFloat64 nor shaderInt64). Under --f64-soft the translator keeps every
binary64 value as the two 32-bit words of its register pair and computes with 32-bit integer instructions only, so
the lowering adds neither the Float64 nor the Int64 capability (a module may still declare Int64 for something else,
such as the 64-bit physical buffer addresses of scalar loads, as it does without the flag). Each operation is one
SPIR-V function, emitted once per module the first time it is used and called with OpFunctionCall, so a module holds
one copy however many instructions use it. The bodies are straight-line code (OpSelect, no branches).

The algorithms are Berkeley SoftFloat 3's (softfloat_addMagsF64 / subMagsF64, softfloat_roundPackToF64,
f64_to_f32 / softfloat_roundPackToF32), written with 32-bit halves. A significand travels as a 64-bit (hi, lo)
pair with its leading bit at bit 62 and the rounding bits in bits 9..0. Rounding follows RDNA2's MODE.FP_ROUND
encoding: 0 nearest even, 1 toward +inf, 2 toward -inf, 3 toward zero. Denormals are kept (MODE.FP_DENORM for
f64 = 3, the default and what the translator assumes for f64/f16). NaN: an operand NaN comes back quiet (bit 51 set)
with its sign and payload, src0's first; inf - inf gives the default NaN 0x7ff80000_00000000. RDNA2's own NaN
payload for f64 is not measured, and SPIR-V does not define the payload on the native path either.

Every shift amount is reduced to 0..31 before the shift: SPIR-V (and MSL) leave a shift by the bit width or more
undefined.
"""

F64_FN_TYPES = {
    # function type id -> OpTypeFunction operands (return type first)
    "%fn_f64s_add": "%v2uint_d %uint %uint %uint %uint %uint",
    "%fn_f64s_scale": "%v2uint_d %uint %uint %uint",
    "%fn_f64s_to_f32": "%uint %uint %uint",
}
# SPIR-V allows one OpTypeFunction per signature: a helper with the signature of another uses that one's type.
FN_TYPE = {"%f64s_add": "%fn_f64s_add", "%f64s_scale": "%fn_f64s_scale", "%f64s_to_f32": "%fn_f64s_to_f32",
           "%f64s_scale_f32": "%fn_f64s_to_f32"}


class SoftF64:
    """Emits the soft-f64 helper functions into a Gen (rdna2spirv.Gen): uses its tmp(), cu() and emit()."""

    def __init__(self, gen):
        self.g = gen
        self.fns = {}       # function id -> list of SPIR-V lines

    # ---- small builders -------------------------------------------------------------------------------------
    def _op(self, ty, op, *args):
        r = self.g.tmp()
        self.g.emit(f"{r} = {op} {ty} " + " ".join(args))
        return r

    def u(self, op, *args):
        return self._op("%uint", op, *args)

    def b(self, op, *args):
        return self._op("%bool", op, *args)

    def c(self, v):
        return self.g.cu(v)

    def sel(self, cond, x, y):
        return self.u("OpSelect", cond, x, y)

    def bsel(self, cond, x, y):
        return self.b("OpSelect", cond, x, y)

    def nz(self, x):
        return self.b("OpINotEqual", x, self.c(0))

    def eq(self, x, y):
        return self.b("OpIEqual", x, y)

    def band(self, *xs):
        r = xs[0]
        for x in xs[1:]:
            r = self.b("OpLogicalAnd", r, x)
        return r

    def bor(self, *xs):
        r = xs[0]
        for x in xs[1:]:
            r = self.b("OpLogicalOr", r, x)
        return r

    def shl(self, x, n):
        return self.u("OpShiftLeftLogical", x, n)

    def shr(self, x, n):
        return self.u("OpShiftRightLogical", x, n)

    def and_(self, x, y):
        return self.u("OpBitwiseAnd", x, y)

    def or_(self, x, y):
        return self.u("OpBitwiseOr", x, y)

    def add(self, x, y):
        return self.u("OpIAdd", x, y)

    def sub(self, x, y):
        return self.u("OpISub", x, y)

    # ---- 64-bit values as (hi, lo) ----------------------------------------------------------------------------
    def add64(self, ah, al, bh, bl):
        lo = self.add(al, bl)
        carry = self.sel(self.b("OpULessThan", lo, al), self.c(1), self.c(0))
        return self.add(self.add(ah, bh), carry), lo

    def sub64(self, ah, al, bh, bl):
        lo = self.sub(al, bl)
        borrow = self.sel(self.b("OpULessThan", al, bl), self.c(1), self.c(0))
        return self.sub(self.sub(ah, bh), borrow), lo

    def shl64(self, h, l, n):
        """(h, l) << n for 0 <= n < 64."""
        m = self.and_(n, self.c(31))
        inv = self.and_(self.sub(self.c(32), m), self.c(31))
        carry = self.sel(self.eq(m, self.c(0)), self.c(0), self.shr(l, inv))
        small_h = self.or_(self.shl(h, m), carry)
        small_l = self.shl(l, m)
        big = self.b("OpUGreaterThanEqual", n, self.c(32))
        return self.sel(big, small_l, small_h), self.sel(big, self.c(0), small_l)

    def shr64_jam(self, h, l, n):
        """(h, l) >> n for any n, with every bit shifted out ORed into bit 0 (SoftFloat's shiftRightJam64)."""
        m = self.and_(n, self.c(31))
        inv = self.and_(self.sub(self.c(32), m), self.c(31))
        m0 = self.eq(m, self.c(0))
        # n < 32
        a_l = self.or_(self.shr(l, m), self.sel(m0, self.c(0), self.shl(h, inv)))
        a_h = self.shr(h, m)
        a_lost = self.band(self.b("OpLogicalNot", m0), self.nz(self.shl(l, inv)))
        # 32 <= n < 64
        b_l = self.shr(h, m)
        b_lost = self.bor(self.nz(l), self.band(self.b("OpLogicalNot", m0), self.nz(self.shl(h, inv))))
        # n >= 64
        c_lost = self.nz(self.or_(h, l))
        lt32 = self.b("OpULessThan", n, self.c(32))
        lt64 = self.b("OpULessThan", n, self.c(64))
        rl = self.sel(lt32, a_l, self.sel(lt64, b_l, self.c(0)))
        rh = self.sel(lt32, a_h, self.c(0))
        lost = self.bsel(lt32, a_lost, self.bsel(lt64, b_lost, c_lost))
        return rh, self.or_(rl, self.sel(lost, self.c(1), self.c(0)))

    def clz64(self, h, l):
        """Leading zeros of (h, l); 64 for zero. FindUMsb(0) is -1, so 31 - FindUMsb(0) = 32."""
        zh = self.sub(self.c(31), self.u("OpExtInst", "%glsl", "FindUMsb", h))
        zl = self.sub(self.c(31), self.u("OpExtInst", "%glsl", "FindUMsb", l))
        return self.sel(self.nz(h), zh, self.add(self.c(32), zl))

    # ---- rounding ---------------------------------------------------------------------------------------------
    def round_pack(self, sign, exp, sh, sl, mode):
        """softfloat_roundPackToF64: the binary64 words (hi, lo) nearest, per `mode`, to sig * 2**(exp - 1084),
        where sig = (sh, sl) has its leading bit at bit 62 (or is a denormal's, with exp 0) and exp is a signed
        32-bit integer. `sign` is 0 or 1."""
        neg = self.nz(sign)
        rne = self.eq(mode, self.c(0))
        up = self.bor(self.band(self.eq(mode, self.c(1)), self.b("OpLogicalNot", neg)),
                      self.band(self.eq(mode, self.c(2)), neg))
        inc = self.sel(rne, self.c(0x200), self.sel(up, self.c(0x3FF), self.c(0)))
        # below the normal range: shift right (jamming) into a denormal, exponent 0
        tiny = self.b("OpSLessThan", exp, self.c(0))
        th, tl = self.shr64_jam(sh, sl, self.sub(self.c(0), exp))
        sh, sl = self.sel(tiny, th, sh), self.sel(tiny, tl, sl)
        exp = self.sel(tiny, self.c(0), exp)
        round_bits = self.and_(sl, self.c(0x3FF))
        ih, il = self.add64(sh, sl, self.c(0), inc)
        carry63 = self.nz(self.and_(ih, self.c(0x80000000)))
        over = self.bor(self.b("OpSGreaterThan", exp, self.c(0x7FD)),
                        self.band(self.eq(exp, self.c(0x7FD)), carry63))
        # (sig + inc) >> 10; a tie under nearest even clears the last bit
        rl = self.or_(self.shr(il, self.c(10)), self.shl(ih, self.c(22)))
        rh = self.shr(ih, self.c(10))
        tie = self.band(rne, self.eq(round_bits, self.c(0x200)))
        rl = self.sel(tie, self.and_(rl, self.c(0xFFFFFFFE)), rl)
        zero = self.b("OpLogicalNot", self.nz(self.or_(rh, rl)))
        exp = self.sel(zero, self.c(0), exp)
        sbit = self.shl(sign, self.c(31))
        hi = self.add(self.add(sbit, self.shl(exp, self.c(20))), rh)
        # overflow: infinity, or the largest finite value when the mode rounds toward zero from this side
        inf_hi = self.or_(sbit, self.c(0x7FF00000))
        to_max = self.eq(inc, self.c(0))
        ov_hi = self.sel(to_max, self.or_(sbit, self.c(0x7FEFFFFF)), inf_hi)
        ov_lo = self.sel(to_max, self.c(0xFFFFFFFF), self.c(0))
        return self.sel(over, ov_hi, hi), self.sel(over, ov_lo, rl)

    def unpack(self, lo, hi):
        """Fields of one binary64: sign (0/1), biased exponent field, the 53-bit significand << 9 as (h, l) with
        the implicit bit for a normal number, the effective exponent max(e, 1), NaN and infinity flags."""
        sign = self.shr(hi, self.c(31))
        e = self.and_(self.shr(hi, self.c(20)), self.c(0x7FF))
        fh = self.and_(hi, self.c(0xFFFFF))
        frac_nz = self.nz(self.or_(fh, lo))
        e_max = self.eq(e, self.c(0x7FF))
        nan = self.band(e_max, frac_nz)
        inf = self.band(e_max, self.b("OpLogicalNot", frac_nz))
        enz = self.nz(e)
        sigh = self.or_(fh, self.sel(enz, self.c(0x100000), self.c(0)))
        h9 = self.or_(self.shl(sigh, self.c(9)), self.shr(lo, self.c(23)))
        l9 = self.shl(lo, self.c(9))
        eeff = self.sel(enz, e, self.c(1))
        return sign, e, h9, l9, eeff, nan, inf

    def quiet(self, hi):
        return self.or_(hi, self.c(0x00080000))

    # ---- functions ----------------------------------------------------------------------------------------------
    def _function(self, name, ret, params, build):
        """Emit function `name` once: build(*param ids) emits the body and returns the result id."""
        if name not in self.fns:
            g = self.g
            outer, g.body = g.body, []
            # Gen.emit tracks VCC/EXEC stores for wave64 emulation; a helper body stores nothing.
            ids = [f"{name}_p{i}" for i in range(len(params))]
            result = build(*ids)
            lines, g.body = g.body, outer
            self.fns[name] = ([f"{name} = OpFunction {ret} Pure {FN_TYPE[name]}"] +
                              [f"{p} = OpFunctionParameter {t}" for p, t in zip(ids, params)] +
                              [f"{name}_entry = OpLabel"] + lines +
                              [f"OpReturnValue {result}", "OpFunctionEnd"])
        return name

    def call(self, name, ret, *args):
        r = self.g.tmp()
        self.g.emit(f"{r} = OpFunctionCall {ret} {name} " + " ".join(args))
        return r

    def words(self, v2):
        lo, hi = self.g.tmp(), self.g.tmp()
        self.g.emit(f"{lo} = OpCompositeExtract %uint {v2} 0")
        self.g.emit(f"{hi} = OpCompositeExtract %uint {v2} 1")
        return lo, hi

    def pair(self, lo, hi):
        r = self.g.tmp()
        self.g.emit(f"{r} = OpCompositeConstruct %v2uint_d {lo} {hi}")
        return r

    def add_f64(self, x, y, mode):
        """x + y, rounded per `mode` (a %uint id, MODE.FP_ROUND[3:2]); x, y and the result are (lo, hi)."""
        self._function("%f64s_add", "%v2uint_d", ["%uint"] * 5, self._add_body)
        return self.words(self.call("%f64s_add", "%v2uint_d", x[0], x[1], y[0], y[1], mode))

    def _add_body(self, alo, ahi, blo, bhi, mode):
        sa, ea, ah9, al9, aeff, nan_a, inf_a = self.unpack(alo, ahi)
        sb, eb, bh9, bl9, beff, nan_b, inf_b = self.unpack(blo, bhi)
        # order by magnitude: X the larger, Y the smaller (equal magnitudes: either)
        mah, mbh = self.and_(ahi, self.c(0x7FFFFFFF)), self.and_(bhi, self.c(0x7FFFFFFF))
        a_ge = self.bor(self.b("OpUGreaterThan", mah, mbh),
                        self.band(self.eq(mah, mbh), self.b("OpUGreaterThanEqual", alo, blo)))
        sx, xh, xl, xe = (self.sel(a_ge, p, q) for p, q in ((sa, sb), (ah9, bh9), (al9, bl9), (aeff, beff)))
        sy, yh, yl, ye = (self.sel(a_ge, q, p) for p, q in ((sa, sb), (ah9, bh9), (al9, bl9), (aeff, beff)))
        yh, yl = self.shr64_jam(yh, yl, self.sub(xe, ye))
        same = self.eq(sx, sy)
        ph, pl = self.add64(xh, xl, yh, yl)
        mh, ml = self.sub64(xh, xl, yh, yl)
        rh, rl = self.sel(same, ph, mh), self.sel(same, pl, ml)
        # normalize the leading bit to bit 62 (exact: see the module docstring of SoftFloat's add/sub mags)
        k = self.sub(self.clz64(rh, rl), self.c(1))
        k = self.and_(k, self.c(63))
        nh, nl = self.shl64(rh, rl, k)
        packed_hi, packed_lo = self.round_pack(sx, self.sub(xe, k), nh, nl, mode)
        # an exact zero: the operands' sign when they agree, else +0 (-0 when rounding toward -inf)
        rzero = self.b("OpLogicalNot", self.nz(self.or_(rh, rl)))
        zsign = self.sel(same, sx, self.sel(self.eq(mode, self.c(2)), self.c(1), self.c(0)))
        hi = self.sel(rzero, self.shl(zsign, self.c(31)), packed_hi)
        lo = self.sel(rzero, self.c(0), packed_lo)
        # infinities and NaNs
        inf_diff = self.band(inf_a, inf_b, self.b("OpLogicalNot", self.eq(sa, sb)))
        hi = self.sel(inf_b, bhi, hi)
        lo = self.sel(inf_b, blo, lo)
        hi = self.sel(inf_a, ahi, hi)
        lo = self.sel(inf_a, alo, lo)
        hi = self.sel(inf_diff, self.c(0x7FF80000), hi)
        lo = self.sel(inf_diff, self.c(0), lo)
        hi = self.sel(nan_b, self.quiet(bhi), hi)
        lo = self.sel(nan_b, blo, lo)
        hi = self.sel(nan_a, self.quiet(ahi), hi)
        lo = self.sel(nan_a, alo, lo)
        return self.pair(lo, hi)

    def scale_f64(self, x, s):
        """x * 2**s rounded to nearest even (VOP3 omod: s = 1, 2 or -1); x and the result are (lo, hi)."""
        self._function("%f64s_scale", "%v2uint_d", ["%uint"] * 3, self._scale_body)
        return self.words(self.call("%f64s_scale", "%v2uint_d", x[0], x[1], self.c(s)))

    def _scale_body(self, lo, hi, s):
        sign, e, h9, l9, eeff, nan, inf = self.unpack(lo, hi)
        k = self.and_(self.sub(self.clz64(h9, l9), self.c(1)), self.c(63))
        nh, nl = self.shl64(h9, l9, k)
        ph, pl = self.round_pack(sign, self.add(self.sub(eeff, k), s), nh, nl, self.c(0))
        zero = self.b("OpLogicalNot", self.nz(self.or_(self.and_(hi, self.c(0x7FFFFFFF)), lo)))
        keep = self.bor(zero, inf)
        rh = self.sel(keep, hi, ph)
        rl = self.sel(keep, lo, pl)
        rh = self.sel(nan, self.quiet(hi), rh)
        rl = self.sel(nan, lo, rl)
        return self.pair(rl, rh)

    def clamp01(self, x):
        """The VOP3 clamp bit on a binary64: +0 for NaN (what RDNA2 gives with MODE.DX10_CLAMP set, measured for
        f32 and f16 in SHADER-186, docs/shader_validation.md), -0 and anything below +0, 1.0 at or above 1.0 (+inf
        included), x otherwise. Inline: a handful of compares."""
        lo, hi = x
        pos = self.b("OpLogicalNot", self.nz(self.shr(hi, self.c(31))))
        mag_nz = self.nz(self.or_(hi, lo))
        nan = self.bor(self.b("OpUGreaterThan", hi, self.c(0x7FF00000)),
                       self.band(self.eq(hi, self.c(0x7FF00000)), self.nz(lo)))
        above0 = self.band(pos, mag_nz, self.b("OpLogicalNot", nan))
        ge1 = self.b("OpUGreaterThanEqual", hi, self.c(0x3FF00000))
        rh = self.sel(above0, self.sel(ge1, self.c(0x3FF00000), hi), self.c(0))
        rl = self.sel(above0, self.sel(ge1, self.c(0), lo), self.c(0))
        return rl, rh

    def clamp01_f32(self, x):
        """The VOP3 clamp bit on binary32 bits, the rule of clamp01: +0 for NaN, -0 and anything below +0, 1.0 at
        or above 1.0 (+inf included), x otherwise. Integer compares, so a NaN is not left to GLSL FClamp, whose
        result for NaN is undefined."""
        pos_nz = self.b("OpULessThan", self.sub(x, self.c(1)), self.c(0x7FFFFFFF))   # 0 < x < 0x80000000
        nan = self.b("OpUGreaterThan", x, self.c(0x7F800000))                         # positive NaNs
        above0 = self.band(pos_nz, self.b("OpLogicalNot", nan))
        ge1 = self.b("OpUGreaterThanEqual", x, self.c(0x3F800000))
        return self.sel(above0, self.sel(ge1, self.c(0x3F800000), x), self.c(0))

    def scale_f32(self, x, s):
        """binary32 bits * 2**s rounded to nearest even (VOP3 omod on an f32 result: s = 1, 2 or -1), f32
        denormals kept: x widened to binary64 (exact), its exponent moved by s (exact: every binary32 times 2**s
        is a normal binary64), then f64_to_f32, the one rounding. ±0 and ±inf keep their sign, a NaN comes back
        quiet with its sign and payload, an overflow is ±inf."""
        self._function("%f64s_scale_f32", "%uint", ["%uint"] * 2, self._scale_f32_body)
        return self.call("%f64s_scale_f32", "%uint", x, self.c(s))

    def _scale_f32_body(self, x, s):
        sbit = self.and_(x, self.c(0x80000000))
        e8 = self.and_(self.shr(x, self.c(23)), self.c(0xFF))
        f23 = self.and_(x, self.c(0x7FFFFF))
        # a denormal's leading bit moved to bit 23, as if its exponent field were 1 - sh
        sh = self.sub(self.c(23), self.u("OpExtInst", "%glsl", "FindUMsb", f23))
        sh = self.and_(sh, self.c(31))
        den = self.eq(e8, self.c(0))
        sig = self.sel(den, self.shl(f23, sh), f23)
        exp = self.sel(den, self.sub(self.c(897), sh), self.add(e8, self.c(896)))      # binary64 exponent field
        exp = self.add(exp, s)
        frac = self.and_(sig, self.c(0x7FFFFF))
        hi = self.or_(self.or_(sbit, self.shl(exp, self.c(20))), self.shr(frac, self.c(3)))
        lo = self.shl(sig, self.c(29))
        # ±0 (any shift of zero is zero), ±inf and NaN keep their exponent field: widened as they are
        keep = self.bor(self.eq(e8, self.c(0xFF)), self.band(den, self.eq(f23, self.c(0))))
        whi = self.or_(self.or_(sbit, self.sel(den, self.c(0), self.c(0x7FF00000))), self.shr(f23, self.c(3)))
        hi = self.sel(keep, whi, hi)
        lo = self.sel(keep, self.shl(f23, self.c(29)), lo)
        return self.to_f32((lo, hi))

    def to_f32(self, x):
        """binary64 (lo, hi) -> binary32 bits, rounded to nearest even (f64_to_f32); f32 denormals kept."""
        self._function("%f64s_to_f32", "%uint", ["%uint"] * 2, self._to_f32_body)
        return self.call("%f64s_to_f32", "%uint", x[0], x[1])

    def _to_f32_body(self, lo, hi):
        sign = self.shr(hi, self.c(31))
        sbit = self.shl(sign, self.c(31))
        e = self.and_(self.shr(hi, self.c(20)), self.c(0x7FF))
        fh = self.and_(hi, self.c(0xFFFFF))
        frac_nz = self.nz(self.or_(fh, lo))
        # 52-bit fraction >> 22 with jamming: 30 bits
        f30 = self.or_(self.shl(fh, self.c(10)), self.shr(lo, self.c(22)))
        f30 = self.or_(f30, self.sel(self.nz(self.and_(lo, self.c(0x3FFFFF))), self.c(1), self.c(0)))
        # softfloat_roundPackToF32(sign, e - 0x381, f30 | 0x40000000), nearest even
        exp = self.sub(e, self.c(0x381))
        sig = self.or_(f30, self.c(0x40000000))
        tiny = self.b("OpSLessThan", exp, self.c(0))
        n = self.sub(self.c(0), exp)
        n31 = self.and_(n, self.c(31))
        jam = self.nz(self.shl(sig, self.and_(self.sub(self.c(32), n31), self.c(31))))
        shifted = self.or_(self.shr(sig, n31), self.sel(self.band(self.nz(n31), jam), self.c(1), self.c(0)))
        shifted = self.sel(self.b("OpULessThan", n, self.c(31)), shifted,
                           self.sel(self.nz(sig), self.c(1), self.c(0)))
        sig = self.sel(tiny, shifted, sig)
        exp = self.sel(tiny, self.c(0), exp)
        round_bits = self.and_(sig, self.c(0x7F))
        over = self.bor(self.b("OpSGreaterThan", exp, self.c(0xFD)),
                        self.band(self.eq(exp, self.c(0xFD)),
                                  self.b("OpUGreaterThanEqual", self.add(sig, self.c(0x40)), self.c(0x80000000))))
        r = self.shr(self.add(sig, self.c(0x40)), self.c(7))
        r = self.sel(self.eq(round_bits, self.c(0x40)), self.and_(r, self.c(0xFFFFFFFE)), r)
        exp = self.sel(self.nz(r), exp, self.c(0))
        fin = self.add(self.add(sbit, self.shl(exp, self.c(23))), r)
        fin = self.sel(over, self.or_(sbit, self.c(0x7F800000)), fin)
        # zero (and f64 denormals that round to zero go through the path above), infinity, NaN
        zero = self.band(self.eq(e, self.c(0)), self.b("OpLogicalNot", frac_nz))
        fin = self.sel(zero, sbit, fin)
        e_max = self.eq(e, self.c(0x7FF))
        inf = self.band(e_max, self.b("OpLogicalNot", frac_nz))
        nan = self.band(e_max, frac_nz)
        fin = self.sel(inf, self.or_(sbit, self.c(0x7F800000)), fin)
        payload = self.or_(self.shl(fh, self.c(3)), self.shr(lo, self.c(29)))
        qnan = self.or_(self.or_(sbit, self.c(0x7FC00000)), payload)
        return self.sel(nan, qnan, fin)

    def helpers(self):
        """The helper functions' lines, in a stable order, and the function types they need."""
        out, types = [], []
        for name in ("%f64s_add", "%f64s_scale", "%f64s_to_f32", "%f64s_scale_f32"):
            if name in self.fns:
                out += self.fns[name]
                t = f"{FN_TYPE[name]} = OpTypeFunction {F64_FN_TYPES[FN_TYPE[name]]}"
                if t not in types:
                    types.append(t)
        return types, out
