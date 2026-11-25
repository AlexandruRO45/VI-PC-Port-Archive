//! Portable 64-lane wave lowering across multi-wave compute threadgroups.
//!
//! Lowers 64-lane ballots, votes (`WaveActiveAnyTrue` / `WaveActiveAllTrue`),
//! bit counts (`WaveActiveCountBits`), prefix bit counts (`WavePrefixCountBits`),
//! cross-half lane reads (`WaveReadLaneAt`), `v_readfirstlane_b32`, `s_ff1_i32_b64`,
//! `v_permlanex16_b32`, and 64-lane reductions onto `groupshared` scratch partitioned
//! per logical 64-lane wave (`wave_id = gindex >> 6`, `lane64 = gindex & 63u`,
//! `half_id = (lane64 >> 5) & 1u`, `lane32 = lane64 & 31u`).
//!
//! Only enabled when every barrier-requiring operation is proven to lie in
//! wave-uniform control flow and every enclosing loop is proven workgroup-uniform
//! (so all `T = nx * ny * nz` threads hit every `GroupMemoryBarrierWithGroupSync()`
//! in lockstep with zero possibility of GPU deadlock).

mod emit;
mod masks;
mod uniformity;

use super::Opnd;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

pub(super) use emit::{
    emit, emit_post_votes, hoist_after, hoist_before, lockstep_branch_condition,
};
pub(super) use uniformity::analyze;

/// A 64-lane collective vote/count/scan emitted for per-lane 1-bit mask operand(s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum MaskOp {
    /// `scc = uint(tb_w64_any(lane, (m & 1u) != 0u))`
    AnyTrue(Opnd),
    /// `scc = uint(!tb_w64_any(lane, (m & 1u) != 0u))`
    NoneTrue(Opnd),
    /// `scc = uint(tb_w64_all(lane, (m & 1u) != 0u))`
    AllTrue(Opnd),
    /// `scc = uint(!tb_w64_all(lane, (m & 1u) != 0u))`
    NotAllTrue(Opnd),
    /// `scc = uint(tb_w64_all(lane, (a & 1u) == (b & 1u)))`
    AllEqual(Opnd, Opnd),
    /// `scc = uint(tb_w64_any(lane, (a & 1u) != (b & 1u)))`
    AnyDiff(Opnd, Opnd),
    /// `scc = uint(tb_w64_all(lane, (ballot_bit(a) == (b & 1u))))`
    AllEqualBallot(Opnd, Opnd),
    /// `scc = uint(tb_w64_any(lane, (ballot_bit(a) != (b & 1u))))`
    AnyDiffBallot(Opnd, Opnd),
    /// `dst = tb_w64_count_bits(lane, (m & 1u) != 0u); scc = uint(dst != 0u);`
    CountBits(Opnd),
    /// `dst = tb_w64_first_active(lane, (m & 1u) != 0u);`
    FirstActive(Opnd),
    /// Post-instruction vote: recompute `scc = uint(tb_w64_any(lane, (m & 1u) != 0u))`.
    PostAnyTrue(Opnd),
}

/// Verified lowering plan for a compute shader executing logical Wave64 waves portably.
#[derive(Clone, Debug, Default)]
pub(super) struct Plan {
    pub(super) waves: u32,
    pub(super) needs_tb_lane: bool,
    pub(super) uses_ballot: bool,
    pub(super) uses_exchange: bool,
    pub(super) uses_extrema_f32: bool,
    /// `v_cmp_*` sites emitting full 64-lane ballots into an even SGPR pair.
    pub(super) ballot_compares: BTreeSet<usize>,
    /// `s_mov_b64 s[lo:hi], exec|vcc` / `s_*_saveexec_b64` sites capturing a 64-lane ballot.
    pub(super) ballot_mask_copies: BTreeSet<usize>,
    /// `v_mov_b32` / `s_mov_b32` directly reading `exec_lo|exec_hi|vcc_lo|vcc_hi` as ballot word.
    pub(super) ballot_word_reads: BTreeSet<usize>,
    /// 64-bit logic ops combining a `Ballot64` SGPR pair with `exec` or `vcc`.
    pub(super) ballot_logic_mask: BTreeSet<usize>,
    /// Instructions consuming a 64-bit ballot SGPR pair as a per-lane mask.
    pub(super) ballot_mask_reads: BTreeSet<usize>,
    /// Instructions performing a 64-lane collective over per-lane mask bit(s).
    pub(super) mask_votes: BTreeMap<usize, MaskOp>,
    /// Barrier-bearing sites enclosed by wave-uniform (not group-uniform) forward skips
    /// that must be hoisted outside `if (!(j...))` so all workgroup threads synchronize.
    pub(super) hoisted_ops: BTreeSet<usize>,
    /// Reduction `v_permlanex16_b32` sites verified to have zero selectors and full EXEC.
    pub(super) reduction_permlane: BTreeSet<usize>,
}

/// Outcome of analyzing a compute shader for portable Wave64 lowering.
pub(super) enum Analysis {
    /// Shader has no Wave64 ops requiring portable lowering.
    NotNeeded,
    /// Verified portable Wave64 plan (drop `[WaveSize(64)]`).
    Portable(Plan),
    /// Shader already has native `[WaveSize(64)]`, but cannot lower portably (keep native `[WaveSize(64)]`).
    KeepNative,
    /// Shader contains general Wave64 ballot/vote ops under divergent or unprovable control flow.
    RefuseDivergent(String),
}

impl Plan {
    pub(super) fn straight_exchange() -> Self {
        Self {
            waves: 1,
            needs_tb_lane: false,
            uses_ballot: false,
            uses_exchange: true,
            ..Self::default()
        }
    }

    pub(super) fn lane_expr(&self) -> &'static str {
        if self.needs_tb_lane {
            "tb_lane"
        } else {
            "gtid.x"
        }
    }

    pub(super) fn scratch_bytes(&self) -> u32 {
        let mask_words = if self.uses_ballot { self.waves * 2 } else { 0 };
        let lane_words = if self.uses_exchange || self.uses_extrema_f32 {
            self.waves * 64
        } else {
            0
        };
        (mask_words + lane_words) * 4
    }

    /// Emit `groupshared` scratch arrays and portable 64-lane HLSL helper functions.
    pub(super) fn declarations(&self) -> String {
        let w = self.waves;
        let mut out = String::new();
        if self.uses_ballot {
            writeln!(out, "groupshared uint tb_w64_mask[{w}][2];").unwrap();
        }
        if self.uses_exchange || self.uses_extrema_f32 {
            writeln!(out, "groupshared uint tb_w64_lanes[{w}][64];").unwrap();
        }
        if self.uses_ballot {
            out.push_str(
                r#"uint2 tb_w64_ballot(uint gindex, bool pred) {
    uint wave_id = gindex >> 6;
    uint lane64 = gindex & 63u;
    uint half_id = (lane64 >> 5) & 1u;
    uint lane32 = lane64 & 31u;
    if (lane32 == 0u) tb_w64_mask[wave_id][half_id] = 0u;
    GroupMemoryBarrierWithGroupSync();
#if __SHADER_TARGET_MAJOR >= 6
    if (WaveGetLaneCount() == 64u && WaveActiveAllEqual(wave_id) && WaveActiveAllTrue(WaveGetLaneIndex() == lane64)) {
        uint4 b = WaveActiveBallot(pred);
        if (lane64 == 0u) {
            tb_w64_mask[wave_id][0] = b.x;
            tb_w64_mask[wave_id][1] = b.y;
        }
    } else if (WaveGetLaneCount() == 32u && WaveActiveAllEqual(gindex >> 5) && WaveActiveAllTrue(WaveGetLaneIndex() == lane32)) {
        uint b = WaveActiveBallot(pred).x;
        if (lane32 == 0u) tb_w64_mask[wave_id][half_id] = b;
    } else
#endif
    {
        if (pred) InterlockedOr(tb_w64_mask[wave_id][half_id], 1u << lane32);
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 mask = uint2(tb_w64_mask[wave_id][0], tb_w64_mask[wave_id][1]);
    GroupMemoryBarrierWithGroupSync();
    return mask;
}
bool tb_w64_any(uint gindex, bool pred) {
    uint2 mask = tb_w64_ballot(gindex, pred);
    return (mask.x | mask.y) != 0u;
}
bool tb_w64_all(uint gindex, bool pred) {
    uint2 mask = tb_w64_ballot(gindex, pred);
    return (mask.x & mask.y) == 0xffffffffu;
}
uint tb_w64_count_bits(uint gindex, bool pred) {
    uint2 mask = tb_w64_ballot(gindex, pred);
    return countbits(mask.x) + countbits(mask.y);
}
uint tb_w64_prefix_count_bits(uint gindex, bool pred) {
    uint2 mask = tb_w64_ballot(gindex, pred);
    uint lane64 = gindex & 63u;
    uint lo_mask = lane64 >= 32u ? 0xffffffffu : ((1u << (lane64 & 31u)) - 1u);
    uint hi_mask = lane64 <= 32u ? 0u : ((1u << (lane64 - 32u)) - 1u);
    return countbits(mask.x & lo_mask) + countbits(mask.y & hi_mask);
}
uint tb_w64_first_active(uint gindex, bool pred) {
    uint2 mask = tb_w64_ballot(gindex, pred);
    return mask.x != 0u ? uint(firstbitlow(mask.x)) : (mask.y != 0u ? 32u + uint(firstbitlow(mask.y)) : 0xffffffffu);
}
"#,
            );
        }
        if self.uses_exchange {
            out.push_str(
                r#"uint tb_reduction_read(uint gindex, uint value, uint source) {
    uint wave_id = gindex >> 6;
    uint lane64 = gindex & 63u;
    tb_w64_lanes[wave_id][lane64] = value;
    GroupMemoryBarrierWithGroupSync();
    uint result = tb_w64_lanes[wave_id][source & 63u];
    GroupMemoryBarrierWithGroupSync();
    return result;
}
"#,
            );
        }
        if self.uses_ballot && self.uses_exchange {
            out.push_str(
                r#"uint tb_w64_read_first_lane(uint gindex, uint value, bool active) {
    uint first = tb_w64_first_active(gindex, active);
    return tb_reduction_read(gindex, value, first == 0xffffffffu ? 0u : first);
}
"#,
            );
        }
        if self.uses_extrema_f32 {
            out.push_str(
                r#"uint tb_w64_extrema_f32(uint gindex, float value, bool maximum) {
    uint wave_id = gindex >> 6;
    uint lane64 = gindex & 63u;
    tb_w64_lanes[wave_id][lane64] = asuint(value);
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 32u; stride > 0u; stride >>= 1) {
        if (lane64 < stride) {
            float a = asfloat(tb_w64_lanes[wave_id][lane64]);
            float b = asfloat(tb_w64_lanes[wave_id][lane64 + stride]);
            tb_w64_lanes[wave_id][lane64] = asuint(maximum ? max(a, b) : min(a, b));
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint result = tb_w64_lanes[wave_id][0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}
"#,
            );
        }
        out
    }
}

#[cfg(test)]
#[path = "../../tests/hlsl/portable_wave64.rs"]
mod tests;
