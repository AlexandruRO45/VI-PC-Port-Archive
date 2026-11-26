//! Changes to the translation of a few of the title's programs that keep AMD's shader
//! compiler from failing on them: amdxc64 8.18.10.0514 (Radeon RX 9060 XT) dereferences
//! null, or ends the process, creating their pipelines as otherwise translated. Found by
//! creating every program's pipeline (2,684 compute; 23,279 others with a stand-in partner
//! stage) in a process of its own. Chosen per program by container digest: every other
//! program translates as before, and these compute the same values, so a host without the
//! fault loses nothing.
//!
//! Kept to this file and a one-line hook where each change acts (`begin` before lowering,
//! `passes` in the clean-up, `hoisted_size` at a raw view's size), so it carries to forks.

use std::cell::Cell;

use super::super::bitcode::opt::{FACTS, FOLD};
use super::super::bitcode::{Attr, V};
use super::{Lower, Source, Why, meeting_mode};

#[derive(Clone, Copy)]
struct Fixes {
    hoist_sizes: bool,
    passes: u32,
}

thread_local! {
    /// The fixes of the program being lowered on this thread.
    static CURRENT: Cell<Fixes> = const { Cell::new(Fixes { hoist_sizes: false, passes: 0 }) };
}

/// (container digest, hoist raw views' sizes, clean-up passes run whatever
/// `VICTORY_SHADER_OPT` says).
const FIXES: [(u128, bool, u32); 13] = [
    // Raw loads whose offset comes from the same view's size, a branch on the loaded value
    // guarding a loop (reduced to 46 instructions with DXC's dxa and dxopt).
    // bvh_skinning:CSComputeSkinningTreeAnimVelocity_d2f99501_Wrapped
    (0x6177_8417_b3a9_0c25_203c_4842_27fe_ebfc, true, 0),
    // GpuParticleUpdateCS_UpdateSkinnedAttachMeshTreeParticles_d5e6e48d_Wrapped
    (0x07d1_ccf3_3c9b_4b4c_7619_eaff_ba86_96ed, true, 0),
    // rtdirectionalshadows:CS_RTShadowTrace32HQ_Tiled_bfd3b062_Wrapped
    (0x7e9f_4d75_149c_5673_6246_56da_a620_e24b, true, 0),
    // rtdirectionalshadows:CS_RTShadowTrace32TemporalHQ_OneOver2RPP_Tiled
    (0xa62d_11a6_bef5_7571_a1ab_8e73_e10c_470b, true, 0),
    // indirect_diffuse_lighting:CS_ScreenProbes_ActivateRays_d60186e6_Wrapped: with no
    // clean-up the process ends (0xFFFFFFFF); folding constants alone is enough.
    (0x5212_83e5_0211_e301_e7fa_ed9e_0299_dc22, false, FOLD),
    // The glints pixel programs build once branches tell their conditions.
    // PS_MIRROR_PBR_MATERIAL_CLEARCOAT_GLINTS
    (0xe416_8a83_c5c6_ff23_73e3_4325_35ad_7c9d, false, FACTS),
    // PS_MIRROR_PBR_MATERIAL_GLINTS
    (0x3024_35a1_3c4c_b152_2a2c_e785_8fab_7e34, false, FACTS),
    // PS_MIRROR_PBR_MATERIAL_TERRAIN_GLINTS
    (0x60f0_c383_b6ba_41b5_895d_4a40_39e7_a781, false, FACTS),
    // PS_MIRROR_PBR_MATERIAL_TERRAIN_GLINTS_TRANSLUCENCY
    (0xb867_ed98_e402_b5f8_8844_35ad_5d48_5b36, false, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_CLEARCOAT_GLINTS_LIGHTING
    (0x8e32_45bb_242a_2e59_17c1_d2ce_d51c_7fc9, false, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_GLINTS_LIGHTING
    (0x5bbf_8eed_9a4f_5133_8b8d_2e36_cd9f_4b66, false, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_TERRAIN_GLINTS_LIGHTING
    (0x3c03_fca2_b4cc_b79b_fb87_eb51_81de_8a70, false, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_TERRAIN_GLINTS_TRANSLUCENCY_LIGHTING
    (0x3df1_012d_c24c_6daf_af0b_c2ce_8fc7_8b7b, false, FACTS),
];

/// Before lowering the program whose container digest is `digest` on this thread.
pub(in crate::emit) fn begin(digest: [u8; 16]) {
    let d = u128::from_be_bytes(digest);
    let (hoist_sizes, passes) = FIXES.iter().find(|f| f.0 == d).map_or((false, 0), |f| (f.1, f.2));
    CURRENT.set(Fixes { hoist_sizes, passes });
}

/// Clean-up passes the program being lowered gets whatever `VICTORY_SHADER_OPT` says.
pub(in crate::emit) fn passes() -> u32 {
    CURRENT.get().passes
}

impl Lower<'_> {
    /// For a program that hoists sizes: the byte size of the raw view `src` names, read in
    /// the entry block and passed through groupshared memory across a group barrier, then
    /// loaded here. The barrier hides from the compiler that a load's offset came from that
    /// view's own size; in the entry block every thread meets it. `None` (the size as read
    /// at the use) outside compute and mesh, or where the view's heap index is not known in
    /// the entry block.
    pub(super) fn hoisted_size(&mut self, src: &Source, props: (u32, u32)) -> Result<Option<V>, Why> {
        let Source::Loaded { load, first } = *src else {
            return Ok(None);
        };
        if !CURRENT.get().hoist_sizes || self.s.threads.is_none() || self.x.res.waterfall.loads.contains(&load) {
            return Ok(None);
        }
        let entry = self.s.entry;
        // As `handle` computes it here, then again in the entry block where it is
        // arithmetic over values known there.
        let index = self.heap_index(load, first, false)?;
        let Some(index) = self.s.m.rematerialize(index, entry) else {
            return Ok(None);
        };
        let i32t = self.s.t.i32;
        let h = self.s.heap_handle(entry, index, false, false, props);
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = self.s.m.undef(i32t);
        let d = self.s.op(entry, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let n = self.s.m.extract(entry, d, 0);
        let slot = self.s.m.fresh_global("view_size", i32t, 3);
        self.s.m.store(entry, slot, n);
        let mode = self.s.m.i32(meeting_mode());
        let void = self.s.t.void;
        self.s.op(entry, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        Ok(Some(self.s.m.load(self.cur, slot)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::bitcode::{Bin, Module};

    #[test]
    fn rematerialize_copies_arithmetic_over_entry_values_only() {
        let mut m = Module::default();
        let i32t = m.int(32);
        let (entry, later) = (m.new_block(), m.new_block());
        let g = m.global("g", i32t, 3);
        let base = m.load(entry, g);
        let (four, five) = (m.i32(4), m.i32(5));
        // In `later`: (base + 4) >> 5, and arithmetic over a value read there.
        let sum = m.bin(later, Bin::Add, base, four);
        let index = m.bin(later, Bin::LShr, sum, five);
        let loaded = m.load(later, g);
        let other = m.bin(later, Bin::Add, loaded, four);
        assert_eq!(m.rematerialize(base, entry), Some(base));
        assert_eq!(m.rematerialize(four, entry), Some(four));
        let copy = m.rematerialize(index, entry).expect("arithmetic over an entry value");
        assert_ne!(copy, index);
        // The copy is in the entry block already.
        assert_eq!(m.rematerialize(copy, entry), Some(copy));
        assert_eq!(m.rematerialize(other, entry), None);
    }

    #[test]
    fn only_the_listed_programs_change() {
        super::begin([0; 16]);
        assert_eq!(super::passes(), 0);
        assert!(!super::CURRENT.get().hoist_sizes);
        super::begin(0x5212_83e5_0211_e301_e7fa_ed9e_0299_dc22u128.to_be_bytes());
        assert_eq!(super::passes(), super::FOLD);
        super::begin(0x6177_8417_b3a9_0c25_203c_4842_27fe_ebfcu128.to_be_bytes());
        assert!(super::CURRENT.get().hoist_sizes);
    }
}
