// Map one graphics stage's user data from the game's PSO cache into the layer's root-signature model.
//
// A graphics PSO in the PS\0\x09 cache (#4358; `docs/pso_cache.md` in that PR) names one or more chunk
// records per stage. A
// record is a list of 0x41 entries (descriptor ranges, 32-bit constants, sampler references) followed by 0x42 slot
// words (0x00SS_IIRR). The cache decoder (tools/psocache_usersig.py) already splits those fields; this file does the
// reduction the layer needs, and refuses what it cannot place instead of guessing:
//
//   one distinct 0x42 SS       -> one D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE
//   each (SS, II) slot pair     -> one D3D12_DESCRIPTOR_RANGE inside that table (II is the D3D12 range type order)
//   each 0x401 entry            -> one D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS of `a` dwords
//   0x2 sampler, 0x100          -> no user-data dword (a static sampler or an unconsumed entry); a note
//
// What the record does not carry, and this file therefore does not invent:
// - the root-parameter order. Tables are emitted where their first entry appears in the record (Order::Record,
//   the default), or grouped by ascending SS (Order::Table). The record does not say which is the console's order;
// - the 0x401 base shader register. The constants parameter is built with register 0 and a note;
// - the stage visibility. The caller passes what it knows; All by default;
// - the 0x42 slots that have no paired 0x41 descriptor ("extras"). A signature built without them would silently
//   omit user data, so a record with extras is refused; the notes still describe them.
//
// One record is one stage's user-data block, so it maps to its own RootSignature. A graphics PSO carries 3 to 5
// distinct records; concatenating them would double-count parameters. The layer currently packs one user-data block
// for all graphics stages (rt::rootsig::pack, docs/d3d12_vulkan.md 3.5), so a synthetic pipeline uses the record of
// the stage whose SGPR window it exercises.
//
// Portable: no d3d12.h, no Vulkan. Tested by runtime/test/pso_user_signature_test.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "root_signature.h"

namespace rt::pso {

// 0x41 entry kinds (#4358, "0x41: user-data entries").
inline constexpr std::uint32_t kKindSrv = 0x300;
inline constexpr std::uint32_t kKindCbv = 0x302;
inline constexpr std::uint32_t kKindUav = 0x303;
inline constexpr std::uint32_t kKindConstants = 0x401;
inline constexpr std::uint32_t kKindSampler = 0x2;
inline constexpr std::uint32_t kKindUnknown100 = 0x100;

// D3D12_DESCRIPTOR_RANGE_TYPE order, as the 0x42 II byte reads it.
inline constexpr std::uint32_t kRangeSrv = 0;
inline constexpr std::uint32_t kRangeUav = 1;
inline constexpr std::uint32_t kRangeCbv = 2;
inline constexpr std::uint32_t kRangeSampler = 3;

// One decoded 0x41 entry. `a`, `b` and `size` are the three bytes of w5; `count` is w4. For a descriptor entry the
// reading is base register `a`, register space `b`, descriptor count `count`. For 0x401 `a` is the constant count
// and `count` repeats it with low confidence, so `a` is what map_user_record uses. The decoder's `offset` and `key`
// are deliberately absent: neither enters a root signature, and the offset's unit (8-byte granular, divisor
// unproven) is not guessed, so ranges append inside their table.
struct UserEntry {
    std::uint32_t kind = 0;
    std::uint32_t count = 0;
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::uint32_t size = 0;  // element size (2, 4 or 8); carried for diagnostics, not read by the mapping
};

// One decoded 0x42 slot word. `table` is SS, `range` is II, `width` is RR.
struct UserSlot {
    std::uint32_t table = 0;
    std::uint32_t range = 0;
    std::uint32_t width = 0;
};

struct UserRecord {
    std::vector<UserEntry> entries;  // 0x41 entries in record order
    std::vector<UserSlot> slots;     // 0x42 words in record order, paired positionally with the descriptor entries
};

enum class Order {
    Record,  // a table sits where its first descriptor entry appears; constants stay where they are listed
    Table,   // tables ascending by SS, then the constants in record order
};

struct MapOptions {
    Order order = Order::Record;
    rt::rootsig::Visibility visibility = rt::rootsig::Visibility::All;
};

struct MapResult {
    bool ok = false;
    std::string error;  // set when !ok
    rt::rootsig::RootSignature signature;
    rt::rootsig::PackResult packed;  // the layer's 2-dwords-per-table layout (rt::rootsig::pack)
    // Constants dwords + one dword per table: the console/compute model measured in docs/d3d12_vulkan.md 3.5 and
    // #4312. Diagnostic only, also set when the layer layout refuses the signature; not a confirmation.
    std::uint32_t console_dwords = 0;
    std::vector<std::string> notes;  // what is unmapped or assumed, in stable phrases a test can match
};

// Maps one decoded graphics-stage record to a RootSignature and packs it with rt::rootsig::pack. Fails with a
// reason when a descriptor has no slot, a slot has no descriptor (the decoded set would be incomplete), a slot's
// range type contradicts its entry, an entry kind is unknown, a count is zero, or the packed layout exceeds the 32
// user SGPRs (spill is not modelled).
[[nodiscard]] MapResult map_user_record(const UserRecord& record, const MapOptions& options = {});

}  // namespace rt::pso
