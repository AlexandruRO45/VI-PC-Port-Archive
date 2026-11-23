// CPU tests for runtime/src/pso_user_signature.{h,cpp}: a decoded graphics-stage user-data record from the game's
// PS\0\x09 PSO cache reduced to a root signature and packed into the 32 user SGPRs. Everything here is hand-built
// (the shapes #4358 measures), with no cache file, game data, GPU or Windows SDK.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "pso_user_signature.h"
#include "root_signature.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool has_note(const rt::pso::MapResult& result, const char* fragment) {
    for (const std::string& note : result.notes)
        if (note.find(fragment) != std::string::npos) return true;
    return false;
}

bool has_error(const rt::pso::MapResult& result, const char* fragment) {
    return result.error.find(fragment) != std::string::npos;
}

rt::pso::UserEntry entry(std::uint32_t kind, std::uint32_t count, std::uint32_t a, std::uint32_t b = 0,
                         std::uint32_t size = 4) {
    rt::pso::UserEntry e;
    e.kind = kind;
    e.count = count;
    e.a = a;
    e.b = b;
    e.size = size;
    return e;
}

rt::pso::UserSlot slot(std::uint32_t table, std::uint32_t range, std::uint32_t width = 1) {
    rt::pso::UserSlot s;
    s.table = table;
    s.range = range;
    s.width = width;
    return s;
}

// The measured shape: one SRV and one CBV range in table 1, four 32-bit constants, one UAV range in table 2.
rt::pso::UserRecord measured_shape() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindSrv, 8, 0, 1), entry(rt::pso::kKindCbv, 4, 2, 1),
                 entry(rt::pso::kKindConstants, 4, 4), entry(rt::pso::kKindUav, 2, 0, 2)};
    r.slots = {slot(1, 0), slot(1, 2), slot(2, 1)};
    return r;
}

void test_measured_shape_record_order() {
    const rt::pso::MapResult r = rt::pso::map_user_record(measured_shape());
    check(r.ok, "measured shape maps");
    check(r.error.empty(), "measured shape has no error");
    check(r.signature.params.size() == 3, "measured shape has three root parameters");
    if (r.signature.params.size() != 3) return;
    const auto& table1 = r.signature.params[0];
    const auto& constants = r.signature.params[1];
    const auto& table2 = r.signature.params[2];
    check(table1.type == rt::rootsig::ParamType::Table, "parameter 0 is a table (record order)");
    check(table1.ranges.size() == 2, "table 1 holds two ranges");
    if (table1.ranges.size() == 2) {
        const auto& srv = table1.ranges[0];
        check(srv.range_type == rt::pso::kRangeSrv, "table 1 range 0 is SRV");
        check(srv.num_descriptors == 8, "SRV range keeps the 0x41 count");
        check(srv.base_shader_register == 0, "SRV range keeps base register a");
        check(srv.register_space == 1, "SRV range keeps register space b");
        check(srv.offset_in_table == rt::rootsig::kRangeOffsetAppend, "ranges append in table order");
        const auto& cbv = table1.ranges[1];
        check(cbv.range_type == rt::pso::kRangeCbv, "table 1 range 1 is CBV");
        check(cbv.num_descriptors == 4 && cbv.base_shader_register == 2 && cbv.register_space == 1,
              "CBV range keeps count, base and space");
    }
    check(constants.type == rt::rootsig::ParamType::Constants, "parameter 1 is 32-bit constants");
    check(constants.num_values == 4, "constants take a dwords (not w4)");
    check(constants.visibility == rt::rootsig::Visibility::All, "visibility defaults to All");
    check(table2.type == rt::rootsig::ParamType::Table && table2.ranges.size() == 1,
          "parameter 2 is table 2 with one range");
    if (table2.ranges.size() == 1) {
        check(table2.ranges[0].range_type == rt::pso::kRangeUav, "table 2 range is UAV");
        check(table2.ranges[0].register_space == 2, "table 2 range keeps its space");
    }
    check(r.packed.ok, "layer layout packs");
    check(r.packed.dword_count == 8, "packed: 2 (table) + 4 (constants) + 2 (table)");
    check(r.packed.d3d12_cost == 6, "D3D12 cost: 1 + 4 + 1");
    check(r.console_dwords == 6, "console model: 1 per table + 4 constants");
    check(has_note(r, "0x401 base shader register"), "constants register unknown is noted");

    const std::vector<std::uint8_t> blob = rt::rootsig::encode_canonical(r.signature);
    const rt::rootsig::DecodeResult decoded = rt::rootsig::decode_canonical(blob.data(), blob.size());
    check(decoded.ok, "mapped signature round-trips through the canonical codec");
    check(decoded.ok && decoded.sig.params.size() == 3, "round-trip keeps the parameter count");
    check(decoded.ok && decoded.sig.params[0].ranges.size() == 2, "round-trip keeps the ranges");
}

void test_table_index_order() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindCbv, 4, 0, 2), entry(rt::pso::kKindSrv, 8, 0, 1),
                 entry(rt::pso::kKindConstants, 2, 2)};
    r.slots = {slot(2, 2), slot(1, 0)};
    rt::pso::MapOptions options;
    options.order = rt::pso::Order::Table;
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r, options);
    check(mapped.ok, "table-index order maps");
    check(mapped.signature.params.size() == 3, "table-index order has three params");
    if (mapped.signature.params.size() != 3) return;
    check(mapped.signature.params[0].type == rt::rootsig::ParamType::Table &&
              mapped.signature.params[0].ranges.size() == 1 &&
              mapped.signature.params[0].ranges[0].range_type == rt::pso::kRangeSrv,
          "SS=1 (SRV) comes first");
    check(mapped.signature.params[1].type == rt::rootsig::ParamType::Table &&
              mapped.signature.params[1].ranges.size() == 1 &&
              mapped.signature.params[1].ranges[0].range_type == rt::pso::kRangeCbv,
          "SS=2 (CBV) comes second");
    check(mapped.signature.params[2].type == rt::rootsig::ParamType::Constants,
          "constants follow the tables");
    check(mapped.packed.dword_count == 6, "table-index order packs to 2 + 2 + 2 dwords");

    const rt::pso::MapResult record_order = rt::pso::map_user_record(r);
    check(record_order.ok && record_order.signature.params[0].ranges[0].range_type == rt::pso::kRangeCbv,
          "record order keeps the first descriptor entry first");
}

void test_interleaved_tables_are_noted() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindSrv, 4, 0, 1), entry(rt::pso::kKindCbv, 2, 0, 2),
                 entry(rt::pso::kKindUav, 2, 0, 1)};
    r.slots = {slot(1, 0), slot(2, 2), slot(1, 1)};
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(mapped.ok, "interleaved tables still map");
    check(mapped.signature.params.size() == 2, "interleaved tables fold into two tables");
    check(mapped.signature.params[0].ranges.size() == 2, "SS=1 receives the later UAV range too");
    check(has_note(mapped, "interleaved"), "interleaved order is noted, not guessed");
}

void test_descriptor_without_slot_is_refused() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindSrv, 4, 0, 1)};
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(!mapped.ok, "descriptor without a slot is refused");
    check(has_error(mapped, "no slot"), "the refusal names the missing slot");
}

void test_kind_and_range_type_must_agree() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindSrv, 4, 0, 1)};
    r.slots = {slot(1, 2)};  // a CBV slot for an SRV entry
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(!mapped.ok, "SRV entry with a CBV slot is refused");
    check(has_error(mapped, "pairs with"), "the refusal names the contradiction");
}

void test_extra_slots_are_refused() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindSrv, 4, 0, 1)};
    r.slots = {slot(1, 0), slot(14, 3, 2)};
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(!mapped.ok, "an extra slot refuses a partial user-data set");
    check(mapped.signature.params.empty(), "the refusal maps no parameters");
    check(has_error(mapped, "extra 0x42 slot"), "the refusal names the extra slot");
    check(has_note(mapped, "extra slot"), "the extra slot is still described in the notes");
}

void test_sampler_and_unknown_are_not_consumed() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindSampler, 1, 0, 7, 4), entry(rt::pso::kKindUnknown100, 1, 0, 6, 2)};
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(mapped.ok, "sampler and 0x100 records map without parameters");
    check(mapped.signature.params.empty(), "neither 0x2 nor 0x100 adds a root parameter");
    check(mapped.packed.dword_count == 0, "neither takes a user-data dword");
    check(has_note(mapped, "sampler"), "the sampler is noted");
    check(has_note(mapped, "0x100"), "the 0x100 entry is noted");
}

void test_empty_record_maps_to_an_empty_signature() {
    const rt::pso::MapResult mapped = rt::pso::map_user_record({});
    check(mapped.ok, "an empty record maps");
    check(mapped.signature.params.empty(), "an empty record has no parameters");
    check(mapped.packed.dword_count == 0, "an empty record packs zero dwords");
}

void test_unknown_kind_is_refused() {
    rt::pso::UserRecord r;
    r.entries = {entry(0x123, 1, 0)};
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(!mapped.ok, "an unknown entry kind is refused");
    check(has_error(mapped, "unknown"), "the refusal names the unknown kind");
}

void test_zero_counts_are_refused() {
    rt::pso::UserRecord zero_range;
    zero_range.entries = {entry(rt::pso::kKindSrv, 0, 0, 1)};
    zero_range.slots = {slot(1, 0)};
    check(!rt::pso::map_user_record(zero_range).ok, "a range with zero descriptors is refused");

    rt::pso::UserRecord zero_constants;
    zero_constants.entries = {entry(rt::pso::kKindConstants, 0, 0)};
    check(!rt::pso::map_user_record(zero_constants).ok, "a 0x401 entry with zero dwords is refused");
}

void test_spill_is_refused_with_the_console_count_kept() {
    rt::pso::UserRecord r;
    r.entries = {entry(rt::pso::kKindConstants, 40, 40)};
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(!mapped.ok, "40 constant dwords are refused");
    check(has_error(mapped, "spill"), "the refusal is the spill rule, not a crash");
    check(mapped.console_dwords == 40, "the console-model count is still reported");
}

void test_console_model_can_fit_where_the_layer_layout_does_not() {
    rt::pso::UserRecord r;
    r.entries.push_back(entry(rt::pso::kKindConstants, 20, 20));
    for (std::uint32_t table = 1; table <= 7; ++table) {
        r.entries.push_back(entry(rt::pso::kKindSrv, 1, 0, table));
        r.slots.push_back(slot(table, 0));
    }
    const rt::pso::MapResult mapped = rt::pso::map_user_record(r);
    check(!mapped.ok, "20 + 7*2 dwords exceed the layer's 32-dword block");
    check(mapped.console_dwords == 27, "one dword per table puts the same record at 27 dwords");
    check(has_error(mapped, "spill"), "the layer's refusal is the spill rule");
}

}  // namespace

int main() {
    test_measured_shape_record_order();
    test_table_index_order();
    test_interleaved_tables_are_noted();
    test_descriptor_without_slot_is_refused();
    test_kind_and_range_type_must_agree();
    test_extra_slots_are_refused();
    test_sampler_and_unknown_are_not_consumed();
    test_empty_record_maps_to_an_empty_signature();
    test_unknown_kind_is_refused();
    test_zero_counts_are_refused();
    test_spill_is_refused_with_the_console_count_kept();
    test_console_model_can_fit_where_the_layer_layout_does_not();
    if (g_failures) std::printf("%d FAILURE(S)\n", g_failures);
    else std::printf("all checks passed\n");
    return g_failures ? 1 : 0;
}
