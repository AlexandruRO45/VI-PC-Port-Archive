// PSO cache graphics user data -> root signature, see pso_user_signature.h.
#include "pso_user_signature.h"

#include <algorithm>
#include <cstdio>
#include <map>

namespace rt::pso {

namespace {

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%x", value);
    return text;
}

bool is_descriptor(std::uint32_t kind) {
    return kind == kKindSrv || kind == kKindCbv || kind == kKindUav;
}

// The range type D3D12_DESCRIPTOR_RANGE_TYPE a descriptor kind must pair with: SRV 0, UAV 1, CBV 2.
std::uint32_t expected_range(std::uint32_t kind) {
    return kind == kKindSrv ? kRangeSrv : kind == kKindUav ? kRangeUav : kRangeCbv;
}

std::string range_name(std::uint32_t range) {
    switch (range) {
        case kRangeSrv: return "SRV";
        case kRangeUav: return "UAV";
        case kRangeCbv: return "CBV";
        case kRangeSampler: return "sampler";
    }
    return "range " + hex(range);
}

void add_note_once(std::vector<std::string>& notes, std::string note) {
    if (std::find(notes.begin(), notes.end(), note) == notes.end()) notes.push_back(std::move(note));
}

}  // namespace

MapResult map_user_record(const UserRecord& record, const MapOptions& options) {
    MapResult out;

    // Validate the entry/slot pairing first, independent of the parameter order. Only descriptor entries take a
    // slot: a sampler (0x2) and the unconsumed 0x100 take none, and every 0x401 is a constants parameter.
    std::size_t descriptors = 0;
    std::uint32_t constant_dwords = 0;
    bool saw_sampler = false;
    bool saw_unknown100 = false;
    for (const UserEntry& entry : record.entries) {
        if (is_descriptor(entry.kind)) {
            if (descriptors < record.slots.size()) {
                const UserSlot& slot = record.slots[descriptors];
                const std::uint32_t expected = expected_range(entry.kind);
                if (slot.range != expected) {
                    out.error = "descriptor entry kind " + hex(entry.kind) + " pairs with a " +
                                range_name(slot.range) + " slot (expected " + range_name(expected) + ")";
                    return out;
                }
            }
            if (entry.count == 0) {
                out.error = "descriptor entry kind " + hex(entry.kind) + " has a zero descriptor count";
                return out;
            }
            ++descriptors;
        } else if (entry.kind == kKindConstants) {
            if (entry.a == 0) {
                out.error = "0x401 entry has a zero constant count";
                return out;
            }
            constant_dwords += entry.a;
        } else if (entry.kind == kKindSampler) {
            saw_sampler = true;
        } else if (entry.kind == kKindUnknown100) {
            saw_unknown100 = true;
        } else {
            out.error = "unknown 0x41 entry kind " + hex(entry.kind);
            return out;
        }
    }
    if (record.slots.size() < descriptors) {
        out.error = std::to_string(descriptors) + " descriptor entries but only " +
                    std::to_string(record.slots.size()) + " 0x42 slots: a descriptor has no slot";
        return out;
    }
    if (record.slots.size() > descriptors) {
        for (std::size_t i = descriptors; i < record.slots.size(); ++i) {
            const UserSlot& slot = record.slots[i];
            add_note_once(out.notes, "extra slot SS=" + std::to_string(slot.table) + " II=" +
                                         std::to_string(slot.range) + " RR=" + std::to_string(slot.width) +
                                         " has no 0x41 descriptor");
        }
        // The extra slots are not decoded (#4358): a signature built without them would silently omit
        // user data a shader may read, so refuse instead of producing a partial mapping.
        out.error = std::to_string(record.slots.size() - descriptors) +
                    " extra 0x42 slot(s) have no 0x41 descriptor; the user-data set is incomplete";
        return out;
    }
    if (saw_sampler) add_note_once(out.notes, "0x2 sampler entry takes no slot and no root-parameter dword");
    if (saw_unknown100) add_note_once(out.notes, "0x100 entry kind is not consumed and not mapped");

    // Descriptor ranges are appended in table order; the record does not carry OffsetInDescriptorsFromTableStart.
    std::map<std::uint32_t, std::size_t> table_param;  // SS -> index in signature.params

    const auto add_table = [&](std::uint32_t ss) -> std::size_t {
        const auto found = table_param.find(ss);
        if (found != table_param.end()) return found->second;
        rt::rootsig::RootParameter param;
        param.type = rt::rootsig::ParamType::Table;
        param.visibility = options.visibility;
        out.signature.params.push_back(param);
        const std::size_t index = out.signature.params.size() - 1;
        table_param.emplace(ss, index);
        return index;
    };

    const auto add_range = [&](std::size_t param, const UserEntry& entry, const UserSlot& slot) {
        rt::rootsig::DescriptorRange range;
        range.range_type = slot.range;
        range.num_descriptors = entry.count;
        range.base_shader_register = entry.a;
        range.register_space = entry.b;
        range.flags = 0;
        range.offset_in_table = rt::rootsig::kRangeOffsetAppend;
        out.signature.params[param].ranges.push_back(range);
    };

    const auto add_constants = [&](const UserEntry& entry) {
        rt::rootsig::RootParameter param;
        param.type = rt::rootsig::ParamType::Constants;
        param.visibility = options.visibility;
        param.num_values = entry.a;
        param.shader_register = 0;
        param.register_space = 0;
        out.signature.params.push_back(param);
        add_note_once(out.notes, "0x401 base shader register is not carried by the record (0 used)");
    };

    if (options.order == Order::Table) {
        // Tables ascending by SS, then the constants in record order. The record does not carry this order; it is
        // one hypothesis, never a measurement.
        std::vector<std::uint32_t> sorted;
        for (std::size_t i = 0; i < descriptors; ++i) sorted.push_back(record.slots[i].table);
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        for (std::uint32_t ss : sorted) add_table(ss);
        std::size_t descriptor = 0;
        for (const UserEntry& entry : record.entries) {
            if (is_descriptor(entry.kind)) {
                const UserSlot& slot = record.slots[descriptor++];
                add_range(table_param.at(slot.table), entry, slot);
            } else if (entry.kind == kKindConstants) {
                add_constants(entry);
            }
        }
    } else {
        std::size_t descriptor = 0;
        bool have_current = false;
        std::uint32_t current_ss = 0;
        for (const UserEntry& entry : record.entries) {
            if (is_descriptor(entry.kind)) {
                const UserSlot& slot = record.slots[descriptor++];
                if (have_current && slot.table != current_ss && table_param.count(slot.table)) {
                    add_note_once(out.notes, "table SS=" + std::to_string(slot.table) + " is interleaved with SS=" +
                                                 std::to_string(current_ss) +
                                                 "; the record order is one hypothesis for the parameter order");
                }
                add_range(add_table(slot.table), entry, slot);
                current_ss = slot.table;
                have_current = true;
            } else if (entry.kind == kKindConstants) {
                add_constants(entry);
            }
        }
    }

    if (out.signature.params.size() > 64) {
        out.error = "more than 64 root parameters";
        return out;
    }
    out.signature.version = rt::rootsig::Version::V1_1;
    out.signature.flags = 0;
    out.signature.static_samplers.clear();
    out.console_dwords = constant_dwords + static_cast<std::uint32_t>(table_param.size());
    out.packed = rt::rootsig::pack(out.signature);
    if (!out.packed.ok) {
        out.error = out.packed.error;
        return out;
    }
    out.ok = true;
    return out;
}

}  // namespace rt::pso
