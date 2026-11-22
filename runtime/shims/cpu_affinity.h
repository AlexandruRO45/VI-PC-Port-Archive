// Title CPU affinity: maps Xbox Series console core masks onto host physical cores.
//
// See cpu_affinity.cpp for the model and shimmed Win32 exports. This header exposes the pure topology model and
// test seam so unit tests can verify the mapping independently of the machine they run on.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace rt::cpu {

struct Topology {
    int cores = 0;
    uint64_t full = 0;
    std::vector<uint64_t> map;  // map[i] = host logical-processor mask for console core i

    std::optional<uint64_t> to_host(uint64_t console_mask) const noexcept;
    uint64_t to_console(uint64_t host_mask) const noexcept;
    uint32_t host_processor(uint32_t console_index) const noexcept;
    uint32_t to_console_index(uint32_t host_number) const noexcept;
};

int parse_title_cores(const char* text) noexcept;
Topology make_topology(const std::vector<uint64_t>& host_cores, int cores);
void set_topology_for_test(const Topology* t) noexcept;  // nullptr = back to the real machine

}  // namespace rt::cpu
