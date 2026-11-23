// AccelerationStructureRanges (runtime/src/acceleration_structure_ranges.h): which structure the KHR path builds into
// or names for an address inside a D3D12 acceleration-structure resource. CPU only: the handles are numbers and the
// create function records its calls instead of calling vkCreateAccelerationStructureKHR.
//
// The case it is for: a BLAS built at GetGPUVirtualAddress() + N of a resource used to find nothing (an exact-key
// lookup of the resource's start), so the whole ExecuteCommandLists was dropped as unresolved_resource.

#include "../src/acceleration_structure_ranges.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

struct Where {
    int buffer;
    std::uint64_t offset;
};

struct Made {
    int buffer;
    std::uint64_t offset;  // in the buffer
    std::uint64_t room;
};

using Ranges = leonida::AccelerationStructureRanges<std::uint64_t, Where>;

struct Creator {
    std::vector<Made> calls;
    bool fail = false;
    std::uint64_t operator()(const Where& where, std::uint64_t offset, std::uint64_t room) {
        if (fail) return 0;
        calls.push_back({where.buffer, where.offset + offset, room});
        return 0x100 + calls.size();  // a distinct non-null handle per call
    }
};

void offsets_inside_one_resource() {
    Ranges ranges;
    Creator create;
    std::vector<std::uint64_t> retired;
    // A committed resource with its own buffer (offset 0) and one in the arena at 0x30000, 0x1000 bytes each.
    ranges.add(0x7000'0000, 0x1000, Where{1, 0}, retired);
    ranges.add(0x9000'0000, 0x1000, Where{2, 0x30000}, retired);

    const std::uint64_t a = ranges.find(0x7000'0300, create);
    check(a != 0, "a 256-byte aligned address inside a resource gets a structure");
    check(create.calls.size() == 1 && create.calls[0].buffer == 1 && create.calls[0].offset == 0x300 &&
              create.calls[0].room == 0xD00,
          "it sits at the address's offset in the resource's buffer and runs to the resource's end");
    check(ranges.find(0x7000'0300, create) == a && create.calls.size() == 1, "the second lookup reuses it");

    const std::uint64_t b = ranges.find(0x9000'0100, create);
    check(b != 0 && b != a, "another resource's address gets its own structure");
    check(create.calls.size() == 2 && create.calls[1].buffer == 2 && create.calls[1].offset == 0x30100 &&
              create.calls[1].room == 0xF00,
          "an arena resource's structure is placed from the resource's offset in the arena buffer");

    check(ranges.find(0x7000'0000 + 0xF00, create) != 0, "the last aligned address of a resource is inside it");
    check(create.calls.back().room == 0x100, "and has the last 256 bytes");
    const std::size_t made = create.calls.size();
    check(ranges.find(0x7000'1000, create) == 0, "the address just past a resource is not in it");
    check(ranges.find(0x6FFF'FF00, create) == 0, "an address before every resource is in none");
    check(ranges.find(0x7000'0310, create) == 0, "an address that is not 256-byte aligned names no structure");
    check(ranges.find(0, create) == 0, "a null address names no structure");
    check(create.calls.size() == made, "nothing is created for an address that names no structure");
}

void removal_hands_back_what_was_made() {
    Ranges ranges;
    Creator create;
    std::vector<std::uint64_t> retired;
    ranges.add(0x1000'0000, 0x10000, Where{1, 0}, retired);
    ranges.add(0x2000'0000, 0x10000, Where{2, 0}, retired);
    const std::uint64_t a = ranges.find(0x1000'0400, create);
    const std::uint64_t b = ranges.find(0x1000'8000, create);
    const std::uint64_t c = ranges.find(0x2000'0200, create);
    check(retired.empty(), "adding new ranges retires nothing");

    ranges.remove(0x1000'0000, retired);
    check(retired.size() == 2 && ((retired[0] == a && retired[1] == b) || (retired[0] == b && retired[1] == a)),
          "removing a resource's range hands back exactly the structures made inside it");
    check(ranges.find(0x1000'0400, create) == 0, "a removed resource's addresses name nothing");
    check(ranges.find(0x2000'0200, create) == c, "the other resource keeps its structure");
    ranges.remove(0x1000'0000, retired);
    check(retired.size() == 2, "removing it again retires nothing more");
    ranges.remove(0x2000'0200, retired);
    check(retired.size() == 2 && ranges.ranges() == 1, "only a range's start removes it");

    // The resource's address is reused by a new resource: the new range starts empty, the old structures go.
    retired.clear();
    ranges.add(0x2000'0000, 0x400, Where{3, 0}, retired);
    check(retired.size() == 1 && retired[0] == c, "a range registered again at the same start retires the old one's");
    check(ranges.find(0x2000'0200, create) != c && create.calls.back().buffer == 3 && create.calls.back().room == 0x200,
          "the new range makes its own structure");
}

void a_failed_create_is_not_kept() {
    Ranges ranges;
    Creator create;
    std::vector<std::uint64_t> retired;
    ranges.add(0x4000'0000, 0x1000, Where{1, 0}, retired);
    create.fail = true;
    check(ranges.find(0x4000'0100, create) == 0, "a structure the driver refuses names nothing");
    create.fail = false;
    check(ranges.find(0x4000'0100, create) != 0, "and the next lookup tries again");
    ranges.remove(0x4000'0000, retired);
    check(retired.size() == 1, "only the structure that was made is handed back");
}

}  // namespace

int main() {
    offsets_inside_one_resource();
    removal_hands_back_what_was_made();
    a_failed_create_is_not_kept();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::puts("AccelerationStructureRanges: offsets, removal and failed creates");
    return 0;
}
