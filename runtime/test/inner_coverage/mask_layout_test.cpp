#include "../../src/inner_coverage_mask.h"
#include <cstdio>
#include <set>
#include <utility>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    using leonida::conservative_raster::InnerCoverageMask;
    // Cross both word and primitive boundaries; every pixel/primitive/instance must be independent.
    const auto layout = InnerCoverageMask::make(5, 7, 3, 2, 56);
    CHECK(layout && layout->bits == 210 && layout->bytes == 28);
    std::set<std::pair<std::uint64_t, std::uint32_t>> addresses;
    for (unsigned i = 0; i < 2; ++i) for (unsigned p = 0; p < 3; ++p)
        for (unsigned y = 0; y < 7; ++y) for (unsigned x = 0; x < 5; ++x) {
            const auto a = layout->address(x, y, p, i);
            CHECK(a && a->byte_offset < layout->bytes && a->byte_offset % 4 == 0);
            CHECK(addresses.emplace(a->byte_offset, a->bit).second);
        }
    CHECK(!layout->address(5, 0, 0, 0) && !layout->address(0, 7, 0, 0));
    CHECK(!layout->address(0, 0, 3, 0) && !layout->address(0, 0, 0, 2));
    CHECK(!InnerCoverageMask::make(0, 7, 3, 2, 56));
    CHECK(!InnerCoverageMask::make(5, 7, 0, 2, 56));
    CHECK(!InnerCoverageMask::make(5, 7, 3, 0, 56));
    CHECK(!InnerCoverageMask::make(5, 7, 3, 2, 27));
    CHECK(!InnerCoverageMask::make(0xffffffffu, 2, 1, 1, ~0ull));
    CHECK(!InnerCoverageMask::make(1, 1, 0xffffffffu, 0xffffffffu, ~0ull));
    CHECK(InnerCoverageMask::make(32, 32, 64, 1, 8192)->bytes == 8192);
    CHECK(InnerCoverageMask::make(1, 1, 1, 1, 4)->address(0, 0, 0, 0)->bit == 1);
    std::puts("PASS: unique mask addresses, bounds, budget and overflow controls");
}
