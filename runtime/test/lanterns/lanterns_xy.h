// Lanterns stage 24: local X and local Y thread IDs (cs_lanterns_xy.asm), and the CPU reference.
// The title's compute shaders enable both local IDs (TIDIG_COMP_CNT 1 or more) and read v1 for Y; every earlier Lanterns
// compute shader has a 1-D local size. Here the group is 16x8 (four wave32 waves) on a 2x3 grid, and each lane's two
// words depend on X and Y separately: its own position, and that of its point mirror in the group, read through LDS
// after the barrier. Swapping X and Y, or a Y that stays 0, moves the words.
#pragma once

#include <array>
#include <cstdint>

namespace lanterns {

constexpr unsigned kXyGroupX = 16, kXyGroupY = 8;   // the local size
constexpr unsigned kXyGroupsX = 2, kXyGroupsY = 3;  // the grid
constexpr unsigned kXyWidth = kXyGroupX * kXyGroupsX, kXyHeight = kXyGroupY * kXyGroupsY;
constexpr std::uint32_t kXyPoison = 0xdeadbeefu;

inline std::uint32_t xy_value(unsigned gx, unsigned gy) { return gx + 1000u * gy; }

// The two words at grid position (gx, gy): its own value and its mirror's in the group.
inline std::array<std::uint32_t, 2> xy_expect(unsigned gx, unsigned gy) {
    const unsigned ox = gx / kXyGroupX * kXyGroupX, oy = gy / kXyGroupY * kXyGroupY;
    const unsigned x = gx - ox, y = gy - oy;
    return {xy_value(gx, gy), xy_value(ox + kXyGroupX - 1 - x, oy + kXyGroupY - 1 - y)};
}

}  // namespace lanterns
