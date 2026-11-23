#pragma once

#include "command_list.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace leonida {

struct PresentScroll {
    Rect destination;
    std::int32_t dx, dy;
};

struct PresentScrollCopy {
    Rect source;
    std::int32_t dst_x, dst_y;
};

struct PresentRegions {
    std::span<const Rect> dirty;
    std::optional<PresentScrollCopy> scroll;
};

inline bool present_rect_fits(const Rect& rect, std::uint32_t width, std::uint32_t height) {
    return rect.left >= 0 && rect.top >= 0 && rect.left < rect.right && rect.top < rect.bottom &&
           std::uint32_t(rect.right) <= width && std::uint32_t(rect.bottom) <= height;
}

inline std::optional<PresentScrollCopy> present_scroll_copy(const PresentScroll& scroll,
                                                           std::uint32_t width, std::uint32_t height) {
    if (!present_rect_fits(scroll.destination, width, height)) return std::nullopt;
    const std::int64_t x = scroll.dx, y = scroll.dy;
    constexpr auto max_coordinate = (std::numeric_limits<std::int32_t>::max)();
    const auto frame_width = (std::min<std::int64_t>)(width, max_coordinate);
    const auto frame_height = (std::min<std::int64_t>)(height, max_coordinate);
    // The scroll offset moves old pixels into the requested destination. Crop that rectangle to the pixels
    // actually available in the old frame, and keep their source coordinates in step.
    const auto left = (std::max<std::int64_t>)(scroll.destination.left, x);
    const auto top = (std::max<std::int64_t>)(scroll.destination.top, y);
    const auto right = (std::min<std::int64_t>)(scroll.destination.right, frame_width + x);
    const auto bottom = (std::min<std::int64_t>)(scroll.destination.bottom, frame_height + y);
    if (right <= left || bottom <= top) return std::nullopt;
    return PresentScrollCopy{{std::int32_t(left - x), std::int32_t(top - y),
                              std::int32_t(right - x), std::int32_t(bottom - y)},
                             std::int32_t(left), std::int32_t(top)};
}

}  // namespace leonida
