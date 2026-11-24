#include <cstdint>

using Draw = int (*)(std::uint32_t);

extern "C" __declspec(dllexport) int frame_render(std::uint32_t selector, Draw draw) {
    return draw(selector);
}
