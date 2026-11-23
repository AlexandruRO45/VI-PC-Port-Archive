// swapchain_blit (runtime/src/present_scaling.h): where Present puts a back buffer in the window's image for each
// DXGI_SCALING (D3D-444). CPU only.
#include "../src/present_scaling.h"

#include <cstdio>

using leonida::SwapchainBlit;
using leonida::PresentScaling;
using leonida::swapchain_blit;

namespace {

int failures = 0;

void expect(PresentScaling s, unsigned sw, unsigned sh, unsigned dw, unsigned dh, const SwapchainBlit& want,
            const char* what, bool right_to_left = false) {
    const SwapchainBlit r = swapchain_blit(s, sw, sh, dw, dh, right_to_left);
    const bool ok = r.src_x == want.src_x && r.src_width == want.src_width && r.src_height == want.src_height &&
                    r.dst_x == want.dst_x &&
                    r.dst_y == want.dst_y && r.dst_width == want.dst_width && r.dst_height == want.dst_height &&
                    r.scaled == want.scaled && r.covers == want.covers;
    if (ok) return;
    ++failures;
    std::printf("FAIL %s: src x%u %ux%u -> dst (%u,%u) %ux%u scaled %d covers %d\n", what, r.src_x, r.src_width, r.src_height,
                r.dst_x, r.dst_y, r.dst_width, r.dst_height, int(r.scaled), int(r.covers));
}

SwapchainBlit rect(unsigned sw, unsigned sh, unsigned x, unsigned y, unsigned w, unsigned h, bool scaled, bool covers) {
    SwapchainBlit r;
    r.src_width = sw, r.src_height = sh, r.dst_x = x, r.dst_y = y, r.dst_width = w, r.dst_height = h;
    r.scaled = scaled, r.covers = covers;
    return r;
}

}  // namespace

int main() {
    using S = PresentScaling;
    // STRETCH: the whole back buffer fills the window, scaled either way.
    expect(S::stretch, 64, 48, 128, 128, rect(64, 48, 0, 0, 128, 128, true, true), "stretch up");
    expect(S::stretch, 256, 256, 128, 64, rect(256, 256, 0, 0, 128, 64, true, true), "stretch down");
    expect(S::stretch, 128, 128, 128, 128, rect(128, 128, 0, 0, 128, 128, false, true), "stretch same size");
    // NONE: texel for texel at the top-left corner; a larger back buffer is cropped, a smaller one leaves the rest.
    expect(S::none, 64, 48, 128, 128, rect(64, 48, 0, 0, 64, 48, false, false), "none smaller");
    expect(S::none, 200, 100, 128, 128, rect(128, 100, 0, 0, 128, 100, false, false), "none wider, shorter");
    expect(S::none, 300, 300, 128, 128, rect(128, 128, 0, 0, 128, 128, false, true), "none larger (cropped)");
    expect(S::none, 128, 128, 128, 128, rect(128, 128, 0, 0, 128, 128, false, true), "none same size");
    // NONE in a right-to-left window (WS_EX_LAYOUTRTL): top-right corner, as host DXGI places it.
    expect(S::none, 64, 48, 128, 128, rect(64, 48, 64, 0, 64, 48, false, false), "none smaller, RTL", true);
    {  // a back buffer wider than an RTL window shows its right part (source columns 64..191 of 192)
        SwapchainBlit want = rect(128, 128, 0, 0, 128, 128, false, true);
        want.src_x = 64;
        expect(S::none, 192, 160, 128, 128, want, "none wider and taller, RTL", true);
        SwapchainBlit wide = rect(128, 100, 0, 0, 128, 100, false, false);
        wide.src_x = 72;
        expect(S::none, 200, 100, 128, 128, wide, "none wider, shorter, RTL", true);
    }
    {
        SwapchainBlit want = rect(128, 128, 0, 0, 128, 128, false, true);
        want.src_x = 172;
        expect(S::none, 300, 300, 128, 128, want, "none larger, RTL", true);
    }
    // STRETCH and ASPECT_RATIO_STRETCH don't depend on the reading direction.
    expect(S::stretch, 64, 48, 128, 128, rect(64, 48, 0, 0, 128, 128, true, true), "stretch, RTL", true);
    expect(S::aspect_ratio_stretch, 64, 32, 128, 128, rect(64, 32, 0, 32, 128, 64, true, false), "aspect wide, RTL",
           true);
    // ASPECT_RATIO_STRETCH: the largest aspect-keeping size, centred (CoreWindow swapchains only).
    expect(S::aspect_ratio_stretch, 64, 32, 128, 128, rect(64, 32, 0, 32, 128, 64, true, false), "aspect wide");
    expect(S::aspect_ratio_stretch, 32, 64, 128, 128, rect(32, 64, 32, 0, 64, 128, true, false), "aspect tall");
    expect(S::aspect_ratio_stretch, 1920, 1080, 1280, 720, rect(1920, 1080, 0, 0, 1280, 720, true, true),
           "aspect same ratio");
    expect(S::aspect_ratio_stretch, 1920, 1080, 1280, 1024, rect(1920, 1080, 0, 152, 1280, 720, true, false),
           "aspect letterbox");
    expect(S::aspect_ratio_stretch, 3, 1, 1000, 1000, rect(3, 1, 0, 333, 1000, 333, true, false), "aspect rounding");
    expect(S::aspect_ratio_stretch, 0, 0, 128, 128, rect(0, 0, 0, 0, 0, 0, false, false), "aspect empty source");
    // Large sizes don't overflow the 32-bit products.
    expect(S::aspect_ratio_stretch, 65536, 65535, 65535, 65535, rect(65536, 65535, 0, 0, 65535, 65534, true, false),
           "aspect large");
    std::printf("present_scaling_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
