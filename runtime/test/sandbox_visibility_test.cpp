#include "sandbox_visibility.h"

#include <cstdio>

namespace {

int failures = 0;

void expect(bool condition, const char* description) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        ++failures;
    }
}

using sandbox_visibility::Bounds;
using sandbox_visibility::Frustum;

Bounds box(float x0, float y0, float z0, float x1, float y1, float z1) {
    Bounds result;
    const float low[3] = {x0, y0, z0}, high[3] = {x1, y1, z1};
    result.include(low);
    result.include(high);
    return result;
}

}  // namespace

int main() {
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const Frustum clip(identity);
    expect(clip.visible(box(-.5f, -.5f, .2f, .5f, .5f, .8f)), "inside");
    expect(!clip.visible(box(-3, -.5f, .2f, -2, .5f, .8f)), "outside left");
    expect(!clip.visible(box(2, -.5f, .2f, 3, .5f, .8f)), "outside right");
    expect(!clip.visible(box(-.5f, -3, .2f, .5f, -2, .8f)), "outside bottom");
    expect(!clip.visible(box(-.5f, 2, .2f, .5f, 3, .8f)), "outside top");
    expect(!clip.visible(box(-.5f, -.5f, -.8f, .5f, .5f, -.2f)), "D3D near plane is z=0");
    expect(!clip.visible(box(-.5f, -.5f, 2, .5f, .5f, 3)), "outside far");
    expect(clip.visible(box(-2, -2, -.5f, 2, 2, 2)), "box enclosing frustum");
    expect(clip.visible(box(1, -.5f, .2f, 2, .5f, .8f)), "touching side");
    expect(clip.visible(box(-.5f, -.5f, -.1f, .5f, .5f, 0)), "touching near plane");
    expect(clip.visible(box(-.5f, -.5f, 1, .5f, .5f, 2)), "touching far plane");
    const float just_outside = std::nextafter(1.f, 2.f);
    expect(clip.visible(box(just_outside, 0, .5f, just_outside, 0, .5f)), "rounding margin at plane");

    const float perspective[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10.f / 9, 1, 0, 0, -10.f / 9, 0};
    const Frustum camera(perspective);
    expect(camera.visible(box(-.5f, -.5f, 2, .5f, .5f, 3)), "perspective inside");
    expect(!camera.visible(box(-.5f, -.5f, -3, .5f, .5f, -2)), "behind camera");
    expect(!camera.visible(box(-.1f, -.1f, .2f, .1f, .1f, .8f)), "before perspective near plane");
    expect(camera.visible(box(-.5f, -.5f, .5f, .5f, .5f, 2)), "near plane crossing");
    expect(camera.visible(box(-.5f, -.5f, 9, .5f, .5f, 11)), "far plane crossing");
    expect(!camera.visible(box(-.5f, -.5f, 11, .5f, .5f, 12)), "beyond perspective far plane");
    expect(camera.visible(box(-1, -1, -1, 1, 1, 2)), "camera inside bounds");

    // Quarter turn, reflection, nonuniform scale and shear, then translation.
    const float rows[4][3] = {{0, -2, 0}, {3, 0, 0}, {1, 0, -4}, {8, 12, 20}};
    const Bounds world = sandbox_visibility::transformed(box(-1, -2, -3, 1, 2, 3), rows);
    expect(world.finite(), "transformed bounds finite");
    expect(world.low[0] <= -1 && world.high[0] >= 17, "rotated X extent");
    expect(world.low[1] <= 10 && world.high[1] >= 14, "rotated Y extent");
    expect(world.low[2] <= 8 && world.high[2] >= 32, "reflected Z extent");
    expect(!clip.visible(world), "transformed outside");

    // Large view translations cancel world coordinates. Keep a point on the side plane.
    float translated[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1e8f, 0, 0, 1};
    expect(Frustum(translated).visible(box(1e8f, 0, .5f, 1e8f, 0, .5f)), "large coordinate cancellation");
    expect(clip.visible(Bounds{}), "empty bounds stay visible");
    Bounds invalid = box(2, 0, .5f, 3, 0, .5f);
    const float nan_point[3] = {std::numeric_limits<float>::quiet_NaN(), 0, 0};
    invalid.include(nan_point);
    const float finite_point[3] = {4, 0, 0};
    invalid.include(finite_point);
    expect(clip.visible(invalid), "nonfinite vertices keep bounds unusable");
    expect(clip.visible(sandbox_visibility::transformed(invalid, rows)), "invalid transformed bounds");
    translated[0] = std::numeric_limits<float>::infinity();
    expect(Frustum(translated).visible(world), "nonfinite projection stays visible");

    if (failures) return 1;
    std::puts("sandbox visibility checks passed");
    return 0;
}
