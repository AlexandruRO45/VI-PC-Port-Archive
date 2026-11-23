// CPU-only unit test for the DLSS backend's translation rules: it drives the pure functions
// (pick_quality / create_flags / effective_mv_scale / preset_key) with no D3D12 device and no NGX, so it
// runs everywhere and guards the game-measured mapping that the .cpp must not drift from.
#include <cstdio>

#include "../src/dlss_backend.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    using leonida::dlss::pick_quality;
    using leonida::dlss::create_flags;
    using leonida::dlss::effective_mv_scale;
    using leonida::dlss::preset_key;
    using nvngx::PerfQualityValue;

    // pick_quality: the measured FFX slots of 1.0.2.1 — 1280x720 -> 1920x1080 is 1.5x.
    check(pick_quality(1.5f) == nvngx::PERF_VALUE_MAX_QUALITY, "1.5x -> MaxQuality (the game's FSR Quality slot)");
    check(pick_quality(2.0f) == nvngx::PERF_VALUE_ULTRA_QUALITY, "2.0x -> UltraQuality");
    check(pick_quality(1.7f) == nvngx::PERF_VALUE_ULTRA_QUALITY, "1.7x -> UltraQuality");
    // The shipped-enum collisions: 1.7237x is Balanced's own ratio and UltraQuality does not win it.
    check(pick_quality(1.7237f) == nvngx::PERF_VALUE_BALANCED, "1.7237x -> Balanced (shipped collision)");
    check(pick_quality(1.72f) == nvngx::PERF_VALUE_BALANCED, "1.72x -> Balanced");
    check(pick_quality(1.705f) == nvngx::PERF_VALUE_ULTRA_QUALITY, "1.705x -> UltraQuality (under the 1.71 cut)");
    check(pick_quality(3.0f) == nvngx::PERF_VALUE_MAX_PERF, "3.0x -> MaxPerf (shipped collision with UltraPerf)");
    check(pick_quality(2.99f) == nvngx::PERF_VALUE_ULTRA_QUALITY, "2.99x -> UltraQuality");
    check(pick_quality(4.0f) == nvngx::PERF_VALUE_MAX_PERF, "4.0x -> MaxPerf (UltraPerformance lives above 3.0)");
    check(pick_quality(1.06f) == nvngx::PERF_VALUE_MAX_QUALITY, "1.06x -> MaxQuality band");
    check(pick_quality(1.05f) == nvngx::PERF_VALUE_DLAA, "1.05x -> DLAA (boundary)");
    check(pick_quality(1.0f) == nvngx::PERF_VALUE_DLAA, "1.0x -> DLAA");

    // create_flags: the measured 1.0.2.1 FSR4 context is HDR + low-res MVs + inverted depth + auto exposure
    // (the game renders motion vectors at render resolution, so MV_LOW_RES rides the default).
    leonida::dlss::Description game{};
    game.upscale_ratio = 1.5f;
    game.hdr = true;
    game.depth_inverted = true;
    game.auto_exposure = true;
    const int expect_game = nvngx::DLSS_FLAG_IS_HDR | nvngx::DLSS_FLAG_MV_LOW_RES |
                            nvngx::DLSS_FLAG_DEPTH_INVERTED | nvngx::DLSS_FLAG_AUTO_EXPOSURE;
    check(create_flags(game) == expect_game, "game-measured context -> HDR|MVLowRes|DepthInverted|AutoExposure");
    leonida::dlss::Description all{};
    all.hdr = all.mv_low_res = all.mv_jittered = all.depth_inverted = all.auto_exposure = true;
    const int expect_all = nvngx::DLSS_FLAG_IS_HDR | nvngx::DLSS_FLAG_MV_LOW_RES | nvngx::DLSS_FLAG_MV_JITTERED |
                           nvngx::DLSS_FLAG_DEPTH_INVERTED | nvngx::DLSS_FLAG_AUTO_EXPOSURE;
    check(create_flags(all) == expect_all, "all set -> all bits");
    leonida::dlss::Description none{};
    check(none.mv_low_res == true, "mv_low_res defaults to true (the game renders MVs at render resolution)");
    check(create_flags(none) == nvngx::DLSS_FLAG_MV_LOW_RES, "defaults -> MV_LOW_RES only");

    // effective_mv_scale: the shipped helper's 0-means-1.0 rule.
    check(effective_mv_scale(0.0f) == 1.0f, "mv scale 0 -> 1.0");
    check(effective_mv_scale(0.5f) == 0.5f, "mv scale passes through");
    check(effective_mv_scale(-0.0f) == 1.0f, "negative zero -> 1.0 as well");

    // preset_key: one key per quality slot.
    check(preset_key(nvngx::PERF_VALUE_MAX_QUALITY) == nvngx::PRESET_KEY_QUALITY, "MaxQuality -> Quality preset key");
    check(preset_key(nvngx::PERF_VALUE_BALANCED) == nvngx::PRESET_KEY_BALANCED, "Balanced -> Balanced key");
    check(preset_key(nvngx::PERF_VALUE_MAX_PERF) == nvngx::PRESET_KEY_PERFORMANCE, "MaxPerf -> Performance key");
    check(preset_key(nvngx::PERF_VALUE_ULTRA_PERFORMANCE) == nvngx::PRESET_KEY_ULTRA_PERFORMANCE,
          "UltraPerformance -> UltraPerformance key");
    check(preset_key(nvngx::PERF_VALUE_ULTRA_QUALITY) == nvngx::PRESET_KEY_ULTRA_QUALITY,
          "UltraQuality -> UltraQuality key");
    check(preset_key(nvngx::PERF_VALUE_DLAA) == nvngx::PRESET_KEY_DLAA, "DLAA -> DLAA key");

    // PresetValue enum: the shipped J/K/L/M numbering must not drift (K=11 transformer, L=12, M=13).
    check(nvngx::PRESET_J == 10 && nvngx::PRESET_K == 11 && nvngx::PRESET_L == 12 && nvngx::PRESET_M == 13,
          "preset enum keeps shipped numbering");

    std::printf("dlss_backend_unit: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
