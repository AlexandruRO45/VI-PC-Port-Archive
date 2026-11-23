// Unit tests for VR mode's settings and arithmetic (runtime/src/vr_config.h): LEONIDA_VR and the screen variables,
// the screen's pose and size, the extension summary and the headset swapchain format. No OpenXR runtime, headset,
// Vulkan instance or GPU: every function is pure, and the environment is a map.
#include "vr_config.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

bool near(float a, float b, float tolerance = 1e-5f) { return std::fabs(a - b) <= tolerance; }

leonida::vr::EnvLookup env(std::map<std::string, std::string> values) {
    return [values](const char* name) -> std::optional<std::string> {
        const auto it = values.find(name);
        if (it == values.end()) return std::nullopt;
        return it->second;
    };
}

}  // namespace

int main() {
    using namespace leonida::vr;

    // LEONIDA_VR: off unless it says openxr; anything else is reported (unknown) and stays flat.
    expect(parse_mode(nullptr) == Mode::off, "unset is off");
    expect(parse_mode("") == Mode::off, "empty is off");
    expect(parse_mode("0") == Mode::off, "0 is off");
    expect(parse_mode("OFF") == Mode::off, "OFF is off");
    expect(parse_mode("openxr") == Mode::openxr, "openxr");
    expect(parse_mode("OpenXR") == Mode::openxr, "OpenXR, any case");
    expect(parse_mode("1") == Mode::unknown, "1 is not a VR mode");
    expect(parse_mode("openvr") == Mode::unknown, "openvr is not a VR mode");
    expect(parse_mode("openxr ") == Mode::unknown, "no trimming");
    expect(mode_from(env({})) == Mode::off, "mode_from: unset");
    expect(mode_from(env({{"LEONIDA_VR", "openxr"}})) == Mode::openxr, "mode_from: openxr");

    // Screen settings: defaults, valid values, and values that are ignored with one warning each.
    {
        const CinemaSettings s = cinema_settings_from(env({}));
        expect(s.warnings.empty(), "defaults: no warnings");
        expect(near(s.config.distance_m, 3.0f) && near(s.config.width_m, 3.2f) && near(s.config.height_m, 0.0f),
               "defaults: 3.0 m away, 3.2 m wide, at eye height");
        // About 56 degrees wide: a comfortable cinema seat, the whole frame in view without turning the head.
        const float fov = horizontal_fov_degrees(s.config);
        expect(fov > 50.0f && fov < 60.0f, "default screen fills 50 to 60 degrees");
    }
    {
        const CinemaSettings s = cinema_settings_from(env({{"LEONIDA_VR_SCREEN_DISTANCE", "2.5"},
                                                           {"LEONIDA_VR_SCREEN_WIDTH", "4"},
                                                           {"LEONIDA_VR_SCREEN_HEIGHT", "-0.25"}}));
        expect(s.warnings.empty(), "valid values: no warnings");
        expect(near(s.config.distance_m, 2.5f) && near(s.config.width_m, 4.0f) && near(s.config.height_m, -0.25f),
               "valid values are taken");
    }
    {
        const CinemaSettings s = cinema_settings_from(env({{"LEONIDA_VR_SCREEN_DISTANCE", "3m"},
                                                           {"LEONIDA_VR_SCREEN_WIDTH", "100"},
                                                           {"LEONIDA_VR_SCREEN_HEIGHT", "nan"}}));
        expect(s.warnings.size() == 3, "three bad values: three warnings");
        expect(near(s.config.distance_m, 3.0f) && near(s.config.width_m, 3.2f) && near(s.config.height_m, 0.0f),
               "bad values keep the defaults");
        expect(!s.warnings.empty() && s.warnings[0].find("LEONIDA_VR_SCREEN_DISTANCE=3m") != std::string::npos,
               "the warning names the variable and its value");
    }
    {
        const CinemaSettings s = cinema_settings_from(env({{"LEONIDA_VR_SCREEN_DISTANCE", "0.4"},
                                                           {"LEONIDA_VR_SCREEN_WIDTH", ""},
                                                           {"LEONIDA_VR_SCREEN_HEIGHT", "inf"}}));
        expect(s.warnings.size() == 2, "below the range and inf are ignored; empty is unset");
        expect(near(s.config.distance_m, 3.0f), "0.4 m is too close");
    }
    {
        const CinemaSettings s = cinema_settings_from(env({{"LEONIDA_VR_SCREEN_DISTANCE", "0.5"},
                                                           {"LEONIDA_VR_SCREEN_WIDTH", "40"},
                                                           {"LEONIDA_VR_SCREEN_HEIGHT", "-10"}}));
        expect(s.warnings.empty(), "range ends are allowed");
    }

    // The screen's placement: centred ahead (-Z, OpenXR is Y up), raised by the height, aspect from the frame.
    {
        CinemaConfig c;
        c.distance_m = 2.0f;
        c.width_m = 3.2f;
        c.height_m = 0.1f;
        const QuadPlacement q = place_screen(c, 1920, 1080);
        expect(near(q.pose.position.x, 0.0f) && near(q.pose.position.y, 0.1f) && near(q.pose.position.z, -2.0f),
               "screen centre 2 m ahead, 0.1 m up");
        expect(near(q.width_m, 3.2f) && near(q.height_m, 1.8f), "16:9 frame: 3.2 m x 1.8 m");
        const QuadPlacement tall = place_screen(c, 1000, 2000);
        expect(near(tall.height_m, 6.4f), "a portrait frame is twice as tall as wide");
        const QuadPlacement none = place_screen(c, 0, 1080);
        expect(none.width_m == 0.0f && none.height_m == 0.0f, "an empty frame gives an empty quad");
        // The visible (+Z) side faces the viewer at the origin: the direction from the screen to the head.
        const Vec3 normal = rotate(q.pose.orientation, {0, 0, 1});
        const Vec3 to_head{-q.pose.position.x, -q.pose.position.y, -q.pose.position.z};
        expect(normal.x * to_head.x + normal.y * to_head.y + normal.z * to_head.z > 0.0f, "screen faces the viewer");
        // Its right edge (+X) is on the viewer's right, its top (+Y) up: not mirrored, not upside down.
        const Vec3 right = rotate(q.pose.orientation, {1, 0, 0}), up = rotate(q.pose.orientation, {0, 1, 0});
        expect(near(right.x, 1.0f) && near(up.y, 1.0f), "screen is upright and not mirrored");
        // The quaternion rotation itself: 90 degrees about +Y turns -Z (ahead) into -X (left).
        const float h = std::sqrt(0.5f);
        const Vec3 turned = rotate(Quat{0, h, 0, h}, {0, 0, -1});
        expect(near(turned.x, -1.0f) && near(turned.y, 0.0f) && near(turned.z, 0.0f), "quaternion rotation");
    }
    {
        CinemaConfig c;
        c.distance_m = 1.0f;
        c.width_m = 2.0f;
        expect(near(horizontal_fov_degrees(c), 90.0f, 1e-3f), "2 m wide at 1 m is 90 degrees");
    }

    // Extensions: what the log line says for a runtime's list.
    {
        const ExtensionSupport none = extension_support({});
        expect(!none.vulkan_enable2 && !none.eye_gaze_interaction && !none.varjo_quad_views, "empty list");
        const ExtensionSupport s = extension_support({"XR_KHR_vulkan_enable2", "XR_EXT_eye_gaze_interaction",
                                                      "XR_FB_foveation", "XR_FB_foveation_vulkan",
                                                      "XR_META_foveation_eye_tracked", "XR_KHR_composition_layer_depth"});
        expect(s.vulkan_enable2 && s.eye_gaze_interaction && s.fb_foveation && s.fb_foveation_vulkan &&
                   s.meta_foveation_eye_tracked && !s.varjo_quad_views && !s.varjo_foveated_rendering,
               "the listed extensions, nothing else");
        const std::string line = extension_summary(s);
        expect(line.find("vulkan_enable2 yes") != std::string::npos, "summary: vulkan_enable2 yes");
        expect(line.find("EXT_eye_gaze_interaction yes") != std::string::npos, "summary: eye gaze yes");
        expect(line.find("VARJO_quad_views no") != std::string::npos, "summary: quad views no");
        expect(line.size() < 200, "summary is one short line");
        expect(extension_support({"XR_KHR_vulkan_enable"}).vulkan_enable2 == false,
               "XR_KHR_vulkan_enable is not XR_KHR_vulkan_enable2");
    }

    // The headset swapchain format: the sRGB format with the window's channel order, or none.
    {
        const std::vector<std::int64_t> both{vk_format_r8g8b8a8_srgb, vk_format_b8g8r8a8_srgb, vk_format_r8g8b8a8_unorm};
        expect(cinema_swapchain_format(vk_format_b8g8r8a8_unorm, both) == vk_format_b8g8r8a8_srgb, "BGRA UNORM window");
        expect(cinema_swapchain_format(vk_format_b8g8r8a8_srgb, both) == vk_format_b8g8r8a8_srgb, "BGRA sRGB window");
        expect(cinema_swapchain_format(vk_format_r8g8b8a8_unorm, both) == vk_format_r8g8b8a8_srgb, "RGBA UNORM window");
        expect(cinema_swapchain_format(vk_format_b8g8r8a8_unorm, {vk_format_r8g8b8a8_srgb}) == 0,
               "no BGRA sRGB offered: none (a copy cannot swap channels)");
        expect(cinema_swapchain_format(vk_format_b8g8r8a8_unorm, {vk_format_b8g8r8a8_unorm}) == 0,
               "a UNORM headset image would be decoded as linear: none");
        expect(cinema_swapchain_format(64 /* VK_FORMAT_A2B10G10R10_UNORM_PACK32, HDR10 */, both) == 0, "HDR10: none");
        expect(cinema_swapchain_format(vk_format_b8g8r8a8_unorm, {}) == 0, "nothing offered");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("vr_config: all checks passed\n");
    return 0;
}
