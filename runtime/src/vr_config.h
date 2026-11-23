#pragma once

// LEONIDA_VR (docs/vr.md): the settings and the arithmetic of VR mode, apart from OpenXR itself so that they are
// tested on every platform without a headset, a runtime or the OpenXR headers. The OpenXR session
// (shims/openxr_session.cpp, built with -DLEONIDA_OPENXR=ON) reads them.
//
// LEONIDA_VR=openxr shows every presented frame on a flat screen in the headset (the "virtual cinema"): an OpenXR quad
// layer in the LOCAL reference space, which is Y up and right-handed with -Z ahead of where the head was when the
// session started. The screen's size and place come from three variables, in metres:
//   LEONIDA_VR_SCREEN_DISTANCE  from the head to the screen's centre   (default 3.0, 0.5 to 20)
//   LEONIDA_VR_SCREEN_WIDTH     the screen's width; the height follows the frame's aspect ratio (default 3.2, 0.25 to 40)
//   LEONIDA_VR_SCREEN_HEIGHT    the centre's height above (or below, negative) the head (default 0, -10 to 10)
// A value that is not a number in range is ignored with a warning, so a typo cannot put the screen out of sight.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace leonida::vr {

enum class Mode { off, openxr, unknown };

// Unset, empty, "0" and "off" are off; "openxr" (any case) is OpenXR; anything else is unknown, which the caller
// reports and treats as off.
inline Mode parse_mode(const char* value) {
    if (!value || !*value) return Mode::off;
    std::string v(value);
    for (char& c : v) c = char((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
    if (v == "0" || v == "off") return Mode::off;
    if (v == "openxr") return Mode::openxr;
    return Mode::unknown;
}

// The value of an environment variable, or nothing when it is unset.
using EnvLookup = std::function<std::optional<std::string>(const char* name)>;

inline std::optional<std::string> process_env(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value) return std::nullopt;
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv(name);
    if (!value) return std::nullopt;
    return std::string(value);
#endif
}

inline Mode mode_from(const EnvLookup& lookup) {
    const std::optional<std::string> value = lookup("LEONIDA_VR");
    return parse_mode(value ? value->c_str() : nullptr);
}

struct CinemaConfig {
    float distance_m = 3.0f;
    float width_m = 3.2f;
    float height_m = 0.0f;
};

struct CinemaSettings {
    CinemaConfig config;
    std::vector<std::string> warnings;  // one line per ignored value
};

namespace detail {
// A whole-string decimal number (strtof in the C locale the runtime keeps); nothing for "", "3m", "nan" or "inf".
inline std::optional<float> parse_float(const std::string& text) {
    if (text.empty()) return std::nullopt;
    char* end = nullptr;
    const float v = std::strtof(text.c_str(), &end);
    if (end != text.c_str() + text.size() || !std::isfinite(v)) return std::nullopt;
    return v;
}
}  // namespace detail

inline CinemaSettings cinema_settings_from(const EnvLookup& lookup) {
    CinemaSettings out;
    struct Field {
        const char* name;
        float* value;
        float low, high;
    };
    const Field fields[] = {
        {"LEONIDA_VR_SCREEN_DISTANCE", &out.config.distance_m, 0.5f, 20.0f},
        {"LEONIDA_VR_SCREEN_WIDTH", &out.config.width_m, 0.25f, 40.0f},
        {"LEONIDA_VR_SCREEN_HEIGHT", &out.config.height_m, -10.0f, 10.0f},
    };
    for (const Field& f : fields) {
        const std::optional<std::string> text = lookup(f.name);
        if (!text || text->empty()) continue;
        const std::optional<float> v = detail::parse_float(*text);
        if (v && *v >= f.low && *v <= f.high) {
            *f.value = *v;
            continue;
        }
        char line[200];
        std::snprintf(line, sizeof(line), "%s=%s is not a number from %g to %g; using %g", f.name, text->c_str(),
                      double(f.low), double(f.high), double(*f.value));
        out.warnings.emplace_back(line);
    }
    return out;
}

struct Vec3 {
    float x = 0, y = 0, z = 0;
};
struct Quat {
    float x = 0, y = 0, z = 0, w = 1;
};
struct Pose {
    Quat orientation;
    Vec3 position;
};

// The screen as an XrCompositionLayerQuad wants it: the pose of its centre and its size in metres.
struct QuadPlacement {
    Pose pose;
    float width_m = 0;
    float height_m = 0;
};

// The screen straight ahead of the starting head position (-Z), raised by height_m, facing the viewer: a quad's
// visible side is its +Z side, so the identity orientation faces back toward the origin. Its height keeps the
// frame's aspect ratio (pixels are square). An empty frame gives an empty quad.
inline QuadPlacement place_screen(const CinemaConfig& config, std::uint32_t pixel_width, std::uint32_t pixel_height) {
    QuadPlacement q;
    q.pose.position = {0.0f, config.height_m, -config.distance_m};
    if (!pixel_width || !pixel_height) return q;
    q.width_m = config.width_m;
    q.height_m = config.width_m * float(pixel_height) / float(pixel_width);
    return q;
}

// The horizontal angle the screen fills, in degrees, seen from the origin (for the log line).
inline float horizontal_fov_degrees(const CinemaConfig& config) {
    return float(2.0 * std::atan(double(config.width_m) / (2.0 * double(config.distance_m))) * 180.0 / 3.14159265358979323846);
}

// Rotates `v` by the unit quaternion `q` (for tests of a pose's facing).
inline Vec3 rotate(const Quat& q, const Vec3& v) {
    // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    const float tx = 2 * (q.y * v.z - q.z * v.y), ty = 2 * (q.z * v.x - q.x * v.z), tz = 2 * (q.x * v.y - q.y * v.x);
    return {v.x + q.w * tx + (q.y * tz - q.z * ty), v.y + q.w * ty + (q.z * tx - q.x * tz),
            v.z + q.w * tz + (q.x * ty - q.y * tx)};
}

// The OpenXR instance extensions VR mode looks for. Only XR_KHR_vulkan_enable2 is enabled; the others are listed so
// that a user's log says what their headset offers for later work (eye-tracked foveated rendering: an eye-gaze
// source, foveation on the runtime's swapchains or a quad-view configuration with VK_KHR_fragment_shading_rate).
struct ExtensionSupport {
    bool vulkan_enable2 = false;
    bool eye_gaze_interaction = false;     // XR_EXT_eye_gaze_interaction
    bool fb_foveation = false;             // XR_FB_foveation (+ XR_FB_foveation_configuration)
    bool fb_foveation_vulkan = false;      // XR_FB_foveation_vulkan (a fragment density map per swapchain image)
    bool meta_foveation_eye_tracked = false;  // XR_META_foveation_eye_tracked
    bool varjo_quad_views = false;         // XR_VARJO_quad_views
    bool varjo_foveated_rendering = false;  // XR_VARJO_foveated_rendering
};

inline ExtensionSupport extension_support(const std::vector<std::string>& names) {
    ExtensionSupport s;
    for (const std::string& n : names) {
        if (n == "XR_KHR_vulkan_enable2") s.vulkan_enable2 = true;
        else if (n == "XR_EXT_eye_gaze_interaction") s.eye_gaze_interaction = true;
        else if (n == "XR_FB_foveation") s.fb_foveation = true;
        else if (n == "XR_FB_foveation_vulkan") s.fb_foveation_vulkan = true;
        else if (n == "XR_META_foveation_eye_tracked") s.meta_foveation_eye_tracked = true;
        else if (n == "XR_VARJO_quad_views") s.varjo_quad_views = true;
        else if (n == "XR_VARJO_foveated_rendering") s.varjo_foveated_rendering = true;
    }
    return s;
}

// One short line: "vulkan_enable2 yes, eye_gaze no, FB_foveation no, ...".
inline std::string extension_summary(const ExtensionSupport& s) {
    const std::pair<const char*, bool> items[] = {
        {"vulkan_enable2", s.vulkan_enable2},
        {"EXT_eye_gaze_interaction", s.eye_gaze_interaction},
        {"FB_foveation", s.fb_foveation},
        {"FB_foveation_vulkan", s.fb_foveation_vulkan},
        {"META_foveation_eye_tracked", s.meta_foveation_eye_tracked},
        {"VARJO_quad_views", s.varjo_quad_views},
        {"VARJO_foveated_rendering", s.varjo_foveated_rendering},
    };
    std::string out;
    for (const auto& [name, on] : items) {
        if (!out.empty()) out += ", ";
        out += name;
        out += on ? " yes" : " no";
    }
    return out;
}

// VkFormat values (vulkan_core.h; openxr_session.cpp checks them against the header) for the choice below.
enum : std::int64_t {
    vk_format_r8g8b8a8_unorm = 37,
    vk_format_r8g8b8a8_srgb = 43,
    vk_format_b8g8r8a8_unorm = 44,
    vk_format_b8g8r8a8_srgb = 50,
};

// The headset swapchain format for a window swapchain in `window_format`, or 0 when none of `offered` fits. The
// frame is copied byte for byte (vkCmdCopyImage), and a window swapchain holds sRGB-encoded bytes whether its format
// is UNORM or sRGB, so the headset image is the sRGB format with the same channel order: the compositor then decodes
// it as sRGB. A blit into it would encode a UNORM source a second time. Other window formats (HDR10) give 0.
inline std::int64_t cinema_swapchain_format(std::int64_t window_format, const std::vector<std::int64_t>& offered) {
    std::int64_t wanted = 0;
    if (window_format == vk_format_b8g8r8a8_unorm || window_format == vk_format_b8g8r8a8_srgb)
        wanted = vk_format_b8g8r8a8_srgb;
    else if (window_format == vk_format_r8g8b8a8_unorm || window_format == vk_format_r8g8b8a8_srgb)
        wanted = vk_format_r8g8b8a8_srgb;
    for (std::int64_t f : offered)
        if (wanted && f == wanted) return f;
    return 0;
}

}  // namespace leonida::vr
