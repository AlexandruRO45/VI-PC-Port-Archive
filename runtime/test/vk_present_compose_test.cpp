// PresentX plane composition (runtime/src/present_compose.comp through PresentComposer, vulkan_present_compose.h) on any
// Vulkan device, headless: no window, swapchain or Windows shim, so it runs where xbox_present_test (Windows only) does
// not, MoltenVK included. An HDR10 plane (ST.2084 Rec.2020 codes in an R16G16B16A16_UNORM texture, one colour per texel,
// point filtered) is composed onto an SDR (sRGB) output and the RGBA16F target is read back, for both HDR10 -> SDR tone
// maps (PresentPlanes::ToneMap, docs/d3d12_vulkan.md section 4.5):
//   1. The double-precision model's anchors. The BT.2390 EETF as Report ITU-R BT.2408-9 (03/2026) Annex 5 defines it
//      (LB = Lmin = 0, LW = 1000, Lmax = SDR white = 203 nits) gives the hand-calculated values below, which libplacebo's
//      bt2390 (knee offset 0.5, no black point) gives too; the shoulder gives the bytes docs section 4.5 lists (243, 250).
//   2. The EETF model's properties on a dense sweep: identity below the knee start KS, value and slope 1 continuous at KS
//      in the PQ domain, non-decreasing, LW and anything above it to exactly SDR white; a peak at or below SDR white is
//      the identity up to SDR white.
//   3. The GPU against the model: grey ramps from 0 to 10 000 nits for peaks 600, 1000 and 4000 at SDR white 203, and
//      SDR white 100 at peak 1000, for both tone maps, within half an 8-bit level of the encoded value.
//   4. Saturated highlights (Rec.709 colours whose largest channel is above SDR white), GPU against the model too. The
//      EETF, applied to max(R, G, B), keeps the largest channel at or below 1.0 and the channel ratios; the shoulder,
//      applied to luminance, leaves the largest channel above 1.0, so it is clipped and the ratios change.
// Exit code 77 (skipped) without a Vulkan device or an R16G16B16A16_UNORM sampled image. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_present.h"  // PresentPlanes; vulkan_present_compose.h (PresentComposer) with it

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition, ...) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "line %d: %s: ", __LINE__, #condition); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while (0)

using V3 = std::array<double, 3>;

// The model, written from the definitions (not from the shader). PQ: the exact rationals of SMPTE ST 2084 / ITU-R BT.2100,
// 1.0 = 10 000 nits. Colours are linear; SDR values have 1.0 = SDR white.
namespace model {

constexpr double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0, c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0,
                 c3 = 2392.0 / 128.0;

double pq_to_nits(double e) {
    const double ep = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / m2);
    return 10000.0 * std::pow(std::max(ep - c1, 0.0) / (c2 - c3 * ep), 1.0 / m1);
}

double nits_to_pq(double nits) {
    const double y = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), m1);
    return std::pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

// Rec.709 <-> Rec.2020 as the presenter converts (docs section 4.5: computed from the primaries; ITU-R BT.2087).
V3 bt709_to_bt2020(const V3& v) {
    return {0.627404 * v[0] + 0.329283 * v[1] + 0.043313 * v[2], 0.069097 * v[0] + 0.919540 * v[1] + 0.011362 * v[2],
            0.016391 * v[0] + 0.088013 * v[1] + 0.895595 * v[2]};
}

V3 bt2020_to_bt709(const V3& v) {
    return {1.660491 * v[0] - 0.587641 * v[1] - 0.072850 * v[2], -0.124550 * v[0] + 1.132900 * v[1] - 0.008349 * v[2],
            -0.018151 * v[0] - 0.100579 * v[1] + 1.118730 * v[2]};
}

double srgb_encode(double c) {
    c = std::clamp(c, 0.0, 1.0);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

double srgb_decode(double e) {
    e = std::max(e, 0.0);
    return e <= 0.04045 ? e / 12.92 : std::pow((e + 0.055) / 1.055, 2.4);
}

// The shoulder (docs section 4.5): identity on luminance up to the knee at 0.8 of SDR white, extended Reinhard (Reinhard
// et al. 2002, eq. 4) on the excess, the content peak Lw = peak / white (at least 1) mapped to 1.0; colour scaled by Yd / Y.
V3 shoulder(const V3& v, double peak, double white) {
    const double knee = 0.8, y = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    if (y <= knee) return v;
    const double t = 1.0 - knee, w = (std::max(peak / white, 1.0) - knee) / t, u = (y - knee) / t;
    const double s = (knee + t * u * (1.0 + u / (w * w)) / (1.0 + u)) / y;
    return {v[0] * s, v[1] * s, v[2] * s};
}

// Report ITU-R BT.2408-9 (03/2026) Annex 5, steps 3 and 4 on the normalized PQ value E1 (0..1): E2 = E1 below KS, else the
// Hermite spline P[E1] with T = (E1 - KS) / (1 - KS).
double eetf_e2(double e1, double ks, double max_lum) {
    if (e1 < ks || ks >= 1.0) return e1;
    const double t = (e1 - ks) / (1.0 - ks), t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ks + (t3 - 2 * t2 + t) * (1 - ks) + (-2 * t3 + 3 * t2) * max_lum;
}

struct Eetf {
    double pw, max_lum, ks;  // PQ^-1(LW); maxLum and KS of steps 1 and 2
};

// LB = 0 and Lmin = 0, so b = minLum = 0 and step 3's black lift b (1 - E2)^4 is nothing; LW at least Lmax.
Eetf eetf_for(double peak, double white) {
    Eetf e{};
    e.pw = nits_to_pq(std::max(peak, white));
    e.max_lum = nits_to_pq(white) / e.pw;  // step 1
    e.ks = 1.5 * e.max_lum - 0.5;          // step 2
    return e;
}

// Nits in, nits out: step 1 (E1, clamped to the source range), steps 3 and 4, step 5 (E4 = E2 PQ^-1(LW)).
double eetf_nits(double nits, const Eetf& e) {
    return pq_to_nits(eetf_e2(std::min(nits_to_pq(nits) / e.pw, 1.0), e.ks, e.max_lum) * e.pw);
}

// The annex's maxRGB option: M = max(R, G, B) through the EETF, the colour scaled by M2 / M.
V3 bt2390(const V3& v, double peak, double white) {
    const double m = std::max({v[0], v[1], v[2]});
    if (!(m > 0.0)) return v;
    const double s = eetf_nits(m * white, eetf_for(peak, white)) / white / m;
    return {v[0] * s, v[1] * s, v[2] * s};
}

// What the pass writes for one HDR10 texel of 16-bit PQ codes on an sRGB output (out_hdr = 0, opaque plane 0).
V3 compose(const std::array<std::uint16_t, 3>& codes, double peak, double white, bool eetf) {
    V3 nits;
    for (int i = 0; i < 3; ++i) nits[i] = pq_to_nits(codes[i] / 65535.0);
    V3 v = bt2020_to_bt709(nits);
    for (double& c : v) c = std::max(c / white, 0.0);
    v = eetf ? bt2390(v, peak, white) : shoulder(v, peak, white);
    return {srgb_encode(v[0]), srgb_encode(v[1]), srgb_encode(v[2])};
}

}  // namespace model

double half_to_double(std::uint16_t h) {
    const int exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    double v = exponent == 0 ? std::ldexp(double(mantissa), -24)
             : exponent == 31 ? (mantissa ? NAN : INFINITY)
                              : std::ldexp(double(mantissa | 0x400), exponent - 25);
    return (h & 0x8000) ? -v : v;
}

std::array<std::uint16_t, 3> encode_hdr10(const V3& rec709_nits) {
    const V3 v = model::bt709_to_bt2020(rec709_nits);
    std::array<std::uint16_t, 3> codes{};
    for (int i = 0; i < 3; ++i)
        codes[i] = std::uint16_t(std::lround(model::nits_to_pq(std::max(v[i], 0.0)) * 65535.0));
    return codes;
}

// 1 and 2: the model alone.
void check_model() {
    {
        const model::Eetf e = model::eetf_for(1000.0, 203.0);
        CHECK(std::abs(e.ks - 0.658555373) < 1e-8, "KS %.9f", e.ks);
        CHECK(std::abs(model::pq_to_nits(e.ks * e.pw) - 87.836267) < 1e-5, "knee start %.6f nits",
              model::pq_to_nits(e.ks * e.pw));
        const double nits[] = {50, 150, 203, 300, 500, 1000, 4000};
        const double want[] = {0.2463054, 0.6661851, 0.7834575, 0.8989678, 0.9795863, 1.0, 1.0};
        for (int i = 0; i < 7; ++i) {
            const double got = model::eetf_nits(nits[i], e) / 203.0;
            CHECK(std::abs(got - want[i]) < 1e-6, "EETF %.0f nits: %.7f, want %.7f", nits[i], got, want[i]);
        }
        const V3 white = model::shoulder({1.0, 1.0, 1.0}, 1000.0, 203.0), brighter = model::shoulder(
                                                                             {300.0 / 203, 300.0 / 203, 300.0 / 203}, 1000.0, 203.0);
        CHECK(std::lround(model::srgb_encode(white[0]) * 255) == 243, "shoulder 203 nits: byte %ld",
              std::lround(model::srgb_encode(white[0]) * 255));
        CHECK(std::lround(model::srgb_encode(brighter[0]) * 255) == 250, "shoulder 300 nits: byte %ld",
              std::lround(model::srgb_encode(brighter[0]) * 255));
    }
    const double cases[][2] = {{1000, 203}, {600, 203}, {4000, 203}, {10000, 203}, {1000, 100}, {150, 203}};
    for (const auto& c : cases) {
        const double peak = c[0], white = c[1];
        const model::Eetf e = model::eetf_for(peak, white);
        const double lw = std::max(peak, white), knee = model::pq_to_nits(std::min(e.ks, 1.0) * e.pw);
        CHECK(std::abs(model::eetf_nits(lw, e) - white) < 1e-9 * white, "peak %.0f white %.0f: LW -> %.9f", peak, white,
              model::eetf_nits(lw, e));
        CHECK(std::abs(model::eetf_nits(2 * lw, e) - white) < 1e-9 * white, "peak %.0f white %.0f: above LW", peak, white);
        if (e.ks < 1.0) {  // the spline joins the 1:1 part with value KS and slope 1 (PQ domain)
            const double h = 1e-7;
            CHECK(std::abs(model::eetf_e2(e.ks, e.ks, e.max_lum) - e.ks) < 1e-12, "peak %.0f: P[KS] != KS", peak);
            const double slope = (model::eetf_e2(e.ks + h, e.ks, e.max_lum) - e.ks) / h;
            CHECK(std::abs(slope - 1.0) < 1e-5, "peak %.0f white %.0f: slope at KS %.8f", peak, white, slope);
            const double end = (model::eetf_e2(1.0, e.ks, e.max_lum) - model::eetf_e2(1.0 - h, e.ks, e.max_lum)) / h;
            CHECK(std::abs(end) < 1e-5, "peak %.0f white %.0f: slope at 1 %.8f", peak, white, end);
        } else {
            CHECK(peak <= white, "KS >= 1 with a peak above SDR white");
        }
        double previous = 0.0;
        bool monotonic = true, identity = true;
        for (int i = 0; i <= 20000; ++i) {
            const double nits = 2 * lw * i / 20000.0, out = model::eetf_nits(nits, e);
            monotonic = monotonic && out >= previous - 1e-9 * white;
            if (nits < knee) identity = identity && std::abs(out - nits) <= 1e-9 * std::max(nits, 1.0);
            previous = out;
        }
        CHECK(monotonic, "peak %.0f white %.0f: not monotonic", peak, white);
        CHECK(identity, "peak %.0f white %.0f: not the identity below %.3f nits", peak, white, knee);
    }
}

struct Gpu {
    vktest::Device& d;
    leonida::PresentComposer& composer;
    VkImage source;
    VkImageView view;
    VkBuffer upload, readback;
    std::uint16_t* up;
    const std::uint16_t* down;
    static constexpr std::uint32_t width = 64;

    // Composes `colours` (HDR10 codes, at most `width`) onto an SDR output; the encoded RGB of each, read back.
    bool run(const std::vector<std::array<std::uint16_t, 3>>& colours, double peak, double white,
             leonida::PresentPlanes::ToneMap tone_map, std::vector<V3>& out) {
        const std::uint32_t n = std::uint32_t(colours.size());
        bool changed = false;
        if (n == 0 || n > width || !composer.ensure_target({n, 1}, changed)) return false;
        std::memset(up, 0, width * 8);
        for (std::uint32_t i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) up[i * 4 + c] = colours[i][c];
            up[i * 4 + 3] = 0xffff;
        }
        leonida::PresentComposePush push{};
        const float fn = float(n);
        const float rect[4] = {0.0f, 0.0f, fn, 1.0f};
        std::memcpy(push.dest0, rect, sizeof rect);
        std::memcpy(push.dest1, rect, sizeof rect);
        std::memcpy(push.src0, rect, sizeof rect);
        std::memcpy(push.src1, rect, sizeof rect);
        push.flags0 = leonida::compose_flag::point | leonida::compose_flag::hdr10;
        push.plane_count = 1;
        push.out_hdr = 0;
        push.sdr_white = float(white);
        push.width = n;
        push.height = 1;
        push.sdr_eotf = std::uint32_t(leonida::PresentPlanes::SdrEotf::srgb);
        push.hdr_peak = float(peak);
        push.tone_map = std::uint32_t(tone_map);
        const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const bool ok = d.run([&](VkCommandBuffer cb) {
            VkImageMemoryBarrier to_copy{};
            to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_copy.srcQueueFamilyIndex = to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_copy.image = source;
            to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                    &to_copy);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {width, 1, 1};
            d.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(cb, upload, source,
                                                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            VkImageMemoryBarrier to_read = to_copy;
            to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                    &to_read);
            composer.record(cb, 0, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, view,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, push);
            d.read_back(cb, composer.target(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, n, 1, 1,
                        readback, 0);
            return true;
        });
        if (!ok) return false;
        out.resize(n);
        for (std::uint32_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) out[i][c] = half_to_double(down[i * 4 + c]);
        return true;
    }
};

const char* name_of(leonida::PresentPlanes::ToneMap t) {
    return t == leonida::PresentPlanes::ToneMap::bt2390 ? "BT.2390 EETF" : "shoulder";
}

// 3: grey ramps, GPU against the model, both tone maps.
bool check_ramps(Gpu& gpu) {
    const double ramp[] = {0, 0.5, 1, 5, 10, 25, 50, 80, 87, 88, 100, 120, 150, 162, 175, 203, 250, 300, 400, 500,
                           600, 750, 1000, 1500, 2000, 4000, 10000};
    const double tolerance = 0.5 / 255.0;
    const double cases[][2] = {{1000, 203}, {600, 203}, {4000, 203}, {1000, 100}};
    for (const auto tone_map : {leonida::PresentPlanes::ToneMap::shoulder, leonida::PresentPlanes::ToneMap::bt2390}) {
        for (const auto& c : cases) {
            const double peak = c[0], white = c[1];
            std::vector<std::array<std::uint16_t, 3>> colours;
            for (double nits : ramp) colours.push_back(encode_hdr10({nits, nits, nits}));
            std::vector<V3> got;
            if (!gpu.run(colours, peak, white, tone_map, got)) return false;
            double worst = 0.0;
            for (std::size_t i = 0; i < colours.size(); ++i) {
                const V3 want = model::compose(colours[i], peak, white,
                                               tone_map == leonida::PresentPlanes::ToneMap::bt2390);
                for (int ch = 0; ch < 3; ++ch) {
                    const double error = std::abs(got[i][ch] - want[ch]);
                    worst = std::max(worst, error);
                    CHECK(error <= tolerance, "%s, peak %.0f, white %.0f, %.1f nits, channel %d: %.6f, model %.6f",
                          name_of(tone_map), peak, white, ramp[i], ch, got[i][ch], want[ch]);
                }
            }
            std::printf("compose %s, HDR10 grey ramp (%zu steps, 0..10000 nits) onto SDR, peak %.0f, white %.0f: "
                        "worst %.5f (%.2f of an 8-bit level)\n",
                        name_of(tone_map), colours.size(), peak, white, worst, worst * 255.0);
        }
    }
    return true;
}

// 4: saturated highlights, GPU against the model, and what each tone map does to them.
bool check_highlights(Gpu& gpu) {
    const double peak = 1000.0, white = 203.0, tolerance = 0.5 / 255.0;
    const V3 hues[] = {{1.0, 0.5, 0.25}, {0.25, 1.0, 0.5}, {0.5, 0.25, 1.0}, {1.0, 1.0, 0.2}, {1.0, 0.1, 0.1}};
    const double scales[] = {1.5, 2.0, 3.0, 4.0};  // the largest channel, in units of SDR white
    std::vector<std::array<std::uint16_t, 3>> colours;
    std::vector<V3> linear;
    for (const V3& hue : hues)
        for (double s : scales) {
            const V3 v{hue[0] * s, hue[1] * s, hue[2] * s};
            colours.push_back(encode_hdr10({v[0] * white, v[1] * white, v[2] * white}));
            linear.push_back(v);
        }
    for (const auto tone_map : {leonida::PresentPlanes::ToneMap::shoulder, leonida::PresentPlanes::ToneMap::bt2390}) {
        const bool eetf = tone_map == leonida::PresentPlanes::ToneMap::bt2390;
        std::vector<V3> got;
        if (!gpu.run(colours, peak, white, tone_map, got)) return false;
        double worst = 0.0, worst_ratio = 0.0;
        int clipped = 0;
        for (std::size_t i = 0; i < colours.size(); ++i) {
            const V3 want = model::compose(colours[i], peak, white, eetf);
            for (int ch = 0; ch < 3; ++ch) {
                const double error = std::abs(got[i][ch] - want[ch]);
                worst = std::max(worst, error);
                CHECK(error <= tolerance, "%s highlight %zu, channel %d: %.6f, model %.6f", name_of(tone_map), i, ch,
                      got[i][ch], want[ch]);
            }
            // Back to linear: the largest channel and the ratios of the other two to it.
            const V3 lin{model::srgb_decode(got[i][0]), model::srgb_decode(got[i][1]), model::srgb_decode(got[i][2])};
            const int top = int(std::max_element(linear[i].begin(), linear[i].end()) - linear[i].begin());
            const V3 unclipped = eetf ? model::bt2390(linear[i], peak, white) : model::shoulder(linear[i], peak, white);
            if (unclipped[top] > 1.0 + 1e-9) ++clipped;
            for (int ch = 0; ch < 3; ++ch) {
                if (ch == top) continue;
                const double ratio_error = std::abs(lin[ch] / lin[top] - linear[i][ch] / linear[i][top]);
                if (eetf) {
                    worst_ratio = std::max(worst_ratio, ratio_error);
                    CHECK(ratio_error < 0.01, "EETF highlight %zu: channel %d / %d is %.4f, the source's %.4f", i, ch, top,
                          lin[ch] / lin[top], linear[i][ch] / linear[i][top]);
                }
            }
        }
        if (eetf) {
            CHECK(clipped == 0, "the EETF on maxRGB left %d highlights above 1.0", clipped);
            std::printf("compose %s, %zu saturated highlights (largest channel 1.5..4x SDR white), peak 1000, white 203: "
                        "worst %.5f; none clipped, channel ratios kept within %.4f\n",
                        name_of(tone_map), colours.size(), worst, worst_ratio);
        } else {
            // The documented difference: on luminance, a saturated colour's largest channel stays above 1.0.
            CHECK(clipped > 0, "the shoulder clipped no highlight");
            std::printf("compose %s, %zu saturated highlights, peak 1000, white 203: worst %.5f; %d of them clipped per "
                        "channel (largest channel above 1.0 after the luminance shoulder), as documented\n",
                        name_of(tone_map), colours.size(), worst, clipped);
        }
    }
    return true;
}

}  // namespace

int main() {
    check_model();
    std::printf("model: BT.2408-9 Annex 5 EETF anchors and properties, shoulder bytes 243 and 250: %s\n",
                failures ? "FAILED" : "ok");

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_present_compose_test: no Vulkan device, GPU part skipped\n");
        return failures ? 1 : 77;
    }
    const VkFormat format = VK_FORMAT_R16G16B16A16_UNORM;
    VkFormatProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(d.physical, format,
                                                                                                 &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        std::fprintf(stderr, "vk_present_compose_test: R16G16B16A16_UNORM can't be sampled here, GPU part skipped\n");
        return failures ? 1 : 77;
    }
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    leonida::VulkanContext context(d.device_proc, d.device, false);
    leonida::PresentComposer composer(context, memory);
    if (!composer.ready()) {
        std::fprintf(stderr, "vk_present_compose_test: the compose pipeline could not be created\n");
        return 1;
    }
    void* up = nullptr;
    void* down = nullptr;
    const VkBuffer upload = d.buffer(Gpu::width * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up);
    const VkBuffer readback = d.buffer(Gpu::width * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &down);
    const VkImage source = d.image(format, Gpu::width, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    if (!upload || !readback || !source) {
        std::fprintf(stderr, "vk_present_compose_test: resources could not be created\n");
        return 1;
    }
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = source;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &view_info, nullptr, &view) != VK_SUCCESS) {
        std::fprintf(stderr, "vk_present_compose_test: vkCreateImageView failed\n");
        return 1;
    }
    Gpu gpu{d, composer, source, view, upload, readback, static_cast<std::uint16_t*>(up),
            static_cast<const std::uint16_t*>(down)};
    const bool ran = check_ramps(gpu) && check_highlights(gpu);
    d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);
    d.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(d.device, view, nullptr);
    if (!ran) {
        std::fprintf(stderr, "vk_present_compose_test: a compose submission failed\n");
        return 1;
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "vk_present_compose_test: %u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("vk_present_compose_test: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
