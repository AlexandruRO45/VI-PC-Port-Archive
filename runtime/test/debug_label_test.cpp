// D3D12 marker payloads (src/debug_label.h). The PIX blobs are built by hand the way Microsoft's WinPixEventRuntime
// headers write them (PIXEventsLegacy.h for the legacy blob, PIXEvents.h with PIX_USE_GPU_MARKERS_V2 for the V2
// blob). No GPU, Windows or game data.
#include "../src/debug_label.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

// Characters packed into 64-bit words, ending with a zero character, as PIXCopyEventStringArgumentSlow does.
void pack(std::vector<std::uint64_t>& words, const std::string& text) {
    std::vector<unsigned char> bytes(text.begin(), text.end());
    bytes.push_back(0);
    while (bytes.size() % 8) bytes.push_back(0);
    for (std::size_t i = 0; i < bytes.size(); i += 8) {
        std::uint64_t w;
        std::memcpy(&w, bytes.data() + i, 8);
        words.push_back(w);
    }
}

void pack(std::vector<std::uint64_t>& words, const std::u16string& text) {
    std::vector<char16_t> units(text.begin(), text.end());
    units.push_back(0);
    while (units.size() % 4) units.push_back(0);
    for (std::size_t i = 0; i < units.size(); i += 4) {
        std::uint64_t w = 0;
        for (int j = 0; j < 4; ++j) w |= std::uint64_t(std::uint16_t(units[i + j])) << (16 * j);
        words.push_back(w);
    }
}

constexpr std::uint64_t legacy_info(std::uint64_t type) { return type << 10; }
constexpr std::uint64_t legacy_string_info(bool ansi) { return (8ull << 55) | (std::uint64_t(ansi) << 54); }
constexpr std::uint64_t v2_info(std::uint64_t type, std::uint64_t metadata, std::uint64_t size) {
    return (metadata << 12) | (type << 7) | size;
}

bool decode(std::uint32_t metadata, const std::vector<std::uint64_t>& words, std::uint64_t context, DecodedLabel& out) {
    return decode_event(metadata, words.data(), std::uint32_t(words.size() * 8), context, out);
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

}  // namespace

int main() {
    DecodedLabel label;

    // WINPIX_EVENT_ANSI_VERSION: the bytes, up to a zero or the size.
    EXPECT(decode_event(pix::ansi_version, "Frame", 6, 0, label) && label.name == "Frame");
    EXPECT(decode_event(pix::ansi_version, "Frame graph", 5, 0, label) && label.name == "Frame");
    EXPECT(label.color == (std::array<float, 4>{1, 1, 1, 1}));
    EXPECT(decode_event(pix::ansi_version, nullptr, 0, 0, label) && label.name.empty());
    EXPECT(!decode_event(pix::ansi_version, nullptr, 4, 0, label));

    // WINPIX_EVENT_UNICODE_VERSION: UTF-16 to UTF-8, surrogate pairs joined, a lone surrogate replaced.
    // Code units spelled out: the source is not read as UTF-8 on every compiler.
    const char16_t wide[] = {u'O', u'm', u'b', u'r', u'e', u' ', 0x00E9, u' ', 0xD83D, 0xDE00, 0};  // "Ombre é 😀"
    EXPECT(decode_event(pix::unicode_version, wide, sizeof(wide), 0, label));
    EXPECT(label.name == "Ombre \xc3\xa9 \xf0\x9f\x98\x80");
    const char16_t lone[] = {u'a', char16_t(0xD800), u'b', 0};
    EXPECT(decode_event(pix::unicode_version, lone, sizeof(lone), 0, label) && label.name == "a\xef\xbf\xbd" "b");

    // Legacy blob (D3D12_EVENT_METADATA = 2): info, colour, string info, string.
    {
        std::vector<std::uint64_t> w{legacy_info(pix::legacy_begin_noargs), 0xFF336699, legacy_string_info(true)};
        pack(w, std::string("Shadow pass"));
        w.push_back(0);
        EXPECT(decode(pix::blob_version, w, 0, label) && label.name == "Shadow pass");
        EXPECT(near(label.color[0], 0x33 / 255.0f) && near(label.color[1], 0x66 / 255.0f) &&
               near(label.color[2], 0x99 / 255.0f) && near(label.color[3], 1.0f));
    }
    {
        // A wide format string with printf arguments after it: the format string is kept as it is.
        std::vector<std::uint64_t> w{legacy_info(pix::legacy_marker_varargs), 0, legacy_string_info(false)};
        pack(w, std::u16string(u"Draw %u"));
        w.push_back(42);
        w.push_back(0);
        EXPECT(decode(pix::blob_version, w, 0, label) && label.name == "Draw %u");
        EXPECT(label.color == (std::array<float, 4>{1, 1, 1, 1}));  // no colour: white
    }
    for (const std::uint64_t type : {pix::legacy_begin_varargs | pix::legacy_on_context,
                                     pix::legacy_begin_noargs | pix::legacy_on_context,
                                     pix::legacy_marker_varargs | pix::legacy_on_context,
                                     pix::legacy_marker_noargs | pix::legacy_on_context, pix::legacy_marker_noargs}) {
        std::vector<std::uint64_t> w{legacy_info(type), 0xFF00FF00, legacy_string_info(true)};
        pack(w, std::string("x"));
        EXPECT(decode(pix::blob_version, w, 0, label) && label.name == "x");
    }
    {
        std::vector<std::uint64_t> w{legacy_info(0x000 /* EndEvent */), 0, legacy_string_info(true)};
        pack(w, std::string("end"));
        label.name = "kept";
        EXPECT(!decode(pix::blob_version, w, 0, label) && label.name == "kept");  // left unchanged
        EXPECT(!decode(pix::blob_version, {legacy_info(pix::legacy_begin_noargs), 0}, 0, label));  // too short
    }

    // V2 blob (WINPIX_EVENT_PIX3BLOB_V2): info, colour, the context on PC, string.
    constexpr std::uint64_t context = 0x00000123456789A0;
    {
        std::vector<std::uint64_t> w{0, 0xFFFF0000, context};
        pack(w, std::string("Lights"));
        w.push_back(0);
        w[0] = v2_info(pix::v2_begin, pix::v2_on_context | pix::v2_ansi | pix::v2_has_color, w.size());
        EXPECT(decode(pix::blob_v2, w, context, label) && label.name == "Lights");
        EXPECT(label.color == (std::array<float, 4>{1, 0, 0, 1}));
    }
    {
        // An Xbox build writes no context word: the word after the colour is already the string.
        std::vector<std::uint64_t> w{0, 0xFF0000FF};
        pack(w, std::u16string(u"Post"));
        w.push_back(0);
        w[0] = v2_info(pix::v2_marker, pix::v2_on_context | pix::v2_has_color, w.size());
        EXPECT(decode(pix::blob_v2, w, context, label) && label.name == "Post");
        EXPECT(label.color == (std::array<float, 4>{0, 0, 1, 1}));
    }
    {
        std::vector<std::uint64_t> w{v2_info(0 /* EndEvent */, pix::v2_ansi, 3), 0};
        pack(w, std::string("e"));
        EXPECT(!decode(pix::blob_v2, w, 0, label));
        EXPECT(!decode(pix::blob_v2, {v2_info(pix::v2_begin, 0, 1)}, 0, label));  // too short
    }
    EXPECT(!decode_event(3, "abc", 4, 0, label));  // a metadata value nobody defined

    std::printf("debug labels: ANSI, UTF-16, legacy and V2 PIX blobs decoded\n");
    return 0;
}
