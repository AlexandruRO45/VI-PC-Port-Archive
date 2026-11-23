#pragma once

// D3D12 event payloads (SetMarker / BeginEvent on command lists and queues) decoded into a label name and colour
// for VK_EXT_debug_utils. No Windows or Vulkan dependency.
//
// The `metadata` argument says how `data` is encoded. The values and layouts are those of Microsoft's
// WinPixEventRuntime headers (github.com/microsoft/PixEvents, include/pix3_win.h, PIXEventsCommon.h,
// PIXEvents.h, PIXEventsLegacy.h):
//   0        WINPIX_EVENT_UNICODE_VERSION: a UTF-16 string of `size` bytes (pix.h in the Windows SDK)
//   1        WINPIX_EVENT_ANSI_VERSION: an 8-bit string of `size` bytes
//   2        WINPIX_EVENT_PIX3BLOB_VERSION (D3D12_EVENT_METADATA), the legacy blob, in 64-bit words:
//              [0] event info, type in bits 10..19
//              [1] colour
//              [2] string info, bit 54 set for an 8-bit string
//              [3..] the format string packed into the words, ending with a zero character
//   6345127  WINPIX_EVENT_PIX3BLOB_V2, the 2023 blob:
//              [0] event info: size in words bits 0..6, type bits 7..11, metadata bits 12..19
//                  (0x01 on a context, 0x02 8-bit string, 0xF0 has a colour)
//              [1] colour
//              [2] the context pointer, on PC builds only (PIX_XBOX builds leave it out)
//              then the format string packed into the words, ending with a zero character
// The format string is used as it is: printf arguments that follow it are not applied. vkd3d-proton
// (libs/vkd3d/command.c, decode_pix_blob) reads the legacy blob the same way.

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

namespace leonida {

struct DecodedLabel {
    std::string name;                                   // UTF-8
    std::array<float, 4> color{1.0f, 1.0f, 1.0f, 1.0f}; // RGBA; white when the payload has no colour
};

namespace pix {

inline constexpr std::uint32_t unicode_version = 0;
inline constexpr std::uint32_t ansi_version = 1;
inline constexpr std::uint32_t blob_version = 2;
inline constexpr std::uint32_t blob_v2 = 6345127;

// Legacy blob event types (PIXEventsLegacy.h).
inline constexpr std::uint64_t legacy_begin_varargs = 0x001, legacy_begin_noargs = 0x002;
inline constexpr std::uint64_t legacy_marker_varargs = 0x007, legacy_marker_noargs = 0x008;
inline constexpr std::uint64_t legacy_on_context = 0x010;  // the _OnContext_ variants: the same value | 0x10
inline constexpr std::uint64_t legacy_type_mask = 0x00000000000FFC00;
inline constexpr unsigned legacy_type_shift = 10;
inline constexpr std::uint64_t string_is_ansi = 0x0040000000000000;  // string info word, bit 54

// V2 event info (PIXEventsCommon.h).
inline constexpr std::uint64_t v2_type_mask = 0x0000000000000F80;
inline constexpr unsigned v2_type_shift = 7;
inline constexpr std::uint64_t v2_metadata_mask = 0x00000000000FF000;
inline constexpr unsigned v2_metadata_shift = 12;
inline constexpr std::uint64_t v2_begin = 0x01, v2_marker = 0x02;
inline constexpr std::uint64_t v2_on_context = 0x01, v2_ansi = 0x02, v2_has_color = 0xF0;

} // namespace pix

namespace detail {

inline void append_utf8(std::string& out, std::uint32_t c) {
    if (c < 0x80) {
        out += char(c);
    } else if (c < 0x800) {
        out += char(0xC0 | (c >> 6));
        out += char(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += char(0xE0 | (c >> 12));
        out += char(0x80 | ((c >> 6) & 0x3F));
        out += char(0x80 | (c & 0x3F));
    } else {
        out += char(0xF0 | (c >> 18));
        out += char(0x80 | ((c >> 12) & 0x3F));
        out += char(0x80 | ((c >> 6) & 0x3F));
        out += char(0x80 | (c & 0x3F));
    }
}

// Up to the first zero character or the end of the bytes.
inline std::string narrow(const unsigned char* bytes, std::size_t size) {
    std::size_t n = 0;
    while (n < size && bytes[n]) ++n;
    return std::string(reinterpret_cast<const char*>(bytes), n);
}

// UTF-16LE to UTF-8; an unpaired surrogate becomes U+FFFD.
inline std::string widen(const unsigned char* bytes, std::size_t size) {
    std::string out;
    const std::size_t units = size / 2;
    const auto unit = [&](std::size_t i) { return std::uint32_t(bytes[2 * i] | (bytes[2 * i + 1] << 8)); };
    for (std::size_t i = 0; i < units; ++i) {
        std::uint32_t c = unit(i);
        if (!c) break;
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < units && unit(i + 1) >= 0xDC00 && unit(i + 1) < 0xE000) {
            c = 0x10000 + ((c - 0xD800) << 10) + (unit(i + 1) - 0xDC00);
            ++i;
        } else if (c >= 0xD800 && c < 0xE000) {
            c = 0xFFFD;
        }
        append_utf8(out, c);
    }
    return out;
}

inline std::uint64_t word(const unsigned char* bytes, std::size_t index) {
    std::uint64_t w;
    std::memcpy(&w, bytes + index * 8, 8);
    return w;
}

// PIX_COLOR is 0xAARRGGBB in the low 32 bits. A colour without alpha (an index colour, or none) is drawn opaque.
inline std::array<float, 4> color(std::uint64_t value) {
    if (!(value & 0xFFFFFFFF)) return {1.0f, 1.0f, 1.0f, 1.0f};
    const auto channel = [&](unsigned shift) { return float((value >> shift) & 0xFF) / 255.0f; };
    const float alpha = (value >> 24) & 0xFF ? channel(24) : 1.0f;
    return {channel(16), channel(8), channel(0), alpha};
}

} // namespace detail

// False when the payload is not a SetMarker/BeginEvent this decoder knows; `out` is then left unchanged.
// `context` is the interface pointer the event was recorded on: in a V2 blob, the word after the colour is
// skipped when it equals it (PC builds write it there; Xbox builds do not).
inline bool decode_event(std::uint32_t metadata, const void* data, std::uint32_t size, std::uint64_t context,
                         DecodedLabel& out) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    if (!bytes && size) return false;
    DecodedLabel label;
    switch (metadata) {
    case pix::unicode_version:
        label.name = detail::widen(bytes, size);
        break;
    case pix::ansi_version:
        label.name = detail::narrow(bytes, size);
        break;
    case pix::blob_version: {
        const std::size_t words = size / 8;
        if (words < 3) return false;
        const std::uint64_t type = ((detail::word(bytes, 0) & pix::legacy_type_mask) >> pix::legacy_type_shift) &
                                   ~pix::legacy_on_context;
        if (type != pix::legacy_begin_varargs && type != pix::legacy_begin_noargs &&
            type != pix::legacy_marker_varargs && type != pix::legacy_marker_noargs)
            return false;
        label.color = detail::color(detail::word(bytes, 1));
        const bool ansi = detail::word(bytes, 2) & pix::string_is_ansi;
        const std::size_t offset = 3 * 8;
        label.name = ansi ? detail::narrow(bytes + offset, size - offset) : detail::widen(bytes + offset, size - offset);
        break;
    }
    case pix::blob_v2: {
        const std::size_t words = size / 8;
        if (words < 2) return false;
        const std::uint64_t info = detail::word(bytes, 0);
        const std::uint64_t type = (info & pix::v2_type_mask) >> pix::v2_type_shift;
        const std::uint64_t flags = (info & pix::v2_metadata_mask) >> pix::v2_metadata_shift;
        if (type != pix::v2_begin && type != pix::v2_marker) return false;
        label.color = detail::color(detail::word(bytes, 1));
        std::size_t first = 2;
        if ((flags & pix::v2_on_context) && words > 2 && context && detail::word(bytes, 2) == context) first = 3;
        const std::size_t offset = first * 8;
        if (offset > size) return false;
        label.name = (flags & pix::v2_ansi) ? detail::narrow(bytes + offset, size - offset)
                                            : detail::widen(bytes + offset, size - offset);
        break;
    }
    default:
        return false;
    }
    out = std::move(label);
    return true;
}

} // namespace leonida
