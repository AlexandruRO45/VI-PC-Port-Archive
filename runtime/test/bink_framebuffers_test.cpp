// Optional Windows test: run the call sequence the retail exe uses to get a Bink 2 picture, through the package's own
// Bink2Scarlett.dll mapped by rt::Loader: BinkOpenWithOptions, BinkGetFrameBuffersInfo, BinkRegisterFrameBuffers,
// BinkDoFrame. The exe imports none of BinkCopyToBuffer, BinkGetSummary or the GPU buffer calls, so this is the
// path a boot movie takes; bink_decode_test and bink_abi_test only cover BinkCopyToBuffer.
//
//   LEONIDA_GAME_DIR=<package root>   (contains Bink2Scarlett.dll and movies.rpf)
//   LEONIDA_BINK_INDEX=<file.csv>     (tools/bink_index.py <package>/movies.rpf --out <file.csv>)
//   LEONIDA_BINK_LIMIT=<n>            (optional: how many streams, spread over the index; default 12)
//
// Each stream is copied from movies.rpf to one temp file (overwritten by the next), opened, and the plane layout and
// decode results are checked. It prints counts and pass/fail only: no frame, pixel or audio data.
// With either variable unset it prints "SKIP (no package/index)" and exits 77 (CTest skip).
#include <windows.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

[[noreturn]] void fatal(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::exit(1);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct Stream {
    uint64_t offset = 0;
    uint32_t frames = 0, width = 0, height = 0, fps_num = 0, fps_den = 0, audio_tracks = 0;
};

// The stream header at `offset` (the same fields bink_decode_test reads from a carved file).
Stream read_stream(std::ifstream& f, uint64_t offset) {
    uint8_t h[0x2C]{};
    f.clear();
    f.seekg((std::streamoff)offset);
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) fatal("index offset is past the end of movies.rpf");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2') fatal("index offset does not start a KB2 stream");
    Stream s;
    s.offset = offset;
    s.frames = read_u32(h + 0x08);
    s.width = read_u32(h + 0x14);
    s.height = read_u32(h + 0x18);
    s.fps_num = read_u32(h + 0x1C);
    s.fps_den = read_u32(h + 0x20);
    s.audio_tracks = read_u32(h + 0x28);
    if (!s.frames || !s.width || !s.height || !s.fps_num || !s.fps_den) fatal("stream header has a zero field");
    if (s.width > 16384 || s.height > 16384) fatal("stream header has an implausible frame size");
    return s;
}

// The "offset" column of tools/bink_index.py's CSV.
std::vector<uint64_t> read_index(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) fatal("cannot open LEONIDA_BINK_INDEX");
    std::string line;
    if (!std::getline(f, line)) fatal("LEONIDA_BINK_INDEX is empty");
    std::vector<std::string> head;
    for (std::stringstream ss(line); std::getline(ss, line, ',');) head.push_back(line);
    const auto col = std::find(head.begin(), head.end(), "offset");
    if (col == head.end()) fatal("LEONIDA_BINK_INDEX has no offset column");
    const size_t at = (size_t)(col - head.begin());
    std::vector<uint64_t> out;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cells;
        for (std::stringstream ss(line); std::getline(ss, line, ',');) cells.push_back(line);
        if (cells.size() <= at) fatal("LEONIDA_BINK_INDEX row has no offset");
        out.push_back(std::strtoull(cells[at].c_str(), nullptr, 10));
    }
    return out;
}



void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    return it->second;
}

// The BINKFRAMEBUFFERS that BinkGetFrameBuffersInfo fills, as measured on the package's 1.0.2.1 streams
// (docs/movies.md): up to three frame sets, each with Y, cR, cB and A planes of {buffer, flag, pitch}.
struct Plane {
    uint8_t* buffer;
    uint32_t flag;   // 1 for every plane the library allocated, 0 for the unused A plane
    uint32_t pitch;  // bytes per row, 8 bits per sample: each plane's width rounded up to 64
};
struct PlaneSet {
    Plane y, cr, cb, a;
    uint8_t reserved[16];  // zero in every stream
};
struct FrameBuffers {
    uint32_t total_frames;
    uint32_t ya_width, ya_height, crcb_width, crcb_height;
    uint32_t reserved;  // 0 in every stream
    PlaneSet frames[3];  // the third set is zero when total_frames is 2
};
static_assert(sizeof(Plane) == 16 && sizeof(PlaneSet) == 80 && sizeof(FrameBuffers) == 24 + 3 * 80, "layout drifted");

uint32_t align64(uint32_t v) { return (v + 63) & ~63u; }

constexpr size_t kInfoBytes = 512;  // BinkGetFrameBuffersInfo is given more room than the structure needs

}  // namespace

int main() {
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    const std::string index_arg = env("LEONIDA_BINK_INDEX");
    if (game_dir_arg.empty() || index_arg.empty()) {
        std::printf("SKIP (no package/index)\n");
        return 77;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path archive = game_dir / "movies.rpf";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("LEONIDA_GAME_DIR does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(archive, ec)) fatal("LEONIDA_GAME_DIR does not contain movies.rpf");

    std::vector<uint64_t> index = read_index(index_arg);
    size_t limit = 12;  // each stream is copied to one temp file first, so the default stays small
    if (!env("LEONIDA_BINK_LIMIT").empty()) limit = std::strtoul(env("LEONIDA_BINK_LIMIT").c_str(), nullptr, 10);
    // Spread the streams over the whole index so the sizes, revisions and aspect ratios all show up.
    std::vector<uint64_t> picked;
    for (size_t i = 0; i < limit && i < index.size(); ++i) picked.push_back(index[i * index.size() / std::min(limit, index.size())]);
    if (picked.empty()) fatal("LEONIDA_BINK_INDEX lists no stream");
    std::ifstream rpf(archive, std::ios::binary);
    if (!rpf) fatal("cannot open movies.rpf");

    try {
        rt::Loader loader(game_dir);
        rt::Module& dll = loader.load(game_dir / "Bink2Scarlett.dll");
        auto fn = [&](const char* name) { return symbol(dll, name); };
        // BinkOpenWithOptions(path, flags, options): the retail exe's open call. Both arguments after the path are 0
        // here (a carved stream needs neither the file-offset flag nor options); which of the two is the flags word
        // was not told apart, so the test does not pass either.
        auto open = reinterpret_cast<void* (*)(const char*, uint64_t, uint64_t)>(fn("BinkOpenWithOptions"));
        auto close = reinterpret_cast<void (*)(void*)>(fn("BinkClose"));
        auto info = reinterpret_cast<void (*)(void*, void*)>(fn("BinkGetFrameBuffersInfo"));
        auto registerfb = reinterpret_cast<void (*)(void*, void*)>(fn("BinkRegisterFrameBuffers"));
        auto do_frame = reinterpret_cast<uint32_t (*)(void*)>(fn("BinkDoFrame"));
        auto get_error = reinterpret_cast<const char* (*)()>(fn("BinkGetError"));

        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "leonida_bink_framebuffers.bk2";
        uint32_t opened = 0;
        for (uint64_t offset : picked) {
            const Stream s = read_stream(rpf, offset);
            {  // one stream, copied out of the archive to one temp file that the next stream overwrites
                uint8_t h[8]{};
                rpf.clear();
                rpf.seekg((std::streamoff)offset);
                rpf.read(reinterpret_cast<char*>(h), 8);
                std::vector<char> bytes((size_t)read_u32(h + 4) + 8);
                rpf.clear();
                rpf.seekg((std::streamoff)offset);
                rpf.read(bytes.data(), (std::streamsize)bytes.size());
                std::ofstream(tmp, std::ios::binary).write(bytes.data(), (std::streamsize)bytes.size());
            }
            const std::string tag = std::to_string(s.width) + "x" + std::to_string(s.height) + " @" + std::to_string(offset);
            void* bink = open(tmp.string().c_str(), 0, 0);
            check(bink != nullptr, tag + ": BinkOpenWithOptions opens the stream");
            if (!bink) {
                std::printf("  %s\n", get_error());
                continue;
            }
            ++opened;

            std::vector<uint8_t> raw(kInfoBytes, 0xCD);
            auto* fb = reinterpret_cast<FrameBuffers*>(raw.data());
            info(bink, raw.data());
            size_t written = 0;
            for (size_t i = 0; i < raw.size(); ++i)
                if (raw[i] != 0xCD) written = i + 1;
            check(written == sizeof(FrameBuffers), tag + ": BinkGetFrameBuffersInfo writes exactly the measured structure");
            check(fb->total_frames == 2, tag + ": two frame sets");
            check(fb->ya_width >= s.width && fb->ya_width - s.width < 32 && fb->ya_height >= s.height && fb->ya_height - s.height < 16 && fb->ya_width % 32 == 0 && fb->ya_height % 16 == 0,
                  tag + ": the Y buffer is the picture, width rounded up to 32 and height to 16");
            check(fb->crcb_width * 2 == fb->ya_width && fb->crcb_height * 2 == fb->ya_height, tag + ": chroma is 4:2:0");
            bool planes_ok = true;
            for (uint32_t f = 0; f < 2; ++f) {
                const PlaneSet& set = fb->frames[f];
                planes_ok = planes_ok && set.y.buffer && set.cr.buffer && set.cb.buffer && !set.a.buffer;
                planes_ok = planes_ok && set.y.flag == 1 && set.cr.flag == 1 && set.cb.flag == 1;
                planes_ok = planes_ok && set.y.pitch == align64(fb->ya_width) && set.cr.pitch == align64(fb->crcb_width) &&
                            set.cb.pitch == align64(fb->crcb_width);
            }
            check(planes_ok, tag + ": Y/cR/cB are allocated, flag 1, each pitch is its width rounded up to 64, no A plane");

            // 1. The library's own buffers: decode frame 0 and see the Y plane written.
            for (uint32_t f = 0; f < 2; ++f) std::memset(fb->frames[f].y.buffer, 0xCD, (size_t)fb->frames[f].y.pitch * fb->ya_height);
            do_frame(bink);
            size_t lib_touched = 0;
            for (uint32_t f = 0; f < 2; ++f) {
                const Plane& y = fb->frames[f].y;
                lib_touched += (size_t)std::count_if(y.buffer, y.buffer + (size_t)y.pitch * fb->ya_height,
                                                     [](uint8_t b) { return b != 0xCD; });
            }
            // A black first frame is a constant Y, so "written" is the test, not "has detail".
            check(lib_touched > 0, tag + ": BinkDoFrame writes the library's Y plane");

            // 2. Caller-owned buffers, in the order a game uses: open, ask for the layout, register its own planes
            // before the first decode, then decode and find the picture in them. (Registering after a decode left
            // the planes untouched on 21 of 515 streams.)
            close(bink);
            bink = open(tmp.string().c_str(), 0, 0);
            check(bink != nullptr, tag + ": the stream opens a second time");
            if (!bink) continue;
            std::vector<uint8_t> raw2(kInfoBytes, 0xCD);
            info(bink, raw2.data());
            std::vector<std::vector<uint8_t>> own;
            FrameBuffers mine = *reinterpret_cast<FrameBuffers*>(raw2.data());
            for (uint32_t f = 0; f < 2; ++f) {
                PlaneSet& set = mine.frames[f];
                for (Plane* p : {&set.y, &set.cr, &set.cb}) {
                    const uint32_t rows = (p == &set.y) ? fb->ya_height : fb->crcb_height;
                    own.emplace_back((size_t)p->pitch * rows, 0xCD);
                    p->buffer = own.back().data();
                }
            }
            registerfb(bink, &mine);
            do_frame(bink);
            size_t touched = 0;
            for (const auto& plane : own)
                touched += (size_t)std::count_if(plane.begin(), plane.end(), [](uint8_t b) { return b != 0xCD; });
            check(touched > 0, tag + ": a frame decoded after BinkRegisterFrameBuffers lands in the caller's planes");
            close(bink);
        }
        check(opened == picked.size(), "every picked stream opened");
        std::printf("streams: %zu picked, %u opened\n", picked.size(), opened);
        std::printf("%s\n", g_failures ? "RESULT FAIL" : "RESULT PASS");
        return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
