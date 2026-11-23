// Optional Windows test: the retail exe's async Bink decode calls, through the package's own Bink2Scarlett.dll mapped by
// rt::Loader: BinkStartAsyncThread, BinkDoFrameAsyncMulti + BinkDoFrameAsyncWait, and the two stop calls. A frame
// decoded that way must equal the same frame from BinkDoFrame, plane for plane.
//
//   LEONIDA_GAME_DIR=<package root>   (contains Bink2Scarlett.dll and movies.rpf)
//   LEONIDA_BINK_INDEX=<file.csv>     (tools/bink_index.py <package>/movies.rpf --out <file.csv>)
//   LEONIDA_BINK_LIMIT=<n>            (optional: how many streams, spread over the index; default 8)
//
// Each stream is copied from movies.rpf to one temp file (overwritten by the next). It prints hashes' pass/fail and
// counts only: no frame, pixel or audio data. With either variable unset it prints "SKIP (no package/index)", exit 77.
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

uint64_t fnv(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// The picture area of the library's six planes (two frame sets of Y, cR, cB) in one hash, so a frame decoded two ways
// compares byte for byte; the padding past the picture is never written and is not compared. Frame 0 writes one set only, so the comparison starts at frame 1, when both have been decoded into.
// Layout of the structure BinkGetFrameBuffersInfo fills: see bink_framebuffers_test.cpp and docs/movies.md.
uint64_t hash_planes(const std::vector<uint8_t>& info, uint32_t width, uint32_t height) {
    uint64_t h = 1469598103934665603ull;
    for (size_t set = 0; set < 2; ++set)
        for (size_t plane = 0; plane < 3; ++plane) {
            const uint8_t* e = info.data() + 0x18 + set * 80 + plane * 16;
            uint8_t* buffer;
            std::memcpy(&buffer, e, sizeof(buffer));
            const uint32_t pitch = read_u32(e + 12);
            const uint32_t w = plane ? (width + 1) / 2 : width, rows = plane ? (height + 1) / 2 : height;
            for (uint32_t r = 0; r < rows; ++r) h = fnv(h, buffer + (size_t)r * pitch, w);
        }
    return h;
}

typedef uint64_t (*Fn3)(uint64_t, uint64_t, uint64_t);

// Calls fn(a, b, c); an access violation is reported as failure instead of killing the test. SEH only: no C++ objects.
int guarded(Fn3 fn, uint64_t a, uint64_t b, uint64_t c, uint64_t* rc) {
    __try {
        *rc = fn(a, b, c);
        return 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 1;
    }
}

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
    size_t limit = 8;
    if (!env("LEONIDA_BINK_LIMIT").empty()) limit = std::strtoul(env("LEONIDA_BINK_LIMIT").c_str(), nullptr, 10);
    std::vector<uint64_t> picked;
    for (size_t i = 0; i < limit && i < index.size(); ++i) picked.push_back(index[i * index.size() / std::min(limit, index.size())]);
    if (picked.empty()) fatal("LEONIDA_BINK_INDEX lists no stream");
    std::ifstream rpf(archive, std::ios::binary);
    if (!rpf) fatal("cannot open movies.rpf");

    try {
        rt::Loader loader(game_dir);
        rt::Module& dll = loader.load(game_dir / "Bink2Scarlett.dll");
        auto F = [&](const char* name) { return reinterpret_cast<Fn3>(symbol(dll, name)); };
        auto call = [&](const char* name, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0) {
            uint64_t rc = 0;
            if (guarded(F(name), a, b, c, &rc)) {
                std::printf("FAIL %s raised an access violation\n", name);
                ++g_failures;
            }
            return rc;
        };

        // The exe's calls, with the argument shapes read off its call sites (docs/movies.md):
        //   BinkStartAsyncThread(thread, 0) -> 1
        //   BinkNextFrame(bink); BinkDoFrameAsyncMulti(bink, u32 thread_ids[], count); BinkDoFrameAsyncWait(bink, ms)
        //   BinkRequestStopAsyncThreadsMulti(thread_ids, count); BinkWaitStopAsyncThreadsMulti(thread_ids, count)
        const uint32_t ids[1] = {0};
        check(call("BinkStartAsyncThread", 0, 0) == 1, "BinkStartAsyncThread(0, 0) returns 1");

        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "leonida_bink_async.bk2";
        constexpr int kFrames = 3;  // frame 0 decoded synchronously, then two more
        uint32_t same = 0, changed = 0, waits[2] = {};
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
            uint64_t hashes[3][kFrames] = {};
            for (int run = 0; run < 3; ++run) {
                const int async = run == 1;
                const uint64_t bink = call("BinkOpenWithOptions", (uint64_t)tmp.string().c_str(), 0, 0);
                check(bink != 0, tag + ": opens");
                if (!bink) break;
                std::vector<uint8_t> info(512, 0xCD);
                call("BinkGetFrameBuffersInfo", bink, (uint64_t)info.data());
                const uint64_t before = hash_planes(info, s.width, s.height);
                for (int f = 0; f < kFrames; ++f) {
                    if (f) call("BinkNextFrame", bink);
                    if (async && f) {
                        call("BinkDoFrameAsyncMulti", bink, (uint64_t)ids, 1);
                        const uint64_t waited = call("BinkDoFrameAsyncWait", bink, 0xffffffffull);  // the exe's argument
                        waits[waited != 0 ? 1 : 0]++;
                    } else {
                        call("BinkDoFrame", bink);
                    }
                    hashes[run][f] = hash_planes(info, s.width, s.height);
                    changed += hashes[run][f] != before;
                }
                call("BinkClose", bink);
            }
            bool twice = true;
            for (int f = 1; f < kFrames; ++f) twice = twice && hashes[0][f] == hashes[2][f];
            if (!twice) std::printf("  (BinkDoFrame is not repeatable here)\n");
            bool equal = true;
            for (int f = 1; f < kFrames; ++f) equal = equal && hashes[0][f] == hashes[1][f];
            check(equal, tag + ": frames 1 and 2 decoded with BinkDoFrameAsyncMulti match BinkDoFrame byte for byte");
            if (equal) ++same;
        }
        std::printf("streams: %zu picked, %u identical\n", picked.size(), same);

        std::printf("BinkDoFrameAsyncWait returned 0 %u times, non-zero %u times\n", waits[0], waits[1]);
        check(waits[0] == 0, "BinkDoFrameAsyncWait(bink, 0xffffffff) returns non-zero every time");
        check(changed > 0, "the decodes wrote picture data (planes differ from the freshly opened ones)");
        call("BinkRequestStopAsyncThreadsMulti", (uint64_t)ids, 1);
        call("BinkWaitStopAsyncThreadsMulti", (uint64_t)ids, 1);
        std::printf("%s\n", g_failures ? "RESULT FAIL" : "RESULT PASS");
        return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
