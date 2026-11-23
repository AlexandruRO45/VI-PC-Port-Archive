// --game-dir's archive count (src/game_dir_check.h) over synthetic folders in %TEMP%: empty files named like the
// package's archives, never their content. No game data.
#include "../src/game_dir_check.h"

#include <process.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

namespace fs = std::filesystem;
using leonida::count_game_dir_archives;

namespace {

void touch(const fs::path& p) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary).put('\0');
}

int run(const fs::path& root) {
    // A folder that does not exist: not a game folder, nothing counted.
    {
        const auto r = count_game_dir_archives(root / "missing");
        CHECK(!r.exists && r.rpf == 0);
    }
    // Only the executable (the case this check is for).
    const fs::path exe_only = root / "exe_only";
    touch(exe_only / "game.exe");
    {
        const auto r = count_game_dir_archives(exe_only);
        CHECK(r.exists && r.rpf == 0);
    }
    // The package's layout: archives at the top level and under pack/<name>/, any case.
    const fs::path pkg = root / "package";
    touch(pkg / "game.exe");
    touch(pkg / "common.rpf");
    touch(pkg / "DATA.RPF");
    touch(pkg / "pack" / "core_a" / "core_a.rpf");
    touch(pkg / "pack" / "core_b" / "core_b_data.rpf");
    touch(pkg / "pack" / "loose.rpf");
    {
        const auto r = count_game_dir_archives(pkg);
        CHECK(r.exists && r.rpf == 5);
    }
    // Neither a folder named like an archive, nor other extensions, nor archives deeper than the package keeps
    // them (or outside pack/) count.
    fs::create_directories(pkg / "folder.rpf");
    touch(pkg / "common.rpf.bak");
    touch(pkg / "notes.txt");
    touch(pkg / "pack" / "core_a" / "deeper" / "x.rpf");
    touch(pkg / "other" / "y.rpf");
    {
        const auto r = count_game_dir_archives(pkg);
        CHECK(r.rpf == 5);
    }
    // A file passed as --game-dir is not a folder.
    {
        const auto r = count_game_dir_archives(pkg / "common.rpf");
        CHECK(!r.exists && r.rpf == 0);
    }
    return 0;
}

}  // namespace

int main() {
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / ("leonida_game_dir_check_" + std::to_string(::_getpid()));
    if (ec) {
        std::fprintf(stderr, "no temporary folder\n");
        return 1;
    }
    fs::remove_all(root, ec);
    int rc = 1;
    try {
        rc = run(root);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
    }
    fs::remove_all(root, ec);
    if (rc == 0) std::printf("game_dir_check: ok\n");
    return rc;
}
