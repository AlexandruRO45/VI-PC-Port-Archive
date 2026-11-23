#pragma once

// Does --game-dir look like the installed package? (docs/boot.md, "Commands")
//
// The title reads its data from the package's archives. Pointed at a folder that holds only the executable, a
// `--run` still gets through TLS, xgameruntime and DirectStorage, then the title's first data reads fail and it ends
// in its own asserts and an access violation (measured on Windows 11 with the 1.0.2.1 executable): nothing in the
// loader's output says why. So `--init` and `--run` count the archives first and warn when there are none.
//
// The package keeps its `.rpf` archives at the top level and under `pack/<name>/` (game/xbox-1.0.2.1.json lists 22),
// so the count covers the top level and two levels below `pack`. It only reads directory entries, opens no file and
// never fails: an unreadable folder counts as empty. A stand-in image (tiny_game) has no archives, so the result is a
// warning, not an error. Tested without game data (game_dir_check_test).

#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>

namespace leonida {

struct GameDirArchives {
    bool exists = false;   // --game-dir is a directory
    std::size_t rpf = 0;   // .rpf files at its top level and under pack/ (two levels)
};

namespace detail {
inline bool is_rpf(const std::filesystem::directory_entry& e) {
    std::error_code ec;
    if (!e.is_regular_file(ec) || ec) return false;
    std::wstring ext = e.path().extension().native();
    for (wchar_t& c : ext)
        if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
    return ext == L".rpf";
}

inline std::size_t count_rpf(const std::filesystem::path& dir, int depth) {
    std::size_t n = 0;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec) return 0;
    for (const std::filesystem::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) break;
        const auto& e = *it;
        if (is_rpf(e)) {
            ++n;
        } else if (depth > 0) {
            std::error_code dir_ec;
            if (e.is_directory(dir_ec) && !dir_ec) n += count_rpf(e.path(), depth - 1);
        }
    }
    return n;
}
}  // namespace detail

inline GameDirArchives count_game_dir_archives(const std::filesystem::path& game_dir) {
    GameDirArchives r;
    std::error_code ec;
    r.exists = std::filesystem::is_directory(game_dir, ec) && !ec;
    if (!r.exists) return r;
    r.rpf = detail::count_rpf(game_dir, 0);
    const std::filesystem::path pack = game_dir / L"pack";
    if (std::filesystem::is_directory(pack, ec) && !ec) r.rpf += detail::count_rpf(pack, 1);
    return r;
}

}  // namespace leonida
