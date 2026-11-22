// Command-line argument parser for gta6rt (extracted from main.cpp so CPU-only tests can exercise it).
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace rt {

enum class RunMode {
    None,
    DryRun,
    Init,
    Run,
};

struct CliOptions {
    RunMode mode = RunMode::None;
    std::filesystem::path image;
    std::filesystem::path game_dir;
    bool trace_d3d12 = false;
    std::vector<std::string> game_args;
};

// Parses argv[1..argc-1] for gta6rt. Returns true on a valid invocation, false on usage error.
inline bool parse_cli_args(int argc, const wchar_t* const* argv, CliOptions& out) {
    out = CliOptions{};
    auto set_mode = [&](RunMode m) -> bool {
        if (out.mode != RunMode::None) return false;
        out.mode = m;
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::wstring_view a = argv[i];
        if (a == L"--") {
            for (++i; i < argc; ++i) out.game_args.push_back(std::filesystem::path(argv[i]).string());
            break;
        }
        if (a == L"--dry-run") {
            if (!set_mode(RunMode::DryRun)) return false;
        } else if (a == L"--init") {
            if (!set_mode(RunMode::Init)) return false;
        } else if (a == L"--run") {
            if (!set_mode(RunMode::Run)) return false;
        } else if (a == L"--trace-d3d12") {
            out.trace_d3d12 = true;
        } else if (a == L"--game-dir" && i + 1 < argc) {
            out.game_dir = argv[++i];
        } else if (out.image.empty()) {
            out.image = std::filesystem::path(a);
        } else {
            return false;
        }
    }
    if (out.image.empty() || out.mode == RunMode::None) return false;
    if (out.game_dir.empty()) out.game_dir = out.image.parent_path();
    return true;
}

}  // namespace rt
