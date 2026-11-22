// CPU-only unit tests for gta6rt CLI argument parsing (runtime/src/cli.h, #886).
#include <cstdio>
#include <initializer_list>
#include <vector>

#include "../src/cli.h"

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

bool parse(std::initializer_list<const wchar_t*> args, rt::CliOptions& out) {
    std::vector<const wchar_t*> argv(args);
    return rt::parse_cli_args(static_cast<int>(argv.size()), argv.data(), out);
}

}  // namespace

int main() {
    rt::CliOptions opts;

    // Valid single modes
    check(parse({L"gta6rt", L"--dry-run", L"dir/game.exe"}, opts), "--dry-run valid");
    check(opts.mode == rt::RunMode::DryRun, "--dry-run sets RunMode::DryRun");
    check(opts.image == std::filesystem::path(L"dir/game.exe"), "--dry-run image path");
    check(opts.game_dir == std::filesystem::path(L"dir"), "--dry-run default game_dir");
    check(!opts.trace_d3d12, "--dry-run default trace_d3d12=false");

    check(parse({L"gta6rt", L"--init", L"game.dll", L"--game-dir", L"root", L"--trace-d3d12"}, opts), "--init valid");
    check(opts.mode == rt::RunMode::Init, "--init sets RunMode::Init");
    check(opts.game_dir == std::filesystem::path(L"root"), "--game-dir override");
    check(opts.trace_d3d12, "--trace-d3d12 enabled");

    check(parse({L"gta6rt", L"--run", L"game.exe", L"--", L"--dry-run", L"arg2"}, opts),
          "--run valid with mode-like game args after --");
    check(opts.mode == rt::RunMode::Run, "--run sets RunMode::Run");
    check(opts.game_args.size() == 2 && opts.game_args[0] == "--dry-run" && opts.game_args[1] == "arg2",
          "args after -- are passed through verbatim and not parsed as mode flags");

    // Mutually exclusive / repeated mode flags must be rejected (#886)
    check(!parse({L"gta6rt", L"--dry-run", L"--run", L"game.exe"}, opts), "reject --dry-run --run");
    check(!parse({L"gta6rt", L"--run", L"--dry-run", L"game.exe"}, opts), "reject --run --dry-run");
    check(!parse({L"gta6rt", L"--dry-run", L"--init", L"game.exe"}, opts), "reject --dry-run --init");
    check(!parse({L"gta6rt", L"--init", L"--run", L"game.exe"}, opts), "reject --init --run");
    check(!parse({L"gta6rt", L"--dry-run", L"--init", L"--run", L"game.exe"}, opts), "reject all three modes");
    check(!parse({L"gta6rt", L"--dry-run", L"--dry-run", L"game.exe"}, opts), "reject repeated --dry-run");
    check(!parse({L"gta6rt", L"--run", L"game.exe", L"--dry-run"}, opts), "reject second mode after image");

    // Missing mode or missing image
    check(!parse({L"gta6rt", L"game.exe"}, opts), "reject missing mode");
    check(!parse({L"gta6rt", L"--dry-run"}, opts), "reject missing image");
    check(!parse({L"gta6rt"}, opts), "reject empty args");
    check(!parse({L"gta6rt", L"--dry-run", L"a.exe", L"b.exe"}, opts), "reject extra positional arg");

    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("cli_test: PASS\n");
    return 0;
}
