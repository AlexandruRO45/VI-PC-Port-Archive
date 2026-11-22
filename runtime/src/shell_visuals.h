#pragma once

// The title's shell visuals: what the Xbox shell shows while a title starts, before its first frame. They come from
// the package's MicrosoftGame.Config (<ShellVisuals SplashScreenImage="..." BackgroundColor="#RRGGBB" .../> and the
// <Executable OverrideDisplayName="..."/> or ShellVisuals DefaultDisplayName), read from the user's own game folder at
// run time; nothing of the package is part of the runtime.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rt::shell {

struct ShellVisuals {
    bool found = false;               // MicrosoftGame.Config was read and has a <ShellVisuals> element
    std::wstring display_name;        // OverrideDisplayName of the first <Executable>, else DefaultDisplayName
    std::filesystem::path splash;     // SplashScreenImage, resolved against the game folder; empty when absent
    uint32_t background = 0x000000;   // BackgroundColor as 0xRRGGBB (black when absent or malformed)
};

// Reads <game_dir>/MicrosoftGame.Config (UTF-8, with or without a BOM, or UTF-16 LE with a BOM).
ShellVisuals read_shell_visuals(const std::filesystem::path& game_dir);

// The same, from the file's text (UTF-8); for tests.
ShellVisuals parse_shell_visuals(const std::string& config_utf8, const std::filesystem::path& game_dir);

// "#RRGGBB" (or "RRGGBB") to 0xRRGGBB; false for anything else.
bool parse_color(const std::string& text, uint32_t& rgb);

// An image decoded by WIC (PNG, JPEG, BMP, ...) to top-down 32-bit BGRA; false with the reason on failure.
struct Image {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> bgra;  // width * height * 4
};
bool decode_image(const std::filesystem::path& path, Image& out, std::string& error);

// Opens a window titled after the title that shows the splash centred and scaled to fit on the background colour,
// as the console shell does, until the user closes it (Esc or the close button) or `limit_ms` passes (0: no limit).
// Without a splash image the window shows the background alone. Returns 0, or 1 when no window could be made.
int show_splash(const ShellVisuals& visuals, uint32_t limit_ms);

}  // namespace rt::shell
