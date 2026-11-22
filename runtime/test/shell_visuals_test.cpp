// Shell visuals (runtime/src/shell_visuals.h): MicrosoftGame.Config parsing, the splash image decode and the splash
// window, from a synthetic config and a PNG this test encodes itself. No game files.
#include <windows.h>
#include <wincodec.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "../src/shell_visuals.h"

namespace {

int g_failed = 0, g_checks = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) ++g_failed;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
}

// A w x h PNG of one colour (0xRRGGBB), written with WIC.
bool write_png(const std::filesystem::path& path, UINT w, UINT h, uint32_t rgb) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) hr = factory->CreateStream(&stream);
    if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
    if (SUCCEEDED(hr)) hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = encoder->CreateNewFrame(&frame, nullptr);
    if (SUCCEEDED(hr)) hr = frame->Initialize(nullptr);
    if (SUCCEEDED(hr)) hr = frame->SetSize(w, h);
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
    if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&format);
    std::string pixels(static_cast<size_t>(w) * h * 4, '\0');
    for (size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i] = static_cast<char>(rgb & 0xFF);
        pixels[i + 1] = static_cast<char>((rgb >> 8) & 0xFF);
        pixels[i + 2] = static_cast<char>((rgb >> 16) & 0xFF);
        pixels[i + 3] = static_cast<char>(0xFF);
    }
    if (SUCCEEDED(hr))
        hr = frame->WritePixels(h, w * 4, static_cast<UINT>(pixels.size()), reinterpret_cast<BYTE*>(pixels.data()));
    if (SUCCEEDED(hr)) hr = frame->Commit();
    if (SUCCEEDED(hr)) hr = encoder->Commit();
    if (frame) frame->Release();
    if (encoder) encoder->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();
    CoUninitialize();
    return SUCCEEDED(hr);
}

}  // namespace

int main() {
    using namespace rt::shell;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "leonida_shell_visuals_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir / "art");

    // Parsing: OverrideDisplayName wins, the splash path is under the game folder, the colour is #RRGGBB.
    const std::string config =
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<Game configVersion=\"1\">\n"
        "  <!-- <ShellVisuals SplashScreenImage=\"commented.png\"/> -->\n"
        "  <ExecutableList><Executable Name=\"title.exe\" Id=\"Game\" OverrideDisplayName=\"Test &amp; Title\"/>"
        "</ExecutableList>\n"
        "  <ShellVisuals DefaultDisplayName=\"Default Name\" BackgroundColor=\"#1A2b3C\"\n"
        "                SplashScreenImage='art\\splash.png' ForegroundText=\"light\"/>\n</Game>\n";
    ShellVisuals v = parse_shell_visuals(config, dir);
    check(v.found, "a <ShellVisuals> element is found (not the commented one)");
    check(v.display_name == L"Test & Title", "OverrideDisplayName names the window, with &amp; unescaped");
    check(v.splash == dir / "art" / "splash.png", "SplashScreenImage resolves under the game folder");
    check(v.background == 0x1A2B3C, "BackgroundColor #1A2b3C reads as 0x1A2B3C");

    ShellVisuals d = parse_shell_visuals("<Game><ShellVisuals DefaultDisplayName=\"Only Default\"/></Game>", dir);
    check(d.found && d.display_name == L"Only Default" && d.splash.empty() && d.background == 0,
          "without an override the default name is used; no splash, black background");
    check(!parse_shell_visuals("<Game><ShellVisualsX/></Game>", dir).found, "a longer element name is not ShellVisuals");
    check(parse_shell_visuals("<Game><ShellVisuals SplashScreenImage=\"..\\outside.png\"/></Game>", dir).splash.empty(),
          "a splash path leaving the game folder is refused");
    check(parse_shell_visuals("<Game><ShellVisuals SplashScreenImage=\"C:\\x.png\"/></Game>", dir).splash.empty(),
          "an absolute splash path is refused");
    uint32_t rgb = 7;
    check(!parse_color("#12345", rgb) && !parse_color("#12345G", rgb) && rgb == 7 && parse_color("ffffff", rgb) &&
              rgb == 0xFFFFFF,
          "colours other than six hex digits are rejected");

    // The file: UTF-8 with a BOM, and UTF-16 LE with a BOM, as package tools write it.
    {
        std::ofstream f(dir / "MicrosoftGame.Config", std::ios::binary);
        f << "\xEF\xBB\xBF" << config;
    }
    check(read_shell_visuals(dir).display_name == L"Test & Title", "a UTF-8 config with a BOM is read");
    {
        const std::wstring wide(config.begin(), config.end());
        std::ofstream f(dir / "MicrosoftGame.Config", std::ios::binary);
        f.write("\xFF\xFE", 2);
        f.write(reinterpret_cast<const char*>(wide.data()), static_cast<std::streamsize>(wide.size() * 2));
    }
    v = read_shell_visuals(dir);
    check(v.found && v.background == 0x1A2B3C, "a UTF-16 LE config with a BOM is read");
    check(!read_shell_visuals(dir / "missing").found, "a folder without a config has no shell visuals");

    // Decode: the PNG comes back as BGRA of its size and colour.
    check(write_png(dir / "art" / "splash.png", 64, 36, 0x336699), "the test PNG is written");
    Image img;
    std::string error;
    check(decode_image(v.splash, img, error) && img.width == 64 && img.height == 36 && img.bgra.size() == 64 * 36 * 4 &&
              img.bgra[0] == 0x99 && img.bgra[1] == 0x66 && img.bgra[2] == 0x33 && img.bgra[3] == 0xFF,
          "the splash decodes to 64x36 BGRA of its colour");
    {
        std::ofstream f(dir / "art" / "broken.png", std::ios::binary);
        f << "not a png";
    }
    check(!decode_image(dir / "art" / "broken.png", img, error) && !error.empty() && img.bgra.empty(),
          "a file that isn't an image fails with a reason");

    // The window: opens, shows, and closes itself after the limit.
    const DWORD start = GetTickCount();
    check(show_splash(v, 300) == 0 && GetTickCount() - start < 10000, "the splash window opens and closes after 300 ms");
    ShellVisuals none = v;
    none.splash = dir / "art" / "broken.png";
    check(show_splash(none, 100) == 0, "an undecodable splash still opens the window on the background");

    std::filesystem::remove_all(dir, ec);
    std::printf("shell_visuals_test: %s (%d checks)\n", g_failed ? "FAILED" : "OK", g_checks);
    return g_failed ? 1 : 0;
}
