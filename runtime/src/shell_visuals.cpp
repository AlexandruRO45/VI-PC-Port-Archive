#include "shell_visuals.h"

#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace rt::shell {

namespace {

std::wstring widen(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), out.data(), n);
    return out;
}

std::string narrow(const std::wstring& wide) {
    if (wide.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr,
                                      nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), out.data(), n, nullptr, nullptr);
    return out;
}

// The XML entities an attribute value can hold; the config needs nothing more than these.
std::string unescape(std::string v) {
    static const std::pair<const char*, const char*> entities[] = {
        {"&quot;", "\""}, {"&apos;", "'"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&amp;", "&"}};
    for (const auto& [from, to] : entities)
        for (size_t at = 0; (at = v.find(from, at)) != std::string::npos; at += std::strlen(to))
            v.replace(at, std::strlen(from), to);
    return v;
}

// The start tag of the first <name ...> element (up to its '>'), or empty. Comments are skipped.
std::string element(const std::string& xml, const char* name) {
    const std::string open = std::string("<") + name;
    for (size_t at = 0; (at = xml.find(open, at)) != std::string::npos; at += open.size()) {
        const size_t comment = xml.rfind("<!--", at);
        if (comment != std::string::npos && xml.find("-->", comment) > at) continue;  // inside a comment
        const char next = at + open.size() < xml.size() ? xml[at + open.size()] : '\0';
        if (next != ' ' && next != '\t' && next != '\r' && next != '\n' && next != '/' && next != '>') continue;
        const size_t end = xml.find('>', at);
        if (end == std::string::npos) return {};
        return xml.substr(at, end - at + 1);
    }
    return {};
}

// The value of attribute `name` in a start tag, or empty. Quotes may be " or '.
std::string attribute(const std::string& tag, const char* name) {
    const std::string key = name;
    for (size_t at = 0; (at = tag.find(key, at)) != std::string::npos; at += key.size()) {
        const char before = at ? tag[at - 1] : ' ';
        if (before != ' ' && before != '\t' && before != '\r' && before != '\n') continue;  // part of a longer name
        size_t p = at + key.size();
        while (p < tag.size() && (tag[p] == ' ' || tag[p] == '\t' || tag[p] == '\r' || tag[p] == '\n')) ++p;
        if (p >= tag.size() || tag[p] != '=') continue;
        ++p;
        while (p < tag.size() && (tag[p] == ' ' || tag[p] == '\t' || tag[p] == '\r' || tag[p] == '\n')) ++p;
        if (p >= tag.size() || (tag[p] != '"' && tag[p] != '\'')) continue;
        const char quote = tag[p];
        const size_t end = tag.find(quote, p + 1);
        if (end == std::string::npos) return {};
        return unescape(tag.substr(p + 1, end - p - 1));
    }
    return {};
}

}  // namespace

bool parse_color(const std::string& text, uint32_t& rgb) {
    std::string hex = text;
    if (!hex.empty() && hex[0] == '#') hex.erase(0, 1);
    if (hex.size() != 6) return false;
    uint32_t value = 0;
    for (char c : hex) {
        value <<= 4;
        if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
        else return false;
    }
    rgb = value;
    return true;
}

ShellVisuals parse_shell_visuals(const std::string& config, const std::filesystem::path& game_dir) {
    ShellVisuals v;
    const std::string visuals = element(config, "ShellVisuals");
    if (visuals.empty()) return v;
    v.found = true;
    std::string name = attribute(element(config, "Executable"), "OverrideDisplayName");
    if (name.empty()) name = attribute(visuals, "DefaultDisplayName");
    v.display_name = widen(name);
    const std::string splash = attribute(visuals, "SplashScreenImage");
    // A package-relative path: refuse anything that could leave the game folder.
    const std::filesystem::path rel = std::filesystem::path(widen(splash)).lexically_normal();
    if (!splash.empty() && rel.is_relative() && !rel.has_root_name() && *rel.begin() != "..")
        v.splash = game_dir / rel;
    uint32_t rgb = 0;
    if (parse_color(attribute(visuals, "BackgroundColor"), rgb)) v.background = rgb;
    return v;
}

ShellVisuals read_shell_visuals(const std::filesystem::path& game_dir) {
    std::ifstream f(game_dir / "MicrosoftGame.Config", std::ios::binary);
    if (!f) return {};
    std::string bytes{std::istreambuf_iterator<char>(f), {}};
    if (bytes.size() >= 2 && static_cast<uint8_t>(bytes[0]) == 0xFF && static_cast<uint8_t>(bytes[1]) == 0xFE) {
        std::wstring wide((bytes.size() - 2) / 2, L'\0');
        std::memcpy(wide.data(), bytes.data() + 2, wide.size() * 2);
        bytes = narrow(wide);
    } else if (bytes.size() >= 3 && bytes.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        bytes.erase(0, 3);
    }
    return parse_shell_visuals(bytes, game_dir);
}

bool decode_image(const std::filesystem::path& path, Image& out, std::string& error) {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninit = SUCCEEDED(init);
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr))
        hr = factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand,
                                                &decoder);
    if (SUCCEEDED(hr)) hr = decoder->GetFrame(0, &frame);
    if (SUCCEEDED(hr)) hr = factory->CreateFormatConverter(&converter);
    if (SUCCEEDED(hr))
        hr = converter->Initialize(frame, GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0,
                                   WICBitmapPaletteTypeCustom);
    UINT w = 0, h = 0;
    if (SUCCEEDED(hr)) hr = converter->GetSize(&w, &h);
    if (SUCCEEDED(hr) && (!w || !h || w > 16384 || h > 16384)) hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) {
        out.width = w;
        out.height = h;
        out.bgra.resize(static_cast<size_t>(w) * h * 4);
        hr = converter->CopyPixels(nullptr, w * 4, static_cast<UINT>(out.bgra.size()), out.bgra.data());
    }
    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (uninit) CoUninitialize();
    if (FAILED(hr)) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "WIC could not decode the image (HRESULT 0x%08lx)", static_cast<unsigned long>(hr));
        error = buf;
        out = {};
        return false;
    }
    return true;
}

namespace {

struct SplashWindow {
    const Image* image = nullptr;
    COLORREF background = RGB(0, 0, 0);
};

LRESULT CALLBACK splash_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    auto* s = reinterpret_cast<SplashWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_NCCREATE:
            SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                              reinterpret_cast<LONG_PTR>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams));
            break;
        case WM_KEYDOWN:
            if (wp == VK_ESCAPE) DestroyWindow(hwnd);
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC dc = BeginPaint(hwnd, &ps);
            RECT rc;
            GetClientRect(hwnd, &rc);
            HBRUSH brush = CreateSolidBrush(s ? s->background : RGB(0, 0, 0));
            FillRect(dc, &rc, brush);
            DeleteObject(brush);
            if (s && s->image && s->image->width) {
                const Image& img = *s->image;
                const int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
                // Fit inside the client area, keeping the aspect ratio.
                const double scale = std::min(static_cast<double>(cw) / img.width, static_cast<double>(ch) / img.height);
                const int dw = static_cast<int>(img.width * scale), dh = static_cast<int>(img.height * scale);
                BITMAPINFO bi{};
                bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
                bi.bmiHeader.biWidth = static_cast<LONG>(img.width);
                bi.bmiHeader.biHeight = -static_cast<LONG>(img.height);  // top-down
                bi.bmiHeader.biPlanes = 1;
                bi.bmiHeader.biBitCount = 32;
                bi.bmiHeader.biCompression = BI_RGB;
                SetStretchBltMode(dc, HALFTONE);
                StretchDIBits(dc, (cw - dw) / 2, (ch - dh) / 2, dw, dh, 0, 0, static_cast<int>(img.width),
                              static_cast<int>(img.height), img.bgra.data(), &bi, DIB_RGB_COLORS, SRCCOPY);
            }
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_SIZE:
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

int show_splash(const ShellVisuals& visuals, uint32_t limit_ms) {
    Image image;
    if (!visuals.splash.empty()) {
        std::string error;
        if (!decode_image(visuals.splash, image, error))
            std::fprintf(stderr, "[rt] shell: splash %ls: %s; showing the background only\n", visuals.splash.c_str(),
                         error.c_str());
    }
    SplashWindow state{image.width ? &image : nullptr,
                       RGB((visuals.background >> 16) & 0xFF, (visuals.background >> 8) & 0xFF,
                           visuals.background & 0xFF)};

    const HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = splash_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"LeonidaShellSplash";
    RegisterClassExW(&wc);  // a second call fails harmlessly with the class already there

    const std::wstring title =
        (visuals.display_name.empty() ? std::wstring(L"Game") : visuals.display_name) + L" (Leonida)";
    RECT r{0, 0, 1280, 720};  // a 16:9 client area, the console's output shape
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, title.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                r.right - r.left, r.bottom - r.top, nullptr, nullptr, instance, &state);
    if (!hwnd) {
        std::fprintf(stderr, "[rt] shell: CreateWindowEx failed (error %lu)\n", GetLastError());
        return 1;
    }
    ShowWindow(hwnd, SW_SHOWNORMAL);
    UpdateWindow(hwnd);
    if (limit_ms) SetTimer(hwnd, 1, limit_ms, [](HWND w, UINT, UINT_PTR, DWORD) { DestroyWindow(w); });
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}

}  // namespace rt::shell
