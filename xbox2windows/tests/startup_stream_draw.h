#pragma once

#include <cstdint>
#include <dxgiformat.h>

struct IDXGIAdapter;

struct StartupBc7Input {
    const char* blocks;
    std::uint32_t width, height;
    bool tiled;
};

int check_stream_draw(IDXGIAdapter* adapter, void* guest, void* pso, const char* pixels,
                      const StartupBc7Input* bc7 = nullptr, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
                      std::uint32_t export_word = 0);
