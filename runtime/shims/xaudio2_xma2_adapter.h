// Standalone XAudio2 XMA2 source-voice adapter: wraps a desktop Windows IXAudio2
// instance so console WAVE_FORMAT_XMA2 (0x0166) source voices decode 2048-byte
// XMA2 packet buffers to interleaved 16-bit PCM via the leonida_xma2_decoder C ABI.
#pragma once

#include <windows.h>
#include <mmreg.h>
#include <xaudio2.h>

#include <cstddef>
#include <cstdint>

#include "../xma_decoder/leonida_xma2_decoder.h"

namespace rt::xaudio2 {

inline constexpr uint16_t kWaveFormatXma2 = 0x0166u;
inline constexpr uint16_t kWaveFormatConsoleOpus = 0x704Fu;

// Standard KSDATAFORMAT_SUBTYPE wave-format GUID layout:
// {0000xxxx-0000-0010-8000-00aa00389b71}
inline constexpr GUID kSubformatXma2 = {
    0x00000166u,
    0x0000u,
    0x0010u,
    {0x80u, 0x00u, 0x00u, 0xAAu, 0x00u, 0x38u, 0x9Bu, 0x71u},
};

inline constexpr GUID kSubformatConsoleOpus = {
    0x0000704Fu,
    0x0000u,
    0x0010u,
    {0x80u, 0x00u, 0x00u, 0xAAu, 0x00u, 0x38u, 0x9Bu, 0x71u},
};

// Injectable function-pointer table matching runtime/xma_decoder/leonida_xma2_decoder.h 1:1.
struct Xma2DecoderApi {
    int32_t (*decoder_create)(uint32_t sample_rate, uint32_t channels,
                              LeonidaXma2Decoder** out_decoder) = nullptr;
    int32_t (*decode_buffer)(LeonidaXma2Decoder* decoder,
                             const uint8_t* input, size_t input_len,
                             LeonidaXma2PcmBlock** out_block) = nullptr;
    void (*decoder_destroy)(LeonidaXma2Decoder* decoder) = nullptr;
    const int16_t* (*pcm_data)(const LeonidaXma2PcmBlock* block) = nullptr;
    size_t (*pcm_sample_count)(const LeonidaXma2PcmBlock* block) = nullptr;
    void (*pcm_destroy)(LeonidaXma2PcmBlock* block) = nullptr;
};

bool is_complete_decoder_api(const Xma2DecoderApi* api) noexcept;

#if defined(LEONIDA_HAS_XMA2_DECODER)
// Returns a static table wired directly to the linked leonida_xma2_decoder symbols.
const Xma2DecoderApi& default_xma2_decoder_api() noexcept;
#endif

// Returns true when format describes WAVE_FORMAT_XMA2 (0x0166), either directly
// or inside a WAVEFORMATEXTENSIBLE (cbSize >= 22) with kSubformatXma2.
bool is_xma2_format(const WAVEFORMATEX* format) noexcept;

// Returns true when format describes the console Opus voice format (0x704F),
// either directly or inside a WAVEFORMATEXTENSIBLE (cbSize >= 22).
bool is_console_opus_format(const WAVEFORMATEX* format) noexcept;

// Translates a mono or stereo XMA2 format descriptor to a 16-bit WAVE_FORMAT_PCM
// descriptor accepted by desktop Windows xaudio2_9.dll.
// Rejects null pointers, non-XMA2 formats, zero sample rates, and channel counts
// outside {1, 2} with XAUDIO2_E_INVALID_CALL.
HRESULT translate_xma2_to_pcm16(const WAVEFORMATEX* source, WAVEFORMATEX* out_pcm) noexcept;

// Maps a LeonidaXma2Status code to the corresponding XAudio2/COM HRESULT.
HRESULT map_xma2_status(int32_t status) noexcept;

using XAudio2CreateFn = HRESULT(STDAPICALLTYPE*)(IXAudio2** ppXAudio2, UINT32 Flags,
                                                 XAUDIO2_PROCESSOR XAudio2Processor);
using CoInitializeExFn = HRESULT(WINAPI*)(LPVOID pvReserved, DWORD dwCoInit);

// Calls create_fn; if it fails with CO_E_NOTINITIALIZED, initializes COM on the
// calling thread with COINIT_MULTITHREADED (via co_init_fn or ::CoInitializeEx)
// and retries create_fn once.
HRESULT create_xaudio2_with_com_retry(
    XAudio2CreateFn create_fn,
    IXAudio2** out_engine,
    UINT32 flags = 0,
    XAUDIO2_PROCESSOR processor = XAUDIO2_DEFAULT_PROCESSOR,
    CoInitializeExFn co_init_fn = nullptr) noexcept;

// Wraps an existing IXAudio2 engine with a COM-refcounted proxy that intercepts
// XMA2 source voices and passes non-XMA2 source voices straight through.
// Takes its own AddRef on `inner` (does not steal the caller's reference) and
// releases it when the wrapper's reference count reaches zero.
HRESULT wrap_xaudio2_for_xma2(IXAudio2* inner,
                              const Xma2DecoderApi* api,
                              IXAudio2** out_wrapped) noexcept;

}  // namespace rt::xaudio2
