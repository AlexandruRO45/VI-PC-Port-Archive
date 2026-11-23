// Synthetic unit tests for the standalone XAudio2 XMA2 source-voice adapter.
// Pure C++ mock IXAudio2 / IXAudio2SourceVoice / fake Xma2DecoderApi: no audio
// hardware, no Rust crate, no game files.
#include "xaudio2_xma2_adapter.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* expr, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL xaudio2_xma2_adapter_test.cpp:%d: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

// --- Fake XMA2 decoder C ABI with strict live-object & event-order tracking ---

struct FakeDecoder {
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    bool destroyed = false;
};

struct FakePcmBlock {
    std::vector<int16_t> samples;
    size_t reported_sample_count = 0;
    bool return_null_data = false;
    bool destroyed = false;
};

struct FakeDecoderState {
    int32_t next_create_status = LEONIDA_XMA2_OK;
    int32_t next_decode_status = LEONIDA_XMA2_OK;
    size_t override_sample_count = 0;
    bool override_null_pcm_data = false;

    int create_calls = 0;
    int decode_calls = 0;
    int decoder_destroy_calls = 0;
    int pcm_destroy_calls = 0;
    int live_decoders = 0;
    int live_pcm_blocks = 0;
    size_t last_decode_input_len = 0;
    std::vector<std::string> events;
};

FakeDecoderState g_dec{};

void reset_fake_decoder() { g_dec = FakeDecoderState{}; }

int32_t fake_decoder_create(uint32_t sample_rate,
                            uint32_t channels,
                            LeonidaXma2Decoder** out_decoder) {
    ++g_dec.create_calls;
    if (out_decoder == nullptr) {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    *out_decoder = nullptr;
    if (g_dec.next_create_status != LEONIDA_XMA2_OK) {
        return g_dec.next_create_status;
    }
    if (sample_rate == 0 || (channels != 1 && channels != 2)) {
        return LEONIDA_XMA2_INVALID_FORMAT;
    }
    auto* dec = new FakeDecoder{sample_rate, channels, false};
    ++g_dec.live_decoders;
    *out_decoder = reinterpret_cast<LeonidaXma2Decoder*>(dec);
    return LEONIDA_XMA2_OK;
}

int32_t fake_decode_buffer(LeonidaXma2Decoder* decoder,
                           const uint8_t* input,
                           size_t input_len,
                           LeonidaXma2PcmBlock** out_block) {
    ++g_dec.decode_calls;
    g_dec.last_decode_input_len = input_len;
    if (out_block == nullptr) {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    *out_block = nullptr;
    if (decoder == nullptr || input == nullptr) {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    if (input_len == 0 || (input_len % LEONIDA_XMA2_PACKET_BYTES) != 0) {
        return LEONIDA_XMA2_BAD_PACKET_SIZE;
    }
    if (g_dec.next_decode_status != LEONIDA_XMA2_OK) {
        return g_dec.next_decode_status;
    }

    auto* dec = reinterpret_cast<FakeDecoder*>(decoder);
    const size_t packets = input_len / LEONIDA_XMA2_PACKET_BYTES;
    // Emit deterministic non-zero PCM pattern tagged by first byte of each packet.
    const size_t samples_per_packet = 4u * dec->channels;
    auto* block = new FakePcmBlock{};
    block->samples.resize(packets * samples_per_packet);
    for (size_t p = 0; p < packets; ++p) {
        const int16_t tag = static_cast<int16_t>(input[p * LEONIDA_XMA2_PACKET_BYTES] + 100);
        for (size_t i = 0; i < samples_per_packet; ++i) {
            block->samples[p * samples_per_packet + i] = static_cast<int16_t>(tag + i);
        }
    }
    block->reported_sample_count =
        g_dec.override_sample_count != 0 ? g_dec.override_sample_count : block->samples.size();
    block->return_null_data = g_dec.override_null_pcm_data;
    ++g_dec.live_pcm_blocks;
    *out_block = reinterpret_cast<LeonidaXma2PcmBlock*>(block);
    return LEONIDA_XMA2_OK;
}

void fake_decoder_destroy(LeonidaXma2Decoder* decoder) {
    if (decoder == nullptr) {
        return;
    }
    auto* dec = reinterpret_cast<FakeDecoder*>(decoder);
    CHECK(!dec->destroyed);
    dec->destroyed = true;
    ++g_dec.decoder_destroy_calls;
    --g_dec.live_decoders;
    g_dec.events.push_back("decoder_destroy");
    delete dec;
}

const int16_t* fake_pcm_data(const LeonidaXma2PcmBlock* block) {
    if (block == nullptr) {
        return nullptr;
    }
    const auto* b = reinterpret_cast<const FakePcmBlock*>(block);
    if (b->return_null_data) {
        return nullptr;
    }
    return b->samples.data();
}

size_t fake_pcm_sample_count(const LeonidaXma2PcmBlock* block) {
    if (block == nullptr) {
        return 0;
    }
    const auto* b = reinterpret_cast<const FakePcmBlock*>(block);
    return b->reported_sample_count;
}

void fake_pcm_destroy(LeonidaXma2PcmBlock* block) {
    if (block == nullptr) {
        return;
    }
    auto* b = reinterpret_cast<FakePcmBlock*>(block);
    CHECK(!b->destroyed);
    b->destroyed = true;
    ++g_dec.pcm_destroy_calls;
    --g_dec.live_pcm_blocks;
    g_dec.events.push_back("pcm_destroy");
    delete b;
}

rt::xaudio2::Xma2DecoderApi make_fake_api() {
    rt::xaudio2::Xma2DecoderApi api{};
    api.decoder_create = &fake_decoder_create;
    api.decode_buffer = &fake_decode_buffer;
    api.decoder_destroy = &fake_decoder_destroy;
    api.pcm_data = &fake_pcm_data;
    api.pcm_sample_count = &fake_pcm_sample_count;
    api.pcm_destroy = &fake_pcm_destroy;
    return api;
}

// --- Pure C++ mock IXAudio2SourceVoice & IXAudio2 ---

struct SubmittedRecord {
    XAUDIO2_BUFFER buffer{};
    bool had_wma = false;
    std::vector<int16_t> copied_pcm;
};

class MockSourceVoice final : public IXAudio2SourceVoice {
public:
    MockSourceVoice(const WAVEFORMATEX& fmt, IXAudio2VoiceCallback* cb)
        : format(fmt), callback(cb) {}

    WAVEFORMATEX format{};
    IXAudio2VoiceCallback* callback = nullptr;
    HRESULT next_submit_hr = S_OK;
    bool fire_end_on_destroy = false;
    bool destroyed = false;
    int start_calls = 0;
    int stop_calls = 0;
    int flush_calls = 0;
    int discontinuity_calls = 0;
    float volume = 1.0f;
    float frequency_ratio = 1.0f;
    std::vector<SubmittedRecord> submitted;

    void STDMETHODCALLTYPE GetVoiceDetails(XAUDIO2_VOICE_DETAILS* pVoiceDetails) noexcept override {
        if (pVoiceDetails) {
            pVoiceDetails->CreationFlags = 0;
            pVoiceDetails->ActiveFlags = 0;
            pVoiceDetails->InputChannels = format.nChannels;
            pVoiceDetails->InputSampleRate = format.nSamplesPerSec;
        }
    }
    HRESULT STDMETHODCALLTYPE SetOutputVoices(const XAUDIO2_VOICE_SENDS*) noexcept override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetEffectChain(const XAUDIO2_EFFECT_CHAIN*) noexcept override { return S_OK; }
    HRESULT STDMETHODCALLTYPE EnableEffect(UINT32, UINT32) noexcept override { return S_OK; }
    HRESULT STDMETHODCALLTYPE DisableEffect(UINT32, UINT32) noexcept override { return S_OK; }
    void STDMETHODCALLTYPE GetEffectState(UINT32, BOOL* pEnabled) noexcept override {
        if (pEnabled) *pEnabled = FALSE;
    }
    HRESULT STDMETHODCALLTYPE SetEffectParameters(UINT32, const void*, UINT32, UINT32) noexcept override {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetEffectParameters(UINT32, void*, UINT32) noexcept override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetFilterParameters(const XAUDIO2_FILTER_PARAMETERS*,
                                                  UINT32) noexcept override {
        return S_OK;
    }
    void STDMETHODCALLTYPE GetFilterParameters(XAUDIO2_FILTER_PARAMETERS* p) noexcept override {
        if (p) *p = XAUDIO2_FILTER_PARAMETERS{};
    }
    HRESULT STDMETHODCALLTYPE SetOutputFilterParameters(IXAudio2Voice*,
                                                        const XAUDIO2_FILTER_PARAMETERS*,
                                                        UINT32) noexcept override {
        return S_OK;
    }
    void STDMETHODCALLTYPE GetOutputFilterParameters(IXAudio2Voice*,
                                                     XAUDIO2_FILTER_PARAMETERS* p) noexcept override {
        if (p) *p = XAUDIO2_FILTER_PARAMETERS{};
    }
    HRESULT STDMETHODCALLTYPE SetVolume(float Volume, UINT32) noexcept override {
        volume = Volume;
        return S_OK;
    }
    void STDMETHODCALLTYPE GetVolume(float* pVolume) noexcept override {
        if (pVolume) *pVolume = volume;
    }
    HRESULT STDMETHODCALLTYPE SetChannelVolumes(UINT32, const float*, UINT32) noexcept override {
        return S_OK;
    }
    void STDMETHODCALLTYPE GetChannelVolumes(UINT32, float*) noexcept override {}
    HRESULT STDMETHODCALLTYPE SetOutputMatrix(IXAudio2Voice*,
                                              UINT32,
                                              UINT32,
                                              const float*,
                                              UINT32) noexcept override {
        return S_OK;
    }
    void STDMETHODCALLTYPE GetOutputMatrix(IXAudio2Voice*, UINT32, UINT32, float*) noexcept override {}

    void STDMETHODCALLTYPE DestroyVoice() noexcept override {
        g_dec.events.push_back("inner_destroy_voice");
        if (fire_end_on_destroy && callback != nullptr && !submitted.empty()) {
            // Simulate XAudio2 finishing an active buffer callback during DestroyVoice.
            callback->OnBufferEnd(submitted.front().buffer.pContext);
        }
        destroyed = true;
    }

    HRESULT STDMETHODCALLTYPE Start(UINT32, UINT32) noexcept override {
        ++start_calls;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Stop(UINT32, UINT32) noexcept override {
        ++stop_calls;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SubmitSourceBuffer(const XAUDIO2_BUFFER* pBuffer,
                                                 const XAUDIO2_BUFFER_WMA* pBufferWMA) noexcept override {
        if (FAILED(next_submit_hr)) {
            return next_submit_hr;
        }
        SubmittedRecord rec{};
        rec.buffer = *pBuffer;
        rec.had_wma = (pBufferWMA != nullptr);
        if (pBuffer->AudioBytes > 0 && pBuffer->pAudioData != nullptr) {
            const size_t count = pBuffer->AudioBytes / sizeof(int16_t);
            rec.copied_pcm.resize(count);
            std::memcpy(rec.copied_pcm.data(), pBuffer->pAudioData, pBuffer->AudioBytes);
        }
        submitted.push_back(std::move(rec));
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE FlushSourceBuffers() noexcept override {
        ++flush_calls;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Discontinuity() noexcept override {
        ++discontinuity_calls;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE ExitLoop(UINT32) noexcept override { return S_OK; }
    void STDMETHODCALLTYPE GetState(XAUDIO2_VOICE_STATE* pVoiceState, UINT32) noexcept override {
        if (pVoiceState) {
            pVoiceState->BuffersQueued = static_cast<UINT32>(submitted.size());
            pVoiceState->pCurrentBufferContext =
                submitted.empty() ? nullptr : submitted.front().buffer.pContext;
            pVoiceState->SamplesPlayed = 0;
        }
    }
    HRESULT STDMETHODCALLTYPE SetFrequencyRatio(float Ratio, UINT32) noexcept override {
        frequency_ratio = Ratio;
        return S_OK;
    }
    void STDMETHODCALLTYPE GetFrequencyRatio(float* pRatio) noexcept override {
        if (pRatio) *pRatio = frequency_ratio;
    }
    HRESULT STDMETHODCALLTYPE SetSourceSampleRate(UINT32) noexcept override { return S_OK; }
};

class MockXAudio2 final : public IXAudio2 {
public:
    ~MockXAudio2() {
        for (auto* v : owned_voices) {
            delete v;
        }
    }

    std::atomic<ULONG> ref_count{1};
    HRESULT next_create_voice_hr = S_OK;
    int create_source_voice_calls = 0;
    int start_engine_calls = 0;
    int stop_engine_calls = 0;
    int commit_changes_calls = 0;
    std::vector<MockSourceVoice*> owned_voices;

    MockSourceVoice* last_voice() const {
        return owned_voices.empty() ? nullptr : owned_voices.back();
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) noexcept override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (riid == IID_IUnknown || riid == IID_IXAudio2) {
            AddRef();
            *ppv = static_cast<IXAudio2*>(this);
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() noexcept override { return ++ref_count; }
    ULONG STDMETHODCALLTYPE Release() noexcept override { return --ref_count; }

    HRESULT STDMETHODCALLTYPE RegisterForCallbacks(IXAudio2EngineCallback*) noexcept override {
        return S_OK;
    }
    void STDMETHODCALLTYPE UnregisterForCallbacks(IXAudio2EngineCallback*) noexcept override {}

    HRESULT STDMETHODCALLTYPE CreateSourceVoice(IXAudio2SourceVoice** ppSourceVoice,
                                                const WAVEFORMATEX* pSourceFormat,
                                                UINT32,
                                                float,
                                                IXAudio2VoiceCallback* pCallback,
                                                const XAUDIO2_VOICE_SENDS*,
                                                const XAUDIO2_EFFECT_CHAIN*) noexcept override {
        ++create_source_voice_calls;
        if (FAILED(next_create_voice_hr)) {
            return next_create_voice_hr;
        }
        auto* voice = new MockSourceVoice(*pSourceFormat, pCallback);
        owned_voices.push_back(voice);
        *ppSourceVoice = voice;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateSubmixVoice(IXAudio2SubmixVoice**,
                                                UINT32,
                                                UINT32,
                                                UINT32,
                                                UINT32,
                                                const XAUDIO2_VOICE_SENDS*,
                                                const XAUDIO2_EFFECT_CHAIN*) noexcept override {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateMasteringVoice(IXAudio2MasteringVoice**,
                                                   UINT32,
                                                   UINT32,
                                                   UINT32,
                                                   LPCWSTR,
                                                   const XAUDIO2_EFFECT_CHAIN*,
                                                   AUDIO_STREAM_CATEGORY) noexcept override {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE StartEngine() noexcept override {
        ++start_engine_calls;
        return S_OK;
    }
    void STDMETHODCALLTYPE StopEngine() noexcept override { ++stop_engine_calls; }
    HRESULT STDMETHODCALLTYPE CommitChanges(UINT32) noexcept override {
        ++commit_changes_calls;
        return S_OK;
    }
    void STDMETHODCALLTYPE GetPerformanceData(XAUDIO2_PERFORMANCE_DATA* p) noexcept override {
        if (p) *p = XAUDIO2_PERFORMANCE_DATA{};
    }
    void STDMETHODCALLTYPE SetDebugConfiguration(const XAUDIO2_DEBUG_CONFIGURATION*,
                                                 void*) noexcept override {}
};

class RecordingCallback final : public IXAudio2VoiceCallback {
public:
    std::vector<void*> starts;
    std::vector<void*> ends;
    std::vector<void*> loops;
    std::vector<std::pair<void*, HRESULT>> errors;
    int stream_ends = 0;
    int pass_starts = 0;
    int pass_ends = 0;

    void STDMETHODCALLTYPE OnVoiceProcessingPassStart(UINT32) noexcept override { ++pass_starts; }
    void STDMETHODCALLTYPE OnVoiceProcessingPassEnd() noexcept override { ++pass_ends; }
    void STDMETHODCALLTYPE OnStreamEnd() noexcept override { ++stream_ends; }
    void STDMETHODCALLTYPE OnBufferStart(void* ctx) noexcept override { starts.push_back(ctx); }
    void STDMETHODCALLTYPE OnBufferEnd(void* ctx) noexcept override { ends.push_back(ctx); }
    void STDMETHODCALLTYPE OnLoopEnd(void* ctx) noexcept override { loops.push_back(ctx); }
    void STDMETHODCALLTYPE OnVoiceError(void* ctx, HRESULT err) noexcept override {
        errors.emplace_back(ctx, err);
    }
};

WAVEFORMATEX make_pcm_format(uint16_t channels, uint32_t rate) {
    WAVEFORMATEX w{};
    w.wFormatTag = WAVE_FORMAT_PCM;
    w.nChannels = channels;
    w.nSamplesPerSec = rate;
    w.wBitsPerSample = 16;
    w.nBlockAlign = static_cast<WORD>(channels * 2u);
    w.nAvgBytesPerSec = rate * w.nBlockAlign;
    w.cbSize = 0;
    return w;
}

WAVEFORMATEX make_xma2_format(uint16_t channels, uint32_t rate) {
    WAVEFORMATEX w{};
    w.wFormatTag = rt::xaudio2::kWaveFormatXma2;
    w.nChannels = channels;
    w.nSamplesPerSec = rate;
    w.wBitsPerSample = 16;
    w.nBlockAlign = static_cast<WORD>(channels * 2u);
    w.nAvgBytesPerSec = 0;
    w.cbSize = 34;
    return w;
}

WAVEFORMATEXTENSIBLE make_xma2_extensible(uint16_t channels, uint32_t rate, uint16_t cb_size = 22) {
    WAVEFORMATEXTENSIBLE ext{};
    ext.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    ext.Format.nChannels = channels;
    ext.Format.nSamplesPerSec = rate;
    ext.Format.wBitsPerSample = 16;
    ext.Format.nBlockAlign = static_cast<WORD>(channels * 2u);
    ext.Format.cbSize = cb_size;
    ext.Samples.wValidBitsPerSample = 16;
    ext.dwChannelMask = channels == 1 ? 0x4u : 0x3u;
    ext.SubFormat = rt::xaudio2::kSubformatXma2;
    return ext;
}

// --- Test 1: Non-XMA2 formats pass through unwrapped ---
void test_non_xma2_passthrough() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();

    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);
    CHECK(wrapped != nullptr);
    CHECK(inner.ref_count.load() == 2);

    RecordingCallback cb;
    const WAVEFORMATEX pcm = make_pcm_format(2, 48000);
    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &pcm, 0, XAUDIO2_DEFAULT_FREQ_RATIO, &cb) == S_OK);
    CHECK(voice == inner.last_voice());
    CHECK(inner.last_voice()->callback == &cb);
    CHECK(g_dec.create_calls == 0);

    WAVEFORMATEX flt = make_pcm_format(6, 48000);
    flt.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    flt.wBitsPerSample = 32;
    flt.nBlockAlign = 24;
    flt.nAvgBytesPerSec = 48000 * 24;
    IXAudio2SourceVoice* float_voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&float_voice, &flt, 0, XAUDIO2_DEFAULT_FREQ_RATIO, nullptr) ==
          S_OK);
    CHECK(float_voice == inner.last_voice());
    CHECK(g_dec.create_calls == 0);

    // Also verify engine-level COM forwarding and refcounting.
    CHECK(wrapped->StartEngine() == S_OK);
    wrapped->StopEngine();
    CHECK(wrapped->CommitChanges(XAUDIO2_COMMIT_ALL) == S_OK);
    CHECK(inner.start_engine_calls == 1);
    CHECK(inner.stop_engine_calls == 1);
    CHECK(inner.commit_changes_calls == 1);

    IXAudio2* qi_xaudio = nullptr;
    CHECK(wrapped->QueryInterface(IID_IXAudio2, reinterpret_cast<void**>(&qi_xaudio)) == S_OK);
    CHECK(qi_xaudio == wrapped);
    CHECK(qi_xaudio->Release() == 1);
    CHECK(wrapped->Release() == 0);
    CHECK(inner.ref_count.load() == 1);
}

// --- Test 2: XMA2 mono & stereo translation (0x0166 + WAVE_FORMAT_EXTENSIBLE) ---
void test_xma2_format_translation() {
    reset_fake_decoder();

    const WAVEFORMATEX mono = make_xma2_format(1, 24000);
    CHECK(rt::xaudio2::is_xma2_format(&mono));
    WAVEFORMATEX out{};
    CHECK(rt::xaudio2::translate_xma2_to_pcm16(&mono, &out) == S_OK);
    CHECK(out.wFormatTag == WAVE_FORMAT_PCM);
    CHECK(out.nChannels == 1);
    CHECK(out.nSamplesPerSec == 24000);
    CHECK(out.wBitsPerSample == 16);
    CHECK(out.nBlockAlign == 2);
    CHECK(out.nAvgBytesPerSec == 48000);
    CHECK(out.cbSize == 0);

    const WAVEFORMATEXTENSIBLE stereo_ext = make_xma2_extensible(2, 48000, 22);
    CHECK(rt::xaudio2::is_xma2_format(&stereo_ext.Format));
    WAVEFORMATEX out_stereo{};
    CHECK(rt::xaudio2::translate_xma2_to_pcm16(&stereo_ext.Format, &out_stereo) == S_OK);
    CHECK(out_stereo.wFormatTag == WAVE_FORMAT_PCM);
    CHECK(out_stereo.nChannels == 2);
    CHECK(out_stereo.nSamplesPerSec == 48000);
    CHECK(out_stereo.wBitsPerSample == 16);
    CHECK(out_stereo.nBlockAlign == 4);
    CHECK(out_stereo.nAvgBytesPerSec == 192000);
    CHECK(out_stereo.cbSize == 0);

    // WAVE_FORMAT_EXTENSIBLE with truncated cbSize < 22 must NOT match XMA2.
    const WAVEFORMATEXTENSIBLE short_ext = make_xma2_extensible(2, 48000, 20);
    CHECK(!rt::xaudio2::is_xma2_format(&short_ext.Format));

    // Verify CreateSourceVoice passes the translated PCM-16 format to inner IXAudio2.
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &stereo_ext.Format) == S_OK);
    CHECK(voice != nullptr);
    CHECK(voice != inner.last_voice());
    CHECK(inner.last_voice()->format.wFormatTag == WAVE_FORMAT_PCM);
    CHECK(inner.last_voice()->format.nChannels == 2);
    CHECK(inner.last_voice()->format.nSamplesPerSec == 48000);
    CHECK(inner.last_voice()->format.wBitsPerSample == 16);
    CHECK(inner.last_voice()->format.nBlockAlign == 4);
    CHECK(inner.last_voice()->format.nAvgBytesPerSec == 192000);
    CHECK(inner.last_voice()->format.cbSize == 0);

    voice->DestroyVoice();
    CHECK(g_dec.live_decoders == 0);
    CHECK(wrapped->Release() == 0);
}

// --- Test 3: Invalid XMA2 formats & unsupported console Opus 0x704F ---
void test_invalid_formats_and_opus() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    // Wrap parameter validation.
    IXAudio2* bad_wrap = nullptr;
    rt::xaudio2::Xma2DecoderApi incomplete{};
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, nullptr) == E_POINTER);
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(nullptr, &api, &bad_wrap) == E_INVALIDARG);
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, nullptr, &bad_wrap) == E_INVALIDARG);
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &incomplete, &bad_wrap) == E_INVALIDARG);

    IXAudio2SourceVoice* voice = reinterpret_cast<IXAudio2SourceVoice*>(0x1);
    CHECK(wrapped->CreateSourceVoice(&voice, nullptr) == XAUDIO2_E_INVALID_CALL);
    CHECK(voice == nullptr);
    CHECK(wrapped->CreateSourceVoice(nullptr, nullptr) == XAUDIO2_E_INVALID_CALL);

    const WAVEFORMATEX zero_hz = make_xma2_format(2, 0);
    CHECK(wrapped->CreateSourceVoice(&voice, &zero_hz) == XAUDIO2_E_INVALID_CALL);

    const WAVEFORMATEX zero_ch = make_xma2_format(0, 48000);
    CHECK(wrapped->CreateSourceVoice(&voice, &zero_ch) == XAUDIO2_E_INVALID_CALL);

    const WAVEFORMATEX six_ch = make_xma2_format(6, 48000);
    CHECK(wrapped->CreateSourceVoice(&voice, &six_ch) == XAUDIO2_E_INVALID_CALL);

    // Console Opus 0x704F (direct and extensible) fails explicitly with E_NOTIMPL.
    WAVEFORMATEX opus{};
    opus.wFormatTag = rt::xaudio2::kWaveFormatConsoleOpus;
    opus.nChannels = 2;
    opus.nSamplesPerSec = 48000;
    CHECK(rt::xaudio2::is_console_opus_format(&opus));
    CHECK(wrapped->CreateSourceVoice(&voice, &opus) == E_NOTIMPL);

    WAVEFORMATEXTENSIBLE opus_ext = make_xma2_extensible(2, 48000);
    opus_ext.SubFormat = rt::xaudio2::kSubformatConsoleOpus;
    CHECK(rt::xaudio2::is_console_opus_format(&opus_ext.Format));
    CHECK(wrapped->CreateSourceVoice(&voice, &opus_ext.Format) == E_NOTIMPL);

    CHECK(inner.create_source_voice_calls == 0);
    CHECK(g_dec.create_calls == 0);
    CHECK(g_dec.live_decoders == 0);

    // If inner CreateSourceVoice fails after decoder creation, decoder must be destroyed cleanly.
    inner.next_create_voice_hr = E_FAIL;
    const WAVEFORMATEX valid_xma2 = make_xma2_format(2, 48000);
    CHECK(wrapped->CreateSourceVoice(&voice, &valid_xma2) == E_FAIL);
    CHECK(voice == nullptr);
    CHECK(g_dec.create_calls == 1);
    CHECK(g_dec.decoder_destroy_calls == 1);
    CHECK(g_dec.live_decoders == 0);

    CHECK(wrapped->Release() == 0);
}

// --- Test 4: Valid 1-packet & multi-packet SubmitSourceBuffer + callback context mapping ---
void test_valid_submission_and_callbacks() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    RecordingCallback cb;
    const WAVEFORMATEX stereo = make_xma2_format(2, 48000);
    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &stereo, 0, XAUDIO2_DEFAULT_FREQ_RATIO, &cb) == S_OK);
    MockSourceVoice* mock_voice = inner.last_voice();
    CHECK(mock_voice != nullptr);

    // Single 2048-byte packet.
    std::vector<uint8_t> pkt1(LEONIDA_XMA2_PACKET_BYTES, 0);
    pkt1[0] = 7;
    void* ctx1 = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1111));
    XAUDIO2_BUFFER buf1{};
    buf1.AudioBytes = static_cast<UINT32>(pkt1.size());
    buf1.pAudioData = pkt1.data();
    buf1.PlayBegin = 1;
    buf1.PlayLength = 4;
    buf1.pContext = ctx1;

    XAUDIO2_BUFFER_WMA dummy_wma{};
    CHECK(voice->SubmitSourceBuffer(&buf1, &dummy_wma) == S_OK);
    CHECK(g_dec.decode_calls == 1);
    CHECK(g_dec.live_pcm_blocks == 1);
    CHECK(mock_voice->submitted.size() == 1);
    CHECK(!mock_voice->submitted[0].had_wma);
    CHECK(mock_voice->submitted[0].buffer.AudioBytes == 8u * sizeof(int16_t));
    CHECK(mock_voice->submitted[0].buffer.Flags == 0);
    CHECK(mock_voice->submitted[0].buffer.PlayBegin == 1);
    CHECK(mock_voice->submitted[0].buffer.PlayLength == 4);
    CHECK(mock_voice->submitted[0].buffer.pContext != ctx1);
    CHECK(mock_voice->submitted[0].copied_pcm[0] == 107);
    CHECK(mock_voice->submitted[0].copied_pcm[7] == 114);

    // GetState translates pCurrentBufferContext back to caller's context.
    XAUDIO2_VOICE_STATE state{};
    voice->GetState(&state);
    CHECK(state.BuffersQueued == 1);
    CHECK(state.pCurrentBufferContext == ctx1);

    // Multi-packet (3 * 2048 bytes) with XAUDIO2_END_OF_STREAM.
    std::vector<uint8_t> pkt3(LEONIDA_XMA2_PACKET_BYTES * 3u, 0);
    pkt3[0] = 1;
    pkt3[LEONIDA_XMA2_PACKET_BYTES] = 2;
    pkt3[LEONIDA_XMA2_PACKET_BYTES * 2u] = 3;
    void* ctx2 = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2222));
    XAUDIO2_BUFFER buf2{};
    buf2.Flags = XAUDIO2_END_OF_STREAM;
    buf2.AudioBytes = static_cast<UINT32>(pkt3.size());
    buf2.pAudioData = pkt3.data();
    buf2.pContext = ctx2;

    CHECK(voice->SubmitSourceBuffer(&buf2, nullptr) == S_OK);
    CHECK(g_dec.decode_calls == 2);
    CHECK(g_dec.last_decode_input_len == LEONIDA_XMA2_PACKET_BYTES * 3u);
    CHECK(g_dec.live_pcm_blocks == 2);
    CHECK(mock_voice->submitted.size() == 2);
    CHECK(mock_voice->submitted[1].buffer.Flags == XAUDIO2_END_OF_STREAM);
    CHECK(mock_voice->submitted[1].buffer.AudioBytes == 24u * sizeof(int16_t));
    CHECK(mock_voice->submitted[1].copied_pcm[0] == 101);
    CHECK(mock_voice->submitted[1].copied_pcm[8] == 102);
    CHECK(mock_voice->submitted[1].copied_pcm[16] == 103);

    // Fire callbacks on inner voice and verify context round-trip + PCM cleanup.
    void* inner_ctx1 = mock_voice->submitted[0].buffer.pContext;
    void* inner_ctx2 = mock_voice->submitted[1].buffer.pContext;
    mock_voice->callback->OnVoiceProcessingPassStart(64);
    mock_voice->callback->OnBufferStart(inner_ctx1);
    mock_voice->callback->OnLoopEnd(inner_ctx1);
    mock_voice->callback->OnVoiceError(inner_ctx1, XAUDIO2_E_DEVICE_INVALIDATED);
    CHECK(g_dec.live_pcm_blocks == 2);  // Still alive until OnBufferEnd!
    mock_voice->callback->OnBufferEnd(inner_ctx1);
    CHECK(g_dec.live_pcm_blocks == 1);
    mock_voice->callback->OnBufferStart(inner_ctx2);
    mock_voice->callback->OnBufferEnd(inner_ctx2);
    mock_voice->callback->OnStreamEnd();
    mock_voice->callback->OnVoiceProcessingPassEnd();

    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(cb.pass_starts == 1);
    CHECK(cb.pass_ends == 1);
    CHECK(cb.stream_ends == 1);
    CHECK(cb.starts.size() == 2 && cb.starts[0] == ctx1 && cb.starts[1] == ctx2);
    CHECK(cb.loops.size() == 1 && cb.loops[0] == ctx1);
    CHECK(cb.errors.size() == 1 && cb.errors[0].first == ctx1 &&
          cb.errors[0].second == XAUDIO2_E_DEVICE_INVALIDATED);
    CHECK(cb.ends.size() == 2 && cb.ends[0] == ctx1 && cb.ends[1] == ctx2);

    // Also verify voice method forwarding.
    CHECK(voice->Start(0, XAUDIO2_COMMIT_NOW) == S_OK);
    CHECK(voice->Stop(0, XAUDIO2_COMMIT_NOW) == S_OK);
    CHECK(voice->FlushSourceBuffers() == S_OK);
    CHECK(voice->Discontinuity() == S_OK);
    CHECK(voice->SetVolume(0.5f) == S_OK);
    float vol = 0.0f;
    voice->GetVolume(&vol);
    CHECK(vol == 0.5f);
    CHECK(voice->SetFrequencyRatio(1.25f) == S_OK);
    float ratio = 0.0f;
    voice->GetFrequencyRatio(&ratio);
    CHECK(ratio == 1.25f);
    CHECK(mock_voice->start_calls == 1);
    CHECK(mock_voice->stop_calls == 1);
    CHECK(mock_voice->flush_calls == 1);
    CHECK(mock_voice->discontinuity_calls == 1);

    voice->DestroyVoice();
    CHECK(g_dec.live_decoders == 0);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(wrapped->Release() == 0);
}

// --- Test 5: Null caller callback still frees decoded PCM blocks on OnBufferEnd ---
void test_null_caller_callback() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    const WAVEFORMATEX mono = make_xma2_format(1, 48000);
    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &mono, 0, XAUDIO2_DEFAULT_FREQ_RATIO, nullptr) ==
          S_OK);
    MockSourceVoice* mock_voice = inner.last_voice();
    CHECK(mock_voice->callback != nullptr);

    std::vector<uint8_t> pkt(LEONIDA_XMA2_PACKET_BYTES, 0x22);
    XAUDIO2_BUFFER buf{};
    buf.AudioBytes = static_cast<UINT32>(pkt.size());
    buf.pAudioData = pkt.data();
    CHECK(voice->SubmitSourceBuffer(&buf, nullptr) == S_OK);
    CHECK(g_dec.live_pcm_blocks == 1);

    mock_voice->callback->OnBufferStart(mock_voice->submitted[0].buffer.pContext);
    mock_voice->callback->OnBufferEnd(mock_voice->submitted[0].buffer.pContext);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(g_dec.pcm_destroy_calls == 1);

    voice->DestroyVoice();
    CHECK(g_dec.live_decoders == 0);
    CHECK(wrapped->Release() == 0);
}

// --- Test 6: Bad packet size, decoder errors, OOM, overflow, and inner submit failure ---
void test_error_paths_no_leaks() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    const WAVEFORMATEX stereo = make_xma2_format(2, 48000);
    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &stereo) == S_OK);
    MockSourceVoice* mock_voice = inner.last_voice();

    // Null buffer / null data / non-multiple of 2048 / 0-byte without EOS.
    CHECK(voice->SubmitSourceBuffer(nullptr, nullptr) == XAUDIO2_E_INVALID_CALL);

    XAUDIO2_BUFFER null_data{};
    null_data.AudioBytes = LEONIDA_XMA2_PACKET_BYTES;
    null_data.pAudioData = nullptr;
    CHECK(voice->SubmitSourceBuffer(&null_data, nullptr) == XAUDIO2_E_INVALID_CALL);

    std::vector<uint8_t> bad_len_data(LEONIDA_XMA2_PACKET_BYTES + 1u, 0);
    XAUDIO2_BUFFER bad_len{};
    bad_len.AudioBytes = static_cast<UINT32>(bad_len_data.size());
    bad_len.pAudioData = bad_len_data.data();
    CHECK(voice->SubmitSourceBuffer(&bad_len, nullptr) == XAUDIO2_E_INVALID_CALL);

    XAUDIO2_BUFFER zero_no_eos{};
    zero_no_eos.AudioBytes = 0;
    zero_no_eos.Flags = 0;
    CHECK(voice->SubmitSourceBuffer(&zero_no_eos, nullptr) == XAUDIO2_E_INVALID_CALL);
    CHECK(g_dec.decode_calls == 0);
    CHECK(mock_voice->submitted.empty());

    std::vector<uint8_t> pkt(LEONIDA_XMA2_PACKET_BYTES, 0);
    XAUDIO2_BUFFER valid_buf{};
    valid_buf.AudioBytes = static_cast<UINT32>(pkt.size());
    valid_buf.pAudioData = pkt.data();

    // Decoder failure.
    g_dec.next_decode_status = LEONIDA_XMA2_DECODER_FAILURE;
    CHECK(voice->SubmitSourceBuffer(&valid_buf, nullptr) == XAUDIO2_E_XMA_DECODER_ERROR);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(mock_voice->submitted.empty());

    // Decoder OOM.
    g_dec.next_decode_status = LEONIDA_XMA2_OUT_OF_MEMORY;
    CHECK(voice->SubmitSourceBuffer(&valid_buf, nullptr) == E_OUTOFMEMORY);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(mock_voice->submitted.empty());

    // Sample-count byte overflow (> XAUDIO2_MAX_BUFFER_BYTES / sizeof(int16_t)).
    g_dec.next_decode_status = LEONIDA_XMA2_OK;
    g_dec.override_sample_count =
        (static_cast<size_t>(XAUDIO2_MAX_BUFFER_BYTES) / sizeof(int16_t)) + 1u;
    CHECK(voice->SubmitSourceBuffer(&valid_buf, nullptr) == XAUDIO2_E_INVALID_CALL);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(mock_voice->submitted.empty());
    g_dec.override_sample_count = 0;

    // Inner voice submission failure must destroy the decoded PCM block and leave 0 leaks.
    mock_voice->next_submit_hr = XAUDIO2_E_INVALID_CALL;
    CHECK(voice->SubmitSourceBuffer(&valid_buf, nullptr) == XAUDIO2_E_INVALID_CALL);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(mock_voice->submitted.empty());
    mock_voice->next_submit_hr = S_OK;

    voice->DestroyVoice();
    CHECK(g_dec.live_decoders == 0);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(wrapped->Release() == 0);
}

// --- Test 7: DestroyVoice destroys inner voice FIRST, frees all in-flight PCM blocks & decoder ---
void test_destroy_voice_ordering_and_in_flight_cleanup() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    RecordingCallback cb;
    const WAVEFORMATEX stereo = make_xma2_format(2, 48000);
    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &stereo, 0, XAUDIO2_DEFAULT_FREQ_RATIO, &cb) == S_OK);
    MockSourceVoice* mock_voice = inner.last_voice();
    mock_voice->fire_end_on_destroy = true;

    std::vector<uint8_t> pkt(LEONIDA_XMA2_PACKET_BYTES, 0x55);
    for (uintptr_t i = 1; i <= 3; ++i) {
        XAUDIO2_BUFFER buf{};
        buf.AudioBytes = static_cast<UINT32>(pkt.size());
        buf.pAudioData = pkt.data();
        buf.pContext = reinterpret_cast<void*>(i);
        CHECK(voice->SubmitSourceBuffer(&buf, nullptr) == S_OK);
    }
    CHECK(g_dec.live_pcm_blocks == 3);
    CHECK(g_dec.live_decoders == 1);

    g_dec.events.clear();
    voice->DestroyVoice();

    CHECK(mock_voice->destroyed);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(g_dec.live_decoders == 0);
    CHECK(g_dec.pcm_destroy_calls == 3);
    CHECK(g_dec.decoder_destroy_calls == 1);
    // Callback fired during inner->DestroyVoice saw translated caller context 1.
    CHECK(cb.ends.size() == 1 && cb.ends[0] == reinterpret_cast<void*>(static_cast<uintptr_t>(1)));
    // Exact destruction order: inner voice first, then PCM blocks, then decoder last.
    CHECK(g_dec.events.size() == 5);
    if (g_dec.events.size() == 5) {
        CHECK(g_dec.events[0] == "inner_destroy_voice");
        CHECK(g_dec.events[1] == "pcm_destroy");
        CHECK(g_dec.events[2] == "pcm_destroy");
        CHECK(g_dec.events[3] == "pcm_destroy");
        CHECK(g_dec.events[4] == "decoder_destroy");
    }

    CHECK(wrapped->Release() == 0);
}

// --- Test 8: 0-byte XAUDIO2_END_OF_STREAM marker forwards without calling decode_buffer ---
void test_zero_byte_end_of_stream_marker() {
    reset_fake_decoder();
    MockXAudio2 inner;
    const auto api = make_fake_api();
    IXAudio2* wrapped = nullptr;
    CHECK(rt::xaudio2::wrap_xaudio2_for_xma2(&inner, &api, &wrapped) == S_OK);

    RecordingCallback cb;
    const WAVEFORMATEX mono = make_xma2_format(1, 48000);
    IXAudio2SourceVoice* voice = nullptr;
    CHECK(wrapped->CreateSourceVoice(&voice, &mono, 0, XAUDIO2_DEFAULT_FREQ_RATIO, &cb) == S_OK);
    MockSourceVoice* mock_voice = inner.last_voice();

    void* eos_ctx = reinterpret_cast<void*>(static_cast<uintptr_t>(0xE05));
    XAUDIO2_BUFFER eos{};
    eos.Flags = XAUDIO2_END_OF_STREAM;
    eos.AudioBytes = 0;
    eos.pAudioData = nullptr;
    eos.pContext = eos_ctx;

    CHECK(voice->SubmitSourceBuffer(&eos, nullptr) == S_OK);
    CHECK(g_dec.decode_calls == 0);
    CHECK(g_dec.live_pcm_blocks == 0);
    CHECK(mock_voice->submitted.size() == 1);
    CHECK(mock_voice->submitted[0].buffer.Flags == XAUDIO2_END_OF_STREAM);
    CHECK(mock_voice->submitted[0].buffer.AudioBytes == 0);
    CHECK(mock_voice->submitted[0].buffer.pAudioData == nullptr);

    mock_voice->callback->OnBufferStart(mock_voice->submitted[0].buffer.pContext);
    mock_voice->callback->OnBufferEnd(mock_voice->submitted[0].buffer.pContext);
    CHECK(cb.starts.size() == 1 && cb.starts[0] == eos_ctx);
    CHECK(cb.ends.size() == 1 && cb.ends[0] == eos_ctx);

    voice->DestroyVoice();
    CHECK(g_dec.live_decoders == 0);
    CHECK(wrapped->Release() == 0);
}

// --- Test 9: create_xaudio2_with_com_retry ---
struct ComRetryState {
    HRESULT first_hr = S_OK;
    HRESULT second_hr = S_OK;
    HRESULT co_init_hr = S_OK;
    int create_calls = 0;
    int co_init_calls = 0;
    DWORD last_co_init_flags = 0;
    IXAudio2* fake_engine = nullptr;
};

ComRetryState g_com{};

HRESULT STDAPICALLTYPE mock_xaudio2_create(IXAudio2** out, UINT32, XAUDIO2_PROCESSOR) {
    ++g_com.create_calls;
    const HRESULT hr = (g_com.create_calls == 1) ? g_com.first_hr : g_com.second_hr;
    if (SUCCEEDED(hr) && out != nullptr) {
        *out = g_com.fake_engine;
    }
    return hr;
}

HRESULT WINAPI mock_co_initialize_ex(LPVOID pvReserved, DWORD dwCoInit) {
    CHECK(pvReserved == nullptr);
    ++g_com.co_init_calls;
    g_com.last_co_init_flags = dwCoInit;
    return g_com.co_init_hr;
}

void test_com_retry_helper() {
    MockXAudio2 dummy_engine;
    IXAudio2* out = nullptr;

    // 1. Retries once on CO_E_NOTINITIALIZED after CoInitializeEx(nullptr, COINIT_MULTITHREADED).
    g_com = ComRetryState{};
    g_com.first_hr = CO_E_NOTINITIALIZED;
    g_com.second_hr = S_OK;
    g_com.co_init_hr = S_OK;
    g_com.fake_engine = &dummy_engine;
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              &mock_xaudio2_create, &out, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) ==
          S_OK);
    CHECK(out == &dummy_engine);
    CHECK(g_com.create_calls == 2);
    CHECK(g_com.co_init_calls == 1);
    CHECK(g_com.last_co_init_flags == COINIT_MULTITHREADED);

    // 2. Does NOT retry on S_OK.
    g_com = ComRetryState{};
    g_com.first_hr = S_OK;
    g_com.fake_engine = &dummy_engine;
    out = nullptr;
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              &mock_xaudio2_create, &out, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) ==
          S_OK);
    CHECK(out == &dummy_engine);
    CHECK(g_com.create_calls == 1);
    CHECK(g_com.co_init_calls == 0);

    // 3. Does NOT retry on non-COM errors (e.g. XAUDIO2_E_INVALID_CALL).
    g_com = ComRetryState{};
    g_com.first_hr = XAUDIO2_E_INVALID_CALL;
    out = &dummy_engine;
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              &mock_xaudio2_create, &out, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) ==
          XAUDIO2_E_INVALID_CALL);
    CHECK(out == nullptr);
    CHECK(g_com.create_calls == 1);
    CHECK(g_com.co_init_calls == 0);

    // 4. Retries at most once even if retry also returns CO_E_NOTINITIALIZED.
    g_com = ComRetryState{};
    g_com.first_hr = CO_E_NOTINITIALIZED;
    g_com.second_hr = CO_E_NOTINITIALIZED;
    g_com.co_init_hr = S_OK;
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              &mock_xaudio2_create, &out, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) ==
          CO_E_NOTINITIALIZED);
    CHECK(g_com.create_calls == 2);
    CHECK(g_com.co_init_calls == 1);

    // 5. Fails without retrying if CoInitializeEx itself fails (other than RPC_E_CHANGED_MODE).
    g_com = ComRetryState{};
    g_com.first_hr = CO_E_NOTINITIALIZED;
    g_com.co_init_hr = E_OUTOFMEMORY;
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              &mock_xaudio2_create, &out, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) ==
          E_OUTOFMEMORY);
    CHECK(g_com.create_calls == 1);
    CHECK(g_com.co_init_calls == 1);

    // 6. Null arguments rejected cleanly.
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              nullptr, &out, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) == E_INVALIDARG);
    CHECK(rt::xaudio2::create_xaudio2_with_com_retry(
              &mock_xaudio2_create, nullptr, 0, XAUDIO2_DEFAULT_PROCESSOR, &mock_co_initialize_ex) ==
          E_INVALIDARG);
}

}  // namespace

int main() {
    test_non_xma2_passthrough();
    test_xma2_format_translation();
    test_invalid_formats_and_opus();
    test_valid_submission_and_callbacks();
    test_null_caller_callback();
    test_error_paths_no_leaks();
    test_destroy_voice_ordering_and_in_flight_cleanup();
    test_zero_byte_end_of_stream_marker();
    test_com_retry_helper();

    if (g_failures != 0) {
        std::fprintf(stderr, "xaudio2_xma2_adapter_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("xaudio2_xma2_adapter_test: PASS (all 9 groups)\n");
    return 0;
}
