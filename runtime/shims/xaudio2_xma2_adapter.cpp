#include "xaudio2_xma2_adapter.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace rt::xaudio2 {
namespace {

static_assert(sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX) == 22,
              "WAVEFORMATEXTENSIBLE must extend WAVEFORMATEX by 22 bytes");

std::atomic<bool> g_opus_logged{false};

class SrwExclusiveGuard {
public:
    explicit SrwExclusiveGuard(SRWLOCK& lock) noexcept : lock_(lock) {
        AcquireSRWLockExclusive(&lock_);
    }
    ~SrwExclusiveGuard() { ReleaseSRWLockExclusive(&lock_); }
    SrwExclusiveGuard(const SrwExclusiveGuard&) = delete;
    SrwExclusiveGuard& operator=(const SrwExclusiveGuard&) = delete;

private:
    SRWLOCK& lock_;
};

bool has_extensible_subformat(const WAVEFORMATEX* format, const GUID& expected) noexcept {
    if (!format || format->wFormatTag != WAVE_FORMAT_EXTENSIBLE || format->cbSize < 22) {
        return false;
    }
    WAVEFORMATEXTENSIBLE ext{};
    std::memcpy(&ext, format, sizeof(ext));
    return std::memcmp(&ext.SubFormat, &expected, sizeof(GUID)) == 0;
}

class Xma2SourceVoiceProxy final : public IXAudio2SourceVoice {
public:
    Xma2SourceVoiceProxy(const Xma2DecoderApi& api,
                         LeonidaXma2Decoder* decoder,
                         IXAudio2VoiceCallback* caller_callback) noexcept
        : api_(api),
          decoder_(decoder),
          caller_callback_(caller_callback),
          callback_proxy_(this) {}

    ~Xma2SourceVoiceProxy() {
        drain_in_flight_blocks();
        if (decoder_ != nullptr) {
            api_.decoder_destroy(decoder_);
            decoder_ = nullptr;
        }
    }

    Xma2SourceVoiceProxy(const Xma2SourceVoiceProxy&) = delete;
    Xma2SourceVoiceProxy& operator=(const Xma2SourceVoiceProxy&) = delete;

    IXAudio2VoiceCallback* callback() noexcept { return &callback_proxy_; }

    void attach_inner_voice(IXAudio2SourceVoice* inner) noexcept { inner_voice_ = inner; }

    // --- IXAudio2Voice methods ---

    void STDMETHODCALLTYPE GetVoiceDetails(XAUDIO2_VOICE_DETAILS* pVoiceDetails) noexcept override {
        inner_voice_->GetVoiceDetails(pVoiceDetails);
    }

    HRESULT STDMETHODCALLTYPE SetOutputVoices(const XAUDIO2_VOICE_SENDS* pSendList) noexcept override {
        return inner_voice_->SetOutputVoices(pSendList);
    }

    HRESULT STDMETHODCALLTYPE SetEffectChain(const XAUDIO2_EFFECT_CHAIN* pEffectChain) noexcept override {
        return inner_voice_->SetEffectChain(pEffectChain);
    }

    HRESULT STDMETHODCALLTYPE EnableEffect(UINT32 EffectIndex, UINT32 OperationSet) noexcept override {
        return inner_voice_->EnableEffect(EffectIndex, OperationSet);
    }

    HRESULT STDMETHODCALLTYPE DisableEffect(UINT32 EffectIndex, UINT32 OperationSet) noexcept override {
        return inner_voice_->DisableEffect(EffectIndex, OperationSet);
    }

    void STDMETHODCALLTYPE GetEffectState(UINT32 EffectIndex, BOOL* pEnabled) noexcept override {
        inner_voice_->GetEffectState(EffectIndex, pEnabled);
    }

    HRESULT STDMETHODCALLTYPE SetEffectParameters(UINT32 EffectIndex,
                                                  const void* pParameters,
                                                  UINT32 ParametersByteSize,
                                                  UINT32 OperationSet) noexcept override {
        return inner_voice_->SetEffectParameters(
            EffectIndex, pParameters, ParametersByteSize, OperationSet);
    }

    HRESULT STDMETHODCALLTYPE GetEffectParameters(UINT32 EffectIndex,
                                                  void* pParameters,
                                                  UINT32 ParametersByteSize) noexcept override {
        return inner_voice_->GetEffectParameters(EffectIndex, pParameters, ParametersByteSize);
    }

    HRESULT STDMETHODCALLTYPE SetFilterParameters(const XAUDIO2_FILTER_PARAMETERS* pParameters,
                                                  UINT32 OperationSet) noexcept override {
        return inner_voice_->SetFilterParameters(pParameters, OperationSet);
    }

    void STDMETHODCALLTYPE GetFilterParameters(XAUDIO2_FILTER_PARAMETERS* pParameters) noexcept override {
        inner_voice_->GetFilterParameters(pParameters);
    }

    HRESULT STDMETHODCALLTYPE SetOutputFilterParameters(
        IXAudio2Voice* pDestinationVoice,
        const XAUDIO2_FILTER_PARAMETERS* pParameters,
        UINT32 OperationSet) noexcept override {
        return inner_voice_->SetOutputFilterParameters(
            pDestinationVoice, pParameters, OperationSet);
    }

    void STDMETHODCALLTYPE GetOutputFilterParameters(
        IXAudio2Voice* pDestinationVoice,
        XAUDIO2_FILTER_PARAMETERS* pParameters) noexcept override {
        inner_voice_->GetOutputFilterParameters(pDestinationVoice, pParameters);
    }

    HRESULT STDMETHODCALLTYPE SetVolume(float Volume, UINT32 OperationSet) noexcept override {
        return inner_voice_->SetVolume(Volume, OperationSet);
    }

    void STDMETHODCALLTYPE GetVolume(float* pVolume) noexcept override {
        inner_voice_->GetVolume(pVolume);
    }

    HRESULT STDMETHODCALLTYPE SetChannelVolumes(UINT32 Channels,
                                                const float* pVolumes,
                                                UINT32 OperationSet) noexcept override {
        return inner_voice_->SetChannelVolumes(Channels, pVolumes, OperationSet);
    }

    void STDMETHODCALLTYPE GetChannelVolumes(UINT32 Channels, float* pVolumes) noexcept override {
        inner_voice_->GetChannelVolumes(Channels, pVolumes);
    }

    HRESULT STDMETHODCALLTYPE SetOutputMatrix(IXAudio2Voice* pDestinationVoice,
                                              UINT32 SourceChannels,
                                              UINT32 DestinationChannels,
                                              const float* pLevelMatrix,
                                              UINT32 OperationSet) noexcept override {
        return inner_voice_->SetOutputMatrix(
            pDestinationVoice, SourceChannels, DestinationChannels, pLevelMatrix, OperationSet);
    }

    void STDMETHODCALLTYPE GetOutputMatrix(IXAudio2Voice* pDestinationVoice,
                                           UINT32 SourceChannels,
                                           UINT32 DestinationChannels,
                                           float* pLevelMatrix) noexcept override {
        inner_voice_->GetOutputMatrix(
            pDestinationVoice, SourceChannels, DestinationChannels, pLevelMatrix);
    }

    void STDMETHODCALLTYPE DestroyVoice() noexcept override {
        // Destroy the inner voice first so XAudio2 drains any active audio-thread callback
        // and guarantees no further callbacks can fire before tearing down our state.
        IXAudio2SourceVoice* inner = inner_voice_;
        inner_voice_ = nullptr;
        if (inner != nullptr) {
            inner->DestroyVoice();
        }
        delete this;
    }

    // --- IXAudio2SourceVoice methods ---

    HRESULT STDMETHODCALLTYPE Start(UINT32 Flags, UINT32 OperationSet) noexcept override {
        return inner_voice_->Start(Flags, OperationSet);
    }

    HRESULT STDMETHODCALLTYPE Stop(UINT32 Flags, UINT32 OperationSet) noexcept override {
        return inner_voice_->Stop(Flags, OperationSet);
    }

    HRESULT STDMETHODCALLTYPE SubmitSourceBuffer(const XAUDIO2_BUFFER* pBuffer,
                                                 const XAUDIO2_BUFFER_WMA* pBufferWMA) noexcept override {
        (void)pBufferWMA;
        if (pBuffer == nullptr) {
            return XAUDIO2_E_INVALID_CALL;
        }

        if (pBuffer->AudioBytes == 0) {
            if ((pBuffer->Flags & XAUDIO2_END_OF_STREAM) == 0) {
                return XAUDIO2_E_INVALID_CALL;
            }
            auto* block = new (std::nothrow) SubmittedBlock{nullptr, pBuffer->pContext};
            if (block == nullptr) {
                return E_OUTOFMEMORY;
            }
            push_block(block);

            XAUDIO2_BUFFER forwarded = *pBuffer;
            forwarded.AudioBytes = 0;
            forwarded.pAudioData = nullptr;
            forwarded.pContext = block;

            const HRESULT hr = inner_voice_->SubmitSourceBuffer(&forwarded, nullptr);
            if (FAILED(hr)) {
                if (remove_block(block)) {
                    delete block;
                }
            }
            return hr;
        }

        if (pBuffer->pAudioData == nullptr ||
            (pBuffer->AudioBytes % LEONIDA_XMA2_PACKET_BYTES) != 0) {
            return XAUDIO2_E_INVALID_CALL;
        }

        LeonidaXma2PcmBlock* pcm = nullptr;
        const int32_t status =
            api_.decode_buffer(decoder_, pBuffer->pAudioData, pBuffer->AudioBytes, &pcm);
        if (status != LEONIDA_XMA2_OK) {
            if (pcm != nullptr) {
                api_.pcm_destroy(pcm);
            }
            return map_xma2_status(status);
        }
        if (pcm == nullptr) {
            return XAUDIO2_E_XMA_DECODER_ERROR;
        }

        const size_t sample_count = api_.pcm_sample_count(pcm);
        const int16_t* samples = api_.pcm_data(pcm);
        constexpr size_t kMaxPcmSamples =
            static_cast<size_t>(XAUDIO2_MAX_BUFFER_BYTES) / sizeof(int16_t);
        if (sample_count > kMaxPcmSamples) {
            api_.pcm_destroy(pcm);
            return XAUDIO2_E_INVALID_CALL;
        }

        const uint32_t pcm_bytes = static_cast<uint32_t>(sample_count * sizeof(int16_t));
        if ((pcm_bytes > 0 && samples == nullptr) ||
            (pcm_bytes == 0 && (pBuffer->Flags & XAUDIO2_END_OF_STREAM) == 0)) {
            api_.pcm_destroy(pcm);
            return XAUDIO2_E_XMA_DECODER_ERROR;
        }

        auto* block = new (std::nothrow) SubmittedBlock{pcm, pBuffer->pContext};
        if (block == nullptr) {
            api_.pcm_destroy(pcm);
            return E_OUTOFMEMORY;
        }
        push_block(block);

        XAUDIO2_BUFFER pcm_buffer = *pBuffer;
        pcm_buffer.AudioBytes = pcm_bytes;
        pcm_buffer.pAudioData =
            pcm_bytes > 0 ? reinterpret_cast<const BYTE*>(samples) : nullptr;
        pcm_buffer.pContext = block;

        const HRESULT hr = inner_voice_->SubmitSourceBuffer(&pcm_buffer, nullptr);
        if (FAILED(hr)) {
            if (remove_block(block)) {
                api_.pcm_destroy(block->pcm);
                delete block;
            }
        }
        return hr;
    }

    HRESULT STDMETHODCALLTYPE FlushSourceBuffers() noexcept override {
        return inner_voice_->FlushSourceBuffers();
    }

    HRESULT STDMETHODCALLTYPE Discontinuity() noexcept override {
        return inner_voice_->Discontinuity();
    }

    HRESULT STDMETHODCALLTYPE ExitLoop(UINT32 OperationSet) noexcept override {
        return inner_voice_->ExitLoop(OperationSet);
    }

    void STDMETHODCALLTYPE GetState(XAUDIO2_VOICE_STATE* pVoiceState, UINT32 Flags) noexcept override {
        inner_voice_->GetState(pVoiceState, Flags);
        if (pVoiceState != nullptr && pVoiceState->pCurrentBufferContext != nullptr) {
            pVoiceState->pCurrentBufferContext =
                translate_context(pVoiceState->pCurrentBufferContext);
        }
    }

    HRESULT STDMETHODCALLTYPE SetFrequencyRatio(float Ratio, UINT32 OperationSet) noexcept override {
        return inner_voice_->SetFrequencyRatio(Ratio, OperationSet);
    }

    void STDMETHODCALLTYPE GetFrequencyRatio(float* pRatio) noexcept override {
        inner_voice_->GetFrequencyRatio(pRatio);
    }

    HRESULT STDMETHODCALLTYPE SetSourceSampleRate(UINT32 NewSourceSampleRate) noexcept override {
        return inner_voice_->SetSourceSampleRate(NewSourceSampleRate);
    }

private:
    struct SubmittedBlock {
        LeonidaXma2PcmBlock* pcm = nullptr;
        void* caller_context = nullptr;
        SubmittedBlock* prev = nullptr;
        SubmittedBlock* next = nullptr;
        bool linked = false;
    };

    class CallbackProxy final : public IXAudio2VoiceCallback {
    public:
        explicit CallbackProxy(Xma2SourceVoiceProxy* owner) noexcept : owner_(owner) {}

        void STDMETHODCALLTYPE OnVoiceProcessingPassStart(UINT32 BytesRequired) noexcept override {
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnVoiceProcessingPassStart(BytesRequired);
            }
        }

        void STDMETHODCALLTYPE OnVoiceProcessingPassEnd() noexcept override {
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnVoiceProcessingPassEnd();
            }
        }

        void STDMETHODCALLTYPE OnStreamEnd() noexcept override {
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnStreamEnd();
            }
        }

        void STDMETHODCALLTYPE OnBufferStart(void* pBufferContext) noexcept override {
            void* caller_ctx = owner_->translate_context(pBufferContext);
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnBufferStart(caller_ctx);
            }
        }

        void STDMETHODCALLTYPE OnBufferEnd(void* pBufferContext) noexcept override {
            void* caller_ctx = owner_->complete_block(pBufferContext);
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnBufferEnd(caller_ctx);
            }
        }

        void STDMETHODCALLTYPE OnLoopEnd(void* pBufferContext) noexcept override {
            void* caller_ctx = owner_->translate_context(pBufferContext);
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnLoopEnd(caller_ctx);
            }
        }

        void STDMETHODCALLTYPE OnVoiceError(void* pBufferContext, HRESULT Error) noexcept override {
            void* caller_ctx = owner_->translate_context(pBufferContext);
            if (owner_->caller_callback_ != nullptr) {
                owner_->caller_callback_->OnVoiceError(caller_ctx, Error);
            }
        }

    private:
        Xma2SourceVoiceProxy* owner_;
    };

    void push_block(SubmittedBlock* block) noexcept {
        SrwExclusiveGuard guard(blocks_lock_);
        block->prev = tail_;
        block->next = nullptr;
        if (tail_ != nullptr) {
            tail_->next = block;
        } else {
            head_ = block;
        }
        tail_ = block;
        block->linked = true;
    }

    SubmittedBlock* find_block_locked(const void* ctx) const noexcept {
        for (SubmittedBlock* cur = head_; cur != nullptr; cur = cur->next) {
            if (cur == ctx) {
                return cur;
            }
        }
        return nullptr;
    }

    void unlink_locked(SubmittedBlock* block) noexcept {
        if (block->prev != nullptr) {
            block->prev->next = block->next;
        } else {
            head_ = block->next;
        }
        if (block->next != nullptr) {
            block->next->prev = block->prev;
        } else {
            tail_ = block->prev;
        }
        block->prev = nullptr;
        block->next = nullptr;
        block->linked = false;
    }

    bool remove_block(SubmittedBlock* block) noexcept {
        SrwExclusiveGuard guard(blocks_lock_);
        if (!block->linked) {
            return false;
        }
        unlink_locked(block);
        return true;
    }

    void* translate_context(void* inner_ctx) noexcept {
        SrwExclusiveGuard guard(blocks_lock_);
        SubmittedBlock* block = find_block_locked(inner_ctx);
        return block != nullptr ? block->caller_context : inner_ctx;
    }

    void* complete_block(void* inner_ctx) noexcept {
        SubmittedBlock* block = nullptr;
        void* caller_ctx = inner_ctx;
        LeonidaXma2PcmBlock* pcm = nullptr;
        {
            SrwExclusiveGuard guard(blocks_lock_);
            block = find_block_locked(inner_ctx);
            if (block != nullptr) {
                caller_ctx = block->caller_context;
                pcm = block->pcm;
                unlink_locked(block);
            }
        }
        if (pcm != nullptr) {
            api_.pcm_destroy(pcm);
        }
        delete block;
        return caller_ctx;
    }

    void drain_in_flight_blocks() noexcept {
        SubmittedBlock* list = nullptr;
        {
            SrwExclusiveGuard guard(blocks_lock_);
            list = head_;
            head_ = nullptr;
            tail_ = nullptr;
        }
        while (list != nullptr) {
            SubmittedBlock* next = list->next;
            if (list->pcm != nullptr) {
                api_.pcm_destroy(list->pcm);
            }
            delete list;
            list = next;
        }
    }

    Xma2DecoderApi api_;
    LeonidaXma2Decoder* decoder_ = nullptr;
    IXAudio2VoiceCallback* caller_callback_ = nullptr;
    CallbackProxy callback_proxy_;
    IXAudio2SourceVoice* inner_voice_ = nullptr;

    SRWLOCK blocks_lock_ = SRWLOCK_INIT;
    SubmittedBlock* head_ = nullptr;
    SubmittedBlock* tail_ = nullptr;
};

class Xma2EngineProxy final : public IXAudio2 {
public:
    Xma2EngineProxy(IXAudio2* inner, const Xma2DecoderApi& api) noexcept
        : inner_(inner), api_(api) {
        inner_->AddRef();
    }

    ~Xma2EngineProxy() {
        if (inner_ != nullptr) {
            inner_->Release();
            inner_ = nullptr;
        }
    }

    Xma2EngineProxy(const Xma2EngineProxy&) = delete;
    Xma2EngineProxy& operator=(const Xma2EngineProxy&) = delete;

    // --- IUnknown ---

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvInterface) noexcept override {
        if (ppvInterface == nullptr) {
            return E_POINTER;
        }
        *ppvInterface = nullptr;
        if (riid == IID_IUnknown || riid == IID_IXAudio2) {
            AddRef();
            *ppvInterface = static_cast<IXAudio2*>(this);
            return S_OK;
        }
        return inner_->QueryInterface(riid, ppvInterface);
    }

    ULONG STDMETHODCALLTYPE AddRef() noexcept override {
        return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    ULONG STDMETHODCALLTYPE Release() noexcept override {
        const ULONG remaining = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (remaining == 0) {
            delete this;
        }
        return remaining;
    }

    // --- IXAudio2 ---

    HRESULT STDMETHODCALLTYPE RegisterForCallbacks(IXAudio2EngineCallback* pCallback) noexcept override {
        return inner_->RegisterForCallbacks(pCallback);
    }

    void STDMETHODCALLTYPE UnregisterForCallbacks(IXAudio2EngineCallback* pCallback) noexcept override {
        inner_->UnregisterForCallbacks(pCallback);
    }

    HRESULT STDMETHODCALLTYPE CreateSourceVoice(
        IXAudio2SourceVoice** ppSourceVoice,
        const WAVEFORMATEX* pSourceFormat,
        UINT32 Flags,
        float MaxFrequencyRatio,
        IXAudio2VoiceCallback* pCallback,
        const XAUDIO2_VOICE_SENDS* pSendList,
        const XAUDIO2_EFFECT_CHAIN* pEffectChain) noexcept override {
        if (ppSourceVoice == nullptr || pSourceFormat == nullptr) {
            if (ppSourceVoice != nullptr) {
                *ppSourceVoice = nullptr;
            }
            return XAUDIO2_E_INVALID_CALL;
        }
        *ppSourceVoice = nullptr;

        if (is_console_opus_format(pSourceFormat)) {
            if (!g_opus_logged.exchange(true, std::memory_order_relaxed)) {
                std::fprintf(stderr,
                             "[xaudio2] unsupported console Opus voice format 0x%04X "
                             "(no runtime Opus decoder)\n",
                             static_cast<unsigned>(kWaveFormatConsoleOpus));
            }
            return E_NOTIMPL;
        }

        if (!is_xma2_format(pSourceFormat)) {
            return inner_->CreateSourceVoice(ppSourceVoice,
                                             pSourceFormat,
                                             Flags,
                                             MaxFrequencyRatio,
                                             pCallback,
                                             pSendList,
                                             pEffectChain);
        }

        WAVEFORMATEX pcm_format{};
        const HRESULT format_hr = translate_xma2_to_pcm16(pSourceFormat, &pcm_format);
        if (FAILED(format_hr)) {
            return format_hr;
        }

        LeonidaXma2Decoder* decoder = nullptr;
        const int32_t dec_status = api_.decoder_create(
            pcm_format.nSamplesPerSec, pcm_format.nChannels, &decoder);
        if (dec_status != LEONIDA_XMA2_OK) {
            if (decoder != nullptr) {
                api_.decoder_destroy(decoder);
            }
            return map_xma2_status(dec_status);
        }
        if (decoder == nullptr) {
            return XAUDIO2_E_XMA_DECODER_ERROR;
        }

        auto* proxy = new (std::nothrow) Xma2SourceVoiceProxy(api_, decoder, pCallback);
        if (proxy == nullptr) {
            api_.decoder_destroy(decoder);
            return E_OUTOFMEMORY;
        }

        IXAudio2SourceVoice* inner_voice = nullptr;
        const HRESULT voice_hr = inner_->CreateSourceVoice(&inner_voice,
                                                           &pcm_format,
                                                           Flags,
                                                           MaxFrequencyRatio,
                                                           proxy->callback(),
                                                           pSendList,
                                                           pEffectChain);
        if (FAILED(voice_hr) || inner_voice == nullptr) {
            delete proxy;
            return FAILED(voice_hr) ? voice_hr : E_FAIL;
        }

        proxy->attach_inner_voice(inner_voice);
        *ppSourceVoice = proxy;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateSubmixVoice(
        IXAudio2SubmixVoice** ppSubmixVoice,
        UINT32 InputChannels,
        UINT32 InputSampleRate,
        UINT32 Flags,
        UINT32 ProcessingStage,
        const XAUDIO2_VOICE_SENDS* pSendList,
        const XAUDIO2_EFFECT_CHAIN* pEffectChain) noexcept override {
        return inner_->CreateSubmixVoice(ppSubmixVoice,
                                         InputChannels,
                                         InputSampleRate,
                                         Flags,
                                         ProcessingStage,
                                         pSendList,
                                         pEffectChain);
    }

    HRESULT STDMETHODCALLTYPE CreateMasteringVoice(
        IXAudio2MasteringVoice** ppMasteringVoice,
        UINT32 InputChannels,
        UINT32 InputSampleRate,
        UINT32 Flags,
        LPCWSTR szDeviceId,
        const XAUDIO2_EFFECT_CHAIN* pEffectChain,
        AUDIO_STREAM_CATEGORY StreamCategory) noexcept override {
        return inner_->CreateMasteringVoice(ppMasteringVoice,
                                            InputChannels,
                                            InputSampleRate,
                                            Flags,
                                            szDeviceId,
                                            pEffectChain,
                                            StreamCategory);
    }

    HRESULT STDMETHODCALLTYPE StartEngine() noexcept override { return inner_->StartEngine(); }

    void STDMETHODCALLTYPE StopEngine() noexcept override { inner_->StopEngine(); }

    HRESULT STDMETHODCALLTYPE CommitChanges(UINT32 OperationSet) noexcept override {
        return inner_->CommitChanges(OperationSet);
    }

    void STDMETHODCALLTYPE GetPerformanceData(XAUDIO2_PERFORMANCE_DATA* pPerfData) noexcept override {
        inner_->GetPerformanceData(pPerfData);
    }

    void STDMETHODCALLTYPE SetDebugConfiguration(
        const XAUDIO2_DEBUG_CONFIGURATION* pDebugConfiguration,
        void* pReserved) noexcept override {
        inner_->SetDebugConfiguration(pDebugConfiguration, pReserved);
    }

private:
    std::atomic<ULONG> ref_count_{1};
    IXAudio2* inner_ = nullptr;
    Xma2DecoderApi api_;
};

}  // namespace

bool is_complete_decoder_api(const Xma2DecoderApi* api) noexcept {
    return api != nullptr && api->decoder_create != nullptr && api->decode_buffer != nullptr &&
           api->decoder_destroy != nullptr && api->pcm_data != nullptr &&
           api->pcm_sample_count != nullptr && api->pcm_destroy != nullptr;
}

#if defined(LEONIDA_HAS_XMA2_DECODER)
const Xma2DecoderApi& default_xma2_decoder_api() noexcept {
    static constexpr Xma2DecoderApi kDefaultApi{
        &leonida_xma2_decoder_create,
        &leonida_xma2_decode_buffer,
        &leonida_xma2_decoder_destroy,
        &leonida_xma2_pcm_data,
        &leonida_xma2_pcm_sample_count,
        &leonida_xma2_pcm_destroy,
    };
    return kDefaultApi;
}
#endif

bool is_xma2_format(const WAVEFORMATEX* format) noexcept {
    if (format == nullptr) {
        return false;
    }
    if (format->wFormatTag == kWaveFormatXma2) {
        return true;
    }
    return has_extensible_subformat(format, kSubformatXma2);
}

bool is_console_opus_format(const WAVEFORMATEX* format) noexcept {
    if (format == nullptr) {
        return false;
    }
    if (format->wFormatTag == kWaveFormatConsoleOpus) {
        return true;
    }
    return has_extensible_subformat(format, kSubformatConsoleOpus);
}

HRESULT translate_xma2_to_pcm16(const WAVEFORMATEX* source, WAVEFORMATEX* out_pcm) noexcept {
    if (source == nullptr || out_pcm == nullptr || !is_xma2_format(source)) {
        return XAUDIO2_E_INVALID_CALL;
    }
    if (source->nSamplesPerSec == 0 ||
        (source->nChannels != 1 && source->nChannels != 2)) {
        return XAUDIO2_E_INVALID_CALL;
    }
    const WORD block_align = static_cast<WORD>(source->nChannels * 2u);
    if (source->nSamplesPerSec > (std::numeric_limits<DWORD>::max)() / block_align) {
        return XAUDIO2_E_INVALID_CALL;
    }

    WAVEFORMATEX pcm{};
    pcm.wFormatTag = WAVE_FORMAT_PCM;
    pcm.nChannels = source->nChannels;
    pcm.nSamplesPerSec = source->nSamplesPerSec;
    pcm.wBitsPerSample = 16;
    pcm.nBlockAlign = block_align;
    pcm.nAvgBytesPerSec = source->nSamplesPerSec * block_align;
    pcm.cbSize = 0;
    *out_pcm = pcm;
    return S_OK;
}

HRESULT map_xma2_status(int32_t status) noexcept {
    switch (status) {
        case LEONIDA_XMA2_OK:
            return S_OK;
        case LEONIDA_XMA2_INVALID_ARGUMENT:
        case LEONIDA_XMA2_INVALID_FORMAT:
        case LEONIDA_XMA2_BAD_PACKET_SIZE:
            return XAUDIO2_E_INVALID_CALL;
        case LEONIDA_XMA2_OUT_OF_MEMORY:
            return E_OUTOFMEMORY;
        case LEONIDA_XMA2_DECODER_FAILURE:
        case LEONIDA_XMA2_PANIC:
        default:
            return XAUDIO2_E_XMA_DECODER_ERROR;
    }
}

HRESULT create_xaudio2_with_com_retry(XAudio2CreateFn create_fn,
                                      IXAudio2** out_engine,
                                      UINT32 flags,
                                      XAUDIO2_PROCESSOR processor,
                                      CoInitializeExFn co_init_fn) noexcept {
    if (out_engine != nullptr) {
        *out_engine = nullptr;
    }
    if (create_fn == nullptr || out_engine == nullptr) {
        return E_INVALIDARG;
    }

    const HRESULT first_hr = create_fn(out_engine, flags, processor);
    if (first_hr != CO_E_NOTINITIALIZED) {
        return first_hr;
    }

    const CoInitializeExFn init = co_init_fn != nullptr ? co_init_fn : &::CoInitializeEx;
    const HRESULT co_hr = init(nullptr, COINIT_MULTITHREADED);
    if (FAILED(co_hr) && co_hr != RPC_E_CHANGED_MODE) {
        return co_hr;
    }

    *out_engine = nullptr;
    return create_fn(out_engine, flags, processor);
}

HRESULT wrap_xaudio2_for_xma2(IXAudio2* inner,
                              const Xma2DecoderApi* api,
                              IXAudio2** out_wrapped) noexcept {
    if (out_wrapped == nullptr) {
        return E_POINTER;
    }
    *out_wrapped = nullptr;
    if (inner == nullptr || !is_complete_decoder_api(api)) {
        return E_INVALIDARG;
    }

    auto* proxy = new (std::nothrow) Xma2EngineProxy(inner, *api);
    if (proxy == nullptr) {
        return E_OUTOFMEMORY;
    }
    *out_wrapped = proxy;
    return S_OK;
}

}  // namespace rt::xaudio2
