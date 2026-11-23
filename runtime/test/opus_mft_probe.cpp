// Run from the scratch directory produced by opus_mft_fixture.py.
// Each stream uses one Windows Media Foundation decoder for two consecutive raw packets.
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mftransform.h>
#include <mferror.h>
#include <wmcodecdsp.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
#define CHECK(call)                                                                                      \
    do {                                                                                                 \
        const HRESULT result = (call);                                                                   \
        if (FAILED(result)) {                                                                            \
            std::fprintf(stderr, "%s failed (%08lx) at %d\n", #call, static_cast<ULONG>(result), __LINE__); \
            return result;                                                                               \
        }                                                                                                \
    } while (false)

HRESULT sample(DWORD size, const BYTE* source, ComPtr<IMFSample>& value) {
    ComPtr<IMFMediaBuffer> buffer;
    CHECK(MFCreateSample(&value));
    CHECK(MFCreateMemoryBuffer(size, &buffer));
    if (source) {
        BYTE* data;
        CHECK(buffer->Lock(&data, nullptr, nullptr));
        std::memcpy(data, source, size);
        CHECK(buffer->Unlock());
        CHECK(buffer->SetCurrentLength(size));
    }
    return value->AddBuffer(buffer.Get());
}

HRESULT check(unsigned channels) {
    std::array<std::uint8_t, 480> input{};
    std::vector<float> expected(2 * 960 * channels);
    std::ifstream packets("packets-" + std::to_string(channels) + ".bin", std::ios::binary);
    std::ifstream reference("reference-" + std::to_string(channels) + ".f32", std::ios::binary);
    if (!packets.read(reinterpret_cast<char*>(input.data()), input.size()) ||
        !reference.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float))) return E_FAIL;
    ComPtr<IMFTransform> decoder;
    CHECK(CoCreateInstance(__uuidof(CMSOpusDecMFT), nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&decoder)));
    ComPtr<IMFMediaType> in;
    CHECK(MFCreateMediaType(&in));
    CHECK(in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio));
    CHECK(in->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Opus));
    CHECK(in->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, 48000));
    CHECK(in->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels));
    // Opus identification header with zero pre-skip and the original 48 kHz input rate.
    const std::array<BYTE, 19> header{'O', 'p', 'u', 's', 'H', 'e', 'a', 'd', 1,
                                      static_cast<BYTE>(channels), 0, 0, 0x80, 0xbb};
    CHECK(in->SetBlob(MF_MT_USER_DATA, header.data(), static_cast<UINT32>(header.size())));
    CHECK(decoder->SetInputType(0, in.Get(), 0));
    for (DWORD i = 0; ; ++i) {
        ComPtr<IMFMediaType> out;
        const HRESULT available = decoder->GetOutputAvailableType(0, i, &out);
        CHECK(available);
        GUID subtype;
        CHECK(out->GetGUID(MF_MT_SUBTYPE, &subtype));
        UINT32 bps = MFGetAttributeUINT32(out.Get(), MF_MT_AUDIO_BITS_PER_SAMPLE, 0);
        UINT32 ch = MFGetAttributeUINT32(out.Get(), MF_MT_AUDIO_NUM_CHANNELS, 0);
        if (ch == channels && subtype == MFAudioFormat_Float && bps == 32 &&
            MFGetAttributeUINT32(out.Get(), MF_MT_AUDIO_SAMPLES_PER_SECOND, 0) == 48000) {
            CHECK(decoder->SetOutputType(0, out.Get(), 0));
            break;
        }
    }
    MFT_OUTPUT_STREAM_INFO info{};
    CHECK(decoder->GetOutputStreamInfo(0, &info));
    CHECK(decoder->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0));
    CHECK(decoder->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0));
    std::vector<float> actual;
    for (unsigned packet = 0; packet < 2; ++packet) {
        ComPtr<IMFSample> in_data;
        CHECK(sample(240, input.data() + packet * 240, in_data));
        CHECK(in_data->SetSampleTime(packet * 200000LL));
        CHECK(in_data->SetSampleDuration(200000LL));
        CHECK(decoder->ProcessInput(0, in_data.Get(), 0));
        ComPtr<IMFSample> out_data;
        if (!(info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES)) {
            CHECK(sample(std::max<DWORD>(info.cbSize, 5760 * channels * 4), nullptr, out_data));
        }
        MFT_OUTPUT_DATA_BUFFER output{};
        output.pSample = out_data.Get();
        DWORD status = 0;
        HRESULT hr = decoder->ProcessOutput(0, 1, &output, &status);
        if (output.pEvents) output.pEvents->Release();
        if (!out_data && output.pSample) out_data.Attach(output.pSample);
        CHECK(hr);
        if (!out_data) return E_UNEXPECTED;
        ComPtr<IMFMediaBuffer> contiguous;
        CHECK(out_data->ConvertToContiguousBuffer(&contiguous));
        BYTE* data;
        DWORD length = 0;
        CHECK(contiguous->Lock(&data, nullptr, &length));
        if (length != 960 * channels * sizeof(float)) {
            contiguous->Unlock();
            return E_UNEXPECTED;
        }
        const auto* pcm = reinterpret_cast<const float*>(data);
        actual.insert(actual.end(), pcm, pcm + 960 * channels);
        CHECK(contiguous->Unlock());
        std::printf("channels=%u packet=%u float32_bytes=%lu\n", channels, packet, length);
    }
    if (actual.size() != expected.size()) return E_FAIL;
    float error = 0.0f;
    float peak = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) return E_FAIL;
        peak = std::max(peak, std::abs(expected[i]));
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::printf("channels=%u samples=%zu max_error=%.9f\n", channels, actual.size(), error);
    return peak > 0.1f && error <= 1.0f / 32768 ? S_OK : E_FAIL;
}

int main() {
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return 1;
    if (FAILED(MFStartup(MF_VERSION))) {
        CoUninitialize();
        return 1;
    }
    const HRESULT mono = check(1);
    const HRESULT stereo = check(2);
    MFShutdown();
    CoUninitialize();
    return SUCCEEDED(mono) && SUCCEEDED(stereo) ? 0 : 1;
}
