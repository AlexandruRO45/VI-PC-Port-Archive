#include "../leonida_xma2_decoder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

template <size_t Size>
void put_bits(std::array<uint8_t, Size>& bytes, size_t& position, uint32_t width,
              uint32_t value) {
  for (uint32_t bit = width; bit > 0; --bit, ++position) {
    if (((value >> (bit - 1)) & 1u) != 0) {
      bytes[position / 8] |= static_cast<uint8_t>(1u << (7 - (position % 8)));
    }
  }
}

std::array<uint8_t, LEONIDA_XMA2_PACKET_BYTES> make_silent_stereo_packet() {
  std::array<uint8_t, LEONIDA_XMA2_PACKET_BYTES> packet{};
  constexpr uint32_t frame_bits = 36;
  size_t position = 0;
  put_bits(packet, position, 6, 1);
  put_bits(packet, position, 15, frame_bits);
  put_bits(packet, position, 3, 1);
  put_bits(packet, position, 8, 0);
  put_bits(packet, position, 15, frame_bits);
  put_bits(packet, position, 1, 1);  // Fixed-size tile header.
  put_bits(packet, position, 1, 0);  // One full-length subframe.
  put_bits(packet, position, 1, 0);  // No optional stereo decorrelation matrix.
  put_bits(packet, position, 8, 0);  // No dynamic range adjustment.
  put_bits(packet, position, 1, 0);  // No trim metadata.
  put_bits(packet, position, 1, 0);  // No fill data.
  put_bits(packet, position, 1, 0);  // No corrupt-subframe marker.
  put_bits(packet, position, 1, 0);  // No stereo transform error.
  put_bits(packet, position, 1, 1);  // Two-channel group without decorrelation.
  put_bits(packet, position, 1, 0);
  put_bits(packet, position, 2, 0);  // No transmitted coefficients: silence.
  put_bits(packet, position, 1, 0);  // Frame alignment bit.
  put_bits(packet, position, 1, 0);  // No additional frame in this packet.
  assert(position == 32 + frame_bits);
  return packet;
}

int main() {
  LeonidaXma2Decoder* decoder = nullptr;
  assert(leonida_xma2_decoder_create(48000, 2, &decoder) == LEONIDA_XMA2_OK);
  assert(decoder != nullptr);

  const auto packet = make_silent_stereo_packet();
  std::array<uint8_t, LEONIDA_XMA2_PACKET_BYTES * 3> input{};
  std::copy(packet.begin(), packet.end(), input.begin() + LEONIDA_XMA2_PACKET_BYTES);
  std::copy(packet.begin(), packet.end(), input.begin() + LEONIDA_XMA2_PACKET_BYTES * 2);
  LeonidaXma2PcmBlock* pcm = nullptr;
  assert(leonida_xma2_decode_buffer(decoder, input.data(), input.size(), &pcm) ==
         LEONIDA_XMA2_OK);
  assert(pcm != nullptr);
  const size_t samples = leonida_xma2_pcm_sample_count(pcm);
  assert(samples > 0);
  assert(samples % 2 == 0);
  const int16_t* data = leonida_xma2_pcm_data(pcm);
  assert(data != nullptr);

  LeonidaXma2PcmBlock* invalid = reinterpret_cast<LeonidaXma2PcmBlock*>(1);
  assert(leonida_xma2_decode_buffer(decoder, input.data(), input.size() - 1, &invalid) ==
         LEONIDA_XMA2_BAD_PACKET_SIZE);
  assert(invalid == nullptr);

  leonida_xma2_decoder_destroy(decoder);
  assert(leonida_xma2_pcm_sample_count(pcm) == samples);
  assert(samples == leonida_xma2_pcm_sample_count(pcm));
  for (size_t i = 0; i < samples; ++i) {
    assert(data[i] == 0);
  }
  leonida_xma2_pcm_destroy(pcm);
  leonida_xma2_decoder_destroy(nullptr);
  return 0;
}
