#ifndef LEONIDA_OPUS_DECODER_H
#define LEONIDA_OPUS_DECODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LeonidaOpusDecoder LeonidaOpusDecoder;

enum LeonidaOpusStatus {
    LEONIDA_OPUS_OK = 0,
    LEONIDA_OPUS_INVALID_ARGUMENT = 1,
    LEONIDA_OPUS_BUFFER_TOO_SMALL = 2,
    LEONIDA_OPUS_BAD_PACKET = 3,
    LEONIDA_OPUS_DECODER_FAILURE = 4
};

// One handle per logical stream, with one or two output channels at 48 kHz.
int32_t leonida_opus_create(uint32_t channels, LeonidaOpusDecoder **out_decoder);

// Decode one raw packet to interleaved native float32 PCM. Both sizes are bytes.
// The caller serializes operations on a handle and supplies non-overlapping,
// valid buffers. On failure, written is zero and destination is untouched.
// A capacity failure can be retried on the same handle with a larger buffer.
// Recreate the handle after LEONIDA_OPUS_DECODER_FAILURE.
int32_t leonida_opus_decode(LeonidaOpusDecoder *decoder,
                            const uint8_t *packet, size_t packet_len,
                            void *destination, size_t capacity, size_t *written);

void leonida_opus_destroy(LeonidaOpusDecoder *decoder);

#ifdef __cplusplus
}
#endif
#endif
