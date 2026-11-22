#![deny(unsafe_op_in_unsafe_fn)]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Mutex;

use xma::{Xma2Stream, XMA_FRAME_SAMPLES, XMA_PACKET_BYTES};

pub const LEONIDA_XMA2_OK: i32 = 0;
pub const LEONIDA_XMA2_INVALID_ARGUMENT: i32 = 1;
pub const LEONIDA_XMA2_INVALID_FORMAT: i32 = 2;
pub const LEONIDA_XMA2_BAD_PACKET_SIZE: i32 = 3;
pub const LEONIDA_XMA2_OUT_OF_MEMORY: i32 = 4;
pub const LEONIDA_XMA2_DECODER_FAILURE: i32 = 5;
pub const LEONIDA_XMA2_PANIC: i32 = 6;

const MAX_FRAMES_PER_PACKET: usize = 64;

#[repr(C)]
pub struct LeonidaXma2Decoder {
    stream: Mutex<Xma2Stream>,
    channels: usize,
}

#[repr(C)]
pub struct LeonidaXma2PcmBlock {
    samples: Box<[i16]>,
}

/// Creates a decoder for one mono or stereo XMA2 stream.
///
/// # Safety
/// `out_decoder` must be writable for one pointer. The returned decoder must be destroyed once,
/// after all calls using it have finished.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_decoder_create(
    sample_rate: u32,
    channels: u32,
    out_decoder: *mut *mut LeonidaXma2Decoder,
) -> i32 {
    if out_decoder.is_null() {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    unsafe { *out_decoder = ptr::null_mut() };
    if sample_rate == 0 || !(channels == 1 || channels == 2) {
        return LEONIDA_XMA2_INVALID_FORMAT;
    }

    match catch_unwind(AssertUnwindSafe(|| {
        Xma2Stream::new(sample_rate, channels as usize)
    })) {
        Ok(Ok(stream)) => {
            let decoder = Box::new(LeonidaXma2Decoder {
                stream: Mutex::new(stream),
                channels: channels as usize,
            });
            unsafe { *out_decoder = Box::into_raw(decoder) };
            LEONIDA_XMA2_OK
        }
        Ok(Err(_)) => LEONIDA_XMA2_INVALID_FORMAT,
        Err(_) => LEONIDA_XMA2_PANIC,
    }
}

/// Decodes a buffer containing one or more complete 2048-byte XMA2 packets.
///
/// # Safety
/// `decoder` must be a live handle returned by `leonida_xma2_decoder_create`.
/// `input` must reference `input_len` readable bytes and `out_block` must be writable.
/// Do not destroy a decoder concurrently with this call. The returned PCM block remains valid
/// until `leonida_xma2_pcm_destroy` is called.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_decode_buffer(
    decoder: *mut LeonidaXma2Decoder,
    input: *const u8,
    input_len: usize,
    out_block: *mut *mut LeonidaXma2PcmBlock,
) -> i32 {
    if out_block.is_null() {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    unsafe { *out_block = ptr::null_mut() };

    if decoder.is_null() || input.is_null() || input_len > isize::MAX as usize {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    if input_len == 0 || !input_len.is_multiple_of(XMA_PACKET_BYTES) {
        return LEONIDA_XMA2_BAD_PACKET_SIZE;
    }

    let decoded = catch_unwind(AssertUnwindSafe(|| {
        let handle = unsafe { &*decoder };
        let bytes = unsafe { slice::from_raw_parts(input, input_len) };
        let mut stream = handle
            .stream
            .lock()
            .map_err(|_| LEONIDA_XMA2_DECODER_FAILURE)?;
        let max_packet_samples = MAX_FRAMES_PER_PACKET
            .checked_mul(XMA_FRAME_SAMPLES)
            .and_then(|samples| samples.checked_mul(handle.channels))
            .ok_or(LEONIDA_XMA2_INVALID_ARGUMENT)?;
        let mut samples = Vec::<i16>::new();

        let (packets, _) = bytes.as_chunks::<XMA_PACKET_BYTES>();
        for packet in packets {
            let requested = samples
                .len()
                .checked_add(max_packet_samples)
                .ok_or(LEONIDA_XMA2_INVALID_ARGUMENT)?;
            if requested > isize::MAX as usize / std::mem::size_of::<i16>() {
                return Err(LEONIDA_XMA2_INVALID_ARGUMENT);
            }
            samples
                .try_reserve(max_packet_samples)
                .map_err(|_| LEONIDA_XMA2_OUT_OF_MEMORY)?;
            stream.decode_packet(packet, &mut samples);
        }

        if !samples.len().is_multiple_of(handle.channels) {
            return Err(LEONIDA_XMA2_DECODER_FAILURE);
        }
        Ok(Box::new(LeonidaXma2PcmBlock {
            samples: samples.into_boxed_slice(),
        }))
    }));

    match decoded {
        Ok(Ok(block)) => {
            unsafe { *out_block = Box::into_raw(block) };
            LEONIDA_XMA2_OK
        }
        Ok(Err(status)) => status,
        Err(_) => LEONIDA_XMA2_PANIC,
    }
}

/// Destroys a decoder. Passing null is a no-op.
///
/// # Safety
/// A non-null pointer must be a live handle returned by `leonida_xma2_decoder_create`,
/// and no calls may be using it concurrently.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_decoder_destroy(decoder: *mut LeonidaXma2Decoder) {
    if !decoder.is_null() {
        unsafe { drop(Box::from_raw(decoder)) };
    }
}

/// Returns the immutable interleaved signed-16 PCM pointer for a block.
///
/// # Safety
/// A non-null pointer must be a live block returned by `leonida_xma2_decode_buffer`.
/// The returned pointer is valid only until the block is destroyed.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_pcm_data(block: *const LeonidaXma2PcmBlock) -> *const i16 {
    unsafe { block.as_ref() }.map_or(ptr::null(), |pcm| pcm.samples.as_ptr())
}

/// Returns the number of signed-16 samples in a block.
///
/// # Safety
/// A non-null pointer must be a live block returned by `leonida_xma2_decode_buffer`.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_pcm_sample_count(block: *const LeonidaXma2PcmBlock) -> usize {
    unsafe { block.as_ref() }.map_or(0, |pcm| pcm.samples.len())
}

/// Destroys a PCM block. Passing null is a no-op.
///
/// # Safety
/// A non-null pointer must be a live block returned by `leonida_xma2_decode_buffer`.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_pcm_destroy(block: *mut LeonidaXma2PcmBlock) {
    if !block.is_null() {
        unsafe { drop(Box::from_raw(block)) };
    }
}
