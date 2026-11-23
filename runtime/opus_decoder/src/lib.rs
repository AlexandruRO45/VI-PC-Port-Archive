use opusdec::{
    celt::{OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET},
    OpusDecoder,
};
use std::alloc::{alloc, Layout};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{ptr, slice};

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 1;
const BUFFER_TOO_SMALL: i32 = 2;
const BAD_PACKET: i32 = 3;
const DECODER_FAILURE: i32 = 4;
const MAX_FRAMES: usize = 5760; // Opus maximum of 120 ms at 48 kHz.

pub struct LeonidaOpusDecoder {
    stream: OpusDecoder,
    scratch: Vec<f32>,
}

/// # Safety
/// `out_decoder` must be writable for one pointer. The caller owns the result.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_create(
    channels: u32,
    out_decoder: *mut *mut LeonidaOpusDecoder,
) -> i32 {
    if out_decoder.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { out_decoder.write(ptr::null_mut()) };
    if !(1..=2).contains(&channels) {
        return INVALID_ARGUMENT;
    }
    let created = catch_unwind(|| {
        let stream = OpusDecoder::new(48000, channels as usize).ok()?;
        let mut scratch = Vec::new();
        let samples = MAX_FRAMES * channels as usize;
        scratch.try_reserve_exact(samples).ok()?;
        scratch.resize(samples, 0.0);
        // Use the global allocator with the same layout that destroy passes to Box.
        let handle =
            unsafe { alloc(Layout::new::<LeonidaOpusDecoder>()).cast::<LeonidaOpusDecoder>() };
        if handle.is_null() {
            return None;
        }
        unsafe { handle.write(LeonidaOpusDecoder { stream, scratch }) };
        Some(handle)
    });
    match created {
        Ok(Some(handle)) => {
            unsafe { out_decoder.write(handle) };
            OK
        }
        _ => DECODER_FAILURE,
    }
}

/// # Safety
/// The handle must be live and used exclusively. The packet must be readable and
/// destination writable for their supplied sizes. All arguments must be disjoint.
/// `written` must be writable and aligned. Recreate the handle after a decoder failure.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_decode(
    decoder: *mut LeonidaOpusDecoder,
    packet: *const u8,
    packet_len: usize,
    destination: *mut c_void,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    if written.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { written.write(0) };
    if decoder.is_null()
        || packet.is_null()
        || destination.is_null()
        || packet_len == 0
        || packet_len > i32::MAX as usize
    {
        return INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        let decoder = unsafe { &mut *decoder };
        let packet = unsafe { slice::from_raw_parts(packet, packet_len) };
        let frame_bytes = decoder.stream.channels() * size_of::<f32>();
        let frame_capacity = (capacity / frame_bytes).min(MAX_FRAMES);
        if frame_capacity == 0 {
            return BUFFER_TOO_SMALL;
        }
        // The backend checks packet duration against frame_capacity before
        // updating stream state. Stage output so errors cannot modify the caller's buffer.
        let frames =
            decoder
                .stream
                .decode_float(packet, &mut decoder.scratch, frame_capacity as i32, false);
        if frames == OPUS_BUFFER_TOO_SMALL {
            return BUFFER_TOO_SMALL;
        }
        if frames == OPUS_INVALID_PACKET {
            return BAD_PACKET;
        }
        if frames < 0 || frames as usize > frame_capacity {
            return DECODER_FAILURE;
        }
        let bytes = frames as usize * frame_bytes;
        unsafe {
            ptr::copy_nonoverlapping(
                decoder.scratch.as_ptr().cast::<u8>(),
                destination.cast(),
                bytes,
            );
            written.write(bytes);
        }
        OK
    }))
    .unwrap_or(DECODER_FAILURE)
}

/// # Safety
/// Pass null or a handle created by this library exactly once, after use.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_destroy(decoder: *mut LeonidaOpusDecoder) {
    if !decoder.is_null() {
        unsafe { drop(Box::from_raw(decoder)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_channel_count_clears_handle() {
        let mut handle = ptr::dangling_mut();
        assert_eq!(
            unsafe { leonida_opus_create(3, &mut handle) },
            INVALID_ARGUMENT
        );
        assert!(handle.is_null());
    }

    #[test]
    fn incomplete_packet_and_empty_input_preserve_destination() {
        let mut decoder = ptr::null_mut();
        assert_eq!(unsafe { leonida_opus_create(1, &mut decoder) }, OK);
        let mut destination = [0xa5u8; 128];
        // Code 3 needs a second byte describing its frame count.
        let packet = [3u8];
        for (length, expected) in [(0, INVALID_ARGUMENT), (1, BAD_PACKET)] {
            let mut written = usize::MAX;
            let status = unsafe {
                leonida_opus_decode(
                    decoder,
                    packet.as_ptr(),
                    length,
                    destination.as_mut_ptr().cast(),
                    destination.len(),
                    &mut written,
                )
            };
            assert_eq!(status, expected);
            assert_eq!(written, 0);
            assert_eq!(destination, [0xa5; 128]);
        }
        unsafe { leonida_opus_destroy(decoder) };
    }
}
