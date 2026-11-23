"""Generate two raw Opus packets and a float32 reference for each original signal."""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from awc_opus import write_ogg_opus


def ffmpeg(args, data=None):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1", *args],
                          input=data, stdout=subprocess.PIPE, check=True, timeout=15).stdout


def generate(channels, temporary, output):
    frames = 960
    samples = [round(9000 * math.sin(i * (0.039 + 0.023 * ch + 0.011 * channels)))
               for i in range(2 * frames) for ch in range(channels)]
    encoded = temporary / f"encoded-{channels}.ogg"
    ffmpeg(["-f", "s16le", "-ar", "48000", "-ac", str(channels), "-i", "pipe:0",
            "-c:a", "libopus", "-application", "lowdelay", "-b:a", "96k", "-vbr", "off",
            "-frame_duration", "20", "-frames:a", "2", str(encoded)], struct.pack(f"<{len(samples)}h", *samples))
    raw = ffmpeg(["-i", str(encoded), "-map", "0:a:0", "-c:a", "copy", "-frames:a", "2", "-f", "data", "pipe:1"])
    if len(raw) != 480:
        raise ValueError("Expected two 240-byte packets at 96 kbit/s CBR")

    # Raw-packet decoding includes encoder priming. Compare from the same sample in the FFmpeg stream.
    ogg = temporary / f"reference-{channels}.ogg"
    write_ogg_opus(str(ogg), [raw[:240], raw[240:]], channels, 17 + channels, 2 * frames, pre_skip=0)
    reference = ffmpeg(["-c:a", "opus", "-i", str(ogg), "-f", "f32le", "pipe:1"])
    if len(reference) != 2 * frames * channels * 4:
        raise ValueError("Unexpected FFmpeg output length")
    (output / f"packets-{channels}.bin").write_bytes(raw)
    (output / f"reference-{channels}.f32").write_bytes(reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="a scratch directory for the four generated inputs")
    directory = parser.parse_args().output
    directory.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        for count in (1, 2):
            generate(count, Path(scratch), directory)
