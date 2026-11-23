"""File names of streamed-music channels in tools/awc_opus.py export_bank (ASSET-309), from synthetic AWC data.

A music channel used to be written as `<id>.ogg` outside export_stems, so a second channel with the same id, or a
channel sharing an Opus stream's id, overwrote that file while the counts still listed both.

    python -I tests/test_awc_music_names.py
"""
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_awc_opus import awc, build_music_bank, format_block, parse_ogg_pages, tag  # noqa: E402


def music_bank(ids, tocs):
    """A one-descriptor music bank whose channel groups carry `ids` (one Opus TOC byte per group)."""
    bank = bytearray(build_music_bank([list(tocs)], 960))
    _, music = awc.parse_awc(bank)
    (desc_at, _), _ = music[0]
    for i, cid in enumerate(ids):
        struct.pack_into("<I", bank, desc_at + 12 + 16 * i, cid)
    return bytes(bank)


def opus_stream_and_music_bank(sid, stream_toc, music_toc):
    """An Opus stream with id `sid` followed by a one-channel music descriptor whose channel also has `sid`."""
    music = music_bank([sid], [music_toc])
    _, desc = awc.parse_awc(music)
    (desc_at, desc_size), (data_at, data_size) = desc[0]
    fmt = format_block(960, 48000, 0x0B, 1)
    stream = bytearray(2048)
    struct.pack_into("<HB", stream, 0, 1, stream_toc)
    header = 16 + 2 * 4 + 4 * 8
    fmt_at = header
    stream_at = fmt_at + len(fmt)
    m_desc_at = stream_at + len(stream)
    m_data_at = m_desc_at + desc_size
    out = bytearray(b"ADAT" + struct.pack("<III", 0, 2, header))
    out += struct.pack("<I", (sid & 0x1FFFFFFF) | (2 << 29))
    out += struct.pack("<I", 0x20 | (2 << 29))
    out += struct.pack("<Q", tag(0x76, fmt_at, len(fmt)))
    out += struct.pack("<Q", tag(0x55, stream_at, len(stream)))
    out += struct.pack("<Q", tag(0x48, m_desc_at, desc_size))
    out += struct.pack("<Q", tag(0x55, m_data_at, data_size))
    assert len(out) == header
    return bytes(out + fmt + stream + music[desc_at:desc_at + desc_size] + music[data_at:data_at + data_size])


def payload(path):
    """The Opus packet bytes of an Ogg file (every page after the two header pages)."""
    with open(path, "rb") as f:
        pages = parse_ogg_pages(f.read())
    return b"".join(page["body"] for page in pages[2:])


class MusicChannelNamesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def export(self, name, bank):
        path = os.path.join(self.tmp, name + ".awc")
        with open(path, "wb") as f:
            f.write(bank)
        counts = awc.export_bank(path, os.path.join(self.tmp, "out"))
        return counts, os.path.join(self.tmp, "out", name)

    def test_two_music_channels_with_one_id_both_export(self):
        counts, od = self.export("music_dup", music_bank([0x123, 0x123], [0x78, 0x70]))
        self.assertEqual(counts["music"], 2)
        self.assertEqual(sorted(os.listdir(od)), ["00000123.ogg", "00000123~2.ogg"])
        self.assertEqual(payload(os.path.join(od, "00000123.ogg")), b"\x78")
        self.assertEqual(payload(os.path.join(od, "00000123~2.ogg")), b"\x70")

    def test_music_channel_sharing_an_opus_stream_id_keeps_the_stream_name(self):
        counts, od = self.export("shared", opus_stream_and_music_bank(0x456, 0x78, 0x70))
        self.assertEqual((counts["opus"], counts["music"]), (1, 1))
        self.assertEqual(sorted(os.listdir(od)), ["00000456.ogg", "00000456~2.ogg"])
        # The stream keeps the name export_stems gives it (and awc_manifest reports); the channel moves aside.
        self.assertEqual(payload(os.path.join(od, "00000456.ogg")), b"\x78")
        self.assertEqual(payload(os.path.join(od, "00000456~2.ogg")), b"\x70")

    def test_distinct_music_ids_keep_plain_names(self):
        counts, od = self.export("music", music_bank([0x100, 0x101], [0x78, 0x70]))
        self.assertEqual(counts["music"], 2)
        self.assertEqual(sorted(os.listdir(od)), ["00000100.ogg", "00000101.ogg"])

    def test_files_written_match_the_counts(self):
        for name, bank in (("music_dup", music_bank([0x123, 0x123, 0x123], [0x78, 0x70, 0x68])),
                           ("shared", opus_stream_and_music_bank(0x456, 0x78, 0x70))):
            with self.subTest(name=name):
                counts, od = self.export(name, bank)
                self.assertEqual(len(os.listdir(od)), sum(counts.values()) - counts["other"])


if __name__ == "__main__":
    unittest.main()
