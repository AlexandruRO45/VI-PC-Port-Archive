"""File names of streamed-music channels in tools/awc_opus.py export_bank (ASSET-309), and the rows
tools/awc_manifest.py lists for them, from synthetic AWC data.

A music channel used to be written as `<id>.ogg` outside export_stems, so a second channel with the same id, or a
channel sharing an Opus stream's id, overwrote that file while the counts still listed both. awc_manifest likewise
let such a channel replace the earlier row with its id.

    python -I tests/test_awc_music_names.py
"""
import contextlib
import io
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "tools")]
from test_awc_opus import awc, build_music_bank, format_block, parse_ogg_pages, tag  # noqa: E402
import awc_manifest  # noqa: E402


def music_bank(ids, tocs, empty=()):
    """A one-descriptor music bank whose channel groups carry `ids` (one Opus TOC byte per group). The groups
    listed in `empty` get no subblocks: still Opus channels, but with no packets, so export_bank writes no file for
    them. The other groups' subblocks follow each other after the channel table, as music_channels reads them."""
    bank = bytearray(build_music_bank([list(tocs)], 960))
    _, music = awc.parse_awc(bank)
    (desc_at, _), (data_at, data_size) = music[0]
    for i, cid in enumerate(ids):
        struct.pack_into("<I", bank, desc_at + 12 + 16 * i, cid)
    if empty:
        bank[data_at + 2048:data_at + data_size] = bytes(data_size - 2048)
        k = 0
        for i, toc in enumerate(tocs):
            struct.pack_into("<I", bank, data_at + 24 * i + 4, 0 if i in empty else 1)
            if i not in empty:
                struct.pack_into("<HB", bank, data_at + 2048 * (1 + k), 1, toc)
                k += 1
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

    def test_a_channel_without_packets_takes_no_name(self):
        # The first 0x123 group has no subblocks, so it holds no packets and is not written: the group after it
        # keeps the plain name instead of moving to `~2`. (A non-Opus group would not do here: music_channels
        # rejects the whole descriptor for one since 2f798a5a.)
        counts, od = self.export("unwritten", music_bank([0x123, 0x123], [0x78, 0x70], empty=(0,)))
        self.assertEqual(counts["music"], 1)
        self.assertEqual(os.listdir(od), ["00000123.ogg"])
        self.assertEqual(payload(os.path.join(od, "00000123.ogg")), b"\x70")


NOTE = re.compile(r"awc_opus exports it as (\S+)")


class ManifestRowsTest(unittest.TestCase):
    """awc_manifest lists one row per stream and per music channel, and names the file export_bank writes."""

    BANKS = {
        "shared": lambda: opus_stream_and_music_bank(0x456, 0x78, 0x70),
        "music_dup": lambda: music_bank([0x123, 0x123, 0x123], [0x78, 0x70, 0x68]),
        "unwritten": lambda: music_bank([0x123, 0x123], [0x78, 0x70], empty=(0,)),
        "music": lambda: music_bank([0x100, 0x101], [0x78, 0x70]),
    }

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def bank(self, name):
        path = os.path.join(self.tmp, name + ".awc")
        with open(path, "wb") as f:
            f.write(self.BANKS[name]())
        return path

    def rows(self, name):
        with contextlib.redirect_stderr(io.StringIO()):
            return awc_manifest._bank_rows(self.bank(name), name)

    def test_a_channel_sharing_a_stream_id_no_longer_replaces_the_stream_row(self):
        rows = self.rows("shared")
        self.assertEqual([(r["stream"], r["streamed_music"]) for r in rows],
                         [("00000456", False), ("00000456", True)])
        self.assertNotIn("exports it as", rows[0]["notes"])
        self.assertEqual(NOTE.search(rows[1]["notes"]).group(1), "00000456~2")

    def test_every_music_channel_with_one_id_has_a_row(self):
        rows = self.rows("music_dup")
        self.assertEqual([r["stream"] for r in rows], ["00000123"] * 3)
        self.assertEqual([(NOTE.search(r["notes"]) or [None, None])[1] for r in rows],
                         [None, "00000123~2", "00000123~3"])

    def test_an_unwritten_channel_is_listed_without_an_exported_name(self):
        rows = self.rows("unwritten")
        self.assertEqual(len(rows), 2)
        self.assertEqual([NOTE.search(r["notes"]) for r in rows], [None, None])

    def test_rows_name_exactly_the_files_export_bank_writes(self):
        for name in self.BANKS:
            with self.subTest(name=name):
                rows = self.rows(name)
                awc.export_bank(self.bank(name), os.path.join(self.tmp, "out"))
                written = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.tmp, "out", name))}
                named = {(NOTE.search(r["notes"]) or [None, r["stream"]])[1] for r in rows
                         if r["channels"] or not r["streamed_music"]}
                self.assertEqual(named, written)

    def test_collect_counts_every_music_channel(self):
        folder = os.path.join(self.tmp, "banks")
        os.makedirs(folder)
        for name in ("shared", "music_dup"):
            with open(os.path.join(folder, name + ".awc"), "wb") as f:
                f.write(self.BANKS[name]())
        with contextlib.redirect_stderr(io.StringIO()):
            banks, rows, streamed = awc_manifest.collect(folder)
        self.assertEqual((banks, len(rows), streamed), (2, 5, 4))


class BankStemsTest(unittest.TestCase):
    def test_written_channels_are_the_ones_with_packets_in_both_modes(self):
        for bank in (music_bank([0x123, 0x123], [0x78, 0x70], empty=(0,)),
                     music_bank([0x100, 0x101], [0xFC, 0x78])):
            _, music = awc.parse_awc(bank)
            full = awc.music_channels(bank, *music[0])
            meta = awc.music_channels(bank, *music[0], collect_packets=False)
            self.assertEqual([c[5] for c in full], [c[5] for c in meta])
            self.assertEqual([bool(c[4]) for c in full], [bool(c[5]) for c in full])
            self.assertEqual(awc.bank_stems([], full), awc.bank_stems([], meta))

    def test_stream_stems_stay_export_stems(self):
        channels = [(0x5, 0, 0, 0x0B, [], 1), (0x7, 0, 0, 0x0B, [], 0), (0x5, 0, 0, 0x0B, [], 2)]
        self.assertEqual(awc.bank_stems([0x5, 0x5], channels),
                         (["00000005", "00000005~2"], ["00000005~3", None, "00000005~4"]))


if __name__ == "__main__":
    unittest.main()
