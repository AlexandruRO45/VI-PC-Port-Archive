"""Tests for the network-mount guard in make_game --find (#8344).

A dead nfs/cifs/sshfs mount blocks os.scandir()/stat() in D-state, where even SIGTERM cannot stop the process, so
find_dumps() must skip network mount points lexically, before any isdir()/scandir()/stat() call.

    python -I tests/test_make_game_find_mounts.py
"""
import contextlib
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402


class FakeDirEntry:
    """The two methods find_dumps() calls, counting them so the test can prove none ran on the network mount."""

    def __init__(self, path, kind="dir"):
        self.path = path
        self.name = os.path.basename(path)
        self.kind = kind
        self.calls = []

    def is_dir(self, follow_symlinks=False):
        self.calls.append("is_dir")
        return self.kind == "dir"

    def is_file(self, follow_symlinks=False):
        self.calls.append("is_file")
        return self.kind == "file"


class MountTableParserTest(unittest.TestCase):
    def test_network_file_systems_are_kept_and_local_ones_ignored(self):
        text = "\n".join((
            "rootfs / rootfs rw 0 0",
            "/dev/sda1 / ext4 rw 0 0",
            "tmpfs /tmp tmpfs rw 0 0",
            "server:/export /mnt/nfs nfs4 rw 0 0",
            "//server/share /mnt/cifs cifs rw 0 0",
            "user@host:/ /mnt/ssh fuse.sshfs rw 0 0",
            "server:/srv /run/media/user/Backup nfs rw 0 0",
            "malformed line",
        ))
        mounts = make_game.network_mounts_from_mounts_text(text)
        self.assertEqual(set(mounts), {"/mnt/nfs", "/mnt/cifs", "/mnt/ssh", "/run/media/user/Backup"})
        self.assertEqual(list(mounts), sorted(mounts))

    def test_octal_escapes_from_proc_are_decoded(self):
        text = r"server:/srv /media/My\040Share nfs4 rw 0 0" + "\n" + r"server:/srv /mnt/back\134slash cifs rw 0 0"
        self.assertEqual(set(make_game.network_mounts_from_mounts_text(text)),
                         {"/media/My Share", "/mnt/back\\slash"})

    def test_empty_text_and_short_lines(self):
        self.assertEqual(make_game.network_mounts_from_mounts_text(""), ())
        self.assertEqual(make_game.network_mounts_from_mounts_text("only one field\n\n"), ())

    def test_network_mounts_is_empty_off_linux(self):
        with mock.patch.object(make_game.sys, "platform", "win32"):
            self.assertEqual(make_game.network_mounts(), ())

    def test_network_mounts_tolerates_an_unreadable_proc(self):
        with mock.patch.object(make_game, "open", side_effect=OSError, create=True):
            self.assertEqual(make_game.network_mounts(), ())


class UnderNetworkMountTest(unittest.TestCase):
    def test_exact_mount_and_children_match(self):
        mounts = ("/mnt/nfs",)
        self.assertTrue(make_game.under_network_mount("/mnt/nfs", mounts))
        self.assertTrue(make_game.under_network_mount("/mnt/nfs/GTA VI", mounts))
        self.assertTrue(make_game.under_network_mount("/mnt/nfs/GTA VI/package", mounts))

    def test_sibling_with_the_same_prefix_does_not_match(self):
        self.assertFalse(make_game.under_network_mount("/mnt/nfs2", ("/mnt/nfs",)))
        self.assertFalse(make_game.under_network_mount("/mnt/nfs2/GTA VI", ("/mnt/nfs",)))
        self.assertFalse(make_game.under_network_mount("/mnt/local", ("/mnt/nfs",)))

    def test_trailing_separator_and_empty_table(self):
        self.assertTrue(make_game.under_network_mount("/mnt/nfs/GTA VI", ("/mnt/nfs/",)))
        self.assertFalse(make_game.under_network_mount("/mnt/nfs", ()))


class FindDumpsSkipsNetworkMountTest(unittest.TestCase):
    """find_dumps() reaches a local copy but never scans or stats the network mount beside it."""

    def setUp(self):
        self.media = "/fake/media"
        self.mount = "/fake/media/nfs"
        self.local = "/fake/media/local"
        self.game = "/fake/media/local/GTA VI"
        self.config = os.path.join(self.game, "MicrosoftGame.Config")
        self.nfs_entry = FakeDirEntry(self.mount)
        self.local_entry = FakeDirEntry(self.local)
        self.game_entry = FakeDirEntry(self.game)
        self.entries = {
            self.media: [self.nfs_entry, self.local_entry],
            self.local: [self.game_entry],
        }
        self.scanned = []
        self.isdir_calls = []

    def fake_scandir(self, path):
        self.scanned.append(path)
        if path == self.mount or path.startswith(self.mount + os.sep):
            raise AssertionError(f"os.scandir({path!r}) entered the network mount")
        return contextlib.nullcontext(iter(self.entries.get(path, [])))

    def fake_isdir(self, path):
        self.isdir_calls.append(path)
        return path == self.media

    def fake_isfile(self, path):
        return path == self.config

    def test_local_copy_found_without_touching_the_mount(self):
        with mock.patch.object(make_game, "discovery_roots",
                               lambda home: [self.media, self.mount]), \
                mock.patch.object(make_game, "network_mounts", lambda: (self.mount,)), \
                mock.patch.object(make_game, "match_version",
                                  lambda d: "1.0.2.1" if d == self.game else None), \
                mock.patch.object(make_game.os, "scandir", self.fake_scandir), \
                mock.patch.object(make_game.os.path, "isdir", self.fake_isdir), \
                mock.patch.object(make_game.os.path, "isfile", self.fake_isfile):
            found = make_game.find_dumps()

        self.assertEqual(found, [(self.game, "1.0.2.1")])
        # The mount is neither a discovery root nor a scanned directory, and its DirEntry is never stat()ed.
        self.assertNotIn(self.mount, self.isdir_calls)
        self.assertFalse(any(p == self.mount or p.startswith(self.mount + os.sep) for p in self.scanned))
        self.assertEqual(self.nfs_entry.calls, [])
        # Positive control: the local sibling and the copy under it were traversed.
        self.assertIn(self.local, self.scanned)
        self.assertIn("is_dir", self.local_entry.calls)


if __name__ == "__main__":
    unittest.main()
