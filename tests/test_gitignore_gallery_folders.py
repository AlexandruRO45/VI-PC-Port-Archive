"""The folders tools/gallery.py reads at the checkout root are ignored, whatever the file extension.

    python -I tests/test_gitignore_gallery_folders.py

tools/gallery.py looks for models and textures in <checkout>/models and <checkout>/textures first, and lists
.glb/.gltf and .dds/.png/.jpg/.webp files there. .gitignore's "never commit the game" block ignores *.dds and
*.glb by extension only, so a converted PNG texture or a glTF export dropped there was one `git add .` away from
a commit. Asks git itself (`git check-ignore --no-index`), so no file is created; skips when git is missing.
"""
import os
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every extension gallery.py lists in each folder, plus a nested path.
GALLERY_PATHS = [
    "models/a.glb", "models/a.gltf", "models/sub/a.gltf",
    "textures/a.dds", "textures/a.png", "textures/a.jpg", "textures/a.webp", "textures/sub/a.webp",
]
# Folders of the same names deeper in the tree are the repo's own; the root anchor leaves them alone.
NOT_IGNORED = ["tools/models/a.py", "docs/textures/a.md"]


def ignored(path):
    """True when git would ignore `path` under the checkout (the path need not exist)."""
    r = subprocess.run(["git", "-C", ROOT, "check-ignore", "-q", "--no-index", path],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore {path}: {r.stderr.strip()}")
    return r.returncode == 0


@unittest.skipIf(shutil.which("git") is None, "git is not on PATH")
class GalleryFoldersIgnoredTest(unittest.TestCase):
    def test_gallery_folders_are_ignored_for_every_extension(self):
        for path in GALLERY_PATHS:
            with self.subTest(path=path):
                self.assertTrue(ignored(path), f"{path} is not ignored")

    def test_same_named_folders_elsewhere_are_not(self):
        for path in NOT_IGNORED:
            with self.subTest(path=path):
                self.assertFalse(ignored(path), f"{path} is ignored")


if __name__ == "__main__":
    unittest.main()
