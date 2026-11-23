"""Unit tests for tools/gallery.py standalone 3D Studio & Asset Gallery server."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import gallery


class TestGalleryCategorization(unittest.TestCase):
    def test_categories(self) -> None:
        self.assertEqual(gallery.categorize_model("ped_lucia_body.glb"), "Персонажи (Characters)")
        self.assertEqual(gallery.categorize_model("veh_infernus.glb"), "Транспорт (Vehicles)")
        self.assertEqual(gallery.categorize_model("w_rifle_m4.glb"), "Оружие (Weapons)")
        self.assertEqual(gallery.categorize_model("build_wall_01.glb"), "Карта и Здания (World)")
        self.assertEqual(gallery.categorize_model("prop_chair_office.glb"), "Объекты и Пропы (Props)")
        self.assertEqual(gallery.categorize_model("tree_palm_02.glb"), "Растительность (Nature)")
        self.assertEqual(gallery.categorize_model("misc_something.glb"), "Прочее (Miscellaneous)")

    def test_folder_inference(self) -> None:
        self.assertEqual(gallery.categorize_model("asset_01.glb", "clean_vehicles"), "Транспорт (Vehicles)")
        self.assertEqual(gallery.categorize_model("model.glb", "unpacked_peds"), "Персонажи (Characters)")


class TestGalleryServer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.tmp.name, "models")
        self.textures_dir = os.path.join(self.tmp.name, "textures")
        self.viewer_dir = os.path.join(self.tmp.name, "viewer")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.textures_dir, exist_ok=True)
        os.makedirs(self.viewer_dir, exist_ok=True)

        # Create dummy assets
        with open(os.path.join(self.models_dir, "test_car.glb"), "wb") as f:
            f.write(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")

        with open(os.path.join(self.textures_dir, "test_tex.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        with open(os.path.join(self.viewer_dir, "studio.html"), "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><body><h1>Studio</h1></body></html>")

        # Configure handler
        gallery.StudioHandler.models_dir = self.models_dir
        gallery.StudioHandler.textures_dir = self.textures_dir
        gallery.StudioHandler.viewer_dir = self.viewer_dir

        # Start server on an open port
        addr = ("127.0.0.1", 0)
        self.httpd = gallery.socketserver.TCPServer(addr, gallery.StudioHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_endpoints(self) -> None:
        base = f"http://127.0.0.1:{self.port}"

        # 1. Studio HTML root
        with urllib.request.urlopen(f"{base}/", timeout=3) as r:
            self.assertEqual(r.status, 200)
            body = r.read().decode("utf-8")
            self.assertIn("<h1>Studio</h1>", body)

        # 2. Models API
        with urllib.request.urlopen(f"{base}/api/models", timeout=3) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["models"][0]["name"], "test_car.glb")
            self.assertEqual(data["models"][0]["category"], "Транспорт (Vehicles)")

        # 3. Textures API
        with urllib.request.urlopen(f"{base}/api/textures", timeout=3) as r:
            self.assertEqual(r.status, 200)
            tdata = json.loads(r.read())
            self.assertEqual(tdata["count"], 1)
            self.assertEqual(tdata["textures"][0]["name"], "test_tex.png")

        # 4. Binary serving
        with urllib.request.urlopen(f"{base}/models/test_car.glb", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"glTF"))

        with urllib.request.urlopen(f"{base}/textures/test_tex.png", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"\x89PNG"))

    def test_unlistable_subfolder_is_reported_not_silently_skipped(self) -> None:
        # os.walk skips a directory it cannot list unless onerror reports it; the listing used to drop
        # the sub-folder's assets and return 200 with no indication the scan was incomplete.
        msub = os.path.join(self.models_dir, "locked_models")
        os.makedirs(msub)
        with open(os.path.join(msub, "hidden.glb"), "wb") as f:
            f.write(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")
        tsub = os.path.join(self.textures_dir, "locked_tex")
        os.makedirs(tsub)
        with open(os.path.join(tsub, "hidden.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        os.chmod(msub, 0o000)
        os.chmod(tsub, 0o000)
        try:
            try:
                os.listdir(msub)
                self.skipTest("the sub-folder is still listable (running as root, or Windows)")
            except OSError:
                pass
            base = f"http://127.0.0.1:{self.port}"
            with urllib.request.urlopen(f"{base}/api/models", timeout=3) as r:
                data = json.loads(r.read())
            self.assertTrue(any(msub in e for e in data.get("scan_errors", [])), data.get("scan_errors"))
            self.assertEqual(data["count"], 1)  # the root model is still listed
            with urllib.request.urlopen(f"{base}/api/textures", timeout=3) as r:
                tdata = json.loads(r.read())
            self.assertTrue(any(tsub in e for e in tdata.get("scan_errors", [])), tdata.get("scan_errors"))
        finally:
            os.chmod(msub, 0o755)
            os.chmod(tsub, 0o755)

    def test_unlistable_subfolder_is_reported_when_chmod_is_not_enough(self) -> None:
        # The chmod case above self-skips on Windows and as root, and CI does not exercise the
        # guard there. Forcing os.scandir to raise for the sub-folders hands os.walk the same
        # PermissionError a real lockout would, on every platform.
        msub = os.path.join(self.models_dir, "locked_models")
        os.makedirs(msub)
        with open(os.path.join(msub, "hidden.glb"), "wb") as f:
            f.write(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")
        tsub = os.path.join(self.textures_dir, "locked_tex")
        os.makedirs(tsub)
        with open(os.path.join(tsub, "hidden.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        real_scandir = os.scandir
        locked = {os.path.normcase(os.path.abspath(msub)), os.path.normcase(os.path.abspath(tsub))}

        def scandir(path, *args, **kwargs):
            if os.path.normcase(os.path.abspath(str(path))) in locked:
                raise PermissionError(13, "Access is denied", str(path))
            return real_scandir(path, *args, **kwargs)

        base = f"http://127.0.0.1:{self.port}"
        with mock.patch.object(gallery.os, "scandir", scandir):
            with urllib.request.urlopen(f"{base}/api/models", timeout=3) as r:
                data = json.loads(r.read())
            with urllib.request.urlopen(f"{base}/api/textures", timeout=3) as r:
                tdata = json.loads(r.read())
        self.assertTrue(any(msub in e for e in data.get("scan_errors", [])), data.get("scan_errors"))
        self.assertEqual(data["count"], 1)  # the root model is still listed
        self.assertTrue(any(tsub in e for e in tdata.get("scan_errors", [])), tdata.get("scan_errors"))
        self.assertEqual(tdata["count"], 1)


if __name__ == "__main__":
    unittest.main()
