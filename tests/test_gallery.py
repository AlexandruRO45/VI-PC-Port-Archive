"""Unit tests for tools/gallery.py standalone 3D Studio & Asset Gallery server."""
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

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

    def _status(self, path: str) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_paths_outside_the_roots_are_refused(self) -> None:
        secret = os.path.join(self.tmp.name, "secret.txt")
        with open(secret, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("SECRET")
        sibling_models = os.path.join(self.tmp.name, "models_extra")
        os.makedirs(sibling_models)
        with open(os.path.join(sibling_models, "x.glb"), "wb") as handle:
            handle.write(b"GLBEXTRA")
        sibling_viewer = os.path.join(self.tmp.name, "viewer_extra")
        os.makedirs(sibling_viewer)
        with open(os.path.join(sibling_viewer, "note.txt"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("SIBLING")
        os.makedirs(os.path.join(self.viewer_dir, "vendor"))
        with open(os.path.join(self.viewer_dir, "vendor", "app.js"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("console.log(1)\n")

        status, body = self._status("/vendor/app.js")
        self.assertEqual(status, 200)
        self.assertIn(b"console.log", body)

        for path in (
            "/vendor/../../secret.txt",
            "/../viewer_extra/note.txt",
            "/models/../models_extra/x.glb",
            "/textures/../secret.txt",
        ):
            status, body = self._status(path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"SECRET", body)
            self.assertNotIn(b"SIBLING", body)
            self.assertNotIn(b"GLBEXTRA", body)

        link = os.path.join(self.models_dir, "link.glb")
        try:
            os.symlink(secret, link)
        except OSError:
            return
        status, body = self._status("/models/link.glb")
        self.assertEqual(status, 404)
        self.assertNotIn(b"SECRET", body)


if __name__ == "__main__":
    unittest.main()
