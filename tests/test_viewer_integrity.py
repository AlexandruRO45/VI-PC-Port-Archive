"""Subresource integrity checks for three.js and CDN imports in viewer/index.html (#8618).

    python -I tests/test_viewer_integrity.py

Verifies that viewer/index.html enforces Subresource Integrity (SRI) hashes on all
remote CDN assets loaded through its import map:
- three.module.js (three@0.160.0)
- GLTFLoader.js (three@0.160.0 jsm addon)
- BufferGeometryUtils.js (three@0.160.0 jsm utility dependency)
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "viewer" / "index.html"

SRI_HASH_RE = re.compile(r"^sha(256|384|512)-[A-Za-z0-9+/=]+$")


class ViewerIntegrityTest(unittest.TestCase):
    def test_index_html_importmap_has_integrity(self):
        content = INDEX_HTML.read_text(encoding="utf-8")
        match = re.search(r'<script type="importmap">\s*(\{[\s\S]*?\})\s*</script>', content)
        self.assertIsNotNone(match, "index.html must contain an import map script tag")

        data = json.loads(match.group(1))
        self.assertIn("imports", data, "import map must contain 'imports'")
        self.assertIn("integrity", data, "import map must contain 'integrity' for CDN security")

        imports = data["imports"]
        integrity = data["integrity"]

        self.assertIn("three", imports)
        three_url = imports["three"]
        self.assertIn(three_url, integrity, f"integrity entry missing for {three_url}")
        self.assertRegex(integrity[three_url], SRI_HASH_RE)

        # GLTFLoader and its dependency BufferGeometryUtils must have integrity entries
        addons_prefix = imports.get("three/addons/", "")
        gltf_url = f"{addons_prefix}loaders/GLTFLoader.js"
        buffer_geom_url = f"{addons_prefix}utils/BufferGeometryUtils.js"

        self.assertIn(gltf_url, integrity, f"integrity entry missing for {gltf_url}")
        self.assertRegex(integrity[gltf_url], SRI_HASH_RE)

        self.assertIn(buffer_geom_url, integrity, f"integrity entry missing for {buffer_geom_url}")
        self.assertRegex(integrity[buffer_geom_url], SRI_HASH_RE)


if __name__ == "__main__":
    unittest.main()
