"""gamedata_xml.py must catalog synthetic XML without reading any game file.

    python -I tests/test_gamedata_xml.py

Fixtures are hand-written documents: nested elements, attributes, macro and platform references,
an include-like manifest, a tail-only fragment and a malformed document. No game data is used.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gamedata_xml.py")

ALPHA = (
    b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?>\n'
    b'<CDataFileMgr__ContentChangeSetDef version="1">\n'
    b'  <id>fixture</id>\n'
    b'  <fileList>\n'
    b'    <Item type="file"><filename>$(common)/movies/fixture.bk2</filename>'
    b'<fileType>bk2</fileType></Item>\n'
    b'    <Item type="file"><filename>platform:/levels/fixture.xml</filename>'
    b'<fileType>xml</fileType></Item>\n'
    b'    <Item type="file"><filename>$(target)/data/fixture.xml</filename>'
    b'<fileType>xml</fileType></Item>\n'
    b'    <Item type="file"><filename>platform:/data/fixture.xml</filename>'
    b'<fileType>xml</fileType></Item>\n'
    b'  </fileList>\n'
    b'  <includedSets><Item ref="synthetic_set"><loadPath>$(root)/fixture</loadPath></Item>'
    b'</includedSets>\n'
    b'  <!-- synthetic note -->\n'
    b'</CDataFileMgr__ContentChangeSetDef>'
)
BETA = (
    b'<timecycle_modifier_data version="1"><modifier name="fixture">'
    b'<postfx_exposure>0 1</postfx_exposure></modifier></timecycle_modifier_data>'
)
TAIL = b'<Item type="x"/>\n</TailRoot>'
BAD = b'<?xml version="1.0"?><Root><Item></Root>'
ROOTLESS = b'plain text, not an XML document\n'


class GamedataXmlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = os.path.join(self.root, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=self.root,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_files_catalog_roots_elements_references_and_values(self):
        alpha = self.write("alpha.xml", ALPHA)
        beta = self.write("beta.xml", BETA)
        out = os.path.join(self.root, "catalog.json")
        csv_path = os.path.join(self.root, "catalog.csv")
        p = self.run_tool(alpha, beta, "--json", out, "--csv", csv_path, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["payloads"], 2)
        self.assertEqual(s["documents"], 2)
        self.assertEqual(s["incomplete"], 0)
        self.assertEqual(s["unattributed"], 0)
        self.assertEqual(s["well_formed"], 2)
        self.assertEqual(s["fragments"], 0)
        self.assertEqual(s["truncated"], 0)
        self.assertEqual(s["malformed"], 0)
        self.assertEqual(s["root_count"], 2)
        self.assertEqual(s["elements"], 21)
        self.assertEqual(s["max_depth"], 4)
        self.assertEqual(s["comments"], 1)
        self.assertEqual(s["comments_header"], 1)
        self.assertEqual(s["references"], {"$(common)": 1, "$(root)": 1, "$(target)": 1})
        self.assertEqual(s["platform"], {"levels": 1, "other": 1})
        self.assertEqual(s["platform_total"], 2)
        self.assertEqual(s["filename"], 4)
        self.assertEqual(s["include_like"]["fileList"], 1)
        self.assertEqual(s["include_like"]["includedSets"], 1)
        self.assertEqual(s["include_like"]["loadPath"], 1)
        self.assertEqual(s["xi_include"], 0)

        rows = {row["source"]: row for row in data["rows"]}
        first = rows["alpha.xml"]
        self.assertEqual(first["root"], "CDataFileMgr__ContentChangeSetDef")
        self.assertEqual(first["category"], "mount / content manifests")
        self.assertEqual(first["element_count"], 18)
        self.assertEqual(first["max_depth"], 4)
        self.assertEqual(first["attributes"], ["ref", "type", "version"])
        self.assertEqual(first["refs"], {"$(common)": 1, "$(root)": 1, "$(target)": 1})
        self.assertEqual(first["platform"], {"levels": 1, "other": 1})
        self.assertEqual(first["filename"], 4)
        self.assertTrue(first["header"])
        self.assertTrue(first["well_formed"])
        self.assertFalse(first["fragment"])
        self.assertEqual(first["signature"], (
            "CDataFileMgr__ContentChangeSetDef|ref,type,version|"
            "CDataFileMgr__ContentChangeSetDef,Item,fileList,fileType,filename,id,includedSets,loadPath"))

        second = rows["beta.xml"]
        self.assertEqual(second["root"], "timecycle_modifier_data")
        self.assertEqual(second["category"], "time, weather, sky, water")
        self.assertEqual(second["element_count"], 3)
        self.assertEqual(second["max_depth"], 3)
        self.assertEqual(second["attributes"], ["name", "version"])
        self.assertFalse(second["header"])
        self.assertEqual(second["signature"],
                         "timecycle_modifier_data|name,version|modifier,postfx_exposure,timecycle_modifier_data")

        root = data["roots"]["CDataFileMgr__ContentChangeSetDef"]
        self.assertEqual(root["files"], 1)
        self.assertEqual(root["documents"], 1)
        self.assertEqual(root["incomplete"], 0)
        self.assertEqual(root["header"], 1)
        self.assertEqual(root["signatures"], 1)
        self.assertEqual(root["category"], "mount / content manifests")
        self.assertEqual(root["elements"], 18)
        self.assertEqual(root["element_names"]["filename"], 4)
        self.assertEqual(root["element_names"]["Item"], 5)
        self.assertEqual(root["attributes"]["type"], 4)
        self.assertEqual(root["attributes"]["ref"], 1)
        self.assertEqual(root["refs"], {"$(common)": 1, "$(root)": 1, "$(target)": 1})
        self.assertEqual(root["platform"], {"levels": 1, "other": 1})
        self.assertEqual(root["filename"], 4)

        with open(csv_path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[0]["root"], "CDataFileMgr__ContentChangeSetDef")
        self.assertEqual(csv_rows[0]["refs"], "$(common)=1;$(root)=1;$(target)=1")
        self.assertEqual(csv_rows[0]["well_formed"], "true")
        self.assertTrue(csv_rows[0]["signature"].startswith("CDataFileMgr__ContentChangeSetDef|"))

        self.assertEqual(s["catalog"]["categories"], 12)
        self.assertEqual(s["catalog"]["unknown_roots"], 0)
        mount = s["categories"]["mount / content manifests"]
        self.assertEqual(mount["files"], 1)
        self.assertEqual(mount["documents"], 1)
        self.assertEqual(mount["roots"], 1)
        self.assertEqual(mount["header"], 1)
        self.assertEqual(mount["incomplete"], 0)
        self.assertEqual(sum(stats["files"] for stats in s["categories"].values()), s["payloads"])
        self.assertEqual(sum(stats["roots"] for stats in s["categories"].values()), s["root_count"])

        # Values, text and comments must not leak into the catalog; names and counts are fine.
        with open(out, encoding="utf-8") as f:
            blob = f.read()
        for value in ("fixture.bk2", "synthetic note", "synthetic_set", "0 1", "fixture.xml"):
            self.assertNotIn(value, blob)

    def test_shared_root_groups_signatures_and_bucket_invariants(self):
        shared_a = self.write("shared_a.xml", (
            b'<?xml version="1.0"?>\n<SharedRoot b="1" a="2">'
            b'<Group><Item value="1"/></Group></SharedRoot>'))
        shared_b = self.write("shared_b.xml", (
            b'<SharedRoot a="9" b="8"><Item value="2"/><Group value="3"/></SharedRoot>'))
        fragment = self.write("fragment.xml", b"not an XML document\n")
        out = os.path.join(self.root, "shared.json")
        p = self.run_tool(shared_a, shared_b, fragment, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["payloads"], 3)
        self.assertEqual(s["documents"], 2)
        self.assertEqual(s["incomplete"], 1)
        self.assertEqual(s["fragments"], 1)
        self.assertEqual(s["unattributed"], 1)
        self.assertEqual(s["root_count"], 1)
        self.assertEqual(s["catalog_roots"], 0)          # SharedRoot is synthetic and unmapped
        self.assertEqual(s["catalog"]["unknown_roots"], 1)

        rows = {row["source"]: row for row in data["rows"]}
        first, second, broken = rows["shared_a.xml"], rows["shared_b.xml"], rows["fragment.xml"]
        self.assertEqual(first["root"], "SharedRoot")
        self.assertEqual(second["root"], "SharedRoot")
        self.assertEqual(first["category"], "uncatalogued")
        self.assertEqual(first["signature"], second["signature"])
        self.assertEqual(first["signature"], "SharedRoot|a,b,value|Group,Item,SharedRoot")
        self.assertEqual(broken["root"], "")
        self.assertEqual(broken["category"], "uncatalogued")
        self.assertFalse(broken["complete"])
        self.assertTrue(broken["fragment"])
        self.assertEqual(broken["signature"], "")

        shared = data["roots"]["SharedRoot"]
        self.assertEqual(shared["files"], 2)
        self.assertEqual(shared["documents"], 2)
        self.assertEqual(shared["incomplete"], 0)
        self.assertEqual(shared["header"], 1)
        self.assertEqual(shared["signatures"], 1)
        self.assertEqual(shared["bytes"], os.path.getsize(shared_a) + os.path.getsize(shared_b))
        unattributed = data["roots"][""]
        self.assertEqual(unattributed["files"], 1)
        self.assertEqual(unattributed["documents"], 0)
        self.assertEqual(unattributed["incomplete"], 1)
        self.assertEqual(unattributed["category"], "uncatalogued")

        # Every payload lands in exactly one root bucket and one category bucket.
        self.assertEqual(sum(stats["files"] for stats in data["roots"].values()), s["payloads"])
        self.assertEqual(sum(stats["documents"] for stats in data["roots"].values()), s["documents"])
        self.assertEqual(sum(stats["bytes"] for stats in data["roots"].values()), s["bytes"])
        self.assertEqual(sum(stats["roots"] for stats in s["categories"].values()), s["root_count"])
        self.assertEqual(sum(stats["files"] for stats in s["categories"].values()), s["payloads"])
        self.assertEqual(sum(stats["documents"] for stats in s["categories"].values()), s["documents"])
        self.assertEqual(sum(stats["bytes"] for stats in s["categories"].values()), s["bytes"])
        self.assertEqual(s["categories"]["uncatalogued"]["unattributed"], 1)
        self.assertEqual(s["categories"]["uncatalogued"]["roots"], 1)
        self.assertEqual(s["categories"]["uncatalogued"]["files"], 3)

    def test_check_reports_tail_only_fragment_and_malformed(self):
        tail = self.write("tail.xml", TAIL)
        p = self.run_tool(tail, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("1 truncated at the start", p.stdout)
        self.assertIn("0 rootless fragments", p.stdout)
        self.assertIn("0 malformed", p.stdout)
        self.assertIn("TailRoot", p.stdout)

        rootless = self.write("rootless.xml", ROOTLESS)
        p = self.run_tool(rootless, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("1 rootless fragment", p.stdout)

        bad = self.write("bad.xml", BAD)
        p = self.run_tool(bad, "--check")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("1 malformed", p.stdout)
        self.assertIn("line 1", p.stdout)

    def test_carve_directory_joins_chunks_and_marks_fragment(self):
        carve = os.path.join(self.root, "carve")
        os.makedirs(carve)
        doc = b'<?xml version="1.0"?>\n<RootA>' + b"<Item value=\"1\"/>" * 20 + b"</RootA>"
        split = doc.index(b"<Item")
        parts = [doc[:split], doc[split:]]
        rows = []
        off = 0x1000
        for part in parts:
            with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            rows.append({"off": off, "stored": len(part), "size": len(part),
                         "head": part[:16].hex(), "pre": ""})
            off += 0x1000
        with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
            f.write(TAIL)
        rows.append({"off": off, "stored": len(TAIL), "size": len(TAIL),
                     "head": TAIL[:16].hex(), "pre": ""})
        off += 0x1000
        fragment = b"<Item>mid"
        with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
            f.write(fragment)
        rows.append({"off": off, "stored": len(fragment), "size": len(fragment),
                     "head": fragment[:16].hex(), "pre": ""})
        with open(os.path.join(carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

        out = os.path.join(self.root, "carve.json")
        p = self.run_tool(carve, "--json", out, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["mode"], "carve")
        self.assertEqual(s["payloads"], 3)
        self.assertEqual(s["documents"], 2)
        self.assertEqual(s["incomplete"], 1)
        self.assertEqual(s["well_formed"], 1)
        self.assertEqual(s["tail_only"], 1)
        self.assertEqual(s["truncated"], 1)
        self.assertEqual(s["fragments"], 1)
        self.assertEqual(s["unattributed"], 1)
        self.assertEqual(s["root_count"], 2)
        self.assertEqual(len(data["rows"]), 3)
        first = data["rows"][0]
        self.assertEqual(first["root"], "RootA")
        self.assertEqual(first["chunks"], 2)
        self.assertEqual(first["bytes"], len(doc))
        self.assertEqual(first["offset_hex"], "1000")
        self.assertEqual(first["element_count"], 21)
        self.assertEqual(first["max_depth"], 2)
        self.assertEqual(first["signature"], "RootA|value|Item,RootA")
        tail = data["rows"][1]
        self.assertEqual(tail["root"], "TailRoot")
        self.assertTrue(tail["tail_only"])
        self.assertTrue(tail["truncated"])
        self.assertFalse(tail["malformed"])
        self.assertEqual(tail["signature"], "")
        broken = data["rows"][2]
        self.assertEqual(broken["root"], "")
        self.assertTrue(broken["fragment"])
        self.assertFalse(broken["complete"])
        self.assertEqual(data["roots"][""]["incomplete"], 1)
        self.assertEqual(data["roots"]["RootA"]["signatures"], 1)
        self.assertEqual(sum(stats["files"] for stats in s["categories"].values()), 3)


if __name__ == "__main__":
    unittest.main()
