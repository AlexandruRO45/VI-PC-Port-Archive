"""The `common.rpf` counts of docs/gamedata_graph.md, re-measured on the reader's own package.

    LEONIDA_GAME_DIR=F:\\CLGame python -I tests/test_gamedata_graph_corpus.py

SKIPs without LEONIDA_GAME_DIR. The carve goes to a temporary folder and only counts are compared: no archive, XML
value, content-set id, target string or file name leaves this test, so the failure output quotes nothing.
"""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gamedata_graph
import gamedata_xml
import rpf9_carve


@unittest.skipUnless(os.environ.get("LEONIDA_GAME_DIR"), "set LEONIDA_GAME_DIR to check the page's counts")
class CommonRpfGraphTest(unittest.TestCase):
    def test_the_counts_docs_gamedata_graph_md_records(self):
        archive = Path(os.environ["LEONIDA_GAME_DIR"]) / "common.rpf"
        if not archive.is_file():
            self.skipTest(f"{archive} is not in this package")
        with tempfile.TemporaryDirectory(prefix="leonida-graph-") as tmp:
            carve = os.path.join(tmp, "common")
            with redirect_stdout(io.StringIO()):        # the carve and the graph both print their own summary
                self.assertEqual(rpf9_carve.main([str(archive), carve]), 0)
            summary = gamedata_graph.build_graph(list(gamedata_xml.load_documents([carve])))["summary"]
            self.assertEqual(gamedata_xml.document_mode([carve]), "carve")

        # "Measured shape": the payloads, the roots and the manifest layer.
        self.assertEqual((summary["payloads"], summary["documents"], summary["parsed"], summary["unparsed"]),
                         (267, 266, 259, 8))
        self.assertEqual((summary["root_count"], summary["rootless"]), (118, 1))
        self.assertEqual(summary["manifest_documents"], 25)
        self.assertEqual((summary["roots"]["CDataFileMgr__ContentChangeSetDef"]["manifest_documents"],
                          summary["roots"]["CDataFileMgr__ContentDef"]["manifest_documents"]), (23, 2))

        # "Graph model": content-set ids, reference sites and the renames the page counts.
        self.assertEqual((summary["content_sets"], summary["defined_content_sets"], summary["dangling_content_sets"],
                          summary["duplicate_set_ids"]), (30, 28, 2, 0))
        self.assertEqual((summary["reference_sites"], summary["target_sites"], summary["include_sites"]),
                         (3400, 3386, 14))
        self.assertEqual(summary["references"], {"filename": 3315, "includedSets": 14, "loadPath": 28,
                                                 "parentFile": 3, "parentPath": 40})
        self.assertEqual(summary["nodes"], {"documents": 267, "content_sets": 30, "targets": 3364, "total": 3661})
        self.assertEqual(summary["edges"]["total"], 3428)
        self.assertEqual({kind: summary["edges"][kind] for kind in
                          ("defines", "includes", "contains_file", "loads_changeset", "parent_path", "parent_file")},
                         {"defines": 28, "includes": 14, "contains_file": 3315, "loads_changeset": 28,
                          "parent_path": 40, "parent_file": 3})
        self.assertEqual((summary["edges"]["document_edges"], summary["edges"]["target_edges"]), (42, 3386))
        self.assertEqual(summary["unique"]["targets"], 3364)

        # "Prefix rules" and "Unresolved references": the four documented rules and `common:/` left alone.
        self.assertEqual(summary["prefixes"], {"$(common)": 587, "$(root)": 28, "$(target)": 23, "common:/": 31,
                                               "platform:/levels": 2346, "platform:/other": 328,
                                               "relative/locator": 43})
        self.assertEqual(summary["resolution"], {"document": 12, "document_unmapped": 3312,
                                                 "include_id_missing": 2, "relative_locator": 43,
                                                 "unrecognized_prefix": 31})

        # The include walk across the 25 manifests.
        self.assertEqual((summary["content_graph"]["root_documents"],
                          summary["content_graph"]["reachable_documents"],
                          summary["content_graph"]["include_edges"],
                          summary["content_graph"]["max_depth"], summary["content_graph"]["cycles"]),
                         (13, 25, 12, 1, 0))
        self.assertEqual((summary["reachable"]["content_sets"], summary["reachable"]["targets"],
                          summary["reachable"]["target_sites"]), (30, 3343, 3343))

        # "What the locators point at": parentPath is resolved inside its own document, never as a document edge.
        inherit = summary["parent_path_inheritance"]
        self.assertEqual((inherit["documents"], inherit["sites"], inherit["resolved_in_document"],
                          inherit["ambiguous"], inherit["unresolved"], inherit["cycles"]), (1, 40, 40, 0, 0, 0))
        self.assertEqual(inherit["chain_depths"], {"1": 35, "2": 3, "3": 1, "4": 1})  # the tool stringifies depths


if __name__ == "__main__":
    unittest.main()
