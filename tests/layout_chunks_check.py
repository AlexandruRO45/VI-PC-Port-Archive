#!/usr/bin/env python3
"""Check the per-chunk layout summary emitted by verify_build.compare_layout."""

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))

from verify_build import compare_layout

LAYOUT = r"""<?xml version="1.0" encoding="utf-8"?>
<Package>
  <Chunk Id="1000">
    <FileGroup SourcePath=".\" DestinationPath="\" Include="MicrosoftGame.Config" />
  </Chunk>
  <Chunk Id="1001">
    <FileGroup SourcePath=".\..\..\content\" DestinationPath="\" Include="game_scarlett_master.exe" />
  </Chunk>
  <Chunk Id="3009" Marker="Launch">
    <FileGroup SourcePath=".\" DestinationPath="\" Include="Update.alignmentchunk" />
  </Chunk>
</Package>
"""


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "layout_test.xml"
        path.write_text(LAYOUT, encoding="utf-8")
        report = compare_layout(
            path,
            {"MicrosoftGame.Config", "Update.alignmentchunk"},
            {"MicrosoftGame.Config", "Update.alignmentchunk"},
        )
    assert report["target_count"] == 3, report
    assert report["launch_marker"] == "3009", report
    assert report["absent_from_disk"] == {"game_scarlett_master.exe": ["1001"]}, report
    assert report["absent_from_torrent"] == {"game_scarlett_master.exe": ["1001"]}, report
    assert report["torrent_paths_outside_layout"] == [], report
    chunks = {chunk["id"]: chunk for chunk in report["chunk_summary"]}
    assert chunks["1001"]["absent_from_disk"] == ["game_scarlett_master.exe"], chunks
    assert chunks["1001"]["absent_from_torrent"] == ["game_scarlett_master.exe"], chunks
    assert chunks["3009"]["marker"] == "Launch", chunks
    assert chunks["3009"]["absent_from_disk"] == [], chunks
    print("layout chunk summary ok")


if __name__ == "__main__":
    main()
