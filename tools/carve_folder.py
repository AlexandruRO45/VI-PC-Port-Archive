"""Is a folder of carved streams a finished rpf9_carve.py run?

    from carve_folder import UNFINISHED, unfinished      # after sys.path.insert(0, <this folder>), as -I needs

rpf9_carve.py writes `carve_unfinished.json` into its output folder before the first stream and removes it after
`index.json` is in place, so a carve that was stopped (Ctrl+C, a full disk, a crash) leaves the marker behind. The
tools that index a folder by listing its `<offset>.bin` files can't tell a stopped carve from a finished one any
other way: they would report the streams that happen to be there as the archive's.

A folder without the marker is not known to be complete: one written by hand, by a test or by an older
rpf9_carve.py has none. Standard library only; no game data.
"""
import json
import os

UNFINISHED = "carve_unfinished.json"   # ends in .json: the folder indexers already skip .json and .csv files


def unfinished(folder):
    """A one-line reason when `folder` holds the marker of a carve that did not finish, else None."""
    marker = os.path.join(folder, UNFINISHED)
    if not os.path.isfile(marker):
        return None
    archive = ""
    try:
        with open(marker, encoding="utf-8") as f:
            archive = str(json.load(f).get("archive") or "")
    except (OSError, ValueError, AttributeError):
        pass
    of = f" of {archive}" if archive else ""
    return (f"{folder} holds an unfinished carve{of} ({UNFINISHED}): its streams are not all of the archive. "
            "Run rpf9_carve.py on it again and let it finish")
