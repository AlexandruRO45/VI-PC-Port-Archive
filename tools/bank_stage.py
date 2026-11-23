"""Hidden staging folders of awc_opus.py, and the rule every walker of the Ogg tree uses to skip them.

A fresh bank is written under `.awc-bank-<bank name>` next to its final folder and renamed into place. A killed run
can leave that folder behind with a truncated file in it, so anything that walks the export tree (audio_wav,
opus_tags) must not treat it as a bank.
"""
STAGE_PREFIX = ".awc-bank-"


def is_stage_name(name):
    return name.startswith(STAGE_PREFIX)


def prune_stage_dirs(dirs):
    """For os.walk: drop staging folders from `dirs` in place so the walk does not enter them."""
    dirs[:] = [d for d in dirs if not is_stage_name(d)]
