"""Names of the hand-written GPU fixtures that the shader builder selects by default."""
from pathlib import Path


# These three cases are built from vs_clip.asm with different signatures.
CLIP_CASES = {"vs_clip": 1, "vs_clip_param4": 4, "vs_clip_undeclared": 1}


def fixture_names(directory):
    return {path.stem for path in Path(directory).glob("*.asm")} | CLIP_CASES.keys()
