"""run_gpu_tests.py gives each per-lane dump test a vk_compute_test buffer that holds every word its lanes store.

    python -I tests/test_run_gpu_out_words.py

vk_compute_test makes a buffer of 64 words for one group unless VKC_OUT_WORDS asks for more, and its V# has the
maximum num_records, so the translator's bounds check does not stop a store past the buffer: the store goes to a
device address outside it. cs_pk_mul_lo_const has 32 lanes that store four words each (512 bytes), and GPU-assisted
validation reported VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819 for lanes 16..31. Here the suite runs with the
harness stood in for (as tests/test_run_gpu_readback_size.py does): no device, no llvm-mc, no game files.
"""
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "runtime" / "test" / "shaders"
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests as rgt  # noqa: E402
from check import CMPX_CLASS_WORDS, CVT_PK_U8_WORDS, PK_MUL_LO_WORDS  # noqa: E402

# The per-lane dump tests: every lane stores at its lane id shifted left (v_lshlrev_b32), plus the store's offset.
PER_LANE = {"cs_pk_mul_lo_const": PK_MUL_LO_WORDS, "cs_cvt_pk_u8": CVT_PK_U8_WORDS,
            "cs_cmpx_class_f32": CMPX_CLASS_WORDS}


def stored_bytes(name: str) -> int:
    """The end of the last byte the fixture's lanes store, from its group size, lane shift and store offsets."""
    text = (SHADERS / f"{name}.asm").read_text(encoding="utf-8")
    lanes = math.prod(int(n) for n in re.search(r"^; group: (\d+),(\d+),(\d+)", text, re.MULTILINE).groups())
    stores = re.findall(r"^buffer_store_dword v\d+, (v\d+), s\[0:3\], 0 offen(?: offset:(\d+))?$", text, re.MULTILINE)
    address = {register for register, _ in stores}
    assert len(address) == 1, f"{name}: stores through {sorted(address)}"
    shift = re.search(rf"^v_lshlrev_b32_e32 {address.pop()}, (\d+), v\d+$", text, re.MULTILINE)
    last = max(int(offset or 0) for _, offset in stores)
    return (lanes - 1) * (1 << int(shift.group(1))) + last + 4


class OutWordsTest(unittest.TestCase):
    def run_test(self, name: str, words: list[int]) -> tuple[int, str, dict | None]:
        """run_suite on one dump test; returns its result, its output and the environment vk_compute_test got."""
        seen: list[dict | None] = []
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            args = SimpleNamespace(filter=f"{name}.asm", build=directory, llvm_mc="unused")

            def launch(cmd: list[str], env: dict | None = None, **kwargs: object) -> tuple[int, str]:
                if cmd[0] == sys.executable:
                    (work / "spirv" / f"{name}.spv").touch()
                    return 0, ""
                self.assertEqual(Path(cmd[0]).stem, "vk_compute_test")
                seen.append(env)
                return 0, "out[0..7] = " + " ".join(str(w) for w in words)

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=launch),
                  redirect_stdout(output)):
                result = rgt.run_suite(args, directory)
        self.assertEqual(len(seen), 1, output.getvalue())
        return result, output.getvalue(), seen[0]

    def test_the_buffer_holds_every_stored_word(self) -> None:
        self.assertEqual(stored_bytes("cs_pk_mul_lo_const"), 512)  # 32 lanes, 16 bytes each
        for name, words in PER_LANE.items():
            with self.subTest(test=name):
                result, output, env = self.run_test(name, words)
                self.assertEqual(result, 0, output)
                buffer_words = max(64, int((env or {}).get("VKC_OUT_WORDS", 0)))  # one group: 64 words at least
                self.assertGreaterEqual(buffer_words * 4, stored_bytes(name), output)


if __name__ == "__main__":
    unittest.main()
