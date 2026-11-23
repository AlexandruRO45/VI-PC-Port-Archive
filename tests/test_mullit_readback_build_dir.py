"""Check harness lookup without launching a GPU program."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import mullit_readback


class MullitBuildDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.build = Path(tmp.name).resolve() / "build"
        self.program = "vk_harness" + mullit_readback.EXE
        self.argv = ["mullit_readback.py", tmp.name, str(self.build)]

    def test_only_harness_is_needed_in_the_configuration_folder(self):
        config = self.build / "RelWithDebInfo"
        config.mkdir(parents=True)
        (config / self.program).touch()
        with (patch.object(sys, "argv", self.argv),
              patch.object(mullit_readback, "run", return_value=(77, "fixture stopped")) as run,
              redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO)):
            self.assertEqual(mullit_readback.main(), 77)
        self.assertEqual(run.call_args.args[0], config / self.program)

    def test_missing_harness_names_only_that_program(self):
        with (patch.object(sys, "argv", self.argv), patch.object(mullit_readback, "run") as run,
              self.assertRaises(SystemExit) as caught):
            mullit_readback.main()
        run.assert_not_called()
        self.assertIn(f"Test programs not found in {self.build}: {self.program}\n", str(caught.exception))
        self.assertIn("RelWithDebInfo", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
