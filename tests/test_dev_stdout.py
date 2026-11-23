"""scripts/dev.py must not require a reconfigurable stdout."""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts" / "dev.py"


def load_dev_module():
    spec = importlib.util.spec_from_file_location("leonida_dev_stdout", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DevStdoutTests(unittest.TestCase):
    def test_discover_does_not_require_a_reconfigurable_stdout(self):
        # A captured stdout (io.StringIO) has no reconfigure(): main() must
        # still print the inventory instead of raising AttributeError.
        module = load_dev_module()
        captured = io.StringIO()
        saved_argv = sys.argv
        sys.argv = ["dev.py", "discover"]
        try:
            with contextlib.redirect_stdout(captured):
                code = module.main()
        finally:
            sys.argv = saved_argv
        self.assertEqual(code, 0)
        self.assertIn("boot-trace", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
