"""Structural and registration checks for the xcrdapi.dll!DStorageGetFactory shim.

    python -I tests/test_xcrdapi_shim.py

Verifies that runtime/shims/xcrdapi.cpp, runtime/shims/registry.cpp, runtime/CMakeLists.txt,
scripts/build_runtime.ps1, and runtime/test/xcrdapi_test.cpp stay wired together cleanly.
Standard library only; no build tools, GPU, or game files.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHIM_SRC = ROOT / "runtime" / "shims" / "xcrdapi.cpp"
REGISTRY_SRC = ROOT / "runtime" / "shims" / "registry.cpp"
CMAKE_SRC = ROOT / "runtime" / "CMakeLists.txt"
FALLBACK_PS1 = ROOT / "scripts" / "build_runtime.ps1"
CPP_TEST_SRC = ROOT / "runtime" / "test" / "xcrdapi_test.cpp"


class XcrdapiShimTests(unittest.TestCase):
    def test_shim_registers_dstorage_get_factory_and_stub_vtable(self):
        text = SHIM_SRC.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'SHIM_TABLE\(\s*xcrdapi\s*,\s*\{"xcrdapi\.dll",\s*"DStorageGetFactory",',
        )
        self.assertIn("if (!ppv) return E_POINTER;", text)
        self.assertIn("if (!riid) return E_INVALIDARG;", text)
        self.assertGreaterEqual(text.count("if (!supports_iid(*riid)) return E_NOINTERFACE;"), 2)
        self.assertNotIn("factory_iid", text)
        self.assertIn("IsEqualIID", text)
        self.assertIn("return E_NOTIMPL;", text)
        self.assertIn("factory_query_interface", text)
        self.assertIn("factory_add_ref", text)
        self.assertIn("factory_release", text)

    def test_registry_classifies_xcrdapi_dstorage_get_factory_as_hresult(self):
        text = REGISTRY_SRC.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'dll == "xcrdapi\.dll".*?\n\s*if \(name == "DStorageGetFactory"\) return ReturnClass::HResult;',
        )

    def test_cmake_and_fallback_compile_and_register_xcrdapi_shim(self):
        cmake = CMAKE_SRC.read_text(encoding="utf-8")
        for target in ("gta6rt", "pe_loader_test", "test_pe_loader_bounds", "xcrdapi_test"):
            with self.subTest(target=target):
                m = re.search(rf"add_executable\(\s*{re.escape(target)}\b(.*?)\)", cmake, re.S)
                self.assertIsNotNone(m, f"missing CMake target {target}")
                self.assertIn("shims/xcrdapi.cpp", m.group(1))
        self.assertIn("add_test(NAME xcrdapi COMMAND xcrdapi_test)", cmake)
        self.assertTrue(CPP_TEST_SRC.is_file())

        ps1 = FALLBACK_PS1.read_text(encoding="utf-8")
        self.assertIn(r'"shims\xcrdapi.cpp"', ps1)


if __name__ == "__main__":
    unittest.main()
