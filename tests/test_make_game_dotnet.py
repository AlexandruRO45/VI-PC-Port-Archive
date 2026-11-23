"""Windows .NET 8+ SDK discovery in make_game.py (#8434).

    python -I tests/test_make_game_dotnet.py
"""
import ntpath
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["dotnet", "--list-sdks"], returncode, stdout=stdout, stderr="")


class DotnetSdkDiscoveryTest(unittest.TestCase):
    def test_path_host_with_sdk_8_returns_bare_dotnet(self) -> None:
        x64 = r"C:\Program Files\dotnet\dotnet.exe"
        with mock.patch("os.name", "nt"), \
                mock.patch("shutil.which", return_value=x64), \
                mock.patch("subprocess.run", return_value=_completed("8.0.400 [C:\\Program Files\\dotnet\\sdk]\n")) as run:
            self.assertEqual(make_game.dotnet_sdk_exe(), "dotnet")
            self.assertTrue(make_game.dotnet_sdk_ok())
        self.assertEqual(run.call_args_list[0].args[0], [x64, "--list-sdks"])

    def test_windows_x86_on_path_falls_back_to_x64_program_files(self) -> None:
        x86 = r"C:\Program Files (x86)\dotnet\dotnet.exe"
        x64 = r"C:\Program Files\dotnet\dotnet.exe"
        existing = {ntpath.normcase(x86), ntpath.normcase(x64)}

        def fake_run(cmd, **_kw):
            if ntpath.normcase(cmd[0]) == ntpath.normcase(x64):
                return _completed("8.0.427 [C:\\Program Files\\dotnet\\sdk]\n")
            return _completed("")

        env = {
            "PATH": r"C:\Program Files (x86)\dotnet;C:\Windows\System32",
            "ProgramFiles": r"C:\Program Files",
        }
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("shutil.which", return_value=x86), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(p) in existing), \
                mock.patch("subprocess.run", side_effect=fake_run) as run:
            self.assertEqual(make_game.dotnet_sdk_exe(), x64)
            self.assertTrue(make_game.dotnet_sdk_ok())
        probed = [call.args[0][0] for call in run.call_args_list[:2]]
        self.assertEqual(probed, [x86, x64])

    def test_windows_x86_on_path_finds_later_custom_path_entry(self) -> None:
        x86 = r"C:\Program Files (x86)\dotnet\dotnet.exe"
        custom = r"D:\Tools\dotnet\dotnet.exe"
        existing = {ntpath.normcase(x86), ntpath.normcase(custom)}

        def fake_run(cmd, **_kw):
            if ntpath.normcase(cmd[0]) == ntpath.normcase(custom):
                return _completed("9.0.100 [D:\\Tools\\dotnet\\sdk]\n")
            return _completed("")

        env = {
            "PATH": r"C:\Program Files (x86)\dotnet;D:\Tools\dotnet",
            "ProgramFiles": r"C:\Program Files",
        }
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("shutil.which", return_value=x86), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(p) in existing), \
                mock.patch("subprocess.run", side_effect=fake_run):
            self.assertEqual(make_game.dotnet_sdk_exe(), custom)

    def test_windows_missing_from_path_checks_dotnet_root_and_programw6432(self) -> None:
        root_exe = r"E:\sdks\dotnet\dotnet.exe"
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, {"DOTNET_ROOT": r"E:\sdks\dotnet"}, clear=True), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(p) == ntpath.normcase(root_exe)), \
                mock.patch("subprocess.run", return_value=_completed("8.0.100 [E:\\sdks\\dotnet\\sdk]\n")):
            self.assertEqual(make_game.dotnet_sdk_exe(), root_exe)

        pf64_exe = r"C:\Program Files\dotnet\dotnet.exe"
        wow64_env = {
            "ProgramW6432": r"C:\Program Files",
            "ProgramFiles": r"C:\Program Files (x86)",
        }
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, wow64_env, clear=True), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(p) == ntpath.normcase(pf64_exe)), \
                mock.patch("subprocess.run", return_value=_completed("8.0.200 [C:\\Program Files\\dotnet\\sdk]\n")):
            self.assertEqual(make_game.dotnet_sdk_exe(), pf64_exe)

    def test_windows_per_user_install_dirs_are_probed(self) -> None:
        local_exe = r"C:\Users\alice\AppData\Local\Microsoft\dotnet\dotnet.exe"
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\alice\AppData\Local"}, clear=True), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(p) == ntpath.normcase(local_exe)), \
                mock.patch("subprocess.run", return_value=_completed("8.0.300 [sdk]\n")):
            self.assertEqual(make_game.dotnet_sdk_exe(), local_exe)

        home_exe = r"C:\Users\alice\.dotnet\dotnet.exe"
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, {"USERPROFILE": r"C:\Users\alice"}, clear=True), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(p) == ntpath.normcase(home_exe)), \
                mock.patch("subprocess.run", return_value=_completed("8.0.300 [sdk]\n")):
            self.assertEqual(make_game.dotnet_sdk_exe(), home_exe)

    def test_duplicate_candidates_are_probed_once(self) -> None:
        x64 = r"C:\Program Files\dotnet\dotnet.exe"
        env = {
            "PATH": r'"C:\PROGRAM FILES\DOTNET\";C:\Program Files\dotnet',
            "DOTNET_ROOT": r"C:\Program Files\dotnet\\",
            "ProgramW6432": r"C:\Program Files",
            "ProgramFiles": r"C:\Program Files",
        }
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("shutil.which", return_value=x64), \
                mock.patch("os.path.isfile", side_effect=lambda p: ntpath.normcase(ntpath.normpath(p)) == ntpath.normcase(x64)), \
                mock.patch("subprocess.run", return_value=_completed("")) as run:
            self.assertIsNone(make_game.dotnet_sdk_exe())
        self.assertEqual(run.call_count, 1)

    def test_nonzero_exit_and_older_sdk_are_rejected_and_bom_accepted(self) -> None:
        with mock.patch("os.name", "posix"), \
                mock.patch("shutil.which", return_value="/usr/bin/dotnet"), \
                mock.patch("subprocess.run", return_value=_completed("8.0.100 [/sdk]\n", returncode=1)):
            self.assertFalse(make_game.dotnet_sdk_ok())

        with mock.patch("os.name", "posix"), \
                mock.patch("shutil.which", return_value="/usr/bin/dotnet"), \
                mock.patch("subprocess.run", return_value=_completed("7.0.400 [/sdk]\n", returncode=0)):
            self.assertFalse(make_game.dotnet_sdk_ok())

        with mock.patch("os.name", "posix"), \
                mock.patch("shutil.which", return_value="/usr/bin/dotnet"), \
                mock.patch("subprocess.run", return_value=_completed("\ufeff8.0.100 [/sdk]\n", returncode=0)):
            self.assertEqual(make_game.dotnet_sdk_exe(), "dotnet")

    def test_rpf9dump_uses_resolved_windows_fallback_host(self) -> None:
        x64 = r"C:\Program Files\dotnet\dotnet.exe"
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "third_party", "rpf9-explorer", "Core"))
            with mock.patch.object(make_game, "ROOT", tmp), \
                    mock.patch.object(make_game, "dotnet_sdk_exe", return_value=x64), \
                    mock.patch.object(make_game, "log"), \
                    mock.patch.object(make_game, "run") as run:
                make_game.rpf9dump(tmp)
        self.assertEqual(run.call_args.args[0][:2], [x64, "build"])


if __name__ == "__main__":
    unittest.main()
