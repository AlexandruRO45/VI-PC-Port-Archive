"""Outil du premier lancement (tools/jour_j.py) sur un paquet synthétique : inventaire, fournisseurs
de chaque import, lancements isolé et natif, rapport écrit hors du dépôt."""
import sys
from pathlib import Path

from pe_builder import Asm, build_pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import jour_j  # noqa: E402

IMPORTS = [("xmem.dll", "XMemAlloc"), ("kernel32.dll", "GetTickCount"), ("inconnu.dll", "Rien")]


def make_package(folder):
    folder.mkdir(parents=True, exist_ok=True)
    build_pe(folder / "game.exe", lambda iat: Asm().mov_eax(7).ret().build(), imports=IMPORTS, dll=False)
    (folder / "MicrosoftGame.config").write_text(
        '<Game><ExecutableList><Executable Name="game.exe" TargetDeviceFamily="Scarlett" /></ExecutableList></Game>',
        encoding="utf-8")
    return folder


def test_report_names_who_provides_each_import(tmp_path, runtime_dir):
    package = make_package(tmp_path / "paquet")
    text = jour_j.report(package, tmp_path / "sortie")
    assert "# Premier lancement : game.exe" in text and "SHA-256" in text
    assert "| `inconnu.dll` | 1 |" in text and "`inconnu.dll` manquantes : `Rien`" in text
    assert "| `xmem.dll` | 1 | 0 | 1 |" in text                 # servi par AnyXboxX
    assert "| `kernel32.dll` | 1 | 0 | 0 | 0 | 1 | 0 |" in text  # servi par Windows
    assert "## Analyse isolée (--rapport)" in text and "## Lancement isolé dans Unicorn (--lancer)" in text


def test_native_step_is_opt_in_and_reports_the_outcome(tmp_path, runtime_dir):
    package = make_package(tmp_path / "paquet")
    assert "Lancement natif" not in jour_j.report(package, tmp_path / "a")
    text = jour_j.report(package, tmp_path / "b", native=True, duration=30)
    assert "## Lancement natif (anyxboxx.exe)" in text and "inconnu.dll" in text


def test_missing_executable_is_reported(tmp_path, capsys):
    (tmp_path / "MicrosoftGame.config").write_text(
        '<Game><ExecutableList><Executable Name="absent.exe" TargetDeviceFamily="Scarlett" /></ExecutableList></Game>', encoding="utf-8")
    assert jour_j.main([str(tmp_path), "--sortie", str(tmp_path / "out")]) == 1
    assert "absent.exe" in capsys.readouterr().err
