"""Épinglage de VIctory sur la branche anyxboxx/fork-7b0-lot5 de LooperSalty/victory (fork fxceless/victory-fork plus sept PR) et réglages par défaut du lanceur."""
import configparser
import re
import subprocess
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
COMMIT_FORK = "fd021bf26a7003749d668e5d9bf6b234bfab2e47"
REGLAGES = (
    "VICTORY_SHADER_LOOP_LIMIT", "VICTORY_LATE_WAIT", "VICTORY_WIDE_BACK",
    "VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_CARD_BUFFERS_MB",
)
# Absentes du code de VIctory au commit épinglé : les poser ne change rien.
ABSENTES = (
    "VICTORY_RT_MISS", "VICTORY_WAVE32_READFIRSTLANE", "VICTORY_WAVE32_CROSSLANE", "VICTORY_SKIP_BVH_BUILDER",
    "VICTORY_CAPPED_DYNAMIC",
)


def _lanceur():
    return (RACINE / "tools" / "lancer_victory.ps1").read_text(encoding="utf-8")


def _variables_posees(texte):
    """Noms VICTORY_* posés par le lanceur : table des réglages et affectations $env:."""
    table = texte[texte.index("$tuned = [ordered]@{"):texte.index("foreach ($name in $tuned.Keys)")]
    noms = set(re.findall(r"^\s*(VICTORY_[A-Z0-9_]+)\s*=", table, re.MULTILINE))
    return noms | set(re.findall(r"\$env:(VICTORY_[A-Z0-9_]+)\s*=", texte))


def test_gitmodules_points_victory_to_the_maintainers_fork():
    declares = configparser.ConfigParser()
    declares.read(RACINE / ".gitmodules", encoding="utf-8")
    urls = {declares[s]["path"]: declares[s]["url"] for s in declares.sections()}
    assert urls["third_party/victory"].removesuffix(".git") == "https://github.com/LooperSalty/victory"


def test_gitlink_is_pinned_to_the_measured_commit():
    sortie = subprocess.run(["git", "ls-files", "-s", "third_party/victory"], cwd=RACINE, capture_output=True,
                            text=True, check=False).stdout
    if not sortie:
        return                                           # arbre sans historique git : rien à vérifier
    assert sortie.split()[1].startswith(COMMIT_FORK)


def test_launcher_sets_the_tuned_settings_and_can_turn_them_off():
    texte = _lanceur()
    for nom in REGLAGES:
        assert nom in texte, nom
    assert "$NoTunedSettings" in texte and "-not $NoTunedSettings" in texte


def test_launcher_does_not_set_the_relay():
    # Fork issue #89: the relay hides the characters' skin and overlaps the subtitles.
    assert "VICTORY_RELAY" not in _variables_posees(_lanceur())


def test_launcher_leaves_out_variables_victory_does_not_read():
    poses = _variables_posees(_lanceur())
    assert not poses & set(ABSENTES)


def test_launcher_turns_dred_on_only_when_asked():
    texte = _lanceur()
    assert "[switch]$Dred" in texte
    table = texte[texte.index("$tuned = [ordered]@{"):texte.index("foreach ($name in $tuned.Keys)")]
    assert "VICTORY_D3D12_DRED" not in table
    assert re.search(r"if \(\$Dred\) \{\s*\$env:VICTORY_D3D12_DRED = '1'", texte)


def test_every_variable_the_launcher_sets_is_read_by_victory():
    victory = RACINE / "third_party" / "victory"
    if not (victory / "Cargo.toml").is_file():
        pytest.skip("sous-module third_party/victory non récupéré")
    code = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in victory.rglob("*.rs")
                     if "target" not in f.relative_to(victory).parts)
    for nom in sorted(_variables_posees(_lanceur()) - {"VICTORY_GAME_DIR", "VICTORY_SKIP_PIPELINES"}):
        assert f'"{nom}"' in code, nom


def test_launcher_keeps_its_parameters_and_builds_with_the_diag_profile():
    texte = _lanceur()
    assert "$GameDir" in texte and "$SkipPipelines" in texte
    assert "--profile diag -p victory --features diagnostics" in texte
