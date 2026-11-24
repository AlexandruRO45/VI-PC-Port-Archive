"""Épinglage de VIctory sur la branche anyxboxx/fork-7b0-lot9-murs de LooperSalty/victory (fork fxceless/victory-fork plus seize PR, dont la précompilation hors ligne, et la xg_xs.dll locale optionnelle comme oracle) et réglages par défaut du lanceur."""
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
COMMIT_FORK = "80d58f8fe2769d6c857eb27abc159b6cf64530a3"
REGLAGES = (
    "VICTORY_SHADER_LOOP_LIMIT", "VICTORY_LATE_WAIT", "VICTORY_WIDE_BACK",
    "VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_CARD_BUFFERS_MB", "VICTORY_DSTORAGE_NOTIFY",
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
    assert "$NoTunedSettings" in texte and "if ($NoTunedSettings)" in texte


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


def test_launcher_uses_microsofts_xg_library_only_when_given():
    texte = _lanceur()
    assert "[string]$XgDll" in texte
    assert "[ValidateSet('real', 'compare')][string]$XgMode = 'real'" in texte
    table = texte[texte.index("$tuned = [ordered]@{"):texte.index("foreach ($name in $tuned.Keys)")]
    assert "VICTORY_XG" not in table
    bloc = texte[texte.index("if ($PSBoundParameters.ContainsKey('XgDll') -and $XgDll) {"):texte.index("& $exe run")]
    assert "Test-Path -LiteralPath $XgDll -PathType Leaf" in bloc and "throw" in bloc
    assert "$env:VICTORY_XG_DLL =" in bloc and "$env:VICTORY_XG_MODE = $XgMode" in bloc


# Faux outil console (.NET Framework, compilé par Windows PowerShell) : copié en cargo.exe et en victory.exe, il note
# « nom arguments » dans FAKE_TOOL_LOG et rend le code de FAKE_TOOL_EXIT_<NOM> (0 par défaut).
FAUX_OUTIL = r"""
using System; using System.IO;
public static class FakeTool {
    public static int Main(string[] args) {
        string name = Path.GetFileNameWithoutExtension(System.Reflection.Assembly.GetEntryAssembly().Location);
        File.AppendAllText(Environment.GetEnvironmentVariable("FAKE_TOOL_LOG"),
                           name + " " + string.Join(" ", args) + Environment.NewLine);
        string code = Environment.GetEnvironmentVariable("FAKE_TOOL_EXIT_" + name.ToUpperInvariant());
        return string.IsNullOrEmpty(code) ? 0 : int.Parse(code);
    }
}
"""
CODE_ECHEC_CARGO = 101
CODE_SORTIE_VICTORY = 7


@pytest.fixture(scope="module")
def faux_outil(tmp_path_factory):
    if sys.platform != "win32" or not shutil.which("powershell"):
        pytest.skip("le lanceur est un script Windows PowerShell")
    dossier = tmp_path_factory.mktemp("faux-outil")
    source, exe = dossier / "FakeTool.cs", dossier / "FakeTool.exe"
    source.write_text(FAUX_OUTIL, encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Add-Type -Path '{source}' -OutputAssembly '{exe}' -OutputType ConsoleApplication"],
                   check=True, capture_output=True, timeout=120)
    return exe


def _lancer(tmp_path, faux_outil, code_cargo=0):
    """Lance une copie du vrai lanceur : faux dépôt, faux jeu, faux cargo sur le PATH, victory.exe déjà compilé."""
    script = tmp_path / "depot" / "tools" / "lancer_victory.ps1"
    script.parent.mkdir(parents=True)
    shutil.copy(RACINE / "tools" / "lancer_victory.ps1", script)
    manifeste = tmp_path / "depot" / "third_party" / "victory" / "Cargo.toml"
    manifeste.parent.mkdir(parents=True)
    manifeste.write_text("", encoding="utf-8")
    jeu = tmp_path / "jeu"
    jeu.mkdir()
    (jeu / "game_scarlett_master.exe").write_bytes(b"")
    cible, outils = tmp_path / "cible", tmp_path / "outils"
    (cible / "diag").mkdir(parents=True)
    outils.mkdir()
    shutil.copy(faux_outil, cible / "diag" / "victory.exe")      # compilation déjà présente (peut-être périmée)
    shutil.copy(faux_outil, outils / "cargo.exe")
    journal = tmp_path / "appels.log"
    env = {**os.environ, "PATH": f"{outils}{os.pathsep}{os.environ['PATH']}", "CARGO_TARGET_DIR": str(cible),
           "FAKE_TOOL_LOG": str(journal), "FAKE_TOOL_EXIT_CARGO": str(code_cargo),
           "FAKE_TOOL_EXIT_VICTORY": str(CODE_SORTIE_VICTORY)}
    fini = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                           "-GameDir", str(jeu)], env=env, capture_output=True, text=True, timeout=120)
    appels = journal.read_text(encoding="utf-8").splitlines() if journal.is_file() else []
    return fini, appels, manifeste, jeu


def test_launcher_rebuilds_even_when_victory_exe_exists(tmp_path, faux_outil):
    # Après un `git submodule update`, l'ancien victory.exe ignorait les réglages du commit épinglé : cargo décide.
    fini, appels, manifeste, jeu = _lancer(tmp_path, faux_outil)
    assert [a.split()[0] for a in appels] == ["cargo", "victory"], (appels, fini.stderr)
    assert appels[0] == f"cargo build --profile diag -p victory --features diagnostics --manifest-path {manifeste}"
    assert appels[1] == f"victory run {jeu / 'game_scarlett_master.exe'} --execute"
    assert fini.returncode == CODE_SORTIE_VICTORY


def test_launcher_stops_when_the_build_fails(tmp_path, faux_outil):
    fini, appels, _, _ = _lancer(tmp_path, faux_outil, code_cargo=CODE_ECHEC_CARGO)
    assert [a.split()[0] for a in appels] == ["cargo"], appels         # l'ancien victory.exe n'est pas lancé
    assert fini.returncode != 0
    assert f"cargo build failed (exit code {CODE_ECHEC_CARGO})" in fini.stderr


def test_no_xg_library_in_the_repository():
    # Règle 1 d'AGENTS.md : la xg_xs.dll du GDK vient de la copie de l'utilisateur, jamais du dépôt.
    sortie = subprocess.run(["git", "ls-files", "*.dll"], cwd=RACINE, capture_output=True, text=True,
                            check=False).stdout
    assert "xg_xs" not in sortie.lower()


def test_precompile_runs_translate_before_the_game_with_the_same_environment():
    texte = _lanceur()
    assert "[switch]$Precompile" in texte
    bloc = texte[texte.index("if ($Precompile) {"):texte.index("& $exe run")]
    assert "translate --cache --pipelines --used-pipelines $shaders $gameExe @dlls" in bloc
    assert "shaders.rpf" in bloc and "XboxBvhBuilder.dll" in bloc and "amd_fidelityfx_upscaler_xbox.dll" in bloc
    assert "throw" in bloc
    # Les variables du jeu (clé du cache) sont posées avant la précompilation, pas après.
    assert texte.index("$env:VICTORY_GAME_DIR =") < texte.index("if ($Precompile) {")
    assert texte.index("if ($Precompile) {") < texte.index("& $exe run")


def test_precompile_is_off_by_default():
    texte = _lanceur()
    assert "[switch]$Precompile" in texte and "Precompile = $true" not in texte
def _lancer_session(dossier, faux_outil, *options, avant=""):
    """Les VICTORY_* reçues par le faux victory.exe (voir `_executer_session`)."""
    return _executer_session(dossier, faux_outil, *options, avant=avant)[0]


def _executer_session(dossier, faux_outil, *options, avant=""):
    """Lance le vrai lanceur dans une session PowerShell, avec un faux victory.exe ; renvoie (VICTORY_* reçues, sortie).

    cargo est le faux outil (`faux_outil`, sur le PATH). Le faux victory.exe est une copie de l'interpréteur Python : `victory.exe run <jeu> --execute` exécute le
    fichier `run` du dossier courant, qui écrit les variables reçues. `avant` est exécuté dans la même session
    juste avant, pour y laisser ce qu'un lancement précédent aurait laissé.
    """
    base = Path(sys.base_prefix)
    (dossier / "repo" / "tools").mkdir(parents=True, exist_ok=True)
    (dossier / "repo" / "third_party" / "victory").mkdir(parents=True, exist_ok=True)
    (dossier / "repo" / "third_party" / "victory" / "Cargo.toml").touch()
    shutil.copy(RACINE / "tools" / "lancer_victory.ps1", dossier / "repo" / "tools")
    (dossier / "jeu").mkdir(exist_ok=True)
    (dossier / "jeu" / "game_scarlett_master.exe").touch()
    diag = dossier / "cible" / "diag"
    diag.mkdir(parents=True, exist_ok=True)
    shutil.copy(base / "python.exe", diag / "victory.exe")
    for dll in base.glob("*.dll"):
        shutil.copy(dll, diag)
    (dossier / "run").write_text("import json, os\nprint(json.dumps({k: v for k, v in os.environ.items() "
                                 "if k.startswith('VICTORY_') and k != 'VICTORY_GAME_DIR'}))\n", encoding="utf-8")
    script = dossier / "repo" / "tools" / "lancer_victory.ps1"
    commande = (f"Get-ChildItem Env:VICTORY_* | Remove-Item; {avant}; "
                f"& '{script}' -GameDir '{dossier / 'jeu'}' {' '.join(options)}")
    outils = dossier / "outils"
    outils.mkdir(exist_ok=True)
    shutil.copy(faux_outil, outils / "cargo.exe")
    env = dict(os.environ, CARGO_TARGET_DIR=str(dossier / "cible"), PYTHONHOME=str(base),
               PATH=f"{outils}{os.pathsep}{os.environ['PATH']}", FAKE_TOOL_LOG=str(dossier / "appels.log"))
    sortie = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", commande], cwd=dossier, env=env,
                            capture_output=True, text=True, timeout=120)
    assert sortie.returncode == 0, sortie.stderr
    return json.loads(sortie.stdout.strip().splitlines()[-1]), sortie.stdout + sortie.stderr


_SANS_PWSH = pytest.mark.skipif(sys.platform != "win32" or shutil.which("pwsh") is None,
                                reason="lanceur PowerShell : Windows et pwsh requis")


@_SANS_PWSH
def test_launcher_runs_with_the_tuned_settings_and_dred_only_when_asked(tmp_path, faux_outil):
    assert set(_lancer_session(tmp_path, faux_outil)) == set(REGLAGES)
    assert _lancer_session(tmp_path, faux_outil, "-Dred")["VICTORY_D3D12_DRED"] == "1"
    assert _lancer_session(tmp_path, faux_outil, "-NoTunedSettings") == {}


@_SANS_PWSH
def test_dred_left_in_the_session_by_an_earlier_launch_is_cleared(tmp_path, faux_outil):
    assert "VICTORY_D3D12_DRED" not in _lancer_session(tmp_path, faux_outil, avant="$env:VICTORY_D3D12_DRED = '1'")


@_SANS_PWSH
def test_no_tuned_settings_clears_those_an_earlier_launch_left(tmp_path, faux_outil):
    assert _lancer_session(tmp_path, faux_outil, "-NoTunedSettings", avant="$env:VICTORY_WIDE_BACK = '1'; $env:VICTORY_LATE_WAIT = '3'") == {}


@_SANS_PWSH
def test_relay_left_in_the_session_is_kept_but_named(tmp_path, faux_outil):
    # Laissé au choix de l'utilisateur, donc pas retiré ; mais le lanceur d'avant le posait (issue #89 du fork).
    recu, sortie = _executer_session(tmp_path, faux_outil, avant="$env:VICTORY_RELAY = '2'")
    assert recu["VICTORY_RELAY"] == "2"
    assert "VICTORY_RELAY=2 is set in this session" in sortie
    assert "VICTORY_RELAY" not in _executer_session(tmp_path, faux_outil)[1]
