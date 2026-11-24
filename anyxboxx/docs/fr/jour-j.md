> English: [day-one.md](../day-one.md)

# Premier lancement d'un exécutable console

Marche à suivre quand l'exécutable du paquet (par exemple `game_scarlett_master.exe`) est disponible. Tout se lance depuis la racine d'AnyXboxX.

## 1. Préparer (une fois)

```powershell
git submodule update --init
cmake -S runtime -B runtime/build -A x64
cmake --build runtime/build --config Release
cargo build --release --manifest-path rust/Cargo.toml
python tools/build_tombox.py d3d12_xs xg_xs xmem
```

## 2. Diagnostic sans risque

```powershell
python tools/jour_j.py ..
```

L'outil :

- fait l'inventaire de l'exécutable : empreinte SHA-256, en-têtes, sections, TLS et imports ;
- indique pour chaque fonction importée qui la fournit : le paquet, AnyXboxX, TomBox, Windows ou personne ;
- lance `--rapport` et `--lancer`, qui exécutent le programme dans Unicorn, isolé du PC.

Le rapport est écrit dans `%LOCALAPPDATA%\AnyXboxX\jour-j\<date>\rapport.md`, hors du dépôt : il nomme des fonctions du jeu et ne doit pas être versionné.

## 3. Lancement réel (facultatif)

```powershell
python tools/jour_j.py .. --natif --duree 120
```

`anyxboxx.exe` exécute alors le code du jeu directement sur le processeur, avec les DLL console de TomBox (`ANYXBOXX_TOMBOX_DIR`) et la trace des appels (`ANYXBOXX_TRACE=1`).

**Ce mode n'isole pas le jeu.** Il est réservé à un exécutable dont la provenance est connue.

## 4. Après le diagnostic

- Les fonctions **manquantes** du rapport deviennent des issues, une par DLL.
- Les **shaders** du jeu passent par `--rdna2 … --liste` puis `--hlsl` ; le cache (`emu/shader_cache.py`) évite de les recompiler.
- Rien du jeu n'entre dans le dépôt : ni l'exécutable, ni ses shaders, ni ses données. Seuls des noms de fonctions et des constats peuvent être publiés.
