# Configurations mesurées

Configurations de testeurs pour lancer le titre sous VIctory, avec les réglages réellement en vigueur et ce qui a été mesuré avec eux. Chaque valeur dit comment elle a été obtenue : **mesuré** (journal VIctory du lancement, processus en cours ou outil système), **observé** (vu à l'écran par le testeur), **lu** (dans le source). Tout vient du lancement de la copie du jeu du testeur ; rien ne vient du code ni des fichiers du jeu. Version anglaise : [../measured-setups.md](../measured-setups.md).

## RTX 4070 Ti SUPER, VIctory `7b0f736` avec `VICTORY_RELAY=2` (minecraftafk072, 24 novembre)

C'est la configuration de `main` à `b547bb2` (après #172), avant que #178 déplace l'épinglage sur `fd021bf` et retire `VICTORY_RELAY` du lanceur. Elle a tenu plus de 18 minutes dans le monde sans perte du périphérique.

### Machine

| | |
| --- | --- |
| Processeur | AMD Ryzen 9 7900X3D (son GPU intégré RDNA 2 est présent ; VIctory ne tourne pas dessus) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, pilote 32.0.16.1742. `nvidia-smi` y liste `victory.exe` (mesuré). |
| Mémoire | 32 Go |
| Écran | 1920x1080 à 240 Hz, sur la RTX |
| Système | Windows 11 Professionnel, build 26300 |

### Logiciels

- AnyXboxX `main` à `b547bb2`.
- `third_party/victory` à `7b0f736` (fork [fxceless/victory-fork](https://github.com/fxceless/victory-fork)), compilé par le lanceur : `cargo build --profile diag -p victory --features diagnostics`.
- Lancé avec `.\tools\lancer_victory.ps1 -GameDir <dossier du jeu>`, sans `-Dred`, `-SkipPipelines` ni `-NoTunedSettings`. Le lanceur exécute `victory.exe run <dossier du jeu>\game_scarlett_master.exe --execute`, sans `--console`, donc avec le profil Series X par défaut (lu).

### Réglages en vigueur

Lus dans l'environnement du `victory.exe` en cours le 24 novembre à 22:18 UTC (mesuré, filtré sur le préfixe `VICTORY_`). Ce sont les valeurs que pose `tools/lancer_victory.ps1` à `b547bb2` (lu).

| Variable | Valeur |
| --- | --- |
| `VICTORY_SHADER_LOOP_LIMIT` | `256` |
| `VICTORY_LATE_WAIT` | `3` |
| `VICTORY_WIDE_BACK` | `1` |
| `VICTORY_RELAY` | `2` |
| `VICTORY_CARD_BUFFERS_MIN_MB` | `1` |
| `VICTORY_CARD_BUFFERS_MB` | `1024` |
| `VICTORY_GAME_DIR` | le dossier du jeu |
| `VICTORY_D3D12_DRED` | absente (DRED coupé) |
| `VICTORY_SKIP_PIPELINES` | absente (aucun pipeline sauté) |

### Résultats

Lancement du 24 novembre à partir de 21:31 UTC, exactement ces réglages, journal VIctory gardé hors du dépôt, chiffres tirés de ses lignes de stats (une toutes les 5 s), mesuré :

| | |
| --- | --- |
| Monde atteint (1 000 draws par image ou plus) | 110 s après le lancement. Le cache de shaders avait été rempli avec la limite de boucles 32 au lancement précédent ; la limite fait partie de la clé du cache, donc une partie a été reconstruite (lu). |
| Temps dans le monde | 18,1 min, puis arrêt par la limite de 20 minutes du testeur |
| Perte du périphérique, lignes `ERROR` | aucune, aucune |
| Lignes `WARN` | 60, surtout des méthodes de création de pipeline non implémentées et des shaders qui ne se traduisent pas |
| Fréquence dans le monde | 5,1 i/s en médiane sur les lignes de stats à 100 draws ou plus, 195 ms par image (106 ms sur le GPU, 126 ms de copie de retour) |
| Par minute | 4,9 à 5,3 i/s de la minute 2 à la minute 15, 4,7 à 4,8 i/s de la minute 16 à la minute 18 |
| Draws par image | environ 4 780 (médiane), environ 345 perdus sans pipeline hôte |

Autres lancements sur la même machine, le même jour (mesuré, d'après leurs journaux VIctory) :

- Avec le lanceur de `7b0f736` (#171 : limite de boucles 32, DRED actif) : 18,8 min dans le monde, ni perte du périphérique ni `ERROR` ; 4,61 i/s de la minute 3 à la minute 6,9.
- Avec ces réglages plus `-Dred` : 6,9 min dans le monde, ni perte du périphérique ni `ERROR` ; 4,70 i/s de la minute 3 à la minute 6,9, contre 5,13 i/s sans DRED. DRED coûte environ 18 ms par image (+9,5 %) sur cette carte.

Un second lancement avec ces réglages, à 22:10 UTC depuis le terminal du testeur, sans journal gardé : toujours en cours après 12 minutes (mesuré avec `Get-Process`) ; le testeur dit qu'il tourne très bien (observé).

Observé par le testeur : avec la limite de boucles 32, les cheveux sont faux ; avec 256, l'écran de chargement et l'intro clignotent un peu (sans doute le clignotement du démarrage de #174). Les problèmes de peau et de sous-titres que l'issue #89 du fork attribue à `VICTORY_RELAY` n'ont pas été vérifiés dans ces lancements.

### Reproduire cette configuration

À `b547bb2`, le lanceur pose lui-même toutes ces valeurs. Il ne compile VIctory que si `third_party\victory\target\diag\victory.exe` manque (lu) : recompilez donc à la main après avoir déplacé le sous-module :

```powershell
git checkout b547bb2
git submodule update --init third_party/victory
cargo build --profile diag -p victory --features diagnostics --manifest-path third_party/victory/Cargo.toml
.\tools\lancer_victory.ps1 -GameDir <dossier du jeu>
```

Ouvrez d'abord un nouveau terminal : le lanceur laisse ses variables dans la session PowerShell, et un `VICTORY_D3D12_DRED` laissé par un lancement précédent garde DRED actif.
