> English: [running-gta6.md](../running-gta6.md)

# Lancer GTA VI (build Xbox Series 1.0.2.1) avec AnyXboxX

Ce guide explique comment installer et lancer le projet, et tout ce qu'il faut installer pour refaire nos essais sur le build Xbox Series 1.0.2.1 de GTA VI, jusqu'au point où nous en sommes : la première cinématique. Il ne fournit pas le jeu et n'explique pas comment l'obtenir : vous apportez votre propre copie. Ne mettez jamais de fichier du jeu dans ce dépôt et ne les joignez à aucune issue.

Le lancement passe par [VIctory](https://github.com/Marvisak/VIctory) (Marvisak, GPL-3.0 ou ultérieure), suivi ici comme sous-module `third_party/victory`, épinglé sur le fork [fxceless/victory-gta6](https://github.com/fxceless/victory-gta6) (GPL-3.0) au commit `7b0f736`. Le script `tools/lancer_victory.ps1` le compile et le lance sur votre dossier du jeu.

## Fork épinglé (24 novembre, nuit)

`third_party/victory` pointe vers `fxceless/victory-gta6` au commit `7b0f736` (« Shader tests: recognise the software adapter from the device's own adapter »). Il était d'abord épinglé sur `d279e69` (« Make host pipeline waits configurable and prewarm pending requests (#1) », contribué par ZazaJr24) ; `7b0f736` ajoute `e9765a7` (« Fix the device loss introduced by the wave64 update ») et `004bf69`. Les commits plus récents, à partir de `1684989`, ne sont pas pris : le mainteneur les a jugés en régression. Ce fork n'est pas un fork GitHub rattaché à VIctory ; sa licence est GPL-3.0, compatible avec AnyXboxX (GPL-3.0 ou ultérieure). Il remplace notre ancienne intégration `anyxboxx/stable`.

Mesuré par le mainteneur le 2026-11-24 (i7-13700KF + RTX 4070 SUPER, Windows 11) en lançant notre propre copie du build 1.0.2.1, d'après le journal de VIctory de l'essai (fréquence d'images et compteurs de dessins tels que le programme les affiche), contre notre ancienne intégration `last` :

| | fork `d279e69` (premier épinglage) | ancienne intégration `last` |
| --- | --- | --- |
| Fréquence dans le monde | 4,0 à 4,3 fps | 2,3 fps |
| Dessins par image | environ 4 580 | environ 2 860 |
| Dessins perdus sans pipeline | 428 | 400 |
| Dessins indirects perdus | 3 | 86 |
| Perte du périphérique | aucune pendant environ 13 minutes, puis perdu (voir limites connues) | non relevée |

Le monde est atteint et affiché. Rien ici ne vient de la lecture du code du jeu.

**Observé par le mainteneur (Windows 11, RTX 4070 Super).** Avec le fork `fxceless/victory-gta6` au commit `d279e69` et les réglages du lanceur, le jeu a été joué à environ 4 fps pendant environ 13 minutes dans la zone de départ (la prison) : le personnage se déplace sous le contrôle du joueur et interagit avec les premiers PNJ, puis le périphérique hôte est perdu (limite connue).

**Mesuré sur `7b0f736` (mainteneur, même machine, d'après le journal VIctory de l'essai `victory-fork-7b0-relay.log`).** Avec `VICTORY_RELAY=2` et les réglages du lanceur, cache de pipelines déjà rempli : monde atteint en environ 1 min 25 ; 3,5 i/s à la première image du monde, puis 4,5 i/s après 2 min, environ 4 975 draws par image, 346 draws perdus faute de pipeline hôte ; aucune perte du périphérique pendant environ 2 min 40 dans le monde, puis le mainteneur a fermé la fenêtre. Un essai antérieur de `7b0f736` sans `VICTORY_RELAY` : 5,4 i/s à environ 3 min, aucune perte, fenêtre fermée. Avis du mainteneur : « légèrement mieux » que `d279e69`. Référence `d279e69` : 4,0 à 4,3 i/s, environ 4 580 draws, environ 430 perdus.

**Limites connues (stabilité).** La perte du périphérique après environ 13 minutes a été observée sur `d279e69` ; la suite de ce paragraphe décrit cet essai. `7b0f736` corrige une perte de périphérique liée au wave64 ; sa tenue longue n'est pas encore mesurée chez nous (notre essai le plus long dessus dure environ 3 minutes). L'issue n°26 du fork rapporte 21,5 minutes stables sur une RTX 4060 Ti avec `VICTORY_RELAY=2` (rapport tiers, non reproduit ici). Sur `d279e69`, le deuxième lancement a tenu environ 13 minutes dans le monde (premiers dessins à 19:07:27 UTC), puis a perdu le périphérique GPU à 19:20:35 UTC : « host GPU device removed after 15 submissions sent ahead (VICTORY_LATE_WAIT=1) of 82 lists », la dernière liste étant un compute shader (`c1488f485ad606fb`). Mesuré, d'après le journal de VIctory de l'essai ; la cause n'est pas connue. Comptez de quelques minutes à environ 13 minutes dans le monde avant une perte du périphérique, puis relancez. Le message cite `VICTORY_LATE_WAIT=1` alors que le lanceur pose `3` : le réglage réellement actif dans cet essai reste à vérifier.

**Premier lancement à froid.** Le premier lancement avec un cache vide a planté au chargement (compilation des pipelines, puis le plantage interne du jeu suivi sous VIctory #1464). Le deuxième lancement, avec le cache rempli, a tenu. Si le premier plante, relancez.

Les réglages de l'essai réussi sont maintenant ceux par défaut de `tools/lancer_victory.ps1` (tableau de la section Lancement). Ils demandent la compilation `diag` (`--profile diag --features diagnostics`). Passez `-NoTunedSettings` pour les désactiver.

## Linux (Proton)

VIctory est un programme Windows ; sous Linux il tourne sous Wine. Mesuré le 24/11/2026 par un contributeur (Ryzen 7 9800X3D, 62 Go, RX 9070 XT avec Mesa 26.2.7 RADV, Fedora 44, noyau 7.3.5, Wine de GE-Proton11-12 avec le vkd3d-proton de Proton Experimental), en lançant notre propre copie du build 1.0.2.1, d'après les journaux de VIctory de six lancements. L'image des lancements 3 à 5 a été décrite par la personne qui jouait ; tout le reste vient des journaux. Ce n'est pas une plateforme officielle : rien ci-dessous n'est vérifié par la CI.

### Réglages nécessaires sous Linux

- `VKD3D_CONFIG=descriptor_heap` : le jeu demande un tas de vues de 1 245 184 entrées, au-delà du plafond par défaut de vkd3d-proton (1 000 000).
- Les remplacements de DLL `d3d12,d3d12core,dxgi=n,b` pour que VIctory utilise vkd3d-proton, et une compilation `diag` croisée pour `x86_64-pc-windows-msvc` (par exemple avec [xwin](https://github.com/Jake-Shadle/xwin)), lancée avec le `wine` de Proton.
- Les réglages du lanceur, sans DRED : `VICTORY_D3D12_DRED` est actif dès que la variable existe (`host.rs`, `dred_on()`), quelle que soit sa valeur, et ses notes coûtent ici environ 670 ms par image (compteur par image de VIctory).
- Mesurés ici, sous Linux seulement : `VICTORY_PIPELINE_WAIT_MS=50` (500 par défaut ; 5 ou 6 pipelines restent « en cours de création » pendant tout le lancement et chaque image les attendait) et `VICTORY_TEXTURE_NO_WAIT=1` (les rafraîchissements de textures n'attendent plus le verrou mémoire pendant que Wine remappe la mémoire du jeu).

### Ce que refuse le compilateur de shaders de Wine

VIctory compile à l'exécution quelques programmes HLSL à lui avec `D3DCompile`. Sous Wine, c'est le compilateur HLSL de vkd3d-shader, qui (testé avec un petit programme sur le même Proton) :

- n'a aucune méthode de `RWByteAddressBuffer` en dehors de `Store*` : `Load`, `InterlockedAdd`, `InterlockedOr`, `InterlockedMax` et `InterlockedCompareExchange` échouent avec E5005. Les intrinsèques globales `Interlocked*` sur un `RWStructuredBuffer<uint>` compilent ;
- refuse `[allow_uav_condition]` (E5017) ;
- compile un shader qui porte `[RootSignature(...)]` mais laisse la signature racine de côté (avertissement W5302) : `CreateRootSignature` depuis le blob échoue alors avec `E_FAIL` (0x80004005).

Effets dans le journal avec le commit épinglé à ce moment-là (`d279e69`) ; `7b0f736`, épinglé maintenant, compile ces programmes de la même façon (lu statiquement) :

| Ligne du journal | Effet |
| --- | --- |
| `argument clamp: clamp:67:10: E5005: Method 'InterlockedMax' is not defined on type 'RWByteAddressBuffer'`, puis `no argument clamp: indirect execution dropped` | Environ 800 dessins et dispatches indirects abandonnés par image (3 dans l'essai Windows du mainteneur). |
| `no pipeline to draw a depth plane from its twin: such planes stay as they are error=Call failed. (0x80004005)` (formats 45 et 20) | La passe qui écrit les classes de matériau 16 bits du jeu dans son plan de profondeur ne tourne jamais ; les passes par matériau du jeu testent cette profondeur à l'égalité, donc beaucoup de surfaces du monde restent sans ombrage. Les personnages et les cheveux n'y passent pas. |
| `no host copies in the card's memory: the compare could not be made` | La vérification de la mémoire écrite par les shaders se fait sur le CPU : 0,6 à 1,1 Go par image. |

### Lancements mesurés

| Lancement | Compilation | Dans le monde | Fin |
| --- | --- | --- | --- |
| 2 | `d279e69` épinglé, réglages du lanceur | 0,1 à 0,6 i/s, environ 800 indirects abandonnés par image | `int3` du jeu lui-même après 8 min 35 s, dans l'intro (suivi sous VIctory #698 : un pool du jeu est plein) |
| 3 | + clamp des arguments réécrit pour Wine (correctif local), DRED coupé, les deux réglages Linux | 3,7 à 5,2 i/s en jeu, 0,4 à 0,8 pendant le streaming | quitté à la main après environ 29 min, sans plantage |
| 4 | + `main` du fork jusqu'à `1684989` (inclut `7b0f736`) et pull requests du fork #21 (DXC pour ces programmes, fermée depuis), #11 (fusionnée depuis), #9 ; limite de boucles 256 | 4,4 à 4,9 i/s ; cheveux corrects | quitté à la main |
| 5 | + VIctory amont #1130/#1067 (descripteur lu comme donnée), #1152 (pas de mélange sur les cibles entières), une somme de vague f16 ; limite de boucles 0 (valeurs par étage) | environ 5 i/s ; visage de Lucia affiché ; monde plus sombre, yeux trop lumineux | quitté à la main |
| 6 | + signature racine de la passe de plan de profondeur construite dans le code quand le blob n'en a pas (correctif local) ; limite de boucles 256 | 3,1 à 4,2 i/s, environ 4 700 dessins par image (jusqu'à 6 260) ; l'erreur du plan de profondeur a disparu | violation d'accès dans le jeu (écriture à l'adresse 0) après environ 11 min dans le monde ; cause non étudiée |

Les lancements 3 à 6 utilisent des correctifs absents du commit épinglé ; ils sont décrits ici pour pouvoir être portés vers le fork. Avec le seul `d279e69`, Linux obtient le lancement 2. Aucun de ces lancements n'utilisait `VICTORY_RELAY=2`, réglage par défaut du lanceur depuis `7b0f736` : il n'est pas mesuré sous Linux.

Ce qui manque encore sous Linux comme sous Windows : le ray tracing (le constructeur BVH du jeu est coupé ; l'activer a planté en quelques secondes dans les essais du fork) et 29 créations de pipelines à partir des pipelines précompilés de la console (VIctory #829).

## Ce que vous obtenez aujourd'hui

Mesuré le 2026-11-24 sur i7-13700KF + RTX 4070 SUPER, Windows 11, avec l'intégration stable :

- Le démarrage, l'écran de profil, les réglages de première exécution, les écrans légaux, le chargement et le menu pause fonctionnent, à 30 i/s dans les menus.
- La première cinématique (crédits d'ouverture de la mission 1) est atteinte.
- La scène 3D derrière les crédits est noire.
- Dans le monde, le jeu tourne à 0,3 à 0,6 image par seconde.
- Il n'est pas jouable.

La section suivante décrit un état plus avancé, mesuré le même après-midi sur une branche d'intégration locale qui n'est pas encore ce que pointe `third_party/victory`.

## État au 24 novembre (après-midi)

Mesuré en lançant notre propre copie du build 1.0.2.1 sous VIctory, sur une branche d'intégration locale (`main` de VIctory + environ 40 pull requests + correctifs locaux pas encore publiés), i7-13700KF + RTX 4070 SUPER. Rien ici ne vient de la lecture du code du jeu : ce sont des mesures d'exécution (i/s, temps, comptes) et la sortie de `victory translate` sur les shaders de notre copie.

- **Monde atteint, personnage contrôlable (observé).** Le tutoriel « Move / Move the camera » (L / R) s'affiche. L'interface est composée par-dessus une image 3D encore noire.
- **Vitesse dans le monde (mesurée) : 0,2 i/s, puis 2,3 à 2,5 i/s.** Causes et correctifs :
  - `VICTORY_D3D12_DRED=1` coûtait environ 2,5 fois (mise en forme de texte et un `VirtualQuery` par draw). Ne l'activer que pour enquêter sur un device removed.
  - La copie retour vers le titre passait 66 % du thread de soumission dans `WriteProcessMemory`, lui-même dans `NtQueryVirtualMemory` (environ 2 ms par appel). Une écriture directe `NtWriteVirtualMemory` fait passer de 1,5 s à 0,13 s par image.
  - Les copies n'appellent plus `VirtualQuery` au préalable ; l'instantané, la comparaison et le hachage des textures sont répartis sur un pool de threads.
  - Reste par image (environ 410 ms) : GPU environ 145 ms, copie retour environ 73 ms, hachage 20 à 60 ms, instantané environ 50 ms.
- **Shaders non traduits en jeu : 456 le matin, environ 170 l'après-midi (compte mesuré en jeu).** Sur le corpus du jeu (26 084 programmes, `victory translate`), les programmes émis sont : VS/PS 13 480 sur 14 038, CS 2 212, MS 8 654 sur 8 765 (comptes tels qu'affichés par l'outil). Motifs résolus dans la journée : descripteur lu comme donnée, mémoire scalaire lue comme donnée, buffer de stride inconnue, préfixe de vague en flot divergent (passes InstanceCulling), masque rangé dans une lane de VGPR, `readfirstlane` d'une adresse uniforme (ColorDownsample). Restent surtout les masques de lanes : « lane mask read as data » (586 programmes) et « holding lanes elsewhere » (282) sur le corpus.
- **Ray tracing.** Le constructeur de BVH du titre provoque un TDR (récupération après dépassement de délai du GPU) ; on le saute avec `VICTORY_SKIP_BVH_BUILDER=1`. La nouvelle option de diagnostic locale `VICTORY_RT_MISS=1` fait manquer tous les rayons : 36 conteneurs de ray tracing sur 47 sont émis au lieu de 18. Une analyse de Codex a trouvé que `ConstructBVH4` contient une scrutation bornée (256 itérations) sur une valeur produite par une autre passe.
- **Composition finale.** Comme dans VIctory #841, la scène existe en 2560x1440 mais la cible présentée reste à zéro.

### Variables utiles

Toutes sont des diagnostics : elles demandent le build `diag` (la fonctionnalité `diagnostics`). Provenance, vérifiée dans le code du worktree d'intégration et contre le `main` de VIctory le 2026-11-24 :

| Variable | Effet | D'où elle vient |
| --- | --- | --- |
| `VICTORY_KEYS` | Appuis de touches scriptés (voir plus bas). | `main` de VIctory |
| `VICTORY_SHADER_LOOP_LIMIT=32` | Les boucles traduites s'arrêtent après ce nombre d'itérations (décimal ou hexadécimal `0x`, au moins 1). | `main` de VIctory |
| `VICTORY_D3D12_DRED=1` | Données de device removed. **Attention : n'importe quelle valeur l'active, et le monde ralentit d'environ 2,5 fois ; ne l'utiliser que pour enquêter sur un device removed.** | `main` de VIctory |
| `VICTORY_SHOW_TARGET=busiest` | Affiche la cible de rendu la plus chargée au lieu de l'image finale. | Pull request ouverte #638 (`Marvisak/VIctory`), dans notre intégration |
| `VICTORY_CAPTURE_BUSY=<draws>[,<nombre>]` | Capture les images de plus de ce nombre de draws (2 images par défaut). | Pull request ouverte #638 |
| `VICTORY_PRODUCERS=1` | Journalise une passe qui lit ce que seul du travail abandonné (non traduit ou sauté) écrit, avec le shader abandonné. | Pull request ouverte #638 |
| `VICTORY_CAPPED_DYNAMIC=262144` | Plus d'emplacements de descripteurs dynamiques sur un tas plafonné (NVIDIA), au prix des plus hauts emplacements du titre. | Essai local, publié seulement sur notre fork (`anyxboxx/essai`, `anyxboxx/stable`) |
| `VICTORY_SKIP_BVH_BUILDER=1` | Écarte les shaders du constructeur de BVH du titre. | Notre fork seulement (`anyxboxx/essai`), pas proposé en amont |
| `VICTORY_PROFILE=<ms>` | Profileur à échantillonnage de tout le processus ; journalise les piles d'appels les plus fréquentes toutes les 5 s (période en ms, 4 par défaut). | Local, non publié |
| `VICTORY_RT_MISS=1` | Tous les rayons manquent (diagnostic de ray tracing, désactivé par défaut). | Local, non publié |

## État au 24 novembre (soir)

Mesuré en lançant notre propre copie du build 1.0.2.1 sous VIctory, sur la branche d'intégration locale (`main` de VIctory + pull requests + correctifs locaux pas tous publiés), i7-13700KF + RTX 4070 SUPER. Ce sont des mesures d'exécution (i/s, comptes, traces) et la sortie de `victory translate` ; rien ne vient de la lecture du code du jeu.

- **Le mode Series S est plus rapide (mesuré).** `victory run --console series-s ...` donne environ 3,1 à 3,3 i/s dans le monde, contre environ 2,0 à 2,4 i/s en mode Series X (environ +50 %), avec le même symptôme d'image. Le jeu rend alors en interne en 1280x720 et présente en 1920x1080. Recommandé pour itérer.
- **Diagnostic de l'image noire (mesuré).** La géométrie existe dans le G-buffer : les personnages de la cinématique sont visibles en 1440p et en 720p. La couleur finale est une teinte uniforme qui s'assombrit avec le temps (exposition), avec les sous-titres dessinés par-dessus. Presque toutes les passes d'éclairage différé sont traduites et liées. Une trace par passe (option locale `VICTORY_INDIRECT_TRACE`, non publiée) montre qu'environ 379 `ExecuteIndirect` de mesh partent avec des arguments à zéro, parce que les passes GPU qui produisent ces arguments ne sont pas encore traduites : le traducteur les refuse avec « an address from a phi » et « a lane mask holding lanes elsewhere read as data ». C'est le chantier en cours.
- **Traduction (mesurée).** Shaders non traduits en jeu : environ 130 à 170 (compte en jeu). Sur le corpus (26 084 programmes, `victory translate`) : VS/PS 13 780 émis sur 14 038, CS 2 212, MS 8 654 sur 8 765 (comptes tels qu'affichés par l'outil).
- **Correctifs publiés en amont** (`Marvisak/VIctory`) : #1215 (vitesse : écriture directe `NtWriteVirtualMemory`, copies sans `VirtualQuery`, profileur), #1225 (`readfirstlane` d'une adresse uniforme : ColorDownsample), #1231 (compteur de boucle uniforme : sondes de lumière). Des commentaires de mesures ont aussi été ajoutés sur #497, #320, #1074, #1198 et #841.
- **Communauté.** Le projet a un serveur Discord (lien en haut du README).

Variable ajoutée au tableau ci-dessus (diagnostic, build `diag`) :

| Variable | Effet | D'où elle vient |
| --- | --- | --- |
| `VICTORY_INDIRECT_TRACE` | Trace par passe des appels `ExecuteIndirect` et de leurs arguments. | Local, non publié |

## Prérequis

- Windows 10 ou 11 x64.
- Un processeur avec AVX2. Intel convient : les instructions propres à AMD sont émulées.
- Un GPU Direct3D 12 avec shader model 6.6 et mesh shaders (NVIDIA RTX série 20 ou plus récent, AMD RDNA 2 ou plus récent).
- 32 Go de RAM conseillés et un fichier d'échange Windows assez grand : le jeu engage plusieurs Gio au démarrage. Sinon, `crashes.log` signale l'erreur Windows 1455 : fermez d'autres applications ou agrandissez le fichier d'échange.
- Votre propre copie du build Xbox Series 1.0.2.1 : un dossier qui contient `game_scarlett_master.exe`. Non fournie, non expliquée ici.
- De l'espace disque pour le cache de shaders (il grossit au fil du jeu ; voir la page d'état pour les chiffres à prévoir).

## Installation

1. [Git](https://git-scm.com/).
2. Rust via [rustup](https://rustup.rs/). La toolchain exacte est fixée par le `rust-toolchain.toml` du sous-module et s'installe seule à la première commande `cargo`.
3. Visual Studio 2022 ou 2026 Build Tools avec la charge « Développement Desktop en C++ » (MSVC).
4. Windows SDK 10.0.26100 (fournit `dxc`).
5. Optionnel : la fonctionnalité Windows « Graphics Tools » (couche de débogage D3D12, utile seulement pour `VICTORY_D3D12_DEBUG`).
6. Python 3.12, seulement pour la suite de tests d'AnyXboxX (voir Tests).

Récupérer le code avec son sous-module :

```powershell
git clone --recurse-submodules https://github.com/LooperSalty/AnyXboxX.git
cd AnyXboxX
# ou, dans un clone existant :
git submodule update --init third_party/victory
```

## Compilation

```powershell
cargo build --profile diag -p victory --features diagnostics --manifest-path third_party/victory/Cargo.toml
```

Le profil `diag` est nécessaire pour les touches scriptées, les captures d'images et les options de diagnostic ci-dessous. La première compilation télécharge et compile toutes les dépendances : comptez plusieurs minutes. L'exécutable est `third_party\victory\target\diag\victory.exe` (ou `%CARGO_TARGET_DIR%\diag\victory.exe` si cette variable est définie). `tools/lancer_victory.ps1` lance cette compilation à votre place quand l'exécutable manque.

## Lancement

```powershell
.\tools\lancer_victory.ps1 -GameDir <dossier du jeu>
```

Le script règle les variables ci-dessous (sauf avec `-NoTunedSettings`, qui ne garde que `VICTORY_GAME_DIR`) puis exécute `victory.exe run <dossier du jeu>\game_scarlett_master.exe --execute` :

| Variable | Valeur | Pourquoi |
| --- | --- | --- |
| `VICTORY_GAME_DIR` | votre dossier du jeu | Aucun `victory.toml` n'est nécessaire. |
| `VICTORY_CAPPED_DYNAMIC` | `262144` | NVIDIA plafonne le tas de descripteurs à 1 000 000 d'entrées. Avec la plage dynamique par défaut, environ 720 dessins indirects par image étaient abandonnés ; avec 262 144 emplacements, ils ne le sont plus. |
| `VICTORY_SHADER_LOOP_LIMIT` | `32` | Les boucles des shaders traduits étaient quasi infinies ; avec une limite de 32 itérations, le GPU est environ 6 fois plus rapide. |
| `VICTORY_RT_MISS` | `1` | Tous les rayons ratent (diagnostic du ray tracing) : plus de programmes de ray tracing se traduisent. |
| `VICTORY_WAVE32_READFIRSTLANE`, `VICTORY_WAVE32_CROSSLANE` | `approx` | `readfirstlane` et opérations inter-voies approchés en wave32. |
| `VICTORY_SKIP_BVH_BUILDER` | `1` | Écarte le constructeur BVH du jeu, qui provoque un délai GPU (TDR). |
| `VICTORY_D3D12_DRED` | `1` | Données de périphérique perdu. Actif dans l'essai mesuré ; il ralentit le monde, désactivez-le (`-NoTunedSettings`) si seule la vitesse compte. |
| `VICTORY_LATE_WAIT` | `3` | Réglage d'attente des pipelines côté hôte, de la pull request n°1 du fork. |
| `VICTORY_WIDE_BACK` | `1` | Option du fork, active dans l'essai mesuré. |
| `VICTORY_RELAY` | `2` | Option du fork, active dans l'essai mesuré de `7b0f736`. |
| `VICTORY_CARD_BUFFERS_MIN_MB`, `VICTORY_CARD_BUFFERS_MB` | `1`, `1024` | Dimensionnement des tampons de carte, en Mo. |

Le premier lancement à froid peut planter pendant la compilation des pipelines : relancez une fois le cache rempli.

`-SkipPipelines <hash,...>` est transmis à `VICTORY_SKIP_PIPELINES` (voir plus bas).

Où sont les choses :

- Sauvegardes, réglages et `crashes.log` : `%LOCALAPPDATA%\victory\data`.
- Données dérivées (caches de shaders et de pipelines) : `%LOCALAPPDATA%\victory\cache`.

Les deux sont volontairement hors du dépôt. Supprimez le dossier du cache pour repartir d'un cache vide ; gardez le dossier des données si vous voulez conserver votre sauvegarde.

## Arriver à la première cinématique

Le titre de la fenêtre affiche la fréquence d'images. La fenêtre doit avoir le focus.

1. Écran de profil : Entrée.
2. Réglages de première exécution : Entrée à chaque écran.
3. Mentions légales : descendre sur « Suivant », attendre la fin du compte à rebours de l'avertissement de photosensibilité, cocher les deux cases, puis « Confirmer ».
4. Chargement, puis la première cinématique (crédits d'ouverture).
5. Échap ouvre le menu pause.

Clavier ou manette. Sans manette XInput branchée, le clavier sert de manette tant que la fenêtre du titre a le focus :

| Touches | Manette |
| --- | --- |
| Entrée, Retour arrière, X, Y | A, B, X, Y |
| Flèches | croix directionnelle |
| W A S D / I J K L | stick gauche / stick droit |
| Q / E, Z / C | gâchettes hautes, gâchettes basses |
| F / H | clics de sticks |
| Échap | Menu |
| Tab | View |

Maintenez chaque touche un instant : le jeu lit l'entrée à 30 Hz, un appui très bref peut être manqué. Les lancements suivants reprennent directement la sauvegarde.

## Lancements scriptés et diagnostics

Ils demandent le build `diag`.

| Variable | Effet |
| --- | --- |
| `VICTORY_KEYS=40:Enter,55:Enter,70:Enter` | Appuie sur les touches à ces secondes, avec ou sans focus. |
| `VICTORY_CAPTURE=<dossier>` et `VICTORY_CAPTURE_FRAMES=<liste>` | Enregistre les images présentées (ces numéros d'image). |
| `VICTORY_SHOW_TARGET=busiest` | Affiche la cible de rendu la plus chargée à la place de l'image finale (par exemple le G-buffer). |
| `VICTORY_D3D12_DRED=1` | En cas de panne du GPU, donne les commandes en cours. Ralentit le monde d'environ 2,5 fois : à n'activer que pour enquêter sur un device removed. |
| `VICTORY_D3D12_DEBUG=1` | Couche de débogage D3D12 (Graphics Tools requis). |
| `VICTORY_GPU_TIMING=1` | Chronomètre chaque liste de commandes et journalise les lentes. |
| `VICTORY_SKIP_PIPELINES=<hash,...>` | Omet les dessins et dispatchs qui utilisent les hash de shader à 16 chiffres listés. |
| `VICTORY_LOG=<cible>=debug,...` | Sortie debug et trace. |

Pour arrêter le jeu : `taskkill /F /IM victory.exe`. La commande `timeout` de Git Bash ne tue pas l'exécutable.

## Anciennes branches

Les branches `anyxboxx/stable` et `anyxboxx/essai` de `LooperSalty/victory` ne sont plus l'épinglage : le fork ci-dessus les remplace. Les sections qui les citent décrivent des mesures antérieures.

## Sécurité

- Sur NVIDIA, rester longtemps dans le monde peut provoquer une réinitialisation du pilote d'affichage (tous les écrans noirs un instant), et un testeur a eu un écran bleu (0x119). Arrêtez le jeu après quelques minutes dans le monde.
- Ne lancez qu'une instance à la fois.

## Tests

```powershell
cargo test -p victory-gpu -p victory-shader --manifest-path third_party/victory/Cargo.toml
```

AnyXboxX lui-même (voir la section Setup d'[AGENTS.md](../../AGENTS.md)) :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check --select W605,F,E9 PrototypeScarlett tests tools
```

## Où signaler un problème

Ouvrez une issue sur [AnyXboxX](https://github.com/LooperSalty/AnyXboxX/issues) ou sur [VIctory](https://github.com/Marvisak/VIctory/issues) (l'issue #264 regroupe les blocages GPU sur NVIDIA). Décrivez le symptôme, votre GPU et votre pilote, et les lignes utiles de `crashes.log`. Ne joignez jamais de fichier du jeu.
