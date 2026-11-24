# Contribuer à AnyXboxX

Merci de ton intérêt. Ces règles s'appliquent à toute personne autre que le mainteneur.

Discussion, aide et rapports de test : [Discord](https://discord.gg/TcX42vRyPU)

**Ta copie du jeu est la bienvenue en local.** Tu peux lancer, mesurer, déboguer et analyser ta propre copie de l'exécutable et du build du jeu pour orienter tes PR. Ce qui reste interdit : ajouter au dépôt des fichiers du jeu ou de la console (voir « Ce qui est refusé »).

## Règle principale : tout passe par une pull request

La branche `main` est protégée. **Personne ne peut y pousser directement.** Pour proposer un changement :

1. Fais un fork du dépôt.
2. Crée une branche depuis `main` : `git checkout -b feat/ma-fonction`.
3. Ajoute tes commits, avec des messages au format `type: description` (`feat`, `fix`, `docs`, `test`, `refactor`, `chore`).
4. Lance les tests : `python -m pytest`, puis `python -m ruff check --select W605,F,E9 PrototypeScarlett tests tools`. Tout doit passer.
5. Ouvre une pull request vers `main` et remplis le modèle proposé : les commandes lancées et leur résultat exact.

Une pull request n'est fusionnée qu'après la relecture et l'accord du mainteneur, et quand les deux checks de la CI passent :

- `lint` (Linux, ~1 min) : ruff et cohérence des README dans les 9 langues ;
- `pytest` (Windows) : build du runtime C++ et du Rust, puis toute la suite. Elle vérifie aussi que le nombre de tests annoncé dans les README est exact.

### Bonnes pratiques

- **Une PR = un comportement.** Deux corrections indépendantes vont dans deux PR, plus faciles à relire.
- **Conflit avec `main`** : fais un rebase (`git fetch && git rebase origin/main`), garde les deux versions quand elles se complètent, relance les tests, puis `git push --force-with-lease`.
- **Check en échec sans rapport avec ta PR** (test GPU sur WARP, par exemple) : signale-le dans la PR, avec le nom du test et la ligne d'erreur ; le mainteneur relancera le job.
- **Aide d'une IA** : indique-le dans la description de la PR.

## Issues

Avant d'écrire du code, tu peux ouvrir une issue :

- **Bogue** : ce que tu as lancé, ce que tu attendais, ce qui s'est passé, le journal ou le rapport `--rapport`.
- **Fonction à émuler** : le nom exact de la fonction (`dll!Fonction`), sa documentation GDK et le programme qui l'appelle.
- **Idée** : le problème que l'idée règle.

Pour un changement important, ouvre d'abord une issue pour en discuter.

## Sources et usage de ta copie du jeu

L'implémentation s'appuie sur des sources publiques : documentation ISA RDNA 2 d'AMD, Microsoft Learn (GDK, Win32, D3D12), tests de LLVM, et code libre sous licence compatible, crédité dans [THIRD_PARTY.md](THIRD_PARTY.md).

Tu peux utiliser ta propre copie de l'exécutable et du build du jeu : le lancer (natif, VIctory via `tools/lancer_victory.ps1`, AnyXboxX), mesurer, déboguer, l'analyser statiquement et dynamiquement, et t'en servir pour orienter tes PR. Les mesures et constats tirés de l'exécution (journaux résumés, comptes, identifiants techniques comme des empreintes de shaders ou des noms de passes, résultats de tests) peuvent être cités dans les PR et dans la doc, avec la manière dont ils ont été obtenus (mesuré, lu statiquement, supposé). La limite : aucun fichier, code ni asset du jeu ou de la console dans le dépôt. Trois choses tirées du jeu sont autorisées, comme dans la politique de VIctory (`docs/asset-policy.md` de VIctory) : les signatures d'octets (motifs courts écrits par nous, avec masques) qui localisent des fonctions ou des données dans l'exécutable, les offsets par build des fonctions qu'on intercepte (sous des noms qu'on choisit), et l'empreinte du build (hash de l'exe, GUID/âge CodeView).

## Ce qui est refusé

- Tout fichier du jeu ou de la console ajouté au dépôt : exécutables, DLL, archives `.rpf`, shaders extraits, textures, sons, sauvegardes, dumps, clés, même partiels et même dans les tests.
- Du code du jeu (sections, routines, code décompilé ou désassemblé au-delà des signatures autorisées) ou des assets copiés du jeu, même relevés sur une copie légitime.
- Les tables de données propriétaires, sauf exception décidée par le mainteneur : seules des tables constantes sans lesquelles un format ne peut pas être décodé et qu'on ne peut pas recalculer (comme les tables BCPack de VIctory) peuvent être ajoutées, chacune listée nommément dans le registre d'exceptions de `THIRD_PARTY.md`, et la licence du dépôt ne les couvre pas.
- Les liens vers des copies illégales de jeux.
- Le code sans test pour une nouvelle fonction émulée.

## Ajouter une fonction émulée

Ajoute un gestionnaire décoré par `@hle(...)` dans `PrototypeScarlett/emu/hle/` et un test dans `tests/test_hle.py`. Cite la signature publique de la fonction (documentation Microsoft GDK ou Win32) en commentaire.
