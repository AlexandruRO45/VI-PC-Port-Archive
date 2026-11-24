## Changement

<!-- Le comportement observable qui change, et pourquoi. Une seule chose relisible par PR. -->

Issue liée : Fixes # / Refs #

## Vérification

<!-- Pour chaque vérification : la commande exacte et son résultat (copie la dernière ligne).
     GPU : carte ou WARP, et la première différence s'il y en a une.
     Si tu n'as pas pu lancer une vérification, dis-le et nomme ce qui manquait. -->

```
python -m pytest                → … passed
python -m ruff check --select W605,F,E9 PrototypeScarlett tests tools → All checks passed!
```

## Oracle et données de test

<!-- Décodeur : comparé aux tables LLVM (`tools/rdna2_mnemonics.py … --verifier`).
     Traduction HLSL : comparée à l'interpréteur de référence, octet par octet.
     Nouvelles données de test : d'où elles viennent (construites dans les tests, LLVM…) et leur licence.
     Cas d'échec : quel test échoue sans ce changement ? -->

## Liste de contrôle

- [ ] La PR couvre un seul comportement relisible.
- [ ] Un test échoue sans ce changement et passe avec.
- [ ] Les commandes et résultats de chaque vérification sont indiqués ci-dessus.
- [ ] Une fonction non implémentée renvoie une erreur (`E_NOTIMPL`), jamais un faux succès.
- [ ] README : la même modification dans les 9 langues (le test `test_readme_sync.py` le vérifie).
- [ ] Aucun fichier de jeu, de console ou de SDK, aucune clé, aucun dump, aucun lien de téléchargement.
- [ ] Constats tirés d'un titre : limités à ce que permet la règle 2 d'`AGENTS.md` (noms, slots, formes d'arguments, valeurs de flags, HRESULT, IID, comptes), avec la manière dont ils ont été obtenus ; aucune adresse ni RVA.
- [ ] Code repris d'un autre projet : licence compatible et crédit dans `THIRD_PARTY.md`.

<!-- Si ce changement a été fait avec l'aide d'une IA, indique-le ici. -->
