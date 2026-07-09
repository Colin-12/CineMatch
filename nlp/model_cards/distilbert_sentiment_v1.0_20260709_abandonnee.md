# Model Card — distilbert_sentiment_v1.0_20260709

## Résumé

| | |
|---|---|
| **Dossier modèle** | `nlp/models/distilbert_sentiment_v1.0_20260709/` (HF checkpoint : `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `bundle_meta.joblib`) |
| **Endpoint cible** | `/sentiment` (classification des avis TMDB) |
| **Type de modèle** | DistilBERT (`distilbert-base-uncased`) fine-tuné, tête de classification 3 classes |
| **Version** | 1.0 |
| **Date d'entraînement** | 2026-07-09 |
| **Auteur** | Personne B (ML & NLP Engineer) |
| **Données** | Base Gold CineMatch — table `avis` (2 082 avis textuels TMDB liés à un couple user_id/film_id), labellisés via la note MovieLens associée (`notation`) |
| **Notebook source** | `nlp/training/sentiment_analysis.ipynb` (exécuté de bout en bout, 0 erreur) |
| **Script d'entraînement** | `nlp/training/model.py` (exécutable via `python -m nlp.training.model`) |

## ⚠️ Statut vs seuils de la feuille de route — **seuils NON atteints**

| Critère | Seuil | Meilleur résultat obtenu | Statut |
|---|---|---|---|
| F1 macro | > 0,70 | 0,423 (`tfidf_logreg`, pas le modèle retenu) | ❌ Non atteint |
| Accuracy | > 0,72 | 0,580 (`baseline_majoritaire`, pas un modèle appris) | ❌ Non atteint |

**Aucun des trois modèles évalués — baseline majoritaire, régression logistique TF-IDF, DistilBERT fine-tuné — n'atteint les seuils cibles.** Ce résultat est documenté ici de façon transparente plutôt que masqué ; voir « Diagnostic et limitations » ci-dessous pour l'analyse des causes.

## Comparaison des modèles (test set, 417 avis)

| Modèle | Accuracy | F1 macro | F1 pondéré | Amélioration F1 macro vs baseline naïve |
|---|---|---|---|---|
| baseline_majoritaire | 0,580 | 0,245 | 0,426 | — |
| tfidf_logreg | 0,540 | **0,423** | 0,521 | +72,9% |
| **distilbert (retenu)** | 0,513 | 0,417 | 0,513 | +70,3% |

**Le modèle retenu en production est DistilBERT fine-tuné**, malgré un F1 macro et une accuracy très légèrement inférieurs à ceux de la régression logistique TF-IDF (écart de 0,006 point de F1 macro — du même ordre que le bruit d'un split unique de 417 exemples). Ce choix est motivé par :

1. **Conformité à la consigne** : la feuille de route demande explicitement un fine-tuning Hugging Face pour `/sentiment` ; le TF-IDF+LogReg sert de baseline de comparaison, pas de modèle cible.
2. **Écart non significatif** : sur un test de 417 exemples, une différence de 0,006 en F1 macro n'est pas discriminante entre les deux approches — aucune n'est réellement meilleure que l'autre ici.
3. **Marge d'amélioration structurelle plus claire côté transformeur** (plus de données, `MAX_LENGTH` plus grand, ré-entraînement) que côté TF-IDF, qui plafonne plus vite avec le volume de données disponible.

Le modèle TF-IDF+LogReg reste disponible comme comparatif documenté dans le notebook, mais n'est pas exporté en tant que modèle de production.

### Rapport de classification détaillé (DistilBERT, test set)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| negatif | 0,300 | 0,284 | 0,292 | 74 |
| neutre | 0,279 | 0,287 | 0,283 | 101 |
| positif | 0,675 | 0,678 | 0,676 | 242 |
| **accuracy** | | | **0,513** | 417 |
| macro avg | 0,418 | 0,416 | 0,417 | 417 |
| weighted avg | 0,512 | 0,513 | 0,513 | 417 |

### Matrice de confusion (DistilBERT)

| | prédit négatif | prédit neutre | prédit positif |
|---|---|---|---|
| **réel négatif** | 21 | 28 | 25 |
| **réel neutre** | 18 | 29 | 54 |
| **réel positif** | 31 | 47 | 164 |

Le modèle ne peine pas seulement sur les paires de classes adjacentes (`negatif`/`neutre`, `neutre`/`positif`) : la confusion `negatif`→`positif` (25 cas) et `positif`→`negatif` (31 cas) est quasi aussi fréquente que les confusions adjacentes, ce qui est cohérent avec un bruit de label diffus plutôt qu'une simple difficulté de frontière entre classes voisines. 203 exemples sur 417 sont mal classés (48,7%).

## Diagnostic et limitations

**1. Supervision faible / bruit du label (cause structurelle principale).** Le label utilisé pour entraîner et évaluer tous les modèles n'est **pas** le sentiment exprimé par l'auteur réel de l'avis TMDB : c'est la note qu'a donnée, sur MovieLens, un utilisateur différent rattaché *synthétiquement* au film de l'avis (voir `nlp/training/features.py::derive_label_from_note` et section 3 du notebook). La lecture commentée des exemples mal classés (section 12/13 du notebook) confirme empiriquement des cas où le texte est clairement positif mais le label dérivé est `neutre` (note 3/5), et inversement.

**2. Volume et couverture très faibles.** Seuls 2 082 avis textuels sont disponibles, contre 1 000 209 notations dans `notation` (Gold) — une couverture de **0,208%**. Une fois divisé en train (1 665) / test (417), chaque classe minoritaire (`negatif` : 297 train / 74 test) dispose de trop peu d'exemples pour qu'un modèle de langage fine-tuné avec des millions de paramètres apprenne une frontière de décision robuste.

**3. Déséquilibre des classes.** Distribution des labels : `positif` 57,9%, `neutre` 24,3%, `negatif` 17,8%. Un classifieur majoritaire atteint déjà 58,0% d'accuracy sans rien apprendre — l'accuracy brute est donc un indicateur trompeur sur ce jeu de données ; le F1 macro (ciblé par `CLAUDE.md`) est la métrique de référence.

**4. Longueur des avis et troncature à 256 tokens.** Les avis sont longs : 252,7 mots / 1 466 caractères en moyenne (médiane 221 mots), jusqu'à 3 786 mots. Tokenisés par `distilbert-base-uncased`, ils font en moyenne 330 tokens (médiane 294) pour une limite `MAX_LENGTH=256` (`nlp/training/model.py`) : **56,5% des avis dépassent cette limite et sont tronqués**, perdant potentiellement le jugement de conclusion exprimé en fin de texte.

**5. Biais de langue.** Vérification lexicale sur les 2 082 avis : corpus quasi exclusivement anglophone (2 063/2 082, ~99,1%). Ce n'est pas une source de bruit observée *dans ce jeu de données*, mais le modèle (base anglophone) n'a été ni entraîné ni évalué sur des avis dans d'autres langues — un biais latent si la couverture linguistique réelle de TMDB évolue en production.

**6. Désaccord label dérivé / sentiment du texte.** Conséquence directe du point 1 : le label ne mesure pas le sentiment de l'auteur de l'avis, mais la note d'un tiers synthétiquement associé. Une piste d'amélioration future serait un label de sentiment humain direct sur un sous-ensemble d'avis, pour dissocier la performance du modèle du bruit de label.

## Recommandation

Avant d'investir davantage de temps de calcul dans un modèle plus gros ou plus d'époques, la piste la plus prometteuse est d'améliorer la donnée elle-même : recolter plus d'avis labellisés, ou remplacer/compléter le label dérivé de la note par un label de sentiment réellement observé sur le texte. En l'état, `/sentiment` doit être exposé avec cette limite documentée auprès de l'équipe plutôt que présenté comme fiable au seuil cible.

## Reproductibilité

- Notebook source : `nlp/training/sentiment_analysis.ipynb` (exécuté de bout en bout, 0 erreur)
- Script d'entraînement : `nlp/training/model.py` (exécutable via `python -m nlp.training.model`, ~80-90 minutes CPU pour 2 époques sur ~1 665 exemples, aucun GPU disponible)
- Fonctions de feature engineering / labellisation : `nlp/training/features.py`
- `random_state=42` utilisé de façon cohérente pour le split train/test (aucune fuite user_id/film_id)
- Smoke test (~100 exemples, 1 époque) validé avant l'entraînement complet, conformément à `CLAUDE.md`
