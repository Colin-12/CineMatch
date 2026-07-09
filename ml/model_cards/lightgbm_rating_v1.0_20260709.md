# Model Card — lightgbm_rating_v1.0_20260709

## Résumé

| | |
|---|---|
| **Fichier modèle** | `ml/models/lightgbm_rating_v1.0_20260709.joblib` |
| **Endpoint cible** | `/prediction` (note prédite utilisateur × film) |
| **Type de modèle** | LightGBM Regressor (gradient boosting sur features tabulaires) |
| **Version** | 1.0 |
| **Date d'entraînement** | 2026-07-09 |
| **Auteur** | Personne B (ML & NLP Engineer) |
| **Données** | Base Gold CineMatch — table `notation` (1 000 209 lignes, 6 040 utilisateurs, 3 883 films) + table `film` enrichie TMDB (couverture ~44%) |

## Décision de modèle retenu

Deux modèles candidats ont été entraînés et évalués sur un split stratifié par utilisateur (80/20, `random_state=42`, aucune fuite train/test) :

| Modèle | RMSE test | MAE test | Amélioration vs baseline moyenne globale | Amélioration vs baseline moyenne par film |
|---|---|---|---|---|
| Baseline — moyenne globale | 1.1164 | 0.9331 | — | — |
| Baseline — moyenne par film | 0.9782 | 0.7816 | +12.4% | — |
| SVD biaisé (factorisation matricielle) | 0.8957 | 0.7004 | +19.77% | +8.43% |
| **LightGBM (retenu)** | **0.9028** | **0.7130** | **+19.13%** | **+7.71%** |

**Le modèle retenu en production est LightGBM**, malgré un RMSE test légèrement supérieur à celui du SVD (0.9028 vs 0.8957, soit ~0.8% relatif — écart faible, potentiellement du même ordre que le bruit d'un split unique). Ce choix est motivé par :

1. **Interprétabilité** : LightGBM permet une analyse SHAP par feature, impossible pour un SVD à facteurs latents sans sémantique explicite.
2. **Analyse d'erreurs actionnable** : LightGBM permet de décomposer les erreurs par genre, décennie, volume de notations — utile pour prioriser les futures améliorations produit.
3. **Réponse directe à la question EDA initiale** : SHAP confirme que les features TMDB (`popularite`, `note_tmdb`, `has_tmdb_data`) apportent une contribution réelle mais modeste, cohérente avec leur couverture partielle du catalogue (~44%).
4. **Cold-start géré explicitement** : les deux modèles gèrent le cold-start via fallback sur la moyenne globale, mais LightGBM généralise ce mécanisme nativement à travers ses features d'agrégats (`user_n_notations=0`, `film_n_notations=0`).

Le SVD entraîné (`ml/models/svd_rating_v1.0_20260709.joblib`) reste disponible comme alternative documentée, mais n'est pas le modèle de production.

## Statut vs seuils de la feuille de route

| Critère | Seuil | Résultat | Statut |
|---|---|---|---|
| RMSE test | < 1.0 | 0.9028 | ✅ Atteint |
| Amélioration vs baseline naïve (moyenne globale) | ≥ 10% | +19.13% | ✅ Atteint |
| Amélioration vs baseline naïve (moyenne par film, référence plus stricte) | ≥ 10% | +7.71% | ⚠️ Non atteint (juste sous le seuil) |

**Note de transparence** : la feuille de route ne précise pas explicitement laquelle des deux baselines naïves sert de référence stricte pour le seuil des 10%. Le choix a été fait de considérer la baseline la plus forte (moyenne par film) comme référence de rigueur dans `model.py`, ce qui fait apparaître un écart honnête : le seuil des 10% est large ment dépassé face à la baseline la plus simple (moyenne globale, +19%), mais manqué de peu (+7.7%) face à la baseline la plus exigeante (moyenne par film). Ce point doit être arbitré/validé avec l'équipe/le PO avant mise en production stricte du critère.

## Features utilisées (30)

Calculées exclusivement à partir du train (aucune fuite vers le test), avec fallback cold-start explicite (moyenne globale, comptages à 0) :

- **Utilisateur** (agrégats train-only) : `user_mean_note`, `user_std_note`, `user_n_notations`
- **Film** (agrégats train-only) : `film_mean_note`, `film_std_note`, `film_n_notations`
- **Métadonnées film** : `annee`, `nb_genres`
- **Genres** (one-hot, 18 catégories MovieLens) : `genre_Action`, `genre_Adventure`, `genre_Animation`, `genre_Children's`, `genre_Comedy`, `genre_Crime`, `genre_Documentary`, `genre_Drama`, `genre_Fantasy`, `genre_Film-Noir`, `genre_Horror`, `genre_Musical`, `genre_Mystery`, `genre_Romance`, `genre_Sci-Fi`, `genre_Thriller`, `genre_War`, `genre_Western`
- **TMDB** (couverture partielle ~44%, NULL gérés explicitement, jamais de `dropna`) : `has_tmdb_data` (flag), `popularite` (imputée par médiane si absente), `note_tmdb` (imputée par médiane si absente), `overview_length` (imputée par médiane si absente)

## Importance des features (SHAP, mean |SHAP value|, top 10)

| Rang | Feature | Importance moyenne |
|---|---|---|
| 1 | `film_mean_note` | 0.4086 |
| 2 | `user_mean_note` | 0.2930 |
| 3 | `user_n_notations` | 0.0460 |
| 4 | `user_std_note` | 0.0342 |
| 5 | `annee` | 0.0110 |
| 6 | `film_n_notations` | 0.0072 |
| 7 | `film_std_note` | 0.0043 |
| 8 | `popularite` (TMDB) | 0.0033 |
| 9 | `genre_Comedy` | 0.0030 |
| 10 | `genre_Horror` | 0.0029 |

Les agrégats user/film dominent très largement le signal (≈ 78% de l'importance cumulée à eux deux). Les features TMDB (`popularite`, `note_tmdb`) et les genres contribuent marginalement — cohérent avec leur couverture partielle (~44% du catalogue) et confirme que ces features sont exploitables mais non déterminantes en l'état.

## Analyse d'erreurs

### Par genre (RMSE)
Les genres les plus « polarisants » (avis dispersés) ont un RMSE plus élevé : Horror (0.9728), Documentary (0.9343), Sci-Fi (0.9301), Musical (0.9299), Comedy (0.9204). À l'inverse, les genres à consensus cinéphile plus fort ont un RMSE plus bas, y compris quand ils sont rares au catalogue : Film-Noir (0.7935, le plus bas malgré sa rareté), War (0.8701), Mystery (0.8765), Crime (0.8834), Drama (0.8837). La rareté d'un genre n'est donc pas le facteur explicatif principal — c'est la dispersion des opinions au sein du genre qui compte.

### Par décennie (RMSE)
Le RMSE **augmente** de façon quasi monotone des films anciens vers les films récents : 1910 → 0.7357, 1950 → 0.7974, 1980 → 0.8773, 1990 → 0.9312, 2000 → 0.9625. C'est l'inverse d'une hypothèse naïve « moins de données anciennes = plus d'erreur ». L'explication retenue : biais de survivance — les films anciens encore présents au catalogue sont des classiques canoniques avec un fort consensus de notation, tandis que les films des années 1990-2000 (cœur du jeu de données MovieLens 1M) couvrent une diversité de qualité bien plus large, avec des avis plus dispersés.

### Par volume de notations du film (le point EDA des 290 films à faible signal)
| Bucket (nb notations) | RMSE |
|---|---|
| (0, 5] — les 290 films identifiés en EDA | 1.0853 |
| (5, 20] | 1.0180 |
| (20, 50] | 0.9325 |
| (50, 200] | 0.9244 |
| (200, ∞) | 0.8960 |
| **Ensemble film_n_notations ≥ 5** | **0.9026** |
| **Ensemble film_n_notations < 5 (290 films)** | **1.1639** |

Les 290 films à faible signal (identifiés dès l'EDA) concentrent l'essentiel de l'erreur résiduelle du modèle (RMSE 1.1639, proche de la baseline naïve moyenne globale). C'est une limite structurelle attendue du cold-start film, malgré le fallback explicite sur la moyenne globale.

## Limitations connues

1. **Couverture TMDB partielle (~44%)** : les features `popularite`, `note_tmdb`, `overview_length` ne sont fiables/informatives que pour un peu moins de la moitié du catalogue ; le flag `has_tmdb_data` neutralise ce biais mais ne comble pas l'information manquante.
2. **Films à faible signal (< 5 notations, 290 films)** : erreur nettement dégradée (RMSE 1.16), proche de la baseline naïve — le modèle n'apporte quasiment aucune valeur ajoutée sur ce segment.
3. **SVD biaisé simplifié** : l'implémentation SVD utilisée (résiduel biais + `scipy.sparse.linalg.svds`) est une simplification par rapport à un Funk-SVD/SGD régularisé ; elle sert de comparatif mais n'a pas été retenue en production.
4. **Seuil d'amélioration de 10%** : atteint largement vs baseline moyenne globale (+19%), mais manqué de peu vs baseline moyenne par film (+7.7%) — voir section « Statut vs seuils » ci-dessus pour la nuance complète.
5. **Split unique pour la comparaison finale** : la comparaison SVD/LightGBM finale repose sur un seul split train/test (80/20) ; le tuning et la sélection de modèle ont utilisé une validation croisée 5-fold stratifiée par utilisateur, mais l'écart final (0.8957 vs 0.9028) reste à confirmer sur plusieurs seeds si du temps est disponible ultérieurement.
6. **Tuning hyperparamètres volontairement limité** : conformément à la consigne de prioriser la robustesse du split et l'analyse d'erreurs, seule une grille réduite (8 combinaisons : `num_leaves` ∈ {15, 31}, `learning_rate` ∈ {0.05, 0.1}, `n_estimators` ∈ {200, 400}) a été explorée pour LightGBM. Un tuning plus poussé (Optuna, espace de recherche élargi) pourrait encore réduire le RMSE.

## Reproductibilité

- Notebook source : `ml/training/prediction_note.ipynb` (exécuté de bout en bout, 0 erreur)
- Script d'entraînement : `ml/training/model.py` (exécutable via `python -m ml.training.model`)
- Fonctions de feature engineering : `ml/training/features.py` (couvertes par `tests/test_features.py`, 20 tests unitaires)
- `random_state=42` utilisé de façon cohérente pour le split et la validation croisée
