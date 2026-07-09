# Model Card — tfidf_logreg_sentiment_v1.1_20260709

## Résumé

| | |
|---|---|
| **Dossier modèle** | `nlp/models/tfidf_logreg_sentiment_v1.1_20260709.joblib` |
| **Endpoint cible** | `/sentiment` (classification des avis TMDB) — **modèle actuellement chargé par `api/routers/sentiment.py::get_bundle()`** |
| **Type de modèle** | TF-IDF (`TfidfVectorizer`) + régression logistique (`LogisticRegression(class_weight="balanced")`) |
| **Version** | 1.1 |
| **Date d'entraînement** | 2026-07-09 |
| **Auteur** | Personne B (ML & NLP Engineer) |
| **Données** | Base Gold CineMatch — table `avis` (1 970 avis textuels TMDB sur 2 082, après exclusion des avis sans `note_auteur`), labellisés directement depuis la note de l'auteur de l'avis (`note_auteur`, échelle 0-10 TMDB), et non plus depuis la note MovieLens d'un utilisateur synthétiquement rattaché (voir « Changement de méthode » ci-dessous) |
| **Notebook source** | `nlp/training/sentiment_analysis.ipynb` |
| **Script d'entraînement** | `nlp/training/model.py` (voir « Reproductibilité » — **ne pas invoquer via `python -m nlp.training.model --tfidf-only`**, voir avertissement dédié) |

## Changement de méthode : label dérivé de `note_auteur` (et non plus de MovieLens)

La version précédente (voir `nlp/model_cards/distilbert_sentiment_v1.0_20260709_abandonnee.md`) dérivait le label de sentiment de la note **MovieLens** d'un utilisateur rattaché *synthétiquement* à l'avis — un signal indirect et bruité, sans lien garanti avec le sentiment réellement exprimé dans le texte de l'avis. Cette approche a été **abandonnée** : `nlp/training/features.py` conserve les fonctions correspondantes (`derive_label_from_note_movielens_legacy`, `add_sentiment_labels_movielens_legacy`) à des fins de comparaison historique documentée, mais elles ne sont plus utilisées pour l'entraînement.

Le label est désormais dérivé directement de `note_auteur` (`features.derive_label_from_note_auteur` / `features.add_sentiment_labels`), la note TMDB donnée par l'auteur de l'avis lui-même — un signal bien plus directement corrélé au sentiment du texte associé. Les avis sans `note_auteur` (112 sur 2 082, soit 5,4%) sont exclus plutôt que de recevoir un label par défaut.

## ⚠️ Statut vs seuils de la feuille de route — **accuracy atteinte, F1 macro non atteint**

| Critère | Seuil | Meilleur résultat obtenu | Statut |
|---|---|---|---|
| F1 macro | > 0,70 | 0,665 (`tfidf_logreg`, modèle retenu) | ❌ Non atteint |
| Accuracy | > 0,72 | 0,774 (`tfidf_logreg`, modèle retenu) | ✅ Atteint |

Le changement de label (note_auteur au lieu de MovieLens) améliore nettement les résultats par rapport à la v1.0 (F1 macro 0,423 → 0,665 ; accuracy 0,540 → 0,774), confirmant que le bruit de label était bien la cause principale des mauvaises performances précédentes. Le seuil d'accuracy est désormais atteint ; le F1 macro reste en dessous du seuil cible, principalement à cause du déséquilibre de classes persistant (voir « Diagnostic et limitations »).

## Comparaison des modèles (test set, 394 avis, split stratifié 80/20, `random_state=42`)

| Modèle | Accuracy | F1 macro | F1 pondéré | Amélioration F1 macro vs baseline naïve |
|---|---|---|---|---|
| baseline_majoritaire | 0,690 | 0,272 | 0,564 | — |
| **tfidf_logreg (retenu, production)** | **0,774** | **0,665** | **0,779** | +144,1% |
| distilbert (entraîné, non retenu) | 0,731 | 0,647 | 0,746 | +137,5% |

**Le modèle retenu en production pour `/sentiment` est TF-IDF + régression logistique.** Contrairement à la v1.0 (où l'écart entre TF-IDF et DistilBERT n'était pas significatif), ici TF-IDF+LogReg **surpasse DistilBERT sur les trois métriques** (accuracy, F1 macro, F1 pondéré) — un écart net et non ambigu :

1. **Performance strictement supérieure** : +0,043 en accuracy, +0,018 en F1 macro, +0,033 en F1 pondéré, en faveur de TF-IDF+LogReg.
2. **Coût de calcul très inférieur** : quelques secondes d'entraînement CPU contre ~111,6 minutes CPU pour DistilBERT (2 époques, `train_runtime` = 6 697 s) — sans aucun gain de performance en contrepartie.
3. **Conformité à la consigne du 09/07/2026** : le fine-tuning DistilBERT tournait en parallèle et ne devait pas bloquer la mise en production de `/sentiment` ; TF-IDF+LogReg sert de référence temporaire pendant que DistilBERT s'entraîne. Une fois DistilBERT disponible et comparé, la consigne prévoyait de basculer l'endpoint dessus **s'il se révélait supérieur** — ce qui n'est pas le cas ici : DistilBERT reste donc documenté (voir `distilbert_sentiment_v1.1_20260709.md`) mais **n'est pas exporté vers l'endpoint**.

### Rapport de classification détaillé (TF-IDF+LogReg, test set)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| negatif | 0,463 | 0,559 | 0,507 | 34 |
| neutre | 0,600 | 0,648 | 0,623 | 88 |
| positif | 0,888 | 0,842 | 0,864 | 272 |
| **accuracy** | | | **0,774** | 394 |
| macro avg | 0,650 | 0,683 | 0,665 | 394 |
| weighted avg | 0,786 | 0,774 | 0,779 | 394 |

### Matrice de confusion (TF-IDF+LogReg)

| | prédit négatif | prédit neutre | prédit positif |
|---|---|---|---|
| **réel négatif** | 19 | 11 | 4 |
| **réel neutre** | 6 | 57 | 25 |
| **réel positif** | 16 | 27 | 229 |

88 exemples sur 394 sont mal classés (22,3%). La confusion dominante reste `positif`↔`neutre` (25 + 27 = 52 cas, soit 59% des erreurs), cohérente avec des frontières de classes adjacentes difficiles à trancher ; les confusions `negatif`↔`positif` (4 + 16 = 20 cas) sont nettement moins fréquentes que dans la v1.0, ce qui confirme un signal de label plus propre.

## Diagnostic et limitations

**1. Déséquilibre des classes toujours présent.** Distribution des 1 970 avis labellisés : `positif` 69,1% (1 361), `neutre` 22,3% (440), `negatif` 8,6% (169). La classe `negatif` reste minoritaire (34 exemples seulement dans le test), ce qui explique la précision/rappel plus faibles sur cette classe (F1 = 0,507) et tire le F1 macro sous le seuil cible malgré une bonne performance globale (F1 pondéré = 0,779). `class_weight="balanced"` atténue mais ne supprime pas cet effet.

**2. Volume de données modeste.** 1 970 avis labellisés (1 576 train / 394 test) restent un jeu de données de taille limitée pour une tâche de classification de texte à 3 classes, en particulier pour la classe minoritaire `negatif` (135 train / 34 test).

**3. F1 macro sous le seuil cible malgré l'amélioration.** Le passage au label `note_auteur` corrige le problème structurel principal de la v1.0 (bruit de label), mais ne compense pas entièrement le déséquilibre de classes : un F1 macro de 0,665 reste sous le seuil de 0,70 fixé par `CLAUDE.md`. Une amélioration future passerait par plus de données pour la classe `negatif` (sur-échantillonnage, collecte ciblée) plutôt que par un changement d'algorithme.

**4. Pas de troncature de texte.** Contrairement à DistilBERT (limité à `MAX_LENGTH=256` tokens), le pipeline TF-IDF traite l'intégralité du texte de chaque avis — un avantage structurel pour les avis longs (moyenne ~250 mots).

## Recommandation

TF-IDF+LogReg est recommandé comme modèle de production pour `/sentiment` en l'état : il atteint le seuil d'accuracy, surpasse DistilBERT sur toutes les métriques, et est trivialement moins coûteux à ré-entraîner. La priorité d'amélioration future est la collecte de davantage d'avis pour la classe `negatif`, plutôt qu'un changement d'architecture de modèle.

## Reproductibilité

- Notebook source : `nlp/training/sentiment_analysis.ipynb`
- Script d'entraînement : `nlp/training/model.py`, fonction `train_tfidf_logreg` / `save_tfidf_reference_model`
- **Invocation recommandée** : `python -c "from nlp.training import model; model.main(tfidf_only=True)"` (import normal du module).
  ⚠️ **Ne pas utiliser `python -m nlp.training.model --tfidf-only`** : cette invocation exécute `nlp/training/model.py` en tant que `__main__`, ce qui fige `SentimentModelBundle.__module__ == "__main__"` dans le fichier `.joblib` sauvegardé et casse son rechargement ultérieur depuis un import normal (ex. `api/routers/sentiment.py`), avec `AttributeError: module '__main__' has no attribute 'SentimentModelBundle'` au chargement. Le bundle actuel (`tfidf_logreg_sentiment_v1.1_20260709.joblib`) a été régénéré via l'invocation recommandée ci-dessus pour corriger ce problème une première fois ; il est recommandé de corriger le point d'entrée CLI (`if __name__ == "__main__":`) pour éviter que ce piège ne se reproduise.
- Fonctions de feature engineering / labellisation : `nlp/training/features.py` (`add_sentiment_labels`, dérivé de `note_auteur`)
- `random_state=42` utilisé de façon cohérente pour le split train/test (`features.stratified_label_train_test_split`)
- Temps d'entraînement : quelques secondes CPU (pas de GPU nécessaire)
