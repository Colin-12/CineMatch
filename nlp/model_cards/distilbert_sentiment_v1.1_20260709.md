# Model Card — distilbert_sentiment_v1.1_20260709

## Résumé

| | |
|---|---|
| **Dossier modèle** | `nlp/models/distilbert_sentiment_v1.1_20260709/` (HF checkpoint : `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `bundle_meta.joblib`) |
| **Endpoint cible** | `/sentiment` (classification des avis TMDB) — **entraîné et évalué, mais NON chargé par l'endpoint** (voir « Statut de production » ci-dessous) |
| **Type de modèle** | DistilBERT (`distilbert-base-uncased`) fine-tuné, tête de classification 3 classes, `WeightedTrainer` (perte `CrossEntropyLoss` pondérée par classe) |
| **Version** | 1.1 |
| **Date d'entraînement** | 2026-07-09 |
| **Auteur** | Personne B (ML & NLP Engineer) |
| **Données** | Base Gold CineMatch — table `avis` (1 970 avis textuels TMDB sur 2 082, après exclusion des avis sans `note_auteur`), labellisés directement depuis la note de l'auteur de l'avis (`note_auteur`), même jeu de données et même label que `tfidf_logreg_sentiment_v1.1_20260709` (voir cette model card pour le détail du changement de méthode par rapport à la v1.0) |
| **Notebook source** | `nlp/training/sentiment_analysis.ipynb` |
| **Script d'entraînement** | `nlp/training/model.py` (exécutable via `python -m nlp.training.model`, ~111,6 minutes CPU pour 2 époques, aucun GPU disponible) |

## Statut de production : **entraîné, évalué, NON retenu**

Ce fine-tuning DistilBERT a été lancé en arrière-plan pendant que TF-IDF+LogReg servait de modèle de référence temporaire pour `/sentiment` (conformément à la consigne : « si le fine-tuning DistilBERT en arrière-plan termine avec de meilleurs résultats, on basculera le endpoint dessus »). **Les résultats ne surpassent pas TF-IDF+LogReg sur aucune des trois métriques** (voir comparaison ci-dessous) : l'endpoint `/sentiment` reste donc chargé sur `tfidf_logreg_sentiment_v1.1_20260709.joblib` (voir `api/routers/sentiment.py::get_bundle()`), sans changement de code nécessaire.

## ⚠️ Statut vs seuils de la feuille de route — **accuracy atteinte, F1 macro non atteint**

| Critère | Seuil | Résultat DistilBERT | Statut |
|---|---|---|---|
| F1 macro | > 0,70 | 0,647 | ❌ Non atteint |
| Accuracy | > 0,72 | 0,731 | ✅ Atteint |

## Comparaison des modèles (test set, 394 avis, split stratifié 80/20, `random_state=42`)

| Modèle | Accuracy | F1 macro | F1 pondéré | Amélioration F1 macro vs baseline naïve |
|---|---|---|---|---|
| baseline_majoritaire | 0,690 | 0,272 | 0,564 | — |
| **tfidf_logreg (retenu, production)** | **0,774** | **0,665** | **0,779** | +144,1% |
| distilbert (ce modèle, non retenu) | 0,731 | 0,647 | 0,746 | +137,5% |

DistilBERT est inférieur à TF-IDF+LogReg de 0,043 en accuracy, 0,018 en F1 macro et 0,033 en F1 pondéré, pour un coût d'entraînement d'environ 111,6 minutes CPU contre quelques secondes pour TF-IDF+LogReg (`train_runtime` = 6 697 s, 198 steps, 2 époques). Voir `tfidf_logreg_sentiment_v1.1_20260709.md` pour la justification complète du choix du modèle de production.

### Rapport de classification détaillé (DistilBERT, test set)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| negatif | 0,478 | 0,647 | 0,550 | 34 |
| neutre | 0,483 | 0,659 | 0,558 | 88 |
| positif | 0,912 | 0,765 | 0,832 | 272 |
| **accuracy** | | | **0,731** | 394 |
| macro avg | 0,625 | 0,690 | 0,647 | 394 |
| weighted avg | 0,779 | 0,731 | 0,746 | 394 |

### Matrice de confusion (DistilBERT)

| | prédit négatif | prédit neutre | prédit positif |
|---|---|---|---|
| **réel négatif** | 22 | 9 | 3 |
| **réel neutre** | 13 | 58 | 17 |
| **réel positif** | 11 | 53 | 208 |

106 exemples sur 394 sont mal classés (26,9%). DistilBERT obtient un meilleur **rappel** sur les classes minoritaires (`negatif` : 0,647 vs 0,559 pour TF-IDF ; `neutre` : 0,659 vs 0,648) mais au prix d'une **précision** nettement plus faible sur ces mêmes classes (`negatif` : 0,478 vs 0,463 — comparable ; `neutre` : 0,483 vs 0,600) et d'un rappel plus faible sur `positif` (0,765 vs 0,842), qui domine le jeu de test (69,1%) : le modèle "sur-corrige" vers les classes minoritaires (via la pondération de la perte), ce qui abîme la performance sur la classe majoritaire sans gain net en F1 macro par rapport à TF-IDF.

## Diagnostic et limitations

**1. Aucun gain par rapport à TF-IDF malgré un coût de calcul ~1000x supérieur.** Contrairement à la v1.0 (où l'écart entre les deux approches n'était pas significatif), ici l'écart est net et défavorable à DistilBERT sur les trois métriques : le volume et la nature du signal disponible (1 970 avis) ne permettent pas à un modèle de langage fine-tuné de dépasser un modèle linéaire beaucoup plus simple sur ce jeu de données.

**2. Déséquilibre des classes.** Distribution des 1 970 avis labellisés : `positif` 69,1%, `neutre` 22,3%, `negatif` 8,6%. La pondération de la perte (`WeightedTrainer`) améliore le rappel sur les classes minoritaires mais dégrade la précision, sans amélioration nette du F1 macro global par rapport à TF-IDF (qui utilise `class_weight="balanced"`, une approche plus légère et tout aussi efficace ici).

**3. Volume de données modeste pour un fine-tuning de transformeur.** 1 970 avis (1 576 train / 394 test) est un jeu de données petit pour fine-tuner ~66M de paramètres (`distilbert-base-uncased`) ; un modèle plus simple comme TF-IDF+LogReg généralise mieux à ce volume.

**4. Troncature à 256 tokens.** Les avis font en moyenne ~250 mots ; une part significative dépasse `MAX_LENGTH=256` (`nlp/training/model.py`) une fois tokenisés et est tronquée, ce qui peut faire perdre le jugement de conclusion en fin de texte — une limite structurelle que TF-IDF (qui traite le texte intégral) n'a pas.

**5. Coût CPU élevé pour un résultat inférieur.** ~111,6 minutes CPU pour 2 époques (aucun GPU disponible dans l'environnement d'entraînement), à comparer aux quelques secondes de TF-IDF+LogReg — un rapport coût/bénéfice défavorable pour un usage en production ou en itération fréquente (ré-entraînement).

## Recommandation

Ne pas basculer `/sentiment` sur ce modèle : TF-IDF+LogReg (`tfidf_logreg_sentiment_v1.1_20260709`) reste le modèle de production. Ce bundle DistilBERT est conservé à des fins de comparaison documentée et de référence si le volume de données venait à augmenter significativement (le fine-tuning de transformeurs tend à mieux passer à l'échelle des données que TF-IDF, mais ce n'est pas le cas observé ici).

## Reproductibilité

- Notebook source : `nlp/training/sentiment_analysis.ipynb`
- Script d'entraînement : `nlp/training/model.py` (exécutable via `python -m nlp.training.model`, ~111,6 minutes CPU pour 2 époques sur 1 576 exemples, aucun GPU disponible)
- Fonctions de feature engineering / labellisation : `nlp/training/features.py` (`add_sentiment_labels`, dérivé de `note_auteur`)
- `random_state=42` utilisé de façon cohérente pour le split train/test (aucune fuite user_id/film_id)
- Smoke test (~100 exemples, 1 époque) validé avant l'entraînement complet, conformément à `CLAUDE.md`
- ⚠️ Le fichier `bundle_meta.joblib` de ce modèle a initialement été sauvegardé avec `SentimentModelBundle.__module__ == "__main__"` (conséquence de l'invocation `python -m nlp.training.model`), ce qui cassait son rechargement via `joblib.load()` depuis un import normal. Corrigé en place (sans ré-entraînement) via un monkeypatch de `sys.modules["__main__"].SentimentModelBundle` suivi d'une ré-sauvegarde. Voir la model card `tfidf_logreg_sentiment_v1.1_20260709.md` pour la recommandation de corriger durablement le point d'entrée CLI du script.
