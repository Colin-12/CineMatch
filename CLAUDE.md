# CLAUDE.md

Contexte projet pour Claude Code sur **CineMatch** (recommandation/analyse de films — pipeline Bronze/Silver/Gold, API FastAPI, dashboard Streamlit). Voir [README.md](README.md) et [docs/milestones/feuille_de_route_equipe.md](docs/milestones/feuille_de_route_equipe.md) pour le contexte complet équipe.

## Conventions Git

- `main` est protégée : jamais de commit direct dessus.
- Branches : `feature/<domaine>-<courte-description>` (ex. `feature/ml-prediction-note`, `feature/nlp-sentiment-finetune`).
- Commits [Conventional Commits](https://www.conventionalcommits.org/) : `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- PR obligatoire avant merge sur `main`, avec au moins 1 relecture croisée (même rapide) par un autre membre.
- Push au moins 2 fois par jour.
- Secrets (`TMDB_API_KEY`, `ANTHROPIC_API_KEY`) uniquement dans `.env`, jamais commit ; `.env.example` tenu à jour (noms de variables sans valeurs).

## Conventions Python

- `snake_case` pour variables/fonctions, `PascalCase` pour les classes.
- Type hints obligatoires sur les fonctions publiques.
- Docstring courte (1-2 lignes) par fonction.
- Formatage `black` avant chaque commit (`flake8` configuré à 88 colonnes, voir `.flake8`).
- Un seul `requirements.txt` à la racine, mis à jour au fil de l'eau.

## Versioning des modèles

- Format de fichier : `<nom_modele>_v<major>.<minor>_<AAAAMMJJ>.joblib`
  - ex. `svd_rating_v1.0_20260709.joblib`
- Chaque modèle versionné est accompagné d'une **model card** (`.md`) :
  - modèles de prédiction de note → `ml/model_cards/`
  - modèles de sentiment/NLP → `nlp/model_cards/`
- Modèles entraînés → `ml/models/` ; scripts d'entraînement → `ml/training/` (idem `nlp/training/` côté NLP).

## Mon périmètre (Personne B — ML & NLP Engineer)

Je suis responsable de deux endpoints et de leur pipeline ML/NLP associé :

### `/prediction` (note prédite)
- Fichiers : [api/routers/prediction.py](api/routers/prediction.py), entraînement dans `ml/training/`, modèles dans `ml/models/`, model card dans `ml/model_cards/`.
- Approche : SVD vs LightGBM (SVD = filet de sécurité si LightGBM trop long), comparé à une baseline naïve.
- Seuils à respecter : **RMSE < 1.0** et **amélioration ≥ 10% vs baseline naïve**.
- Schéma de sortie : `PredictionNote` (`api/schemas/gold.py`) — `user_id`, `film_id`, `note_predite`.

### `/sentiment` (classification des avis TMDB)
- Fichiers : [api/routers/sentiment.py](api/routers/sentiment.py), fine-tuning Hugging Face dans `nlp/training/` (checkpoints sur Drive pendant l'entraînement), modèles dans `nlp/models/` (à créer si besoin), model card dans `nlp/model_cards/`.
- Seuils à respecter : **F1 > 0.70** et **accuracy > 0.72** ; tester sur ~100 exemples avant l'entraînement complet.
- Schéma de sortie : `SentimentScore` (`api/schemas/gold.py`) — `film_id`, `score`, `label`.

### Point d'attention données

La table Gold `film` a été enrichie récemment avec des champs TMDB : `overview`, `popularite`, `note_tmdb`, `affiche_path` (voir [pipeline/transform_gold.py](pipeline/transform_gold.py) et [api/schemas/gold.py](api/schemas/gold.py)). **Couverture partielle du catalogue** — ces colonnes peuvent être `NULL`. En tenir compte dans le feature engineering (ex. imputation ou exclusion explicite) plutôt que de supposer leur présence.

### Hors périmètre

Les endpoints `/fiche`, `/comparaison`, `/recommandation` (LLM, Personne C) et le pipeline Bronze/Silver/Gold/CI (Personne A) ne sont pas de mon ressort — coordination via PR/revue croisée si des changements de contrat sont nécessaires (schéma Gold, format des endpoints).
