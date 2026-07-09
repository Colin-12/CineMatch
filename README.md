# CineMatch

Plateforme de recommandation et d'analyse de films : pipeline data (Bronze/Silver/Gold), API FastAPI (prédiction, sentiment, fonctionnalités LLM) et dashboard Streamlit.

## Structure du repo

```
CineMatch/
├── data/            # datalake bronze/silver (non versionné, cf. .gitignore) ; gold = Supabase Postgres
├── pipeline/         # flows Prefect (ingestion, transformation)
├── api/              # FastAPI (routers: fiche, comparaison, recommandation, prediction, sentiment)
├── ml/                # entraînement + modèles versionnés (prédiction de note)
├── nlp/               # entraînement + modèles versionnés (sentiment)
├── llm/               # prompts versionnés (Anthropic Claude)
├── dashboard/         # Streamlit
├── tests/             # tests unitaires/intégration
├── docs/milestones/   # documents de milestones
└── .github/workflows/ # CI (pytest, flake8)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # puis renseigner les clés API
```

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

## Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

## Lancer le pipeline

Le Gold est stocké dans une base Postgres (Supabase) — `DATABASE_URL` doit être
renseignée dans `.env` (connection string "Session pooler", pas la connexion
directe : le host direct `db.xxx.supabase.co` est IPv6-only et peut ne pas
résoudre selon le réseau).

```bash
python pipeline/ingestion_bronze.py   # MovieLens 1M + recherche/avis TMDB (nécessite TMDB_API_KEY)
python pipeline/transform_silver.py   # nettoyage/typage/dédoublonnage -> data/silver/*.csv
python pipeline/transform_gold.py     # charge film/utilisateur/notation/avis dans Supabase
```

Idempotent : chaque étape peut être rejouée sans créer de doublons (Bronze
saute le téléchargement/les appels déjà en cache, Gold fait un `TRUNCATE` +
rechargement complet). Pipeline complet testé à ~1 min pour l'ensemble
Bronze→Gold (seuil Milestone 3 : < 10 min).

Sans `TMDB_API_KEY`, l'ingestion TMDB (enrichissement des films + avis) est
ignorée proprement (warning, pas d'erreur) — seul MovieLens est chargé.

## Conventions

Voir la feuille de route équipe (`docs/milestones/`) pour les conventions Git, Python, secrets et versioning des modèles.
