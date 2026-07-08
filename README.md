# CineMatch

Plateforme de recommandation et d'analyse de films : pipeline data (Bronze/Silver/Gold), API FastAPI (prédiction, sentiment, fonctionnalités LLM) et dashboard Streamlit.

## Structure du repo

```
CineMatch/
├── data/            # datalake bronze/silver/gold (non versionné, cf. .gitignore)
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

```bash
python pipeline/ingestion_bronze.py
python pipeline/transform_silver.py
python pipeline/transform_gold.py
```

## Conventions

Voir la feuille de route équipe (`docs/milestones/`) pour les conventions Git, Python, secrets et versioning des modèles.
