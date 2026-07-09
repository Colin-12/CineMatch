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
├── llm/               # prompts versionnés (Google Gemini)
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

Gold est chargé dans Supabase (source de vérité, partagée par l'équipe) **et**
exporté en snapshot Parquet dans `data/gold/*.parquet` (film, utilisateur,
notation, avis) — pratique pour Personne B afin d'itérer sur l'entraînement
ML en local sans requêter Supabase à chaque run.

Idempotent : chaque étape peut être rejouée sans créer de doublons (Bronze
saute le téléchargement/les appels déjà en cache, Gold fait un `TRUNCATE` +
rechargement complet côté Supabase, et écrase le Parquet local). Pipeline
complet testé à ~1 min pour l'ensemble Bronze→Gold (seuil Milestone 3 : < 10 min).

Sans `TMDB_API_KEY`, l'ingestion TMDB (enrichissement des films + avis) est
ignorée proprement (warning, pas d'erreur) — seul MovieLens est chargé.

## État d'avancement (J3)

### ✅ Data & Pipeline (Personne A)

- Bronze : MovieLens 1M (téléchargement + extraction) + enrichissement TMDB
  (recherche par film + avis), requêtes parallélisées (~1min20 pour 1800
  films au lieu de ~35-40min en séquentiel).
- Silver : nettoyage/typage/dédoublonnage (movies, users, ratings, avis).
  Les avis sont construits à partir des reviews TMDB, rattachés à un
  utilisateur MovieLens ayant réellement noté le film (tirage aléatoire
  sans remise parmi les vrais votants, pour respecter les FK).
- Gold : tables `film` (enrichie TMDB : synopsis/popularité/note/affiche),
  `utilisateur`, `notation`, `avis` chargées dans Supabase (Postgres) +
  snapshot Parquet local dans `data/gold/`. `sentiment_score` et
  `prediction_note` existent en schéma mais restent vides (Personne B).
- Pipeline complet Bronze→Gold : ~1 min, idempotent (testé sur double
  exécution, 0 doublon).
- CI GitHub Actions (flake8 + pytest) verte, config flake8 alignée sur
  black (88 col).

**Chiffres actuels** : 3 883 films, 6 040 utilisateurs, 1 000 209
notations, 2 082 avis (856 films couverts).

### ✅ LLM & Restitution (Personne C)

Provider : **Google Gemini** (`gemini-2.5-flash`, thinking désactivé pour
la latence) — changement par rapport à la feuille de route initiale
(Anthropic), pour raison de budget.

- `GET /fiche?titre=...` — fiche narrative structurée (accroche, résumé,
  ambiance, pourquoi la regarder). Recherche par titre dans le catalogue
  Gold ; si absent, Gemini répond depuis ses connaissances
  (`source: "connaissances_llm"`). ~3-4s.
- `GET /comparaison?titre_1=...&titre_2=...` — compare 2 films (points
  communs, différences clés, note comparative appuyée sur les vraies
  notes TMDB quand disponibles). Chaque film résolu indépendamment
  (catalogue ou connaissances LLM). ~3-4s.
- `GET /recommandation?duree_max_heures=...&genre=...&note_min=...` —
  ≥5 suggestions justifiées à partir d'un formulaire d'envies, basé
  uniquement sur les connaissances de Gemini (pas de filtrage catalogue).
  ~3-4s.
- Dashboard Streamlit : les 3 vues LLM branchées sur l'API réelle
  (affiches TMDB, badges de source, gestion d'erreurs propre). Vues
  Prédiction/Sentiment/Admin en placeholder (dépendent de Personne B).

### ⬜ ML & NLP (Personne B)

- Prédiction de note (SVD/LightGBM sur `notation`, 1M lignes disponibles)
  et fine-tuning sentiment (sur `avis`, 2 082 lignes disponibles, labels
  à définir) — pas encore démarré à ce stade.
- Endpoints `/prediction` et `/sentiment` : stubs FastAPI en place,
  logique à implémenter.

## Conventions

Voir la feuille de route équipe (`docs/milestones/`) pour les conventions Git, Python, secrets et versioning des modèles.
