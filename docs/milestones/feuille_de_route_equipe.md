# CineMatch — Feuille de route équipe (3 personnes)

Repo : https://github.com/Colin-12/CineMatch
Période restante : J2 (8 juillet, après-midi) → J4 (10 juillet, 17h)
Milestones 1 à 6 : rédigés. Reste : implémentation technique + démo.

---

## 0. Structure de repo à créer en premier (30 min, ensemble)

Avant toute répartition, une personne (idéalement Personne A) pousse ce squelette pour que tout le monde clone la même base.

**Action immédiate** : Personne A crée cette arborescence + un README minimal + `.env.example` + `requirements.txt` de base, commit `chore: init project structure`, push sur `main`. Tout le monde clone ensuite.

---

## 1. Conventions communes (à respecter par tous)

### Git

- **Branche protégée** : `main` — jamais de commit direct dessus.
- **Nommage des branches** : `feature/<domaine>-<courte-description>`
  - ex. `feature/data-ingestion-bronze`, `feature/ml-prediction-note`, `feature/llm-fiche-narrative`
- **Commits** (Conventional Commits) :
  - `feat: ajoute l'endpoint /prediction`
  - `fix: corrige le calcul RMSE`
  - `docs: ajoute model card sentiment`
  - `chore: met à jour requirements.txt`
  - `test: ajoute test unitaire endpoint fiche`
- **Pull Request obligatoire** avant merge sur `main`, avec au minimum 1 relecture croisée (même rapide) par un autre membre.
- Push **au moins 2 fois par jour**.

### Python

- `snake_case` pour variables/fonctions, `PascalCase` pour les classes.
- Type hints obligatoires sur les fonctions publiques.
- Docstring courte (1-2 lignes) par fonction.
- Formatage : `black` avant chaque commit si possible.
- Un seul `requirements.txt` à la racine, mis à jour au fil de l'eau.

### Secrets

- Clés API (`TMDB_API_KEY`, `ANTHROPIC_API_KEY`) **uniquement** dans `.env`, jamais commit.
- `.env.example` tenu à jour avec les noms de variables (sans valeurs).

### Versioning des modèles

- Format : `<nom_modele>_v<major>.<minor>_<AAAAMMJJ>.joblib`
  - ex. `svd_rating_v1.0_20260709.joblib`
- Chaque modèle versionné est accompagné d'une **model card** (`.md`) dans `ml/model_cards/` ou `nlp/model_cards/`.

---

## 2. Le point critique : définir les contrats AVANT de coder (aujourd'hui, 30-45 min)

1. Schéma exact des tables Gold (`FILM`, `UTILISATEUR`, `NOTATION`, `AVIS`, `SENTIMENT_SCORE`, `PREDICTION_NOTE`).
2. Contrat des endpoints FastAPI (requête / réponse JSON attendues).
3. Un jeu de données d'exemple (10-20 lignes en dur, JSON) pour tester chaque module isolément.

---

## 3. Répartition des rôles

### 🟦 Personne A — Data & Pipeline Engineer

Datalake, ETL, FastAPI (socle commun), CI/CD de base. Chemin critique : livrer un échantillon Gold exploitable au plus tôt.

- Ingestion Bronze : MovieLens 1M + API TMDB (throttling, cache).
- Transformation Silver : nettoyage, typage, dédoublonnage.
- Transformation Gold : tables `FILM`, `UTILISATEUR`, `NOTATION`, `AVIS`.
- Orchestration Prefect (idempotence, 0 doublon après double exécution).
- Squelette FastAPI + routing de base.
- CI GitHub Actions : `pytest` + `flake8`.

**Seuil** : pipeline Bronze→Gold < 10 min, 0 doublon après double exécution.

### 🟩 Personne B — ML & NLP Engineer

Option C (prédiction de note) + Option B (sentiment, fine-tuning Hugging Face).

- Exploration data, choix SVD vs LightGBM.
- Entraînement + baseline naïve (amélioration ≥ 10%).
- Export `.joblib` versionné + model card.
- Endpoint `/prediction`.
- Fine-tuning NLP (Colab), décision source des avis **aujourd'hui**.
- Checkpoints sur Drive.
- Endpoint `/sentiment` + model card NLP.

**Seuil** : RMSE < 1.0 et amélioration ≥ 10% vs baseline ; F1 > 0.70 et accuracy > 0.72.

**Filet de sécurité** : SVD basique si LightGBM trop long ; test NLP sur 100 exemples avant entraînement complet.

### 🟨 Personne C — LLM & Restitution Engineer

3 fonctionnalités LLM (fiche, comparaison, recommandation) + dashboard Streamlit + démo finale.

- Prompts (`llm/prompts/`) + appels API Anthropic (`claude-sonnet-4-6`).
- Endpoints `/fiche`, `/comparaison`, `/recommandation`.
- Dashboard Streamlit : 5 vues + vue admin.
- Démarrage immédiat en mockant les réponses de B et A.
- Script de démo finale.

**Seuil** : fiche ≤ 5s, comparaison ≤ 8s, recommandation ≤ 5s (≥ 5 suggestions justifiées) ; dashboard < 3s/vue.

---

## 4. Points de synchronisation obligatoires

| Moment | Objectif |
|---|---|
| Aujourd'hui (J2), avant de se séparer | Valider les contrats (schéma Gold, endpoints, jeu de données mock) |
| Fin J2 / début J3 | Personne A partage un premier échantillon Gold réel → B et C remplacent leurs mocks |
| Fin J3 | Vérification croisée : endpoints, dashboard, CI verte |
| J4 matin | Répétition de la démo complète, chasse aux erreurs non gérées |

---

## 5. Rappel des risques (Milestone 2) qui touchent la répartition

- **R1** (Prefect non maîtrisé) → Personne A : si blocage, script séquentiel simple.
- **R7** (source des avis non définie) → Personne B : décision aujourd'hui.
- **R9** (charge J3 sous-estimée) → simplifier en SVD basique si besoin ; LLM peut absorber le planning.
- **R10** (dérive planning ETL) → jalon go/no-go fin J2 : sinon, B et C continuent sur mock + CSV manuel.
