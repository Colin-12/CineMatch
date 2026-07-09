"""Endpoint `/prediction` : note predite (LightGBM), features live depuis Gold.

Contrairement a `ml/training/model.py::predict_note` (qui reutilise le
`FeatureContext` fige au moment de l'entrainement -- agregats calcules une
seule fois sur le train et serialises dans le bundle), cet endpoint
reconstruit les features utilisateur/film a la volee, a chaque requete, a
partir de l'etat courant de la base Gold (`notation`, `film`). Seuls les
poids du modele LightGBM deja entraine et le vocabulaire de genres qu'il
attend en entree sont reutilises tels quels (contrat structurel du modele,
independant des donnees) : aucune moyenne ni agregat de note n'est mis en
cache. Choix assume : un aller-retour base de donnees (plusieurs requetes
SQL legeres + une lecture complete de la table `film`, ~3883 lignes) par
appel, sans contrainte de latence stricte sur cet endpoint.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from api.schemas.gold import PredictionNote
from ml.training import features
from ml.training.model import ModelBundle, load_model

load_dotenv()

router = APIRouter(prefix="/prediction", tags=["prediction"])

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "ml"
    / "models"
    / "lightgbm_rating_v1.0_20260709.joblib"
)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    """Charge le bundle LightGBM une seule fois par processus (poids figes).

    Seuls `bundle.model`, `bundle.context.genre_vocab` et
    `bundle.context.feature_cols` sont utilises ici (contrat du modele) ; les
    agregats `bundle.context.user_aggregates`/`film_aggregates`/`films_features`
    (figes au moment de l'entrainement) sont volontairement ignores : les
    features sont recalculees live pour chaque requete (voir docstring module).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")
    bundle = load_model(MODEL_PATH)
    if bundle.model_type != "lightgbm":
        raise ValueError(
            f"Modele attendu 'lightgbm', trouve '{bundle.model_type}' dans {MODEL_PATH}"
        )
    return bundle


def get_connection() -> psycopg2.extensions.connection:
    """Ouvre une connexion a la base Gold (`DATABASE_URL`)."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


# --------------------------------------------------------------------------
# Recuperation live des donnees Gold necessaires a un couple (user_id, film_id)
# --------------------------------------------------------------------------


def fetch_film_exists(conn: psycopg2.extensions.connection, film_id: int) -> bool:
    """Verifie l'existence du film (404 sinon : pas de metadonnees a exploiter)."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM film WHERE film_id = %s;", (film_id,))
        return cur.fetchone() is not None


def fetch_full_film_table(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Recupere le catalogue complet courant (petit volume, ~3883 lignes).

    Necessaire pour recalculer, a l'identique de l'entrainement mais sur
    l'etat courant de Gold, le vocabulaire de genres (`build_genre_vocab`) et
    les medianes d'imputation TMDB (`compute_tmdb_imputation_values`).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT film_id, annee, genres, overview, popularite, note_tmdb FROM film;"
        )
        rows = cur.fetchall()
    return pd.DataFrame(
        rows,
        columns=["film_id", "annee", "genres", "overview", "popularite", "note_tmdb"],
    )


def fetch_global_mean(conn: psycopg2.extensions.connection) -> float:
    """Moyenne globale courante des notes (recalculee a chaque requete)."""
    with conn.cursor() as cur:
        cur.execute("SELECT AVG(note) FROM notation;")
        (value,) = cur.fetchone()
    return float(value) if value is not None else 0.0


def fetch_notation_aggregate(
    conn: psycopg2.extensions.connection, column: str, entity_id: int
) -> dict[str, Any]:
    """Agregat (moyenne/ecart-type/nombre) courant sur `notation`, filtre sur
    `user_id` ou `film_id`. Utilise STDDEV_SAMP (ddof=1), coherent avec le
    `.std()` pandas utilise a l'entrainement (`ml/training/features.py`)."""
    if column not in {"user_id", "film_id"}:
        raise ValueError(f"colonne inattendue : {column}")
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT AVG(note), STDDEV_SAMP(note), COUNT(*) "
            f"FROM notation WHERE {column} = %s;",
            (entity_id,),
        )
        mean_note, std_note, n_notations = cur.fetchone()
    return {
        "mean": float(mean_note) if mean_note is not None else None,
        "std": float(std_note) if std_note is not None else 0.0,
        "n": int(n_notations),
    }


# --------------------------------------------------------------------------
# Assemblage de la ligne de features (reutilise ml/training/features.py)
# --------------------------------------------------------------------------


def build_live_feature_row(
    conn: psycopg2.extensions.connection,
    user_id: int,
    film_id: int,
    bundle: ModelBundle,
) -> pd.DataFrame:
    """Construit la ligne de features (1 x n_colonnes) pour un couple donne,
    entierement recalculee depuis l'etat courant de Gold (pas de precalcul)."""
    if not fetch_film_exists(conn, film_id):
        raise HTTPException(status_code=404, detail=f"film_id {film_id} introuvable")

    full_film_df = fetch_full_film_table(conn)

    genre_vocab = features.build_genre_vocab(full_film_df)
    if genre_vocab != bundle.context.genre_vocab:
        # Contrat structurel rompu (nouveau genre apparu cote Gold depuis
        # l'entrainement) : on refuse de predire plutot que de mal aligner
        # les colonnes one-hot attendues par le modele.
        raise HTTPException(
            status_code=500,
            detail=(
                "Vocabulaire de genres courant incompatible avec celui attendu "
                "par le modele entraine ; reentrainement necessaire."
            ),
        )

    imputation_values = features.compute_tmdb_imputation_values(full_film_df)
    films_features = features.prepare_film_features(
        full_film_df, genre_vocab, imputation_values
    )
    film_row = films_features.loc[films_features["film_id"] == film_id].copy()

    global_mean = fetch_global_mean(conn)
    user_agg = fetch_notation_aggregate(conn, "user_id", user_id)
    film_agg = fetch_notation_aggregate(conn, "film_id", film_id)

    film_row["user_id"] = user_id
    film_row["user_mean_note"] = user_agg["mean"] if user_agg["n"] > 0 else global_mean
    film_row["user_std_note"] = user_agg["std"]
    film_row["user_n_notations"] = user_agg["n"]
    film_row["film_mean_note"] = film_agg["mean"] if film_agg["n"] > 0 else global_mean
    film_row["film_std_note"] = film_agg["std"]
    film_row["film_n_notations"] = film_agg["n"]

    return film_row.reset_index(drop=True)


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------


@router.get("", response_model=PredictionNote)
def predict_note(user_id: int, film_id: int) -> PredictionNote:
    """Predit la note qu'un utilisateur donnerait a un film. Seuil : RMSE < 1.0."""
    bundle = get_bundle()
    conn = get_connection()
    try:
        feature_row = build_live_feature_row(conn, user_id, film_id, bundle)
    finally:
        conn.close()

    feature_matrix = feature_row[bundle.context.feature_cols]
    raw_prediction = bundle.model.predict(feature_matrix)
    note_predite = float(features.clip_ratings(np.asarray(raw_prediction))[0])

    return PredictionNote(user_id=user_id, film_id=film_id, note_predite=note_predite)
