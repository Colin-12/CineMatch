"""Transformation Gold : tables FILM, UTILISATEUR, NOTATION, AVIS prêtes à consommer."""

import io
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from psycopg2.extras import execute_values

load_dotenv()

SILVER_DIR = Path(__file__).resolve().parent.parent / "data" / "silver"

DDL = """
CREATE TABLE IF NOT EXISTS film (
    film_id INTEGER PRIMARY KEY,
    titre TEXT NOT NULL,
    annee INTEGER,
    genres TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS utilisateur (
    user_id INTEGER PRIMARY KEY,
    age INTEGER
);

CREATE TABLE IF NOT EXISTS notation (
    user_id INTEGER NOT NULL REFERENCES utilisateur(user_id),
    film_id INTEGER NOT NULL REFERENCES film(film_id),
    note REAL NOT NULL,
    timestamp BIGINT NOT NULL,
    PRIMARY KEY (user_id, film_id)
);

CREATE TABLE IF NOT EXISTS avis (
    user_id INTEGER NOT NULL REFERENCES utilisateur(user_id),
    film_id INTEGER NOT NULL REFERENCES film(film_id),
    texte TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    PRIMARY KEY (user_id, film_id)
);

CREATE TABLE IF NOT EXISTS sentiment_score (
    film_id INTEGER PRIMARY KEY REFERENCES film(film_id),
    score REAL NOT NULL,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_note (
    user_id INTEGER NOT NULL,
    film_id INTEGER NOT NULL,
    note_predite REAL NOT NULL,
    PRIMARY KEY (user_id, film_id)
);
"""


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(os.environ["DATABASE_URL"])


@task
def create_schema() -> None:
    """Crée les tables Gold si elles n'existent pas déjà (idempotent)."""
    logger = get_run_logger()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    finally:
        conn.close()
    logger.info(
        "Schéma Gold vérifié/créé (film, utilisateur, notation, avis, "
        "sentiment_score, prediction_note)."
    )


@task
def load_utilisateurs() -> int:
    """Charge silver/users.csv dans la table utilisateur (full refresh idempotent)."""
    logger = get_run_logger()
    df = pd.read_csv(SILVER_DIR / "users.csv")
    df = df.astype(object).where(df.notna(), None)
    rows = list(df[["user_id", "age"]].itertuples(index=False, name=None))

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE utilisateur CASCADE")
            execute_values(
                cur,
                "INSERT INTO utilisateur (user_id, age) VALUES %s",
                rows,
                page_size=1000,
            )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"utilisateur: {len(rows)} lignes chargées")
    return len(rows)


@task
def load_films() -> int:
    """Charge silver/movies.csv dans la table film (full refresh idempotent)."""
    logger = get_run_logger()
    df = pd.read_csv(SILVER_DIR / "movies.csv")
    df = df.astype(object).where(df.notna(), None)
    df["genres"] = df["genres"].apply(
        lambda g: g.split("|") if isinstance(g, str) else []
    )
    rows = list(
        df[["film_id", "titre", "annee", "genres"]].itertuples(index=False, name=None)
    )

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE film CASCADE")
            execute_values(
                cur,
                "INSERT INTO film (film_id, titre, annee, genres) VALUES %s",
                rows,
                page_size=1000,
            )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"film: {len(rows)} lignes chargées")
    return len(rows)


@task
def load_notations() -> int:
    """Charge silver/ratings.csv dans notation via COPY (full refresh idempotent)."""
    logger = get_run_logger()
    df = pd.read_csv(SILVER_DIR / "ratings.csv")

    buffer = io.StringIO()
    df[["user_id", "film_id", "note", "timestamp"]].to_csv(
        buffer, index=False, header=False, sep="\t"
    )
    buffer.seek(0)

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE notation CASCADE")
            cur.copy_expert(
                "COPY notation (user_id, film_id, note, timestamp) FROM STDIN "
                "WITH (FORMAT csv, DELIMITER E'\\t')",
                buffer,
            )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"notation: {len(df)} lignes chargées")
    return len(df)


@task
def load_avis() -> int:
    """Charge silver/avis.csv dans la table avis via COPY (full refresh idempotent).

    Avis TMDB déjà rattachés aux utilisateurs par clean_avis (transform_silver).
    """
    logger = get_run_logger()
    avis_path = SILVER_DIR / "avis.csv"
    if not avis_path.exists():
        logger.warning("silver/avis.csv absent, chargement des avis ignoré.")
        return 0

    df = pd.read_csv(avis_path)

    buffer = io.StringIO()
    df[["user_id", "film_id", "texte", "timestamp"]].to_csv(
        buffer, index=False, header=False
    )
    buffer.seek(0)

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE avis CASCADE")
            cur.copy_expert(
                "COPY avis (user_id, film_id, texte, timestamp) "
                "FROM STDIN WITH (FORMAT csv)",
                buffer,
            )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"avis: {len(df)} lignes chargées")
    return len(df)


@task
def build_gold_tables(utilisateurs: int, films: int, notations: int, avis: int) -> None:
    """Log récapitulatif."""
    logger = get_run_logger()
    logger.info(
        f"Gold peuplé : {utilisateurs} utilisateurs, {films} films, "
        f"{notations} notations, {avis} avis. sentiment_score/prediction_note "
        "restent vides (à peupler par Personne B, modèles ML/NLP)."
    )


@flow(name="transform-gold")
def transform_gold() -> None:
    create_schema()
    utilisateurs = load_utilisateurs()
    films = load_films()
    notations = load_notations()
    avis = load_avis()
    build_gold_tables(utilisateurs, films, notations, avis)


if __name__ == "__main__":
    transform_gold()
