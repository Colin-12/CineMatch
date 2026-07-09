"""Transformation Silver : nettoyage, typage, dédoublonnage."""

import json
import random
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIELENS_DIR = DATA_DIR / "bronze" / "movielens"
TMDB_CACHE_DIR = DATA_DIR / "bronze" / "tmdb_cache"
TMDB_REVIEWS_CACHE_DIR = DATA_DIR / "bronze" / "tmdb_reviews_cache"
SILVER_DIR = DATA_DIR / "silver"


def _split_title_year(title: str) -> tuple[str, int | None]:
    """Sépare 'Toy Story (1995)' en ('Toy Story', 1995)."""
    if title.endswith(")") and "(" in title[-6:]:
        titre, annee_str = title.rsplit("(", 1)
        annee_str = annee_str.rstrip(")")
        return titre.strip(), int(annee_str) if annee_str.isdigit() else None
    return title, None


@task
def clean_movies() -> pd.DataFrame:
    """Charge movies.dat, type/dédoublonne par movie_id, enrichit avec le cache TMDB."""
    logger = get_run_logger()
    df = pd.read_csv(
        MOVIELENS_DIR / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["film_id", "titre_brut", "genres"],
    )
    df[["titre", "annee"]] = df["titre_brut"].apply(
        lambda t: pd.Series(_split_title_year(t))
    )
    df["film_id"] = df["film_id"].astype(int)
    df["annee"] = df["annee"].astype("Int64")
    df["genres"] = df["genres"].astype(str)
    df = df.drop_duplicates(subset="film_id", keep="last").drop(columns="titre_brut")

    tmdb_rows = []
    for cache_file in TMDB_CACHE_DIR.glob("*.json"):
        film_id = int(cache_file.stem)
        payload = json.loads(cache_file.read_text())
        results = payload.get("results") or []
        if not results:
            continue
        best = results[0]
        tmdb_rows.append(
            {
                "film_id": film_id,
                "tmdb_id": best.get("id"),
                "overview": best.get("overview"),
                "popularite": best.get("popularity"),
                "note_tmdb": best.get("vote_average"),
                "affiche_path": best.get("poster_path"),
            }
        )
    if tmdb_rows:
        df = df.merge(
            pd.DataFrame(tmdb_rows).drop_duplicates(subset="film_id"),
            on="film_id",
            how="left",
        )

    logger.info(f"clean_movies: {len(df)} films (dont {len(tmdb_rows)} enrichis TMDB)")
    return df


@task
def clean_users() -> pd.DataFrame:
    """Charge users.dat, type et dédoublonne par user_id."""
    logger = get_run_logger()
    df = pd.read_csv(
        MOVIELENS_DIR / "users.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["user_id", "genre", "age", "occupation", "code_postal"],
    )
    df["user_id"] = df["user_id"].astype(int)
    df["age"] = df["age"].astype(int)
    df["occupation"] = df["occupation"].astype(int)
    df["genre"] = df["genre"].astype(str)
    df["code_postal"] = df["code_postal"].astype(str)
    df = df.drop_duplicates(subset="user_id", keep="last")

    logger.info(f"clean_users: {len(df)} utilisateurs")
    return df


@task
def clean_ratings() -> pd.DataFrame:
    """Charge ratings.dat, type et dédoublonne par (user_id, film_id).

    Garde le timestamp le plus récent en cas de doublon.
    """
    logger = get_run_logger()
    df = pd.read_csv(
        MOVIELENS_DIR / "ratings.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["user_id", "film_id", "note", "timestamp"],
    )
    df["user_id"] = df["user_id"].astype(int)
    df["film_id"] = df["film_id"].astype(int)
    df["note"] = df["note"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "film_id"], keep="last"
    )

    logger.info(f"clean_ratings: {len(df)} notations")
    return df


@task
def clean_avis(ratings: pd.DataFrame) -> pd.DataFrame:
    """Construit les avis à partir du cache d'avis TMDB.

    Chaque avis est rattaché à un utilisateur MovieLens ayant réellement noté
    le film (tirage aléatoire sans remise parmi les vrais votants, pour
    respecter les FK). Conserve la vraie note de l'auteur de l'avis TMDB
    (`author_details.rating`, 0-10, ~95% de couverture) dans `note_auteur` —
    utile comme label réel pour l'entraînement du modèle de sentiment
    (à la différence du rattachement user_id, qui reste synthétique).
    """
    logger = get_run_logger()
    random.seed(42)
    raters_by_film = ratings.groupby("film_id")["user_id"].apply(list).to_dict()

    rows = []
    for cache_file in TMDB_REVIEWS_CACHE_DIR.glob("*.json"):
        film_id = int(cache_file.stem)
        reviews = json.loads(cache_file.read_text()).get("results") or []
        candidates = raters_by_film.get(film_id, [])
        if not reviews or not candidates:
            continue

        sample_size = min(len(reviews), len(candidates))
        sampled_users = random.sample(candidates, sample_size)
        for review, user_id in zip(reviews[:sample_size], sampled_users):
            texte = (review.get("content") or "").strip()
            if not texte:
                continue
            created_at = review.get("created_at")
            timestamp = int(pd.Timestamp(created_at).timestamp()) if created_at else 0
            note_auteur = (review.get("author_details") or {}).get("rating")
            rows.append(
                {
                    "user_id": user_id,
                    "film_id": film_id,
                    "texte": texte,
                    "timestamp": timestamp,
                    "note_auteur": note_auteur,
                }
            )

    df = pd.DataFrame(
        rows, columns=["user_id", "film_id", "texte", "timestamp", "note_auteur"]
    )
    df = df.drop_duplicates(subset=["user_id", "film_id"], keep="first")

    avec_note = df["note_auteur"].notna().sum()
    logger.info(
        f"clean_avis: {len(df)} avis construits (TMDB, rattachement "
        f"synthétique aux utilisateurs), {avec_note} avec note_auteur réelle"
    )
    return df


@task
def write_silver(
    movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame, avis: pd.DataFrame
) -> None:
    """Écrit les DataFrames nettoyés dans data/silver/ (CSV, écrasé à chaque run)."""
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(SILVER_DIR / "movies.csv", index=False)
    users.to_csv(SILVER_DIR / "users.csv", index=False)
    ratings.to_csv(SILVER_DIR / "ratings.csv", index=False)
    avis.to_csv(SILVER_DIR / "avis.csv", index=False)


@flow(name="transform-silver")
def transform_silver() -> None:
    movies = clean_movies()
    users = clean_users()
    ratings = clean_ratings()
    avis = clean_avis(ratings)
    write_silver(movies, users, ratings, avis)


if __name__ == "__main__":
    transform_silver()
