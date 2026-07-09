"""Ingestion Bronze : MovieLens 1M (local) + API TMDB (cache, rate-limit)."""

import json
import os
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

load_dotenv()

BRONZE_DIR = Path(__file__).resolve().parent.parent / "data" / "bronze"
MOVIELENS_DIR = BRONZE_DIR / "movielens"
TMDB_CACHE_DIR = BRONZE_DIR / "tmdb_cache"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_FILES = ["movies.dat", "ratings.dat", "users.dat"]

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_RATE_LIMIT_SECONDS = 0.3


@task
def ingest_movielens() -> Path:
    """Télécharge (si absent) et extrait users/movies/ratings dans data/bronze/movielens/."""
    logger = get_run_logger()
    MOVIELENS_DIR.mkdir(parents=True, exist_ok=True)

    if all((MOVIELENS_DIR / name).exists() for name in MOVIELENS_FILES):
        logger.info("MovieLens 1M déjà présent, ingestion ignorée (idempotent).")
        return MOVIELENS_DIR

    logger.info(f"Téléchargement de MovieLens 1M depuis {MOVIELENS_URL}")
    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()

    zip_path = BRONZE_DIR / "ml-1m.zip"
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path) as archive:
        for name in MOVIELENS_FILES:
            with archive.open(f"ml-1m/{name}") as src, open(MOVIELENS_DIR / name, "wb") as dst:
                dst.write(src.read())

    zip_path.unlink()
    logger.info(f"MovieLens 1M extrait dans {MOVIELENS_DIR}")
    return MOVIELENS_DIR


def _parse_movielens_titles() -> list[tuple[int, str, int]]:
    """Lit movies.dat et retourne une liste de (movie_id, titre, annee)."""
    movies_path = MOVIELENS_DIR / "movies.dat"
    results = []
    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            movie_id, title, _genres = line.strip().split("::")
            if title.endswith(")") and "(" in title[-6:]:
                titre, annee_str = title.rsplit("(", 1)
                titre = titre.strip()
                annee_str = annee_str.rstrip(")")
                annee = int(annee_str) if annee_str.isdigit() else 0
            else:
                titre, annee = title, 0
            results.append((int(movie_id), titre, annee))
    return results


def _tmdb_cache_path(movie_id: int) -> Path:
    return TMDB_CACHE_DIR / f"{movie_id}.json"


@task
def ingest_tmdb(limit: int | None = None) -> Path:
    """Enrichit chaque film MovieLens via l'API TMDB (recherche titre/année), cache local + throttling."""
    logger = get_run_logger()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.warning("TMDB_API_KEY absente de l'environnement, ingestion TMDB ignorée.")
        return TMDB_CACHE_DIR

    TMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    movies = _parse_movielens_titles()
    if limit is not None:
        movies = movies[:limit]

    session = requests.Session()
    for movie_id, titre, annee in movies:
        cache_path = _tmdb_cache_path(movie_id)
        if cache_path.exists():
            continue

        params = {"api_key": api_key, "query": titre}
        if annee:
            params["year"] = annee

        response = session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=10)
        time.sleep(TMDB_RATE_LIMIT_SECONDS)

        if response.status_code != 200:
            logger.warning(f"TMDB: échec requête pour '{titre}' ({movie_id}): {response.status_code}")
            continue

        cache_path.write_text(json.dumps(response.json(), ensure_ascii=False))

    logger.info(f"Cache TMDB peuplé dans {TMDB_CACHE_DIR}")
    return TMDB_CACHE_DIR


@flow(name="ingestion-bronze")
def ingestion_bronze(tmdb_limit: int | None = None) -> None:
    ingest_movielens()
    ingest_tmdb(limit=tmdb_limit)


if __name__ == "__main__":
    ingestion_bronze()
