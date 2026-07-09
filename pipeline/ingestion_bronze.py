"""Ingestion Bronze : MovieLens 1M (local) + API TMDB (cache, rate-limit)."""

import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

load_dotenv()

BRONZE_DIR = Path(__file__).resolve().parent.parent / "data" / "bronze"
MOVIELENS_DIR = BRONZE_DIR / "movielens"
TMDB_CACHE_DIR = BRONZE_DIR / "tmdb_cache"
TMDB_REVIEWS_CACHE_DIR = BRONZE_DIR / "tmdb_reviews_cache"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_FILES = ["movies.dat", "ratings.dat", "users.dat"]

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_MAX_WORKERS = 10


def _get_with_backoff(session: requests.Session, url: str, params: dict) -> requests.Response | None:
    """GET avec retry/backoff sur 429 (rate limit TMDB)."""
    for _ in range(4):
        response = session.get(url, params=params, timeout=10)
        if response.status_code != 429:
            return response
        time.sleep(float(response.headers.get("Retry-After", 1)))
    return response


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


def _fetch_movie_search(session: requests.Session, api_key: str, movie_id: int, titre: str, annee: int) -> str | None:
    params = {"api_key": api_key, "query": titre}
    if annee:
        params["year"] = annee

    response = _get_with_backoff(session, f"{TMDB_BASE_URL}/search/movie", params)
    if response is None or response.status_code != 200:
        return f"'{titre}' ({movie_id}): {response.status_code if response else 'timeout'}"

    _tmdb_cache_path(movie_id).write_text(json.dumps(response.json(), ensure_ascii=False))
    return None


@task
def ingest_tmdb(limit: int | None = None) -> Path:
    """Enrichit chaque film MovieLens via l'API TMDB (recherche titre/année), cache local, requêtes parallélisées."""
    logger = get_run_logger()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.warning("TMDB_API_KEY absente de l'environnement, ingestion TMDB ignorée.")
        return TMDB_CACHE_DIR

    TMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    movies = _parse_movielens_titles()
    if limit is not None:
        movies = movies[:limit]
    movies = [m for m in movies if not _tmdb_cache_path(m[0]).exists()]

    session = requests.Session()
    errors = 0
    with ThreadPoolExecutor(max_workers=TMDB_MAX_WORKERS) as executor:
        futures = [executor.submit(_fetch_movie_search, session, api_key, mid, titre, annee) for mid, titre, annee in movies]
        for future in as_completed(futures):
            error = future.result()
            if error:
                errors += 1
                logger.warning(f"TMDB: échec requête pour {error}")

    logger.info(f"Cache TMDB peuplé dans {TMDB_CACHE_DIR} ({len(movies)} films interrogés, {errors} échecs)")
    return TMDB_CACHE_DIR


def _tmdb_reviews_cache_path(movie_id: int) -> Path:
    return TMDB_REVIEWS_CACHE_DIR / f"{movie_id}.json"


def _fetch_movie_reviews(session: requests.Session, api_key: str, movie_id: int, tmdb_id: int) -> str | None:
    response = _get_with_backoff(session, f"{TMDB_BASE_URL}/movie/{tmdb_id}/reviews", {"api_key": api_key})
    if response is None or response.status_code != 200:
        return f"movie_id={movie_id} (tmdb_id={tmdb_id}): {response.status_code if response else 'timeout'}"

    _tmdb_reviews_cache_path(movie_id).write_text(json.dumps(response.json(), ensure_ascii=False))
    return None


@task
def ingest_tmdb_reviews(limit: int | None = None) -> Path:
    """Récupère les avis TMDB (/movie/{tmdb_id}/reviews) pour les films déjà identifiés via ingest_tmdb, cache local, requêtes parallélisées."""
    logger = get_run_logger()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.warning("TMDB_API_KEY absente de l'environnement, ingestion des avis TMDB ignorée.")
        return TMDB_REVIEWS_CACHE_DIR

    TMDB_REVIEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_files = sorted(TMDB_CACHE_DIR.glob("*.json"))
    if limit is not None:
        cache_files = cache_files[:limit]

    targets = []
    for cache_file in cache_files:
        movie_id = int(cache_file.stem)
        if _tmdb_reviews_cache_path(movie_id).exists():
            continue
        results = json.loads(cache_file.read_text()).get("results") or []
        if not results or not results[0].get("id"):
            continue
        targets.append((movie_id, results[0]["id"]))

    session = requests.Session()
    errors = 0
    with ThreadPoolExecutor(max_workers=TMDB_MAX_WORKERS) as executor:
        futures = [executor.submit(_fetch_movie_reviews, session, api_key, mid, tmdb_id) for mid, tmdb_id in targets]
        for future in as_completed(futures):
            error = future.result()
            if error:
                errors += 1
                logger.warning(f"TMDB reviews: échec requête pour {error}")

    logger.info(f"Cache avis TMDB peuplé dans {TMDB_REVIEWS_CACHE_DIR} ({len(targets)} films interrogés, {errors} échecs)")
    return TMDB_REVIEWS_CACHE_DIR


@flow(name="ingestion-bronze")
def ingestion_bronze(tmdb_limit: int | None = None) -> None:
    ingest_movielens()
    ingest_tmdb(limit=tmdb_limit)
    ingest_tmdb_reviews(limit=tmdb_limit)


if __name__ == "__main__":
    ingestion_bronze()
