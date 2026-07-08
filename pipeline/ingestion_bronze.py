"""Ingestion Bronze : MovieLens 1M (local) + API TMDB (cache, rate-limit)."""

from prefect import flow, task


@task
def ingest_movielens() -> None:
    """Charge les fichiers MovieLens 1M dans data/bronze/."""
    raise NotImplementedError


@task
def ingest_tmdb() -> None:
    """Interroge l'API TMDB avec cache local et throttling (0.3s entre appels)."""
    raise NotImplementedError


@flow(name="ingestion-bronze")
def ingestion_bronze() -> None:
    ingest_movielens()
    ingest_tmdb()


if __name__ == "__main__":
    ingestion_bronze()
