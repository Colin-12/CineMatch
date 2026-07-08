"""Transformation Gold : tables FILM, UTILISATEUR, NOTATION, AVIS prêtes à consommer."""

from prefect import flow, task


@task
def build_gold_tables() -> None:
    """Construit les tables Gold à partir de data/silver/, de façon idempotente."""
    raise NotImplementedError


@flow(name="transform-gold")
def transform_gold() -> None:
    build_gold_tables()


if __name__ == "__main__":
    transform_gold()
