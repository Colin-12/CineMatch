"""Transformation Silver : nettoyage, typage, dédoublonnage."""

from prefect import flow, task


@task
def clean_and_dedupe() -> None:
    """Nettoie et déduplique les données de data/bronze/ vers data/silver/."""
    raise NotImplementedError


@flow(name="transform-silver")
def transform_silver() -> None:
    clean_and_dedupe()


if __name__ == "__main__":
    transform_silver()
