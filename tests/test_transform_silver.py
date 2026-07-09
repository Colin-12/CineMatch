"""Tests unitaires de `pipeline/transform_silver.py::clean_avis`.

Objets JSON reconstruits à la main (pas de dépendance à de vraies données
Bronze) pour valider le parsing de `author_details.rating`, en particulier
sa robustesse quand le champ est explicitement `null` ou totalement absent.
"""

import json

import pandas as pd
import pytest

from pipeline.transform_silver import clean_avis


@pytest.fixture
def ratings() -> pd.DataFrame:
    """Trois utilisateurs ayant réellement noté le film 1 (candidats FK)."""
    return pd.DataFrame(
        {
            "user_id": [10, 20, 30],
            "film_id": [1, 1, 1],
            "note": [4.0, 3.0, 5.0],
            "timestamp": [1000, 1001, 1002],
        }
    )


@pytest.fixture
def reviews_cache_dir(tmp_path, monkeypatch):
    """Écrit un cache TMDB reviews factice pour le film 1 et pointe dessus."""
    cache_dir = tmp_path / "tmdb_reviews_cache"
    cache_dir.mkdir()

    reviews = {
        "results": [
            # rating renseigné
            {
                "content": "Great movie, loved it.",
                "created_at": "2020-01-01T00:00:00.000Z",
                "author_details": {"rating": 8},
            },
            # rating explicitement null (avis laissé sans note par l'auteur)
            {
                "content": "Mixed feelings about this one.",
                "created_at": "2020-01-02T00:00:00.000Z",
                "author_details": {"rating": None},
            },
            # author_details totalement absent de l'objet review
            {
                "content": "No rating field at all here.",
                "created_at": "2020-01-03T00:00:00.000Z",
            },
        ]
    }
    (cache_dir / "1.json").write_text(json.dumps(reviews), encoding="utf-8")

    monkeypatch.setattr("pipeline.transform_silver.TMDB_REVIEWS_CACHE_DIR", cache_dir)
    return cache_dir


def test_clean_avis_captures_note_auteur_when_present(reviews_cache_dir, ratings):
    """`author_details.rating` rempli doit être capturé tel quel (float)."""
    df = clean_avis(ratings)
    row = df.loc[df["texte"] == "Great movie, loved it."].iloc[0]
    assert row["note_auteur"] == 8.0


def test_clean_avis_handles_null_rating_without_error(reviews_cache_dir, ratings):
    """`author_details.rating` explicitement `null` doit devenir `None`, pas planter."""
    df = clean_avis(ratings)
    row = df.loc[df["texte"] == "Mixed feelings about this one."].iloc[0]
    assert pd.isna(row["note_auteur"])


def test_clean_avis_handles_missing_author_details(reviews_cache_dir, ratings):
    """Absence totale de la clé `author_details` doit aussi donner `None`."""
    df = clean_avis(ratings)
    row = df.loc[df["texte"] == "No rating field at all here."].iloc[0]
    assert pd.isna(row["note_auteur"])


def test_clean_avis_keeps_synthetic_user_attachment_as_fallback(
    reviews_cache_dir, ratings
):
    """Le rattachement synthétique user_id (tirage parmi les votants réels) doit
    rester en place : chaque avis reste attribué à un des candidats FK, et
    aucun avis n'est perdu quand `note_auteur` est absent/null."""
    df = clean_avis(ratings)
    assert len(df) == 3
    assert set(df["user_id"]).issubset({10, 20, 30})
    assert (df["film_id"] == 1).all()
    # Une seule ligne avec note_auteur non nul sur les trois avis.
    assert df["note_auteur"].notna().sum() == 1
