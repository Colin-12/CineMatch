"""Tests de l'endpoint `/sentiment` (score agrege depuis les avis Gold).

Comme `tests/test_prediction.py`, ce sont des tests d'integration : l'endpoint
recupere les avis textuels a la volee depuis la base Gold (voir
`api/routers/sentiment.py`), une connexion `DATABASE_URL` vivante est donc
necessaire. Si la base n'est pas accessible (ex. CI sans Postgres
provisionne), les tests sont proprement `skip` plutot que d'echouer.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app

load_dotenv()

client = TestClient(app)


def _fetch_film_with_avis() -> int | None:
    """Recupere un `film_id` reel ayant au moins un avis textuel, ou None si
    la base Gold n'est pas accessible."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    try:
        conn = psycopg2.connect(database_url)
    except psycopg2.OperationalError:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT film_id FROM avis WHERE texte IS NOT NULL "
                "AND texte != '' LIMIT 1;"
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return row[0] if row else None


@pytest.fixture(scope="module")
def film_with_avis() -> int:
    film_id = _fetch_film_with_avis()
    if film_id is None:
        pytest.skip("Base Gold indisponible (DATABASE_URL absent ou injoignable).")
    return film_id


def test_get_sentiment_known_film_returns_coherent_response(
    film_with_avis: int,
) -> None:
    """Reponse coherente (schema SentimentScore, score dans [0, 1], label
    valide) pour un film reel possedant des avis textuels."""
    response = client.get(f"/sentiment/{film_with_avis}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["film_id"] == film_with_avis
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["label"] in {"negatif", "neutre", "positif"}


def test_get_sentiment_unknown_film_returns_404() -> None:
    """Un film_id inexistant dans Gold doit renvoyer 404, pas un score."""
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL absent.")
    try:
        response = client.get("/sentiment/999999999")
    except psycopg2.OperationalError:
        pytest.skip("Base Gold indisponible.")
    assert response.status_code == 404
