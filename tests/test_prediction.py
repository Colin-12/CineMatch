"""Tests de l'endpoint `/prediction` (features live depuis Gold).

Ce sont des tests d'integration : l'endpoint recalcule ses features a la
volee depuis la base Gold (voir `api/routers/prediction.py`), une connexion
`DATABASE_URL` vivante est donc necessaire par construction. Si la base
n'est pas accessible (ex. environnement CI sans Postgres provisionne, cf.
`.github/workflows/ci.yml` qui ne demarre pas de service Postgres), les
tests sont proprement `skip` plutot que de faire echouer la pipeline.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app

load_dotenv()

client = TestClient(app)


def _fetch_known_pair() -> tuple[int, int] | None:
    """Recupere un couple (user_id, film_id) reel depuis `notation`, ou None
    si la base Gold n'est pas accessible."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    try:
        conn = psycopg2.connect(database_url)
    except psycopg2.OperationalError:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, film_id FROM notation LIMIT 1;")
            row = cur.fetchone()
    finally:
        conn.close()
    return (row[0], row[1]) if row else None


@pytest.fixture(scope="module")
def known_pair() -> tuple[int, int]:
    pair = _fetch_known_pair()
    if pair is None:
        pytest.skip("Base Gold indisponible (DATABASE_URL absent ou injoignable).")
    return pair


def test_predict_note_known_pair_returns_coherent_response(
    known_pair: tuple[int, int],
) -> None:
    """Reponse coherente (schema PredictionNote, note dans [1, 5]) pour un
    couple user_id/film_id reel issu de Gold."""
    user_id, film_id = known_pair
    response = client.get(
        "/prediction", params={"user_id": user_id, "film_id": film_id}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == user_id
    assert payload["film_id"] == film_id
    assert 1.0 <= payload["note_predite"] <= 5.0


def test_predict_note_unknown_film_returns_404() -> None:
    """Un film_id inexistant dans Gold doit renvoyer 404, pas une prediction."""
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL absent.")
    try:
        response = client.get(
            "/prediction", params={"user_id": 1, "film_id": 999_999_999}
        )
    except psycopg2.OperationalError:
        pytest.skip("Base Gold indisponible.")
    assert response.status_code == 404


def test_predict_note_cold_start_user_falls_back_gracefully(
    known_pair: tuple[int, int],
) -> None:
    """Un user_id jamais vu (cold start) doit quand meme renvoyer une note
    valide (fallback moyenne globale cote utilisateur), pas une erreur."""
    _, film_id = known_pair
    unseen_user_id = 999_999_999
    response = client.get(
        "/prediction", params={"user_id": unseen_user_id, "film_id": film_id}
    )
    assert response.status_code == 200
    payload = response.json()
    assert 1.0 <= payload["note_predite"] <= 5.0
