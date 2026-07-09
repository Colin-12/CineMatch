"""Endpoint `/sentiment` : score de sentiment agrege des avis TMDB d'un film.

**Modele charge actuellement : TF-IDF + regression logistique** (bundle
`nlp/models/tfidf_logreg_sentiment_v1.1_<date>.joblib`), entraine sur le
label derive de `avis.note_auteur` (voir `nlp/training/features.py` et
`nlp/training/model.py`). C'est un **choix de reference temporaire** : le
fine-tuning DistilBERT complet (`python -m nlp.training.model`) tourne en
parallele (~60-90+ min CPU) et n'est pas encore disponible/valide au moment
ou cet endpoint a ete implemente. Une fois DistilBERT entraine et confirme
superieur aux seuils cibles (F1 macro > 0.70, accuracy > 0.72 -- voir
`CLAUDE.md`), `get_bundle()` ci-dessous sera bascule sur le bundle
`distilbert_sentiment_v*` (meme fonction `load_model`, seul le chemin/type
de modele change) -- voir `nlp/model_cards/` pour la comparaison chiffree
des deux approches au moment du switch.

Le score renvoye (`SentimentScore.score`, dans [0, 1]) est une moyenne, sur
tous les avis textuels du film, de l'esperance ponderee par les probabilites
de classe de chaque avis (`features.expected_sentiment_score` : negatif ->
0.0, neutre -> 0.5, positif -> 1.0). Le label agrege
(`features.derive_label_from_score`) reutilise des seuils proportionnels a
ceux de l'entrainement (`derive_label_from_note_auteur`), pour rester
coherent d'une echelle a l'autre.
"""

import glob
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from api.schemas.gold import SentimentScore
from nlp.training import features
from nlp.training.model import MODELS_DIR, SentimentModelBundle, load_model

load_dotenv()

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@lru_cache(maxsize=1)
def get_bundle() -> SentimentModelBundle:
    """Charge le bundle TF-IDF+LogReg le plus recent, une seule fois par
    processus (poids figes). Voir docstring module pour le plan de bascule
    vers DistilBERT."""
    candidates = sorted(glob.glob(str(MODELS_DIR / "tfidf_logreg_sentiment_v*.joblib")))
    if not candidates:
        raise FileNotFoundError(
            "Aucun modele tfidf_logreg sauvegarde trouve dans "
            f"{MODELS_DIR} -- executez d'abord "
            "`python -m nlp.training.model --tfidf-only`."
        )
    model_path = Path(candidates[-1])
    bundle = load_model(model_path)
    if bundle.model_type != "tfidf_logreg":
        raise ValueError(
            f"Modele attendu 'tfidf_logreg', trouve '{bundle.model_type}' "
            f"dans {model_path}"
        )
    return bundle


def get_connection() -> psycopg2.extensions.connection:
    """Ouvre une connexion a la base Gold (`DATABASE_URL`)."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def fetch_film_exists(conn: psycopg2.extensions.connection, film_id: int) -> bool:
    """Verifie l'existence du film (404 sinon)."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM film WHERE film_id = %s;", (film_id,))
        return cur.fetchone() is not None


def fetch_avis_textes(conn: psycopg2.extensions.connection, film_id: int) -> list[str]:
    """Recupere les textes d'avis TMDB disponibles pour ce film."""
    with conn.cursor() as cur:
        cur.execute("SELECT texte FROM avis WHERE film_id = %s;", (film_id,))
        rows = cur.fetchall()
    return [row[0] for row in rows if row[0]]


def score_film_sentiment(
    bundle: SentimentModelBundle, textes: list[str]
) -> tuple[float, str]:
    """Agrege le score/label de sentiment sur l'ensemble des avis d'un film.

    Un seul appel `predict_proba` sur tous les textes (plutot qu'un par
    avis) : plus rapide, et le pipeline TF-IDF+LogReg le supporte nativement.
    """
    proba_matrix = bundle.sklearn_model.predict_proba(textes)
    labels_order = list(bundle.sklearn_model.classes_)
    per_avis_scores = [
        features.expected_sentiment_score(proba, labels=labels_order)
        for proba in proba_matrix
    ]
    score = float(np.mean(per_avis_scores))
    label = features.derive_label_from_score(score)
    return score, label


@router.get("/{film_id}", response_model=SentimentScore)
def get_sentiment(film_id: int) -> SentimentScore:
    """Retourne le score de sentiment agrege des avis d'un film.

    Seuil : F1 > 0.70, accuracy > 0.72 (evalues hors ligne, voir model card).
    404 si le film est inconnu de Gold, ou si aucun avis textuel n'est
    disponible pour lui (rien a scorer).
    """
    bundle = get_bundle()
    conn = get_connection()
    try:
        if not fetch_film_exists(conn, film_id):
            raise HTTPException(
                status_code=404, detail=f"film_id {film_id} introuvable"
            )
        textes = fetch_avis_textes(conn, film_id)
    finally:
        conn.close()

    if not textes:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun avis textuel disponible pour film_id {film_id}",
        )

    score, label = score_film_sentiment(bundle, textes)
    return SentimentScore(film_id=film_id, score=score, label=label)
