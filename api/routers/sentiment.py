from fastapi import APIRouter

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.get("/{film_id}")
def get_sentiment(film_id: int) -> dict:
    """Retourne le score de sentiment agrégé des avis d'un film. Seuil : F1 > 0.70, accuracy > 0.72."""
    raise NotImplementedError
