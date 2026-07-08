from fastapi import APIRouter

router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.get("")
def predict_note(user_id: int, film_id: int) -> dict:
    """Prédit la note qu'un utilisateur donnerait à un film. Seuil : RMSE < 1.0."""
    raise NotImplementedError
