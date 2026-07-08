from fastapi import APIRouter

router = APIRouter(prefix="/fiche", tags=["fiche"])


@router.get("/{film_id}")
def get_fiche(film_id: int) -> dict:
    """Retourne la fiche narrative LLM d'un film. Seuil : réponse <= 5s."""
    raise NotImplementedError
