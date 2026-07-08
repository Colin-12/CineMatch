from fastapi import APIRouter

router = APIRouter(prefix="/comparaison", tags=["comparaison"])


@router.get("")
def compare_films(film_id_1: int, film_id_2: int) -> dict:
    """Compare deux films via LLM. Seuil : réponse <= 8s."""
    raise NotImplementedError
