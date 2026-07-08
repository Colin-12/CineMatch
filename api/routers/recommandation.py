from fastapi import APIRouter

router = APIRouter(prefix="/recommandation", tags=["recommandation"])


@router.get("/{user_id}")
def get_recommandations(user_id: int) -> dict:
    """Retourne au moins 5 suggestions justifiées par LLM. Seuil : réponse <= 5s."""
    raise NotImplementedError
