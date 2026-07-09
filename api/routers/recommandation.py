"""Endpoint /recommandation : suggère des films via Gemini (formulaire d'envies)."""

from fastapi import APIRouter
from pydantic import BaseModel

from api.llm_common import generate_structured

router = APIRouter(prefix="/recommandation", tags=["recommandation"])

SYSTEM_INSTRUCTION = (
    "Tu es un conseiller cinéma qui recommande des films réels (jamais "
    "inventés), en français, à partir de tes connaissances. Tu réponds "
    "aux envies exprimées par l'utilisateur (durée maximale, genre, note "
    "minimale). Tu dois proposer au moins 5 films distincts qui "
    "correspondent le mieux possible à ces critères, chacun avec une "
    "justification concrète liée aux critères demandés. Réponds "
    "strictement au format JSON demandé."
)


class Suggestion(BaseModel):
    titre: str
    annee: int | None = None
    duree_heures: float | None = None
    note_estimee: float | None = None
    justification: str


class Recommandations(BaseModel):
    suggestions: list[Suggestion]


def _build_prompt(
    duree_max_heures: float, genre: str, note_min: float, min_suggestions: int
) -> str:
    return (
        f"Durée souhaitée : {duree_max_heures}h maximum\n"
        f"Genre recherché : {genre}\n"
        f"Note minimale souhaitée : {note_min}/10\n\n"
        f"Propose au moins {min_suggestions} films réels qui correspondent "
        "le mieux possible à ces critères."
    )


@router.get("")
def get_recommandations(duree_max_heures: float, genre: str, note_min: float) -> dict:
    """Retourne au moins 5 suggestions de films via LLM selon un formulaire
    d'envies (durée max, genre, note minimale). Seuil : réponse <= 5s.

    Basé uniquement sur les connaissances de Gemini (pas de filtrage sur
    notre catalogue) : les films proposés ne sont pas garantis présents
    dans data Gold.
    """
    min_suggestions = 5
    prompt = _build_prompt(duree_max_heures, genre, note_min, min_suggestions)

    result = generate_structured(
        SYSTEM_INSTRUCTION, prompt, Recommandations, max_output_tokens=1500
    )
    suggestions = result.suggestions if result else []

    if len(suggestions) < min_suggestions:
        retry_prompt = (
            prompt + f"\n\n(Réponse précédente incomplète : {len(suggestions)} "
            f"films seulement, il en faut au moins {min_suggestions}.)"
        )
        result = generate_structured(
            SYSTEM_INSTRUCTION, retry_prompt, Recommandations, max_output_tokens=1500
        )
        if result and len(result.suggestions) > len(suggestions):
            suggestions = result.suggestions

    return {
        "criteres": {
            "duree_max_heures": duree_max_heures,
            "genre": genre,
            "note_min": note_min,
        },
        "source": "connaissances_llm",
        "suggestions": [s.model_dump() for s in suggestions],
    }
