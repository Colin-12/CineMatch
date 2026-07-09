"""Endpoint /fiche : génère une fiche narrative de film via Gemini."""

from fastapi import APIRouter
from pydantic import BaseModel

from api.llm_common import generate_structured, resolve_film

router = APIRouter(prefix="/fiche", tags=["fiche"])

SYSTEM_INSTRUCTION_GROUNDED = (
    "Tu es un critique de cinéma qui rédige des fiches de films courtes et "
    "engageantes pour une application de recommandation, en français. Tu "
    "t'appuies uniquement sur les faits fournis (titre, année, genres, "
    "synopsis) — tu n'inventes aucun détail d'intrigue absent du synopsis. "
    "Si le synopsis est absent, reste général (ton, genre, époque) sans "
    "inventer d'intrigue. Réponds strictement au format JSON demandé."
)

SYSTEM_INSTRUCTION_UNGROUNDED = (
    "Tu es un critique de cinéma qui rédige des fiches de films courtes et "
    "engageantes, en français. Ce film n'est pas dans notre catalogue : "
    "réponds à partir de tes propres connaissances si tu reconnais ce "
    "titre. Si tu n'es pas sûr qu'un film porte exactement ce titre, "
    "dis-le explicitement plutôt que d'inventer une intrigue plausible. "
    "Réponds strictement au format JSON demandé."
)


class FicheNarrative(BaseModel):
    accroche: str
    resume: str
    ambiance: str
    pourquoi_regarder: str


def _build_grounded_prompt(film: dict) -> str:
    genres = ", ".join(film["genres"]) if film["genres"] else "non précisés"
    overview = film["overview"] or "Synopsis non disponible."
    return (
        f"Titre : {film['titre']}\n"
        f"Année : {film['annee']}\n"
        f"Genres : {genres}\n"
        f"Synopsis (TMDB) : {overview}\n\n"
        "Rédige la fiche à partir de ces informations."
    )


def _build_ungrounded_prompt(titre: str) -> str:
    return (
        f"Titre recherché : {titre}\n\n"
        "Ce film n'est pas dans notre catalogue. Rédige une fiche si tu "
        "reconnais ce titre, sinon indique-le dans le champ 'accroche' "
        "et laisse les autres champs vides."
    )


@router.get("")
def get_fiche(titre: str) -> dict:
    """Retourne la fiche narrative LLM d'un film recherché par titre.

    Seuil : réponse <= 5s. Si le film est en base (Gold), la fiche est
    ancrée sur nos données réelles ; sinon Gemini répond depuis ses
    propres connaissances (source="connaissances_llm").
    """
    film, grounded = resolve_film(titre)

    if grounded:
        fiche = generate_structured(
            SYSTEM_INSTRUCTION_GROUNDED, _build_grounded_prompt(film), FicheNarrative
        )
    else:
        fiche = generate_structured(
            SYSTEM_INSTRUCTION_UNGROUNDED,
            _build_ungrounded_prompt(titre),
            FicheNarrative,
        )

    return {
        "film_id": film["film_id"],
        "titre": film["titre"],
        "annee": film["annee"],
        "genres": film["genres"],
        "affiche_path": film["affiche_path"],
        "source": "catalogue" if grounded else "connaissances_llm",
        "fiche": fiche.model_dump() if fiche else None,
    }
