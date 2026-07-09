"""Endpoint /fiche : génère une fiche narrative de film via Gemini."""

import os

from dotenv import load_dotenv
from fastapi import APIRouter
from google import genai
from google.genai import types
from pydantic import BaseModel

from api.db import fetch_film_by_titre

load_dotenv()

router = APIRouter(prefix="/fiche", tags=["fiche"])

GEMINI_MODEL = "gemini-2.5-flash"

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


def _generate(system_instruction: str, prompt: str) -> FicheNarrative | None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=500,
        response_mime_type="application/json",
        response_schema=FicheNarrative,
    )

    for attempt_prompt in (
        prompt,
        prompt + "\n\n(Réponse précédente invalide, corrige le format JSON.)",
    ):
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=attempt_prompt, config=config
        )
        if response.parsed is not None:
            return response.parsed
    return None


@router.get("")
def get_fiche(titre: str) -> dict:
    """Retourne la fiche narrative LLM d'un film recherché par titre.

    Seuil : réponse <= 5s. Si le film est en base (Gold), la fiche est
    ancrée sur nos données réelles ; sinon Gemini répond depuis ses
    propres connaissances (source="connaissances_llm").
    """
    film = fetch_film_by_titre(titre)

    if film is not None:
        fiche = _generate(SYSTEM_INSTRUCTION_GROUNDED, _build_grounded_prompt(film))
        source = "catalogue"
        meta = {
            "film_id": film["film_id"],
            "titre": film["titre"],
            "annee": film["annee"],
            "genres": film["genres"],
            "affiche_path": film["affiche_path"],
        }
    else:
        fiche = _generate(
            SYSTEM_INSTRUCTION_UNGROUNDED, _build_ungrounded_prompt(titre)
        )
        source = "connaissances_llm"
        meta = {
            "film_id": None,
            "titre": titre,
            "annee": None,
            "genres": [],
            "affiche_path": None,
        }

    return {
        **meta,
        "source": source,
        "fiche": fiche.model_dump() if fiche else None,
    }
