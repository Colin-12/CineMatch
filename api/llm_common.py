"""Utilitaires partagés par les routers LLM (fiche, comparaison, recommandation)."""

import os
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from api.db import fetch_film_by_titre

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def generate_structured(
    system_instruction: str,
    prompt: str,
    schema: type[SchemaT],
    max_output_tokens: int = 500,
) -> SchemaT | None:
    """Appelle Gemini en sortie JSON structurée.

    Thinking désactivé pour la latence. Un 2e essai est fait si la 1ère
    réponse ne respecte pas le schéma demandé.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
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


def resolve_film(titre: str) -> tuple[dict, bool]:
    """Cherche un film par titre en Gold.

    Retourne (meta, grounded) : grounded=True si trouvé en base (données
    réelles), False sinon (meta ne contient alors que le titre recherché).
    """
    film = fetch_film_by_titre(titre)
    if film is not None:
        return (
            {
                "film_id": film["film_id"],
                "titre": film["titre"],
                "annee": film["annee"],
                "genres": film["genres"],
                "overview": film["overview"],
                "note_tmdb": film["note_tmdb"],
                "affiche_path": film["affiche_path"],
            },
            True,
        )
    return (
        {
            "film_id": None,
            "titre": titre,
            "annee": None,
            "genres": [],
            "overview": None,
            "note_tmdb": None,
            "affiche_path": None,
        },
        False,
    )
