"""Endpoint /comparaison : compare deux films via Gemini."""

from fastapi import APIRouter
from pydantic import BaseModel

from api.llm_common import generate_structured, resolve_film

router = APIRouter(prefix="/comparaison", tags=["comparaison"])

SYSTEM_INSTRUCTION = (
    "Tu es un critique de cinéma qui compare deux films pour aider un "
    "spectateur à choisir, en français. Pour chaque film marqué "
    "'catalogue', appuie-toi uniquement sur les faits fournis (titre, "
    "année, genres, synopsis, note TMDB) sans inventer d'intrigue absente "
    "du synopsis. Pour un film marqué 'connaissances_llm' (absent de "
    "notre catalogue), réponds à partir de tes propres connaissances si "
    "tu reconnais le titre, et dis-le explicitement si tu n'es pas sûr. "
    "Réponds strictement au format JSON demandé."
)


class ComparaisonNarrative(BaseModel):
    points_communs: str
    differences_cles: str
    note_comparative: str


def _describe_film(label: str, film: dict, grounded: bool) -> str:
    if not grounded:
        return f"{label} : {film['titre']} (absent de notre catalogue)"

    genres = ", ".join(film["genres"]) if film["genres"] else "non précisés"
    overview = film["overview"] or "Synopsis non disponible."
    note = film["note_tmdb"] if film["note_tmdb"] is not None else "non disponible"
    return (
        f"{label} : {film['titre']} ({film['annee']})\n"
        f"Genres : {genres}\n"
        f"Note TMDB : {note}\n"
        f"Synopsis : {overview}"
    )


def _build_prompt(
    film_1: dict, grounded_1: bool, film_2: dict, grounded_2: bool
) -> str:
    return (
        f"{_describe_film('Film A', film_1, grounded_1)}\n\n"
        f"{_describe_film('Film B', film_2, grounded_2)}\n\n"
        "Compare ces deux films : points communs, différences clés, et une "
        "note comparative (qui l'emporte sur quels aspects, en t'appuyant "
        "sur les notes TMDB réelles quand elles sont disponibles)."
    )


@router.get("")
def compare_films(titre_1: str, titre_2: str) -> dict:
    """Compare deux films (recherchés par titre) via LLM. Seuil : réponse <= 8s."""
    film_1, grounded_1 = resolve_film(titre_1)
    film_2, grounded_2 = resolve_film(titre_2)

    comparaison = generate_structured(
        SYSTEM_INSTRUCTION,
        _build_prompt(film_1, grounded_1, film_2, grounded_2),
        ComparaisonNarrative,
        max_output_tokens=700,
    )

    def _meta(film: dict, grounded: bool) -> dict:
        return {
            "film_id": film["film_id"],
            "titre": film["titre"],
            "annee": film["annee"],
            "genres": film["genres"],
            "note_tmdb": film["note_tmdb"],
            "affiche_path": film["affiche_path"],
            "source": "catalogue" if grounded else "connaissances_llm",
        }

    return {
        "film_1": _meta(film_1, grounded_1),
        "film_2": _meta(film_2, grounded_2),
        "comparaison": comparaison.model_dump() if comparaison else None,
    }
