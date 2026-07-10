"""Endpoint /film : recherche d'un film par titre dans le catalogue Gold (sans LLM)."""

from fastapi import APIRouter, HTTPException

from api.db import fetch_film_by_titre

router = APIRouter(prefix="/film", tags=["film"])


@router.get("")
def get_film(titre: str) -> dict:
    """Cherche un film par titre dans le catalogue Gold. 404 si absent.

    Recherche exacte insensible à la casse, sinon correspondance partielle
    la plus proche (voir `api.db.fetch_film_by_titre`).
    """
    film = fetch_film_by_titre(titre)
    if film is None:
        raise HTTPException(
            status_code=404, detail="Film introuvable dans le catalogue"
        )
    return film
