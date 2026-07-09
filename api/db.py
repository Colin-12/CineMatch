"""Connexion à la base Gold (Supabase Postgres), partagée par les routers."""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def get_connection() -> psycopg2.extensions.connection:
    """Ouvre une connexion à la base Gold. À fermer par l'appelant."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def fetch_film(film_id: int) -> dict | None:
    """Récupère un film Gold par son id, ou None si absent."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT film_id, titre, annee, genres, overview, popularite, "
                "note_tmdb, affiche_path FROM film WHERE film_id = %s",
                (film_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def fetch_film_by_titre(titre: str) -> dict | None:
    """Cherche un film Gold par titre (exact insensible à la casse, sinon
    correspondance partielle la plus proche). None si aucun match."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT film_id, titre, annee, genres, overview, popularite, "
                "note_tmdb, affiche_path FROM film WHERE LOWER(titre) = LOWER(%s)",
                (titre,),
            )
            match = cur.fetchone()
            if match:
                return match

            cur.execute(
                "SELECT film_id, titre, annee, genres, overview, popularite, "
                "note_tmdb, affiche_path FROM film WHERE titre ILIKE %s "
                "ORDER BY LENGTH(titre) ASC LIMIT 1",
                (f"%{titre}%",),
            )
            return cur.fetchone()
    finally:
        conn.close()
