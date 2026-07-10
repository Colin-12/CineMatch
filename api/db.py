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


def fetch_admin_stats() -> dict:
    """Statistiques Gold pour la vue Admin : compteurs, couverture, fraîcheur."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM film")
            (n_films,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM utilisateur")
            (n_utilisateurs,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM notation")
            (n_notations,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM avis")
            (n_avis,) = cur.fetchone()

            cur.execute("SELECT COUNT(*) FROM film WHERE overview IS NOT NULL")
            (n_films_enrichis,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM avis WHERE note_auteur IS NOT NULL")
            (n_avis_notes,) = cur.fetchone()
            cur.execute("SELECT COUNT(DISTINCT film_id) FROM avis")
            (n_films_avec_avis,) = cur.fetchone()

            cur.execute("SELECT MAX(timestamp) FROM notation")
            (derniere_notation,) = cur.fetchone()
            cur.execute("SELECT MAX(timestamp) FROM avis")
            (dernier_avis,) = cur.fetchone()
    finally:
        conn.close()

    def _pct(part: int, total: int) -> float:
        return round(100 * part / total, 1) if total else 0.0

    return {
        "counts": {
            "film": n_films,
            "utilisateur": n_utilisateurs,
            "notation": n_notations,
            "avis": n_avis,
        },
        "coverage": {
            "films_enrichis_tmdb_pct": _pct(n_films_enrichis, n_films),
            "avis_avec_note_auteur_pct": _pct(n_avis_notes, n_avis),
            "films_avec_avis_pct": _pct(n_films_avec_avis, n_films),
        },
        "fraicheur": {
            "derniere_notation_timestamp": derniere_notation,
            "dernier_avis_timestamp": dernier_avis,
        },
    }
