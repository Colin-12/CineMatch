"""Script jetable d'EDA sur la base Gold : sparsité, distribution des notes,
couverture des colonnes TMDB. A lancer ponctuellement, pas de graphiques.

Usage : python ml/training/eda_exploration.py
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

MIN_NOTATIONS = 5


def _connect() -> psycopg2.extensions.connection:
    """Ouvre une connexion à la base Gold via DATABASE_URL."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def print_section(title: str) -> None:
    """Affiche un séparateur de section lisible dans la sortie texte."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def explore_notation_counts(cur) -> None:
    """Affiche le nombre de notations et le nombre d'utilisateurs/films distincts."""
    print_section("1. Volumetrie de NOTATION")
    cur.execute("SELECT COUNT(*) FROM notation;")
    n_notations = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM notation;")
    n_users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT film_id) FROM notation;")
    n_films = cur.fetchone()[0]
    print(f"Nombre de notations       : {n_notations}")
    print(f"Utilisateurs distincts     : {n_users}")
    print(f"Films distincts (notes)    : {n_films}")


def explore_rating_distribution(cur) -> None:
    """Affiche la distribution des notes (comptage par valeur + stats de base)."""
    print_section("2. Distribution des notes")
    cur.execute("""
        SELECT note, COUNT(*) AS n
        FROM notation
        GROUP BY note
        ORDER BY note;
        """)
    rows = cur.fetchall()
    total = sum(n for _, n in rows)
    for note, n in rows:
        pct = 100 * n / total if total else 0
        print(f"  note={note:<5} n={n:<8} ({pct:5.1f}%)")

    cur.execute("""
        SELECT
            AVG(note)::numeric(10,3),
            STDDEV(note)::numeric(10,3),
            MIN(note),
            MAX(note)
        FROM notation;
        """)
    avg, std, mn, mx = cur.fetchone()
    print(f"\nMoyenne={avg}  Ecart-type={std}  Min={mn}  Max={mx}")


def explore_sparsity(cur) -> None:
    """Calcule la sparsite user x film (1 - notations / (users * films))."""
    print_section("3. Sparsite de la matrice user x film")
    cur.execute("SELECT COUNT(*) FROM notation;")
    n_notations = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM notation;")
    n_users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT film_id) FROM notation;")
    n_films = cur.fetchone()[0]

    taille_matrice = n_users * n_films
    densite = n_notations / taille_matrice if taille_matrice else 0
    sparsite = 1 - densite

    print(f"Taille matrice (users x films) : {n_users} x {n_films} = {taille_matrice}")
    print(f"Notations existantes            : {n_notations}")
    print(f"Densite                         : {densite:.6f} ({100 * densite:.4f}%)")
    print(f"Sparsite                        : {sparsite:.6f} ({100 * sparsite:.4f}%)")


def explore_low_activity(cur) -> None:
    """Compte les utilisateurs et films ayant moins de MIN_NOTATIONS notations."""
    print_section(f"4. Utilisateurs / films avec moins de {MIN_NOTATIONS} notations")

    cur.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT user_id FROM notation
            GROUP BY user_id
            HAVING COUNT(*) < %s
        ) sub;
        """,
        (MIN_NOTATIONS,),
    )
    n_users_low = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT film_id FROM notation
            GROUP BY film_id
            HAVING COUNT(*) < %s
        ) sub;
        """,
        (MIN_NOTATIONS,),
    )
    n_films_low = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT user_id) FROM notation;")
    n_users_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT film_id) FROM notation;")
    n_films_total = cur.fetchone()[0]

    pct_users = 100 * n_users_low / n_users_total if n_users_total else 0
    pct_films = 100 * n_films_low / n_films_total if n_films_total else 0

    print(
        f"Utilisateurs < {MIN_NOTATIONS} notations : {n_users_low} / {n_users_total} "
        f"({pct_users:.1f}%)"
    )
    print(
        f"Films < {MIN_NOTATIONS} notations         : {n_films_low} / {n_films_total} "
        f"({pct_films:.1f}%)"
    )


def explore_tmdb_coverage(cur) -> None:
    """Affiche le taux de couverture (non-NULL) des colonnes TMDB de la table film."""
    print_section("5. Couverture des colonnes TMDB (table film)")
    cur.execute("SELECT COUNT(*) FROM film;")
    n_films = cur.fetchone()[0]

    colonnes = ["overview", "popularite", "note_tmdb", "affiche_path"]
    for col in colonnes:
        cur.execute(f"SELECT COUNT({col}) FROM film WHERE {col} IS NOT NULL;")
        n_non_null = cur.fetchone()[0]
        pct = 100 * n_non_null / n_films if n_films else 0
        print(f"  {col:<15} : {n_non_null:>6} / {n_films:<6} non-NULL ({pct:5.1f}%)")

    print(f"\nTotal films : {n_films}")


def main() -> None:
    """Lance l'etat des lieux complet de la base Gold et affiche les resultats."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            explore_notation_counts(cur)
            explore_rating_distribution(cur)
            explore_sparsity(cur)
            explore_low_activity(cur)
            explore_tmdb_coverage(cur)
    finally:
        conn.close()

    print_section("Fin de l'etat des lieux")


if __name__ == "__main__":
    main()
