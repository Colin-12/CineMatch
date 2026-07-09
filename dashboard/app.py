"""Dashboard Streamlit CineMatch (fiche, comparaison, recommandation, prédiction)."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 15

st.set_page_config(page_title="CineMatch", layout="wide")

PAGES = [
    "Fiche",
    "Comparaison",
    "Recommandation",
    "Prédiction de note",
    "Sentiment",
    "Admin",
]

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"


def _call_api(path: str, params: dict) -> dict | None:
    """Appelle l'API CineMatch, affiche une erreur Streamlit propre en cas d'échec."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Impossible de joindre l'API sur {API_BASE_URL}. "
            "Vérifie qu'elle tourne (`uvicorn api.main:app`)."
        )
    except requests.exceptions.Timeout:
        st.error("L'API a mis trop de temps à répondre.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur API ({e.response.status_code}) : {e.response.text}")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
    return None


def _source_badge(source: str) -> None:
    if source == "catalogue":
        st.caption("📚 Source : catalogue CineMatch (données vérifiées)")
    else:
        st.caption(
            "🤖 Source : connaissances Gemini (non vérifié dans notre catalogue)"
        )


def page_fiche() -> None:
    st.title("CineMatch — Fiche narrative")
    titre = st.text_input("Titre du film", placeholder="ex. Toy Story")

    if st.button("Générer la fiche", type="primary") and titre:
        with st.spinner("Génération en cours..."):
            data = _call_api("/fiche", {"titre": titre})

        if data is None:
            return
        if data["fiche"] is None:
            st.warning("Le film n'a pas pu être identifié.")
            return

        col_img, col_info = st.columns([1, 3])
        with col_img:
            if data.get("affiche_path"):
                st.image(f"{TMDB_IMAGE_BASE}{data['affiche_path']}")
        with col_info:
            st.header(data["titre"])
            if data.get("annee"):
                st.caption(f"{data['annee']} — {', '.join(data.get('genres') or [])}")
            _source_badge(data["source"])

            fiche = data["fiche"]
            st.subheader(fiche["accroche"])
            st.write(fiche["resume"])
            st.markdown(f"**Ambiance :** {fiche['ambiance']}")
            st.markdown(f"**Pourquoi le regarder :** {fiche['pourquoi_regarder']}")


def page_comparaison() -> None:
    st.title("CineMatch — Comparaison")
    col1, col2 = st.columns(2)
    titre_1 = col1.text_input("Film A", placeholder="ex. Toy Story")
    titre_2 = col2.text_input("Film B", placeholder="ex. Jumanji")

    if st.button("Comparer", type="primary") and titre_1 and titre_2:
        with st.spinner("Comparaison en cours..."):
            data = _call_api("/comparaison", {"titre_1": titre_1, "titre_2": titre_2})

        if data is None:
            return

        col1, col2 = st.columns(2)
        for col, film in ((col1, data["film_1"]), (col2, data["film_2"])):
            with col:
                if film.get("affiche_path"):
                    st.image(f"{TMDB_IMAGE_BASE}{film['affiche_path']}", width=200)
                st.subheader(film["titre"])
                if film.get("annee"):
                    st.caption(
                        f"{film['annee']} — {', '.join(film.get('genres') or [])}"
                    )
                if film.get("note_tmdb"):
                    st.caption(f"Note TMDB : {film['note_tmdb']}/10")
                _source_badge(film["source"])

        if data["comparaison"] is None:
            st.warning("La comparaison n'a pas pu être générée.")
            return

        comparaison = data["comparaison"]
        st.markdown(f"**Points communs :** {comparaison['points_communs']}")
        st.markdown(f"**Différences clés :** {comparaison['differences_cles']}")
        st.markdown(f"**Note comparative :** {comparaison['note_comparative']}")


def page_recommandation() -> None:
    st.title("CineMatch — Recommandation")
    st.caption(
        "Suggestions basées sur les connaissances de Gemini (pas de "
        "filtrage sur notre catalogue)."
    )

    col1, col2, col3 = st.columns(3)
    duree_max_heures = col1.number_input(
        "Durée max (heures)", min_value=0.5, max_value=5.0, value=2.0, step=0.5
    )
    genre = col2.text_input("Genre", placeholder="ex. Comédie")
    note_min = col3.slider("Note minimale", min_value=0.0, max_value=10.0, value=6.0)

    if st.button("Recommander", type="primary") and genre:
        with st.spinner("Recherche de suggestions..."):
            data = _call_api(
                "/recommandation",
                {
                    "duree_max_heures": duree_max_heures,
                    "genre": genre,
                    "note_min": note_min,
                },
            )

        if data is None:
            return
        if not data["suggestions"]:
            st.warning("Aucune suggestion n'a pu être générée.")
            return

        for suggestion in data["suggestions"]:
            with st.container(border=True):
                meta = []
                if suggestion.get("annee"):
                    meta.append(str(suggestion["annee"]))
                if suggestion.get("duree_heures"):
                    meta.append(f"{suggestion['duree_heures']}h")
                if suggestion.get("note_estimee"):
                    meta.append(f"{suggestion['note_estimee']}/10")
                st.subheader(suggestion["titre"])
                if meta:
                    st.caption(" — ".join(meta))
                st.write(suggestion["justification"])


page = st.sidebar.radio("Navigation", PAGES)

if page == "Fiche":
    page_fiche()
elif page == "Comparaison":
    page_comparaison()
elif page == "Recommandation":
    page_recommandation()
else:
    st.title(f"CineMatch — {page}")
    st.info("Vue en construction (dépend des modèles ML/NLP de Personne B).")
