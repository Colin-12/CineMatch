"""Dashboard Streamlit CineMatch : fiche, comparaison, recommandation, ML/NLP."""

import os
from datetime import datetime, timezone

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 15
GITHUB_REPO = "Colin-12/CineMatch"

# Métriques modèles vs seuils Milestone 3 (statique, cf. model cards
# ml/model_cards/lightgbm_rating_v1.0_20260709.md et
# nlp/model_cards/tfidf_logreg_sentiment_v1.1_20260709.md du 2026-07-09).
# À remettre à jour manuellement si Colin réentraîne/republie un modèle.
MODEL_METRICS = [
    {
        "endpoint": "/prediction (LightGBM)",
        "metrique": "RMSE",
        "valeur": "0.9028",
        "seuil": "< 1.0",
        "ok": True,
    },
    {
        "endpoint": "/prediction (LightGBM)",
        "metrique": "Amélioration vs baseline (moy. globale)",
        "valeur": "+19.1%",
        "seuil": "≥ 10%",
        "ok": True,
    },
    {
        "endpoint": "/prediction (LightGBM)",
        "metrique": "Amélioration vs baseline (moy. par film)",
        "valeur": "+7.7%",
        "seuil": "≥ 10%",
        "ok": False,
    },
    {
        "endpoint": "/sentiment (TF-IDF+LogReg)",
        "metrique": "Accuracy",
        "valeur": "0.774",
        "seuil": "> 0.72",
        "ok": True,
    },
    {
        "endpoint": "/sentiment (TF-IDF+LogReg)",
        "metrique": "F1 macro",
        "valeur": "0.665",
        "seuil": "> 0.70",
        "ok": False,
    },
]

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


def page_prediction() -> None:
    st.title("CineMatch — Prédiction de note")
    st.caption(
        "Note prédite (échelle MovieLens 1-5) via LightGBM, features "
        "recalculées en direct sur les données Gold."
    )

    col1, col2 = st.columns(2)
    user_id = col1.number_input("ID utilisateur", min_value=1, step=1, value=1)
    film_id = col2.number_input("ID film", min_value=1, step=1, value=1)

    if st.button("Prédire", type="primary"):
        with st.spinner("Prédiction en cours..."):
            data = _call_api(
                "/prediction", {"user_id": int(user_id), "film_id": int(film_id)}
            )

        if data is None:
            return

        note = data["note_predite"]
        st.metric("Note prédite", f"{note:.2f} / 5")
        st.progress(min(max(note / 5, 0.0), 1.0))


def page_sentiment() -> None:
    st.title("CineMatch — Sentiment")
    st.caption(
        "Score de sentiment agrégé des avis TMDB d'un film (TF-IDF + "
        "régression logistique, entraîné sur note_auteur)."
    )

    titre = st.text_input("Titre du film", placeholder="ex. Toy Story")

    if st.button("Analyser", type="primary") and titre:
        with st.spinner("Analyse en cours..."):
            film = _call_api("/film", {"titre": titre})
            if film is None:
                return
            data = _call_api(f"/sentiment/{film['film_id']}", {})

        if data is None:
            return

        st.caption(f"Film analysé : {film['titre']} ({film.get('annee', 'N/A')})")

        label_display = {
            "positif": ("🟢 Positif", "normal"),
            "neutre": ("🟠 Neutre", "off"),
            "negatif": ("🔴 Négatif", "inverse"),
        }
        label_text, _ = label_display.get(data["label"], (data["label"], "off"))

        col1, col2 = st.columns(2)
        col1.metric("Sentiment agrégé", label_text)
        col2.metric("Score", f"{data['score']:.2f} / 1.00")
        st.progress(min(max(data["score"], 0.0), 1.0))


def _fetch_ci_status() -> dict | None:
    """Statut du dernier run CI GitHub Actions sur main (API publique, sans auth)."""
    try:
        response = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/commits/main/check-runs",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _format_timestamp(ts: int | None) -> str:
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def page_admin() -> None:
    st.title("CineMatch — Admin")

    st.subheader("📊 Chiffres clés Gold")
    stats = _call_api("/admin/stats", {})
    if stats is not None:
        counts = stats["counts"]
        cols = st.columns(4)
        cols[0].metric("Films", f"{counts['film']:,}")
        cols[1].metric("Utilisateurs", f"{counts['utilisateur']:,}")
        cols[2].metric("Notations", f"{counts['notation']:,}")
        cols[3].metric("Avis", f"{counts['avis']:,}")

        coverage = stats["coverage"]
        cols = st.columns(3)
        cols[0].metric("Films enrichis TMDB", f"{coverage['films_enrichis_tmdb_pct']}%")
        cols[1].metric(
            "Avis avec note auteur", f"{coverage['avis_avec_note_auteur_pct']}%"
        )
        cols[2].metric("Films avec avis", f"{coverage['films_avec_avis_pct']}%")

    st.divider()
    st.subheader("🩺 Santé pipeline & CI")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fraîcheur des données**")
        if stats is not None:
            fraicheur = stats["fraicheur"]
            derniere_notation = _format_timestamp(
                fraicheur["derniere_notation_timestamp"]
            )
            dernier_avis = _format_timestamp(fraicheur["dernier_avis_timestamp"])
            st.write(f"Dernière notation : {derniere_notation}")
            st.write(f"Dernier avis : {dernier_avis}")

    with col2:
        st.markdown("**CI GitHub Actions (main)**")
        checks = _fetch_ci_status()
        if checks is None:
            st.caption("Statut CI indisponible.")
        else:
            for run in checks.get("check_runs", []):
                icon = "✅" if run.get("conclusion") == "success" else "❌"
                st.write(f"{icon} {run['name']} — {run.get('conclusion')}")

    st.divider()
    st.subheader("🎯 Métriques modèles vs seuils Milestone 3")
    st.caption(
        "Valeurs statiques issues des model cards (2026-07-09) — à mettre à "
        "jour manuellement en cas de réentraînement."
    )
    for row in MODEL_METRICS:
        icon = "✅" if row["ok"] else "❌"
        st.write(
            f"{icon} **{row['endpoint']}** — {row['metrique']} : "
            f"{row['valeur']} (seuil {row['seuil']})"
        )


page = st.sidebar.radio("Navigation", PAGES)

if page == "Fiche":
    page_fiche()
elif page == "Comparaison":
    page_comparaison()
elif page == "Recommandation":
    page_recommandation()
elif page == "Prédiction de note":
    page_prediction()
elif page == "Sentiment":
    page_sentiment()
elif page == "Admin":
    page_admin()
else:
    st.title(f"CineMatch — {page}")
    st.info("Vue en construction.")
