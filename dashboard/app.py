"""Dashboard Streamlit CineMatch : fiche, comparaison, recommandation, prédiction, sentiment + vue admin."""

import streamlit as st

st.set_page_config(page_title="CineMatch", layout="wide")

PAGES = ["Fiche", "Comparaison", "Recommandation", "Prédiction de note", "Sentiment", "Admin"]

page = st.sidebar.radio("Navigation", PAGES)

st.title(f"CineMatch — {page}")
st.info("Vue en construction.")
