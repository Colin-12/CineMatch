"""Fonctions pures de feature engineering pour la classification de sentiment.

Philosophie (miroir de ``ml/training/features.py``) : chaque fonction ici est
pure et testable unitairement (DataFrame/array in, DataFrame/array out, pas
d'I/O, pas d'effet de bord, pas d'appel reseau ni de connexion base de
donnees). Le chargement des donnees Gold (avis + notation) est isole dans
``nlp/training/model.py``.

Historique du choix de label (important, a garder trace honnetement) :
la toute premiere version de ce module derivait le label de sentiment de la
note MovieLens laissee par l'utilisateur auquel l'avis TMDB avait ete
rattache *synthetiquement* (tirage aleatoire parmi les votants reels du film,
pour respecter les contraintes de cle etrangere -- voir
``pipeline/transform_silver.py::clean_avis``). Ce label etait donc une
approximation par supervision faible (distant supervision) : la note utilisee
n'etait pas celle de l'auteur reel de l'avis. Cette approche a ete
**abandonnee** une fois la colonne Gold ``avis.note_auteur`` disponible
(``author_details.rating`` de l'API TMDB, capture par
``pipeline/transform_silver.py::clean_avis``, ~94.6% de couverture) : elle est
conservee ci-dessous sous les noms ``derive_label_from_note_movielens_legacy``
/ ``add_sentiment_labels_movielens_legacy`` uniquement a titre de trace
historique et de comparaison documentee dans le notebook, **pas** utilisee
pour l'entrainement du modele de production.

Approche retenue (actuelle) : le label est derive directement de
``note_auteur`` (0-10, la vraie note laissee par l'auteur de l'avis TMDB).
Les avis sans ``note_auteur`` (TMDB ne l'a pas fournie, ~5.4% des cas) sont
**exclus** du jeu d'entrainement/evaluation plutot que de retomber sur
l'ancien label bruite -- on prefere un jeu de donnees plus petit mais a la
supervision fiable qu'un jeu plus grand mais bruite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

LABEL_NEGATIF = "negatif"
LABEL_NEUTRE = "neutre"
LABEL_POSITIF = "positif"
LABELS: list[str] = [LABEL_NEGATIF, LABEL_NEUTRE, LABEL_POSITIF]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Ancienne approche (ABANDONNEE) : label derive de la note MovieLens de
# l'utilisateur synthetiquement rattache. Conservee uniquement pour trace
# historique / comparaison documentee dans le notebook.
# ---------------------------------------------------------------------------


def derive_label_from_note_movielens_legacy(note: float) -> str:
    """ANCIEN label (abandonne) : derive d'une note MovieLens (1-5) qui n'est
    pas celle de l'auteur reel de l'avis (rattachement synthetique). Conserve
    a titre de comparaison historique -- voir ``derive_label_from_note_auteur``
    pour l'approche retenue."""
    if note <= 2:
        return LABEL_NEGATIF
    if note >= 4:
        return LABEL_POSITIF
    return LABEL_NEUTRE


def add_sentiment_labels_movielens_legacy(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> pd.DataFrame:
    """ANCIENNE approche (abandonnee) : jointure avis x notation puis derive
    le label depuis la note MovieLens de l'utilisateur synthetiquement
    rattache. Conservee pour comparaison historique documentee dans le
    notebook -- voir ``add_sentiment_labels`` pour l'approche retenue.

    Ne supprime aucune ligne d'avis : un avis sans notation correspondante
    (ne devrait pas arriver vu la contrainte FK Gold, mais defensif) est
    exclu explicitement via un inner join documente, pas un dropna silencieux.
    """
    merged = avis_df.merge(
        notation_df[["user_id", "film_id", "note"]],
        on=["user_id", "film_id"],
        how="inner",
    )
    merged["label"] = merged["note"].apply(derive_label_from_note_movielens_legacy)
    return merged


# ---------------------------------------------------------------------------
# Approche retenue (actuelle) : label derive de note_auteur (vraie note de
# l'auteur de l'avis TMDB, 0-10, author_details.rating).
# ---------------------------------------------------------------------------


def derive_label_from_note_auteur(note_auteur: float) -> str:
    """Derive un label de sentiment a partir de ``note_auteur`` (0-10), la
    vraie note laissee par l'auteur de l'avis TMDB lui-meme
    (``author_details.rating``).

    Seuils choisis pour rester lisibles/explicables (convention usuelle des
    notes sur 10 : <=4 = mauvais, >=7 = bon, 5-6 = mitige) plutot que calibres
    finement sur la distribution empirique -- documente dans le notebook avec
    la repartition de classes qui en resulte.
    """
    if note_auteur <= 4:
        return LABEL_NEGATIF
    if note_auteur >= 7:
        return LABEL_POSITIF
    return LABEL_NEUTRE


def add_sentiment_labels(avis_df: pd.DataFrame) -> pd.DataFrame:
    """Derive le label de sentiment directement depuis ``avis.note_auteur``
    (approche retenue, remplace l'ancien rattachement via la note MovieLens).

    Les avis sans ``note_auteur`` (non renseignee par TMDB) sont **exclus**
    explicitement (pas de fallback sur l'ancien label bruite) : on prefere un
    jeu de donnees plus petit mais fiable.
    """
    labeled = avis_df.dropna(subset=["note_auteur"]).copy()
    labeled["label"] = labeled["note_auteur"].apply(derive_label_from_note_auteur)
    return labeled


def compute_text_length(df: pd.DataFrame, text_col: str = "texte") -> pd.DataFrame:
    """Ajoute longueur en caracteres et en mots du texte de l'avis."""
    result = df.copy()
    result["longueur_car"] = result[text_col].str.len()
    result["longueur_mots"] = result[text_col].str.split().str.len()
    return result


def label_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """Distribution (comptage) des labels, ordonnee selon LABELS."""
    counts = df[label_col].value_counts()
    return counts.reindex(LABELS).fillna(0).astype(int)


def stratified_label_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/test stratifie sur le label de sentiment (pas de fuite)."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Encode les labels texte en entiers (LABEL2ID)."""
    return labels.map(LABEL2ID).to_numpy()


def decode_labels(label_ids: np.ndarray) -> list[str]:
    """Decode des identifiants entiers vers les labels texte (ID2LABEL)."""
    return [ID2LABEL[int(i)] for i in label_ids]


def predict_baseline_majority(n_rows: int, majority_label: str) -> np.ndarray:
    """Baseline naive : predit toujours la classe majoritaire du train."""
    return np.full(n_rows, majority_label, dtype=object)


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> pd.DataFrame:
    """Matrice de confusion sous forme de DataFrame lisible (lignes=reel)."""
    labels = labels or LABELS
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"reel_{label}" for label in labels],
        columns=[f"predit_{label}" for label in labels],
    )


def compute_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> pd.DataFrame:
    """Precision/rappel/F1/support par classe, sous forme de DataFrame."""
    from sklearn.metrics import classification_report

    labels = labels or LABELS
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).T


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> float:
    """F1 macro (moyenne non ponderee par le support des classes)."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=labels or LABELS, average="macro"))


def weighted_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> float:
    """F1 pondere par le support de chaque classe."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=labels or LABELS, average="weighted"))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy globale."""
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_true, y_pred))


def extract_misclassified(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    text_col: str = "texte",
    max_chars: int = 300,
    note_col: str = "note_auteur",
) -> pd.DataFrame:
    """Extrait les lignes mal classees avec un apercu tronque du texte.

    Retourne les colonnes utiles a l'analyse d'erreurs : texte tronque, note
    d'origine (``note_auteur`` par defaut -- l'approche retenue ; passer
    ``note_col="note"`` pour analyser la comparaison historique avec l'ancien
    label MovieLens), label reel, label predit.
    """
    result = test_df.copy().reset_index(drop=True)
    result["label_reel"] = y_true
    result["label_predit"] = y_pred
    result["texte_apercu"] = result[text_col].str.slice(0, max_chars)
    mask = result["label_reel"] != result["label_predit"]
    cols = [
        "user_id",
        "film_id",
        note_col,
        "label_reel",
        "label_predit",
        "texte_apercu",
    ]
    return result.loc[mask, cols].reset_index(drop=True)


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clippe des probabilites dans [0, 1] (garde-fou numerique)."""
    return np.clip(probabilities, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Agregation d'un score de sentiment continu (endpoint /sentiment, SentimentScore)
# ---------------------------------------------------------------------------

_SCORE_BY_LABEL: dict[str, float] = {
    LABEL_NEGATIF: 0.0,
    LABEL_NEUTRE: 0.5,
    LABEL_POSITIF: 1.0,
}


def expected_sentiment_score(
    proba: np.ndarray, labels: list[str] | None = None
) -> float:
    """Score continu dans [0, 1] : esperance ponderee par les probabilites.

    Mappe negatif -> 0.0, neutre -> 0.5, positif -> 1.0, pondere par la
    probabilite predite de chaque classe. ``labels`` donne l'ordre des
    colonnes de ``proba`` (ex. ``pipeline.classes_`` d'un classifieur
    sklearn) ; par defaut, l'ordre canonique ``LABELS``. Sert a agreger un
    sentiment par avis, puis plusieurs avis d'un meme film (moyenne), en un
    score unique exploitable par l'endpoint `/sentiment`
    (`api/routers/sentiment.py`, schema `SentimentScore.score`).
    """
    labels = labels or LABELS
    weights = np.array([_SCORE_BY_LABEL[label] for label in labels])
    return float(np.dot(np.asarray(proba), weights))


def derive_label_from_score(score: float) -> str:
    """Label agrege a partir d'un score continu dans [0, 1].

    Seuils proportionnels a ceux de ``derive_label_from_note_auteur``
    (echelle 0-10 divisee par 10, pour rester coherent avec le seuillage
    utilise a l'entrainement) : <=0.4 negatif, >=0.7 positif, sinon neutre.
    """
    if score <= 0.4:
        return LABEL_NEGATIF
    if score >= 0.7:
        return LABEL_POSITIF
    return LABEL_NEUTRE
