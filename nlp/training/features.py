"""Fonctions pures de feature engineering pour la classification de sentiment.

Philosophie (miroir de ``ml/training/features.py``) : chaque fonction ici est
pure et testable unitairement (DataFrame/array in, DataFrame/array out, pas
d'I/O, pas d'effet de bord, pas d'appel reseau ni de connexion base de
donnees). Le chargement des donnees Gold (avis + notation) est isole dans
``nlp/training/model.py``.

Contexte metier important (a documenter aussi dans le model card) : la table
``avis`` contient de vrais textes d'avis TMDB, mais leur rattachement a un
``user_id`` MovieLens est **synthetique** (tirage aleatoire parmi les
utilisateurs ayant reellement note le film, pour respecter les contraintes de
cle etrangere -- voir ``pipeline/transform_silver.py::clean_avis``). Le label
de sentiment utilise ici pour l'entrainement est donc derive de la note
laissee par cet utilisateur MovieLens sur le film, et NON de la note que
l'auteur reel de l'avis TMDB aurait pu donner. Il s'agit donc d'un label par
supervision faible (distant supervision), structurellement bruite : un
desaccord entre le label derive de la note et le sentiment reel du texte
n'est pas seulement possible, il est attendu pour une fraction des exemples.
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


def derive_label_from_note(note: float) -> str:
    """Deriv un label de sentiment a partir d'une note MovieLens (1-5)."""
    if note <= 2:
        return LABEL_NEGATIF
    if note >= 4:
        return LABEL_POSITIF
    return LABEL_NEUTRE


def add_sentiment_labels(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> pd.DataFrame:
    """Jointure avis x notation (user_id, film_id) puis derive la colonne label.

    Ne supprime aucune ligne d'avis : un avis sans notation correspondante
    (ne devrait pas arriver vu la contrainte FK Gold, mais defensif) est
    exclu explicitement via un inner join documente, pas un dropna silencieux.
    """
    merged = avis_df.merge(
        notation_df[["user_id", "film_id", "note"]],
        on=["user_id", "film_id"],
        how="inner",
    )
    merged["label"] = merged["note"].apply(derive_label_from_note)
    return merged


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
) -> pd.DataFrame:
    """Extrait les lignes mal classees avec un apercu tronque du texte.

    Retourne les colonnes utiles a l'analyse d'erreurs : texte tronque,
    note d'origine, label reel, label predit.
    """
    result = test_df.copy().reset_index(drop=True)
    result["label_reel"] = y_true
    result["label_predit"] = y_pred
    result["texte_apercu"] = result[text_col].str.slice(0, max_chars)
    mask = result["label_reel"] != result["label_predit"]
    cols = ["user_id", "film_id", "note", "label_reel", "label_predit", "texte_apercu"]
    return result.loc[mask, cols].reset_index(drop=True)


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clippe des probabilites dans [0, 1] (garde-fou numerique)."""
    return np.clip(probabilities, 0.0, 1.0)
