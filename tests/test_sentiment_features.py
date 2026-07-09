"""Tests unitaires des fonctions pures de feature engineering sentiment
(nlp.training.features).

Deux familles de tests cohabitent, a l'image du module source :
- l'approche RETENUE (label derive de ``note_auteur``, la vraie note TMDB de
  l'auteur de l'avis) ;
- l'ANCIENNE approche ABANDONNEE (label derive de la note MovieLens de
  l'utilisateur synthetiquement rattache), conservee pour comparaison
  historique documentee dans le notebook.
"""

import numpy as np
import pandas as pd
import pytest

from nlp.training import features


@pytest.fixture
def avis_df() -> pd.DataFrame:
    """Avis avec `note_auteur` (0-10), dont un sans note_auteur (a exclure).

    Au moins 2 avis par classe (positif/neutre/negatif) une fois le label
    derive, pour que le split stratifie (qui exige >=2 membres par classe)
    reste exercable sur ce jeu de test.
    """
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "film_id": [10, 10, 20, 20, 30, 30, 40, 40],
            "texte": [
                "A great movie, loved every minute of it.",
                "Terrible, a waste of time.",
                "It was fine, nothing special.",
                "Amazing cinematography and story.",
                "I hated the ending.",
                "Decent but forgettable, middle of the road.",
                "Awful acting, I almost left the theater.",
                "An average film, could go either way.",
            ],
            "timestamp": list(range(8)),
            "note_auteur": [9.0, 2.0, 6.0, 8.0, 3.0, 5.0, 2.5, None],
        }
    )


@pytest.fixture
def notation_df() -> pd.DataFrame:
    """Notations MovieLens (utilisees uniquement par l'ancienne approche)."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "film_id": [10, 10, 20, 20, 30, 30, 40, 40],
            "note": [5.0, 1.0, 3.0, 4.0, 2.0, 3.0, 1.0, 3.0],
        }
    )


# ---------------------------------------------------------------------------
# Approche retenue : label derive de note_auteur
# ---------------------------------------------------------------------------


def test_derive_label_from_note_auteur_negatif() -> None:
    assert features.derive_label_from_note_auteur(1.0) == features.LABEL_NEGATIF
    assert features.derive_label_from_note_auteur(4.0) == features.LABEL_NEGATIF


def test_derive_label_from_note_auteur_neutre() -> None:
    assert features.derive_label_from_note_auteur(5.0) == features.LABEL_NEUTRE
    assert features.derive_label_from_note_auteur(6.0) == features.LABEL_NEUTRE


def test_derive_label_from_note_auteur_positif() -> None:
    assert features.derive_label_from_note_auteur(7.0) == features.LABEL_POSITIF
    assert features.derive_label_from_note_auteur(10.0) == features.LABEL_POSITIF


def test_add_sentiment_labels_derives_from_note_auteur(avis_df: pd.DataFrame) -> None:
    labeled = features.add_sentiment_labels(avis_df)
    row = labeled[labeled["user_id"] == 1].iloc[0]
    assert row["label"] == features.LABEL_POSITIF
    row2 = labeled[labeled["user_id"] == 2].iloc[0]
    assert row2["label"] == features.LABEL_NEGATIF


def test_add_sentiment_labels_excludes_missing_note_auteur(
    avis_df: pd.DataFrame,
) -> None:
    """L'avis sans note_auteur (user_id=8) doit etre exclu, pas fallback."""
    labeled = features.add_sentiment_labels(avis_df)
    assert len(labeled) == len(avis_df) - 1
    assert 8 not in set(labeled["user_id"])


# ---------------------------------------------------------------------------
# Ancienne approche (abandonnee), conservee pour comparaison historique
# ---------------------------------------------------------------------------


def test_derive_label_from_note_movielens_legacy_negatif() -> None:
    assert (
        features.derive_label_from_note_movielens_legacy(1.0) == features.LABEL_NEGATIF
    )
    assert (
        features.derive_label_from_note_movielens_legacy(2.0) == features.LABEL_NEGATIF
    )


def test_derive_label_from_note_movielens_legacy_neutre() -> None:
    assert (
        features.derive_label_from_note_movielens_legacy(3.0) == features.LABEL_NEUTRE
    )


def test_derive_label_from_note_movielens_legacy_positif() -> None:
    assert (
        features.derive_label_from_note_movielens_legacy(4.0) == features.LABEL_POSITIF
    )
    assert (
        features.derive_label_from_note_movielens_legacy(5.0) == features.LABEL_POSITIF
    )


def test_add_sentiment_labels_movielens_legacy_merges_and_labels(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    merged = features.add_sentiment_labels_movielens_legacy(avis_df, notation_df)
    assert len(merged) == len(avis_df)
    row = merged[merged["user_id"] == 1].iloc[0]
    assert row["label"] == features.LABEL_POSITIF
    row2 = merged[merged["user_id"] == 2].iloc[0]
    assert row2["label"] == features.LABEL_NEGATIF


def test_add_sentiment_labels_movielens_legacy_drops_unmatched_rows(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    """Un avis sans notation correspondante (cas defensif) est exclu."""
    partial_notation = notation_df.iloc[:-1]
    merged = features.add_sentiment_labels_movielens_legacy(avis_df, partial_notation)
    assert len(merged) == len(avis_df) - 1


# ---------------------------------------------------------------------------
# Fonctions independantes de la source du label
# ---------------------------------------------------------------------------


def test_compute_text_length(avis_df: pd.DataFrame) -> None:
    result = features.compute_text_length(avis_df)
    row = result[result["user_id"] == 1].iloc[0]
    assert row["longueur_car"] == len("A great movie, loved every minute of it.")
    assert row["longueur_mots"] == 8


def test_label_distribution_orders_by_labels(avis_df: pd.DataFrame) -> None:
    labeled = features.add_sentiment_labels(avis_df)
    dist = features.label_distribution(labeled)
    assert list(dist.index) == features.LABELS
    assert dist.sum() == len(labeled)


def test_stratified_label_split_keeps_size(avis_df: pd.DataFrame) -> None:
    labeled = features.add_sentiment_labels(avis_df)
    train_df, test_df = features.stratified_label_train_test_split(
        labeled, test_size=0.4, random_state=0
    )
    assert len(train_df) + len(test_df) == len(labeled)


def test_encode_decode_labels_roundtrip() -> None:
    labels = pd.Series(
        [features.LABEL_NEGATIF, features.LABEL_POSITIF, features.LABEL_NEUTRE]
    )
    encoded = features.encode_labels(labels)
    decoded = features.decode_labels(encoded)
    assert decoded == list(labels)


def test_predict_baseline_majority() -> None:
    preds = features.predict_baseline_majority(3, features.LABEL_POSITIF)
    np.testing.assert_array_equal(
        preds, np.array([features.LABEL_POSITIF] * 3, dtype=object)
    )


def test_compute_confusion_matrix_shape() -> None:
    y_true = np.array(
        [features.LABEL_POSITIF, features.LABEL_NEGATIF, features.LABEL_NEUTRE]
    )
    y_pred = np.array(
        [features.LABEL_POSITIF, features.LABEL_POSITIF, features.LABEL_NEUTRE]
    )
    cm = features.compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.loc["reel_positif", "predit_positif"] == 1
    assert cm.loc["reel_negatif", "predit_positif"] == 1


def test_macro_f1_perfect_predictions() -> None:
    y_true = np.array(
        [features.LABEL_POSITIF, features.LABEL_NEGATIF, features.LABEL_NEUTRE]
    )
    assert features.macro_f1(y_true, y_true) == pytest.approx(1.0)


def test_accuracy_known_value() -> None:
    y_true = np.array(["a", "b", "c", "d"])
    y_pred = np.array(["a", "b", "x", "y"])
    assert features.accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_extract_misclassified_only_returns_errors(avis_df: pd.DataFrame) -> None:
    labeled = features.add_sentiment_labels(avis_df)
    y_true = labeled["label"].to_numpy()
    y_pred = y_true.copy()
    y_pred[0] = (
        features.LABEL_NEGATIF
        if y_true[0] != features.LABEL_NEGATIF
        else features.LABEL_NEUTRE
    )
    misclassified = features.extract_misclassified(labeled, y_true, y_pred)
    assert len(misclassified) == 1
    assert misclassified.iloc[0]["label_reel"] != misclassified.iloc[0]["label_predit"]
    assert "note_auteur" in misclassified.columns


def test_extract_misclassified_with_legacy_note_col(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    """Le parametre note_col permet de rejouer l'analyse avec l'ancien label."""
    merged = features.add_sentiment_labels_movielens_legacy(avis_df, notation_df)
    y_true = merged["label"].to_numpy()
    y_pred = y_true.copy()
    y_pred[0] = (
        features.LABEL_NEGATIF
        if y_true[0] != features.LABEL_NEGATIF
        else features.LABEL_NEUTRE
    )
    misclassified = features.extract_misclassified(
        merged, y_true, y_pred, note_col="note"
    )
    assert "note" in misclassified.columns


def test_clip_probabilities_bounds() -> None:
    clipped = features.clip_probabilities(np.array([-0.2, 0.5, 1.4]))
    np.testing.assert_array_equal(clipped, np.array([0.0, 0.5, 1.0]))


# ---------------------------------------------------------------------------
# Agregation du score de sentiment continu (endpoint /sentiment)
# ---------------------------------------------------------------------------


def test_expected_sentiment_score_all_positive() -> None:
    proba = np.array([0.0, 0.0, 1.0])  # ordre LABELS : negatif, neutre, positif
    assert features.expected_sentiment_score(proba) == pytest.approx(1.0)


def test_expected_sentiment_score_all_negative() -> None:
    proba = np.array([1.0, 0.0, 0.0])
    assert features.expected_sentiment_score(proba) == pytest.approx(0.0)


def test_expected_sentiment_score_mixed_respects_label_order() -> None:
    """Avec un ordre de labels different de LABELS, le score doit rester
    correct (verifie que le parametre `labels` est bien utilise)."""
    proba_positif_only = np.array([1.0, 0.0, 0.0])
    score = features.expected_sentiment_score(
        proba_positif_only,
        labels=[features.LABEL_POSITIF, features.LABEL_NEUTRE, features.LABEL_NEGATIF],
    )
    assert score == pytest.approx(1.0)


def test_derive_label_from_score_thresholds() -> None:
    assert features.derive_label_from_score(0.0) == features.LABEL_NEGATIF
    assert features.derive_label_from_score(0.4) == features.LABEL_NEGATIF
    assert features.derive_label_from_score(0.5) == features.LABEL_NEUTRE
    assert features.derive_label_from_score(0.7) == features.LABEL_POSITIF
    assert features.derive_label_from_score(1.0) == features.LABEL_POSITIF
