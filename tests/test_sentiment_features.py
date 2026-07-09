"""Tests unitaires des fonctions pures de feature engineering sentiment
(nlp.training.features)."""

import numpy as np
import pandas as pd
import pytest

from nlp.training import features


@pytest.fixture
def avis_df() -> pd.DataFrame:
    """Avis synthetiques (texte + FK user/film) sans note attachee."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "film_id": [10, 10, 20, 20, 30, 30],
            "texte": [
                "A great movie, loved every minute of it.",
                "Terrible, a waste of time.",
                "It was fine, nothing special.",
                "Amazing cinematography and story.",
                "I hated the ending.",
                "An average film, could go either way.",
            ],
            "timestamp": list(range(6)),
        }
    )


@pytest.fixture
def notation_df() -> pd.DataFrame:
    """Notations correspondantes (1 note par paire user/film de avis_df)."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "film_id": [10, 10, 20, 20, 30, 30],
            "note": [5.0, 1.0, 3.0, 4.0, 2.0, 3.0],
        }
    )


def test_derive_label_from_note_negatif() -> None:
    assert features.derive_label_from_note(1.0) == features.LABEL_NEGATIF
    assert features.derive_label_from_note(2.0) == features.LABEL_NEGATIF


def test_derive_label_from_note_neutre() -> None:
    assert features.derive_label_from_note(3.0) == features.LABEL_NEUTRE


def test_derive_label_from_note_positif() -> None:
    assert features.derive_label_from_note(4.0) == features.LABEL_POSITIF
    assert features.derive_label_from_note(5.0) == features.LABEL_POSITIF


def test_add_sentiment_labels_merges_and_labels(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    merged = features.add_sentiment_labels(avis_df, notation_df)
    assert len(merged) == len(avis_df)
    row = merged[merged["user_id"] == 1].iloc[0]
    assert row["label"] == features.LABEL_POSITIF
    row2 = merged[merged["user_id"] == 2].iloc[0]
    assert row2["label"] == features.LABEL_NEGATIF


def test_add_sentiment_labels_drops_unmatched_rows(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    """Un avis sans notation correspondante (cas defensif) est exclu."""
    partial_notation = notation_df.iloc[:-1]
    merged = features.add_sentiment_labels(avis_df, partial_notation)
    assert len(merged) == len(avis_df) - 1


def test_compute_text_length(avis_df: pd.DataFrame) -> None:
    result = features.compute_text_length(avis_df)
    row = result[result["user_id"] == 1].iloc[0]
    assert row["longueur_car"] == len("A great movie, loved every minute of it.")
    assert row["longueur_mots"] == 8


def test_label_distribution_orders_by_labels(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    merged = features.add_sentiment_labels(avis_df, notation_df)
    dist = features.label_distribution(merged)
    assert list(dist.index) == features.LABELS
    assert dist.sum() == len(merged)


def test_stratified_label_split_keeps_size(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    merged = features.add_sentiment_labels(avis_df, notation_df)
    # 3 classes distinctes, train et test doivent chacun couvrir >= 1 exemplaire/classe.
    train_df, test_df = features.stratified_label_train_test_split(
        merged, test_size=0.5, random_state=0
    )
    assert len(train_df) + len(test_df) == len(merged)


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


def test_extract_misclassified_only_returns_errors(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> None:
    merged = features.add_sentiment_labels(avis_df, notation_df)
    y_true = merged["label"].to_numpy()
    y_pred = y_true.copy()
    y_pred[0] = (
        features.LABEL_NEGATIF
        if y_true[0] != features.LABEL_NEGATIF
        else features.LABEL_NEUTRE
    )
    misclassified = features.extract_misclassified(merged, y_true, y_pred)
    assert len(misclassified) == 1
    assert misclassified.iloc[0]["label_reel"] != misclassified.iloc[0]["label_predit"]


def test_clip_probabilities_bounds() -> None:
    clipped = features.clip_probabilities(np.array([-0.2, 0.5, 1.4]))
    np.testing.assert_array_equal(clipped, np.array([0.0, 0.5, 1.0]))
