"""Tests unitaires des fonctions pures de feature engineering (ml.training.features)."""

import numpy as np
import pandas as pd
import pytest

from ml.training import features


@pytest.fixture
def notation_df() -> pd.DataFrame:
    """Notations synthetiques : 4 utilisateurs, 3 films, notes variees."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            "film_id": [10, 20, 30, 10, 20, 30, 10, 20, 10, 30],
            "note": [5.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 2.0],
            "timestamp": list(range(10)),
        }
    )


@pytest.fixture
def film_df() -> pd.DataFrame:
    """Films synthetiques avec couverture TMDB partielle (film 30 = NULL partout)."""
    return pd.DataFrame(
        {
            "film_id": [10, 20, 30],
            "titre": ["Film A", "Film B", "Film C"],
            "annee": [2001, 1999, None],
            "genres": [["Action", "Drama"], ["Comedy"], ["Drama"]],
            "overview": ["Un heros.", "Une comedie.", None],
            "popularite": [12.5, 8.0, None],
            "note_tmdb": [7.2, 6.5, None],
            "affiche_path": ["/a.jpg", "/b.jpg", None],
        }
    )


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def test_stratified_split_keeps_every_user_in_both_sets(
    notation_df: pd.DataFrame,
) -> None:
    train_df, test_df = features.stratified_user_train_test_split(
        notation_df, test_size=0.5, random_state=0
    )
    assert len(train_df) + len(test_df) == len(notation_df)
    # Chaque utilisateur avec >= 2 notations doit apparaitre dans les deux sets.
    for user_id in [1, 2]:
        assert user_id in train_df["user_id"].values
        assert user_id in test_df["user_id"].values


def test_stratified_split_no_row_leakage(notation_df: pd.DataFrame) -> None:
    train_df, test_df = features.stratified_user_train_test_split(
        notation_df, test_size=0.5, random_state=1
    )
    train_pairs = set(zip(train_df["user_id"], train_df["film_id"]))
    test_pairs = set(zip(test_df["user_id"], test_df["film_id"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_kfold_indices_cover_all_rows_without_overlap(
    notation_df: pd.DataFrame,
) -> None:
    folds = features.stratified_user_kfold_indices(
        notation_df, n_splits=2, random_state=0
    )
    assert len(folds) == 2
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) + len(val_idx) == len(notation_df)


# --------------------------------------------------------------------------
# Agregats train-only
# --------------------------------------------------------------------------


def test_compute_global_mean(notation_df: pd.DataFrame) -> None:
    assert features.compute_global_mean(notation_df) == pytest.approx(
        notation_df["note"].mean()
    )


def test_compute_user_aggregates(notation_df: pd.DataFrame) -> None:
    agg = features.compute_user_aggregates(notation_df)
    row = agg[agg["user_id"] == 1].iloc[0]
    assert row["user_mean_note"] == pytest.approx(4.0)
    assert row["user_n_notations"] == 3


def test_compute_film_aggregates(notation_df: pd.DataFrame) -> None:
    agg = features.compute_film_aggregates(notation_df)
    row = agg[agg["film_id"] == 10].iloc[0]
    assert row["film_mean_note"] == pytest.approx((5.0 + 1.0 + 4.0 + 5.0) / 4)
    assert row["film_n_notations"] == 4


def test_merge_user_features_cold_start_fallback(notation_df: pd.DataFrame) -> None:
    """Utilisateur absent des agregats (cold-start) -> moyenne globale, pas de NaN."""
    train_agg = features.compute_user_aggregates(
        notation_df[notation_df["user_id"] != 4]
    )
    global_mean = features.compute_global_mean(notation_df)

    unseen_user_df = pd.DataFrame({"user_id": [4], "film_id": [10]})
    merged = features.merge_user_features(unseen_user_df, train_agg, global_mean)

    assert merged["user_mean_note"].iloc[0] == pytest.approx(global_mean)
    assert merged["user_n_notations"].iloc[0] == 0
    assert not merged.isna().any().any()


def test_merge_film_features_cold_start_fallback(notation_df: pd.DataFrame) -> None:
    train_agg = features.compute_film_aggregates(
        notation_df[notation_df["film_id"] != 30]
    )
    global_mean = features.compute_global_mean(notation_df)

    unseen_film_df = pd.DataFrame({"user_id": [1], "film_id": [30]})
    merged = features.merge_film_features(unseen_film_df, train_agg, global_mean)

    assert merged["film_mean_note"].iloc[0] == pytest.approx(global_mean)
    assert merged["film_n_notations"].iloc[0] == 0
    assert not merged.isna().any().any()


# --------------------------------------------------------------------------
# Features film (genres + TMDB)
# --------------------------------------------------------------------------


def test_build_genre_vocab_is_sorted_and_deduplicated(film_df: pd.DataFrame) -> None:
    vocab = features.build_genre_vocab(film_df)
    assert vocab == sorted(set(vocab))
    assert "Action" in vocab and "Comedy" in vocab and "Drama" in vocab


def test_encode_genres_one_hot(film_df: pd.DataFrame) -> None:
    vocab = features.build_genre_vocab(film_df)
    encoded = features.encode_genres(film_df, vocab)
    film_a = encoded[encoded["film_id"] == 10].iloc[0]
    assert film_a["genre_Action"] == 1
    assert film_a["genre_Comedy"] == 0
    assert film_a["nb_genres"] == 2


def test_add_has_tmdb_flag(film_df: pd.DataFrame) -> None:
    flagged = features.add_has_tmdb_flag(film_df)
    flags = dict(zip(flagged["film_id"], flagged["has_tmdb_data"]))
    assert flags[10] == 1
    assert flags[20] == 1
    assert flags[30] == 0  # note_tmdb NULL -> pas de donnees TMDB


def test_impute_tmdb_features_no_nan_remains(film_df: pd.DataFrame) -> None:
    imputation_values = features.compute_tmdb_imputation_values(film_df)
    imputed = features.impute_tmdb_features(film_df, imputation_values)
    assert not imputed["popularite"].isna().any()
    assert not imputed["note_tmdb"].isna().any()
    assert not imputed["overview_length"].isna().any()
    # Le film sans overview recoit la mediane, pas 0 artificiel.
    assert imputed.loc[imputed["film_id"] == 30, "overview_length"].iloc[
        0
    ] == pytest.approx(imputation_values["overview_length"])


def test_prepare_film_features_never_drops_rows(film_df: pd.DataFrame) -> None:
    vocab = features.build_genre_vocab(film_df)
    imputation_values = features.compute_tmdb_imputation_values(film_df)
    result = features.prepare_film_features(film_df, vocab, imputation_values)
    assert len(result) == len(film_df)  # pas de dropna
    assert not result.isna().any().any()


# --------------------------------------------------------------------------
# Assemblage complet
# --------------------------------------------------------------------------


def test_build_feature_matrix_has_no_missing_values(
    notation_df: pd.DataFrame, film_df: pd.DataFrame
) -> None:
    train_df, test_df = features.stratified_user_train_test_split(
        notation_df, test_size=0.5, random_state=0
    )
    global_mean = features.compute_global_mean(train_df)
    user_agg = features.compute_user_aggregates(train_df)
    film_agg = features.compute_film_aggregates(train_df)
    vocab = features.build_genre_vocab(film_df)
    imputation_values = features.compute_tmdb_imputation_values(film_df)
    films_features = features.prepare_film_features(film_df, vocab, imputation_values)

    train_matrix = features.build_feature_matrix(
        train_df, films_features, user_agg, film_agg, global_mean
    )
    test_matrix = features.build_feature_matrix(
        test_df, films_features, user_agg, film_agg, global_mean
    )

    cols = features.feature_columns(vocab)
    assert not train_matrix[cols].isna().any().any()
    assert not test_matrix[cols].isna().any().any()


# --------------------------------------------------------------------------
# Baselines et metriques
# --------------------------------------------------------------------------


def test_predict_baseline_global() -> None:
    preds = features.predict_baseline_global(3, 3.5)
    np.testing.assert_array_equal(preds, np.array([3.5, 3.5, 3.5]))


def test_predict_baseline_film_mean() -> None:
    feature_df = pd.DataFrame({"film_mean_note": [3.0, 4.5]})
    preds = features.predict_baseline_film_mean(feature_df)
    np.testing.assert_array_equal(preds, np.array([3.0, 4.5]))


def test_rmse_zero_for_perfect_predictions() -> None:
    assert features.rmse(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(0.0)


def test_rmse_known_value() -> None:
    assert features.rmse(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(
        np.sqrt((1 + 4) / 2)
    )


def test_mae_known_value() -> None:
    assert features.mae(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(
        1.5
    )


def test_clip_ratings_bounds() -> None:
    clipped = features.clip_ratings(np.array([0.0, 3.0, 6.0]))
    np.testing.assert_array_equal(clipped, np.array([1.0, 3.0, 5.0]))
