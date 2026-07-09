"""Feature engineering pour la prediction de note (`/prediction`).

Fonctions pures : chaque fonction prend des DataFrames/valeurs en entree et
retourne un resultat, sans acces base de donnees ni etat cache. Les agregats
user/film DOIVENT etre calcules uniquement sur le train (fonctions
`compute_*_aggregates`) puis fusionnes sur train ET test via `merge_*_features`
pour eviter toute fuite de la cible entre les deux ensembles.

Les colonnes TMDB (`overview`, `popularite`, `note_tmdb`, `affiche_path`) ont
une couverture partielle (~44% du catalogue, cf. eda_exploration.py). Elles ne
sont jamais droppees : un flag `has_tmdb_data` explicite leur absence, et les
valeurs manquantes sont imputees (jamais de `dropna`).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Vocabulaire fixe des genres MovieLens (ordre stable pour le one-hot).
GENRE_VOCAB_FALLBACK = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

RATING_MIN = 1.0
RATING_MAX = 5.0


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def stratified_user_train_test_split(
    notation_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scinde les notations en train/test en stratifiant par `user_id`.

    Chaque utilisateur retrouve la meme proportion de ses notations dans le
    train et dans le test (pas d'utilisateur exclusivement cote test), ce qui
    evite un cold-start artificiel tout en testant sur des paires
    (user, film) jamais vues a l'entrainement.
    """
    train_df, test_df = train_test_split(
        notation_df,
        test_size=test_size,
        random_state=random_state,
        stratify=notation_df["user_id"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def stratified_user_kfold_indices(
    notation_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Renvoie les indices (train, val) d'une validation croisee a `n_splits` folds.

    Stratifie par `user_id` : chaque utilisateur est reparti proportionnellement
    entre les folds, comme pour le split train/test.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(notation_df, notation_df["user_id"]))


# --------------------------------------------------------------------------
# Agregats calcules sur le train uniquement (evite la fuite train/test)
# --------------------------------------------------------------------------


def compute_global_mean(train_notation: pd.DataFrame) -> float:
    """Calcule la moyenne globale des notes du train (baseline naive #1)."""
    return float(train_notation["note"].mean())


def compute_user_aggregates(train_notation: pd.DataFrame) -> pd.DataFrame:
    """Calcule moyenne/ecart-type/nombre de notations par utilisateur (train only)."""
    agg = train_notation.groupby("user_id")["note"].agg(
        user_mean_note="mean", user_std_note="std", user_n_notations="count"
    )
    agg["user_std_note"] = agg["user_std_note"].fillna(0.0)
    return agg.reset_index()


def compute_film_aggregates(train_notation: pd.DataFrame) -> pd.DataFrame:
    """Calcule moyenne/ecart-type/nombre de notations par film (train only).

    C'est aussi la baseline naive #2 (moyenne par film).
    """
    agg = train_notation.groupby("film_id")["note"].agg(
        film_mean_note="mean", film_std_note="std", film_n_notations="count"
    )
    agg["film_std_note"] = agg["film_std_note"].fillna(0.0)
    return agg.reset_index()


def merge_user_features(
    df: pd.DataFrame, user_aggregates: pd.DataFrame, global_mean: float
) -> pd.DataFrame:
    """Fusionne les agregats utilisateur (calcules sur train) sur `df`.

    Les utilisateurs absents des agregats (cold-start) recoivent la moyenne
    globale et un compte de 0 plutot qu'un NaN ou un dropna.
    """
    merged = df.merge(user_aggregates, on="user_id", how="left")
    merged["user_mean_note"] = merged["user_mean_note"].fillna(global_mean)
    merged["user_std_note"] = merged["user_std_note"].fillna(0.0)
    merged["user_n_notations"] = merged["user_n_notations"].fillna(0).astype(int)
    return merged


def merge_film_features(
    df: pd.DataFrame, film_aggregates: pd.DataFrame, global_mean: float
) -> pd.DataFrame:
    """Fusionne les agregats film (calcules sur train) sur `df`.

    Les films absents des agregats (cold-start) recoivent la moyenne globale
    et un compte de 0 plutot qu'un NaN ou un dropna.
    """
    merged = df.merge(film_aggregates, on="film_id", how="left")
    merged["film_mean_note"] = merged["film_mean_note"].fillna(global_mean)
    merged["film_std_note"] = merged["film_std_note"].fillna(0.0)
    merged["film_n_notations"] = merged["film_n_notations"].fillna(0).astype(int)
    return merged


# --------------------------------------------------------------------------
# Features film (metadata + TMDB) : pas de fuite, calculables sur tout le catalogue
# --------------------------------------------------------------------------


def build_genre_vocab(film_df: pd.DataFrame) -> list[str]:
    """Deduit le vocabulaire de genres a partir de `film_df['genres']`.

    Pas de risque de fuite : les genres sont des metadonnees intrinseques du
    film, independantes de la cible (note), donc calculables sur tout le
    catalogue (train + test).
    """
    genres: set[str] = set()
    for row in film_df["genres"]:
        genres.update(row)
    return sorted(genres) if genres else list(GENRE_VOCAB_FALLBACK)


def encode_genres(film_df: pd.DataFrame, genre_vocab: list[str]) -> pd.DataFrame:
    """Encode les genres en colonnes one-hot `genre_<nom>` + `nb_genres`."""
    encoded = film_df[["film_id"]].copy()
    genre_sets = film_df["genres"].apply(set)
    for genre in genre_vocab:
        encoded[f"genre_{genre}"] = genre_sets.apply(lambda s, g=genre: int(g in s))
    encoded["nb_genres"] = film_df["genres"].apply(len)
    return encoded


def add_has_tmdb_flag(film_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `has_tmdb_data` (1 si le film a ete enrichi via TMDB, 0 sinon).

    Se base sur `note_tmdb` : les 4 colonnes TMDB sont couvertes par le meme
    sous-ensemble de films (cf. eda_exploration.py, ~44% de couverture chacune).
    """
    result = film_df.copy()
    result["has_tmdb_data"] = result["note_tmdb"].notna().astype(int)
    return result


def compute_tmdb_imputation_values(train_film_df: pd.DataFrame) -> dict[str, float]:
    """Calcule les valeurs d'imputation (mediane) pour les colonnes TMDB.

    A calculer sur les films presents dans le train uniquement, pour rester
    coherent avec la philosophie "pas de fuite" (meme si les colonnes TMDB ne
    sont pas derivees de la cible, on reste conservateur en n'utilisant que
    l'information disponible a l'entrainement).
    """
    return {
        "popularite": float(train_film_df["popularite"].median(skipna=True) or 0.0),
        "note_tmdb": float(train_film_df["note_tmdb"].median(skipna=True) or 0.0),
        "overview_length": float(
            train_film_df["overview"].fillna("").str.len().median() or 0.0
        ),
    }


def impute_tmdb_features(
    film_df: pd.DataFrame, imputation_values: dict[str, float]
) -> pd.DataFrame:
    """Impute les colonnes TMDB manquantes (jamais de `dropna`).

    `overview` est reduite a sa longueur (`overview_length`), imputee comme
    les autres colonnes numeriques TMDB. `affiche_path` n'est pas utilisee
    comme feature numerique (chemin d'image, deja couvert par `has_tmdb_data`).
    """
    result = film_df.copy()
    result["overview_length"] = result["overview"].fillna("").str.len()
    result.loc[result["overview_length"] == 0, "overview_length"] = imputation_values[
        "overview_length"
    ]
    result["popularite"] = result["popularite"].fillna(imputation_values["popularite"])
    result["note_tmdb"] = result["note_tmdb"].fillna(imputation_values["note_tmdb"])
    return result


def prepare_film_features(
    film_df: pd.DataFrame,
    genre_vocab: list[str],
    imputation_values: dict[str, float],
) -> pd.DataFrame:
    """Pipeline complet des features film : genres + flag TMDB + imputation.

    Combine `encode_genres`, `add_has_tmdb_flag` et `impute_tmdb_features` en
    une seule table de features indexee par `film_id`, prete a etre fusionnee
    sur les notations.
    """
    with_flag = add_has_tmdb_flag(film_df)
    imputed = impute_tmdb_features(with_flag, imputation_values)
    genres_encoded = encode_genres(film_df, genre_vocab)

    feature_cols = [
        "film_id",
        "annee",
        "has_tmdb_data",
        "popularite",
        "note_tmdb",
        "overview_length",
    ]
    films_features = imputed[feature_cols].merge(genres_encoded, on="film_id")
    films_features["annee"] = films_features["annee"].fillna(
        films_features["annee"].median()
    )
    return films_features


# --------------------------------------------------------------------------
# Assemblage final
# --------------------------------------------------------------------------


def build_feature_matrix(
    notation_df: pd.DataFrame,
    films_features: pd.DataFrame,
    user_aggregates: pd.DataFrame,
    film_aggregates: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """Assemble la matrice de features finale (X + y) pour train ou test.

    `user_aggregates` et `film_aggregates` doivent avoir ete calcules sur le
    train (voir `compute_user_aggregates`/`compute_film_aggregates`) meme
    quand on construit les features du test, afin d'eviter toute fuite.
    """
    df = notation_df.merge(films_features, on="film_id", how="left")
    df = merge_user_features(df, user_aggregates, global_mean)
    df = merge_film_features(df, film_aggregates, global_mean)
    return df


def feature_columns(genre_vocab: list[str]) -> list[str]:
    """Liste les colonnes de features (hors identifiants et cible) pour LightGBM."""
    return [
        "annee",
        "has_tmdb_data",
        "popularite",
        "note_tmdb",
        "overview_length",
        "nb_genres",
        "user_mean_note",
        "user_std_note",
        "user_n_notations",
        "film_mean_note",
        "film_std_note",
        "film_n_notations",
    ] + [f"genre_{g}" for g in genre_vocab]


# --------------------------------------------------------------------------
# Baselines naives
# --------------------------------------------------------------------------


def predict_baseline_global(n_rows: int, global_mean: float) -> np.ndarray:
    """Baseline naive #1 : prediction constante = moyenne globale du train."""
    return np.full(shape=n_rows, fill_value=global_mean)


def predict_baseline_film_mean(feature_df: pd.DataFrame) -> np.ndarray:
    """Baseline naive #2 : prediction = moyenne du film (fallback moyenne globale).

    `feature_df` doit contenir la colonne `film_mean_note` produite par
    `merge_film_features` (qui gere deja le fallback cold-start).
    """
    return feature_df["film_mean_note"].to_numpy()


# --------------------------------------------------------------------------
# Metriques
# --------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def clip_ratings(predictions: np.ndarray) -> np.ndarray:
    """Borne les predictions dans l'intervalle valide des notes [1, 5]."""
    return np.clip(predictions, RATING_MIN, RATING_MAX)
