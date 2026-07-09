"""Entrainement des modeles de prediction de note pour l'endpoint `/prediction`.

Trois modeles sont compares sur un split train/test stratifie par utilisateur :
baseline naive (moyenne globale / moyenne par film), SVD biaise (factorisation
matricielle) et LightGBM (avec un tuning leger par grille + validation croisee
5-fold sur le train). Le meilleur modele est exporte au format joblib
versionne (cf. convention `<nom>_v<major>.<minor>_<AAAAMMJJ>.joblib`).

Seuils cibles (feuille de route equipe) : RMSE < 1.0 et amelioration >= 10%
vs la baseline naive la plus forte (moyenne par film).

Usage (depuis la racine du repo, pour que les imports `ml.training.*`
resolvent) : python -m ml.training.model
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import ParameterGrid

from ml.training import features

load_dotenv()

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_VERSION = "1.0"

RMSE_THRESHOLD = 1.0
MIN_IMPROVEMENT_VS_BASELINE = 0.10

DEFAULT_LIGHTGBM_PARAM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}


def print_section(title: str) -> None:
    """Affiche un separateur de section lisible dans la sortie texte."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# --------------------------------------------------------------------------
# Chargement des donnees Gold
# --------------------------------------------------------------------------


def load_gold_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge `notation` et `film` depuis la base Gold (`DATABASE_URL`)."""
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, film_id, note, timestamp FROM notation;")
            notation_df = pd.DataFrame(
                cur.fetchall(), columns=["user_id", "film_id", "note", "timestamp"]
            )

            cur.execute("""
                SELECT film_id, titre, annee, genres, overview, popularite,
                       note_tmdb, affiche_path
                FROM film;
                """)
            film_df = pd.DataFrame(
                cur.fetchall(),
                columns=[
                    "film_id",
                    "titre",
                    "annee",
                    "genres",
                    "overview",
                    "popularite",
                    "note_tmdb",
                    "affiche_path",
                ],
            )
    finally:
        conn.close()
    return notation_df, film_df


# --------------------------------------------------------------------------
# Preparation des datasets (features)
# --------------------------------------------------------------------------


@dataclass
class FeatureContext:
    """Artefacts calcules sur le train, necessaires pour reproduire les features
    a l'inference (evite toute fuite train/test)."""

    global_mean: float
    user_aggregates: pd.DataFrame
    film_aggregates: pd.DataFrame
    films_features: pd.DataFrame
    genre_vocab: list[str]
    feature_cols: list[str]


def build_feature_context(
    train_notation: pd.DataFrame, film_df: pd.DataFrame
) -> FeatureContext:
    """Calcule tous les artefacts de feature engineering a partir du train."""
    global_mean = features.compute_global_mean(train_notation)
    user_aggregates = features.compute_user_aggregates(train_notation)
    film_aggregates = features.compute_film_aggregates(train_notation)

    genre_vocab = features.build_genre_vocab(film_df)
    train_film_ids = set(train_notation["film_id"])
    train_film_df = film_df[film_df["film_id"].isin(train_film_ids)]
    imputation_values = features.compute_tmdb_imputation_values(train_film_df)
    films_features = features.prepare_film_features(
        film_df, genre_vocab, imputation_values
    )

    return FeatureContext(
        global_mean=global_mean,
        user_aggregates=user_aggregates,
        film_aggregates=film_aggregates,
        films_features=films_features,
        genre_vocab=genre_vocab,
        feature_cols=features.feature_columns(genre_vocab),
    )


def build_matrix(notation_df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
    """Assemble la matrice de features pour un DataFrame de notations donne."""
    return features.build_feature_matrix(
        notation_df,
        context.films_features,
        context.user_aggregates,
        context.film_aggregates,
        context.global_mean,
    )


# --------------------------------------------------------------------------
# Baselines naives
# --------------------------------------------------------------------------


def evaluate_baselines(
    test_matrix: pd.DataFrame, global_mean: float
) -> dict[str, dict[str, float]]:
    """Evalue les deux baselines naives (moyenne globale, moyenne par film)."""
    y_test = test_matrix["note"].to_numpy()

    pred_global = features.predict_baseline_global(len(test_matrix), global_mean)
    pred_film_mean = features.predict_baseline_film_mean(test_matrix)

    return {
        "baseline_moyenne_globale": {
            "rmse": features.rmse(y_test, pred_global),
            "mae": features.mae(y_test, pred_global),
        },
        "baseline_moyenne_par_film": {
            "rmse": features.rmse(y_test, pred_film_mean),
            "mae": features.mae(y_test, pred_film_mean),
        },
    }


# --------------------------------------------------------------------------
# SVD biaise (factorisation matricielle)
# --------------------------------------------------------------------------


class BiasedSVDRecommender:
    """Factorisation matricielle biaisee : note = mu + biais_user + biais_film
    + <facteur_user, facteur_film>, entrainee par SVD tronque (scipy) sur la
    matrice des residus (note - biais), remplie a 0 pour les couples non
    observes.
    """

    def __init__(self, n_factors: int = 20) -> None:
        self.n_factors = n_factors
        self.global_mean_ = 0.0
        self.user_bias_: dict[int, float] = {}
        self.film_bias_: dict[int, float] = {}
        self.user_factors_: dict[int, np.ndarray] = {}
        self.film_factors_: dict[int, np.ndarray] = {}

    def fit(self, train_notation: pd.DataFrame) -> "BiasedSVDRecommender":
        """Entraine le modele sur les notations du train uniquement."""
        self.global_mean_ = float(train_notation["note"].mean())

        user_ids = sorted(train_notation["user_id"].unique())
        film_ids = sorted(train_notation["film_id"].unique())
        user_index = {u: i for i, u in enumerate(user_ids)}
        film_index = {f: i for i, f in enumerate(film_ids)}

        user_mean = train_notation.groupby("user_id")["note"].mean()
        film_mean = train_notation.groupby("film_id")["note"].mean()
        self.user_bias_ = (user_mean - self.global_mean_).to_dict()
        self.film_bias_ = (film_mean - self.global_mean_).to_dict()

        rows, cols, values = [], [], []
        for user_id, film_id, note in zip(
            train_notation["user_id"], train_notation["film_id"], train_notation["note"]
        ):
            residual = (
                note
                - self.global_mean_
                - self.user_bias_.get(user_id, 0.0)
                - self.film_bias_.get(film_id, 0.0)
            )
            rows.append(user_index[user_id])
            cols.append(film_index[film_id])
            values.append(residual)

        residual_matrix = coo_matrix(
            (values, (rows, cols)), shape=(len(user_ids), len(film_ids))
        ).tocsr()

        k = min(self.n_factors, min(residual_matrix.shape) - 1)
        u, s, vt = svds(residual_matrix, k=k)
        # svds renvoie les valeurs singulieres en ordre croissant -> on retrie.
        order = np.argsort(-s)
        u, s, vt = u[:, order], s[order], vt[order, :]

        user_latent = u * s
        film_latent = vt.T

        self.user_factors_ = {uid: user_latent[idx] for uid, idx in user_index.items()}
        self.film_factors_ = {fid: film_latent[idx] for fid, idx in film_index.items()}
        self.n_factors = k
        return self

    def predict(self, user_id: int, film_id: int) -> float:
        """Predit la note d'un couple (user_id, film_id), avec fallback cold-start."""
        pred = (
            self.global_mean_
            + self.user_bias_.get(user_id, 0.0)
            + self.film_bias_.get(film_id, 0.0)
        )
        u_vec = self.user_factors_.get(user_id)
        f_vec = self.film_factors_.get(film_id)
        if u_vec is not None and f_vec is not None:
            pred += float(np.dot(u_vec, f_vec))
        return pred

    def predict_batch(self, notation_df: pd.DataFrame) -> np.ndarray:
        """Predit un lot de couples (user_id, film_id)."""
        return np.array(
            [
                self.predict(u, f)
                for u, f in zip(notation_df["user_id"], notation_df["film_id"])
            ]
        )


# --------------------------------------------------------------------------
# LightGBM (tuning leger par grille + validation croisee 5-fold)
# --------------------------------------------------------------------------


def grid_search_lightgbm(
    train_matrix: pd.DataFrame,
    feature_cols: list[str],
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grille simple + validation croisee 5-fold (stratifiee par utilisateur).

    Priorise la robustesse (folds sans fuite, memes agregats recalcules par
    fold en amont via `train_matrix`) plutot qu'un tuning pousse (Optuna) : le
    train est deja feature-engineere une fois pour toutes ici car les
    agregats user/film utilises sont ceux du train global (pas de recalcul
    par fold), ce qui est une simplification assumee pour le temps imparti.
    """
    grid = list(ParameterGrid(param_grid or DEFAULT_LIGHTGBM_PARAM_GRID))
    folds = features.stratified_user_kfold_indices(
        train_matrix, n_splits=n_splits, random_state=random_state
    )

    records = []
    for params in grid:
        fold_rmses = []
        for train_idx, val_idx in folds:
            fold_train = train_matrix.iloc[train_idx]
            fold_val = train_matrix.iloc[val_idx]
            model = lgb.LGBMRegressor(random_state=random_state, verbosity=-1, **params)
            model.fit(fold_train[feature_cols], fold_train["note"])
            preds = features.clip_ratings(model.predict(fold_val[feature_cols]))
            fold_rmses.append(features.rmse(fold_val["note"].to_numpy(), preds))
        records.append(
            {
                "params": params,
                "cv_rmse_mean": float(np.mean(fold_rmses)),
                "cv_rmse_std": float(np.std(fold_rmses)),
            }
        )

    records.sort(key=lambda r: r["cv_rmse_mean"])
    best_params = records[0]["params"]
    results_df = pd.DataFrame(
        [
            {
                **r["params"],
                "cv_rmse_mean": r["cv_rmse_mean"],
                "cv_rmse_std": r["cv_rmse_std"],
            }
            for r in records
        ]
    )
    return best_params, results_df


def train_final_lightgbm(
    train_matrix: pd.DataFrame, feature_cols: list[str], params: dict[str, Any]
) -> lgb.LGBMRegressor:
    """Entraine le modele LightGBM final sur tout le train avec les meilleurs params."""
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1, **params)
    model.fit(train_matrix[feature_cols], train_matrix["note"])
    return model


# --------------------------------------------------------------------------
# Sauvegarde / chargement (joblib)
# --------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Paquet auto-suffisant pour l'inference : modele + artefacts de features."""

    # "baseline_moyenne_globale" | "baseline_moyenne_par_film" | "svd" | "lightgbm"
    model_type: str
    model: Any
    context: FeatureContext
    metrics: dict[str, float]
    trained_at: str


def model_filename(model_type: str, version: str = MODEL_VERSION) -> str:
    """Construit le nom de fichier versionne : `<nom>_v<version>_<AAAAMMJJ>.joblib`."""
    today = date.today().strftime("%Y%m%d")
    return f"{model_type}_rating_v{version}_{today}.joblib"


def save_model(bundle: ModelBundle) -> Path:
    """Sauvegarde le bundle au format joblib versionne dans `ml/models/`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / model_filename(bundle.model_type)
    joblib.dump(bundle, path)
    return path


def load_model(path: str | Path) -> ModelBundle:
    """Charge un bundle sauvegarde par `save_model`."""
    return joblib.load(path)


def predict_note(bundle: ModelBundle, user_id: int, film_id: int) -> float:
    """Predit la note d'un couple (user_id, film_id) a partir d'un bundle charge.

    Reutilise exactement le pipeline de features du train (contexte serialise
    dans le bundle) : aucune fuite, aucun recalcul divergent entre
    entrainement et inference.
    """
    row = pd.DataFrame({"user_id": [user_id], "film_id": [film_id]})
    feature_row = build_matrix(row, bundle.context)

    if bundle.model_type == "svd":
        pred = bundle.model.predict(user_id, film_id)
    elif bundle.model_type == "lightgbm":
        pred = float(bundle.model.predict(feature_row[bundle.context.feature_cols])[0])
    elif bundle.model_type == "baseline_moyenne_globale":
        pred = bundle.context.global_mean
    elif bundle.model_type == "baseline_moyenne_par_film":
        pred = float(feature_row["film_mean_note"].iloc[0])
    else:
        raise ValueError(f"model_type inconnu : {bundle.model_type}")

    return float(features.clip_ratings(np.array([pred]))[0])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> None:
    """Pipeline complet : chargement, split, baselines, SVD, LightGBM, export."""
    print_section("Chargement des donnees Gold")
    notation_df, film_df = load_gold_data()
    print(f"{len(notation_df)} notations, {len(film_df)} films")

    print_section("Split train/test stratifie par utilisateur (80/20)")
    train_notation, test_notation = features.stratified_user_train_test_split(
        notation_df, test_size=0.2, random_state=42
    )
    print(f"train={len(train_notation)}  test={len(test_notation)}")

    context = build_feature_context(train_notation, film_df)
    train_matrix = build_matrix(train_notation, context)
    test_matrix = build_matrix(test_notation, context)

    print_section("Baselines naives")
    baseline_metrics = evaluate_baselines(test_matrix, context.global_mean)
    for name, metrics in baseline_metrics.items():
        print(f"  {name:<28} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
    reference_baseline_rmse = min(m["rmse"] for m in baseline_metrics.values())

    print_section("SVD biaise (factorisation matricielle)")
    svd_model = BiasedSVDRecommender(n_factors=20).fit(train_notation)
    svd_preds = features.clip_ratings(svd_model.predict_batch(test_notation))
    svd_metrics = {
        "rmse": features.rmse(test_notation["note"].to_numpy(), svd_preds),
        "mae": features.mae(test_notation["note"].to_numpy(), svd_preds),
    }
    print(f"  RMSE={svd_metrics['rmse']:.4f}  MAE={svd_metrics['mae']:.4f}")

    print_section("LightGBM : grid search 5-fold CV (stratifie par utilisateur)")
    best_params, cv_results = grid_search_lightgbm(train_matrix, context.feature_cols)
    print(cv_results.sort_values("cv_rmse_mean").to_string(index=False))
    print(f"\nMeilleurs parametres : {best_params}")

    lgbm_model = train_final_lightgbm(train_matrix, context.feature_cols, best_params)
    lgbm_preds = features.clip_ratings(
        lgbm_model.predict(test_matrix[context.feature_cols])
    )
    lgbm_metrics = {
        "rmse": features.rmse(test_matrix["note"].to_numpy(), lgbm_preds),
        "mae": features.mae(test_matrix["note"].to_numpy(), lgbm_preds),
    }
    print(
        f"\n  LightGBM (test) RMSE={lgbm_metrics['rmse']:.4f} "
        f" MAE={lgbm_metrics['mae']:.4f}"
    )

    print_section("Comparatif final (RMSE / MAE sur le test)")
    all_results = {
        **{k: v for k, v in baseline_metrics.items()},
        "svd": svd_metrics,
        "lightgbm": lgbm_metrics,
    }
    for name, metrics in sorted(all_results.items(), key=lambda kv: kv[1]["rmse"]):
        improvement = (
            100 * (reference_baseline_rmse - metrics["rmse"]) / reference_baseline_rmse
        )
        print(
            f"  {name:<28} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
            f"amelioration={improvement:+.1f}%"
        )

    best_name = min(("svd", "lightgbm"), key=lambda n: all_results[n]["rmse"])
    best_model = svd_model if best_name == "svd" else lgbm_model
    best_metrics = all_results[best_name]
    improvement = (
        reference_baseline_rmse - best_metrics["rmse"]
    ) / reference_baseline_rmse

    print_section("Verification des seuils")
    print(f"Modele retenu       : {best_name}")
    print(
        f"RMSE                : {best_metrics['rmse']:.4f}  (seuil < {RMSE_THRESHOLD})"
    )
    print(
        f"Amelioration vs baseline naive la plus forte : {100 * improvement:.1f}%  "
        f"(seuil >= {100 * MIN_IMPROVEMENT_VS_BASELINE:.0f}%)"
    )
    rmse_ok = best_metrics["rmse"] < RMSE_THRESHOLD
    improvement_ok = improvement >= MIN_IMPROVEMENT_VS_BASELINE
    print(f"Seuil RMSE respecte         : {rmse_ok}")
    print(f"Seuil amelioration respecte : {improvement_ok}")

    bundle = ModelBundle(
        model_type=best_name,
        model=best_model,
        context=context,
        metrics={**best_metrics, "improvement_vs_baseline": improvement},
        trained_at=date.today().isoformat(),
    )
    path = save_model(bundle)
    print_section("Export")
    print(f"Modele sauvegarde : {path}")


if __name__ == "__main__":
    main()
