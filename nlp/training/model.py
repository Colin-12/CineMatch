"""Entrainement du modele de classification de sentiment (endpoint /sentiment).

Structure miroir de ``ml/training/model.py`` : chargement Gold (psycopg2),
baselines (naive puis classique), entrainement du modele final (fine-tuning
DistilBERT via Hugging Face Trainer), sauvegarde/chargement versionnes.

Seuils cibles (voir CLAUDE.md) : F1 (macro) > 0.70 et accuracy > 0.72.

Label de sentiment (voir docstring de ``nlp/training/features.py`` pour
l'historique complet) : derive de ``avis.note_auteur`` (0-10, la vraie note
laissee par l'auteur de l'avis TMDB, `author_details.rating`, ~94.6% de
couverture) ; les avis sans ``note_auteur`` sont exclus. Une ANCIENNE approche
(abandonnee), qui derivait le label de la note MovieLens de l'utilisateur
synthetiquement rattache a l'avis (rattachement documente dans
``pipeline/transform_silver.py::clean_avis``), est conservee dans
``features.derive_label_from_note_movielens_legacy`` /
``features.add_sentiment_labels_movielens_legacy`` a titre de comparaison
historique documentee dans le notebook, mais n'est plus utilisee ici.

Usage :
    python -m nlp.training.model
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.training import features

load_dotenv()

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_VERSION = "1.1"  # v1.1 : label derive de note_auteur (remplace la note MovieLens)
F1_MACRO_THRESHOLD = 0.70
ACCURACY_THRESHOLD = 0.72

PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
SMOKE_TEST_N_ROWS = 100
RANDOM_STATE = 42


def print_section(title: str) -> None:
    """Affiche un titre de section lisible dans la console."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_gold_avis_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les tables Gold `avis` (avec `note_auteur`) et `notation`.

    `notation` n'est plus necessaire pour deriver le label (voir
    `build_labeled_dataset`) ; elle reste chargee et retournee pour permettre
    la comparaison historique documentee dans le notebook avec l'ancienne
    approche (`features.add_sentiment_labels_movielens_legacy`).
    """
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, film_id, texte, timestamp, note_auteur FROM avis"
            )
            avis_rows = cur.fetchall()
            avis_df = pd.DataFrame(
                avis_rows,
                columns=["user_id", "film_id", "texte", "timestamp", "note_auteur"],
            )

            cur.execute("SELECT user_id, film_id, note FROM notation")
            notation_rows = cur.fetchall()
            notation_df = pd.DataFrame(
                notation_rows, columns=["user_id", "film_id", "note"]
            )
    finally:
        conn.close()
    return avis_df, notation_df


def build_labeled_dataset(avis_df: pd.DataFrame) -> pd.DataFrame:
    """Derivation du label (depuis `note_auteur`) + longueur du texte.

    Approche retenue : voir `features.add_sentiment_labels`. Exclut les avis
    sans `note_auteur` (pas de fallback sur l'ancien label bruite).
    """
    labeled = features.add_sentiment_labels(avis_df)
    labeled = features.compute_text_length(labeled)
    return labeled


def build_labeled_dataset_legacy_movielens(
    avis_df: pd.DataFrame, notation_df: pd.DataFrame
) -> pd.DataFrame:
    """ANCIENNE approche (abandonnee) : label derive de la note MovieLens de
    l'utilisateur synthetiquement rattache. Conservee pour comparaison
    historique documentee dans le notebook -- voir `build_labeled_dataset`
    pour l'approche retenue en production."""
    labeled = features.add_sentiment_labels_movielens_legacy(avis_df, notation_df)
    labeled = features.compute_text_length(labeled)
    return labeled


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def evaluate_baseline_majority(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, float]:
    """Baseline naive : predit toujours la classe majoritaire du train."""
    majority_label = train_df["label"].value_counts().idxmax()
    y_pred = features.predict_baseline_majority(len(test_df), majority_label)
    y_true = test_df["label"].to_numpy()
    return {
        "accuracy": features.accuracy(y_true, y_pred),
        "macro_f1": features.macro_f1(y_true, y_pred),
        "weighted_f1": features.weighted_f1(y_true, y_pred),
    }


def train_tfidf_logreg(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    """Baseline classique : TF-IDF (uni+bigrammes) + regression logistique.

    class_weight="balanced" pour compenser le desequilibre des classes
    (negatif/neutre sous-representes face a positif).
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    pipeline.fit(train_df["texte"], train_df["label"])
    y_pred = pipeline.predict(test_df["texte"])
    y_true = test_df["label"].to_numpy()
    metrics = {
        "accuracy": features.accuracy(y_true, y_pred),
        "macro_f1": features.macro_f1(y_true, y_pred),
        "weighted_f1": features.weighted_f1(y_true, y_pred),
    }
    return pipeline, metrics, y_pred


# ---------------------------------------------------------------------------
# Fine-tuning DistilBERT (Hugging Face)
# ---------------------------------------------------------------------------


def _build_hf_dataset(df: pd.DataFrame, tokenizer: Any) -> Any:
    """Construit un datasets.Dataset tokenise a partir d'un DataFrame labelise."""
    from datasets import Dataset

    frame = pd.DataFrame(
        {
            "texte": df["texte"].tolist(),
            "label": features.encode_labels(df["label"]).tolist(),
        }
    )
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def _tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["texte"], truncation=True, max_length=MAX_LENGTH, padding=False
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=["texte"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def _compute_metrics_for_trainer(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": features.accuracy(labels, preds),
        "macro_f1": features.macro_f1(labels, preds, labels=[0, 1, 2]),
        "weighted_f1": features.weighted_f1(labels, preds, labels=[0, 1, 2]),
    }


def _make_weighted_trainer_class(class_weights: np.ndarray) -> type:
    """Cree une sous-classe Trainer avec CrossEntropyLoss ponderee par classe.

    Necessaire car les classes negatif/neutre sont minoritaires face a
    positif (voir EDA) ; sans ponderation le modele tend a sur-predire
    la classe majoritaire.
    """
    import torch
    from transformers import Trainer

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=weights_tensor.to(logits.device)
            )
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def train_distilbert(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    num_train_epochs: float = 2.0,
    per_device_batch_size: int = 16,
    learning_rate: float = 5e-5,
    seed: int = RANDOM_STATE,
) -> tuple[Any, Any, dict[str, float], np.ndarray]:
    """Fine-tune DistilBERT pour la classification 3-classes de sentiment.

    num_train_epochs=2 et batch=16 par defaut : compromis assume vis-a-vis du
    budget de calcul CPU (pas de GPU disponible). Un calibrage prealable a
    mesure ~1.3s/exemple/epoque (max_length=256, batch=16) ; 2 epoques sur les
    ~1665 exemples d'entrainement representent deja ~75-90 minutes CPU, un
    budget juge raisonnable pour cet exercice (voir notebook, section
    limites). Retourne (model, tokenizer, metrics, y_pred_labels_texte).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=len(features.LABELS),
        id2label=features.ID2LABEL,
        label2id=features.LABEL2ID,
    )

    train_dataset = _build_hf_dataset(train_df, tokenizer)
    test_dataset = _build_hf_dataset(test_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_counts = train_df["label"].map(features.LABEL2ID).value_counts().sort_index()
    class_counts = class_counts.reindex(range(len(features.LABELS))).fillna(1)
    class_weights = (
        class_counts.sum() / (len(features.LABELS) * class_counts)
    ).to_numpy()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=seed,
        report_to=[],
        disable_tqdm=False,
    )

    trainer_class = _make_weighted_trainer_class(class_weights)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_for_trainer,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    raw_predictions = trainer.predict(test_dataset)
    y_pred_ids = np.argmax(raw_predictions.predictions, axis=-1)
    y_pred_labels = np.array(features.decode_labels(y_pred_ids), dtype=object)

    metrics = {
        "accuracy": eval_metrics["eval_accuracy"],
        "macro_f1": eval_metrics["eval_macro_f1"],
        "weighted_f1": eval_metrics["eval_weighted_f1"],
    }
    return model, tokenizer, metrics, y_pred_labels


def run_smoke_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validation rapide (~100 exemples, 1 epoque) avant l'entrainement complet.

    Objectif : detecter tot un probleme de plomberie (tokenisation, forme des
    tenseurs, API Trainer) sans attendre le run complet (~20-30 min CPU).
    """
    print_section("SMOKE TEST (~100 exemples, 1 epoque)")
    small_train = train_df.sample(
        n=min(SMOKE_TEST_N_ROWS, len(train_df)), random_state=RANDOM_STATE
    )
    small_test = test_df.sample(
        n=min(SMOKE_TEST_N_ROWS // 2, len(test_df)), random_state=RANDOM_STATE
    )
    checkpoint_dir = MODELS_DIR / "_smoke_test_checkpoints"
    start = time.time()
    _, _, metrics, _ = train_distilbert(
        small_train,
        small_test,
        output_dir=checkpoint_dir,
        num_train_epochs=1.0,
        per_device_batch_size=8,
    )
    elapsed = time.time() - start
    print(f"Smoke test termine en {elapsed:.1f}s.")
    print(f"Metriques (non representatives, {SMOKE_TEST_N_ROWS} exemples) : {metrics}")
    print("Plomberie validee, poursuite vers l'entrainement complet.")


# ---------------------------------------------------------------------------
# Bundle, sauvegarde, chargement, inference
# ---------------------------------------------------------------------------


@dataclass
class SentimentModelBundle:
    """Bundle auto-suffisant pour l'inference (mirroir de ModelBundle ml/)."""

    model_type: str  # "baseline_majoritaire" | "tfidf_logreg" | "distilbert"
    sklearn_model: Any = None
    hf_model_dir: str | None = None
    label2id: dict[str, int] = field(default_factory=lambda: dict(features.LABEL2ID))
    id2label: dict[int, str] = field(default_factory=lambda: dict(features.ID2LABEL))
    max_length: int = MAX_LENGTH
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""


def model_filename(model_type: str, version: str = MODEL_VERSION) -> str:
    """Nom de fichier/dossier versionne : <nom>_sentiment_v<version>_<AAAAMMJJ>."""
    today = date.today().strftime("%Y%m%d")
    return f"{model_type}_sentiment_v{version}_{today}"


def save_model(
    bundle: SentimentModelBundle, hf_model: Any = None, hf_tokenizer: Any = None
) -> Path:
    """Sauvegarde le bundle. Format HF (dossier) si distilbert, sinon joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = model_filename(bundle.model_type)

    if bundle.model_type == "distilbert":
        if hf_model is None or hf_tokenizer is None:
            raise ValueError(
                "hf_model et hf_tokenizer requis pour sauvegarder distilbert"
            )
        model_dir = MODELS_DIR / base_name
        model_dir.mkdir(parents=True, exist_ok=True)
        hf_model.save_pretrained(model_dir)
        hf_tokenizer.save_pretrained(model_dir)
        bundle.hf_model_dir = str(model_dir)
        joblib.dump(bundle, model_dir / "bundle_meta.joblib")
        return model_dir

    path = MODELS_DIR / f"{base_name}.joblib"
    joblib.dump(bundle, path)
    return path


def load_model(path: Path) -> SentimentModelBundle:
    """Charge un bundle sauvegarde (dossier HF ou fichier joblib)."""
    path = Path(path)
    if path.is_dir():
        bundle: SentimentModelBundle = joblib.load(path / "bundle_meta.joblib")
        return bundle
    return joblib.load(path)


def evaluate_saved_distilbert(
    bundle: SentimentModelBundle, test_df: pd.DataFrame
) -> tuple[dict[str, float], np.ndarray]:
    """Recharge un DistilBERT deja fine-tune et l'evalue sur test_df.

    Ne relance PAS d'entrainement : sert a reutiliser dans le notebook le
    modele deja entraine par ``python -m nlp.training.model`` (fine-tuning
    complet ~80-90 min CPU), pour garder le notebook executable en quelques
    minutes tout en analysant les vraies predictions du modele retenu.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(bundle.hf_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bundle.hf_model_dir)
    test_dataset = _build_hf_dataset(test_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(MODELS_DIR / "_eval_tmp"),
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_for_trainer,
    )
    raw_predictions = trainer.predict(test_dataset)
    y_pred_ids = np.argmax(raw_predictions.predictions, axis=-1)
    y_pred_labels = np.array(features.decode_labels(y_pred_ids), dtype=object)
    metrics = {
        "accuracy": features.accuracy(raw_predictions.label_ids, y_pred_ids),
        "macro_f1": features.macro_f1(
            raw_predictions.label_ids, y_pred_ids, labels=[0, 1, 2]
        ),
        "weighted_f1": features.weighted_f1(
            raw_predictions.label_ids, y_pred_ids, labels=[0, 1, 2]
        ),
    }
    return metrics, y_pred_labels


def predict_sentiment(bundle: SentimentModelBundle, texte: str) -> tuple[str, float]:
    """Predit (label, score_confiance) pour un texte d'avis donne."""
    if bundle.model_type == "distilbert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(bundle.hf_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(bundle.hf_model_dir)
        model.eval()
        inputs = tokenizer(
            texte, truncation=True, max_length=bundle.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        label_id = int(np.argmax(probs))
        return bundle.id2label[label_id], float(
            features.clip_probabilities(probs[label_id])
        )

    if bundle.model_type == "tfidf_logreg":
        proba = bundle.sklearn_model.predict_proba([texte])[0]
        classes = bundle.sklearn_model.classes_
        best_idx = int(np.argmax(proba))
        return str(classes[best_idx]), float(
            features.clip_probabilities(proba[best_idx])
        )

    # baseline_majoritaire : pas de notion de confiance graduee
    majority_label = bundle.metrics.get("majority_label", "positif")
    return majority_label, 1.0


# ---------------------------------------------------------------------------
# Orchestration complete
# ---------------------------------------------------------------------------


def save_tfidf_reference_model(
    tfidf_model: Pipeline, tfidf_metrics: dict[str, float]
) -> Path:
    """Sauvegarde TF-IDF+LogReg comme modele de reference pour `/sentiment`.

    Utilise le temps que le fine-tuning DistilBERT complet (~60-90+ min CPU)
    tourne : `api/routers/sentiment.py` charge ce bundle en attendant, puis
    bascule sur le bundle `distilbert` une fois celui-ci disponible et
    verifie superieur (voir model card `tfidf_logreg_sentiment_v*`).
    """
    bundle = SentimentModelBundle(
        model_type="tfidf_logreg",
        sklearn_model=tfidf_model,
        metrics=tfidf_metrics,
        trained_at=date.today().isoformat(),
    )
    return save_model(bundle)


def main(tfidf_only: bool = False) -> None:
    print_section("CHARGEMENT DES DONNEES GOLD (avis + notation)")
    avis_df, _notation_df = load_gold_avis_data()
    n_avant_filtre = len(avis_df)
    labeled_df = build_labeled_dataset(avis_df)
    n_exclus = n_avant_filtre - len(labeled_df)
    print(
        f"Avis labelises (note_auteur) : {len(labeled_df)} "
        f"({n_exclus} exclus sur {n_avant_filtre}, sans note_auteur)"
    )
    print(features.label_distribution(labeled_df))

    train_df, test_df = features.stratified_label_train_test_split(
        labeled_df, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train : {len(train_df)}  Test : {len(test_df)}")

    print_section("BASELINE NAIVE (classe majoritaire)")
    baseline_metrics = evaluate_baseline_majority(train_df, test_df)
    print(baseline_metrics)

    print_section("BASELINE CLASSIQUE (TF-IDF + regression logistique)")
    tfidf_model, tfidf_metrics, _ = train_tfidf_logreg(train_df, test_df)
    print(tfidf_metrics)

    if tfidf_only:
        print_section("SAUVEGARDE DU MODELE DE REFERENCE (TF-IDF+LogReg)")
        saved_path = save_tfidf_reference_model(tfidf_model, tfidf_metrics)
        print(f"Modele de reference sauvegarde : {saved_path}")
        print(
            "Fine-tuning DistilBERT non lance (--tfidf-only) : "
            "executez `python -m nlp.training.model` (sans l'option) pour "
            "le modele final."
        )
        return

    run_smoke_test(train_df, test_df)

    print_section("ENTRAINEMENT COMPLET (DistilBERT fine-tune)")
    checkpoint_dir = MODELS_DIR / "_distilbert_checkpoints"
    hf_model, hf_tokenizer, distilbert_metrics, _ = train_distilbert(
        train_df, test_df, output_dir=checkpoint_dir
    )
    print(distilbert_metrics)

    print_section("COMPARAISON DES MODELES")
    comparison = pd.DataFrame(
        [
            {"modele": "baseline_majoritaire", **baseline_metrics},
            {"modele": "tfidf_logreg", **tfidf_metrics},
            {"modele": "distilbert", **distilbert_metrics},
        ]
    )
    print(comparison.to_string(index=False))

    print_section("VERIFICATION DES SEUILS (DistilBERT)")
    f1 = distilbert_metrics["macro_f1"]
    acc = distilbert_metrics["accuracy"]
    f1_ok = f1 > F1_MACRO_THRESHOLD
    acc_ok = acc > ACCURACY_THRESHOLD
    print(f"F1 macro > {F1_MACRO_THRESHOLD} : {f1_ok} ({f1:.4f})")
    print(f"Accuracy > {ACCURACY_THRESHOLD} : {acc_ok} ({acc:.4f})")

    print_section("SAUVEGARDE DU MODELE FINAL (DistilBERT)")
    bundle = SentimentModelBundle(
        model_type="distilbert",
        metrics=distilbert_metrics,
        trained_at=date.today().isoformat(),
    )
    saved_path = save_model(bundle, hf_model=hf_model, hf_tokenizer=hf_tokenizer)
    print(f"Modele sauvegarde : {saved_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tfidf-only",
        action="store_true",
        help=(
            "N'entraine/sauvegarde que la baseline TF-IDF+LogReg (quelques "
            "secondes), sans lancer le fine-tuning DistilBERT complet. Utile "
            "pour publier un modele de reference pendant qu'un entrainement "
            "DistilBERT tourne deja en parallele."
        ),
    )
    args = parser.parse_args()
    main(tfidf_only=args.tfidf_only)
