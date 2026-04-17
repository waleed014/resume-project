"""Domain (category) classifier using sentence embeddings + Logistic Regression."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from .config import SETTINGS
from .embeddings import encode_one, encode_texts

logger = logging.getLogger(__name__)


def train(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train a Logistic Regression classifier on the prebuilt embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if not SETTINGS.embeddings_path.exists():
        raise FileNotFoundError("Embeddings not built. Run scripts/build_embeddings.py")

    X = np.load(SETTINGS.embeddings_path)
    with open(SETTINGS.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    y_raw = meta["categories"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Drop classes with fewer than 2 samples (stratify requirement).
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= 2]
    mask = np.isin(y, keep_classes)
    X, y = X[mask], y[mask]

    # Ensure test set is at least as large as #classes for stratify; otherwise drop stratify.
    n_classes = len(np.unique(y))
    eff_test = max(test_size, float(n_classes) / max(len(y), 1))
    eff_test = min(eff_test, 0.5)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=eff_test, random_state=random_state, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=eff_test, random_state=random_state
        )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    labels_used = np.unique(np.concatenate([y_te, y_pred]))
    report = classification_report(
        y_te,
        y_pred,
        labels=labels_used,
        target_names=[le.classes_[i] for i in labels_used],
        output_dict=True,
        zero_division=0,
    )

    SETTINGS.classifier_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, SETTINGS.classifier_path)
    joblib.dump(le, SETTINGS.label_encoder_path)
    logger.info(
        "Saved classifier to %s (accuracy=%.3f)",
        SETTINGS.classifier_path,
        report["accuracy"],
    )
    return report


def load() -> Tuple[object, object]:
    if not SETTINGS.classifier_path.exists():
        raise FileNotFoundError(
            "Classifier not trained. Run `python -m scripts.train_classifier`."
        )
    clf = joblib.load(SETTINGS.classifier_path)
    le = joblib.load(SETTINGS.label_encoder_path)
    return clf, le


def predict(text: str, top_k: int = 3) -> list[dict]:
    """Predict the top-k most likely domains for a resume text."""
    clf, le = load()
    vec = encode_one(text).reshape(1, -1)
    probs = clf.predict_proba(vec)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [
        {"category": str(le.inverse_transform([i])[0]), "probability": float(probs[i])}
        for i in order
    ]
