#!/usr/bin/env python3
"""
evaluate_model.py — Model Evaluation

Recreates the dataset (same seed as training), runs the trained model
on the validation set, and prints confusion matrix + classification report.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "fatigue_model.keras"

SIGNAL_FILES = [
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
]
WALKING_DOWNSTAIRS_LABEL = 3
BUFFER_SIZE = 128


def load_inertial_signals(split):
    signals = []
    for sig in SIGNAL_FILES:
        filepath = DATA_DIR / split / "Inertial Signals" / sig.format(split)
        signals.append(np.loadtxt(filepath))
    return np.stack(signals, axis=-1)


def load_labels(split):
    return np.loadtxt(DATA_DIR / split / f"y_{split}.txt", dtype=int)


def synthesize_fatigue(X_clean, rng):
    X_fat = X_clean.copy()
    n, steps, ch = X_fat.shape
    X_fat += rng.normal(0, 0.05, X_fat.shape)
    for i in range(n):
        for c in range(3):
            sig = X_fat[i, :, c]
            thresh = np.percentile(np.abs(sig), 90)
            mask = np.abs(sig) >= thresh
            sig[mask] *= rng.uniform(1.5, 3.0, size=np.sum(mask))
            X_fat[i, :, c] = sig
    return X_fat


def main():
    rng = np.random.default_rng(seed=42)

    # ── Recreate dataset (same as training) ──
    print("Loading data...")
    X_all = np.concatenate([load_inertial_signals("train"), load_inertial_signals("test")])
    y_all = np.concatenate([load_labels("train"), load_labels("test")])

    X_class0 = X_all[y_all == WALKING_DOWNSTAIRS_LABEL]
    X_class1 = synthesize_fatigue(X_class0, rng)

    X = np.concatenate([X_class0, X_class1])
    y = np.concatenate([np.zeros(len(X_class0)), np.ones(len(X_class1))])

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_val, y_val = X[split:], y[split:]

    # ── Load model ──
    print("Loading model...")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)

    # ── Predict ──
    print("Running predictions on validation set...\n")
    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob > 0.6).astype(int)
    y_true = y_val.astype(int)

    # ── Metrics ──
    print("=" * 50)
    print("  MODEL EVALUATION (Validation Set)")
    print("=" * 50)

    print(f"\n  Samples      : {len(y_true)}")
    print(f"  Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision    : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall       : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score     : {f1_score(y_true, y_pred):.4f}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Optimal  Fatigued")
    print(f"  Actual Optimal   {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"  Actual Fatigued  {cm[1][0]:>5}    {cm[1][1]:>5}")

    # ── Full classification report ──
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Optimal (0)", "Fatigued (1)"],
        digits=4,
    ))


if __name__ == "__main__":
    main()
