#!/usr/bin/env python3
"""
train_model.py — Data Pipeline & 1D-CNN for Fatigue Detection

Loads raw UCI HAR Inertial Signals (total_acc + body_gyro),
filters WALKING_DOWNSTAIRS as "optimal form" (Class 0),
synthesizes "fatigued" data (Class 1), trains a 1D-CNN,
and exports the model.
"""

import os
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"

SIGNAL_FILES = [
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
]

WALKING_DOWNSTAIRS_LABEL = 3  # from activity_labels.txt
TIMESTEPS = 128
N_FEATURES = 6


def load_signal_file(filepath: str) -> np.ndarray:
    """Load a single signal file. Each line = 128 space-separated floats."""
    return np.loadtxt(filepath)


def load_inertial_signals(split: str) -> np.ndarray:
    """
    Load all 6 signal channels for a given split ('train' or 'test').
    Returns shape (N, 128, 6).
    """
    signals = []
    for sig_template in SIGNAL_FILES:
        filename = sig_template.format(split)
        filepath = DATA_DIR / split / "Inertial Signals" / filename
        print(f"  Loading {filepath.name}...")
        data = load_signal_file(filepath)  # (N, 128)
        signals.append(data)

    # Stack channels: list of (N,128) → (N, 128, 6)
    return np.stack(signals, axis=-1)


def load_labels(split: str) -> np.ndarray:
    """Load activity labels for a given split."""
    filepath = DATA_DIR / split / f"y_{split}.txt"
    return np.loadtxt(filepath, dtype=int)


# ─────────────────────────────────────────────
# 2. FILTER WALKING_DOWNSTAIRS
# ─────────────────────────────────────────────

def extract_class0(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Keep only WALKING_DOWNSTAIRS windows → Class 0 (optimal form)."""
    mask = y == WALKING_DOWNSTAIRS_LABEL
    return X[mask]


# ─────────────────────────────────────────────
# 3. SYNTHESIZE CLASS 1 (FATIGUE)
# ─────────────────────────────────────────────

def synthesize_fatigue(X_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Duplicate Class 0 data and corrupt it to simulate fatigue:
      - Add high-frequency Gaussian noise (muscle tremor / instability)
      - Amplify top-K impact peaks in accelerometer channels (joint failure)
    """
    X_fatigued = X_clean.copy()
    n_samples, n_steps, n_channels = X_fatigued.shape

    # --- Inject instability: Gaussian noise on ALL channels ---
    noise = rng.normal(loc=0.0, scale=0.05, size=X_fatigued.shape)
    X_fatigued += noise

    # --- Simulate impact spikes: amplify peaks in acc channels (0,1,2) ---
    for i in range(n_samples):
        for ch in range(3):  # total_acc_x, y, z only
            signal = X_fatigued[i, :, ch]
            # Find the top 10% peak indices
            threshold = np.percentile(np.abs(signal), 90)
            peak_mask = np.abs(signal) >= threshold
            # Multiply peaks by a random factor in [1.5, 3.0]
            n_peaks = np.sum(peak_mask)
            factors = rng.uniform(1.5, 3.0, size=n_peaks)
            signal[peak_mask] *= factors
            X_fatigued[i, :, ch] = signal

    return X_fatigued


# ─────────────────────────────────────────────
# 4. BUILD 1D-CNN
# ─────────────────────────────────────────────

def build_model():
    """
    1D-CNN architecture:
      Conv1D(64) → MaxPool → Conv1D(128) → MaxPool
      → Flatten → Dense(64, dropout=0.5) → Sigmoid
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # Block 1: micro-patterns (sharp spikes, tremors)
        layers.Conv1D(64, kernel_size=5, activation="relu",
                      input_shape=(TIMESTEPS, N_FEATURES)),
        layers.MaxPooling1D(pool_size=2),

        # Block 2: macro-patterns (stride rhythm, impact sequences)
        layers.Conv1D(128, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),

        # Classification head
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────────────────────────
# 5. MAIN — TRAIN & EXPORT
# ─────────────────────────────────────────────

def main():
    rng = np.random.default_rng(seed=42)

    # --- Load raw signals ---
    print("\n📡 Loading UCI HAR Inertial Signals...")
    print("  [train]")
    X_train_raw = load_inertial_signals("train")
    y_train_raw = load_labels("train")
    print("  [test]")
    X_test_raw = load_inertial_signals("test")
    y_test_raw = load_labels("test")

    # --- Merge train+test, then filter WALKING_DOWNSTAIRS ---
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
    print(f"\n📊 Total windows loaded: {X_all.shape[0]}")

    X_class0 = extract_class0(X_all, y_all)
    print(f"🚶 WALKING_DOWNSTAIRS windows (Class 0): {X_class0.shape[0]}")

    # --- Synthesize fatigued data ---
    print("⚡ Synthesizing fatigue corruption (Class 1)...")
    X_class1 = synthesize_fatigue(X_class0, rng)

    # --- Combine & shuffle ---
    X = np.concatenate([X_class0, X_class1], axis=0)
    y = np.concatenate([
        np.zeros(len(X_class0)),   # Class 0: optimal
        np.ones(len(X_class1)),    # Class 1: fatigued
    ])

    shuffle_idx = rng.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]
    print(f"🔀 Combined dataset: {X.shape[0]} windows  "
          f"(Class 0: {int(np.sum(y == 0))}, Class 1: {int(np.sum(y == 1))})")

    # --- 80/20 split ---
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"✂️  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # --- Build & train ---
    print("\n🧠 Building 1D-CNN...")
    model = build_model()
    model.summary()

    print("\n🏋️ Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1,
    )

    # --- Evaluate ---
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ Final validation — Loss: {val_loss:.4f}  Accuracy: {val_acc:.4f}")

    # --- Export ---
    MODEL_DIR.mkdir(exist_ok=True)
    export_path = MODEL_DIR / "fatigue_model.keras"
    model.save(export_path)
    print(f"💾 Model exported to {export_path}")


if __name__ == "__main__":
    main()
