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
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
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
    mask = np.isin(y, [1, 2, 3])  # 1: WALKING, 2: WALKING_UPSTAIRS, 3: WALKING_DOWNSTAIRS
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
    noise = rng.normal(loc=0.0, scale=0.1, size=X_fatigued.shape)
    X_fatigued += noise

    # --- Simulate impact spikes: amplify peaks in acc channels (0,1,2) ---
    for i in range(n_samples):
        for ch in range(3):  # body_acc_x, y, z only
            signal = X_fatigued[i, :, ch]
            # Find the top 10% peak indices
            threshold = np.percentile(np.abs(signal), 90)
            peak_mask = np.abs(signal) >= threshold
            # Multiply peaks by a random factor in [2.0, 4.0]
            n_peaks = np.sum(peak_mask)
            factors = rng.uniform(2.0, 4.0, size=n_peaks)
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
      → Flatten → Dense(64, dropout=0.6) → Sigmoid
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2

    model = keras.Sequential([
        # Block 1: micro-patterns (sharp spikes, tremors)
        layers.Conv1D(64, kernel_size=5, activation="relu",
                      kernel_regularizer=l2(0.001),
                      input_shape=(TIMESTEPS, N_FEATURES)),
        layers.MaxPooling1D(pool_size=2),

        # Block 2: macro-patterns (stride rhythm, impact sequences)
        layers.Conv1D(128, kernel_size=5, activation="relu",
                      kernel_regularizer=l2(0.001)),
        layers.MaxPooling1D(pool_size=2),

        # Classification head
        layers.Flatten(),
        layers.Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        layers.Dropout(0.6),
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

    # --- Build TRAIN set from UCI HAR train split only (no leakage) ---
    print(f"\n📊 Train split windows: {X_train_raw.shape[0]}")
    rng_train = np.random.default_rng(seed=42)

    X_class0_train = extract_class0(X_train_raw, y_train_raw)
    print(f"🚶 Dynamic-activity windows — Train (Class 0): {X_class0_train.shape[0]}")

    print("⚡ Synthesizing fatigue corruption for training set (Class 1)...")
    X_class1_train = synthesize_fatigue(X_class0_train, rng_train)

    X_train_all = np.concatenate([X_class0_train, X_class1_train], axis=0)
    y_train_all = np.concatenate([
        np.zeros(len(X_class0_train)),
        np.ones(len(X_class1_train)),
    ])
    shuffle_train = rng_train.permutation(len(X_train_all))
    X_train, y_train = X_train_all[shuffle_train], y_train_all[shuffle_train]
    print(f"🔀 Train set: {X_train.shape[0]} windows  "
          f"(Class 0: {int(np.sum(y_train == 0))}, Class 1: {int(np.sum(y_train == 1))})")

    # --- Build VAL/TEST set from UCI HAR test split only (no leakage) ---
    print(f"\n📊 Test split windows: {X_test_raw.shape[0]}")
    rng_test = np.random.default_rng(seed=99)

    X_class0_test = extract_class0(X_test_raw, y_test_raw)
    print(f"🚶 Dynamic-activity windows — Test (Class 0): {X_class0_test.shape[0]}")

    print("⚡ Synthesizing fatigue corruption for test set (Class 1)...")
    X_class1_test = synthesize_fatigue(X_class0_test, rng_test)

    X_test_all = np.concatenate([X_class0_test, X_class1_test], axis=0)
    y_test_all = np.concatenate([
        np.zeros(len(X_class0_test)),
        np.ones(len(X_class1_test)),
    ])
    shuffle_test = rng_test.permutation(len(X_test_all))
    X_val, y_val = X_test_all[shuffle_test], y_test_all[shuffle_test]
    print(f"🔀 Val/Test set: {X_val.shape[0]} windows  "
          f"(Class 0: {int(np.sum(y_val == 0))}, Class 1: {int(np.sum(y_val == 1))})")

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
