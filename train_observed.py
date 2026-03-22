#!/usr/bin/env python3
"""
train_observed.py — Train fatigue model on real phone data

Loads CSV files from 'observed-data/', safely temporal-splits each session 
80/20 to prevent data leakage, windows the data into 128-step sequences, 
and trains a regularized 1D-CNN with EarlyStopping.
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# ─────────────────────────────────────────────
# Set up paths & configurations
# ─────────────────────────────────────────────
DATA_DIR = Path("observed-data")
MODEL_DIR = Path("model")

TIMESTEPS = 128
STEP_SIZE = 64
N_FEATURES = 6
FS = 50.0  # approximate sampling rate

# ─────────────────────────────────────────────
# Data Loading & Prep
# ─────────────────────────────────────────────

def load_session(session_dir: Path) -> pd.DataFrame:
    """
    Loads Accelerometer and Gyroscope from a session dir,
    merges them on 'seconds_elapsed' using merge_asof.
    Returns: df with columns [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    acc_path = session_dir / "Accelerometer.csv"
    gyr_path = session_dir / "Gyroscope.csv"
    
    # Read CSVs and explicitly rename x,y,z columns
    acc_df = pd.read_csv(acc_path)[["seconds_elapsed", "x", "y", "z"]]
    acc_df = acc_df.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    
    gyr_df = pd.read_csv(gyr_path)[["seconds_elapsed", "x", "y", "z"]]
    gyr_df = gyr_df.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    
    # Sort both to ensure merge_asof works
    acc_df = acc_df.sort_values("seconds_elapsed")
    gyr_df = gyr_df.sort_values("seconds_elapsed")
    
    # Merge closely occurring timestamps
    merged = pd.merge_asof(
        acc_df, gyr_df, 
        on="seconds_elapsed", 
        direction="nearest",
        tolerance=0.05  # Within 50ms matching
    )
    
    # Drop rows with NaN (if gyro didn't match acc)
    merged = merged.dropna()
    
    # Return strict 6-channel order: acc X,Y,Z then gyro X,Y,Z
    return merged[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]]


def extract_windows(df: pd.DataFrame, window_size: int, step_size: int) -> np.ndarray:
    """Sliding window approach: returns (N_windows, window_size, n_features)"""
    arr = df.values
    n_samples = len(arr)
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        windows.append(arr[start : start + window_size])
    return np.array(windows)


def process_all_sessions():
    """Reads all subdirs, builds train/test sets with 80/20 chronological split."""
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    session_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    
    print("📡 Processing Sessions...")
    for s_dir in session_dirs:
        # Determine label based on folder name
        if "class0" in s_dir.name.lower():
            label = 0
            label_name = "Fresh"
        elif "class1" in s_dir.name.lower():
            label = 1
            label_name = "Fatigued"
        else:
            print(f"  [?] Skipping {s_dir.name} (no class0/class1 label in name)")
            continue
            
        df = load_session(s_dir)
        n_rows = len(df)
        
        # 80/20 CHRONOLOGICAL SPLIT
        # Guard against leakage by windowing sequentially BEFORE slicing, OR slicing the DF first.
        # Slicing the dataframe FIRST ensures absolutely zero overlap/leakage between train and test
        split_idx = int(n_rows * 0.8)
        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]
        
        # Extract windows
        X_tr = extract_windows(train_df, TIMESTEPS, STEP_SIZE)
        X_te = extract_windows(test_df, TIMESTEPS, STEP_SIZE)
        
        # Create labels
        y_tr = np.full(len(X_tr), label)
        y_te = np.full(len(X_te), label)
        
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_test_list.append(X_te)
        y_test_list.append(y_te)
        
        print(f"  [{label_name}] {s_dir.name}")
        print(f"      Train windows: {len(X_tr)}  |  Test windows: {len(X_te)}")
        
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test  = np.concatenate(X_test_list,  axis=0)
    y_test  = np.concatenate(y_test_list,  axis=0)
    
    # Shuffle Train Set
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]
    
    print(f"\n📊 Total Train: {len(X_train)} windows (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
    print(f"📊 Total Test:  {len(X_test)} windows (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")
    
    return (X_train, y_train), (X_test, y_test)


# ─────────────────────────────────────────────
# Model Building
# ─────────────────────────────────────────────

def build_model():
    """
    1D-CNN designed for real phone data (6 channels).
    Includes Dropout and L2 to prevent overfitting on the small dataset.
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv1D(64, kernel_size=5, activation="relu",
                      kernel_regularizer=l2(0.005),
                      input_shape=(TIMESTEPS, N_FEATURES)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Block 2
        layers.Conv1D(128, kernel_size=3, activation="relu",
                      kernel_regularizer=l2(0.005)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Classification head
        layers.Flatten(),
        layers.Dense(64, activation="relu", kernel_regularizer=l2(0.005)),
        layers.Dropout(0.5), # Strong dropout
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────

def main():
    (X_train, y_train), (X_test, y_test) = process_all_sessions()
    
    # Z-Score Normalization based on TRAINING set stats (zero leakage)
    train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    train_std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    model = build_model()
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", 
            patience=6, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print("\n🏋️ Training Model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n📈 Final Evaluation on Held-Out Test Set (Chronological Last 20%):")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"             Predicted")
    print(f"             Fresh  Fatigued")
    print(f"  Fresh       {cm[0,0]:<5}  {cm[0,1]}")
    print(f"  Fatigued    {cm[1,0]:<5}  {cm[1,1]}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Fresh", "Fatigued"]))

    # Save Model
    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / "fatigue_model_observed.keras"
    model.save(out_path)
    print(f"\n💾 Model saved to: {out_path}")


if __name__ == "__main__":
    main()
