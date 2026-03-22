#!/usr/bin/env python3
"""
server.py — Real-Time Fatigue Detection Inference Server

Receives live sensor data from Sensor Logger (HTTP Push),
buffers 128 samples, runs the 1D-CNN, and serves a 3D dashboard.
"""

import math
import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BUFFER_SIZE = 128
PREDICTION_THRESHOLD = 0.6
MODEL_PATH = Path(__file__).parent / "model" / "fatigue_model.keras"

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# Global state — shared between request handler and frontend polling
state_lock = threading.Lock()
global_state = {
    "pitch": 0.0,
    "roll": 0.0,
    "yaw": 0.0,
    "qw": 1.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "has_orientation": False,
    "prediction": "Optimal",
    "confidence": 0.0,
    "buffer_fill": 0,
    "buffer_max": BUFFER_SIZE,
    "latest_gravity": (0.0, 0.0, 9.81), # default resting gravity
}

# Rolling buffer for sensor data: each item = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
sensor_buffer = deque(maxlen=BUFFER_SIZE)

# Separate buffer for debug charts (stores more history for visualization)
CHART_BUFFER_SIZE = 200
chart_buffer = deque(maxlen=CHART_BUFFER_SIZE)
orient_buffer = deque(maxlen=CHART_BUFFER_SIZE)  # [pitch, roll, yaw]

# Model loaded at startup
model = None


def load_model():
    """Load the trained 1D-CNN."""
    global model
    import tensorflow as tf
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")


# ─────────────────────────────────────────────
# ORIENTATION CALCULATION
# ─────────────────────────────────────────────

def compute_pitch_roll(acc_x: float, acc_y: float, acc_z: float):
    """
    Compute pitch and roll from raw accelerometer using atan2.
    Returns degrees.
    """
    pitch = math.atan2(acc_y, math.sqrt(acc_x ** 2 + acc_z ** 2))
    roll = math.atan2(-acc_x, acc_z)
    return math.degrees(pitch), math.degrees(roll)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference():
    """Run the 1D-CNN on the current buffer and update global state."""
    global global_state

    if model is None or len(sensor_buffer) < BUFFER_SIZE:
        return

    # Shape: (1, 128, 6)
    window_data = np.array(list(sensor_buffer))
    
    # Calculate standard deviation of body acceleration (first 3 columns)
    acc_std = np.std(window_data[:, :3], axis=0)
    mean_std = np.mean(acc_std)
    
    STATIONARY_THRESHOLD = 0.05  # threshold for stationary detection in 'g'
    if mean_std < STATIONARY_THRESHOLD:
        prob = 0.0
        label = "Optimal"
    else:
        window = window_data.reshape(1, BUFFER_SIZE, 6)
        prob = float(model.predict(window, verbose=0)[0][0])
        label = "Fatigued" if prob > PREDICTION_THRESHOLD else "Optimal"

    with state_lock:
        global_state["prediction"] = label
        global_state["confidence"] = round(prob, 4)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the 3D dashboard."""
    return render_template("index.html")


@app.route("/stream", methods=["POST"])
def stream():
    """
    Receive sensor data from Sensor Logger (HTTP Push).
    Handles many different JSON formats by scanning keys flexibly.
    """
    global _stream_log_count

    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            raw = request.get_data(as_text=True)
            print(f"⚠️  No JSON parsed. Raw body ({len(raw)} bytes): {raw[:300]}")
            return jsonify({"status": "error", "msg": "no JSON"}), 400

        # Log the first 5 payloads so the user can see the format
        if _stream_log_count < 5:
            _stream_log_count += 1
            import json
            print(f"\n📦 Payload #{_stream_log_count}:")
            print(json.dumps(payload, indent=2, default=str)[:1000])

        # Sensor Logger sends: {"messageId":..., "payload": [{name, time, values}, ...]}
        # Unwrap the outer wrapper if present
        if isinstance(payload, dict) and "payload" in payload and isinstance(payload["payload"], list):
            readings = payload["payload"]
        elif isinstance(payload, list):
            readings = payload
        else:
            readings = [payload]

        # Collect acc + gyro from this batch, then pair them
        batch_acc = None
        batch_gyro = None

        for reading in readings:
            if not isinstance(reading, dict):
                continue

            sensor_name = str(reading.get("name", reading.get("sensor", ""))).lower()

            # Skip uncalibrated sensors — use calibrated only
            if "uncalibrated" in sensor_name:
                continue

            # Get values sub-object
            values_obj = reading.get("values", reading.get("payload"))

            # --- Orientation / attitude data (quaternion or euler) ---
            if any(k in sensor_name for k in ("orient", "attitude", "rotation vector", "game rotation")):
                if isinstance(values_obj, dict):
                    _process_orientation(values_obj)
                continue

            # --- Accelerometer / Gyroscope ---
            values = _extract_xyz(values_obj) if isinstance(values_obj, dict) else None
            if values is None:
                values = _extract_xyz(reading)
            if values is None:
                continue

            if any(k in sensor_name for k in ("acc", "accelero", "linear", "gravity")):
                if "gravity" in sensor_name:
                    with state_lock:
                        global_state["latest_gravity"] = values
                    print(f"🌍 Gravity received: X={values[0]:.2f} | Y={values[1]:.2f} | Z={values[2]:.2f}")
                    continue
                else:
                    batch_acc = values
            elif any(k in sensor_name for k in ("gyro", "rotation", "angular")):
                batch_gyro = values

            # If we have both from this batch, emit a sample immediately
            if batch_acc and batch_gyro:
                _process_sample(*batch_acc, *batch_gyro)
                batch_acc = None
                batch_gyro = None

        # Handle leftovers via pending state (acc/gyro split across batches)
        if batch_acc and not batch_gyro:
            with state_lock:
                global_state["_pending_acc"] = batch_acc
                pending_gyro = global_state.pop("_pending_gyro", None)
            if pending_gyro:
                _process_sample(*batch_acc, *pending_gyro)
        elif batch_gyro and not batch_acc:
            with state_lock:
                global_state["_pending_gyro"] = batch_gyro
                pending_acc = global_state.pop("_pending_acc", None)
            if pending_acc:
                _process_sample(*pending_acc, *batch_gyro)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        print(f"❌ Error processing stream: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": str(e)}), 500


# Counter for debug logging
_stream_log_count = 0


def _process_orientation(values_obj):
    """Extract orientation quaternion or euler angles and update global state."""
    keys = {k.lower(): k for k in values_obj.keys()}

    with state_lock:
        global_state["has_orientation"] = True

        # Try quaternion first (qw, qx, qy, qz)
        qw = qx = qy = qz = None
        for k, orig in keys.items():
            try:
                v = float(values_obj[orig])
            except (TypeError, ValueError):
                continue
            if k in ("qw", "w"):
                qw = v
            elif k in ("qx", "x"):
                qx = v
            elif k in ("qy", "y"):
                qy = v
            elif k in ("qz", "z"):
                qz = v

        if qw is not None and qx is not None and qy is not None and qz is not None:
            global_state["qw"] = round(qw, 6)
            global_state["qx"] = round(qx, 6)
            global_state["qy"] = round(qy, 6)
            global_state["qz"] = round(qz, 6)
            # Also derive euler for the stats display
            import math
            # Roll (x), Pitch (y), Yaw (z) from quaternion
            sinr = 2 * (qw * qx + qy * qz)
            cosr = 1 - 2 * (qx * qx + qy * qy)
            roll = math.degrees(math.atan2(sinr, cosr))
            sinp = 2 * (qw * qy - qz * qx)
            sinp = max(-1, min(1, sinp))
            pitch = math.degrees(math.asin(sinp))
            siny = 2 * (qw * qz + qx * qy)
            cosy = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.degrees(math.atan2(siny, cosy))
            global_state["pitch"] = round(pitch, 2)
            global_state["roll"] = round(roll, 2)
            global_state["yaw"] = round(yaw, 2)
        orient_buffer.append([round(pitch, 2), round(roll, 2), round(yaw, 2)])
        return

        # Try euler angles (pitch, roll, yaw)
        for k, orig in keys.items():
            try:
                v = float(values_obj[orig])
            except (TypeError, ValueError):
                continue
            if "pitch" in k:
                global_state["pitch"] = round(math.degrees(v) if abs(v) < 10 else v, 2)
            elif "roll" in k:
                global_state["roll"] = round(math.degrees(v) if abs(v) < 10 else v, 2)
            elif "yaw" in k:
                global_state["yaw"] = round(math.degrees(v) if abs(v) < 10 else v, 2)
        orient_buffer.append([
            global_state.get("pitch", 0),
            global_state.get("roll", 0),
            global_state.get("yaw", 0),
        ])


def _extract_xyz(obj):
    """Extract (x, y, z) from a dict, trying multiple key conventions."""
    if not isinstance(obj, dict):
        return None
    keys = {k.lower(): k for k in obj.keys()}

    # Try simple x/y/z
    for x_key, y_key, z_key in [
        ("x", "y", "z"),
        ("xvalue", "yvalue", "zvalue"),
    ]:
        if x_key in keys and y_key in keys and z_key in keys:
            return (float(obj[keys[x_key]]), float(obj[keys[y_key]]), float(obj[keys[z_key]]))

    # Try any keys containing x, y, z
    x_val = y_val = z_val = None
    for k, orig in keys.items():
        v = obj[orig]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if "x" in k and x_val is None:
            x_val = fv
        elif "y" in k and y_val is None:
            y_val = fv
        elif "z" in k and z_val is None:
            z_val = fv
    if x_val is not None and y_val is not None and z_val is not None:
        return (x_val, y_val, z_val)

    return None


def _find_acc(obj):
    """Find accelerometer values in a flat dict."""
    keys = {k.lower(): k for k in obj.keys()}
    for prefix in ["accelerometeracceleration", "acc_", "accel"]:
        xk = next((k for k in keys if prefix in k and "x" in k), None)
        yk = next((k for k in keys if prefix in k and "y" in k), None)
        zk = next((k for k in keys if prefix in k and "z" in k), None)
        if xk and yk and zk:
            return (float(obj[keys[xk]]), float(obj[keys[yk]]), float(obj[keys[zk]]))
    return None


def _find_gyro(obj):
    """Find gyroscope values in a flat dict."""
    keys = {k.lower(): k for k in obj.keys()}
    for prefix in ["gyrorotation", "gyro_", "gyro"]:
        xk = next((k for k in keys if prefix in k and "x" in k), None)
        yk = next((k for k in keys if prefix in k and "y" in k), None)
        zk = next((k for k in keys if prefix in k and "z" in k), None)
        if xk and yk and zk:
            return (float(obj[keys[xk]]), float(obj[keys[yk]]), float(obj[keys[zk]]))
    return None


_ema_state = None

def _process_sample(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """Add a complete 6-channel sample to the buffer and update state."""
    global _ema_state
    
    with state_lock:
        grav = global_state.get("latest_gravity", (0.0, 0.0, 9.81))

    # Subtract gravity and convert to g
    body_acc_x = (acc_x - grav[0]) / 9.81
    body_acc_y = (acc_y - grav[1]) / 9.81
    body_acc_z = (acc_z - grav[2]) / 9.81

    # Apply EMA Smoothing
    alpha = 0.3
    if _ema_state is None:
        _ema_state = [body_acc_x, body_acc_y, body_acc_z, gyro_x, gyro_y, gyro_z]
    else:
        _ema_state[0] = alpha * body_acc_x + (1 - alpha) * _ema_state[0]
        _ema_state[1] = alpha * body_acc_y + (1 - alpha) * _ema_state[1]
        _ema_state[2] = alpha * body_acc_z + (1 - alpha) * _ema_state[2]
        _ema_state[3] = alpha * gyro_x + (1 - alpha) * _ema_state[3]
        _ema_state[4] = alpha * gyro_y + (1 - alpha) * _ema_state[4]
        _ema_state[5] = alpha * gyro_z + (1 - alpha) * _ema_state[5]

    sample = list(_ema_state)
    sensor_buffer.append(sample)
    chart_buffer.append(sample)

    with state_lock:
        # Only compute pitch/roll from atan2 if no orientation sensor
        if not global_state.get("has_orientation"):
            pitch, roll = compute_pitch_roll(acc_x, acc_y, acc_z)
            global_state["pitch"] = round(pitch, 2)
            global_state["roll"] = round(roll, 2)
        global_state["buffer_fill"] = len(sensor_buffer)

    # Run inference when buffer is full
    if len(sensor_buffer) == BUFFER_SIZE:
        run_inference()


@app.route("/state", methods=["GET"])
def get_state():
    """Return the current state for the frontend."""
    with state_lock:
        # Filter out internal keys
        public_state = {k: v for k, v in global_state.items() if not k.startswith("_")}
        return jsonify(public_state)


@app.route("/data", methods=["GET"])
def get_data():
    """Return recent raw sensor values for debug charts."""
    data = list(chart_buffer)
    odata = list(orient_buffer)
    return jsonify({
        "acc_x":  [d[0] for d in data],
        "acc_y":  [d[1] for d in data],
        "acc_z":  [d[2] for d in data],
        "gyro_x": [d[3] for d in data],
        "gyro_y": [d[4] for d in data],
        "gyro_z": [d[5] for d in data],
        "length": len(data),
        "orient_pitch": [d[0] for d in odata],
        "orient_roll":  [d[1] for d in odata],
        "orient_yaw":   [d[2] for d in odata],
        "orient_length": len(odata),
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    load_model()

    # Print local IP for easy phone configuration
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n🌐 Server starting on http://0.0.0.0:5143")
    print(f"📱 Point Sensor Logger HTTP Push to: http://{local_ip}:5143/stream")
    print(f"🖥️  Open dashboard at: http://localhost:5143\n")

    app.run(host="0.0.0.0", port=5143, debug=False, threaded=True)
