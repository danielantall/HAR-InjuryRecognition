#!/usr/bin/env python3
"""
debug_sensor.py — Logs all incoming HTTP requests from Sensor Logger.

Run this INSTEAD of server.py to inspect the raw JSON payloads
your phone is sending, so you can verify field names and format.

Usage:
    python debug_sensor.py
    Then point Sensor Logger HTTP Push to http://<your-ip>:5050/stream
"""

import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

packet_count = 0


@app.route("/stream", methods=["POST"])
def stream():
    global packet_count
    packet_count += 1

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    raw_body = request.get_data(as_text=True)

    # Header info
    print(f"\n{'='*70}")
    print(f"📦 Packet #{packet_count}  |  {timestamp}")
    print(f"{'='*70}")
    print(f"  Content-Type : {request.content_type}")
    print(f"  Content-Len  : {request.content_length}")

    # Parse JSON
    try:
        payload = request.get_json(force=True, silent=True)

        if payload is None:
            print(f"  ⚠️  Could not parse JSON")
            print(f"  Raw body: {raw_body[:500]}")
        elif isinstance(payload, list):
            print(f"  Format     : Array of {len(payload)} object(s)")
            for i, item in enumerate(payload):
                print(f"\n  --- Item [{i}] ---")
                _log_object(item, indent=4)
        else:
            print(f"  Format     : Single object")
            _log_object(payload, indent=4)

    except Exception as e:
        print(f"  ❌ Parse error: {e}")
        print(f"  Raw body: {raw_body[:500]}")

    return jsonify({"status": "ok"}), 200


def _log_object(obj, indent=2):
    """Pretty-print a dict with type annotations."""
    prefix = " " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            val_type = type(value).__name__
            if isinstance(value, dict):
                print(f"{prefix}{key} ({val_type}):")
                _log_object(value, indent + 4)
            elif isinstance(value, list):
                print(f"{prefix}{key} ({val_type}, len={len(value)}): {json.dumps(value[:3])}{'...' if len(value) > 3 else ''}")
            else:
                print(f"{prefix}{key} ({val_type}): {value}")
    else:
        print(f"{prefix}{obj}")


@app.route("/", methods=["GET"])
def index():
    return f"<h3>debug_sensor.py — {packet_count} packets received</h3>"


@app.route("/state", methods=["GET"])
def state():
    return jsonify({"packet_count": packet_count, "status": "debug_mode"})


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n🔍 Sensor Logger Debug Server")
    print(f"{'='*40}")
    print(f"📱 Point Sensor Logger to: http://{local_ip}:5143/stream")
    print(f"🖥️  Status page: http://localhost:5143")
    print(f"{'='*40}")
    print(f"Waiting for packets...\n")

    app.run(host="0.0.0.0", port=5143, debug=False, threaded=True)
