"""
CENTRAL MONITORING DASHBOARD  (head-office / cloud side)
========================================================
Receives compact prediction summaries, replenishment alerts and heartbeats from
store edge nodes, via:
  * HTTP  : POST /api/ingest           (default)
  * MQTT  : subscribes retail/+/#      (when COMM_MODE=mqtt or both)

  Dashboard UI ... http://localhost:8050
Env: DASH_PORT=8050  COMM_MODE=http|mqtt|both  MQTT_HOST=localhost  MQTT_PORT=1883
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.getenv("DASH_PORT", "8050"))
COMM_MODE = os.getenv("COMM_MODE", "http").lower()
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DB_PATH = os.getenv("DASH_DB", str(ROOT / "dashboard" / "central.db"))

lock = threading.RLock()
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, received TEXT, channel TEXT,
    node_id TEXT, topic TEXT, store_id TEXT, product_id TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS latest(store_id TEXT, product_id TEXT, payload TEXT, received TEXT,
    PRIMARY KEY(store_id, product_id));
CREATE TABLE IF NOT EXISTS nodes(node_id TEXT PRIMARY KEY, last_seen REAL, payload TEXT, channel TEXT);
""")
db.commit()
mqtt_state = {"enabled": COMM_MODE in ("mqtt", "both"), "connected": False, "messages": 0}


def store_event(envelope: dict, channel: str):
    topic = envelope.get("topic")
    p = envelope.get("payload") or {}
    node = envelope.get("node_id", "unknown")
    now = datetime.now().isoformat(timespec="seconds")
    with lock:
        if topic == "heartbeat":
            db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?)", (node, time.time(), json.dumps(p), channel))
        else:
            db.execute("INSERT INTO events(received,channel,node_id,topic,store_id,product_id,payload) "
                       "VALUES (?,?,?,?,?,?,?)",
                       (now, channel, node, topic, p.get("store_id"), p.get("product_id"), json.dumps(p)))
            if topic == "prediction":
                db.execute("INSERT OR REPLACE INTO latest VALUES (?,?,?,?)",
                           (p["store_id"], p["product_id"], json.dumps(p), now))
        db.commit()


def start_mqtt():
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        mqtt_state["error"] = "paho-mqtt not installed"
        return

    def on_connect(c, *_a):
        mqtt_state["connected"] = True
        c.subscribe("retail/#", qos=1)

    def on_disconnect(*_a):
        mqtt_state["connected"] = False

    def on_message(_c, _u, msg):
        try:
            store_event(json.loads(msg.payload), "mqtt")
            mqtt_state["messages"] += 1
        except Exception as e:
            mqtt_state["error"] = str(e)

    def loop():
        while True:
            try:
                try:
                    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="central-dashboard")
                except AttributeError:
                    c = mqtt.Client(client_id="central-dashboard")
                c.on_connect, c.on_disconnect, c.on_message = on_connect, on_disconnect, on_message
                c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
                c.loop_forever()
            except Exception as e:
                mqtt_state.update(connected=False, error=str(e))
                time.sleep(3)

    threading.Thread(target=loop, daemon=True).start()


app = Flask(__name__, template_folder=str(ROOT / "dashboard" / "templates"))


@app.post("/api/ingest")
def ingest():
    env = request.get_json(silent=True)
    if not isinstance(env, dict) or "topic" not in env:
        return jsonify({"ok": False, "error": "expected {node_id, topic, payload}"}), 400
    store_event(env, "http")
    return jsonify({"ok": True})


@app.get("/api/state")
def state():
    with lock:
        latest = [json.loads(r["payload"]) | {"received": r["received"]}
                  for r in db.execute("SELECT * FROM latest ORDER BY store_id, product_id")]
        alerts = [json.loads(r["payload"]) | {"received": r["received"], "channel": r["channel"]}
                  for r in db.execute("SELECT * FROM events WHERE topic='alert' ORDER BY id DESC LIMIT 25")]
        feed = [{"received": r["received"], "channel": r["channel"], "node_id": r["node_id"],
                 **{k: v for k, v in json.loads(r["payload"]).items()
                    if k in ("store_id", "product_name", "predicted_demand", "status", "latency_ms")}}
                for r in db.execute("SELECT * FROM events WHERE topic='prediction' ORDER BY id DESC LIMIT 15")]
        counts = {r["topic"]: r["n"] for r in db.execute("SELECT topic, COUNT(*) n FROM events GROUP BY topic")}
        chan = {r["channel"]: r["n"] for r in db.execute("SELECT channel, COUNT(*) n FROM events GROUP BY channel")}
        nodes = [{"node_id": r["node_id"], "online": time.time() - r["last_seen"] < 15,
                  "seconds_ago": int(time.time() - r["last_seen"]), "channel": r["channel"],
                  **json.loads(r["payload"])} for r in db.execute("SELECT * FROM nodes")]
    lat = [x["latency_ms"] for x in latest if x.get("latency_ms") is not None]
    return jsonify({
        "latest": latest, "alerts": alerts, "feed": feed, "nodes": nodes, "counts": counts,
        "channels": chan, "mqtt": mqtt_state, "comm_mode": COMM_MODE,
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
        "server_time": datetime.now().strftime("%H:%M:%S"),
    })


@app.post("/api/reset")
def reset():
    with lock:
        db.executescript("DELETE FROM events; DELETE FROM latest;")
        db.commit()
    return jsonify({"ok": True})


@app.get("/")
def home():
    return render_template("dashboard.html")


if __name__ == "__main__":
    if mqtt_state["enabled"]:
        start_mqtt()
    print(f"\n  Central dashboard  ->  http://localhost:{PORT}   (receiving via {COMM_MODE.upper()})\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
