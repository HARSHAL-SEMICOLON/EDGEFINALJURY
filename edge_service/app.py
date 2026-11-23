"""
Smart Retail EDGE SERVICE  (runs on the in-store edge device / laptop)
=====================================================================
Flask microservice that performs LOCAL ML inference and replenishment decisions.

  Store UI ............ http://localhost:5000
  REST API ............ /api/predict, /api/predict/batch, /api/sales, /api/inventory,
                        /api/context, /api/catalog, /api/history, /api/alerts, /api/health
  Upstream comms ...... HTTP POST -> central dashboard  (default)
                        MQTT publish -> retail/<store>/<topic>   (COMM_MODE=mqtt or both)
  Store-and-forward ... every result is written to a local SQLite outbox first and
                        synced when the dashboard/broker is reachable (works offline).

Env vars:
  EDGE_PORT=5000  DASHBOARD_URL=http://localhost:8050  COMM_MODE=http|mqtt|both
  MQTT_HOST=localhost  MQTT_PORT=1883  EDGE_DB=edge_service/edge_store.db
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_service.engine import EdgeEngine, ValidationError  # noqa: E402

PORT = int(os.getenv("EDGE_PORT", "5000"))
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:8050").rstrip("/")
COMM_MODE = os.getenv("COMM_MODE", "http").lower()
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DB_PATH = os.getenv("EDGE_DB", str(ROOT / "edge_service" / "edge_store.db"))
NODE_ID = os.getenv("EDGE_NODE_ID", "pune-edge-01")


class Publisher(threading.Thread):
    """Background sync of the outbox to the central dashboard (HTTP) and/or MQTT broker."""

    def __init__(self, engine: EdgeEngine):
        super().__init__(daemon=True)
        self.engine = engine
        self.state = {"mode": COMM_MODE, "dashboard_url": DASHBOARD_URL, "http_ok": None,
                      "mqtt_ok": None, "last_sync": None, "last_error": None}
        self.mqtt = None
        if COMM_MODE in ("mqtt", "both"):
            self._connect_mqtt()

    def _connect_mqtt(self):
        try:
            import paho.mqtt.client as mqtt
            try:
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{NODE_ID}-pub")
            except AttributeError:  # paho < 2.0
                c = mqtt.Client(client_id=f"{NODE_ID}-pub")
            c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
            c.loop_start()
            self.mqtt = c
            self.state["mqtt_ok"] = True
        except Exception as e:  # broker down -> keep queueing, retry later
            self.mqtt = None
            self.state.update(mqtt_ok=False, last_error=f"MQTT: {e}")

    def _send(self, topic: str, payload: dict) -> bool:
        ok = False
        envelope = {"node_id": NODE_ID, "topic": topic, "payload": payload}
        if COMM_MODE in ("http", "both"):
            try:
                r = requests.post(f"{DASHBOARD_URL}/api/ingest", json=envelope, timeout=2)
                ok = r.ok
                self.state["http_ok"] = r.ok
            except requests.RequestException as e:
                self.state.update(http_ok=False, last_error=f"HTTP: {type(e).__name__}")
        if COMM_MODE in ("mqtt", "both"):
            if self.mqtt is None:
                self._connect_mqtt()
            if self.mqtt is not None:
                store = payload.get("store_id", "all").lower()
                info = self.mqtt.publish(f"retail/{store}/{topic}", json.dumps(envelope), qos=1)
                mqtt_ok = info.rc == 0
                self.state["mqtt_ok"] = mqtt_ok
                ok = ok or mqtt_ok if COMM_MODE == "both" else mqtt_ok
        return ok

    def run(self):
        last_hb = 0.0
        while True:
            try:
                rows = self.engine.q("SELECT id, topic, payload FROM outbox WHERE sent=0 ORDER BY id LIMIT 50")
                for r in rows:
                    if not self._send(r["topic"], json.loads(r["payload"])):
                        break  # upstream unreachable -> keep the rest queued
                    self.engine.execute("UPDATE outbox SET sent=1 WHERE id=?", (r["id"],))
                    self.state["last_sync"] = time.strftime("%H:%M:%S")
                if time.time() - last_hb > 5:
                    self._send("heartbeat", {"store_id": "all", "node_id": NODE_ID,
                                             "model": self.engine.meta["model_type"],
                                             "trained_at": self.engine.meta["trained_at"],
                                             **self.engine.stats()})
                    last_hb = time.time()
            except Exception as e:  # never let the sync thread die
                self.state["last_error"] = str(e)
            time.sleep(2)


def create_app(db_path: str = DB_PATH, start_publisher: bool = True) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "edge_service" / "templates"))
    engine = EdgeEngine(db_path)
    pub = Publisher(engine)
    if start_publisher:
        pub.start()
    app.config["engine"], app.config["publisher"] = engine, pub

    @app.errorhandler(ValidationError)
    def bad_input(e):
        return jsonify({"ok": False, "error": str(e)}), 400

    @app.errorhandler(Exception)
    def unexpected(e):
        code = getattr(e, "code", 500)
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), code if isinstance(code, int) else 500

    def body():
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()
        if not isinstance(data, dict):
            raise ValidationError("request body must be a JSON object")
        return data

    @app.get("/")
    def home():
        return render_template("index.html", node_id=NODE_ID, dashboard_url=DASHBOARD_URL)

    @app.get("/api/health")
    def health():
        m = engine.meta
        return jsonify({"ok": True, "node_id": NODE_ID, "model": m["model_type"],
                        "model_params": m["model_params"], "trained_at": m["trained_at"],
                        "test_metrics": m["metrics_test"], "model_size_mb": m["model_size_mb"],
                        "comms": pub.state, **engine.stats()})

    @app.get("/api/catalog")
    def catalog():
        inv = engine.q("SELECT * FROM inventory ORDER BY store_id, product_id")
        return jsonify({"stores": engine.stores, "products": engine.products, "inventory": inv})

    @app.get("/api/context")
    def context():
        return jsonify(engine.context(request.args.get("store_id", "").upper(),
                                      request.args.get("product_id", "").upper()))

    @app.post("/api/predict")
    def predict():
        return jsonify({"ok": True, **engine.predict(body(), source=request.headers.get("X-Source", "api"))})

    @app.post("/api/predict/batch")
    def predict_batch():
        b = body()
        if "items" in b:
            res = []
            for item in b["items"]:
                try:
                    res.append({"ok": True, **engine.predict(item, source="batch")})
                except ValidationError as e:
                    res.append({"ok": False, "error": str(e), "input": item})
            return jsonify({"ok": True, "results": res})
        store = str(b.get("store_id", "")).upper()
        t = time.perf_counter()
        res = engine.scan_store(store)
        summary = {s: sum(r["status"] == s for r in res) for s in ("CRITICAL", "REORDER", "OVERSTOCK", "OK")}
        return jsonify({"ok": True, "store_id": store, "summary": summary, "results": res,
                        "total_ms": round((time.perf_counter() - t) * 1000, 1)})

    BATCH_REQUIRED = ("store_id", "product_id")
    BATCH_OPTIONAL = ("date", "unit_price", "discount_pct", "promo_flag", "current_stock",
                      "lead_time_days", "safety_stock_days")
    MAX_BATCH_ROWS = 200

    @app.get("/api/batch/template")
    def batch_template():
        csv_text = (ROOT / "data" / "sample_batch.csv").read_text()
        return app.response_class(csv_text, mimetype="text/csv",
                                  headers={"Content-Disposition": "attachment; filename=batch_template.csv"})

    @app.post("/api/predict/upload")
    def predict_upload():
        """Batch prediction from an uploaded CSV. Bad rows are reported, good rows still predicted."""
        f = request.files.get("file")
        if f is None or not f.filename:
            raise ValidationError("upload a CSV file in form field 'file'")
        if not f.filename.lower().endswith(".csv"):
            raise ValidationError(f"'{f.filename}' is not a .csv file")
        try:
            text = f.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValidationError("file is not UTF-8 text - save it as 'CSV UTF-8' from Excel")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValidationError("CSV is empty")
        cols = [c.strip().lower() for c in reader.fieldnames]
        missing = [c for c in BATCH_REQUIRED if c not in cols]
        if missing:
            raise ValidationError(f"CSV is missing required column(s): {', '.join(missing)}. "
                                  f"Required: {', '.join(BATCH_REQUIRED)}; optional: {', '.join(BATCH_OPTIONAL)}")
        rows = []
        for raw in reader:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            if any(row.values()):
                rows.append(row)
        if not rows:
            raise ValidationError("CSV has a header but no data rows")
        if len(rows) > MAX_BATCH_ROWS:
            raise ValidationError(f"{len(rows)} rows - max {MAX_BATCH_ROWS} per batch on this edge node")

        t = time.perf_counter()
        results = []
        for i, row in enumerate(rows, start=2):  # row 1 = header, as seen in Excel
            item = {k: row[k] for k in BATCH_REQUIRED + BATCH_OPTIONAL if row.get(k, "") != ""}
            try:
                r = engine.predict(item, source="batch-csv")
                results.append({"row": i, "ok": True, **r})
            except ValidationError as e:
                results.append({"row": i, "ok": False, "error": str(e), "input": item})
        ok = [r for r in results if r["ok"]]
        summary = {s: sum(r["status"] == s for r in ok) for s in ("CRITICAL", "REORDER", "OVERSTOCK", "OK")}
        return jsonify({"ok": True, "file": f.filename, "rows": len(rows), "predicted": len(ok),
                        "rejected": len(rows) - len(ok), "summary": summary,
                        "units_to_order": sum(r["reorder_qty"] for r in ok),
                        "total_ms": round((time.perf_counter() - t) * 1000, 1), "results": results})

    @app.post("/api/sales")
    def sales():
        b = body()
        items = b.get("items", [b])
        out = [engine.ingest_sale(i) for i in items]
        return jsonify({"ok": True, "ingested": len(out), "results": out})

    @app.post("/api/inventory")
    def inventory():
        b = body()
        store, prod = str(b.get("store_id", "")).upper(), str(b.get("product_id", "")).upper()
        if "add_units" in b:
            new = engine.set_stock(store, prod, EdgeEngine._num(b, "add_units", lo=0, integer=True), add=True)
        else:
            new = engine.set_stock(store, prod, EdgeEngine._num(b, "current_stock", lo=0, integer=True))
        return jsonify({"ok": True, "store_id": store, "product_id": prod, "current_stock": new})

    @app.get("/api/history")
    def history():
        n = min(int(request.args.get("limit", 50)), 500)
        rows = engine.q("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,))
        for r in rows:
            r["product_name"] = engine.products[r["product_id"]]["name"]
        return jsonify(rows)

    @app.get("/api/alerts")
    def alerts():
        return jsonify(engine.q("SELECT * FROM alerts ORDER BY id DESC LIMIT 50"))

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\n  Smart Retail EDGE service  ->  http://localhost:{PORT}   (comms: {COMM_MODE} -> "
          f"{DASHBOARD_URL if COMM_MODE != 'mqtt' else f'mqtt://{MQTT_HOST}:{MQTT_PORT}'})\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
