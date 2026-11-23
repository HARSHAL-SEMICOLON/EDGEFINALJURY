"""
Edge inference engine: loads the trained Random Forest locally, keeps a small
SQLite "edge store" (sales history, inventory, predictions, alerts, outbox) and
turns raw model output into replenishment decisions.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from ml.cleaning import clean  # noqa: E402
from ml.features import MIN_HISTORY_DAYS, FestivalCalendar, build_inference_row  # noqa: E402

MAX_FORECAST_AHEAD_DAYS = 60
FORECAST_DAYS = 7


class ValidationError(ValueError):
    """Bad input -> HTTP 400 with a clear message."""


class EdgeEngine:
    def __init__(self, db_path: Path | str, model_dir: Path | str | None = None,
                 data_dir: Path | str | None = None):
        self.model_dir = Path(model_dir or ROOT / "models")
        self.data_dir = Path(data_dir or ROOT / "data")
        self.db_path = str(db_path)
        self.lock = threading.RLock()
        self.started = time.time()

        self.meta = json.loads((self.model_dir / "metadata.json").read_text())
        self.model = joblib.load(self.model_dir / self.meta["model_file"])
        self.model.set_params(n_jobs=1)  # single-row inference is fastest single-threaded
        self.cal = FestivalCalendar(self.meta["festivals"])
        self.products = self.meta["products"]
        self.stores = self.meta["stores"]

        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------ storage
    def _init_db(self):
        with self.lock:
            c = self.db
            c.executescript("""
            CREATE TABLE IF NOT EXISTS sales_history(
                store_id TEXT, product_id TEXT, date TEXT, units_sold REAL,
                unit_price REAL, discount_pct REAL, promo_flag INTEGER,
                PRIMARY KEY(store_id, product_id, date));
            CREATE TABLE IF NOT EXISTS inventory(
                store_id TEXT, product_id TEXT, current_stock INTEGER,
                lead_time_days INTEGER, safety_stock_days INTEGER, updated_at TEXT,
                PRIMARY KEY(store_id, product_id));
            CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, store_id TEXT, product_id TEXT,
                target_date TEXT, predicted_units REAL, forecast_7d TEXT, current_stock INTEGER,
                reorder_point REAL, status TEXT, reorder_qty INTEGER, latency_ms REAL,
                source TEXT, inputs TEXT);
            CREATE TABLE IF NOT EXISTS alerts(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, store_id TEXT, product_id TEXT,
                severity TEXT, message TEXT);
            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, topic TEXT, payload TEXT,
                sent INTEGER DEFAULT 0);
            """)
            if c.execute("SELECT COUNT(*) FROM sales_history").fetchone()[0] == 0:
                self._seed()
            c.commit()

    def _seed(self):
        """First boot: load the last 90 days of cleaned POS history + inventory snapshot."""
        raw = pd.read_csv(self.data_dir / "retail_sales.csv")
        raw["date"] = pd.to_datetime(raw["date"])
        cutoff = raw["date"].max() - pd.Timedelta(days=89)
        df, _ = clean(raw[raw["date"] >= cutoff])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        rows = df[["store_id", "product_id", "date", "units_sold", "unit_price",
                   "discount_pct", "promo_flag"]].itertuples(index=False, name=None)
        self.db.executemany("INSERT OR REPLACE INTO sales_history VALUES (?,?,?,?,?,?,?)", list(rows))
        inv = pd.read_csv(self.data_dir / "inventory.csv")
        now = datetime.now().isoformat(timespec="seconds")
        self.db.executemany(
            "INSERT OR REPLACE INTO inventory VALUES (?,?,?,?,?,?)",
            [(r.store_id, r.product_id, int(r.current_stock), int(r.lead_time_days),
              int(r.safety_stock_days), now) for r in inv.itertuples()])

    def q(self, sql, args=(), one=False):
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
        return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]

    def execute(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------ context
    def last_history_date(self, store_id, product_id) -> date:
        r = self.q("SELECT MAX(date) d FROM sales_history WHERE store_id=? AND product_id=?",
                   (store_id, product_id), one=True)
        return date.fromisoformat(r["d"])

    def history(self, store_id, product_id, days=28):
        rows = self.q("SELECT date, units_sold, unit_price, discount_pct, promo_flag FROM sales_history "
                      "WHERE store_id=? AND product_id=? ORDER BY date DESC LIMIT ?",
                      (store_id, product_id, days))
        return rows[::-1]

    def inventory(self, store_id, product_id):
        return self.q("SELECT * FROM inventory WHERE store_id=? AND product_id=?",
                      (store_id, product_id), one=True)

    def context(self, store_id, product_id):
        self._check_ids(store_id, product_id)
        hist = self.history(store_id, product_id, 28)
        inv = self.inventory(store_id, product_id)
        last = hist[-1]
        next_day = self.last_history_date(store_id, product_id) + timedelta(days=1)
        return {
            "store_id": store_id, "store_name": self.stores[store_id],
            "product_id": product_id, "product_name": self.products[product_id]["name"],
            "category": self.products[product_id]["category"],
            "last_sales_date": hist[-1]["date"], "suggested_date": str(next_day),
            "unit_price": round(last["unit_price"], 2),
            "reference_price": self.meta["ref_price"][product_id],
            "recent_sales": [int(h["units_sold"]) for h in hist],
            "current_stock": inv["current_stock"], "lead_time_days": inv["lead_time_days"],
            "safety_stock_days": inv["safety_stock_days"],
            "upcoming_festivals": self.cal.upcoming(next_day, 30),
        }

    # ------------------------------------------------------------------ validation
    def _check_ids(self, store_id, product_id):
        if store_id not in self.stores:
            raise ValidationError(f"Unknown store_id '{store_id}'. Valid: {sorted(self.stores)}")
        if product_id not in self.products:
            raise ValidationError(f"Unknown product_id '{product_id}'. Use /api/catalog for valid products")

    @staticmethod
    def _num(payload, key, default=None, *, lo=None, hi=None, integer=False):
        v = payload.get(key, default)
        if v is None or v == "":
            if default is None:
                raise ValidationError(f"'{key}' is required")
            v = default
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValidationError(f"'{key}' must be a number, got {payload.get(key)!r}")
        if math.isnan(v) or math.isinf(v):
            raise ValidationError(f"'{key}' must be a finite number")
        if lo is not None and v < lo:
            raise ValidationError(f"'{key}' must be >= {lo}, got {v:g}")
        if hi is not None and v > hi:
            raise ValidationError(f"'{key}' must be <= {hi}, got {v:g}")
        return int(round(v)) if integer else v

    # ------------------------------------------------------------------ prediction
    def predict(self, payload: dict, source: str = "api") -> dict:
        t0 = time.perf_counter()
        store_id = str(payload.get("store_id", "")).strip().upper()
        product_id = str(payload.get("product_id", "")).strip().upper()
        self._check_ids(store_id, product_id)
        warnings = []

        ctx_last = self.last_history_date(store_id, product_id)
        raw_date = payload.get("date") or str(ctx_last + timedelta(days=1))
        try:
            target = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            raise ValidationError(f"'date' must be YYYY-MM-DD, got {raw_date!r}")
        ahead = (target - ctx_last).days
        if ahead > MAX_FORECAST_AHEAD_DAYS:
            raise ValidationError(f"date is {ahead} days after the latest sales data ({ctx_last}); "
                                  f"max supported horizon is {MAX_FORECAST_AHEAD_DAYS} days")
        if ahead < -365 * 3:
            raise ValidationError("date is too far in the past")

        last_row = self.history(store_id, product_id, 1)[0]
        unit_price = self._num(payload, "unit_price", last_row["unit_price"], lo=0.01, hi=100000)
        discount = self._num(payload, "discount_pct", 0, lo=0, hi=90)
        promo = int(self._num(payload, "promo_flag", 1 if discount > 0 else 0, lo=0, hi=1, integer=True))
        inv = self.inventory(store_id, product_id)
        stock = self._num(payload, "current_stock", inv["current_stock"], lo=0, hi=1_000_000, integer=True)
        lead = self._num(payload, "lead_time_days", inv["lead_time_days"], lo=1, hi=30, integer=True)
        safety_days = self._num(payload, "safety_stock_days", inv["safety_stock_days"], lo=0, hi=30,
                                integer=True)

        ref = self.meta["ref_price"][product_id]
        if unit_price > ref * 1.6 or unit_price < ref * 0.4:
            warnings.append(f"Price Rs.{unit_price:g} is far from the usual Rs.{ref:g} - outside the range the "
                            "model was trained on, so the estimate is less reliable.")
        if discount >= 40:
            warnings.append(f"{discount:g}% discount is higher than any promotion in the training data (max 25%).")

        # demand history: user-supplied (what-if) or from the edge store
        if payload.get("recent_sales") not in (None, "", []):
            rs = payload["recent_sales"]
            if isinstance(rs, str):
                rs = [x for x in rs.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
            try:
                hist = [float(x) for x in rs]
            except (TypeError, ValueError):
                raise ValidationError("'recent_sales' must be a list of numbers")
            if len(hist) < MIN_HISTORY_DAYS:
                raise ValidationError(f"'recent_sales' needs at least {MIN_HISTORY_DAYS} daily values "
                                      f"(oldest -> newest), got {len(hist)}")
            if any(h < 0 for h in hist):
                raise ValidationError("'recent_sales' cannot contain negative values")
            hist = hist[-MIN_HISTORY_DAYS:]
            typical = np.median([h["units_sold"] for h in self.history(store_id, product_id, 28)])
            if np.mean(hist) > typical * 4 or (typical > 0 and np.mean(hist) < typical * 0.15):
                warnings.append("Supplied recent sales are very different from this store's normal level "
                                f"(~{typical:.0f}/day). Prediction follows the supplied history.")
            gap = 1
        else:
            hist = [h["units_sold"] for h in self.history(store_id, product_id, MIN_HISTORY_DAYS)]
            gap = max(ahead, 1)

        # recursive forecast: fill any gap up to target, then 7 days from target
        series = list(hist)
        start = target - timedelta(days=gap - 1)
        days = [start + timedelta(days=i) for i in range(gap - 1 + FORECAST_DAYS)]
        preds, X_target = [], None
        for d in days:
            X = build_inference_row(store_id=store_id, product_id=product_id, target_date=d,
                                    unit_price=unit_price, discount_pct=discount, promo_flag=promo,
                                    history=series[-MIN_HISTORY_DAYS:], cal=self.cal, meta=self.meta)
            p = max(float(self.model.predict(X)[0]), 0.0)
            if d == target:
                X_target = X
            series.append(p)
            preds.append((d, p))
        fc = preds[gap - 1:]
        forecast = [{"date": str(d), "units": round(p, 1)} for d, p in fc]

        # spread across the forest's trees -> simple 80% band for the target day
        tree_preds = np.array([t.predict(X_target.values)[0] for t in self.model.estimators_])
        lo, hi = np.percentile(tree_preds, [10, 90])

        decision = self.replenishment(fc, stock, lead, safety_days)
        latency = (time.perf_counter() - t0) * 1000

        result = {
            "store_id": store_id, "store_name": self.stores[store_id],
            "product_id": product_id, "product_name": self.products[product_id]["name"],
            "category": self.products[product_id]["category"],
            "target_date": str(target),
            "predicted_demand": round(fc[0][1], 1),
            "prediction_interval_80": [round(max(lo, 0), 1), round(hi, 1)],
            "forecast_7d": forecast,
            "forecast_7d_total": round(sum(p for _, p in fc), 1),
            "inputs_used": {"unit_price": unit_price, "discount_pct": discount, "promo_flag": promo,
                            "current_stock": stock, "lead_time_days": lead, "safety_stock_days": safety_days,
                            "history_source": "user-supplied" if gap == 1 and payload.get("recent_sales") else
                            "edge store (local POS history)"},
            "festival_context": self.cal.upcoming(target, 14),
            **decision,
            "warnings": warnings,
            "edge": {"inference": "local", "model": self.meta["model_type"],
                     "latency_ms": round(latency, 1), "node": "store-edge-" + store_id.lower()},
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self._record(result, source)
        return result

    @staticmethod
    def replenishment(fc, stock, lead, safety_days):
        daily = [p for _, p in fc]
        lead_demand = sum(daily[:lead]) if lead <= len(daily) else sum(daily) / len(daily) * lead
        avg = sum(daily) / len(daily)
        safety = avg * safety_days
        reorder_point = lead_demand + safety
        week = sum(daily)
        days_cover = stock / avg if avg > 0 else float("inf")
        if stock < daily[0]:
            status, msg = "CRITICAL", "Stock will run out tomorrow - urgent replenishment"
        elif stock < reorder_point:
            status, msg = "REORDER", (f"Stock covers ~{days_cover:.1f} days but supplier lead time is {lead} days "
                                      f"(+{safety_days}d safety) - place order now")
        elif stock > 3 * week:
            status, msg = "OVERSTOCK", f"Stock covers ~{days_cover:.0f} days - pause ordering / consider promotion"
        else:
            status, msg = "OK", f"Stock covers ~{days_cover:.1f} days - no action needed"
        # order-up-to level: demand over the review horizon (2x lead time, max 7 days) + safety stock.
        # Perishables (short lead time) are therefore not over-ordered.
        horizon = min(len(daily), 2 * lead)
        order_up_to = sum(daily[:horizon]) + safety
        qty = int(math.ceil(max(0.0, order_up_to - stock))) if status in ("CRITICAL", "REORDER") else 0
        return {"current_stock": stock, "reorder_point": round(reorder_point, 1),
                "days_of_cover": round(days_cover, 1) if days_cover != float("inf") else None,
                "status": status, "recommendation": msg, "reorder_qty": qty}

    def _record(self, r, source):
        self.execute(
            "INSERT INTO predictions(ts,store_id,product_id,target_date,predicted_units,forecast_7d,"
            "current_stock,reorder_point,status,reorder_qty,latency_ms,source,inputs) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["ts"], r["store_id"], r["product_id"], r["target_date"], r["predicted_demand"],
             json.dumps(r["forecast_7d"]), r["current_stock"], r["reorder_point"], r["status"],
             r["reorder_qty"], r["edge"]["latency_ms"], source, json.dumps(r["inputs_used"])))
        if r["status"] in ("CRITICAL", "REORDER"):
            msg = (f"{r['product_name']} @ {r['store_name']}: {r['status']} - stock {r['current_stock']} vs "
                   f"reorder point {r['reorder_point']:.0f}. Order {r['reorder_qty']} units.")
            self.execute("INSERT INTO alerts(ts,store_id,product_id,severity,message) VALUES (?,?,?,?,?)",
                         (r["ts"], r["store_id"], r["product_id"], r["status"], msg))
            self.enqueue("alert", {"store_id": r["store_id"], "product_id": r["product_id"],
                                   "severity": r["status"], "message": msg, "ts": r["ts"]})
        # only a compact summary leaves the store (not raw POS data)
        self.enqueue("prediction", {k: r[k] for k in (
            "ts", "store_id", "store_name", "product_id", "product_name", "category", "target_date",
            "predicted_demand", "forecast_7d_total", "current_stock", "reorder_point", "status",
            "reorder_qty", "days_of_cover")} | {"latency_ms": r["edge"]["latency_ms"]})

    def enqueue(self, topic, payload):
        self.execute("INSERT INTO outbox(ts,topic,payload) VALUES (?,?,?)",
                     (datetime.now().isoformat(timespec="seconds"), topic, json.dumps(payload)))

    # ------------------------------------------------------------------ store operations
    def scan_store(self, store_id, source="scan"):
        if store_id not in self.stores:
            raise ValidationError(f"Unknown store_id '{store_id}'")
        out = [self.predict({"store_id": store_id, "product_id": pid}, source=source) for pid in self.products]
        order = {"CRITICAL": 0, "REORDER": 1, "OVERSTOCK": 2, "OK": 3}
        return sorted(out, key=lambda r: (order[r["status"]], r["product_id"]))

    def ingest_sale(self, payload):
        store_id = str(payload.get("store_id", "")).upper()
        product_id = str(payload.get("product_id", "")).upper()
        self._check_ids(store_id, product_id)
        units = self._num(payload, "units_sold", lo=0, hi=100000)
        try:
            d = date.fromisoformat(str(payload.get("date"))[:10])
        except ValueError:
            raise ValidationError("'date' must be YYYY-MM-DD")
        last = self.history(store_id, product_id, 1)[0]
        price = self._num(payload, "unit_price", last["unit_price"], lo=0.01)
        disc = self._num(payload, "discount_pct", 0, lo=0, hi=90)
        promo = int(self._num(payload, "promo_flag", 1 if disc > 0 else 0, lo=0, hi=1, integer=True))
        self.execute("INSERT OR REPLACE INTO sales_history VALUES (?,?,?,?,?,?,?)",
                     (store_id, product_id, str(d), units, price, disc, promo))
        inv = self.inventory(store_id, product_id)
        new_stock = max(0, inv["current_stock"] - int(units))
        self.set_stock(store_id, product_id, new_stock)
        return {"store_id": store_id, "product_id": product_id, "date": str(d),
                "units_sold": units, "stock_after_sale": new_stock}

    def set_stock(self, store_id, product_id, stock, add=False):
        self._check_ids(store_id, product_id)
        inv = self.inventory(store_id, product_id)
        new = inv["current_stock"] + int(stock) if add else int(stock)
        if new < 0:
            raise ValidationError("stock cannot be negative")
        self.execute("UPDATE inventory SET current_stock=?, updated_at=? WHERE store_id=? AND product_id=?",
                     (new, datetime.now().isoformat(timespec="seconds"), store_id, product_id))
        return new

    def stats(self):
        return {
            "predictions": self.q("SELECT COUNT(*) n FROM predictions", one=True)["n"],
            "alerts": self.q("SELECT COUNT(*) n FROM alerts", one=True)["n"],
            "pending_sync": self.q("SELECT COUNT(*) n FROM outbox WHERE sent=0", one=True)["n"],
            "synced": self.q("SELECT COUNT(*) n FROM outbox WHERE sent=1", one=True)["n"],
            "history_rows": self.q("SELECT COUNT(*) n FROM sales_history", one=True)["n"],
            "uptime_s": int(time.time() - self.started),
        }
