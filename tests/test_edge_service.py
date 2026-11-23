"""Automated tests for the edge service: normal, varied and abnormal inputs.
Run:  python -m pytest -q tests
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_service.app import create_app  # noqa: E402
from edge_service.engine import EdgeEngine  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("edge") / "test.db"
    app = create_app(str(db), start_publisher=False)
    return app.test_client()


def post(c, url, body):
    return c.post(url, json=body)


def test_health(client):
    j = client.get("/api/health").get_json()
    assert j["ok"] and j["model"] == "RandomForestRegressor"


def test_normal_prediction(client):
    r = post(client, "/api/predict", {"store_id": "STR02", "product_id": "P201"})
    j = r.get_json()
    assert r.status_code == 200
    assert j["predicted_demand"] > 0
    assert len(j["forecast_7d"]) == 7
    assert j["status"] in {"CRITICAL", "REORDER", "OVERSTOCK", "OK"}
    assert j["prediction_interval_80"][0] <= j["predicted_demand"] <= j["prediction_interval_80"][1] * 1.5
    assert j["edge"]["inference"] == "local"


def test_promotion_increases_demand(client):
    base = post(client, "/api/predict", {"store_id": "STR01", "product_id": "P401", "date": "2026-11-24"}).get_json()
    promo = post(client, "/api/predict", {"store_id": "STR01", "product_id": "P401", "date": "2026-11-24",
                                          "discount_pct": 25, "promo_flag": 1}).get_json()
    assert promo["predicted_demand"] > base["predicted_demand"]


def test_low_stock_flags_replenishment(client):
    j = post(client, "/api/predict", {"store_id": "STR01", "product_id": "P201", "current_stock": 5}).get_json()
    assert j["status"] == "CRITICAL" and j["reorder_qty"] > 0


def test_overstock(client):
    j = post(client, "/api/predict", {"store_id": "STR01", "product_id": "P502", "current_stock": 5000}).get_json()
    assert j["status"] == "OVERSTOCK" and j["reorder_qty"] == 0


def test_abnormal_price_warns(client):
    j = post(client, "/api/predict", {"store_id": "STR01", "product_id": "P203", "unit_price": 450}).get_json()
    assert j["ok"] and j["warnings"]


@pytest.mark.parametrize("payload,needle", [
    ({"store_id": "STR01", "product_id": "P999"}, "Unknown product"),
    ({"store_id": "XX", "product_id": "P201"}, "Unknown store"),
    ({"store_id": "STR01", "product_id": "P201", "unit_price": -1}, "unit_price"),
    ({"store_id": "STR01", "product_id": "P201", "discount_pct": 150}, "discount_pct"),
    ({"store_id": "STR01", "product_id": "P201", "current_stock": -3}, "current_stock"),
    ({"store_id": "STR01", "product_id": "P201", "unit_price": "abc"}, "number"),
    ({"store_id": "STR01", "product_id": "P201", "date": "31-12-2026"}, "YYYY-MM-DD"),
    ({"store_id": "STR01", "product_id": "P201", "date": "2027-08-01"}, "horizon"),
    ({"store_id": "STR01", "product_id": "P201", "recent_sales": [1, 2]}, "at least 28"),
    ({"store_id": "STR01", "product_id": "P201", "recent_sales": [-1] * 28}, "negative"),
])
def test_invalid_inputs_rejected(client, payload, needle):
    r = post(client, "/api/predict", payload)
    assert r.status_code == 400
    assert needle.lower() in r.get_json()["error"].lower()


def test_pos_sale_updates_stock_and_history(client):
    ctx = client.get("/api/context?store_id=STR03&product_id=P602").get_json()
    r = post(client, "/api/sales", {"store_id": "STR03", "product_id": "P602",
                                     "date": ctx["suggested_date"], "units_sold": 20}).get_json()
    assert r["results"][0]["stock_after_sale"] == max(0, ctx["current_stock"] - 20)
    ctx2 = client.get("/api/context?store_id=STR03&product_id=P602").get_json()
    assert ctx2["recent_sales"][-1] == 20


def test_store_scan(client):
    j = post(client, "/api/predict/batch", {"store_id": "STR01"}).get_json()
    assert len(j["results"]) == 24 and sum(j["summary"].values()) == 24


def test_replenishment_rule():
    fc = [(None, 10.0)] * 7
    assert EdgeEngine.replenishment(fc, 5, 2, 1)["status"] == "CRITICAL"
    assert EdgeEngine.replenishment(fc, 25, 2, 1)["status"] == "REORDER"   # rop = 20 + 10 = 30
    assert EdgeEngine.replenishment(fc, 50, 2, 1)["status"] == "OK"
    assert EdgeEngine.replenishment(fc, 300, 2, 1)["status"] == "OVERSTOCK"
    assert EdgeEngine.replenishment(fc, 25, 2, 1)["reorder_qty"] == 25     # 4-day horizon 40 + 10 - 25
    assert EdgeEngine.replenishment(fc, 25, 5, 2)["reorder_qty"] == 65     # 7-day horizon 70 + 20 - 25


# ---------------- batch CSV upload ----------------
import io  # noqa: E402

DATA = ROOT / "data"


def upload(client, name, content: bytes):
    return client.post("/api/predict/upload", data={"file": (io.BytesIO(content), name)},
                       content_type="multipart/form-data")


def test_batch_csv_all_valid(client):
    j = upload(client, "sample_batch.csv", (DATA / "sample_batch.csv").read_bytes()).get_json()
    assert j["ok"] and j["rows"] == 12 and j["predicted"] == 12 and j["rejected"] == 0
    assert sum(j["summary"].values()) == 12


def test_batch_csv_bad_rows_reported_not_fatal(client):
    j = upload(client, "errs.csv", (DATA / "sample_batch_with_errors.csv").read_bytes()).get_json()
    assert j["rows"] == 10 and j["predicted"] == 3 and j["rejected"] == 7
    bad = [r for r in j["results"] if not r["ok"]]
    assert all(r["error"] for r in bad) and bad[0]["row"] == 3   # row numbers match Excel


@pytest.mark.parametrize("name,content,needle", [
    ("x.csv", b"store,product\nSTR01,P201\n", "missing required"),
    ("x.csv", b"", "empty"),
    ("x.csv", b"store_id,product_id\n", "no data rows"),
    ("x.txt", b"store_id,product_id\nSTR01,P201\n", "not a .csv"),
])
def test_batch_csv_invalid_files(client, name, content, needle):
    r = upload(client, name, content)
    assert r.status_code == 400 and needle in r.get_json()["error"]


def test_batch_template_download(client):
    r = client.get("/api/batch/template")
    assert r.status_code == 200 and r.data.startswith(b"store_id,product_id")
