"""
Store device simulator - talks to the EDGE SERVICE over HTTP, like a POS terminal
or a store manager's handheld would.

  python simulator/store_simulator.py tests          # normal + abnormal input test cases
  python simulator/store_simulator.py pos --days 3   # stream POS sales day-by-day, watch alerts fire
  python simulator/store_simulator.py scan --store STR01
  python simulator/store_simulator.py batch --file data/sample_batch_with_errors.csv

Env/arg: --edge http://localhost:5000
"""
import argparse
import random
import sys
import time
from datetime import date, timedelta

import requests

GREEN, RED, YEL, DIM, END = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
if sys.platform == "win32":
    import os
    os.system("")  # enable ANSI colours in Windows terminal

TEST_CASES = [
    # (description, payload, expect_ok)
    ("Normal: milk, next day, regular price", {"store_id": "STR02", "product_id": "P201"}, True),
    ("Normal: chips on 20% promotion", {"store_id": "STR01", "product_id": "P401",
                                        "discount_pct": 20, "promo_flag": 1}, True),
    ("Normal: weekend date (Sat 28-Nov)", {"store_id": "STR03", "product_id": "P402",
                                           "date": "2026-11-28"}, True),
    ("Normal: low stock tea powder", {"store_id": "STR01", "product_id": "P301", "current_stock": 10}, True),
    ("Normal: overstocked detergent", {"store_id": "STR02", "product_id": "P504", "current_stock": 2000}, True),
    ("Varied: festival what-if (Christmas eve)", {"store_id": "STR02", "product_id": "P404",
                                                  "date": "2026-12-24"}, True),
    ("Varied: what-if sales spike history", {"store_id": "STR01", "product_id": "P303",
                                             "recent_sales": [40] * 21 + [90, 95, 110, 120, 130, 140, 150]}, True),
    ("Abnormal: price 5x normal (warns)", {"store_id": "STR01", "product_id": "P203", "unit_price": 450}, True),
    ("Abnormal: 60% discount (warns)", {"store_id": "STR01", "product_id": "P402", "discount_pct": 60}, True),
    ("Invalid: unknown product", {"store_id": "STR01", "product_id": "P999"}, False),
    ("Invalid: unknown store", {"store_id": "STR77", "product_id": "P201"}, False),
    ("Invalid: negative price", {"store_id": "STR01", "product_id": "P201", "unit_price": -20}, False),
    ("Invalid: discount 150%", {"store_id": "STR01", "product_id": "P201", "discount_pct": 150}, False),
    ("Invalid: negative stock", {"store_id": "STR01", "product_id": "P201", "current_stock": -5}, False),
    ("Invalid: text in number field", {"store_id": "STR01", "product_id": "P201", "unit_price": "fifty"}, False),
    ("Invalid: bad date format", {"store_id": "STR01", "product_id": "P201", "date": "22/11/2026"}, False),
    ("Invalid: date 6 months ahead", {"store_id": "STR01", "product_id": "P201", "date": "2027-05-30"}, False),
    ("Invalid: only 5 days of history", {"store_id": "STR01", "product_id": "P201",
                                         "recent_sales": [10, 12, 9, 11, 13]}, False),
    ("Invalid: negative sales history", {"store_id": "STR01", "product_id": "P201",
                                         "recent_sales": [-3] + [20] * 27}, False),
]


def run_tests(edge):
    print(f"\nSending {len(TEST_CASES)} test inputs to edge service {edge}\n")
    print(f"{'#':>2}  {'Case':42s} {'HTTP':>4}  {'Result':60s}")
    print("-" * 115)
    passed = 0
    for i, (name, payload, expect_ok) in enumerate(TEST_CASES, 1):
        r = requests.post(f"{edge}/api/predict", json=payload, headers={"X-Source": "simulator-tests"}, timeout=30)
        j = r.json()
        ok = r.status_code == 200
        if ok:
            res = (f"{j['predicted_demand']:>6.1f} units  {j['status']:9s} stock {j['current_stock']:>5}"
                   f"  order {j['reorder_qty']:>4}  {j['edge']['latency_ms']:.0f}ms")
            if j["warnings"]:
                res += f"  {YEL}⚠ {len(j['warnings'])} warning(s){END}"
        else:
            res = f"rejected: {j['error'][:70]}"
        good = ok == expect_ok
        passed += good
        mark = f"{GREEN}PASS{END}" if good else f"{RED}FAIL{END}"
        print(f"{i:>2}  {name:42s} {r.status_code:>4}  {res}  {mark}")
    print("-" * 115)
    print(f"{passed}/{len(TEST_CASES)} cases behaved as expected "
          "(valid inputs predicted, abnormal inputs warned, invalid inputs rejected)\n")


def run_pos(edge, store, days, delay):
    """Replay realistic daily POS sales into the edge node; stock drops, model re-forecasts, alerts fire."""
    cat = requests.get(f"{edge}/api/catalog", timeout=10).json()
    products = list(cat["products"])
    print(f"\nStreaming {days} day(s) of POS sales for {store} into {edge} ...\n")
    for _ in range(days):
        ctx = requests.get(f"{edge}/api/context", params={"store_id": store, "product_id": products[0]},
                           timeout=10).json()
        d = ctx["suggested_date"]
        items = []
        for pid in products:
            c = requests.get(f"{edge}/api/context", params={"store_id": store, "product_id": pid}, timeout=10).json()
            base = sum(c["recent_sales"][-7:]) / 7
            dow = date.fromisoformat(d).weekday()
            mult = 1.2 if dow >= 5 else 1.0
            items.append({"store_id": store, "product_id": pid, "date": d,
                          "units_sold": max(0, round(random.gauss(base * mult, base * 0.15)))})
        requests.post(f"{edge}/api/sales", json={"items": items}, timeout=30)
        scan = requests.post(f"{edge}/api/predict/batch", json={"store_id": store}, timeout=120).json()
        s = scan["summary"]
        print(f"{d}  POS uploaded {len(items)} products -> edge re-forecast in {scan['total_ms']:.0f} ms  |  "
              f"{RED}CRITICAL {s['CRITICAL']}{END}  {YEL}REORDER {s['REORDER']}{END}  "
              f"OVERSTOCK {s['OVERSTOCK']}  {GREEN}OK {s['OK']}{END}")
        for r in scan["results"]:
            if r["status"] == "CRITICAL":
                print(f"    {RED}! {r['product_name']:22s} stock {r['current_stock']:>4} < tomorrow "
                      f"{r['predicted_demand']:.0f} -> order {r['reorder_qty']}{END}")
        time.sleep(delay)
    print("\nDone. Check the central dashboard for synced alerts.\n")


def run_batch(edge, path):
    from pathlib import Path
    p = Path(path)
    with p.open("rb") as fh:
        r = requests.post(f"{edge}/api/predict/upload", files={"file": (p.name, fh, "text/csv")}, timeout=300)
    j = r.json()
    if not r.ok:
        sys.exit(f"{RED}Batch rejected: {j.get('error')}{END}")
    print(f"\n{p.name}: {j['rows']} rows -> {GREEN}{j['predicted']} predicted{END}, "
          f"{RED}{j['rejected']} rejected{END}  |  {j['summary']}  |  order {j['units_to_order']} units  "
          f"|  {j['total_ms']:.0f} ms\n")
    for x in j["results"]:
        if x["ok"]:
            w = f"  {YEL}⚠ warning{END}" if x["warnings"] else ""
            print(f"  row {x['row']:>3}  {x['store_id']} {x['product_name']:22s} {x['target_date']}  "
                  f"{x['predicted_demand']:>6.1f}  {x['status']:9s} order {x['reorder_qty']:>4}{w}")
        else:
            print(f"  row {x['row']:>3}  {RED}REJECTED{END}  {x['error'][:90]}")
    print()


def run_scan(edge, store):
    scan = requests.post(f"{edge}/api/predict/batch", json={"store_id": store}, timeout=120).json()
    print(f"\n{store}: {scan['summary']}  ({scan['total_ms']:.0f} ms)\n")
    for r in scan["results"]:
        print(f"  {r['product_id']} {r['product_name']:22s} tomorrow {r['predicted_demand']:>6.1f}  "
              f"7d {r['forecast_7d_total']:>7.1f}  stock {r['current_stock']:>5}  {r['status']:9s} "
              f"order {r['reorder_qty']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tests", "pos", "scan", "batch"])
    ap.add_argument("--edge", default="http://localhost:5000")
    ap.add_argument("--store", default="STR01")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--file", default="data/sample_batch.csv")
    a = ap.parse_args()
    try:
        requests.get(f"{a.edge}/api/health", timeout=5)
    except requests.RequestException:
        sys.exit(f"Edge service not reachable at {a.edge} - start it first (python edge_service/app.py)")
    {"tests": lambda: run_tests(a.edge), "pos": lambda: run_pos(a.edge, a.store, a.days, a.delay),
     "scan": lambda: run_scan(a.edge, a.store), "batch": lambda: run_batch(a.edge, a.file)}[a.mode]()
