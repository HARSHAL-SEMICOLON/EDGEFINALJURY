# Smart Retail Product Demand Prediction System — Edge Analytics

**Course:** Applied Edge Analytics (23CSE4019) · Jury Examination · Semester VII · Problem Statement #3
**Goal:** Predict product-wise demand for a retail store **locally on an edge node**, compare it with current stock, flag products that need replenishment, and sync the results to a central dashboard over **HTTP / MQTT**.

---

## 1. Quick start (Windows)

| Step | Do this | Result |
|---|---|---|
| 1 | Double-click **`setup.bat`** (once, needs internet, ~2–3 min) | Creates `.venv`, installs packages, generates the dataset, trains the models |
| 2 | Double-click **`start_demo.bat`** | Opens **Edge UI → http://localhost:5000** and **Dashboard → http://localhost:8050** |
| 3 | `run_simulator.bat tests` | Sends 19 normal / abnormal / invalid inputs to the edge service |
| 4 | `run_simulator.bat pos --store STR01 --days 3` | Streams POS sales into the edge, stock drops, alerts fire |
| 5 | `run_simulator.bat batch --file data\sample_batch_with_errors.csv` | Batch CSV → 3 predicted, 7 bad rows rejected with reasons |
| 6 | `run_tests.bat` | 26 automated pytest checks |

> Needs Python 3.10+ (tick **"Add to PATH"** when installing). Nothing needs the internet after setup, so the demo runs offline.
> Manual run: `python dashboard\app.py` and `python edge_service\app.py` in two terminals.

**Docker (optional, full MQTT stack):** `docker compose up --build` starts Mosquitto, the edge service and the dashboard (HTTP + MQTT).

---

## 2. Architecture

```
 ┌──────────────── STORE (EDGE) ───────────────────────────┐        ┌──── HEAD OFFICE ────┐
 │                                                         │        │                     │
 │  POS / handheld simulator ──HTTP──►  EDGE SERVICE :5000 │  HTTP  │  CENTRAL DASHBOARD  │
 │  Store manager UI (browser) ─────►   Flask microservice ├──POST─►│  Flask :8050        │
 │                                      • Random Forest    │  /api/ │  • KPIs, alerts     │
 │                                        (local inference)│  ingest│  • demand vs stock  │
 │                                      • 7-day forecast   │        │  • node heartbeats  │
 │                                      • replenishment    │  MQTT  │                     │
 │                                        rules + alerts   ├─retail/►  (subscribes       │
 │                                      • SQLite edge store│  <store>/ retail/#)         │
 │                                        + outbox (store- │  topic │                     │
 │                                        and-forward)     │        │                     │
 └─────────────────────────────────────────────────────────┘        └─────────────────────┘
   raw POS data stays in the store; only compact prediction summaries + alerts are sent upstream
```

| Component | File | Role |
|---|---|---|
| Dataset generator | `data/generate_dataset.py` | 3 Pune stores × 24 products × 1,056 days (Jan 2024 – 21 Nov 2026) |
| Feature engineering | `ml/features.py` | **Same code** for training and edge inference (no training/serving skew) |
| Data cleaning | `ml/cleaning.py` | Duplicates, negative returns, outliers, missing prices, date gaps |
| Training & evaluation | `ml/train_model.py` | Baseline vs Linear vs Decision Tree vs **Random Forest** → `models/`, `reports/` |
| Edge engine | `edge_service/engine.py` | Local inference, 7-day recursive forecast, replenishment logic, SQLite store |
| Edge API + UI | `edge_service/app.py`, `templates/index.html` | REST microservice + store manager UI + background sync thread |
| Central dashboard | `dashboard/app.py`, `templates/dashboard.html` | Receives HTTP/MQTT, shows KPIs, alerts, product-wise estimates |
| Simulator | `simulator/store_simulator.py` | POS terminal / test-case client |
| Tests | `tests/test_edge_service.py` | 26 tests: normal, abnormal and invalid inputs, batch CSV, business rules |

---

### Batch CSV format
```
store_id,product_id,date,unit_price,discount_pct,promo_flag,current_stock,lead_time_days
STR01,P201,2026-11-22,63.4,0,0,180,2
STR01,P503,,,,,,          <- blanks are filled from the edge store (last price, stock, lead time)
```
Samples: `data/sample_batch.csv` (12 valid rows) and `data/sample_batch_with_errors.csv` (validation demo).

---

## 3. Dataset & preprocessing

- **Columns:** date, store_id, product_id, product_name, category, unit_price, discount_pct, promo_flag, festival, units_sold (target)
- **6 categories:** Staples, Dairy, Beverages, Snacks, Personal Care, Fresh Produce
- **Patterns built in:** weekend uplift, salary-week (1st–5th) uplift, category seasonality (summer beverages, winter tea/coffee), **Indian festivals** (Diwali, Ganesh Chaturthi, Dussehra, Holi, Eid, Sankranti, Christmas) with a 7-day pre-festival ramp, promotions and price elasticity, ~6%/year growth, realistic noise
- **Dirty data injected on purpose and cleaned:**

| Issue | Rows | Fix |
|---|---|---|
| Duplicate POS uploads | 228 | dropped |
| Negative units (returns) | 152 | treated as missing → interpolated |
| Data-entry outliers (>5× median) | 62 | treated as missing → interpolated |
| Missing price | 304 | interpolated per product |

- **Split:** time-based, no shuffling → **train** 29 Jan 2024 – 30 Jun 2026 (63,648 rows) · **test** 1 Jul – 21 Nov 2026 (10,368 rows, includes Ganesh Chaturthi, Dussehra and Diwali 2026)

**Features (26):** store/product/category codes · day of week, day of month, month, week, weekend, salary week · is_festival, days_to_festival, days_since_festival, big-festival window · unit_price, discount_pct, promo_flag, effective_price, price_index · lag_1, lag_7, lag_14, lag_28, rolling mean 7/28, rolling std 7, same-weekday mean of the last 4 weeks.

---

## 4. Model results (unseen test period)

| Model | MAE | RMSE | R² | WAPE |
|---|---|---|---|---|
| Seasonal naive (same day last week) | 17.85 | 31.76 | 0.516 | 35.9% |
| Linear Regression | 11.70 | 19.90 | 0.810 | 23.5% |
| Decision Tree Regressor | 12.61 | 21.59 | 0.777 | 25.3% |
| **Random Forest Regressor** ✅ | **11.24** | **19.27** | **0.822** | **22.6%** |

- **Random Forest** (150 trees, max_depth 18, min_samples_leaf 4, max_features 0.6) is deployed: best on every metric, robust to noise, handles mixed categorical and numeric features, and gives an uncertainty band from the spread of its trees.
- **Top features:** rolling mean 28d (0.38), rolling mean 7d (0.31), same-weekday mean (0.11), lag-1 (0.06), day of week.
- **Edge profile:** model 36 MB · **~6 ms per single prediction** · a full API request (8 model calls + interval) takes ~100 ms · a 24-product store scan takes ~2.5 s on a laptop CPU.
- Plots in `reports/`: `model_comparison.png`, `feature_importance.png`, `actual_vs_predicted.png`, `pred_vs_actual_scatter.png`.

---

## 5. Replenishment logic (stock vs predicted demand)

```
reorder_point = demand over supplier lead time  +  avg daily demand × safety-stock days
CRITICAL   stock < tomorrow's predicted demand          → urgent order
REORDER    stock < reorder_point                        → place order now
OVERSTOCK  stock > 3 × 7-day predicted demand           → pause ordering
OK         otherwise
order qty  = demand over (2 × lead time, max 7 days) + safety stock − current stock
```
Lead time: Dairy/Fresh 2 days, Snacks 3 days, others 5 days. Perishables are never ordered up to a full week.

---

## 6. Edge service API (port 5000)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict` | Product input → demand, 7-day forecast, 80% band, status, order qty, latency |
| POST | `/api/predict/upload` | **Batch CSV upload** (form field `file`, max 200 rows) → per-row prediction or rejection reason |
| GET | `/api/batch/template` | Download the CSV template |
| POST | `/api/predict/batch` | `{"store_id":"STR01"}` → whole-store scan, or `{"items":[...]}` (JSON batch) |
| POST | `/api/sales` | POS ingestion: updates local history and decrements stock |
| POST | `/api/inventory` | Set stock or `add_units` (restock received) |
| GET | `/api/context` | Auto-fill: last price, stock, last 28 days of sales, upcoming festivals |
| GET | `/api/catalog`, `/api/history`, `/api/alerts`, `/api/health` | Catalogue, records, alerts, node status |

Example:
```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" ^
  -d "{\"store_id\":\"STR02\",\"product_id\":\"P201\",\"discount_pct\":10,\"current_stock\":300}"
```

**Communication modes** (env `COMM_MODE`): `http` (default) → `POST dashboard/api/ingest` · `mqtt` → topics `retail/<store>/prediction|alert|heartbeat` · `both`.
**Store-and-forward:** every result goes into a local SQLite **outbox** first. If the dashboard or broker is down, predictions still work and the results are queued; they sync automatically when the link returns.

---

## 7. Jury demo script (~6 minutes)

1. **Problem (30 s):** stock-outs lose sales and overstock wastes money, perishables especially. Edge processing means predictions work even when the internet is down, respond in milliseconds, keep raw sales data inside the store, and cut bandwidth because only summaries go upstream.
2. **Data and model (60 s):** open `reports/model_comparison.png` and `actual_vs_predicted.png`. Random Forest R² 0.82 vs 0.52 for the naive baseline; Diwali spikes are visible.
3. **Live prediction (90 s):** Edge UI → STR02 · Toned Milk → **Predict**. Show demand, 80% range, 7-day chart, stock vs reorder point and the "inferred locally in ~100 ms" line. Then change **discount to 20% + promotion** → demand rises. Set **stock to 20** → CRITICAL with an order quantity.
4. **Communication (45 s):** switch to the Dashboard; the prediction and alert appear, tagged HTTP. Show the edge node heartbeat.
5. **Store scan (30 s):** **Scan whole store** → 24 products ranked CRITICAL / REORDER / OK.
6. **Abnormal and invalid inputs (45 s):** price −5 → rejected; price ₹450 for paneer → prediction with a warning; then `run_simulator.bat tests` → 19/19 cases behave as expected.
7. **Batch prediction (45 s):** Edge UI → *Batch prediction · CSV upload* → choose `data/sample_batch_with_errors.csv` → **Run batch at the edge**. 3 rows are predicted and 7 bad rows are rejected with their row numbers and reasons; the batch doesn't crash. Click **Export results CSV** to download the results.
8. **Edge resilience (45 s):** close the dashboard window, make 2 predictions → the edge chip shows "Offline — N queued". Restart the dashboard → the queue flushes automatically.
9. **POS stream (optional):** `run_simulator.bat pos --store STR01 --days 2` → stock falls and alerts appear on the dashboard.

---

## 8. Testing summary

| Type | Cases | Expected | Status |
|---|---|---|---|
| Normal | milk, promo chips, weekend, low stock, overstock | prediction + correct status | ✅ |
| Varied | festival date, what-if spike in sales history | prediction follows the scenario | ✅ |
| Abnormal | price 5× normal, 60% discount | prediction **with warning** (outside training range) | ✅ |
| Invalid | unknown product/store, negative price/stock, discount >90, text in a number field, bad date, >60 days ahead, <28 days history, negative history | **HTTP 400** with a clear reason | ✅ |
| Batch CSV | 12-row valid file; 10-row file with 7 bad rows; missing column, empty file, header only, non-CSV | all predicted / bad rows reported with Excel row number, good rows still predicted / whole file rejected with reason | ✅ |
| Business rules | CRITICAL / REORDER / OK / OVERSTOCK thresholds, order quantity | exact values | ✅ |
| Comms | HTTP sync, MQTT sync, offline queue + recovery | delivered / queued / flushed | ✅ |

---

## 9. Limitations (state these honestly in the viva)

- The dataset is **simulated** with realistic Indian retail patterns, not a real chain's POS export; retrain on real data before any real use.
- Random Forests **cannot extrapolate**: prices or discounts outside the training range produce a warning, and the estimate stays near known levels.
- The 7-day forecast is **recursive** (each day's prediction feeds the next), so error grows with the horizon; the 80% band comes from tree spread and is not a calibrated interval.
- **Festival peaks are under-predicted** (Diwali 2026 milk: actual ~750 vs predicted ~350, see `reports/actual_vs_predicted.png`). The training history holds only **two Diwalis**, so the forest averages the spike away. Tried and rejected: a ratio-to-28-day-mean target (worse, R² 0.78) and festival-identity / days-to-Diwali features (no gain). Fix with more years of data or a separate festival-uplift multiplier learned from past festivals; until then, store managers should add a manual festival buffer.
- The 7-day forecast assumes the entered price and discount stay the same all week.
- The model is not retrained on the edge automatically; the planned path is periodic retraining in the cloud and pushing the new model to the edge.

---

## 10. Likely viva questions

- **Why the edge and not the cloud?** Works offline (store-and-forward), millisecond latency at the counter, sales data stays local (privacy), lower bandwidth because each summary is a few hundred bytes instead of the full POS stream, and it scales because each store computes for itself.
- **Why Random Forest over Decision Tree?** Bagging many decorrelated trees reduces variance: R² 0.82 vs 0.78 here. It also gives an uncertainty estimate from tree spread.
- **Why a time-based split?** A random split leaks future information through the lag features; the model has to be judged on genuinely future dates.
- **Why lag features dominate?** Recent demand level is the strongest signal; calendar, festival and promo features adjust around it.
- **HTTP vs MQTT?** HTTP is request/response and simple. MQTT is publish/subscribe and lightweight, with QoS and decoupling, suited to many edge nodes and unreliable networks.
- **Why is Diwali under-predicted?** Rare event: only 2 occurrences in training, and tree ensembles average extremes. Next step: more history plus a festival-uplift model.
- **How to scale to 500 stores?** One edge node per store, an MQTT broker cluster, the central dashboard subscribes to `retail/#`, and models are versioned and pushed from the cloud (e.g. Azure IoT Edge modules).
