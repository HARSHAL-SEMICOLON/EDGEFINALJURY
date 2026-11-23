"""
Smart Retail Product Demand Prediction - Model Training
--------------------------------------------------------
1. Load raw POS history, clean it (duplicates, negative returns, outliers, missing prices, gaps)
2. Engineer features (calendar, festival, price/promo, lags & rolling demand)
3. Time-based split: train <= 2026-06-30 | test = 2026-07-01 .. 2026-11-21 (unseen future)
4. Compare: Seasonal-naive baseline, Linear Regression, Decision Tree Regressor, Random Forest Regressor
5. Export best edge-friendly model (Random Forest) + metadata + evaluation reports/plots

Run:  python ml/train_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.features import FEATURES, TARGET, FestivalCalendar, add_training_features  # noqa: E402
from ml.cleaning import clean  # noqa: E402

DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
SPLIT_DATE = "2026-07-01"


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": round(float(mean_absolute_error(y, p)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
        "R2": round(float(r2_score(y, p)), 4),
        "WAPE_%": round(float(np.abs(y - p).sum() / y.sum() * 100), 2),
    }


def main():
    t0 = time.time()
    raw = pd.read_csv(DATA / "retail_sales.csv")
    products = pd.read_csv(DATA / "products.csv")
    stores = pd.read_csv(DATA / "stores.csv")
    festivals = pd.read_csv(DATA / "festivals.csv")
    cal = FestivalCalendar(festivals.itertuples(index=False, name=None))

    df, clean_rep = clean(raw)
    print("Cleaning:", clean_rep)

    meta = {
        "store_codes": {s: i for i, s in enumerate(sorted(stores.store_id))},
        "product_codes": {p: i for i, p in enumerate(sorted(products.product_id))},
        "category_codes": {c: i for i, c in enumerate(sorted(products.category.unique()))},
        "ref_price": df.groupby("product_id")["unit_price"].median().round(2).to_dict(),
        "products": {r.product_id: {"name": r.product_name, "category": r.category}
                     for r in products.itertuples()},
        "stores": {r.store_id: r.store_name for r in stores.itertuples()},
    }

    feat = add_training_features(df, cal, meta)
    train = feat[feat["date"] < SPLIT_DATE]
    test = feat[feat["date"] >= SPLIT_DATE]
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    print(f"Train rows {len(train):,} ({train.date.min().date()}..{train.date.max().date()}) | "
          f"Test rows {len(test):,} ({test.date.min().date()}..{test.date.max().date()})")

    results, preds, fitted = {}, {}, {}
    preds["Seasonal Naive (lag-7)"] = test["lag_7"].values
    results["Seasonal Naive (lag-7)"] = metrics(yte, preds["Seasonal Naive (lag-7)"])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=14, min_samples_leaf=10, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=18, min_samples_leaf=4, max_features=0.6,
            n_jobs=-1, random_state=42),
    }
    for name, model in candidates.items():
        s = time.time()
        model.fit(Xtr, ytr)
        tt = time.time() - s
        p = np.clip(model.predict(Xte), 0, None)
        preds[name], fitted[name] = p, model
        results[name] = metrics(yte, p) | {"train_time_s": round(tt, 2)}
        print(f"{name:26s} {results[name]}")

    rf = fitted["Random Forest Regressor"]
    model_path = MODELS / "demand_rf.joblib"
    joblib.dump(rf, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6

    # edge latency benchmark (single-row inference, as the edge service does it)
    one = Xte.iloc[[0]]
    rf.set_params(n_jobs=1)
    for _ in range(5):
        rf.predict(one)
    s = time.perf_counter()
    for _ in range(100):
        rf.predict(one)
    lat_ms = (time.perf_counter() - s) / 100 * 1000
    s = time.perf_counter()
    rf.predict(Xte)
    batch_ms = (time.perf_counter() - s) * 1000

    # per-category accuracy (RF)
    tcopy = test[["category", TARGET]].copy()
    tcopy["pred"] = preds["Random Forest Regressor"]
    per_cat = {c: metrics(g[TARGET], g["pred"]) for c, g in tcopy.groupby("category")}

    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    meta.update({
        "features": FEATURES,
        "target": TARGET,
        "model_file": model_path.name,
        "model_type": "RandomForestRegressor",
        "model_params": {k: v for k, v in rf.get_params().items() if k in
                         ("n_estimators", "max_depth", "min_samples_leaf", "max_features")},
        "sklearn_version": sklearn.__version__,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "train_range": [str(train.date.min().date()), str(train.date.max().date())],
        "test_range": [str(test.date.min().date()), str(test.date.max().date())],
        "metrics_test": results["Random Forest Regressor"],
        "model_size_mb": round(size_mb, 2),
        "edge_latency_ms_single": round(lat_ms, 2),
        "festivals": festivals.values.tolist(),
    })
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

    report = {
        "cleaning": clean_rep,
        "split": {"train_rows": int(len(train)), "test_rows": int(len(test)), "split_date": SPLIT_DATE},
        "model_comparison": results,
        "random_forest_per_category": per_cat,
        "feature_importance": imp.round(4).to_dict(),
        "edge_profile": {"model_size_mb": round(size_mb, 2), "single_inference_ms": round(lat_ms, 2),
                         f"batch_{len(Xte)}_rows_ms": round(batch_ms, 1)},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))

    # ---------------- plots ----------------
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    names = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].barh(names, [results[n]["MAE"] for n in names], color=["#9aa5b1", "#9aa5b1", "#f0a04b", "#2f7d6d"])
    ax[0].set_title("MAE on unseen test period (lower is better)")
    ax[1].barh(names, [results[n]["R2"] for n in names], color=["#9aa5b1", "#9aa5b1", "#f0a04b", "#2f7d6d"])
    ax[1].set_title("R² on unseen test period (higher is better)")
    ax[1].set_yticklabels([])
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp.head(15)[::-1]
    ax.barh(top.index, top.values, color="#2f7d6d")
    ax.set_title("Random Forest - top 15 feature importances")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png")
    plt.close(fig)

    tcopy = test[["date", "store_id", "product_id", TARGET]].copy()
    tcopy["RF"] = preds["Random Forest Regressor"]
    tcopy["DT"] = preds["Decision Tree Regressor"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a, (sid, pid) in zip(axes, [("STR02", "P201"), ("STR01", "P403")]):
        g = tcopy[(tcopy.store_id == sid) & (tcopy.product_id == pid)]
        a.plot(g.date, g[TARGET], color="#333", lw=1.2, label="Actual")
        a.plot(g.date, g.RF, color="#2f7d6d", lw=1.4, label="Random Forest")
        a.plot(g.date, g.DT, color="#f0a04b", lw=1, alpha=.8, label="Decision Tree")
        for fd, fn in festivals.values:
            fdt = pd.Timestamp(fd)
            if g.date.min() <= fdt <= g.date.max():
                a.axvline(fdt, color="#c0392b", ls=":", lw=1)
                a.text(fdt, a.get_ylim()[1] * 0.95, fn, fontsize=7, color="#c0392b", ha="center")
        a.set_title(f"{meta['stores'][sid]} - {meta['products'][pid]['name']}")
        a.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "actual_vs_predicted.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(yte, preds["Random Forest Regressor"], s=3, alpha=.25, color="#2f7d6d")
    lim = [0, float(max(yte.max(), preds["Random Forest Regressor"].max()))]
    ax.plot(lim, lim, color="#c0392b", lw=1)
    ax.set_xlabel("Actual units")
    ax.set_ylabel("Predicted units")
    ax.set_title("Random Forest - predicted vs actual (test)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "pred_vs_actual_scatter.png")
    plt.close(fig)

    print(f"\nSaved {model_path.name} ({size_mb:.1f} MB), single-row edge latency {lat_ms:.1f} ms")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
