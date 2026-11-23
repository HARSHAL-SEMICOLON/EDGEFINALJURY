"""
Smart Retail Product Demand Prediction - Dataset Generator
-----------------------------------------------------------
Creates a realistic historical sales dataset for 3 Pune retail stores x 24 products
(Jan 2024 -> Nov 2026, daily), with:
  * weekly pattern (weekend uplift), salary-day effect (1st-5th of month)
  * category seasonality (summer beverages, winter tea, festive sweets/dry fruits)
  * Indian festival effects (Diwali, Holi, Ganesh Chaturthi, Dussehra, Eid, Sankranti, Christmas)
  * promotions / discounts with price elasticity, slow business growth trend
  * realistic noise (over-dispersed counts)
  * ~1% "dirty" rows (missing price, duplicates, negative returns, outliers) so the
    cleaning step in training is meaningful.

Outputs:
  data/products.csv          product master
  data/stores.csv            store master
  data/retail_sales.csv      raw historical sales (with dirty rows)
  data/inventory.csv         current stock snapshot per store-product
  data/festivals.csv         festival calendar used for feature engineering

Run:  python data/generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent

START, END = "2024-01-01", "2026-11-21"   # history ends the day before the jury week

STORES = [
    # store_id, name, size factor
    ("STR01", "Kothrud SuperMart", 1.00),
    ("STR02", "Hinjewadi Daily Needs", 1.25),
    ("STR03", "Hadapsar FreshMart", 0.80),
]

# product_id, name, category, base_daily_units, base_price (INR), price_elasticity
PRODUCTS = [
    ("P101", "Basmati Rice 5kg", "Staples", 14, 649, -0.6),
    ("P102", "Wheat Atta 10kg", "Staples", 18, 459, -0.5),
    ("P103", "Toor Dal 1kg", "Staples", 22, 165, -0.7),
    ("P104", "Sugar 1kg", "Staples", 26, 48, -0.4),
    ("P201", "Toned Milk 1L", "Dairy", 120, 56, -0.3),
    ("P202", "Curd 400g", "Dairy", 45, 35, -0.5),
    ("P203", "Paneer 200g", "Dairy", 30, 90, -0.8),
    ("P204", "Butter 100g", "Dairy", 24, 58, -0.7),
    ("P301", "Tea Powder 500g", "Beverages", 16, 285, -0.6),
    ("P302", "Instant Coffee 100g", "Beverages", 9, 320, -0.8),
    ("P303", "Cold Drink 2L", "Beverages", 28, 95, -1.1),
    ("P304", "Mango Juice 1L", "Beverages", 18, 110, -1.0),
    ("P401", "Potato Chips 52g", "Snacks", 60, 20, -1.2),
    ("P402", "Cream Biscuits 120g", "Snacks", 55, 30, -1.0),
    ("P403", "Namkeen Mix 400g", "Snacks", 20, 110, -0.9),
    ("P404", "Chocolate Bar 50g", "Snacks", 40, 45, -1.1),
    ("P501", "Bath Soap 4x100g", "Personal Care", 15, 199, -0.8),
    ("P502", "Shampoo 340ml", "Personal Care", 8, 285, -0.9),
    ("P503", "Toothpaste 150g", "Personal Care", 17, 105, -0.6),
    ("P504", "Detergent Powder 1kg", "Personal Care", 19, 130, -0.7),
    ("P601", "Onion 1kg", "Fresh Produce", 70, 40, -0.5),
    ("P602", "Potato 1kg", "Fresh Produce", 75, 32, -0.4),
    ("P603", "Tomato 1kg", "Fresh Produce", 60, 38, -0.6),
    ("P604", "Banana 1 dozen", "Fresh Produce", 35, 60, -0.7),
]

FESTIVALS = [
    ("2024-01-14", "Makar Sankranti"), ("2024-03-25", "Holi"), ("2024-04-11", "Eid al-Fitr"),
    ("2024-09-07", "Ganesh Chaturthi"), ("2024-10-12", "Dussehra"), ("2024-11-01", "Diwali"),
    ("2024-12-25", "Christmas"),
    ("2025-01-14", "Makar Sankranti"), ("2025-03-14", "Holi"), ("2025-03-31", "Eid al-Fitr"),
    ("2025-08-27", "Ganesh Chaturthi"), ("2025-10-02", "Dussehra"), ("2025-10-21", "Diwali"),
    ("2025-12-25", "Christmas"),
    ("2026-01-14", "Makar Sankranti"), ("2026-03-04", "Holi"), ("2026-03-20", "Eid al-Fitr"),
    ("2026-09-14", "Ganesh Chaturthi"), ("2026-10-20", "Dussehra"), ("2026-11-08", "Diwali"),
    ("2026-12-25", "Christmas"),
    ("2027-01-14", "Makar Sankranti"),
]

# how strongly each category reacts to a festival (multiplier at peak)
FESTIVAL_CAT_BOOST = {
    "Staples": 1.35, "Dairy": 1.45, "Beverages": 1.40, "Snacks": 1.70,
    "Personal Care": 1.25, "Fresh Produce": 1.20,
}
BIG_FESTIVALS = {"Diwali": 1.6, "Ganesh Chaturthi": 1.2, "Dussehra": 1.1}

# month-of-year seasonality per category (index 1..12)
SEASON = {
    "Beverages":     [0.85, 0.9, 1.05, 1.3, 1.45, 1.25, 0.95, 0.9, 0.95, 1.0, 0.95, 0.9],
    "Dairy":         [1.0, 1.0, 1.0, 1.05, 1.1, 1.05, 0.98, 0.97, 1.0, 1.03, 1.06, 1.02],
    "Snacks":        [1.0, 0.98, 1.0, 1.0, 1.02, 1.05, 1.08, 1.05, 1.03, 1.08, 1.12, 1.05],
    "Staples":       [1.0, 0.98, 1.0, 1.02, 1.0, 1.03, 1.05, 1.02, 1.0, 1.05, 1.08, 1.0],
    "Personal Care": [1.0, 1.0, 1.02, 1.05, 1.08, 1.05, 1.0, 0.98, 1.0, 1.04, 1.06, 1.0],
    "Fresh Produce": [1.05, 1.0, 0.98, 0.95, 0.92, 1.0, 1.05, 1.06, 1.03, 1.02, 1.02, 1.05],
}
# product-level overrides (tea/coffee peak in winter, mango juice in summer)
PRODUCT_SEASON = {
    "P301": [1.25, 1.15, 1.0, 0.85, 0.8, 0.85, 1.0, 1.0, 0.95, 1.0, 1.1, 1.25],
    "P302": [1.2, 1.15, 1.0, 0.9, 0.85, 0.9, 1.0, 1.0, 0.95, 1.0, 1.1, 1.2],
    "P304": [0.7, 0.8, 1.1, 1.6, 1.8, 1.4, 0.9, 0.8, 0.8, 0.85, 0.8, 0.7],
}
DOW = [0.92, 0.88, 0.9, 0.95, 1.05, 1.22, 1.18]  # Mon..Sun


def festival_multiplier(dates: pd.DatetimeIndex, category: str) -> np.ndarray:
    """Ramp-up in the 7 days before a festival, peak on the day, small spill-over after."""
    mult = np.ones(len(dates))
    for fdate, fname in FESTIVALS:
        f = pd.Timestamp(fdate)
        delta = (f - dates).days.values  # >0 means festival is ahead
        boost = FESTIVAL_CAT_BOOST[category] * BIG_FESTIVALS.get(fname, 1.0) - 1.0
        pre = (delta >= 1) & (delta <= 7)
        mult[pre] *= 1 + boost * (1 - (delta[pre] - 1) / 7) * 0.8
        mult[delta == 0] *= 1 + boost
        post = (delta >= -2) & (delta <= -1)
        mult[post] *= 1 + boost * 0.25
    return mult


def main():
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    t = np.arange(n) / 365.0
    rows = []
    for sid, sname, sfac in STORES:
        for pid, pname, cat, base, price, elast in PRODUCTS:
            season = np.array(PRODUCT_SEASON.get(pid, SEASON[cat]))[dates.month - 1]
            dow = np.array(DOW)[dates.dayofweek]
            salary = np.where(dates.day <= 5, 1.08, 1.0)
            trend = 1 + 0.06 * t  # ~6% yearly growth
            fest = festival_multiplier(dates, cat)

            # promotions: ~10% of days, clustered in 3-5 day runs, bigger near festivals
            promo = np.zeros(n, dtype=int)
            i = 0
            while i < n:
                if RNG.random() < 0.035:
                    run = RNG.integers(3, 6)
                    promo[i:i + run] = 1
                    i += run
                else:
                    i += 1
            near_fest = fest > 1.05
            promo[near_fest & (RNG.random(n) < 0.35)] = 1
            discount = np.where(promo == 1, RNG.choice([5, 10, 15, 20, 25], n,
                                                       p=[.2, .35, .25, .15, .05]), 0)
            # inflation drift in list price (~5%/yr) + small noise
            unit_price = price * (1 + 0.05 * t) * RNG.normal(1, 0.01, n)
            eff_price = unit_price * (1 - discount / 100)
            price_effect = (eff_price / (price * (1 + 0.05 * t))) ** elast
            promo_effect = np.where(promo == 1, 1.12, 1.0)  # visibility / end-cap display

            mu = base * sfac * season * dow * salary * trend * fest * price_effect * promo_effect
            # over-dispersed counts: gamma-poisson (negative binomial)
            shape = 30.0
            lam = RNG.gamma(shape, mu / shape)
            units = RNG.poisson(lam)

            df = pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "store_id": sid, "store_name": sname,
                "product_id": pid, "product_name": pname, "category": cat,
                "unit_price": np.round(unit_price, 2),
                "discount_pct": discount, "promo_flag": promo,
                "units_sold": units,
            })
            rows.append(df)

    sales = pd.concat(rows, ignore_index=True)
    fest_map = {d: name for d, name in FESTIVALS}
    sales["festival"] = sales["date"].map(fest_map).fillna("None")

    # ---- inject realistic data-quality problems (handled in training) ----
    m = len(sales)
    idx = RNG.choice(m, int(m * 0.004), replace=False)
    sales.loc[idx, "unit_price"] = np.nan                       # missing price
    idx = RNG.choice(m, int(m * 0.002), replace=False)
    sales.loc[idx, "units_sold"] = -RNG.integers(1, 5, len(idx))  # returns booked as negative
    idx = RNG.choice(m, int(m * 0.0008), replace=False)
    sales.loc[idx, "units_sold"] = sales.loc[idx, "units_sold"] * 12  # data-entry outliers
    dup = sales.sample(int(m * 0.003), random_state=1)           # duplicate POS uploads
    sales = pd.concat([sales, dup], ignore_index=True).sort_values(
        ["date", "store_id", "product_id"]).reset_index(drop=True)

    sales.to_csv(OUT / "retail_sales.csv", index=False)
    pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "category", "base_daily_units",
                                    "base_price", "price_elasticity"]).to_csv(OUT / "products.csv", index=False)
    pd.DataFrame(STORES, columns=["store_id", "store_name", "size_factor"]).to_csv(OUT / "stores.csv", index=False)
    pd.DataFrame(FESTIVALS, columns=["date", "festival"]).to_csv(OUT / "festivals.csv", index=False)

    # ---- current inventory snapshot: deliberately mixed (some low, some overstocked) ----
    inv = []
    for sid, sname, sfac in STORES:
        for pid, pname, cat, base, price, elast in PRODUCTS:
            weekly = base * sfac * 7
            cover_days = RNG.choice([0.8, 3, 6, 9, 12, 16, 26], p=[.06, .10, .12, .22, .24, .16, .10])
            inv.append({
                "store_id": sid, "product_id": pid,
                "current_stock": int(round(weekly / 7 * cover_days)),
                "lead_time_days": 2 if cat in ("Dairy", "Fresh Produce") else (3 if cat == "Snacks" else 5),
                "safety_stock_days": 1 if cat in ("Dairy", "Fresh Produce") else 2,
            })
    pd.DataFrame(inv).to_csv(OUT / "inventory.csv", index=False)

    print(f"retail_sales.csv : {len(sales):,} rows  ({sales['date'].min()} -> {sales['date'].max()})")
    print(f"stores={len(STORES)} products={len(PRODUCTS)} festivals={len(FESTIVALS)}")


if __name__ == "__main__":
    main()
