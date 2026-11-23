"""Data cleaning shared by training and edge seeding."""
import numpy as np
import pandas as pd

from ml.features import TARGET


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"raw_rows": int(len(raw))}
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    before = len(df)
    df = df.drop_duplicates(subset=["date", "store_id", "product_id"], keep="first")
    rep["duplicates_removed"] = int(before - len(df))

    neg = df[TARGET] < 0
    rep["negative_units_fixed"] = int(neg.sum())
    df.loc[neg, TARGET] = np.nan  # returns are not demand -> impute

    med = df.groupby(["store_id", "product_id"])[TARGET].transform("median")
    out = df[TARGET] > 5 * med
    rep["outliers_fixed"] = int(out.sum())
    df.loc[out, TARGET] = np.nan

    rep["missing_price_filled"] = int(df["unit_price"].isna().sum())
    parts = []
    for (s, p), g in df.groupby(["store_id", "product_id"]):
        g = g.set_index("date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)
        g.index.name = "date"
        for c in ["store_id", "store_name", "product_id", "product_name", "category"]:
            g[c] = g[c].ffill().bfill()
        g["unit_price"] = g["unit_price"].interpolate().ffill().bfill()
        g["discount_pct"] = g["discount_pct"].fillna(0)
        g["promo_flag"] = g["promo_flag"].fillna(0).astype(int)
        g[TARGET] = g[TARGET].interpolate(limit_direction="both").round()
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True)
    rep["clean_rows"] = int(len(df))
    return df, rep
