"""
Shared feature engineering for training AND edge inference.
Keeping one implementation guarantees the edge service builds exactly the same
feature vector the model was trained on (no training/serving skew).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FEATURES = [
    # identity
    "store_code", "product_code", "category_code",
    # calendar
    "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend", "is_salary_week",
    # festival
    "is_festival", "days_to_festival", "days_since_festival", "is_big_festival_window",
    # price / promotion
    "unit_price", "discount_pct", "promo_flag", "effective_price", "price_index",
    # demand history (lags & rolling stats)
    "lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7",
    "same_weekday_mean_4",
]
TARGET = "units_sold"
BIG_FESTIVALS = {"Diwali", "Ganesh Chaturthi", "Dussehra"}
MIN_HISTORY_DAYS = 28


def _to_ts(d) -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return d.normalize()
    if isinstance(d, (date, datetime)):
        return pd.Timestamp(d).normalize()
    return pd.Timestamp(str(d)).normalize()


class FestivalCalendar:
    def __init__(self, festivals: Iterable[tuple[str, str]]):
        items = sorted((_to_ts(d), n) for d, n in festivals)
        self.dates = np.array([d.value for d, _ in items], dtype="int64")
        self.names = [n for _, n in items]

    def features(self, dates: pd.DatetimeIndex | Sequence) -> pd.DataFrame:
        dts = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        v = dts.values.astype("int64")
        day = 86_400 * 10**9
        pos = np.searchsorted(self.dates, v, side="left")           # next festival on/after date
        nxt = np.where(pos < len(self.dates), self.dates[np.minimum(pos, len(self.dates) - 1)], np.iinfo("int64").max)
        prv_pos = np.searchsorted(self.dates, v, side="right") - 1  # previous festival on/before
        prv = np.where(prv_pos >= 0, self.dates[np.maximum(prv_pos, 0)], np.iinfo("int64").min)
        days_to = np.where(nxt == np.iinfo("int64").max, 60, (nxt - v) // day)
        days_since = np.where(prv == np.iinfo("int64").min, 60, (v - prv) // day)
        next_name = np.array([self.names[p] if p < len(self.names) else "" for p in pos])
        big_window = np.isin(next_name, list(BIG_FESTIVALS)) & (days_to <= 7)
        return pd.DataFrame({
            "is_festival": (days_to == 0).astype(int),
            "days_to_festival": np.clip(days_to, 0, 30).astype(int),
            "days_since_festival": np.clip(days_since, 0, 10).astype(int),
            "is_big_festival_window": big_window.astype(int),
            "upcoming_festival": np.where(days_to <= 14, next_name, "None"),
        }, index=range(len(dts)))

    def upcoming(self, d, horizon=30):
        t = _to_ts(d).value
        out = []
        for dv, n in zip(self.dates, self.names):
            dd = (dv - t) // (86_400 * 10**9)
            if 0 <= dd <= horizon:
                out.append({"festival": n, "date": str(pd.Timestamp(dv).date()), "days_away": int(dd)})
        return out


def calendar_features(dates) -> pd.DataFrame:
    dts = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame({
        "day_of_week": dts.dayofweek,
        "day_of_month": dts.day,
        "month": dts.month,
        "week_of_year": dts.isocalendar().week.astype(int).values,
        "is_weekend": (dts.dayofweek >= 5).astype(int),
        "is_salary_week": (dts.day <= 5).astype(int),
    }, index=range(len(dts)))


def history_features(history: Sequence[float]) -> dict:
    """history = daily units ordered oldest -> newest, ending the day BEFORE the target date."""
    h = np.asarray(history, dtype=float)
    if len(h) < MIN_HISTORY_DAYS:
        raise ValueError(f"need at least {MIN_HISTORY_DAYS} days of sales history, got {len(h)}")
    return {
        "lag_1": h[-1], "lag_7": h[-7], "lag_14": h[-14], "lag_28": h[-28],
        "roll_mean_7": h[-7:].mean(), "roll_mean_28": h[-28:].mean(),
        "roll_std_7": h[-7:].std(ddof=1),
        "same_weekday_mean_4": np.mean([h[-7], h[-14], h[-21], h[-28]]),
    }


def add_training_features(df: pd.DataFrame, cal: FestivalCalendar, meta: dict) -> pd.DataFrame:
    """df: cleaned, gap-free daily rows per store-product, sorted by date."""
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    g = df.groupby(["store_id", "product_id"])[TARGET]
    s1 = g.shift(1)
    df["lag_1"] = s1
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_21"] = g.shift(21)
    df["lag_28"] = g.shift(28)
    grp = s1.groupby([df["store_id"], df["product_id"]])
    df["roll_mean_7"] = grp.transform(lambda x: x.rolling(7).mean())
    df["roll_mean_28"] = grp.transform(lambda x: x.rolling(28).mean())
    df["roll_std_7"] = grp.transform(lambda x: x.rolling(7).std())
    df["same_weekday_mean_4"] = df[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)

    cal_df = calendar_features(df["date"])
    fest_df = cal.features(df["date"])
    df = pd.concat([df, cal_df, fest_df.drop(columns=["upcoming_festival"])], axis=1)
    df = encode_static(df, meta)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def encode_static(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    df["store_code"] = df["store_id"].map(meta["store_codes"])
    df["product_code"] = df["product_id"].map(meta["product_codes"])
    df["category_code"] = df["category"].map(meta["category_codes"])
    df["effective_price"] = df["unit_price"] * (1 - df["discount_pct"] / 100.0)
    ref = df["product_id"].map(meta["ref_price"])
    df["price_index"] = df["effective_price"] / ref
    return df


def build_inference_row(*, store_id: str, product_id: str, target_date, unit_price: float,
                        discount_pct: float, promo_flag: int, history: Sequence[float],
                        cal: FestivalCalendar, meta: dict) -> pd.DataFrame:
    prod = meta["products"][product_id]
    row = {
        "store_id": store_id, "product_id": product_id, "category": prod["category"],
        "unit_price": float(unit_price), "discount_pct": float(discount_pct), "promo_flag": int(promo_flag),
    }
    row.update(history_features(history))
    base = pd.DataFrame([row])
    base = pd.concat([base, calendar_features([target_date]),
                      cal.features([target_date]).drop(columns=["upcoming_festival"])], axis=1)
    base = encode_static(base, meta)
    return base[FEATURES]
