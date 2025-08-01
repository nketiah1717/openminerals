
from typing import Tuple

import numpy as np
import pandas as pd



def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet("data.parquet")
    fx = pd.read_csv("fx_rates_intraday.csv")
    return df, fx


def normalize(df: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trading data by aligning timestamps with forex rates,
    converting SHFE prices to USD, and computing spread and mid prices.
    """
    # work on copies to avoid side effects
    df_copy = df.copy()
    fx_copy = fx.copy()

    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], utc=True)
    fx_copy["timestamp"] = pd.to_datetime(fx_copy["timestamp"], utc=True)

    fx_rates = fx_copy[["timestamp", "bid"]].rename(columns={"bid": "forex_bid"})

    df_sorted = df_copy.sort_values("timestamp")
    fx_sorted = fx_rates.sort_values("timestamp")

    merged = pd.merge_asof(df_sorted, fx_sorted, on="timestamp", direction="backward")

    clean = merged.dropna(subset=["ask", "bid", "forex_bid"]).copy()

    if "id" not in clean.columns:
        raise KeyError("Missing id")

    is_shfe = clean["id"].str.startswith("shfe", na=False)

    # calculating rates for both sides for future buys and sells
    clean["ask_usd"] = np.where(is_shfe, clean["ask"] / clean["forex_bid"], clean["ask"])
    clean["bid_usd"] = np.where(is_shfe, clean["bid"] / clean["forex_bid"], clean["bid"])

    clean["spread_usd"] = clean["ask_usd"] - clean["bid_usd"]
    clean["mid_usd"] = (clean["ask_usd"] + clean["bid_usd"]) / 2

    clean.to_parquet("normalized_data.parquet", index=False)
    return clean



if __name__ == "__main__":
    raw_df, fx_df = load_data()
    normalized_df = normalize(raw_df, fx_df)

