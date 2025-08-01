
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from typing import List, Tuple


def load_normalized_data(path: str = "normalized_data.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns of mid prices grouped by instrument id."""
    df["logret"] = (
        df.groupby("id")["mid_usd"]
        .transform(lambda x: np.log(x) - np.log(x.shift(1)))
    )
    return df



def get_logret_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot log return series into wide format with timestamps as index."""
    dedup = df.drop_duplicates(subset=["timestamp", "id"])
    return dedup.pivot(index="timestamp", columns="id", values="logret")



def find_cointegrated_pairs(
    pivot: pd.DataFrame, threshold: float = 0.5, min_obs: int = 500
) -> pd.DataFrame:
    """
    Identify cointegrated pairs based on correlation threshold.
    Returns DataFrame sorted by p-value of cointegration test.
    """
    corr = pivot.corr()
    pairs = [
        (a, b, corr.loc[a, b])
        for a in corr.columns
        for b in corr.columns
        if a < b and corr.loc[a, b] > threshold
    ]

    records = []
    for a, b, coeff in pairs:
        s_a = pivot[a].dropna()
        s_b = pivot[b].dropna()
        idx = s_a.index.intersection(s_b.index)
        if len(idx) < min_obs:
            continue
        y = s_a.loc[idx]
        x = s_b.loc[idx]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        beta = model.params.iloc[1]
        p_val = coint(y, x)[1]
        records.append({
            "A": a,
            "B": b,
            "corr": coeff,
            "beta": beta,
            "resid_std": model.resid.std(),
            "pval": p_val,
        })
    return pd.DataFrame(records).sort_values("pval").reset_index(drop=True)


def strategy_preparation(
    df: pd.DataFrame,
    A_id: str = "lme_0",
    B_id: str = "shfe_0"
) -> pd.DataFrame:
    """
    Build spread and z-score series for a specified cointegrated pair.
    """
    dedup = df.drop_duplicates(subset=["timestamp", "id"])
    mid = dedup.pivot(index="timestamp", columns="id", values="mid_usd")
    ask = dedup.pivot(index="timestamp", columns="id", values="ask_usd")
    bid = dedup.pivot(index="timestamp", columns="id", values="bid_usd")
    spread_df = dedup.pivot(index="timestamp", columns="id", values="spread_usd")

    a = mid[A_id].dropna()
    b = mid[B_id].dropna()
    idx = a.index.intersection(b.index)

    y = a.loc[idx]
    x = b.loc[idx]
    model = sm.OLS(y, sm.add_constant(x)).fit()
    beta = model.params.iloc[1]

    spread = y - beta * x
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

    signals = pd.DataFrame({
        "timestamp": idx,
        "A": y,
        "B": x,
        "ask_A": ask[A_id].loc[idx],
        "bid_A": bid[A_id].loc[idx],
        "ask_B": ask[B_id].loc[idx],
        "bid_B": bid[B_id].loc[idx],
        "spread_A": spread_df[A_id].loc[idx],
        "spread_B": spread_df[B_id].loc[idx],
        "model_spread": spread,
        "zscore": z,
        "beta": beta
    }).dropna().reset_index(drop=True)

    filename = f"spread_signals_{A_id}_{B_id}.csv"
    signals.to_csv(filename, index=False)
    return signals

if __name__ == "__main__":
    df = load_normalized_data()
    df = compute_log_returns(df)
    pivot = get_logret_pivot(df)
    df_pairs = find_cointegrated_pairs(pivot)
    print("\nTop cointegrated pairs:")
    print(df_pairs.head(5))
    #generating signals for the TOP pair
    signals = strategy_preparation(df, A_id="lme_0", B_id="shfe_0")
