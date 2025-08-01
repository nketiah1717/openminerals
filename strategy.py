
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_strategy(
    df: pd.DataFrame,
    A_id: str,
    B_id: str,
    z_entry: float = 6.0,
    z_exit: float = 0.0,
    notional: float = 100000.0
) -> pd.DataFrame:
    """
    Execute pairs trading based on z-score signals.
    A_id and B_id identify the instruments for naming outputs.
    """
    if df.empty:
        raise ValueError("Signals DataFrame is empty")

    # Contract sizes and commission rates
    contract_sizes = {"lme": 25, "shfe": 5}
    commission_rates = {"lme": 0.00015625, "shfe": 0.00005}

    key_a = "lme" if "lme" in A_id else "shfe"
    key_b = "lme" if "lme" in B_id else "shfe"
    size_a = contract_sizes[key_a]
    size_b = contract_sizes[key_b]
    rate_a = commission_rates[key_a]
    rate_b = commission_rates[key_b]

    position = 0
    entry_price_a = entry_price_b = qty_a = qty_b = 0.0
    pnl_list = []

    for row in df.itertuples(index=False):
        z = row.zscore
        ask_a, bid_a = row.ask_A, row.bid_A
        ask_b, bid_b = row.ask_B, row.bid_B
        spread_a, spread_b = row.spread_A, row.spread_B
        ts = row.timestamp

        #no info on price step so we use spread as slippage
        slippage_a = spread_a
        slippage_b = spread_b

        if position == 0:
            if z > z_entry: #simple strategy logic
                position = -1
                entry_price_a = bid_a - slippage_a #executing on market + slippage (worst scenario)
                entry_price_b = ask_b + slippage_b
                qty_a = notional / entry_price_a
                qty_b = notional / entry_price_b
            elif z < -z_entry:
                position = 1
                entry_price_a = ask_a + slippage_a
                entry_price_b = bid_b - slippage_b
                qty_a = notional / entry_price_a
                qty_b = notional / entry_price_b




        elif position == 1 and z >= z_exit:
            exit_price_a = bid_a - slippage_a
            exit_price_b = ask_b + slippage_b
            comm_a = rate_a * exit_price_a * size_a * qty_a * 2  #comissions for a roundtrip
            comm_b = rate_b * exit_price_b * size_b * qty_b * 2
            pnl = (
                (exit_price_a - entry_price_a) * qty_a
                - (exit_price_b - entry_price_b) * qty_b
                - comm_a - comm_b
            )

            pnl_list.append((ts, pnl))
            position = 0

        elif position == -1 and z <= z_exit:
            exit_price_a = ask_a + slippage_a
            exit_price_b = bid_b - slippage_b
            comm_a = rate_a * exit_price_a * size_a * qty_a * 2
            comm_b = rate_b * exit_price_b * size_b * qty_b * 2
            pnl = (
                (entry_price_a - exit_price_a) * qty_a
                - (entry_price_b - exit_price_b) * qty_b
                - comm_a - comm_b
            )
            pnl_list.append((ts, pnl))
            position = 0

    pnl_df = pd.DataFrame(pnl_list, columns=["timestamp", "pnl"])
    pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])
    pnl_df["cum_pnl"] = pnl_df["pnl"].cumsum()

    out_file = f"pnl_{A_id}_{B_id}.csv"
    pnl_df.to_csv(out_file, index=False)

    # Metrics
    num_trades = len(pnl_df)
    win_rate = (pnl_df["pnl"] > 0).mean()
    avg_pnl = pnl_df["pnl"].mean()
    pnl_std = pnl_df["pnl"].std()
    sharpe = (avg_pnl / pnl_std) * np.sqrt(252) if pnl_std > 0 else float("nan")

    print(f"\n==== Trade Statistics for {A_id} vs {B_id} ====")
    print(f"Total trades  : {num_trades}")
    print(f"Win rate      : {win_rate:.2%}")
    print(f"Average PnL   : {avg_pnl:.2f} USD")
    print(f"Sharpe ratio  : {sharpe:.2f}")

    return pnl_df



def plot_equity_curve(df: pd.DataFrame, output_path: str = "equity_commissions_excluded.png") -> None:
    """
    Plot cumulative PnL curve for given pair.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["cum_pnl"], label="Equity Curve")
    plt.title("Cumulative PnL — Pair")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative PnL (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    signals = pd.read_csv("spread_signals_lme_0_shfe_0.csv")
    df_pnl = run_strategy(signals, A_id="lme_0", B_id="shfe_0")
    plot_equity_curve(df_pnl)

