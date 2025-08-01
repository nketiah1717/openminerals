
# Pairs Trading Signal Generation and Backtest

This project builds a research and backtesting pipeline for identifying cointegrated futures pairs (e.g., LME and SHFE instruments), generating spread signals, and simulating a realistic trading strategy with slippage. The entire pipeline includes:

- data normalization and cleaning,
- cointegration pair discovery,
- signal construction (spread & z-score),
- strategy execution with slippage,
- metric logging and equity curve plotting.

---

##  Project Structure

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `data.py`          | Loads raw futures and FX data, converts SHFE quotes to USD, computes mid/spread |
| `research.py`      | Computes log returns, finds cointegrated pairs, builds spread/zscore signals  |
| `strategy.py`      | Simulates a trading strategy with slippage and computes trade performance     |
| `normalized_data.parquet` | Output from `data.py`, cleaned and normalized dataset                 |
| `spread_signals_*.csv`    | Signals for a given cointegrated pair                               |
| `pnl_*.csv`         | Realized trade PnL for the selected pair                                   |

---

## Data Processing (`data.py`)

1. Load futures data (`data.parquet`) and FX quotes (`fx_rates_intraday.csv`)
2. Merge intraday timestamps with FX rates
3. Convert SHFE quotes to USD:
   ```python
   ask_usd = ask / forex_bid
   bid_usd = bid / forex_bid
   ```
4. Compute:
   - `mid_usd = (ask_usd + bid_usd) / 2`
   - `spread_usd = ask_usd - bid_usd`
5. Save to `normalized_data.parquet`

---

##  Cointegration Research (`research.py`)

### Step 1: Log-Returns and Pivot

```python
logret = log(mid_usd) - log(mid_usd.shift(1))
pivot = df.pivot(index='timestamp', columns='id', values='logret')
```

### Step 2: Pair Selection

Pairs are filtered using:
- Pearson correlation > 0.5
- Cointegration test (Engle-Granger)
- Minimum overlapping observations: 500

### Example output:
```text
Top cointegrated pairs:
        A       B      corr      beta  resid_std  pval
0   lme_0  shfe_0  0.572618  0.464204   0.000339   0.0
1  shfe_0  shfe_1  0.558133  0.687012   0.000426   0.0
2  shfe_0  shfe_2  0.574786  0.518008   0.000420   0.0
3  shfe_1  shfe_2  0.641464  0.468493   0.000339   0.0
```

### Step 3: Signal Construction

For the selected pair:

```python
spread = A - beta * B
zscore = (spread - rolling_mean) / rolling_std
```

Resulting file: `spread_signals_lme_0_shfe_0.csv`

---

##  Strategy Logic (`strategy.py`)

### Entry:
- Long spread: `zscore < -z_entry`
- Short spread: `zscore > +z_entry`

### Exit:
- Close long: `zscore >= z_exit`
- Close short: `zscore <= z_exit`

### Slippage model:
- Slippage is modeled as full bid/ask spread:
  ```python
  entry_price_a = ask + spread_a
  entry_price_b = bid - spread_b
  ```

### Position sizing:
- Dollar-neutral: `$100000` notional per leg

### PnL computation:
- Only realized PnL is stored, based on execution price with slippage

---

##  Output: Trade Metrics

Sample output:
```
==== Trade Statistics for lme_0 vs shfe_0 ====
Total trades  : 38
Win rate      : 55.26%
Average PnL   : 0.74 USD
Sharpe ratio  : 1.93
```

### Cumulative Equity Curve

Saved to: `equity_curve_lme0_shfe0.png`

---

## How to Run

```bash
python data.py           # produces normalized_data.parquet
python research.py       # finds cointegrated pairs and builds signals
python strategy.py       # runs backtest and prints metrics
```

---

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, statsmodels

Install via:

```bash
pip install -r requirements.txt
```

---

##  Notes

- All timestamps are aligned intraday in UTC
- SHFE quotes are converted to USD using backward-aligned FX bid
- No lookahead bias in merging or signal generation
- Execution assumes market orders with full spread slippage

## ⚠️ Commission Model Note

This implementation presents an architectural framework for such strategies. However, it is possible that the assignment contains an error in the commission formula because the calculated costs appear unrealistically high.

Alternatively, it may overlook that we're dealing with futures contracts, which follow a different execution specification. A futures contract has a defined tick size and tick value. Therefore, in the context of total notional value, commissions are significantly smaller, and strategy survivability is much more reasonable.

With the current commission model, realistic execution is not viable.
