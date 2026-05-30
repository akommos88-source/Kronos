"""
Kronos US Stock Prediction (5-minute bars)

Usage:
    python predict_us_stock.py --ticker AAPL --lookback 200 --pred_len 20

Requirements:
    pip install yfinance
    pip install -r requirements.txt
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import yfinance as yf

from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_5min_data(ticker: str, lookback: int, pred_len: int) -> pd.DataFrame:
    """
    Fetch enough 5-minute bars to cover lookback + pred_len candles.
    yfinance 5m data goes back ~60 days; we fetch 30 days to be safe.
    """
    total_bars = lookback + pred_len
    raw = yf.download(ticker, period="30d", interval="5m", progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

    # yfinance returns MultiIndex columns when auto_adjust=True
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw.index = pd.to_datetime(raw.index)

    # Drop pre/post market if timezone-aware index
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("America/New_York")
        raw = raw.between_time("09:30", "16:00")

    if len(raw) < total_bars:
        raise ValueError(
            f"Only {len(raw)} bars available for {ticker}, need at least {total_bars}. "
            "Try reducing --lookback or --pred_len."
        )

    # Use the most recent total_bars candles
    raw = raw.iloc[-total_bars:].copy()
    raw = raw.reset_index().rename(columns={"Datetime": "timestamp", "index": "timestamp"})
    return raw


def plot_prediction(ticker: str, hist_df: pd.DataFrame, pred_df: pd.DataFrame, pred_len: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(f"{ticker} — 5-min Kronos Prediction (next {pred_len} bars)", fontsize=14)

    # ---- Close price ----
    # Show last 60 bars of history + all predictions side by side
    hist_tail = hist_df.tail(60)
    hist_times = hist_tail["timestamp"]
    pred_times = pred_df.index

    ax1.plot(hist_times, hist_tail["close"], color="steelblue", linewidth=1.5, label="Historical close")
    ax1.plot(pred_times, pred_df["close"], color="tomato", linewidth=1.5, linestyle="--", label="Predicted close")
    ax1.axvline(x=hist_times.iloc[-1], color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ---- Volume ----
    ax2.bar(hist_times, hist_tail["volume"], color="steelblue", alpha=0.6, width=0.003, label="Historical volume")
    ax2.bar(pred_times, pred_df["volume"], color="tomato", alpha=0.6, width=0.003, label="Predicted volume")
    ax2.axvline(x=hist_times.iloc[-1], color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Volume")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out_path = f"{ticker}_prediction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Kronos 5-min US stock prediction")
    parser.add_argument("--ticker",    type=str,   default="AAPL",  help="Stock ticker symbol")
    parser.add_argument("--lookback",  type=int,   default=200,     help="Number of historical bars to feed the model")
    parser.add_argument("--pred_len",  type=int,   default=20,      help="Number of bars to predict")
    parser.add_argument("--model",     type=str,   default="NeoQuasar/Kronos-small",
                        help="HuggingFace model ID or local path for Kronos predictor")
    parser.add_argument("--tokenizer", type=str,   default="NeoQuasar/Kronos-Tokenizer-base",
                        help="HuggingFace model ID or local path for KronosTokenizer")
    parser.add_argument("--top_p",     type=float, default=0.9)
    parser.add_argument("--temp",      type=float, default=1.0,     help="Sampling temperature")
    parser.add_argument("--samples",   type=int,   default=5,       help="Monte Carlo samples (averaged)")
    args = parser.parse_args()

    # ---- Data ----
    print(f"Fetching {args.ticker} 5-min data...")
    df = fetch_5min_data(args.ticker, args.lookback, args.pred_len)

    x_df        = df.iloc[:args.lookback][["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = pd.to_datetime(df.iloc[:args.lookback]["timestamp"])
    y_timestamp = pd.to_datetime(df.iloc[args.lookback : args.lookback + args.pred_len]["timestamp"])

    # yfinance doesn't provide a separate 'amount' column; KronosPredictor fills it automatically
    print(f"Historical bars: {len(x_df)}, Prediction bars: {len(y_timestamp)}")

    # ---- Model ----
    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    print(f"Loading predictor from {args.model} ...")
    model = Kronos.from_pretrained(args.model)

    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print(f"Running inference on {predictor.device} ...")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.temp,
        top_p=args.top_p,
        sample_count=args.samples,
        verbose=True,
    )

    # ---- Results ----
    print("\n--- Predicted OHLCV (first 5 rows) ---")
    print(pred_df[["open", "high", "low", "close", "volume"]].head())

    last_close = x_df["close"].iloc[-1]
    pred_close = pred_df["close"].iloc[-1]
    direction  = "UP" if pred_close > last_close else "DOWN"
    change_pct = (pred_close - last_close) / last_close * 100
    print(f"\nLast historical close : {last_close:.4f}")
    print(f"Predicted close (t+{args.pred_len}): {pred_close:.4f}  [{direction}  {change_pct:+.2f}%]")

    plot_prediction(args.ticker, df, pred_df, args.pred_len)


if __name__ == "__main__":
    main()
