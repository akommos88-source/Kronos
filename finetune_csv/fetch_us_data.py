"""
Fetch 5-minute OHLCV data for one or more US tickers via yfinance
and save in the format expected by CustomKlineDataset.

Required columns: timestamps, open, high, low, close, volume, amount

Usage:
    python fetch_us_data.py --tickers AAPL MSFT TSLA --output data/us_5min.csv
    python fetch_us_data.py --tickers AAPL --period 60d --output data/AAPL_5min.csv
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf


def fetch_ticker(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval="5m", progress=False, auto_adjust=True)
    if raw.empty:
        print(f"  WARNING: no data returned for {ticker}, skipping.")
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    # Keep only regular market hours (09:30–16:00 ET)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("America/New_York")
    raw = raw.between_time("09:30", "16:00")

    raw = raw[["open", "high", "low", "close", "volume"]].dropna()

    # Derive 'amount' = volume * avg(ohlc) — a reasonable proxy
    raw["amount"] = raw["volume"] * (raw["open"] + raw["high"] + raw["low"] + raw["close"]) / 4

    raw = raw.reset_index().rename(columns={"Datetime": "timestamps", "index": "timestamps"})
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)  # strip tz for CSV

    raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    raw = raw.sort_values("timestamps").reset_index(drop=True)

    print(f"  {ticker}: {len(raw)} bars  [{raw['timestamps'].iloc[0]} → {raw['timestamps'].iloc[-1]}]")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Fetch US 5-min data for Kronos fine-tuning")
    parser.add_argument("--tickers", nargs="+", required=True, help="One or more ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--period",  type=str, default="60d",
                        help="yfinance period string (max 60d for 5m). Default: 60d")
    parser.add_argument("--output",  type=str, default="data/us_5min.csv",
                        help="Output CSV path. Default: data/us_5min.csv")
    args = parser.parse_args()

    all_frames = []
    print(f"Fetching {len(args.tickers)} ticker(s) ...")
    for ticker in args.tickers:
        df = fetch_ticker(ticker.upper(), args.period)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data fetched — check your ticker symbols and internet connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("timestamps").reset_index(drop=True)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"\nSaved {len(combined):,} rows → {args.output}")
    print(f"Date range : {combined['timestamps'].min()}  →  {combined['timestamps'].max()}")
    print(f"Columns    : {list(combined.columns)}")


if __name__ == "__main__":
    main()
