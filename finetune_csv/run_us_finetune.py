"""
End-to-end fine-tuning runner for US equity 5-min data.

Steps:
  1. Fetch data from yfinance  (skipped if CSV already exists)
  2. Fine-tune KronosTokenizer
  3. Fine-tune Kronos predictor

Usage:
    # Full pipeline — fetch AAPL + MSFT + TSLA then train
    python run_us_finetune.py --tickers AAPL MSFT TSLA

    # Use an existing CSV, custom config
    python run_us_finetune.py --data_path data/us_5min.csv --config configs/config_us_5min.yaml

    # Skip tokenizer training (already done), only train predictor
    python run_us_finetune.py --tickers AAPL --skip_tokenizer
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def patch_config_data_path(config_path: str, data_path: str, out_path: str):
    """Write a copy of config_path with data.data_path replaced."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_path"] = os.path.abspath(data_path)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Kronos US stock fine-tuning pipeline")
    parser.add_argument("--tickers",        nargs="+", default=[],
                        help="Ticker symbols to fetch (e.g. AAPL MSFT TSLA)")
    parser.add_argument("--period",         default="60d",
                        help="yfinance period for data fetch (max 60d for 5m). Default: 60d")
    parser.add_argument("--data_path",      default="data/us_5min.csv",
                        help="Path to existing or target CSV. Default: data/us_5min.csv")
    parser.add_argument("--config",         default="configs/config_us_5min.yaml",
                        help="YAML config path. Default: configs/config_us_5min.yaml")
    parser.add_argument("--skip_tokenizer", action="store_true",
                        help="Skip tokenizer training (use if already fine-tuned)")
    parser.add_argument("--skip_predictor", action="store_true",
                        help="Skip predictor training")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = args.data_path if os.path.isabs(args.data_path) \
                 else os.path.join(script_dir, args.data_path)

    # ------------------------------------------------------------------ #
    # 1. Fetch data
    # ------------------------------------------------------------------ #
    if args.tickers:
        run(
            [sys.executable, os.path.join(script_dir, "fetch_us_data.py"),
             "--tickers"] + args.tickers + [
             "--period", args.period,
             "--output", data_path],
            f"Fetching 5-min data for: {', '.join(args.tickers)}"
        )
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"No tickers provided and data file not found: {data_path}\n"
                "Pass --tickers AAPL ... to fetch data first."
            )
        print(f"Using existing data file: {data_path}")

    # ------------------------------------------------------------------ #
    # 2. Patch config with the resolved data path
    # ------------------------------------------------------------------ #
    config_src = args.config if os.path.isabs(args.config) \
                 else os.path.join(script_dir, args.config)
    config_run = os.path.join(script_dir, "configs", "_run_config.yaml")
    os.makedirs(os.path.dirname(config_run), exist_ok=True)
    patch_config_data_path(config_src, data_path, config_run)
    print(f"Active config written to: {config_run}")

    # ------------------------------------------------------------------ #
    # 3. Fine-tune tokenizer
    # ------------------------------------------------------------------ #
    if not args.skip_tokenizer:
        run(
            [sys.executable, os.path.join(script_dir, "finetune_tokenizer.py"),
             "--config", config_run],
            "Stage 1 / 2 — Fine-tuning KronosTokenizer"
        )
    else:
        print("\nSkipping tokenizer training (--skip_tokenizer set).")

    # ------------------------------------------------------------------ #
    # 4. Fine-tune predictor
    # ------------------------------------------------------------------ #
    if not args.skip_predictor:
        run(
            [sys.executable, os.path.join(script_dir, "finetune_base_model.py"),
             "--config", config_run],
            "Stage 2 / 2 — Fine-tuning Kronos predictor"
        )
    else:
        print("\nSkipping predictor training (--skip_predictor set).")

    print("\n" + "="*60)
    print("  Fine-tuning complete!")
    print(f"  Fine-tuned models saved under: finetuned/us_5min_finetune/")
    print("  To predict with the fine-tuned model, pass:")
    print("    --tokenizer finetuned/us_5min_finetune/tokenizer/best_model")
    print("    --model     finetuned/us_5min_finetune/basemodel/best_model")
    print("  to predict_us_stock.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
