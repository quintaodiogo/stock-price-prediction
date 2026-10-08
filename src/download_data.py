"""Baixa os 20 ativos diários do Yahoo Finance para data/raw/<NOME>_1d.csv.

Histórico máximo, preços ajustados (auto_adjust). Volume zero (câmbio, VIX, juros e
séries antigas de índices) é gravado como vazio. Uso: python src/download_data.py [NOME ...]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from assets import ASSETS, COLUMNS  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(ticker: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
        if not df.empty:
            return df
        time.sleep(2 * attempt)
    raise RuntimeError(f"Yahoo não devolveu dados para {ticker}")


def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df[COLUMNS].sort_values("date").drop_duplicates("date", keep="last")
    df = df[df["date"] < pd.Timestamp.today().normalize()]  # candle de hoje ainda está aberto
    # Volume zero nunca é volume real: ausente (inclui câmbio, VIX, juros e séries antigas de índices)
    df["Volume"] = df["Volume"].where(df["Volume"] > 0)
    return df.reset_index(drop=True)


def main(names: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        ticker, _ = ASSETS[name]
        df = clean(fetch(ticker), name)
        df.to_csv(RAW_DIR / f"{name}_1d.csv", index=False)
        print(f"{name:10s} {ticker:10s} {len(df):6d} linhas  {df.date.min().date()} -> {df.date.max().date()}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(ASSETS)
    unknown = [n for n in wanted if n not in ASSETS]
    if unknown:
        sys.exit(f"Ativos desconhecidos: {unknown}. Opções: {list(ASSETS)}")
    main(wanted)
