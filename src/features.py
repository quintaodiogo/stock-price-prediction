"""Normalização, features, alvos, painel e folds. Nenhuma feature usa dado posterior a t.

Série base x_t: retorno log do Close (ativos de preço) ou diferença do Close (US10Y, VIX).
x_t fica ausente se o preço não é positivo (WTI em 20/04/2020) ou se há um buraco longo nos dados.
Escala s_t = desvio-padrão de x nos 252 dias até t (só passado). Features e alvos são divididos por s_t.

Uso: python src/features.py   ->  data/processed/panel.parquet e data/processed/folds.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from assets import ASSETS, LEVEL_ASSETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

SCALE_WINDOW = 252
SCALE_MIN = 63
MAX_GAP_DAYS = {"cripto": 3, "default": 7}   # buraco maior que isso invalida o retorno do dia
LAGS = 5
EMBARGO_DAYS = 10                            # dias corridos (>= 5 pregões do maior horizonte)
FIRST_TEST_YEAR = 2010


def load_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}_1d.csv", parse_dates=["date"])


def base_series(df: pd.DataFrame, name: str) -> pd.Series:
    """x_t: retorno log (ou diferença para US10Y/VIX); NaN em preço <= 0 e após buracos longos."""
    close = df["Close"].where(df["Close"] > 0) if name not in LEVEL_ASSETS else df["Close"]
    if name in LEVEL_ASSETS:
        x = close.diff()
    else:
        x = np.log(close / close.shift())
    gap = df["date"].diff().dt.days
    limit = MAX_GAP_DAYS.get(ASSETS[name][1], MAX_GAP_DAYS["default"])
    return x.where(gap <= limit).replace([np.inf, -np.inf], np.nan)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    return (up / (up + down).replace(0, np.nan)).rename("rsi")


def asset_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    x = base_series(df, name)
    scale = x.rolling(SCALE_WINDOW, min_periods=SCALE_MIN).std()
    out = pd.DataFrame({"date": df["date"], "asset": name, "asset_class": ASSETS[name][1], "scale": scale})

    # --- features: só dados até t -------------------------------------------------
    for k in range(LAGS):
        out[f"f_ret_lag{k}"] = x.shift(k) / scale
    for w in (5, 21, 63):
        out[f"f_vol{w}"] = x.rolling(w, min_periods=w).std() / scale
    for w in (21, 63):
        out[f"f_mom{w}"] = x.rolling(w, min_periods=w).sum() / (scale * np.sqrt(w))
    out["f_rsi14"] = rsi(df["Close"])
    rng = ((df["High"] - df["Low"]) / df["Close"].where(df["Close"] > 0)).where(df["High"] >= df["Low"])
    out["f_range"] = rng / rng.rolling(SCALE_WINDOW, min_periods=SCALE_MIN).mean().replace(0, np.nan)
    out["f_volrel"] = df["Volume"] / df["Volume"].rolling(21, min_periods=21).mean()

    # --- alvos: dados após t, em unidades de s_t ---------------------------------
    fut1 = x.shift(-1)
    fut5 = pd.concat([x.shift(-k) for k in range(1, 6)], axis=1)
    out["y_ret_h1"] = fut1 / scale
    out["y_ret_h5"] = fut5.sum(axis=1, skipna=False) / (scale * np.sqrt(5))
    out["y_dir_h1"] = (fut1 > 0).astype(float).where(fut1.notna() & (fut1 != 0))
    out["y_vol_h5"] = fut5.std(axis=1, skipna=False) / scale
    return out


def build_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [asset_frame(df, name) for name, df in raw.items()]
    panel = pd.concat(frames, ignore_index=True)
    feats = [c for c in panel if c.startswith("f_") and c not in ("f_volrel", "f_range")]
    targets = ["y_ret_h1", "y_ret_h5", "y_vol_h5"]
    panel = panel.dropna(subset=["scale"] + feats + targets)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["date", "asset"]).reset_index(drop=True)


def make_folds(last_date: pd.Timestamp, first_year: int = FIRST_TEST_YEAR, embargo_days: int = EMBARGO_DAYS) -> list[dict]:
    """Walk-forward expansivo: teste de um ano; treino termina `embargo_days`+1 antes do teste."""
    folds = []
    for year in range(first_year, last_date.year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = min(pd.Timestamp(year=year, month=12, day=31), last_date)
        folds.append({
            "fold": len(folds),
            "train_end": str((test_start - pd.Timedelta(days=embargo_days + 1)).date()),
            "embargo_end": str((test_start - pd.Timedelta(days=1)).date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        })
    return folds


def main() -> None:
    raw = {n: load_raw(n) for n in ASSETS}
    panel = build_panel(raw)
    folds = make_folds(panel["date"].max())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_DIR / "panel.parquet", index=False)
    (OUT_DIR / "folds.json").write_text(json.dumps(folds, indent=2))
    print(f"painel: {len(panel)} linhas, {panel['asset'].nunique()} ativos, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}, {len(folds)} folds")
    print(panel.groupby("asset")["date"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
