"""Agrega os minis da B3 (MT5, 1 minuto, contrato contínuo sem ajuste) para data/raw/<NOME>_1d.csv.

Entrada: data/intraday/<NOME>_M1.csv (colunas <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL>,
separadas por tab). Saída: mesmas colunas dos demais ativos (date, Open, High, Low, Close, Volume),
com Volume = <VOL> somado no dia. Uso: python src/minis_to_daily.py [WIN WDO]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from assets import COLUMNS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INTRADAY_DIR = ROOT / "data" / "intraday"
RAW_DIR = ROOT / "data" / "raw"
MINIS = ["WIN", "WDO"]


def to_daily(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, sep="\t")
    m.columns = [c.strip("<>").capitalize() for c in m.columns]
    m["date"] = pd.to_datetime(m["Date"], format="%Y.%m.%d")
    m = m.sort_values(["date", "Time"])
    g = m.groupby("date")
    df = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
        "Close": g["Close"].last(), "Volume": g["Vol"].sum(),
    }).reset_index()
    df["Volume"] = df["Volume"].where(df["Volume"] > 0)
    return df[COLUMNS]


def main(names: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        df = to_daily(INTRADAY_DIR / f"{name}_M1.csv")
        df.to_csv(RAW_DIR / f"{name}_1d.csv", index=False)
        print(f"{name:5s} {len(df):6d} linhas  {df.date.min().date()} -> {df.date.max().date()}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or MINIS
    unknown = [n for n in wanted if n not in MINIS]
    if unknown:
        sys.exit(f"Ativos desconhecidos: {unknown}. Opções: {MINIS}")
    main(wanted)
