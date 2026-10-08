"""Valida os CSVs de data/raw: datas duplicadas, buracos, splits/saltos, zeros e valores incoerentes.

Uso: python src/validate_data.py        (imprime a tabela e grava docs/dados_validacao.md)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from assets import ASSETS, LEVEL_ASSETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
JUMP = 0.20          # |retorno log| diário acima disso é suspeito (split, erro ou evento extremo)
GAP_DAYS = {"cripto": 2, "default": 6}   # buraco = dias corridos entre duas datas seguidas


def validate(name: str) -> dict:
    df = pd.read_csv(RAW_DIR / f"{name}_1d.csv", parse_dates=["date"])
    cls = ASSETS[name][1]
    px = df[["Open", "High", "Low", "Close"]]
    gaps = df["date"].diff().dt.days
    limit = GAP_DAYS.get(cls, GAP_DAYS["default"])
    if name in LEVEL_ASSETS:
        move = df["Close"].diff()
        jumps = move[(df["Close"].shift() > 0) & (move.abs() / df["Close"].shift() > 0.5)]
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            move = np.log(df["Close"] / df["Close"].shift())
        jumps = move[move.abs() > JUMP]
    return {
        "ativo": name,
        "linhas": len(df),
        "inicio": df["date"].min().date(),
        "fim": df["date"].max().date(),
        "datas_duplicadas": int(df["date"].duplicated().sum()),
        "fora_de_ordem": int(not df["date"].is_monotonic_increasing),
        "buracos": int((gaps > limit).sum()),
        "maior_buraco_dias": int(gaps.max()),
        "nulos_preco": int(px.isna().sum().sum()),
        "preco_nao_positivo": int((px <= 0).sum().sum()) if name not in LEVEL_ASSETS else int((px < 0).sum().sum()),
        "high_menor_low": int((df["High"] < df["Low"]).sum()),
        "close_fora_hl": int(((df["Close"] > df["High"] * 1.0001) | (df["Close"] < df["Low"] * 0.9999)).sum()),
        "saltos_suspeitos": len(jumps),
        "retorno_zero_%": round(100 * float((df["Close"].diff() == 0).mean()), 1),
        "volume_nulo_%": round(100 * float(df["Volume"].isna().mean()), 1),
            "_jumps": [(df.loc[i, "date"].date(), round(float(v), 3)) for i, v in jumps.items()],
    }


def main() -> None:
    rows = [validate(n) for n in ASSETS if (RAW_DIR / f"{n}_1d.csv").exists()]
    table = pd.DataFrame(rows).drop(columns="_jumps")
    print(table.to_string(index=False))
    for r in rows:
        if r["_jumps"]:
            print(f"\nSaltos suspeitos em {r['ativo']}: {r['_jumps'][:10]}")
    if len(rows) != len(ASSETS):
        sys.exit(f"Faltam arquivos: {len(ASSETS) - len(rows)}")


if __name__ == "__main__":
    main()
