"""Baselines obrigatórios (README, seção 4.1), no mesmo walk-forward do modelo geral.

- zero:            retorno previsto = 0; direção = 0,5. Não se aplica à volatilidade.
- media_historica: média do alvo no treino de cada ativo (para direção, a frequência de alta).
- vol_passada:     só y_vol_h5. Volatilidade realizada dos últimos 20 pregões (média de 4 blocos
                   de 5 dias), na mesma unidade do alvo (dividida pela escala s_t de hoje).

Uso: python src/models/baselines.py   ->  reports/results/preds_<baseline>.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RET_TARGETS = ["y_ret_h1", "y_ret_h5", "y_dir_h1"]
MAX_BLOCK_DAYS = 14   # 5 pregões para trás não podem passar disso em dias corridos (senão há buraco)


def past_vol(panel: pd.DataFrame) -> pd.Series:
    """Vol. realizada passada, só com dados até t.

    r_vol_h5 na linha t-5 é o desvio de x_{t-4..t}, já conhecido em t. Média dos blocos que
    terminam em t, t-5, t-10 e t-15 (20 pregões), dividida pela escala de hoje.
    """
    p = panel.sort_values(["asset", "date"])
    g = p.groupby("asset")
    blocks = []
    for k in (5, 10, 15, 20):
        vol = g["r_vol_h5"].shift(k)
        span = (p["date"] - g["date"].shift(k)).dt.days
        blocks.append(vol.where(span <= MAX_BLOCK_DAYS * k / 5))
    est = pd.concat(blocks, axis=1).mean(axis=1, skipna=False) / p["scale"]
    return est.reindex(panel.index)


def run(panel: pd.DataFrame, folds: list[dict]) -> dict[str, list[pd.DataFrame]]:
    out = {"zero": [], "media_historica": [], "vol_passada": []}
    panel = panel.assign(_past_vol=past_vol(panel))
    for fold in folds:
        train, test = common.split(panel, fold)
        if train.empty or test.empty:
            continue
        for target in common.TARGETS:
            tr = train.dropna(subset=[target])
            if target in RET_TARGETS:
                zero = 0.5 if target == "y_dir_h1" else 0.0
                out["zero"].append(common.preds_frame(test, fold, target, np.full(len(test), zero)))
            # média por ativo; ativo sem histórico no treino usa a média geral
            by_asset = tr.groupby("asset")[target].mean()
            mean_pred = test["asset"].map(by_asset).fillna(tr[target].mean())
            out["media_historica"].append(common.preds_frame(test, fold, target, mean_pred))
            if target == "y_vol_h5":
                vp = test["_past_vol"].fillna(test["asset"].map(by_asset))
                out["vol_passada"].append(common.preds_frame(test, fold, target, vp))
    return out


def main() -> None:
    panel, folds = common.load_panel(), common.load_folds()
    info = {"painel": common.panel_signature(panel, folds)}
    for name, frames in run(panel, folds).items():
        path = common.save_preds(frames, name, {**info, "modelo": name})
        preds = pd.read_parquet(path)
        print(f"\n=== {name}: {len(preds)} previsões -> {path.relative_to(common.ROOT)}")
        print(common.quick_metrics(preds).to_string())


if __name__ == "__main__":
    main()
