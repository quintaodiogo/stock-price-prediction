"""Peças comuns da Pessoa B: carregar painel e folds, separar treino/teste, salvar previsões.

Contrato de saída (README, seção 7): reports/results/preds_<modelo>.parquet com as colunas
date, asset, fold, target, y_true, y_pred.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "panel.parquet"
FOLDS_PATH = ROOT / "data" / "processed" / "folds.json"
RESULTS_DIR = ROOT / "reports" / "results"
MODELS_DIR = ROOT / "models"

TARGETS = ["y_ret_h1", "y_ret_h5", "y_dir_h1", "y_vol_h5"]
PRED_COLUMNS = ["date", "asset", "fold", "target", "y_true", "y_pred"]
SEED = 42


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        sys.exit(f"Painel não encontrado em {PANEL_PATH}. Rode antes: python src/features.py --universe sem_minis_cripto")
    return pd.read_parquet(PANEL_PATH)


def load_folds() -> list[dict]:
    return json.loads(FOLDS_PATH.read_text())


def feature_cols(panel: pd.DataFrame) -> list[str]:
    """Só colunas f_*. Nunca usar y_* nem r_* como entrada: elas olham para o futuro."""
    return [c for c in panel.columns if c.startswith("f_")]


def split(panel: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino = date <= train_end; teste = test_start <= date <= test_end (embargo entre os dois)."""
    d = panel["date"]
    train = panel[d <= pd.Timestamp(fold["train_end"])]
    test = panel[(d >= pd.Timestamp(fold["test_start"])) & (d <= pd.Timestamp(fold["test_end"]))]
    return train, test


def preds_frame(test: pd.DataFrame, fold: dict, target: str, y_pred) -> pd.DataFrame:
    """Monta as previsões no formato do contrato; descarta linhas sem alvo (ex.: y_dir_h1 com retorno zero)."""
    out = pd.DataFrame({
        "date": test["date"].values,
        "asset": test["asset"].values,
        "fold": fold["fold"],
        "target": target,
        "y_true": test[target].values,
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    return out.dropna(subset=["y_true", "y_pred"])[PRED_COLUMNS]


def panel_signature(panel: pd.DataFrame, folds: list[dict]) -> dict:
    """Resumo do painel usado, para saber depois com qual versão dos dados cada resultado foi gerado."""
    return {
        "linhas": int(len(panel)),
        "ativos": sorted(panel["asset"].unique().tolist()),
        "inicio": str(panel["date"].min().date()),
        "fim": str(panel["date"].max().date()),
        "folds": len(folds),
        "features": feature_cols(panel),
    }


def save_preds(frames: list[pd.DataFrame], model: str, info: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"preds_{model}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    if info is not None:
        (RESULTS_DIR / f"run_{model}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return path


def quick_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Conferência rápida (a avaliação oficial é o evaluate.py da Pessoa C).

    Por alvo: IC (Spearman) e RMSE médios entre folds, com desvio-padrão; AUC para a direção.
    """
    rows = []
    for (target, fold), g in preds.groupby(["target", "fold"]):
        const = g["y_pred"].nunique() == 1
        ic = np.nan if const else spearmanr(g["y_true"], g["y_pred"]).statistic
        row = {"target": target, "fold": fold, "ic": ic,
               "rmse": float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean()))}
        if target == "y_dir_h1":
            row["auc"] = 0.5 if const else roc_auc_score(g["y_true"], g["y_pred"])
        rows.append(row)
    per_fold = pd.DataFrame(rows)
    stats = per_fold.drop(columns="fold").groupby("target").agg(["mean", "std"])
    stats.columns = [f"{m}_{s}" for m, s in stats.columns]
    return stats.round(4)
