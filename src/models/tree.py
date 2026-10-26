"""LightGBM no walk-forward, em três modos (README, seção 4, experimento 1b):

- geral:  um modelo por fold e alvo, treinado em todos os ativos empilhados;
- classe: um modelo por classe de ativo (indice, commodity, cambio, juros, acao_br);
- ativo:  um modelo por ativo (o baseline "treinado só no ativo alvo").

Hiperparâmetros fixos por enquanto; o número de árvores sai de early stopping numa validação
interna (últimos 20% das datas do treino, com embargo), e o modelo final é retreinado no treino
inteiro com esse número. O ano de teste nunca é usado para escolher nada.

Uso:
  python src/models/tree.py --mode geral
  python src/models/tree.py --mode geral --exclude IBOVESPA      (cenário "ativo não visto")
  python src/models/tree.py --mode ativo --targets y_vol_h5
-> reports/results/preds_lgbm_<modo>[_sem_<ativos>].parquet  (modelos gerais em models/)
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": common.SEED,
    "deterministic": True,
    "num_threads": 0,
}
MAX_TREES = 1000
EARLY_STOP = 50
VAL_FRAC = 0.2
EMBARGO_DAYS = 10
GROUP_COL = {"geral": None, "classe": "asset_class", "ativo": "asset"}


def objective(target: str) -> dict:
    if target == "y_dir_h1":
        return {"objective": "binary", "metric": "binary_logloss"}
    return {"objective": "regression", "metric": "l2"}


def inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validação interna = últimos 20% das datas do treino; embargo entre ela e o resto."""
    dates = np.sort(train["date"].unique())
    val_start = pd.Timestamp(dates[int(len(dates) * (1 - VAL_FRAC))])
    fit = train[train["date"] <= val_start - pd.Timedelta(days=EMBARGO_DAYS + 1)]
    val = train[train["date"] >= val_start]
    return fit, val


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, feats: list[str],
                min_rows: int = 500) -> tuple[np.ndarray | None, lgb.Booster | None]:
    train = train.dropna(subset=[target])
    if len(train) < min_rows or test.empty:
        return None, None
    params = {**PARAMS, **objective(target),
              "min_child_samples": int(min(PARAMS["min_child_samples"], max(20, len(train) // 50)))}
    fit, val = inner_split(train)
    n_trees = 100
    if len(fit) >= min_rows // 2 and len(val) > 0:
        probe = lgb.train(params, lgb.Dataset(fit[feats], fit[target]), MAX_TREES,
                          valid_sets=[lgb.Dataset(val[feats], val[target])],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        n_trees = max(probe.best_iteration, 10)
    model = lgb.train(params, lgb.Dataset(train[feats], train[target]), n_trees)
    return model.predict(test[feats]), model


def run(panel: pd.DataFrame, folds: list[dict], mode: str, targets: list[str],
        exclude: list[str], save_models: bool) -> list[pd.DataFrame]:
    feats = common.feature_cols(panel)
    group_col = GROUP_COL[mode]
    frames = []
    for fold in folds:
        train, test = common.split(panel, fold)
        train = train[~train["asset"].isin(exclude)]          # ativo não visto: fora do treino
        if mode != "geral":                                     # por classe/ativo: só grupos com treino
            test = test[test[group_col].isin(train[group_col].unique())]
        for target in targets:
            groups = [(None, train, test)] if group_col is None else [
                (g, train[train[group_col] == g], test[test[group_col] == g]) for g in test[group_col].unique()]
            for g, tr, te in groups:
                pred, model = fit_predict(tr, te, target, feats)
                if pred is None:
                    continue
                frames.append(common.preds_frame(te, fold, target, pred))
                if save_models and mode == "geral":
                    common.MODELS_DIR.mkdir(exist_ok=True)
                    model.save_model(str(common.MODELS_DIR / f"lgbm_{tag(mode, exclude)}_{target}_fold{fold['fold']}.txt"))
        print(f"  fold {fold['fold']:2d} ({fold['test_start'][:4]}) ok", flush=True)
    return frames


def tag(mode: str, exclude: list[str]) -> str:
    return mode + (f"_sem_{'-'.join(sorted(exclude))}" if exclude else "")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(GROUP_COL), default="geral")
    ap.add_argument("--targets", nargs="+", default=common.TARGETS, choices=common.TARGETS)
    ap.add_argument("--exclude", nargs="*", default=[], help="ativos fora do treino (cenário 'ativo não visto')")
    ap.add_argument("--no-save-models", action="store_true")
    args = ap.parse_args(argv)

    panel, folds = common.load_panel(), common.load_folds()
    unknown = set(args.exclude) - set(panel["asset"])
    if unknown:
        sys.exit(f"Ativos fora do painel: {sorted(unknown)}")
    name = "lgbm_" + tag(args.mode, args.exclude)
    print(f"{name}: {len(folds)} folds, alvos {args.targets}")
    t0 = time.time()
    frames = run(panel, folds, args.mode, args.targets, args.exclude, not args.no_save_models)
    info = {"modelo": name, "modo": args.mode, "excluidos": args.exclude, "params": PARAMS,
            "max_trees": MAX_TREES, "early_stop": EARLY_STOP, "val_frac": VAL_FRAC,
            "painel": common.panel_signature(panel, folds)}
    path = common.save_preds(frames, name, info)
    print(f"\n=== {name}: {time.time() - t0:.0f}s -> {path.relative_to(common.ROOT)}")
    print(common.quick_metrics(pd.read_parquet(path)).to_string())


if __name__ == "__main__":
    main()
