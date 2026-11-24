"""Transfer learning com LightGBM (Pessoa C): zero-shot, fine-tuning e ativo não visto.

Para cada fold do walk-forward e cada alvo:

- zero_shot:  modelo geral (todos os ativos empilhados, igual ao `lgbm_geral` da Pessoa B) aplicado a cada ativo,
              sem nenhum treino extra nele.
- fine_tune:  parte do modelo geral e continua o boosting só nas linhas de treino do ativo, com learning rate
              menor e poucas árvores. O número de árvores extras (0 a 100) é escolhido numa validação interna
              (últimos 20% das datas do treino do ativo, com embargo). Como o modelo geral não pode ter visto
              essa validação, a escolha usa uma versão do modelo geral treinada só na parte "ajuste".
- unseen:     para cada ativo, o modelo geral é treinado SEM ele e SEM os irmãos próximos (siblings.py) e
              testado nele. É o teste real de generalização.

Os baselines (lgbm_ativo = treino só no ativo, lgbm_geral etc.) vêm da Pessoa B (src/models/run_1b.py).

Uso:
  python src/transfer/run_transfer.py                          (3 cenários, y_vol_h5)
  python src/transfer/run_transfer.py --scenarios unseen --targets y_vol_h5 y_dir_h1
  python src/transfer/run_transfer.py --universe sem_minis_cripto
-> reports/results/preds_transfer_<cenario>.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "transfer"))
import common  # noqa: E402
import tree  # noqa: E402
from siblings import excluded_for  # noqa: E402

FT_LR = 0.01                       # learning rate do fine-tuning (o geral usa 0,03)
FT_GRID = [0, 10, 25, 50, 100]     # árvores extras candidatas
MIN_ASSET_ROWS = 500
SCENARIOS = ["zero_shot", "fine_tune", "unseen"]


def params_for(target: str, n_rows: int, lr: float | None = None) -> dict:
    p = {**tree.PARAMS, **tree.objective(target),
         "min_child_samples": int(min(tree.PARAMS["min_child_samples"], max(20, n_rows // 50)))}
    if lr:
        p["learning_rate"] = lr
    return p


def train_general(train: pd.DataFrame, target: str, feats: list[str]) -> tuple[lgb.Booster, lgb.Booster]:
    """(modelo completo, modelo treinado só na parte de ajuste) com o mesmo número de árvores."""
    train = train.dropna(subset=[target])
    fit, val = tree.inner_split(train)
    p = params_for(target, len(train))
    probe = lgb.train(p, lgb.Dataset(fit[feats], fit[target]), tree.MAX_TREES,
                      valid_sets=[lgb.Dataset(val[feats], val[target])],
                      callbacks=[lgb.early_stopping(tree.EARLY_STOP, verbose=False)])
    n = max(probe.best_iteration, 10)
    full = lgb.train(p, lgb.Dataset(train[feats], train[target]), n)
    fit_only = lgb.train(p, lgb.Dataset(fit[feats], fit[target]), n)
    return full, fit_only


def loss(target: str, y: np.ndarray, pred: np.ndarray) -> float:
    if target == "y_dir_h1":
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(pred) + (1 - y) * np.log(1 - pred)).mean())
    return float(((y - pred) ** 2).mean())


def fine_tune(full: lgb.Booster, fit_only: lgb.Booster, tr: pd.DataFrame, te: pd.DataFrame,
              target: str, feats: list[str]) -> tuple[np.ndarray, int]:
    """Continua o boosting do modelo geral no ativo. Devolve previsões do teste e árvores extras usadas."""
    tr = tr.dropna(subset=[target])
    fit, val = tree.inner_split(tr)
    n_extra = 0
    if len(fit) >= MIN_ASSET_ROWS // 2 and len(val) > 0:
        p = params_for(target, len(fit), FT_LR)
        ext = lgb.train(p, lgb.Dataset(fit[feats], fit[target], free_raw_data=False), max(FT_GRID), init_model=fit_only)
        base = fit_only.current_iteration()
        scores = {n: loss(target, val[target].values, ext.predict(val[feats], num_iteration=base + n)) for n in FT_GRID}
        n_extra = min(scores, key=scores.get)
    if n_extra == 0:
        return full.predict(te[feats]), 0
    p = params_for(target, len(tr), FT_LR)
    tuned = lgb.train(p, lgb.Dataset(tr[feats], tr[target], free_raw_data=False), n_extra, init_model=full)
    return tuned.predict(te[feats]), n_extra


def run(panel: pd.DataFrame, folds: list[dict], targets: list[str], scenarios: list[str]) -> tuple[dict, list[dict]]:
    feats = common.feature_cols(panel)
    out = {s: [] for s in scenarios}
    log = []
    for fold in folds:
        train, test = common.split(panel, fold)
        for target in targets:
            if {"zero_shot", "fine_tune"} & set(scenarios):
                full, fit_only = train_general(train, target, feats)
                for asset in test["asset"].unique():
                    te = test[test["asset"] == asset]
                    if "zero_shot" in scenarios:
                        out["zero_shot"].append(common.preds_frame(te, fold, target, full.predict(te[feats])))
                    if "fine_tune" in scenarios:
                        tr = train[train["asset"] == asset]
                        if tr[target].notna().sum() < MIN_ASSET_ROWS:
                            continue
                        pred, n_extra = fine_tune(full, fit_only, tr, te, target, feats)
                        out["fine_tune"].append(common.preds_frame(te, fold, target, pred))
                        log.append({"fold": fold["fold"], "target": target, "asset": asset, "arvores_extras": n_extra})
            if "unseen" in scenarios:
                for asset in test["asset"].unique():
                    te = test[test["asset"] == asset]
                    tr = train[~train["asset"].isin(excluded_for(asset))]
                    pred, _ = tree.fit_predict(tr, te, target, feats)
                    if pred is not None:
                        out["unseen"].append(common.preds_frame(te, fold, target, pred))
        print(f"  fold {fold['fold']:2d} ({fold['test_start'][:4]}) ok", flush=True)
    return out, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS)
    ap.add_argument("--folds", type=int, default=None, help="só os N últimos folds (teste rápido)")
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    panel, folds = common.load_panel(), common.load_folds()
    if args.folds:
        folds = folds[-args.folds:]
    t0 = time.time()
    print(f"transfer: {args.scenarios}, alvos {args.targets}, {len(folds)} folds")
    out, log = run(panel, folds, args.targets, args.scenarios)
    for s, frames in out.items():
        info = {"modelo": f"transfer_{s}", "cenario": s, "alvos": args.targets, "ft_lr": FT_LR, "ft_grid": FT_GRID,
                "painel": common.panel_signature(panel, folds),
                **({"fine_tune_arvores": log} if s == "fine_tune" else {}),
                **({"excluidos_por_ativo": {a: excluded_for(a) for a in sorted(panel["asset"].unique())}} if s == "unseen" else {})}
        path = common.save_preds(frames, f"transfer_{s}", info)
        print(f"{s}: {path.relative_to(common.ROOT)}")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
