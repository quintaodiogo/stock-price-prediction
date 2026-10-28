"""Ajuste de hiperparâmetros do LightGBM (etapa 4 da Pessoa B), sem olhar o ano de teste.

Em cada fold, e separadamente para cada modo (geral, classe, ativo):
  1. o treino do fold é dividido em ajuste + validação interna (últimos 20% das datas, com embargo,
     a mesma divisão do tree.py);
  2. cada combinação candidata é treinada no ajuste, com early stopping na validação;
  3. fica a combinação com a menor perda na validação;
  4. o modelo final é retreinado no treino inteiro com ela e prevê o ano de teste.
O ano de teste só aparece no passo 4, então o resultado não é otimista por construção. Os três
modos recebem o mesmo ajuste, o que torna a comparação geral x classe x ativo mais justa.

Candidatas: a configuração atual do tree.py mais N_CANDIDATES sorteadas (semente fixa) da grade GRID.

Uso:
  python src/models/tune.py                          (os 3 modos, só y_vol_h5)
  python src/models/tune.py --modes geral --targets y_vol_h5 y_dir_h1
  python src/models/tune.py --folds 0 8 --n 3        (teste rápido)
-> reports/results/preds_lgbm_<modo>_tuned.parquet e run_lgbm_<modo>_tuned.json (escolhas por fold)
Depois: python src/models/run_1b.py --models lgbm_ativo_tuned lgbm_classe_tuned lgbm_geral_tuned \
        lgbm_geral vol_passada --ref lgbm_ativo_tuned --out comparacao_tuned.csv
"""
import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import tree  # noqa: E402

GRID = {
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [50, 200, 500],
    "feature_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0],
    "learning_rate": [0.02, 0.05],
}
N_CANDIDATES = 8
TUNED_KEYS = list(GRID)


def candidates(n: int) -> list[dict]:
    """A configuração atual do tree.py primeiro, depois `n` sorteadas da grade (sem repetir)."""
    base = {k: tree.PARAMS[k] for k in TUNED_KEYS}
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    rng = np.random.default_rng(common.SEED)
    picked = [grid[i] for i in rng.permutation(len(grid))]
    out = [base] + [c for c in picked if c != base]
    return out[:n + 1]


def scale_min_child(params: dict, n_rows: int) -> dict:
    """Grupos pequenos (modo ativo) não suportam folhas grandes: mesma regra do tree.py."""
    return {**params, "min_child_samples": int(min(params["min_child_samples"], max(20, n_rows // 50)))}


def tune_and_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, feats: list[str],
                     cands: list[dict], min_rows: int = 500):
    train = train.dropna(subset=[target])
    if len(train) < min_rows or test.empty:
        return None, None
    fit, val = tree.inner_split(train)
    if len(fit) < min_rows // 2 or val.empty:
        return None, None
    obj = tree.objective(target)
    # o mesmo Dataset serve a todas as candidatas; feature_pre_filter=False deixa mudar min_child_samples
    d_fit = lgb.Dataset(fit[feats], fit[target], free_raw_data=False, params={"feature_pre_filter": False})
    d_val = lgb.Dataset(val[feats], val[target], reference=d_fit, free_raw_data=False,
                        params={"feature_pre_filter": False})
    best = None
    for i, cand in enumerate(cands):
        params = scale_min_child({**tree.PARAMS, **obj, **cand}, len(train))
        m = lgb.train(params, d_fit, tree.MAX_TREES, valid_sets=[d_val],
                      callbacks=[lgb.early_stopping(tree.EARLY_STOP, verbose=False)])
        score = m.best_score["valid_0"][obj["metric"]]
        if best is None or score < best[0]:
            best = (score, i, params, max(m.best_iteration, 10))
    score, i, params, n_trees = best
    model = lgb.train(params, lgb.Dataset(train[feats], train[target]), n_trees)
    return model.predict(test[feats]), {"candidata": i, "arvores": n_trees, "val": round(float(score), 6)}


def run(panel, folds, mode, targets, cands):
    feats = common.feature_cols(panel)
    group_col = tree.GROUP_COL[mode]
    frames, log = [], []
    for fold in folds:
        t0 = time.time()
        train, test = common.split(panel, fold)
        if mode != "geral":
            test = test[test[group_col].isin(train[group_col].unique())]
        for target in targets:
            groups = [("todos", train, test)] if group_col is None else [
                (g, train[train[group_col] == g], test[test[group_col] == g]) for g in test[group_col].unique()]
            for g, tr, te in groups:
                pred, info = tune_and_predict(tr, te, target, feats, cands)
                if pred is None:
                    continue
                frames.append(common.preds_frame(te, fold, target, pred))
                log.append({"fold": fold["fold"], "target": target, "grupo": g, **info})
        print(f"  {mode:6s} fold {fold['fold']:2d} ({fold['test_start'][:4]}) ok  {time.time() - t0:.0f}s", flush=True)
    return frames, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["geral", "classe", "ativo"], choices=list(tree.GROUP_COL))
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS)
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="só estes folds (teste rápido)")
    ap.add_argument("--n", type=int, default=N_CANDIDATES, help="combinações sorteadas além da atual")
    args = ap.parse_args(argv)

    panel, all_folds = common.load_panel(), common.load_folds()
    folds = [f for f in all_folds if args.folds is None or f["fold"] in args.folds]
    cands = candidates(args.n)
    print(f"{len(cands)} candidatas ({len(cands) - 1} sorteadas + a atual), {len(folds)} folds, "
          f"alvos {args.targets}, modos {args.modes}")
    for mode in args.modes:
        # teste rápido (--folds) salva com outro nome para não sobrescrever o resultado completo
        name = f"lgbm_{mode}_tuned" + ("_teste" if args.folds is not None else "")
        t0 = time.time()
        frames, log = run(panel, folds, mode, args.targets, cands)
        chosen = Counter(r["candidata"] for r in log)
        info = {"modelo": name, "modo": mode, "candidatas": cands, "escolhas": log,
                "vezes_escolhida": {str(k): v for k, v in sorted(chosen.items())},
                "params_fixos": tree.PARAMS, "painel": common.panel_signature(panel, all_folds)}
        path = common.save_preds(frames, name, info)
        print(f"\n=== {name}: {time.time() - t0:.0f}s -> {path.relative_to(common.ROOT)}")
        print("candidata escolhida (nº de vezes):", dict(sorted(chosen.items())), "(0 = configuração atual)")
        print(common.quick_metrics(pd.read_parquet(path)).to_string(), "\n")


if __name__ == "__main__":
    main()
