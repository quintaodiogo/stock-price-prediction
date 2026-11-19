"""Modelos mais simples que o LightGBM, para justificar a escolha dele (pedido do professor).

Dois modelos, cada um na versão certa para o tipo de alvo:

  linear  regressão linear com penalização (Ridge) nos alvos numéricos (retornos e volatilidade) e
          regressão logística no alvo de direção (y_dir_h1, sobe ou cai: classificação);
  arvore  UMA árvore de decisão (CART): árvore de regressão nos alvos numéricos e árvore de
          classificação na direção. O LightGBM é uma soma de centenas de árvores assim.

Mesmas regras do LightGBM: mesmo walk-forward, mesmas 13 features, nada do ano de teste usado para escolher
nada. Os valores vazios (amplitude e volume) são preenchidos com a mediana do treino, com uma coluna extra
avisando que estava vazio; o modelo linear também recebe as features padronizadas (média 0, desvio 1,
calculados só no treino). Um hiperparâmetro por modelo é escolhido na validação interna (os últimos 20% das
datas do treino, com embargo; igual ao tree.py):
  Ridge: força da penalização alpha em {0,1; 1; 10; 100}     logística: C em {0,01; 0,1; 1}
  árvore: profundidade máxima em {3, 5, 7, 9}, com pelo menos 1% das linhas do treino em cada folha

Uso:
  python src/models/simple.py --model linear --universe todos_sem_alinhar
  python src/models/simple.py --model arvore --universe todos_sem_alinhar
  python src/models/simple.py --compare --universe todos_sem_alinhar
-> preds_linear_geral.parquet, preds_arvore_geral.parquet e modelos_simples.csv em reports/results/<painel>/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

GROUP_COL = {"geral": None, "classe": "asset_class", "ativo": "asset"}
GRID = {
    ("linear", "reg"): [0.1, 1.0, 10.0, 100.0],     # alpha do Ridge
    ("linear", "clf"): [0.01, 0.1, 1.0],            # C da logística
    ("arvore", "reg"): [3, 5, 7, 9],                # profundidade máxima
    ("arvore", "clf"): [3, 5, 7, 9],
}
MIN_ROWS = 500


def kind(target: str) -> str:
    return "clf" if target == "y_dir_h1" else "reg"


def build(model: str, task: str, param, n_train: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    impute = SimpleImputer(strategy="median", add_indicator=True)
    if model == "linear":
        est = Ridge(alpha=param) if task == "reg" else LogisticRegression(C=param, max_iter=1000)
        return make_pipeline(impute, StandardScaler(), est)
    leaf = max(20, n_train // 100)
    est = (DecisionTreeRegressor if task == "reg" else DecisionTreeClassifier)(
        max_depth=param, min_samples_leaf=leaf, random_state=common.SEED)
    return make_pipeline(impute, est)


def predict(est, X: pd.DataFrame, task: str) -> np.ndarray:
    return est.predict_proba(X)[:, 1] if task == "clf" else est.predict(X)


def val_loss(y: np.ndarray, p: np.ndarray, task: str) -> float:
    if task == "clf":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((y - p) ** 2))


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame, target: str, feats: list[str]):
    """Escolhe o hiperparâmetro na validação interna, retreina no treino inteiro e prevê o teste."""
    import tree  # noqa: E402  (mesma validação interna do LightGBM)
    train = train.dropna(subset=[target])
    if len(train) < MIN_ROWS or test.empty:
        return None, None
    task = kind(target)
    grid = GRID[(model, task)]
    best = grid[len(grid) // 2]
    fit, val = tree.inner_split(train)
    if len(fit) >= MIN_ROWS // 2 and len(val) > 0 and (task == "reg" or fit[target].nunique() == 2):
        losses = {}
        for param in grid:
            est = build(model, task, param, len(fit)).fit(fit[feats], fit[target])
            losses[param] = val_loss(val[target].to_numpy(), predict(est, val[feats], task), task)
        best = min(losses, key=losses.get)
    est = build(model, task, best, len(train)).fit(train[feats], train[target])
    return predict(est, test[feats], task), best


def run(panel: pd.DataFrame, folds: list[dict], model: str, mode: str, targets: list[str]):
    feats = common.feature_cols(panel)
    group_col = GROUP_COL[mode]
    frames, chosen = [], []
    for fold in folds:
        train, test = common.split(panel, fold)
        if group_col is not None:
            test = test[test[group_col].isin(train[group_col].unique())]
        for target in targets:
            groups = [(None, train, test)] if group_col is None else [
                (g, train[train[group_col] == g], test[test[group_col] == g]) for g in test[group_col].unique()]
            for g, tr, te in groups:
                pred, param = fit_predict(model, tr, te, target, feats)
                if pred is None:
                    continue
                frames.append(common.preds_frame(te, fold, target, pred))
                chosen.append({"fold": fold["fold"], "target": target, "grupo": g, "parametro": param})
        print(f"  fold {fold['fold']:2d} ({fold['test_start'][:4]}) ok", flush=True)
    return frames, chosen


# ---------------------------------------------------------------------------- comparação
def metric_of(target: str) -> str:
    return "auc" if target == "y_dir_h1" else "ic"


def paired(base: str, others: list[str], targets: list[str]) -> pd.DataFrame:
    """Cada modelo em `others` contra `base`, mesmas linhas, ano a ano (IC; AUC na direção)."""
    import run_1b  # noqa: E402
    rows = []
    for other in others:
        if not all((common.RESULTS_DIR / f"preds_{m}.parquet").exists() for m in (base, other)):
            print(f"  aviso: falta preds de {base} ou {other}")
            continue
        allp = run_1b.load_all([base, other])
        pf = run_1b.per_fold(allp[allp["target"].isin(targets)])
        for target, g in pf.groupby("target"):
            met = metric_of(target)
            w = g.pivot_table(index="fold", columns="model", values=met)
            if other not in w or base not in w:
                continue
            d = (w[base] - w[other]).dropna()
            rows.append({"target": target, "medida": met.upper(), "modelo": other,
                         "valor_modelo": w[other].mean(), f"valor_{base}": w[base].mean(),
                         f"vantagem_{base}": d.mean(), "desvio": d.std(),
                         f"anos_{base}_melhor": f"{int((d > 0).sum())} de {len(d)}"})
    return pd.DataFrame(rows).set_index(["target", "modelo"]) if rows else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear", "arvore"], help="modelo a treinar")
    ap.add_argument("--mode", choices=list(GROUP_COL), default="geral")
    ap.add_argument("--targets", nargs="+", default=common.TARGETS, choices=common.TARGETS)
    ap.add_argument("--compare", action="store_true", help="só compara com o LightGBM (lgbm_geral)")
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    pd.set_option("display.width", 200)

    if not args.compare:
        if not args.model:
            sys.exit("Escolha --model linear ou --model arvore (ou use --compare).")
        panel, folds = common.load_panel(), common.load_folds()
        name = f"{args.model}_{args.mode}"
        print(f"{name}: {len(folds)} folds, alvos {args.targets}")
        t0 = time.time()
        frames, chosen = run(panel, folds, args.model, args.mode, args.targets)
        info = {"modelo": name, "grid": {f"{m}_{t}": v for (m, t), v in GRID.items() if m == args.model},
                "escolhidos": chosen, "painel": common.panel_signature(panel, folds)}
        path = common.save_preds(frames, name, info)
        print(f"\n=== {name}: {time.time() - t0:.0f}s -> {path.relative_to(common.ROOT)}")
        print(common.quick_metrics(pd.read_parquet(path)).to_string())
        return

    others = [m for m in ["linear_geral", "arvore_geral", "vol_passada", "media_historica"]
              if (common.RESULTS_DIR / f"preds_{m}.parquet").exists()]
    table = paired("lgbm_geral", others, args.targets)
    if table.empty:
        sys.exit("Nada para comparar. Rode antes: tree.py --mode geral e simple.py --model linear/arvore.")
    out = common.RESULTS_DIR / "modelos_simples.csv"
    table.round(4).to_csv(out)
    print("\n=== LightGBM geral contra modelos mais simples (mesmas linhas; vantagem > 0 = LightGBM melhor)\n")
    print(table.round(4).to_string())
    print(f"\n-> {out.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
