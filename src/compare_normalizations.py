"""Compara as normalizações: mesmo painel (datas comuns), mesmos alvos, mesmos folds; só as features mudam.

Para cada normalização: walk-forward (src/features.py: embargo), gradient boosting (scikit-learn) com
hiperparâmetros fixos, avaliação nos alvos CRUS (direção do retorno, retorno e volatilidade futuros).
Métricas por fold e média: AUC da direção (0,5 = acaso), IC de Spearman do retorno de 1 dia,
IC de Spearman e RMSE (log) da volatilidade de 5 dias. Uso:
    python src/compare_normalizations.py [--universe todos|sem_minis|sem_minis_cripto] [--norms a b ...]
Saída: reports/results/normalizacao_<universo>.csv (por fold) e _resumo.csv (média e desvio entre folds).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import features as ft  # noqa: E402
import normalization as nz  # noqa: E402
from assets import UNIVERSES  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "reports" / "results"
PARAMS = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=50, random_state=0)


def evaluate_fold(panel: pd.DataFrame, fold: dict, feats: list[str]) -> dict | None:
    tr = panel[panel["date"] <= fold["train_end"]]
    te = panel[(panel["date"] >= fold["test_start"]) & (panel["date"] <= fold["test_end"])]
    if len(tr) < 500 or len(te) < 100:
        return None
    out = {"fold": fold["fold"], "test_start": fold["test_start"], "n_train": len(tr), "n_test": len(te)}
    # direção
    d_tr, d_te = tr.dropna(subset=["y_dir_h1"]), te.dropna(subset=["y_dir_h1"])
    clf = HistGradientBoostingClassifier(**PARAMS).fit(d_tr[feats], d_tr["y_dir_h1"])
    out["auc_dir"] = roc_auc_score(d_te["y_dir_h1"], clf.predict_proba(d_te[feats])[:, 1])
    # retorno 1 dia (treino em unidades de s_t; IC contra o retorno cru)
    reg = HistGradientBoostingRegressor(**PARAMS).fit(tr[feats], tr["y_ret_h1"])
    out["ic_ret_h1"] = spearmanr(reg.predict(te[feats]), te["r_ret_h1"]).statistic
    # volatilidade 5 dias: alvo log(y_vol_h5); avalia RMSE na escala log e IC contra a vol crua
    v_tr, v_te = tr[tr["y_vol_h5"] > 0], te[(te["y_vol_h5"] > 0) & (te["r_vol_h5"] > 0)]
    vol = HistGradientBoostingRegressor(**PARAMS).fit(v_tr[feats], np.log(v_tr["y_vol_h5"]))
    pred = vol.predict(v_te[feats])
    out["ic_vol_h5"] = spearmanr(pred, v_te["r_vol_h5"] / v_te["scale"]).statistic
    out["rmse_logvol"] = float(np.sqrt(np.mean((pred - np.log(v_te["y_vol_h5"])) ** 2)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="todos", choices=list(UNIVERSES))
    ap.add_argument("--norms", nargs="*", default=nz.ALL_NORMS)
    args = ap.parse_args(argv)
    raw = {n: ft.load_raw(n) for n in UNIVERSES[args.universe]}
    rows = []
    for norm in args.norms:
        panel = ft.build_panel(raw, norm)
        first = max(ft.FIRST_TEST_YEAR, panel["date"].min().year + 1)
        folds = ft.make_folds(panel["date"].max(), first)
        feats = [c for c in panel if c.startswith("f_")]
        for fold in folds:
            r = evaluate_fold(panel, fold, feats)
            if r:
                rows.append({"norm": norm, **r})
        print(f"{norm:22s} ok", flush=True)
    res = pd.DataFrame(rows)
    metrics = ["auc_dir", "ic_ret_h1", "ic_vol_h5", "rmse_logvol"]
    g = res.groupby("norm")[metrics]
    summary = g.mean().add_suffix("_media").join(g.std().add_suffix("_dp")).join(
        res.assign(auc_gt=res["auc_dir"] > 0.5).groupby("norm")["auc_gt"].sum().rename("folds_auc>0.5"))
    summary["n_folds"] = res.groupby("norm").size()
    summary = summary.loc[[n for n in args.norms if n in summary.index]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / f"normalizacao_{args.universe}.csv", index=False)
    summary.round(4).to_csv(RESULTS / f"normalizacao_{args.universe}_resumo.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
