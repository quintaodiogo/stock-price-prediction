"""Experimento 1b: modelo geral x por classe x por ativo x baselines, no mesmo walk-forward.

Roda o que faltar (baselines e os três modos do LightGBM) e compara todos nas MESMAS linhas
(date, asset, target presentes em todos os modelos). Por alvo:
  ic / rmse / auc: média entre folds (desvio entre parênteses na tabela salva);
  vence_ativo_%:   em quantos folds o modelo tem IC maior que o modelo por ativo.

Uso: python src/models/run_1b.py [--rerun]
-> reports/results/comparacao_1b.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402
import common  # noqa: E402
import tree  # noqa: E402

BASE_MODELS = ["zero", "media_historica", "vol_passada", "lgbm_ativo", "lgbm_classe", "lgbm_geral"]
OPTIONAL_MODELS = ["gru_geral",                    # src/models/seq.py
                   "lgbm_ativo_tuned", "lgbm_classe_tuned", "lgbm_geral_tuned"]   # src/models/tune.py
# os opcionais entram na comparação se as previsões existirem
MODELS = BASE_MODELS + OPTIONAL_MODELS


def ensure_preds(rerun: bool) -> None:
    path = lambda m: common.RESULTS_DIR / f"preds_{m}.parquet"  # noqa: E731
    if rerun or not all(path(m).exists() for m in ("zero", "media_historica", "vol_passada")):
        baselines.main()
    for mode in ("ativo", "classe", "geral"):
        if rerun or not path(f"lgbm_{mode}").exists():
            tree.main(["--mode", mode])


def load_all(models: list[str] | None = None) -> pd.DataFrame:
    keys = ["date", "asset", "fold", "target"]
    found = [m for m in (models or MODELS) if (common.RESULTS_DIR / f"preds_{m}.parquet").exists()]
    frames = [pd.read_parquet(common.RESULTS_DIR / f"preds_{m}.parquet").assign(model=m) for m in found]
    allp = pd.concat(frames, ignore_index=True)
    # só linhas que todos os modelos aplicáveis ao alvo previram
    n_models = allp.groupby("target")["model"].nunique()
    count = allp.groupby(keys)["model"].transform("nunique")
    return allp[count == allp["target"].map(n_models)]


def per_fold(allp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target, fold), g in allp.groupby(["model", "target", "fold"]):
        rows.append({"model": model, "target": target, "fold": fold,
                     **common.quick_metrics(g.assign(fold=0)).iloc[0].filter(like="_mean")
                     .rename(lambda c: c.replace("_mean", "")).to_dict()})
    return pd.DataFrame(rows)


def summarize(pf: pd.DataFrame, ref_model: str = "lgbm_ativo") -> pd.DataFrame:
    ref = pf[pf["model"] == ref_model].set_index(["target", "fold"])["ic"]
    ref_ic = pf.set_index(["target", "fold"]).index.map(ref)
    # sem IC (previsão constante) ou sem referência para o alvo: fica fora da contagem
    ok = pf["ic"].notna() & pd.notna(np.asarray(ref_ic, dtype=float))
    pf = pf.assign(vence_ativo=(pf["ic"] > ref_ic).where(ok).astype(float))
    agg = pf.groupby(["target", "model"]).agg(
        ic=("ic", "mean"), ic_std=("ic", "std"), rmse=("rmse", "mean"),
        auc=("auc", "mean") if "auc" in pf else ("ic", "size"),
        vence_ativo_pct=("vence_ativo", lambda s: 100 * s.mean()), folds=("fold", "nunique"))
    agg.loc[agg.index.get_level_values("model") == ref_model, "vence_ativo_pct"] = np.nan
    order = {m: i for i, m in enumerate(MODELS)}
    return agg.sort_index(key=lambda idx: idx.map(order) if idx.name == "model" else idx).round(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="refaz todas as previsões")
    ap.add_argument("--models", nargs="+", default=None, choices=MODELS,
                    help="só estes modelos (padrão: todos os que tiverem previsões)")
    ap.add_argument("--ref", default="lgbm_ativo", choices=MODELS,
                    help="modelo de referência da coluna vence_ativo_pct")
    ap.add_argument("--out", default="comparacao_1b.csv")
    args = ap.parse_args()
    ensure_preds(args.rerun)
    models = args.models and list(dict.fromkeys(args.models + [args.ref]))
    table = summarize(per_fold(load_all(models)), args.ref)
    out = common.RESULTS_DIR / args.out
    table.to_csv(out)
    pd.set_option("display.width", 200)
    print(f"\n=== Experimento 1b (mesmas linhas para todos os modelos; vence_ativo_pct contra {args.ref})\n")
    print(table.to_string())
    print(f"\n-> {out.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
