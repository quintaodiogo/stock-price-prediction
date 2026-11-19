"""Avaliação (Pessoa C): métricas, backtest de direção, tabelas e gráficos por ativo e por classe.

Lê qualquer `reports/results/preds_<modelo>.parquet` no formato do contrato (README, seção 7) e compara os
modelos nas MESMAS linhas (date, asset, fold, target). Não depende de como o modelo foi treinado.

Saídas em reports/results/ (tabelas) e reports/figures/ (gráficos):
  avaliacao_geral_<alvo>.csv    IC, RMSE (e AUC) médios entre folds, desvio e % de folds em que vence o modelo por ativo
  avaliacao_ativo_<alvo>.csv    IC por ativo e modelo (todas as linhas de teste do ativo, folds juntos)
  avaliacao_classe_<alvo>.csv   idem por classe de ativo
  tabela_final_<alvo>.csv       baseline x treino só no ativo x zero-shot x fine-tuning x ativo não visto, por ativo
  backtest_direcao.csv          Sharpe e acerto de uma estratégia long/short sobre a probabilidade de alta (y_dir_h1)

Uso:
  python src/evaluate.py                       (alvo y_vol_h5)
  python src/evaluate.py --target y_dir_h1
  python src/evaluate.py --universe sem_minis_cripto
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
import common  # noqa: E402

KEYS = ["date", "asset", "fold", "target"]
# ordem de apresentação; os que não tiverem previsões ficam de fora
BASELINES = ["zero", "media_historica", "vol_passada"]
STAGES = {                       # rótulo da tabela final -> modelo
    "baseline": "vol_passada",
    "so_ativo": "lgbm_ativo",
    "geral": "lgbm_geral",
    "zero_shot": "transfer_zero_shot",
    "fine_tune": "transfer_fine_tune",
    "nao_visto": "transfer_unseen",
}
DEFAULT_MODELS = BASELINES + ["lgbm_ativo", "lgbm_classe", "lgbm_geral", "gru_geral",
                              "transfer_zero_shot", "transfer_fine_tune", "transfer_unseen"]
REF_MODEL = "lgbm_ativo"


def load(models: list[str], target: str) -> pd.DataFrame:
    """Previsões empilhadas, só das linhas que todos os modelos disponíveis para o alvo previram."""
    frames = []
    for m in models:
        p = common.RESULTS_DIR / f"preds_{m}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d = d[d["target"] == target]
            if len(d):
                frames.append(d.assign(model=m))
    if not frames:
        sys.exit(f"Sem previsões para {target} em {common.RESULTS_DIR}. Rode os modelos antes.")
    allp = pd.concat(frames, ignore_index=True)
    n = allp["model"].nunique()
    keep = allp.groupby(KEYS)["model"].transform("nunique") == n
    return allp[keep].reset_index(drop=True)


def ic(g: pd.DataFrame) -> float:
    return np.nan if g["y_pred"].nunique() < 2 else float(spearmanr(g["y_true"], g["y_pred"]).statistic)


def metrics(g: pd.DataFrame, target: str) -> dict:
    row = {"ic": ic(g), "rmse": float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean()))}
    if target == "y_dir_h1":
        row["auc"] = 0.5 if g["y_pred"].nunique() < 2 else float(roc_auc_score(g["y_true"], g["y_pred"]))
    return row


def by_fold(allp: pd.DataFrame, target: str) -> pd.DataFrame:
    return pd.DataFrame([{"model": m, "fold": f, **metrics(g, target)}
                         for (m, f), g in allp.groupby(["model", "fold"])])


def general_table(allp: pd.DataFrame, target: str) -> pd.DataFrame:
    pf = by_fold(allp, target)
    ref = pf[pf["model"] == REF_MODEL].set_index("fold")["ic"]
    ref_g = pf[pf["model"] == "lgbm_geral"].set_index("fold")["ic"]
    pf["vence_geral"] = np.where(pf["ic"].notna() & pf["fold"].map(ref_g).notna(), (pf["ic"] > pf["fold"].map(ref_g)).astype(float), np.nan)
    pf["vence"] = np.where(pf["ic"].notna() & pf["fold"].map(ref).notna(), (pf["ic"] > pf["fold"].map(ref)).astype(float), np.nan)
    cols = {"ic": ("ic", "mean"), "ic_std": ("ic", "std"), "rmse": ("rmse", "mean"),
            "vence_so_ativo_pct": ("vence", lambda s: 100 * s.mean()),
            "vence_geral_pct": ("vence_geral", lambda s: 100 * s.mean()), "folds": ("fold", "nunique")}
    if "auc" in pf:
        cols["auc"] = ("auc", "mean")
    t = pf.groupby("model").agg(**cols)
    if REF_MODEL in t.index:
        t.loc[REF_MODEL, "vence_so_ativo_pct"] = np.nan
    if "lgbm_geral" in t.index:
        t.loc["lgbm_geral", "vence_geral_pct"] = np.nan
    order = {m: i for i, m in enumerate(DEFAULT_MODELS)}
    return t.sort_index(key=lambda i: i.map(lambda m: order.get(m, 99))).round(4)


def group_table(allp: pd.DataFrame, target: str, col: str) -> pd.DataFrame:
    rows = [{col: k, "model": m, **metrics(g, target)} for (k, m), g in allp.groupby([col, "model"])]
    t = pd.DataFrame(rows).pivot(index=col, columns="model", values="ic")
    return t[[m for m in DEFAULT_MODELS if m in t.columns]].round(4)


def final_table(ic_asset: pd.DataFrame, rmse_asset: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por ativo, uma coluna por etapa (README, seção 6: "pronto quando")."""
    out = pd.DataFrame(index=ic_asset.index)
    for stage, model in STAGES.items():
        if model in ic_asset.columns:
            out[f"ic_{stage}"] = ic_asset[model]
    for stage, model in STAGES.items():
        if model in rmse_asset.columns:
            out[f"rmse_{stage}"] = rmse_asset[model]
    if {"ic_fine_tune", "ic_so_ativo"} <= set(out.columns):
        out["ganho_fine_tune_vs_ativo"] = out["ic_fine_tune"] - out["ic_so_ativo"]
    if {"ic_nao_visto", "ic_so_ativo"} <= set(out.columns):
        out["ganho_nao_visto_vs_ativo"] = out["ic_nao_visto"] - out["ic_so_ativo"]
    return out.round(4)


def backtest_direction(models: list[str], panel: pd.DataFrame) -> pd.DataFrame:
    """Long/short diário sobre a probabilidade de alta: posição +1 se p > 0,5, -1 se p < 0,5, 0 se p = 0,5.

    Retorno do dia = média entre os ativos de posição x y_ret_h1 (retorno do dia seguinte, já dividido pela
    volatilidade do ativo, então cada ativo tem risco parecido). Sem custos de transação, sem alavancagem.
    Sharpe anualizado = média / desvio dos retornos diários x sqrt(252).
    """
    ret = panel[["date", "asset", "y_ret_h1"]]
    rows = []
    loaded = {}
    for m in models:
        p = common.RESULTS_DIR / f"preds_{m}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d = d[d["target"] == "y_dir_h1"].merge(ret, on=["date", "asset"]).dropna(subset=["y_ret_h1"])
            if len(d):
                loaded[m] = d
    if not loaded:
        return pd.DataFrame()
    idx = [d.set_index(["date", "asset"]).index for d in loaded.values()]
    common_idx = idx[0]
    for i in idx[1:]:
        common_idx = common_idx.intersection(i)     # mesmas linhas para todos
    for m, d in loaded.items():
        d = d[d.set_index(["date", "asset"]).index.isin(common_idx)].copy()
        d["pnl"] = np.sign(d["y_pred"] - 0.5) * d["y_ret_h1"]
        daily = d.groupby("date")["pnl"].mean()
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan
        traded = d[d["pnl"] != 0]
        rows.append({"model": m, "sharpe": sharpe, "retorno_medio_dia": float(daily.mean()),
                     "acerto_pct": float(100 * (traded["pnl"] > 0).mean()) if len(traded) else np.nan,
                     "dias": int(len(daily))})
    return pd.DataFrame(rows).round(4)


def plots(target: str, by_class: pd.DataFrame, final: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = common.ROOT / "reports" / "figures" / (common.UNIVERSE or "")
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    ax = by_class.plot.bar(figsize=(10, 4.5), width=0.85)
    ax.set_ylabel("IC (Spearman)")
    ax.set_title(f"IC por classe de ativo ({target})")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=7, ncol=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    p = fig_dir / f"ic_por_classe_{target}.png"
    plt.savefig(p, dpi=130)
    plt.close()
    saved.append(p)

    cols = [c for c in ("ganho_fine_tune_vs_ativo", "ganho_nao_visto_vs_ativo") if c in final.columns]
    if cols:
        order = final[cols[0]].sort_values().index
        ax = final.loc[order, cols].plot.barh(figsize=(8, 9), width=0.8)
        ax.set_xlabel("diferença de IC em relação ao modelo treinado só no ativo")
        ax.set_title(f"Ganho de transferência por ativo ({target})")
        ax.axvline(0, color="k", lw=0.6)
        plt.tight_layout()
        p = fig_dir / f"ganho_transferencia_{target}.png"
        plt.savefig(p, dpi=130)
        plt.close()
        saved.append(p)
    return saved


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="y_vol_h5", choices=common.TARGETS)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--no-plots", action="store_true")
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    t, out = args.target, common.RESULTS_DIR
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    allp = load(args.models, t)
    classes = common.load_panel()[["asset", "asset_class"]].drop_duplicates().set_index("asset")["asset_class"]
    allp["asset_class"] = allp["asset"].map(classes)
    print(f"{allp['model'].nunique()} modelos, {allp.groupby(KEYS).ngroups} linhas em comum, "
          f"folds {allp['fold'].min()} a {allp['fold'].max()}")

    gen = general_table(allp, t)
    gen.to_csv(out / f"avaliacao_geral_{t}.csv")
    print("\n=== Geral (média entre folds)\n", gen.to_string())

    asset_ic = group_table(allp, t, "asset")
    asset_ic.to_csv(out / f"avaliacao_ativo_{t}.csv")
    class_ic = group_table(allp, t, "asset_class")
    class_ic.to_csv(out / f"avaliacao_classe_{t}.csv")
    print("\n=== IC por classe\n", class_ic.to_string())

    rmse_asset = pd.DataFrame([{"asset": a, "model": m, "rmse": metrics(g, t)["rmse"]}
                               for (a, m), g in allp.groupby(["asset", "model"])]).pivot(index="asset", columns="model", values="rmse")
    final = final_table(asset_ic, rmse_asset)
    final.to_csv(out / f"tabela_final_{t}.csv")
    print(f"\n=== Tabela final por ativo -> {(out / f'tabela_final_{t}.csv').relative_to(common.ROOT)}\n", final.to_string())

    if t == "y_dir_h1" or (out / "preds_lgbm_geral.parquet").exists():
        bt = backtest_direction([m for m in args.models if m not in ("vol_passada",)], common.load_panel())
        if len(bt):
            bt.to_csv(out / "backtest_direcao.csv", index=False)
            print("\n=== Backtest de direção (long/short, sem custos)\n", bt.to_string(index=False))

    if not args.no_plots:
        for p in plots(t, class_ic, final):
            print("gráfico:", p.relative_to(common.ROOT))


if __name__ == "__main__":
    main()
