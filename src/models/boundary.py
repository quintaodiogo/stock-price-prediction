"""Condições de contorno: em que bordas dos dados e em que situações os resultados valem.

Duas partes:

1. Bordas dos dados (por ativo): onde a série começa e termina no painel, quanto se perde no início
   (aquecimento das janelas móveis), quantos retornos são descartados por buracos longos ou preço não
   positivo, e quanto falta de volume e de amplitude (features que ficam vazias).

2. Domínio de validade do modelo: o IC dos modelos recortado por situação, nas MESMAS linhas para todos:
     regime    agitação do mercado no dia da previsão (média da f_z_std21 de todos os ativos naquele dia),
               em terços: calmo, normal, agitado; e os 5% de dias mais agitados ("extremo")
     classe    classe do ativo
     historico anos de histórico que o ativo tinha no painel no dia da previsão (<2, 2-5, 5-10, 10+)
     ano       ano de teste (mostra crises como 2020 e 2022)
   E a cobertura: quantas linhas de teste cada modelo consegue prever, por anos de histórico (o modelo
   por ativo não existe nos primeiros anos de um ativo novo; o geral prevê desde o primeiro dia).
   O IC de cada recorte é calculado ano a ano e depois tirada a média (como nas outras tabelas). Recortes
   com menos de 200 linhas num ano ficam fora daquele ano.

Uso (depois de rodar run_1b.py no painel):
  python src/models/boundary.py --universe todos_sem_alinhar
  python src/models/boundary.py --universe todos_sem_alinhar --targets y_vol_h5 y_ret_h1
-> reports/results/<painel>/contorno/bordas.csv e contorno_<recorte>.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODELS = ["vol_passada", "media_historica", "lgbm_ativo", "lgbm_classe", "lgbm_geral"]
MIN_ROWS = 200
HIST_BINS = [0, 2, 5, 10, np.inf]
HIST_LABELS = ["1) menos de 2 anos", "2) 2 a 5 anos", "3) 5 a 10 anos", "4) 10 anos ou mais"]


# ------------------------------------------------------------------ 1. bordas dos dados
def data_edges(panel: pd.DataFrame) -> pd.DataFrame:
    import features  # noqa: E402
    rows = []
    for asset, g in panel.groupby("asset"):
        row = {"ativo": asset, "classe": g["asset_class"].iloc[0],
               "inicio_painel": g["date"].min().date(), "fim_painel": g["date"].max().date(),
               "linhas": len(g),
               "sem_volume_%": 100 * g["f_volrel"].isna().mean(),
               "sem_amplitude_%": 100 * g["f_range"].isna().mean()}
        path = features.RAW_DIR / f"{asset}_1d.csv"
        if path.exists():
            raw = features.load_raw(asset).sort_values("date").reset_index(drop=True)
            x = features.base_series(raw, asset)
            row.update({
                "inicio_bruto": raw["date"].min().date(),
                "pregoes_brutos": len(raw),
                "aquecimento_pregoes": int((raw["date"] < g["date"].min()).sum()),
                "retornos_descartados": int(x.iloc[1:].isna().sum()),
                "maior_buraco_dias": int(raw["date"].diff().dt.days.max()),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("ativo")


# ------------------------------------------------------------------ 2. domínio de validade
def load_preds(models: list[str], targets: list[str], same_rows: bool = True) -> pd.DataFrame:
    frames = []
    for m in models:
        path = common.RESULTS_DIR / f"preds_{m}.parquet"
        if path.exists():
            p = pd.read_parquet(path)
            frames.append(p[p["target"].isin(targets)].assign(model=m))
    if not frames:
        sys.exit(f"Sem previsões em {common.RESULTS_DIR}. Rode antes: python src/models/run_1b.py --universe <painel>")
    p = pd.concat(frames, ignore_index=True)
    if not same_rows:
        return p
    # mesmas linhas para todos os modelos que previram o alvo
    n = p.groupby("target")["model"].nunique()
    keys = ["date", "asset", "target"]
    return p[p.groupby(keys)["model"].transform("nunique") == p["target"].map(n)]


def labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Rótulos de cada (data, ativo): regime de mercado, classe, anos de histórico."""
    stress = panel.groupby("date")["f_z_std21"].mean().rename("stress")
    first = panel.groupby("asset")["date"].min().rename("primeira_data")
    lab = panel[["date", "asset", "asset_class"]].merge(stress, on="date").merge(first, on="asset")
    years = (lab["date"] - lab["primeira_data"]).dt.days / 365.25
    lab["historico"] = pd.cut(years, HIST_BINS, labels=HIST_LABELS, right=False).astype(str)
    lab["classe"] = lab["asset_class"]
    return lab[["date", "asset", "stress", "classe", "historico"]]


def add_regime(p: pd.DataFrame) -> pd.DataFrame:
    """Terços da agitação calculados nos dias de teste; 'extremo' = 5% mais agitados (também contados em 'agitado')."""
    days = p.drop_duplicates("date")[["date", "stress"]]
    q1, q2, q95 = days["stress"].quantile([1 / 3, 2 / 3, 0.95])
    p = p.assign(regime=np.select([p["stress"] <= q1, p["stress"] <= q2], ["1 calmo", "2 normal"], "3 agitado"))
    extreme = p[p["stress"] >= q95].assign(regime="4 extremo (5% mais agitados)")
    print(f"  regimes (agitação média do mercado): calmo <= {q1:.3f} < normal <= {q2:.3f} < agitado; extremo >= {q95:.3f}")
    return pd.concat([p, extreme], ignore_index=True)


def ic_by(p: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for (target, model, value, fold), g in p.groupby(["target", "model", col, "fold"]):
        if len(g) < MIN_ROWS or g["y_pred"].nunique() < 2:
            continue
        rows.append({"target": target, "model": model, col: value, "fold": fold, "n": len(g),
                     "ic": spearmanr(g["y_true"], g["y_pred"]).statistic})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    table = df.pivot_table(index=["target", col], columns="model", values="ic", aggfunc="mean")
    extra = df[df["model"] == df["model"].iloc[0]].groupby(["target", col]).agg(anos=("fold", "nunique"), linhas=("n", "sum"))
    order = [m for m in MODELS if m in table.columns]
    return table[order].join(extra)


def coverage(p_all: pd.DataFrame, panel: pd.DataFrame, lab: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """% das linhas de teste que cada modelo consegue prever, por anos de histórico do ativo.

    O modelo por ativo precisa de pelo menos 500 linhas de treino do próprio ativo; antes disso não prevê.
    """
    folds = common.load_folds()
    start, end = pd.Timestamp(folds[0]["test_start"]), pd.Timestamp(folds[-1]["test_end"])
    test = lab[(lab["date"] >= start) & (lab["date"] <= end)]
    rows = []
    for target in targets:
        ok = panel.loc[panel[target].notna(), ["date", "asset"]] if target in panel else panel[["date", "asset"]]
        base = test.merge(ok, on=["date", "asset"])
        total = base.groupby("historico").size()
        for model, g in p_all[p_all["target"] == target].groupby("model"):
            hit = base.merge(g[["date", "asset"]], on=["date", "asset"]).groupby("historico").size()
            for h in total.index:
                rows.append({"target": target, "historico": h, "model": model,
                             "cobertura_%": 100 * hit.get(h, 0) / total[h]})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).pivot_table(index=["target", "historico"], columns="model", values="cobertura_%")
    t = t[[m for m in MODELS if m in t.columns]]
    return t.join(base.groupby("historico").size().rename("linhas_teste"), on="historico")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    out = common.RESULTS_DIR / "contorno"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    panel = common.load_panel()
    edges = data_edges(panel)
    edges.round(2).to_csv(out / "bordas.csv")
    print("\n=== 1. Bordas dos dados, por ativo\n")
    print(edges.round(1).to_string())

    lab = labels(panel)
    cov = coverage(load_preds(args.models, args.targets, same_rows=False), panel, lab, args.targets)
    if not cov.empty:
        cov.round(1).to_csv(out / "cobertura_historico.csv")
        print("\n=== Cobertura: % das linhas de teste que cada modelo consegue prever, por anos de histórico\n")
        print(cov.round(1).to_string())
    p = load_preds(args.models, args.targets).merge(lab, on=["date", "asset"], how="left")
    p["ano"] = p["date"].dt.year
    print("\n=== 2. IC por recorte (média dos anos de teste; mesmas linhas para todos os modelos)")
    for col, data in [("regime", add_regime(p)), ("classe", p), ("historico", p), ("ano", p)]:
        table = ic_by(data, col)
        if table.empty:
            continue
        table.round(4).to_csv(out / f"contorno_{col}.csv")
        print(f"\n--- por {col}\n")
        print(table.round(3).to_string())
    # ativos novos: só os modelos que já preveem nos primeiros anos (o por ativo e o por classe ainda não existem)
    early = [m for m in args.models if m not in ("lgbm_ativo", "lgbm_classe")]
    pe = load_preds(early, args.targets).merge(lab, on=["date", "asset"], how="left")
    table = ic_by(pe, "historico")
    if not table.empty:
        table.round(4).to_csv(out / "contorno_historico_inicio.csv")
        print("\n--- por historico, incluindo os primeiros anos (só modelos que já preveem ali)\n")
        print(table.round(3).to_string())
    print(f"\n-> {out.relative_to(common.ROOT)}/")


if __name__ == "__main__":
    main()
