"""Vantagem de cada modelo sobre a régua, em vários painéis: ajuda a decidir qual painel usar.

Para cada painel (pasta de reports/results/<painel>/), por alvo e modelo: IC e RMSE médios por ano de
teste, a vantagem sobre a régua (`vol_passada` na volatilidade, `media_historica` nos outros alvos) e em
quantos anos o modelo ganhou da régua.

Quando os painéis têm os mesmos alvos nas linhas em comum (é o caso do painel completo e do cortado com
--start, que mudam só o treino), a comparação usa SÓ essas linhas comuns: mesmos ativos, mesmos dias. Se os
alvos não batem (alinhado x sem alinhar), cada painel usa as próprias linhas e o script avisa: aí só a
vantagem sobre a régua do próprio painel é comparável, não o IC em si.

Uso (depois de rodar run_1b.py em cada painel):
  python src/models/edge_report.py --panels todos_sem_alinhar todos_sem_alinhar_desde2008
  python src/models/edge_report.py --panels todos_sem_alinhar todos_sem_alinhar_desde2008 --from-year 2012
-> reports/results/vantagem_<painéis>.csv e vantagem_<painéis>_por_ano.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODELS = ["media_historica", "vol_passada", "lgbm_ativo", "lgbm_classe", "lgbm_geral"]
KEYS = ["date", "asset", "target"]


def regua(target: str) -> str:
    return "vol_passada" if target == "y_vol_h5" else "media_historica"


def load_panel_preds(panel: str, models: list[str], targets: list[str]) -> pd.DataFrame:
    base = common.ROOT / "reports" / "results" / panel
    frames = []
    for m in models:
        path = base / f"preds_{m}.parquet"
        if path.exists():
            p = pd.read_parquet(path)
            frames.append(p[p["target"].isin(targets)].assign(model=m))
        else:
            print(f"  aviso: falta {path.relative_to(common.ROOT)}")
    if not frames:
        sys.exit(f"Nenhuma previsão em reports/results/{panel}/. Rode: python src/models/run_1b.py --universe {panel}")
    p = pd.concat(frames, ignore_index=True)
    # dentro do painel: só linhas que todos os modelos do alvo previram (como no run_1b)
    n = p.groupby("target")["model"].nunique()
    return p[p.groupby(KEYS)["model"].transform("nunique") == p["target"].map(n)]


def same_targets(preds: dict[str, pd.DataFrame]) -> bool:
    """True se, nas linhas comuns a todos os painéis, o y_true é o mesmo em todos."""
    ys = [p.drop_duplicates(KEYS).set_index(KEYS)["y_true"] for p in preds.values()]
    idx = ys[0].index
    for y in ys[1:]:
        idx = idx.intersection(y.index)
    if len(idx) == 0:
        return False
    return all(np.allclose(ys[0].loc[idx].values, y.loc[idx].values, rtol=1e-6, atol=1e-9, equal_nan=True) for y in ys[1:])


def restrict_common(preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    keys = None
    for p in preds.values():
        k = pd.MultiIndex.from_frame(p[KEYS].drop_duplicates())
        keys = k if keys is None else keys.intersection(k)
    return {name: p[pd.MultiIndex.from_frame(p[KEYS]).isin(keys)] for name, p in preds.items()}


def per_year(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, year), g in p.groupby(["target", "model", p["date"].dt.year]):
        const = g["y_pred"].nunique() < 2
        rows.append({"target": target, "model": model, "ano": year, "n": len(g),
                     "ic": np.nan if const else spearmanr(g["y_true"], g["y_pred"]).statistic,
                     "rmse": float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean()))})
    py = pd.DataFrame(rows)
    ref = py[py["model"] == py["target"].map(regua)].set_index(["target", "ano"])["ic"]
    py["vantagem"] = py["ic"] - py.set_index(["target", "ano"]).index.map(ref).to_numpy(dtype=float)
    return py


def summarize(py: pd.DataFrame) -> pd.DataFrame:
    g = py.groupby(["target", "model"])
    out = g.agg(ic=("ic", "mean"), ic_std=("ic", "std"), rmse=("rmse", "mean"),
                vantagem=("vantagem", "mean"), anos=("ano", "nunique"))
    out["vence_regua"] = g["vantagem"].apply(lambda s: f"{int((s > 0).sum())} de {int(s.notna().sum())}")
    is_ref = out.index.get_level_values("model") == out.index.get_level_values("target").map(regua)
    out.loc[is_ref, ["vantagem", "vence_regua"]] = [np.nan, "régua"]
    return out


def paired_diff(by_year: pd.DataFrame, panels: list[str]) -> pd.DataFrame:
    """IC de cada painel menos o do primeiro, por modelo, pareado por ano (só faz sentido com as mesmas linhas)."""
    wide = by_year.pivot_table(index=["target", "model", "ano"], columns="painel", values="ic")
    rows = []
    for other in panels[1:]:
        d = (wide[other] - wide[panels[0]]).dropna()
        for (target, model), g in d.groupby(level=["target", "model"]):
            rows.append({"target": target, "painel": other, "model": model, "dif_ic": g.mean(), "dif_std": g.std(),
                         "anos_melhor": "igual" if (g.abs() < 1e-12).all() else f"{int((g > 0).sum())} de {len(g)}"})
    return pd.DataFrame(rows).set_index(["target", "painel", "model"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", required=True, help="pastas de reports/results/ a comparar")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS)
    ap.add_argument("--from-year", type=int, default=None, help="só anos de teste a partir deste (ex.: 2012)")
    ap.add_argument("--own-rows", action="store_true", help="cada painel com as próprias linhas, mesmo com alvos iguais")
    args = ap.parse_args(argv)

    preds = {name: load_panel_preds(name, args.models, args.targets) for name in args.panels}
    if args.from_year:
        preds = {k: v[v["date"].dt.year >= args.from_year] for k, v in preds.items()}
    paired = False
    if len(preds) > 1 and not args.own_rows:
        if same_targets(preds):
            paired = True
            preds = restrict_common(preds)
            print("Alvos iguais nos painéis: comparação nas MESMAS linhas (mesmos ativos e dias).")
        else:
            print("AVISO: os alvos diferem entre os painéis (ex.: alinhado x sem alinhar). Cada painel usa as próprias\n"
                  "linhas; compare só a vantagem sobre a régua, não o IC em si.")

    tables, years = [], []
    for name, p in preds.items():
        py = per_year(p).assign(painel=name)
        years.append(py)
        tables.append(summarize(py).assign(painel=name))
    table = pd.concat(tables).reset_index().set_index(["target", "painel", "model"]).sort_index(level=[0, 1], sort_remaining=False)
    by_year = pd.concat(years)

    diff = paired_diff(by_year, args.panels) if paired else None

    stem = "vantagem_" + "_x_".join(args.panels) + (f"_desde{args.from_year}" if args.from_year else "")
    out = common.ROOT / "reports" / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / f"{stem}.csv")
    by_year.round(4).to_csv(out / f"{stem}_por_ano.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== Vantagem sobre a régua (anos de teste: {by_year['ano'].min()} a {by_year['ano'].max()})\n")
    print(table.round(4).to_string())
    if diff is not None:
        diff.round(4).to_csv(out / f"{stem}_diferenca.csv")
        print(f"\n=== Cada painel menos {args.panels[0]}, mesmo modelo, mesmas linhas, ano a ano\n")
        print(diff.round(4).to_string())
    print(f"\n-> reports/results/{stem}.csv e {stem}_por_ano.csv")


if __name__ == "__main__":
    main()
