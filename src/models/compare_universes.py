"""Compara os resultados de dois conjuntos de ativos, por exemplo sem e com minis e cripto.

Pergunta 1, "perdemos ou ganhamos ao incluir os ativos novos?":
  compara cada modelo consigo mesmo nos dois conjuntos, nas MESMAS linhas (mesma data, mesmo ativo,
  mesmo alvo); --strict exige as mesmas linhas para todos os modelos ao mesmo tempo.
  Só entram os ativos que existem nos dois conjuntos e os dias que os dois previram; assim a
  diferença vem só de como cada modelo foi treinado (mais ativos, porém menos anos de histórico).
  Por alvo e modelo: IC e RMSE médios por ano de teste nos dois conjuntos, a diferença, e em
  quantos anos o conjunto B teve IC maior que o A.

Pergunta 2, "como o modelo se sai nos ativos novos?":
  só no conjunto B, para cada ativo que não existe no A (ex.: WIN, WDO, BTC, ETH), o IC de cada
  modelo ao longo do tempo (todas as datas de teste do ativo).

Antes, rode os modelos nos dois conjuntos (cada um na sua pasta):
  python src/models/prepare_universe.py --universe sem_minis_cripto
  python src/models/run_1b.py --universe sem_minis_cripto
  python src/models/prepare_universe.py --universe todos
  python src/models/run_1b.py --universe todos
Uso:
  python src/models/compare_universes.py --a sem_minis_cripto --b todos
-> reports/results/comparacao_<a>_vs_<b>.csv e comparacao_<a>_vs_<b>_novos.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DEFAULT_MODELS = ["media_historica", "vol_passada", "lgbm_ativo", "lgbm_classe", "lgbm_geral"]
KEYS = ["date", "asset", "target"]


def load(universe: str, models: list[str]) -> pd.DataFrame:
    base = common.ROOT / "reports" / "results" / universe
    frames = []
    for m in models:
        path = base / f"preds_{m}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path).assign(model=m))
        else:
            print(f"  aviso: {path.relative_to(common.ROOT)} não existe; {m} fica fora para '{universe}'")
    if not frames:
        sys.exit(f"Nenhuma previsão em reports/results/{universe}/. Rode: python src/models/run_1b.py --universe {universe}")
    return pd.concat(frames, ignore_index=True)


def metrics(g: pd.DataFrame, target: str) -> dict:
    """IC (Spearman), RMSE e, na direção, AUC de um grupo de previsões."""
    const = g["y_pred"].nunique() < 2 or g["y_true"].nunique() < 2
    out = {"ic": np.nan if const else spearmanr(g["y_true"], g["y_pred"]).statistic,
           "rmse": float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean())), "n": len(g)}
    if target == "y_dir_h1":
        out["auc"] = np.nan if const else roc_auc_score(g["y_true"], g["y_pred"])
    return out


def common_rows(a: pd.DataFrame, b: pd.DataFrame, strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Linhas (data, ativo, alvo) previstas nos DOIS conjuntos.

    Padrão: modelo a modelo (o lgbm_geral de A e o de B nas mesmas linhas). Com strict=True, só as
    linhas que TODOS os modelos dos dois conjuntos previram (mais justo entre modelos, mas o
    modelo mais restrito, em geral o por ativo, encurta o período de todos).
    """
    if not strict:
        pairs = [common_rows(a[a["model"] == m], b[b["model"] == m], strict=True)
                 for m in sorted(set(a["model"]) & set(b["model"]))]
        return pd.concat([x for x, _ in pairs]), pd.concat([y for _, y in pairs])
    both = pd.concat([a.assign(um="a:" + a["model"]), b.assign(um="b:" + b["model"])], ignore_index=True)
    need = both.groupby("target")["um"].nunique()
    have = both.groupby(KEYS)["um"].nunique()
    keep = have[have.values == have.index.get_level_values("target").map(need).values].index
    idx = lambda d: pd.MultiIndex.from_frame(d[KEYS])  # noqa: E731
    return a[idx(a).isin(keep)], b[idx(b).isin(keep)]


def check_same_targets(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> None:
    """Avisa se o alvo de uma mesma (data, ativo) é diferente nos dois conjuntos.

    Acontece ao comparar painel alinhado com não alinhado: o alinhado descarta os dias em que algum
    mercado fechou, então "a volatilidade dos próximos 5 dias" não é a mesma conta nos dois.
    """
    ya = a.drop_duplicates(KEYS).set_index(KEYS)["y_true"]
    yb = b.drop_duplicates(KEYS).set_index(KEYS)["y_true"]
    both = ya.to_frame("a").join(yb.rename("b"), how="inner")
    if len(both) and (both["a"] - both["b"]).abs().gt(1e-9).mean() > 0.01:
        print(f"\n  ATENÇÃO: os alvos de '{name_a}' e '{name_b}' não são iguais nos mesmos dias e ativos "
              f"({(both['a'] - both['b']).abs().gt(1e-9).mean():.0%} das linhas diferem).\n"
              "  Provavelmente um painel é alinhado e o outro não. A comparação abaixo NÃO é justa:\n"
              "  compare dois conjuntos alinhados, ou dois sem alinhar (--no-align nos dois).\n")


def compare_same_rows(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str,
                      strict: bool = False) -> pd.DataFrame:
    a, b = common_rows(a, b, strict)
    rows = []
    for (target, model), ga in a.groupby(["target", "model"]):
        gb = b[(b["target"] == target) & (b["model"] == model)]
        per_year = []
        for year, ya in ga.groupby(ga["date"].dt.year):
            yb = gb[gb["date"].dt.year == year]
            per_year.append({"ano": year, "a": metrics(ya, target), "b": metrics(yb, target)})
        ic_a = np.array([p["a"]["ic"] for p in per_year], dtype=float)
        ic_b = np.array([p["b"]["ic"] for p in per_year], dtype=float)
        ok = ~np.isnan(ic_a) & ~np.isnan(ic_b)
        row = {"target": target, "model": model,
               f"ic_{name_a}": np.nanmean(ic_a) if ok.any() else np.nan,
               f"ic_{name_b}": np.nanmean(ic_b) if ok.any() else np.nan,
               "ic_diferenca": np.nanmean(ic_b - ic_a) if ok.any() else np.nan,
               "ic_std_anos": np.nanstd(ic_a, ddof=1) if ok.sum() > 1 else np.nan,
               # empate em todos os anos (ex.: modelo por ativo, que não muda): sem vencedor, não "0%"
               f"{name_b}_vence_pct": (100 * (ic_b[ok] > ic_a[ok]).mean()
                                       if ok.any() and np.abs(ic_b[ok] - ic_a[ok]).max() > 1e-12 else np.nan),
               f"rmse_{name_a}": np.mean([p["a"]["rmse"] for p in per_year]),
               f"rmse_{name_b}": np.mean([p["b"]["rmse"] for p in per_year])}
        if target == "y_dir_h1":
            row[f"auc_{name_a}"] = np.nanmean([p["a"].get("auc", np.nan) for p in per_year])
            row[f"auc_{name_b}"] = np.nanmean([p["b"].get("auc", np.nan) for p in per_year])
        row.update({"anos": len(per_year), "primeiro_ano": min(p["ano"] for p in per_year),
                    "linhas": len(ga), "ativos": ga["asset"].nunique()})
        rows.append(row)
    order = {m: i for i, m in enumerate(DEFAULT_MODELS)}
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "model"], key=lambda s: s.map(order) if s.name == "model" else s) \
              .set_index(["target", "model"]).round(4)


def new_assets(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """IC de cada modelo, ao longo do tempo, nos ativos que só existem no conjunto B."""
    novos = sorted(set(b["asset"]) - set(a["asset"]))
    if not novos:
        return pd.DataFrame()
    g = b[b["asset"].isin(novos)]
    rows = [{"target": t, "asset": s, "model": m, **metrics(x, t),
             "periodo": f"{x['date'].min():%Y-%m} a {x['date'].max():%Y-%m}"}
            for (t, s, m), x in g.groupby(["target", "asset", "model"])]
    df = pd.DataFrame(rows)
    ic = df.pivot_table(index=["target", "asset"], columns="model", values="ic").round(3)
    # período e nº de dias do modelo geral, para saber quanto pesa cada número
    ref = df[df["model"] == ("lgbm_geral" if "lgbm_geral" in set(df["model"]) else df["model"].iloc[0])]
    return ic.join(ref.set_index(["target", "asset"])[["n", "periodo"]].rename(columns={"n": "dias"}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="sem_minis_cripto", help="conjunto de referência")
    ap.add_argument("--b", default="todos", help="conjunto com os ativos a mais")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--targets", nargs="+", default=common.TARGETS, choices=common.TARGETS)
    ap.add_argument("--strict", action="store_true",
                    help="mesmas linhas para TODOS os modelos (o período fica o do modelo mais restrito)")
    args = ap.parse_args(argv)

    a = load(args.a, args.models)
    b = load(args.b, args.models)
    a, b = a[a["target"].isin(args.targets)], b[b["target"].isin(args.targets)]
    shared = sorted(set(a["model"]) & set(b["model"]))
    a, b = a[a["model"].isin(shared)], b[b["model"].isin(shared)]

    check_same_targets(a, b, args.a, args.b)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    table = compare_same_rows(a, b, args.a, args.b, args.strict)
    out = common.ROOT / "reports" / "results" / f"comparacao_{args.a}_vs_{args.b}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\n=== 1) Mesmos ativos e mesmos dias: '{args.a}' x '{args.b}'")
    print("    Cada modelo é comparado consigo mesmo nos dois conjuntos, nos mesmos dias e ativos.")
    print("    ic_diferenca > 0 e vence_pct > 50: incluir os ativos novos AJUDOU nesses ativos.\n")
    print(table.to_string())
    print(f"\n-> {out.relative_to(common.ROOT)}")

    novos = new_assets(a, b)
    if not novos.empty:
        out2 = out.with_name(out.stem + "_novos.csv")
        novos.to_csv(out2)
        print(f"\n=== 2) Ativos que só existem em '{args.b}': IC de cada modelo ao longo do tempo\n")
        print(novos.to_string())
        print(f"\n-> {out2.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
