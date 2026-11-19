"""A variabilidade de preço dos OUTROS ativos ajuda a prever? Features de mercado e o experimento.

Features novas (todas com dados até o pregão ANTERIOR a t, para não usar o fechamento de um mercado que
fecha depois do ativo no mesmo dia; ex.: Nova York fecha depois de Tóquio):

  f_mkt_std5    agitação do mercado na última semana: média, entre os OUTROS ativos, da f_z_std5 deles
  f_mkt_std21   idem com a janela de um mês (f_z_std21)
  f_mkt_move    tamanho do último movimento do mercado: média, entre os outros ativos, de |f_z_lag0|
  f_cls_std21   agitação da própria classe no último mês: média da f_z_std21 dos OUTROS ativos da classe

"Outros" = o próprio ativo fica fora da média (leave-one-out), para a feature trazer só informação dos
demais. Cada ativo entra com o valor mais recente que tinha até o pregão anterior (no máximo 5 datas de
atraso, por causa de feriados e fins de semana da cripto). Média de mercado exige pelo menos 5 ativos;
a de classe, pelo menos 1 outro ativo da classe.

Experimento: os mesmos modelos com e sem essas features, nas mesmas linhas.
  - por ativo com mercado x por ativo: o ativo sozinho ganha algo ao "ver" os outros?
  - geral com mercado x geral: o modelo geral já captura isso ao treinar com todos?

Uso:
  python src/models/tree.py --mode ativo --market --universe todos_sem_alinhar
  python src/models/tree.py --mode geral --market --universe todos_sem_alinhar
  python src/models/market.py --universe todos_sem_alinhar
-> reports/results/<painel>/mercado_comparacao.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MAX_STALE = 5          # datas do calendário do painel que um valor pode ser repetido (feriados)
MIN_MARKET = 5         # mínimo de outros ativos na média de mercado
MARKET_COLS = ["f_mkt_std5", "f_mkt_std21", "f_mkt_move", "f_cls_std21"]


def _lagged_wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tabela datas x ativos com o valor de `col` de cada ativo até a data ANTERIOR do calendário do painel."""
    wide = panel.pivot(index="date", columns="asset", values=col).sort_index()
    return wide.ffill(limit=MAX_STALE).shift(1)


def _leave_one_out_mean(wide: pd.DataFrame, groups: dict[str, list[str]], min_others: int) -> pd.DataFrame:
    """Para cada ativo, média dos OUTROS ativos do mesmo grupo, por data."""
    out = {}
    for members in groups.values():
        sub = wide[members]
        total, count = sub.sum(axis=1, min_count=1), sub.notna().sum(axis=1)
        for a in members:
            own = sub[a]
            n = count - own.notna()
            out[a] = ((total - own.fillna(0)) / n).where(n >= min_others)
    return pd.DataFrame(out)


def _long(wide: pd.DataFrame, name: str) -> pd.Series:
    return wide.stack(future_stack=True).rename(name)


def add_market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Devolve o painel com as colunas MARKET_COLS (nomes f_*, então entram como features dos modelos)."""
    panel = panel.drop(columns=[c for c in MARKET_COLS if c in panel])
    assets = sorted(panel["asset"].unique())
    cls = panel.drop_duplicates("asset").set_index("asset")["asset_class"]
    everyone = {"todos": assets}
    by_class = {c: sorted(cls[cls == c].index) for c in cls.unique()}

    panel = panel.assign(_abs_lag0=panel["f_z_lag0"].abs())
    parts = [
        _long(_leave_one_out_mean(_lagged_wide(panel, "f_z_std5"), everyone, MIN_MARKET), "f_mkt_std5"),
        _long(_leave_one_out_mean(_lagged_wide(panel, "f_z_std21"), everyone, MIN_MARKET), "f_mkt_std21"),
        _long(_leave_one_out_mean(_lagged_wide(panel, "_abs_lag0"), everyone, MIN_MARKET), "f_mkt_move"),
        _long(_leave_one_out_mean(_lagged_wide(panel, "f_z_std21"), by_class, 1), "f_cls_std21"),
    ]
    feats = pd.concat(parts, axis=1)
    feats.index.names = ["date", "asset"]
    out = panel.drop(columns="_abs_lag0").merge(feats.reset_index(), on=["date", "asset"], how="left")
    return out


# ---------------------------------------------------------------------------- comparação
PAIRS = [("lgbm_ativo", "lgbm_ativo_mercado"), ("lgbm_classe", "lgbm_classe_mercado"),
         ("lgbm_geral", "lgbm_geral_mercado")]


def compare(pairs: list[tuple[str, str]], targets: list[str]) -> pd.DataFrame:
    import run_1b  # noqa: E402  (importa o LightGBM; só aqui)
    rows = []
    for base, new in pairs:
        if not all((common.RESULTS_DIR / f"preds_{m}.parquet").exists() for m in (base, new)):
            print(f"  aviso: falta preds de {base} ou {new}; par fora da comparação")
            continue
        allp = run_1b.load_all([base, new])
        allp = allp[allp["target"].isin(targets)]
        pf = run_1b.per_fold(allp).pivot_table(index=["target", "fold"], columns="model", values="ic")
        for target, g in pf.groupby(level="target"):
            d = g[new] - g[base]
            rows.append({"target": target, "modelo": base, "ic_sem": g[base].mean(), "ic_com": g[new].mean(),
                         "ganho": d.mean(), "ganho_std": d.std(),
                         "anos_melhor": f"{int((d > 0).sum())} de {int(d.notna().sum())}"})
    if not rows:
        sys.exit("Nenhum par com previsões. Rode antes tree.py com e sem --market (ver o topo deste arquivo).")
    return pd.DataFrame(rows).set_index(["target", "modelo"]).sort_index()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=common.TARGETS, choices=common.TARGETS)
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    table = compare(PAIRS, args.targets)
    out = common.RESULTS_DIR / "mercado_comparacao.csv"
    table.round(4).to_csv(out)
    pd.set_option("display.width", 200)
    print("\n=== Com x sem features de mercado (IC médio por ano de teste; ganho = com - sem, mesmas linhas)\n")
    print(table.round(4).to_string())
    print(f"\n-> {out.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
