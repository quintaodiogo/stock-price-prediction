"""Quais features o modelo geral mais usa: importância por ganho, média entre os anos de teste.

Lê os modelos salvos pelo tree.py (models/[<painel>/]lgbm_<modelo>_<alvo>_fold<N>.txt). Para cada ano de
teste, a importância de cada feature é a parte (%) da melhora total do modelo que veio das divisões feitas
com ela ("gain"); depois tira a média e o desvio entre os anos. Também soma por grupo de feature.

Uso:
  python src/models/importance.py --universe todos_sem_alinhar
  python src/models/importance.py --universe todos_sem_alinhar --model geral_mercado   (com features de mercado)
-> reports/results/<painel>/importancia_<modelo>_<alvo>.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NAMES = {   # nome técnico (docs/features.md)
    "f_z_lag0": "Retorno normalizado (t)",
    "f_z_lag1": "Retorno normalizado (t-1)",
    "f_z_lag2": "Retorno normalizado (t-2)",
    "f_z_lag3": "Retorno normalizado (t-3)",
    "f_z_lag4": "Retorno normalizado (t-4)",
    "f_z_std5": "Volatilidade realizada relativa, 5 dias",
    "f_z_std21": "Volatilidade realizada relativa, 21 dias",
    "f_z_std63": "Volatilidade realizada relativa, 63 dias",
    "f_z_mean21": "Momento ajustado ao risco, 21 dias",
    "f_z_mean63": "Momento ajustado ao risco, 63 dias",
    "f_rsi14": "Índice de força relativa (IFR), 14 dias",
    "f_range": "Amplitude diária relativa",
    "f_volrel": "Volume relativo, 21 dias",
    "f_mkt_std5": "Volatilidade relativa do mercado, 5 dias",
    "f_mkt_std21": "Volatilidade relativa do mercado, 21 dias",
    "f_mkt_move": "Movimento absoluto do mercado (t-1)",
    "f_cls_std21": "Volatilidade relativa da classe, 21 dias",
}
GROUPS = [("f_z_lag", "retornos recentes"), ("f_z_std", "volatilidade do próprio ativo"),
          ("f_z_mean", "tendência"), ("f_rsi", "IFR"), ("f_range", "amplitude"), ("f_volrel", "volume"),
          ("f_mkt", "mercado"), ("f_cls", "mercado")]


def group_of(feat: str) -> str:
    return next((g for prefix, g in GROUPS if feat.startswith(prefix)), "outro")


def importance(files: list[Path]) -> pd.DataFrame:
    import lightgbm as lgb  # noqa: E402
    per_fold = {}
    for f in files:
        booster = lgb.Booster(model_file=str(f))
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        per_fold[int(re.search(r"fold(\d+)", f.stem).group(1))] = 100 * gain / gain.sum()
    df = pd.DataFrame(per_fold).fillna(0)
    out = pd.DataFrame({"nome": [NAMES.get(c, c) for c in df.index], "grupo": [group_of(c) for c in df.index],
                        "importancia_%": df.mean(axis=1), "desvio_%": df.std(axis=1)}, index=df.index)
    return out.sort_values("importancia_%", ascending=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="geral", help="parte do nome depois de lgbm_ (geral, geral_mercado...)")
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS)
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    pd.set_option("display.width", 200)
    for target in args.targets:
        files = sorted(common.MODELS_DIR.glob(f"lgbm_{args.model}_{target}_fold*.txt"))
        if not files:
            print(f"Nenhum modelo lgbm_{args.model}_{target}_fold*.txt em {common.MODELS_DIR}. "
                  f"Rode: python src/models/tree.py --mode geral{' --market' if 'mercado' in args.model else ''}"
                  f"{' --universe ' + args.universe if args.universe else ''}")
            continue
        table = importance(files)
        out = common.RESULTS_DIR / f"importancia_{args.model}_{target}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.round(2).to_csv(out)
        print(f"\n=== Importância (ganho, % do total) - lgbm_{args.model}, {target}, {len(files)} anos de teste\n")
        print(table.round(1).to_string())
        print("\nPor grupo:")
        print(table.groupby("grupo")["importancia_%"].sum().sort_values(ascending=False).round(1).to_string())
        print(f"\n-> {out.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
