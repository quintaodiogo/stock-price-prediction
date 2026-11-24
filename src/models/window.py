"""Quanto dado é preciso para treinar o modelo geral sem perder desempenho? (pedido do professor)

Treina o LightGBM geral várias vezes, cada uma estudando só os últimos N anos antes de cada ano de prova
(N = 1, 2, 3, 5 e 10, por padrão), e compara com o modelo que estuda todo o passado. As provas são as mesmas
17 (2010 a 2026); só muda quanto passado o modelo vê. Assim a pergunta "qual a menor quantidade de dados que
não perde desempenho" é respondida em todos os anos, e não só depois de uma data.

Leitura (mesmas regras do report.py): diferença menor que 0,005 de IC = praticamente igual; "perde de forma
consistente" = pior em pelo menos 70% dos anos. A menor janela que fica "praticamente igual" é a resposta.

Complementa o corte fixo em 2021 (pandemia), que é feito com prepare_universe.py --start (docs/modelo.md, 19).

Uso:
  python src/models/window.py --universe todos_sem_alinhar                 (treina o que faltar e compara)
  python src/models/window.py --universe todos_sem_alinhar --years 1 3     (só algumas janelas)
-> reports/results/<painel>/janela_treino.csv e janela_treino_por_ano.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

YEARS = [1, 2, 3, 5, 10]
MIN_IC = 0.005


def name(n: int) -> str:
    return f"lgbm_geral_janela{n}a"


def verdict(diff: float, worse: int, total: int) -> str:
    if total == 0 or pd.isna(diff):
        return "sem dados"
    if abs(diff) < MIN_IC:
        return "praticamente igual"
    if diff < 0 and worse >= 0.7 * total:
        return "perde de forma consistente"
    if diff > 0 and worse <= 0.3 * total:
        return "ganha de forma consistente"
    return "empate (muda de sinal entre os anos)"


def compare(years: list[int], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import run_1b  # noqa: E402
    models = ["lgbm_geral", "vol_passada"] + [name(n) for n in years]
    allp = run_1b.load_all([m for m in models if (common.RESULTS_DIR / f"preds_{m}.parquet").exists()])
    pf = run_1b.per_fold(allp[allp["target"] == target])
    w = pf.pivot_table(index="fold", columns="model", values="ic")
    rows = [{"janela": "todo o passado", "anos": None, "ic": w["lgbm_geral"].mean(), "dif_vs_completo": 0.0,
             "anos_pior": "—", "leitura": "referência"}]
    for n in years:
        m = name(n)
        if m not in w:
            continue
        d = w[m] - w["lgbm_geral"]
        worse = int((d < 0).sum())
        rows.append({"janela": f"últimos {n} ano{'s' if n > 1 else ''}", "anos": n, "ic": w[m].mean(),
                     "dif_vs_completo": d.mean(), "anos_pior": f"{worse} de {len(d)}",
                     "leitura": verdict(d.mean(), worse, len(d))})
    if "vol_passada" in w:
        rows.append({"janela": "régua (não treina)", "anos": None, "ic": w["vol_passada"].mean(),
                     "dif_vs_completo": (w["vol_passada"] - w["lgbm_geral"]).mean(), "anos_pior": "—",
                     "leitura": "piso"})
    return pd.DataFrame(rows), w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=YEARS, help="janelas de treino, em anos")
    ap.add_argument("--target", default="y_vol_h5", choices=common.TARGETS)
    ap.add_argument("--no-train", action="store_true", help="só compara o que já existe")
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    import tree  # noqa: E402

    if not (common.RESULTS_DIR / "preds_lgbm_geral.parquet").exists():
        sys.exit("Falta o modelo geral completo. Rode antes: python src/models/run_1b.py --universe <painel>")
    if not args.no_train:
        for n in args.years:
            if (common.RESULTS_DIR / f"preds_{name(n)}.parquet").exists():
                print(f"{name(n)}: já existe, pulando")
                continue
            uni = ["--universe", args.universe] if args.universe else []
            tree.main(["--mode", "geral", "--targets", args.target, "--train-years", str(n), "--no-save-models"] + uni)

    table, by_year = compare(args.years, args.target)
    out = common.RESULTS_DIR / "janela_treino.csv"
    table.round(4).to_csv(out, index=False)
    by_year.round(4).to_csv(common.RESULTS_DIR / "janela_treino_por_ano.csv")
    pd.set_option("display.width", 200)
    print(f"\n=== Quanto passado o modelo geral precisa estudar ({args.target}; IC médio nos anos de prova)\n")
    print(table.round(4).to_string(index=False))
    ok = table[(table["leitura"] == "praticamente igual") & table["anos"].notna()]
    if len(ok):
        print(f"\nMenor janela praticamente igual ao completo: {int(ok['anos'].min())} ano(s).")
    print(f"\n-> {out.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
