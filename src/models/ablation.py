"""Qual ativo mais impacta o modelo geral? Tira um ativo por vez do treino e mede a perda.

Para cada ativo X do painel, treina o modelo geral SEM X (mesmo walk-forward, mesmos hiperparâmetros) e
compara com o modelo geral completo:

  nos outros ativos   IC do modelo sem X menos IC do completo, nas linhas dos OUTROS ativos (mesmas linhas
                      nos dois). Negativo = tirar X piorou os outros, ou seja, X ajuda (colocar X ganha o
                      mesmo valor com o sinal trocado). Média entre os anos de teste e em quantos anos piorou.
  no próprio ativo    IC ao longo do tempo no próprio X: visto (modelo completo) x nunca visto (modelo sem X).
                      É a pergunta da transferência (Pessoa C), aqui como subproduto.

Só entram os anos de teste em que X já estava no treino (antes disso os dois modelos são o mesmo).

Ruído: mudar só a semente aleatória do LightGBM já muda um pouco o IC. O script treina o modelo completo com
outras sementes (--noise-seeds) e mede essa variação nas mesmas linhas; um efeito menor que o ruído não deve
ser lido como impacto do ativo.

É demorado (um modelo geral por ativo; 32 ativos = 32 treinos). Cada treino fica salvo em
reports/results/<painel>/ablacao/; se parar no meio, rode de novo o mesmo comando e ele continua de onde parou.

Uso:
  python src/models/ablation.py --universe todos_sem_alinhar
  python src/models/ablation.py --universe todos_sem_alinhar --assets BTC ETH WIN WDO     (só alguns)
  python src/models/ablation.py --universe todos_sem_alinhar --by classe                  (uma classe inteira por vez)
-> reports/results/<painel>/ablacao_<ativo|classe>.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

KEEP = ["date", "asset", "fold", "target", "y_true", "y_pred"]


def _ic(g: pd.DataFrame) -> float:
    if len(g) < 3 or g["y_pred"].nunique() < 2 or g["y_true"].nunique() < 2:
        return np.nan
    return spearmanr(g["y_true"], g["y_pred"]).statistic


def _rmse(g: pd.DataFrame) -> float:
    return float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean()))


def run_geral(panel, folds, targets, exclude, seed=None) -> pd.DataFrame:
    import tree  # noqa: E402  (importa o LightGBM só quando precisa treinar)
    old = tree.PARAMS["seed"]
    if seed is not None:
        tree.PARAMS["seed"] = seed
    try:
        frames = tree.run(panel, folds, "geral", targets, exclude, save_models=False)
    finally:
        tree.PARAMS["seed"] = old
    return pd.concat(frames, ignore_index=True)[KEEP]


def cached(path: Path, make) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    df = make()
    df.to_parquet(path, index=False)
    print(f"  -> {path.name} ({time.time() - t0:.0f}s)", flush=True)
    return df


def per_fold(p: pd.DataFrame, rows: pd.Series) -> pd.DataFrame:
    """IC e RMSE por alvo e ano de teste, só nas linhas marcadas em `rows`."""
    out = [{"target": t, "fold": f, "ic": _ic(g), "rmse": _rmse(g)} for (t, f), g in p[rows].groupby(["target", "fold"])]
    return pd.DataFrame(out).set_index(["target", "fold"])


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mesmas linhas (date, asset, target) nos dois conjuntos de previsões."""
    k = ["date", "asset", "target"]
    m = a.merge(b[k + ["y_pred"]], on=k, suffixes=("_a", "_b"))
    return (m.drop(columns="y_pred_b").rename(columns={"y_pred_a": "y_pred"}),
            m.drop(columns="y_pred_a").rename(columns={"y_pred_b": "y_pred"}))


def effect(full: pd.DataFrame, other: pd.DataFrame, removed: list[str],
           folds_ok: set[int] | None = None) -> dict[str, dict]:
    """Por alvo: mudança no IC/RMSE dos outros ativos e IC no(s) ativo(s) tirado(s).

    `folds_ok`: anos de teste em que o ativo tirado estava no treino; nos outros os dois modelos são iguais
    e a diferença seria zero por construção, então ficam fora da média.
    """
    a, b = align(full, other)
    if folds_ok is not None:
        a, b = a[a["fold"].isin(folds_ok)], b[b["fold"].isin(folds_ok)]
    others = ~a["asset"].isin(removed)
    fa, fb = per_fold(a, others), per_fold(b, others)
    d_ic, d_rmse = fb["ic"] - fa["ic"], fb["rmse"] - fa["rmse"]
    res = {}
    for target in fa.index.get_level_values("target").unique():
        di = d_ic.loc[target]
        own_a, own_b = a[~others & (a["target"] == target)], b[~others & (b["target"] == target)]
        res[target] = {
            "ic_outros_completo": fa.loc[target, "ic"].mean(),
            "dif_ic_outros": di.mean(), "dif_ic_outros_std": di.std(),
            "anos_pior": f"{int((di < -1e-12).sum())} de {int(di.notna().sum())}",
            "dif_rmse_outros": d_rmse.loc[target].mean(),
            "ic_proprio_visto": _ic(own_a), "ic_proprio_nao_visto": _ic(own_b),
            "linhas_proprio": len(own_a),
        }
    return res


def noise(full: pd.DataFrame, seeds: list[pd.DataFrame], removed: list[str], folds_ok: set[int]) -> dict[str, float]:
    """Maior mudança de IC nos outros ativos causada só por trocar a semente."""
    worst: dict[str, float] = {}
    for s in seeds:
        for target, r in effect(full, s, removed, folds_ok).items():
            worst[target] = max(worst.get(target, 0.0), abs(r["dif_ic_outros"]))
    return worst


def groups_to_remove(panel: pd.DataFrame, by: str, only: list[str] | None) -> dict[str, list[str]]:
    if by == "classe":
        cls = panel.drop_duplicates("asset").set_index("asset")["asset_class"]
        groups = {c: sorted(cls[cls == c].index) for c in sorted(cls.unique())}
    else:
        groups = {a: [a] for a in sorted(panel["asset"].unique())}
    if only:
        unknown = set(only) - set(groups)
        if unknown:
            sys.exit(f"Fora do painel: {sorted(unknown)}. Opções: {', '.join(groups)}")
        groups = {k: v for k, v in groups.items() if k in only}
    return groups


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS,
                    help="padrão: só volatilidade (nos retornos não há sinal para perder)")
    ap.add_argument("--by", choices=["ativo", "classe"], default="ativo", help="tirar um ativo ou uma classe por vez")
    ap.add_argument("--assets", nargs="+", default=None, help="só estes ativos (ou classes, com --by classe)")
    ap.add_argument("--noise-seeds", type=int, default=2, help="sementes extras para medir o ruído (0 = não mede)")
    common.add_universe_arg(ap)
    args = ap.parse_args(argv)
    common.use_universe(args.universe)

    panel, folds = common.load_panel(), common.load_folds()
    folder = common.RESULTS_DIR / "ablacao"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(args.targets)
    groups = groups_to_remove(panel, args.by, args.assets)
    print(f"Ablação ({args.by}, alvos {args.targets}): {len(groups)} treinos + 1 completo + {args.noise_seeds} de ruído, "
          f"{len(folds)} folds cada")

    full = cached(folder / f"preds_completo_{suffix}.parquet", lambda: run_geral(panel, folds, args.targets, []))
    seeds = [cached(folder / f"preds_semente{s}_{suffix}.parquet",
                    lambda s=s: run_geral(panel, folds, args.targets, [], seed=common.SEED + s))
             for s in range(1, args.noise_seeds + 1)]

    rows = []
    for i, (name, members) in enumerate(groups.items(), 1):
        print(f"[{i}/{len(groups)}] sem {name}", flush=True)
        other = cached(folder / f"preds_sem_{name}_{suffix}.parquet",
                       lambda m=members: run_geral(panel, folds, args.targets, m))
        first = panel.loc[panel["asset"].isin(members), "date"].min()
        folds_ok = {f["fold"] for f in folds if pd.Timestamp(f["train_end"]) >= first}
        if not folds_ok:
            print(f"  {name} não aparece em nenhum treino; fica fora")
            continue
        nz = noise(full, seeds, members, folds_ok) if seeds else {}
        for target, r in effect(full, other, members, folds_ok).items():
            ruido = nz.get(target, np.nan)
            rows.append({"target": target, args.by: name, **r, "anos_no_treino": len(folds_ok), "ruido": ruido,
                         "acima_do_ruido": bool(abs(r["dif_ic_outros"]) > ruido) if seeds else None})

    table = pd.DataFrame(rows).sort_values(["target", "dif_ic_outros"]).set_index(["target", args.by])
    out = common.RESULTS_DIR / f"ablacao_{args.by}.csv"
    table.round(4).to_csv(out)
    pd.set_option("display.width", 220)
    print("\n=== Tirar do treino do modelo geral (dif < 0: o ativo ajudava os outros)\n")
    print(table.round(4).to_string())
    print(f"\n-> {out.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
