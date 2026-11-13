"""Gera o painel e os folds de um conjunto de ativos numa pasta só dele.

Usa as funções do src/features.py (Pessoa A) sem mudá-las; a única diferença é onde grava:
data/processed/<conjunto>/panel.parquet e folds.json, em vez de data/processed/. Assim os
conjuntos não se sobrescrevem e dá para comparar resultados com e sem minis e cripto.

Conjuntos (src/assets.py):
  sem_minis_cripto  28 ativos, desde 2008  (o usado nos resultados até agora)
  sem_minis         30 ativos (+ BTC, ETH), desde 2018
  todos             32 ativos (+ BTC, ETH, WIN, WDO), desde 2021

Com --no-align o painel NÃO é cortado nas datas em que todos os ativos negociaram: cada ativo entra
com o próprio histórico (os 28 desde 2008, a cripto desde 2014/2017, os minis desde 2020). Assim incluir
minis e cripto não encurta o histórico dos outros. Grava em data/processed/<conjunto>_sem_alinhar/.

Uso:
  python src/models/prepare_universe.py --universe todos
  python src/models/prepare_universe.py --universe todos --no-align     (-> todos_sem_alinhar)
  python src/models/prepare_universe.py --universe sem_minis_cripto
Depois, os scripts de modelo com o mesmo --universe, por exemplo:
  python src/models/run_1b.py --universe todos
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import features  # noqa: E402
import normalization as nz  # noqa: E402
from assets import ASSETS, UNIVERSES  # noqa: E402


def missing_raw(names: list[str]) -> list[str]:
    return [n for n in names if not (features.RAW_DIR / f"{n}_1d.csv").exists()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=list(UNIVERSES))
    ap.add_argument("--norm", default="vol_scaled", choices=nz.ALL_NORMS)
    ap.add_argument("--no-align", action="store_true",
                    help="não corta o painel nas datas comuns: cada ativo entra com o próprio histórico. "
                         "Grava em <conjunto>_sem_alinhar (ex.: todos_sem_alinhar)")
    args = ap.parse_args(argv)
    if args.no_align and args.norm == nz.CROSS_SECTION:
        sys.exit("A normalização cross_section compara os ativos no mesmo dia e precisa do painel alinhado.")
    folder = args.universe + ("_sem_alinhar" if args.no_align else "")

    names = UNIVERSES[args.universe]
    missing = missing_raw(names)
    if missing:
        minis = [n for n in missing if ASSETS[n][1] == "mini"]
        others = [n for n in missing if n not in minis]
        msg = [f"Faltam dados brutos em data/raw/ para: {', '.join(missing)}."]
        if others:
            msg.append(f"  -> python src/download_data.py {' '.join(others)}")
        if minis:
            msg.append("  -> minis: coloque data/intraday/WIN_M1.csv e WDO_M1.csv (arquivos do MT5, peça à Pessoa A)"
                       " e rode: python src/minis_to_daily.py")
        sys.exit("\n".join(msg))

    raw = {n: features.load_raw(n) for n in names}
    panel = features.build_panel(raw, args.norm, align=not args.no_align)
    first = max(features.FIRST_TEST_YEAR, panel["date"].min().year + 1)
    folds = features.make_folds(panel["date"].max(), first)

    common.use_universe(folder)
    common.PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(common.PANEL_PATH, index=False)
    common.FOLDS_PATH.write_text(json.dumps(folds, indent=2))
    (common.PANEL_PATH.parent / "info.json").write_text(json.dumps(
        {"universe": args.universe, "norm": args.norm, "alinhado": not args.no_align,
         **common.panel_signature(panel, folds)},
        indent=2, ensure_ascii=False))
    print(f"painel ({folder}, {args.norm}): {len(panel)} linhas, {panel['asset'].nunique()} ativos, "
          f"{panel['date'].nunique()} datas{' comuns' if not args.no_align else ''}, {panel['date'].min().date()} -> {panel['date'].max().date()}, "
          f"{len(folds)} folds -> {common.PANEL_PATH.parent.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
