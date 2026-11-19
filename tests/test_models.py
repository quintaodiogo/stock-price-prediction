"""Testes da Pessoa B: baselines sem vazamento e previsões no formato do contrato."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
import baselines  # noqa: E402
import common  # noqa: E402
import features  # noqa: E402
from test_no_leakage import synthetic  # noqa: E402

CUT = pd.Timestamp("2017-06-30")


def _raw():
    return {"SP500": synthetic("SP500", seed=1), "VIX": synthetic("VIX", seed=2, level=True),
            "BTC": synthetic("BTC", seed=3)}


def test_past_vol_ignores_future():
    """Trocar os preços futuros por lixo não pode mudar a vol. passada de datas até o corte."""
    raw = _raw()
    broken = {k: v.copy() for k, v in raw.items()}
    for v in broken.values():
        v.loc[v["date"] > CUT, ["Open", "High", "Low", "Close"]] *= 3.0
    a, b = (features.build_panel(r) for r in (raw, broken))
    a["pv"], b["pv"] = baselines.past_vol(a), baselines.past_vol(b)
    ka = a[a["date"] <= CUT].set_index(["date", "asset"])["pv"]
    kb = b.set_index(["date", "asset"]).loc[ka.index, "pv"]
    assert ka.notna().sum() > 100
    pd.testing.assert_series_equal(ka, kb, check_exact=False, rtol=1e-9)


def test_preds_follow_contract():
    panel = features.build_panel(_raw())
    folds = features.make_folds(panel["date"].max(), first_year=2016)
    out = baselines.run(panel, folds)
    for frames in out.values():
        preds = pd.concat(frames)
        assert list(preds.columns) == common.PRED_COLUMNS
        assert preds["y_true"].notna().all() and np.isfinite(preds["y_pred"]).all()
        for f in folds:
            dates = preds.loc[preds["fold"] == f["fold"], "date"]
            assert dates.min() >= pd.Timestamp(f["test_start"]) and dates.max() <= pd.Timestamp(f["test_end"])


def test_tree_inner_validation_has_embargo():
    import tree
    panel = features.build_panel(_raw())
    fit, val = tree.inner_split(panel)
    gap = val["date"].min() - fit["date"].max()
    assert len(fit) and len(val) and gap.days > tree.EMBARGO_DAYS


def test_features_never_include_targets():
    panel = features.build_panel(_raw())
    cols = common.feature_cols(panel)
    assert cols and not any(c.startswith(("y_", "r_")) for c in cols)


def test_tune_candidates_start_with_current_config():
    import tree
    import tune
    cands = tune.candidates(8)
    assert len(cands) == 9
    assert cands[0] == {k: tree.PARAMS[k] for k in tune.TUNED_KEYS}
    assert len({tuple(sorted(c.items())) for c in cands}) == 9      # sem repetição
    assert all(set(c) == set(tune.GRID) for c in cands)


def test_universe_folders_do_not_overwrite_each_other():
    import importlib
    c = importlib.reload(common)
    default = (c.PANEL_PATH, c.RESULTS_DIR, c.MODELS_DIR)
    c.use_universe(None)                                   # sem conjunto: nada muda
    assert (c.PANEL_PATH, c.RESULTS_DIR, c.MODELS_DIR) == default
    c.use_universe("todos")
    assert c.PANEL_PATH.parent.name == "todos" and c.RESULTS_DIR.name == "todos" and c.MODELS_DIR.name == "todos"
    importlib.reload(common)                               # volta ao padrão para os outros testes


def test_compare_universes_uses_same_rows_per_model():
    import compare_universes as cu
    d = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    row = lambda m, dates, asset="SP500": pd.DataFrame(  # noqa: E731
        {"date": dates, "asset": asset, "fold": 0, "target": "y_vol_h5", "y_true": 1.0, "y_pred": 1.0, "model": m})
    a = pd.concat([row("lgbm_geral", d), row("lgbm_ativo", d[:1])])
    b = pd.concat([row("lgbm_geral", d[1:]), row("lgbm_ativo", d), row("lgbm_geral", d, asset="BTC")])
    ra, rb = cu.common_rows(a, b)                           # modelo a modelo
    assert set(ra.loc[ra.model == "lgbm_geral", "date"]) == set(d[1:])   # só os dias dos dois lados
    assert set(ra.loc[ra.model == "lgbm_ativo", "date"]) == set(d[:1])
    assert "BTC" not in set(rb["asset"])                    # ativo que só existe em B fica fora
    sa, sb = cu.common_rows(a, b, strict=True)              # todos os modelos juntos: só o dia comum a todos
    assert sa["date"].nunique() == 0 or set(sa["date"]) <= set(d[:1])


def test_unaligned_panel_keeps_each_asset_independent():
    """Sem alinhar, incluir um ativo novo não muda nenhuma linha dos outros (base da comparação justa)."""
    raw = _raw()
    small = {k: v for k, v in raw.items() if k != "BTC"}
    p_all = features.build_panel(raw, align=False)
    p_small = features.build_panel(small, align=False)
    a = p_all[p_all["asset"] != "BTC"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, p_small.reset_index(drop=True))


# ---------------------------------------------------------------- pendências do professor
def _raw_many():
    """Oito ativos de classes diferentes (as features de mercado pedem pelo menos 5 outros ativos)."""
    names = ["SP500", "NASDAQ", "DAX", "GOLD", "WTI", "EURUSD", "BTC", "PETR4"]
    return {n: synthetic(n, seed=i + 10) for i, n in enumerate(names)}


def test_history_cut_keeps_remaining_rows_identical():
    """--start só tira linhas antigas: as que sobram são iguais às do painel completo (features já calculadas)."""
    import prepare_universe as pu
    panel = features.build_panel(_raw(), align=False)
    cut = pu.cut(panel, "2016-01-01")
    assert cut["date"].min() >= pd.Timestamp("2016-01-01")
    expected = panel[panel["date"] >= pd.Timestamp("2016-01-01")].reset_index(drop=True)
    pd.testing.assert_frame_equal(cut, expected)
    assert pu.folder_name("todos", True, "2008-01-01") == "todos_sem_alinhar_desde2008"


def test_market_features_use_only_previous_days():
    """Mudar os preços a partir de uma data não pode mudar as features de mercado até essa data (inclusive)."""
    import market
    raw = _raw_many()
    broken = {k: v.copy() for k, v in raw.items()}
    for v in broken.values():
        v.loc[v["date"] >= CUT, ["Open", "High", "Low", "Close"]] *= 3.0
    a, b = (market.add_market_features(features.build_panel(r, align=False)) for r in (raw, broken))
    cols = market.MARKET_COLS
    ka = a[a["date"] <= CUT].set_index(["date", "asset"])[cols]
    kb = b.set_index(["date", "asset"]).loc[ka.index, cols]
    assert ka["f_mkt_std21"].notna().sum() > 100
    pd.testing.assert_frame_equal(ka, kb, check_exact=False, rtol=1e-9)


def test_market_features_leave_the_asset_out():
    """A feature de mercado de um ativo é a média dos OUTROS: mudar o próprio ativo não a altera."""
    import market
    raw = _raw_many()
    changed = dict(raw, SP500=synthetic("SP500", seed=999))
    a, b = (market.add_market_features(features.build_panel(r, align=False)) for r in (raw, changed))
    key = ["date", "asset"]
    sa = a[a["asset"] == "SP500"].set_index(key)[["f_mkt_std21", "f_mkt_move"]]
    sb = b[b["asset"] == "SP500"].set_index(key).loc[sa.index, ["f_mkt_std21", "f_mkt_move"]]
    pd.testing.assert_frame_equal(sa, sb, check_exact=False, rtol=1e-9)
    other_a = a[a["asset"] == "GOLD"].set_index(key)["f_mkt_std21"]
    other_b = b[b["asset"] == "GOLD"].set_index(key).loc[other_a.index, "f_mkt_std21"]
    assert not np.allclose(other_a.fillna(0), other_b.fillna(0))   # os outros ativos enxergam a mudança


def test_ablation_effect_is_zero_for_identical_models():
    import ablation
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2020-01-01", periods=300)
    p = pd.DataFrame({"date": np.tile(d, 3), "asset": np.repeat(["A", "B", "C"], 300), "fold": np.tile(d.year - 2020, 3),
                      "target": "y_vol_h5", "y_true": rng.normal(size=900)})
    p["y_pred"] = p["y_true"] + rng.normal(size=900)
    r = ablation.effect(p, p.copy(), ["A"])["y_vol_h5"]
    assert r["dif_ic_outros"] == 0 and r["ic_proprio_visto"] == r["ic_proprio_nao_visto"]


def test_edge_report_detects_same_targets():
    import edge_report as er
    d = pd.bdate_range("2020-01-01", periods=50)
    p = pd.DataFrame({"date": d, "asset": "A", "target": "y_vol_h5", "y_true": np.arange(50.0), "y_pred": 1.0})
    assert er.same_targets({"x": p, "y": p.iloc[10:]})
    assert not er.same_targets({"x": p, "y": p.assign(y_true=p["y_true"] * 2)})
