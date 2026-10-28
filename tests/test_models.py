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
