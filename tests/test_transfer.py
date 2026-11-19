"""Testes da Pessoa C: irmãos fora do treino, transferência sem vazamento e métricas do evaluate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "src/models", "src/transfer"):
    sys.path.insert(0, str(ROOT / sub))
import common  # noqa: E402
import evaluate  # noqa: E402
import features  # noqa: E402
import run_transfer  # noqa: E402
from siblings import excluded_for  # noqa: E402
from test_no_leakage import synthetic  # noqa: E402


def test_siblings_removed():
    assert {"SP500", "NASDAQ", "DOWJONES"} <= set(excluded_for("SP500"))
    assert {"IBOVESPA", "PETR4", "VALE3"} <= set(excluded_for("IBOVESPA"))
    assert "IBOVESPA" in excluded_for("PETR4") and "VALE3" not in excluded_for("PETR4")
    assert excluded_for("VIX") == ["VIX"]


def _panel():
    raw = {"SP500": synthetic("SP500", seed=1), "NASDAQ": synthetic("NASDAQ", seed=4),
           "VIX": synthetic("VIX", seed=2, level=True), "GOLD": synthetic("GOLD", seed=3)}
    return features.build_panel(raw)


def test_unseen_never_trains_on_target_asset(monkeypatch):
    """No cenário ativo não visto, nem o ativo nem os irmãos podem estar no treino."""
    panel = _panel()
    folds = features.make_folds(panel["date"].max(), first_year=2016)[-1:]
    seen = []
    real = run_transfer.tree.fit_predict

    def spy(train, test, target, feats, min_rows=500):
        seen.append((set(test["asset"]), set(train["asset"])))
        return real(train, test, target, feats, min_rows)

    monkeypatch.setattr(run_transfer.tree, "fit_predict", spy)
    out, _ = run_transfer.run(panel, folds, ["y_vol_h5"], ["unseen"])
    assert seen
    for test_assets, train_assets in seen:
        (a,) = test_assets
        assert not train_assets & set(excluded_for(a))
    preds = pd.concat(out["unseen"])
    assert list(preds.columns) == common.PRED_COLUMNS


def test_fine_tune_follows_contract_and_dates():
    panel = _panel()
    folds = features.make_folds(panel["date"].max(), first_year=2016)[-1:]
    out, log = run_transfer.run(panel, folds, ["y_vol_h5"], ["zero_shot", "fine_tune"])
    f = folds[0]
    for s in ("zero_shot", "fine_tune"):
        p = pd.concat(out[s])
        assert list(p.columns) == common.PRED_COLUMNS and np.isfinite(p["y_pred"]).all()
        assert p["date"].min() >= pd.Timestamp(f["test_start"]) and p["date"].max() <= pd.Timestamp(f["test_end"])
    assert all(r["arvores_extras"] in run_transfer.FT_GRID for r in log)


def test_ic_and_final_table():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    good = pd.DataFrame({"y_true": y, "y_pred": y + rng.normal(size=500)})
    const = pd.DataFrame({"y_true": y, "y_pred": 0.0})
    assert evaluate.ic(good) > 0.4
    assert np.isnan(evaluate.ic(const))
    ic = pd.DataFrame({"lgbm_ativo": [0.2, 0.3], "transfer_fine_tune": [0.25, 0.28]}, index=["A", "B"])
    t = evaluate.final_table(ic, pd.DataFrame(index=ic.index))
    assert np.allclose(t["ganho_fine_tune_vs_ativo"], [0.05, -0.02])
