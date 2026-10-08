"""Garante que nenhuma feature usa dado futuro e que os folds respeitam o embargo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import features  # noqa: E402

FEATURES = lambda df: [c for c in df if c.startswith("f_")]  # noqa: E731


def synthetic(name: str, n: int = 900, seed: int = 0, level: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    if level:
        close = np.abs(15 + np.cumsum(rng.normal(0, 0.5, n))) + 1
    else:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"date": dates, "Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": rng.uniform(1e6, 2e6, n)})


@pytest.fixture
def raw():
    return {"SP500": synthetic("SP500", seed=1), "VIX": synthetic("VIX", seed=2, level=True),
            "BTC": synthetic("BTC", seed=3)}


def test_features_do_not_change_when_future_is_removed(raw):
    """Se cortarmos o futuro, as features de datas passadas têm de ser idênticas."""
    full = features.build_panel(raw)
    cut = pd.Timestamp("2017-06-30")
    trunc = features.build_panel({k: v[v["date"] <= cut] for k, v in raw.items()})
    a = full[full["date"] <= cut].set_index(["date", "asset"])
    b = trunc.set_index(["date", "asset"])
    common = b.index.intersection(a.index)
    assert len(common) > 100
    cols = FEATURES(full) + ["scale"]
    pd.testing.assert_frame_equal(a.loc[common, cols], b.loc[common, cols], check_exact=False, rtol=1e-9)


def test_features_do_not_change_when_future_is_corrupted(raw):
    """Trocar os preços futuros por lixo não pode alterar nenhuma feature passada."""
    cut = pd.Timestamp("2017-06-30")
    base = features.build_panel(raw)
    broken = {k: v.copy() for k, v in raw.items()}
    for v in broken.values():
        v.loc[v["date"] > cut, ["Open", "High", "Low", "Close", "Volume"]] = 12345.0
    other = features.build_panel(broken)
    a = base[base["date"] <= cut].set_index(["date", "asset"])
    b = other[other["date"] <= cut].set_index(["date", "asset"])
    cols = FEATURES(base) + ["scale"]
    pd.testing.assert_frame_equal(a[cols], b.loc[a.index, cols], check_exact=False, rtol=1e-9)


def test_targets_look_forward(raw):
    """y_ret_h1 deve ser o retorno do dia seguinte dividido pela escala de hoje."""
    panel = features.build_panel(raw)
    df = raw["SP500"].sort_values("date").reset_index(drop=True)
    x = np.log(df["Close"] / df["Close"].shift())
    row = panel[(panel["asset"] == "SP500")].iloc[300]
    i = df.index[df["date"] == row["date"]][0]
    assert row["y_ret_h1"] == pytest.approx(x[i + 1] / row["scale"])
    assert row["y_dir_h1"] == float(x[i + 1] > 0)


def test_nonpositive_price_and_gap_give_no_return():
    df = synthetic("WTI", n=60)
    df.loc[30, "Close"] = -5.0
    x = features.base_series(df, "WTI")
    assert np.isnan(x[30]) and np.isnan(x[31])
    df2 = synthetic("USDBRL", n=60)
    df2.loc[30:, "date"] = df2.loc[30:, "date"] + pd.Timedelta(days=400)
    assert np.isnan(features.base_series(df2, "USDBRL")[30])


def test_level_assets_use_difference():
    df = synthetic("VIX", n=30, level=True)
    x = features.base_series(df, "VIX")
    assert x[5] == pytest.approx(df["Close"][5] - df["Close"][4])


def test_folds_respect_embargo():
    folds = features.make_folds(pd.Timestamp("2026-10-02"))
    assert folds[0]["test_start"] == "2010-01-01" and folds[-1]["test_end"] == "2026-10-02"
    for f in folds:
        gap = pd.Timestamp(f["test_start"]) - pd.Timestamp(f["train_end"])
        assert gap.days > features.EMBARGO_DAYS >= 7
        assert pd.Timestamp(f["train_end"]) < pd.Timestamp(f["embargo_end"]) < pd.Timestamp(f["test_start"])
    for a, b in zip(folds, folds[1:]):
        assert pd.Timestamp(a["test_end"]) < pd.Timestamp(b["test_start"])
