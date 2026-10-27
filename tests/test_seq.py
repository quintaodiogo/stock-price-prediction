"""Testes da rede sequencial: janelas sem futuro e sem misturar ativos."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
import common  # noqa: E402
import features  # noqa: E402
import seq  # noqa: E402
from test_models import _raw  # noqa: E402

L = 10


@pytest.fixture(scope="module")
def win():
    panel = features.build_panel(_raw())
    return seq.Windows(panel, common.feature_cols(panel), L)


def test_window_uses_only_same_asset_and_past(win):
    p = win.panel
    rows = np.where(win.valid)[0]
    idx = rows[:, None] - win.offsets                      # linhas que entram em cada janela
    assert (p["asset"].to_numpy()[idx] == p["asset"].to_numpy()[rows][:, None]).all()
    d = p["date"].to_numpy()
    assert (d[idx[:, -1]] == d[rows]).all()                # última linha da janela = data da previsão
    assert (np.diff(d[idx].astype("int64"), axis=1) > 0).all()   # janela em ordem, sem futuro


def test_batch_shape_and_content(win):
    x = win.standardize(np.arange(len(win.panel)))
    rows = np.where(win.valid)[0][:7]
    b = win.batch(x, rows)
    assert tuple(b.shape) == (7, L, len(win.feats))
    assert torch.equal(b[:, -1], x[rows])
    assert torch.isfinite(b).all()


def test_standardize_uses_only_train_rows(win):
    train = np.arange(len(win.panel) // 2)
    a = win.standardize(train)
    raw = win.raw.copy()
    win.raw[len(win.panel) // 2:] *= 1000                  # mexer só fora do treino
    b = win.standardize(train)
    win.raw = raw
    torch.testing.assert_close(a[train], b[train])
