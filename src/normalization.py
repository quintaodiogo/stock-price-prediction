"""Alinhamento de datas e variantes de normalização, todas usando só dados até t.

Cada normalizador recebe a série base `x` (retorno log, ou diferença em US10Y/VIX) e o `close`
do ativo, e devolve `z`, a série que alimenta as features. `cross_section_z` é a exceção: precisa
de todos os ativos na mesma data, por isso roda sobre a tabela larga (datas x ativos).
Janela móvel padrão: 252 pregões, mínimo 63 observações.
"""
import numpy as np
import pandas as pd

WINDOW = 252
MIN_OBS = 63
EWMA_ALPHA = 0.06        # ~ lambda 0,94 (RiskMetrics)
CLIP = 3.0


def _roll(s: pd.Series):
    return s.rolling(WINDOW, min_periods=MIN_OBS)


def _exp(s: pd.Series):
    return s.expanding(min_periods=MIN_OBS)


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).replace([np.inf, -np.inf], np.nan)


def _logprice(close: pd.Series, level: bool) -> pd.Series:
    return close if level else np.log(close.where(close > 0))


def raw(x, close, level):
    return x


def vol_scaled(x, close, level):
    return _div(x, _roll(x).std())


def vol_scaled_clip(x, close, level):
    return vol_scaled(x, close, level).clip(-CLIP, CLIP)


def ewma_vol(x, close, level):
    return _div(x, np.sqrt((x ** 2).ewm(alpha=EWMA_ALPHA, min_periods=MIN_OBS).mean()))


def zscore_rolling(x, close, level):
    return _div(x - _roll(x).mean(), _roll(x).std())


def zscore_expanding(x, close, level):
    return _div(x - _exp(x).mean(), _exp(x).std())


def minmax_rolling(x, close, level):
    lo, hi = _roll(x).min(), _roll(x).max()
    return _div(x - lo, hi - lo)


def minmax_expanding(x, close, level):
    lo, hi = _exp(x).min(), _exp(x).max()
    return _div(x - lo, hi - lo)


def robust_rolling(x, close, level):
    med = _roll(x).median()
    mad = _roll((x - med).abs()).median() * 1.4826   # MAD com a mediana de hoje; só passado
    return _div(x - med, mad)


def rank_rolling(x, close, level):
    """Percentil de x_t entre as últimas 252 observações (inclui x_t). Resultado em (0, 1]."""
    return _roll(x).apply(lambda a: (a <= a[-1]).mean(), raw=True)


def minmax_price_rolling(x, close, level):
    """Posição do (log) preço no intervalo [mínimo, máximo] dos últimos 252 pregões."""
    p = _logprice(close, level)
    lo, hi = _roll(p).min(), _roll(p).max()
    return _div(p - lo, hi - lo)


def zscore_price_rolling(x, close, level):
    p = _logprice(close, level)
    return _div(p - _roll(p).mean(), _roll(p).std())


def cross_section_z(x_wide: pd.DataFrame) -> pd.DataFrame:
    """z-score entre ativos na mesma data (usa só a própria data; exige >= 5 ativos)."""
    mean, std = x_wide.mean(axis=1), x_wide.std(axis=1)
    z = x_wide.sub(mean, axis=0).div(std.where(std > 0), axis=0)
    return z.where(x_wide.notna().sum(axis=1) >= 5)


NORMALIZERS = {
    "raw": raw,                                  # referência: retorno sem normalizar
    "vol_scaled": vol_scaled,                    # x / desvio móvel (sem subtrair média)
    "vol_scaled_clip": vol_scaled_clip,          # idem, limitado a +-3
    "ewma_vol": ewma_vol,                        # x / volatilidade EWMA
    "zscore_rolling": zscore_rolling,            # (x - média móvel) / desvio móvel
    "zscore_expanding": zscore_expanding,        # (x - média expansiva) / desvio expansivo
    "minmax_rolling": minmax_rolling,            # min-max do retorno na janela móvel, em [0, 1]
    "minmax_expanding": minmax_expanding,        # min-max do retorno desde o início
    "robust_rolling": robust_rolling,            # (x - mediana) / MAD móvel
    "rank_rolling": rank_rolling,                # percentil na janela móvel
    "minmax_price_rolling": minmax_price_rolling,  # min-max do log preço na janela móvel
    "zscore_price_rolling": zscore_price_rolling,  # z-score do log preço na janela móvel
}
CROSS_SECTION = "cross_section_z"                # normalização entre ativos, tratada no painel
ALL_NORMS = list(NORMALIZERS) + [CROSS_SECTION]


def align_dates(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Mantém só as datas em que TODOS os ativos têm candle (interseção)."""
    common = None
    for df in raw_data.values():
        d = set(df["date"])
        common = d if common is None else common & d
    return {n: df[df["date"].isin(common)].reset_index(drop=True) for n, df in raw_data.items()}
