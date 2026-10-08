"""Catálogo dos 20 ativos: nome do arquivo -> ticker do Yahoo Finance e classe."""

ASSETS = {
    # nome: (ticker, classe)
    "SP500": ("^GSPC", "indice"),
    "NASDAQ": ("^IXIC", "indice"),
    "DOWJONES": ("^DJI", "indice"),
    "FTSE100": ("^FTSE", "indice"),
    "DAX": ("^GDAXI", "indice"),
    "NIKKEI225": ("^N225", "indice"),
    "HANGSENG": ("^HSI", "indice"),
    "IBOVESPA": ("^BVSP", "indice"),
    "GOLD": ("GC=F", "commodity"),
    "SILVER": ("SI=F", "commodity"),
    "WTI": ("CL=F", "commodity"),
    "COPPER": ("HG=F", "commodity"),
    "EURUSD": ("EURUSD=X", "cambio"),
    "USDJPY": ("JPY=X", "cambio"),
    "GBPUSD": ("GBPUSD=X", "cambio"),
    "USDBRL": ("BRL=X", "cambio"),
    "BTC": ("BTC-USD", "cripto"),
    "ETH": ("ETH-USD", "cripto"),
    "US10Y": ("^TNX", "juros"),
    "VIX": ("^VIX", "juros"),
}

# Ativos que são níveis (taxa ou índice de volatilidade): usar diferença, não retorno log.
LEVEL_ASSETS = {"US10Y", "VIX"}

# Sem volume real no Yahoo (câmbio, VIX, juros): volume zero é ausência, não zero.
NO_VOLUME_ASSETS = {n for n, (_, c) in ASSETS.items() if c == "cambio"} | LEVEL_ASSETS

COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume"]
