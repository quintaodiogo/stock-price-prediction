"""Catálogo dos ativos (20 globais, ações da B3 e minis): nome do arquivo -> ticker do Yahoo Finance e classe."""

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
    # Ações da B3 (Yahoo usa o sufixo .SA)
    "PETR4": ("PETR4.SA", "acao_br"),
    "VALE3": ("VALE3.SA", "acao_br"),
    "ITUB4": ("ITUB4.SA", "acao_br"),
    "BBDC4": ("BBDC4.SA", "acao_br"),
    "BBAS3": ("BBAS3.SA", "acao_br"),
    "ABEV3": ("ABEV3.SA", "acao_br"),
    "WEGE3": ("WEGE3.SA", "acao_br"),
    "B3SA3": ("B3SA3.SA", "acao_br"),
    "SUZB3": ("SUZB3.SA", "acao_br"),
    "RENT3": ("RENT3.SA", "acao_br"),
    # Minis da B3: não vêm do Yahoo; gerados por src/minis_to_daily.py a partir de data/intraday/
    "WIN": (None, "mini"),  # minicontrato de Ibovespa
    "WDO": (None, "mini"),  # minicontrato de dólar
}

# Ativos que são níveis (taxa ou índice de volatilidade): usar diferença, não retorno log.
LEVEL_ASSETS = {"US10Y", "VIX"}

# Sem volume real no Yahoo (câmbio, VIX, juros): volume zero é ausência, não zero.
NO_VOLUME_ASSETS = {n for n, (_, c) in ASSETS.items() if c == "cambio"} | LEVEL_ASSETS

COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume"]

# Conjuntos de ativos para o painel (todos alinhados na interseção de datas de cada conjunto).
# O alinhamento é limitado pelo ativo mais curto: os minis começam em 2020, a cripto (ETH) em 2017.
UNIVERSES = {
    "todos": list(ASSETS),
    "sem_minis": [n for n, (_, c) in ASSETS.items() if c != "mini"],
    "sem_minis_cripto": [n for n, (_, c) in ASSETS.items() if c not in ("mini", "cripto")],
}
