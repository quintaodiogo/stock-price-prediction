"""Ativos "irmãos": saem do treino junto com o ativo alvo no cenário "ativo não visto" (README, seção 5).

Sem isso, o S&P 500 "não visto" seria previsto por um modelo que viu Nasdaq e Dow Jones, que se movem quase juntos.
"""
GROUPS = [
    {"SP500", "NASDAQ", "DOWJONES"},
    {"FTSE100", "DAX"},
    {"GOLD", "SILVER"},
    {"EURUSD", "GBPUSD"},
    {"ITUB4", "BBDC4", "BBAS3"},          # bancos
    {"WIN", "IBOVESPA"},
    {"WDO", "USDBRL"},
    {"BTC", "ETH"},
]
# O Ibovespa é composto pelas ações da B3: ao esconder o índice, esconde-se também as ações;
# ao esconder uma ação, esconde-se o índice (mas não as outras ações).
IBOV_COMPONENTS = {"PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3", "SUZB3", "RENT3"}


def excluded_for(asset: str) -> list[str]:
    """Ativo + irmãos próximos, a retirar do treino quando `asset` é o ativo não visto."""
    out = {asset}
    for g in GROUPS:
        if asset in g:
            out |= g
    if asset in ("IBOVESPA", "WIN"):
        out |= IBOV_COMPONENTS | {"IBOVESPA", "WIN"}
    if asset in IBOV_COMPONENTS:
        out |= {"IBOVESPA", "WIN"}
    return sorted(out)
