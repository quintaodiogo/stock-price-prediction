"""Relatório para LER os resultados das pendências do professor (seções 12 a 16 de docs/modelo.md).

Junta numa página HTML as tabelas que os scripts gravaram, com um gráfico por pergunta, uma explicação de
como ler cada tabela e uma frase automática do que aconteceu, seguindo regras fixas (abaixo). A frase é um
ponto de partida: confira sempre com a tabela ao lado.

Regras usadas nas frases (iguais para todas as perguntas):
  - "melhorou de forma consistente": diferença média > 0 E melhor em pelo menos 70% dos anos de teste;
  - "piorou de forma consistente":   diferença média < 0 E melhor em no máximo 30% dos anos;
  - fora disso: "empate" (a diferença muda de sinal de um ano para outro e não dá para afirmar nada);
  - antes de tudo: diferença média menor que 0,005 de IC = "praticamente igual", mesmo que o sinal se repita
    (o IC dos modelos varia cerca de 0,03 a 0,08 entre anos; 0,005 não muda nenhuma conclusão);
  - na ablação, um ativo só "tem impacto" se a mudança passar do ruído (mudar só a semente aleatória) E de 0,005.

Uso (depois de rodar os comandos das seções 12 a 16):
  python src/models/report.py --universe todos_sem_alinhar --cut todos_sem_alinhar_desde2008
-> reports/results/todos_sem_alinhar/relatorio_pendencias.html  (abra no navegador)
"""
import argparse
import html
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

LABEL = {"vol_passada": "régua (vol. passada)", "media_historica": "média histórica", "lgbm_ativo": "por ativo",
         "lgbm_classe": "por classe", "lgbm_geral": "geral"}


# ------------------------------------------------------------------ utilidades
def num(x, d=3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def frac(s: str) -> tuple[int, int]:
    """'13 de 17' -> (13, 17)."""
    try:
        k, n = str(s).split(" de ")
        return int(k), int(n)
    except ValueError:
        return 0, 0


MIN_IC = 0.005   # diferença de IC abaixo disso é desprezível na prática


def verdict(mean: float, k: int, n: int) -> str:
    if n == 0 or pd.isna(mean):
        return "sem dados"
    if abs(mean) < MIN_IC:
        return "ficou praticamente igual (diferença menor que 0,005 de IC)"
    if mean > 0 and k >= 0.7 * n:
        return "melhorou de forma consistente"
    if mean < 0 and k <= 0.3 * n:
        return "piorou de forma consistente"
    return "empate (a diferença muda de sinal entre os anos)"


def read(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def table(df: pd.DataFrame, cols: dict[str, str], digits: int = 3) -> str:
    """HTML de uma tabela com as colunas renomeadas (cols = {coluna: título})."""
    head = "".join(f"<th>{html.escape(t)}</th>" for t in cols.values())
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"<td class='n'>{num(v, digits)}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def svg(fig) -> str:
    import matplotlib.pyplot as plt  # noqa: E402
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return "<figure>" + buf.getvalue().split("?>", 1)[-1] + "</figure>"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.grid": True, "grid.color": "#ddd", "grid.linewidth": 0.5})
    return plt


def missing(cmd: str) -> str:
    return f"<p class='falta'>Ainda sem resultado. Rode: <code>{html.escape(cmd)}</code></p>"


# ------------------------------------------------------------------ seção 12: painel
def sec_panel(full: str, cut: str) -> str:
    out = ["<h2>12. Histórico completo ou cortado</h2>",
           "<p><b>Pergunta.</b> Tirar do treino os dados antigos (antes do corte) melhora a previsão? O teste usa os "
           "mesmos ativos e os mesmos dias de teste nos dois painéis; só o treino muda.</p>"]
    stem = f"vantagem_{full}_x_{cut}"
    res = common.ROOT / "reports" / "results"
    summ, diff, years = read(res / f"{stem}.csv"), read(res / f"{stem}_diferenca.csv"), read(res / f"{stem}_por_ano.csv")
    if summ is None or diff is None:
        return "".join(out) + missing(f"python src/models/edge_report.py --panels {full} {cut}")
    s = summ[summ["target"] == "y_vol_h5"].copy()
    s["modelo"] = s["model"].map(LABEL).fillna(s["model"])
    s["painel_"] = s["painel"].map({full: "completo", cut: "cortado"})
    out.append("<p><b>Como ler.</b> <i>IC</i> = quanto a ordem das previsões acerta a ordem real (0 = nada, 1 = "
               "perfeito). <i>Vantagem</i> = IC do modelo menos IC da régua, ano a ano, depois a média. "
               "<i>Vence a régua</i> = em quantos anos o modelo ganhou dela.</p>")
    out.append(table(s, {"painel_": "Painel", "modelo": "Modelo", "ic": "IC", "ic_std": "Desvio entre anos",
                         "rmse": "RMSE", "vantagem": "Vantagem sobre a régua", "vence_regua": "Vence a régua"}))
    d = diff[diff["target"] == "y_vol_h5"].copy()
    d["modelo"] = d["model"].map(LABEL).fillna(d["model"])
    out.append("<p><b>Diferença cortado − completo</b>, mesmo modelo, ano a ano (positivo = cortar ajudou):</p>")
    out.append(table(d, {"modelo": "Modelo", "dif_ic": "Diferença de IC", "dif_std": "Desvio", "anos_melhor": "Anos em que o cortado foi melhor"}))
    g = d[d["model"] == "lgbm_geral"]
    if len(g):
        k, n = frac(g["anos_melhor"].iloc[0])
        v = verdict(g["dif_ic"].iloc[0], k, n)
        out.append(f"<p class='leitura'><b>O que aconteceu (modelo geral):</b> cortando o histórico, o IC {v}: diferença "
                   f"média {num(g['dif_ic'].iloc[0])}, melhor em {k} de {n} anos. ")
        if v.startswith("melhorou"):
            out.append("O histórico antigo atrapalhava: pela regra da seção 12, o painel oficial passa a ser o cortado.</p>")
        elif v.startswith("piorou"):
            out.append("O histórico antigo ajuda: fica o painel completo.</p>")
        else:
            out.append("O histórico antigo não muda o resultado: fica o painel completo (menos escolhas arbitrárias). A "
                       "queda de vantagem da seção 11.1 deve vir dos anos de teste; confira a versão com <code>--from-year 2012</code>.</p>")
    if years is not None:
        plt = plt_setup()
        y = years[(years["target"] == "y_vol_h5") & years["model"].isin(["lgbm_geral", "vol_passada"])]
        fig, ax = plt.subplots(figsize=(7, 2.8))
        for (m, p), gg in y.groupby(["model", "painel"]):
            ax.plot(gg["ano"], gg["ic"], marker="o", ms=3, lw=1.2, ls="-" if p == full else "--",
                    color="#1f4e79" if m == "lgbm_geral" else "#999",
                    label=f"{LABEL[m]}, {'completo' if p == full else 'cortado'}")
        ax.set_xlabel("ano de teste"); ax.set_ylabel("IC"); ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.set_title("IC por ano, volatilidade (y_vol_h5)", loc="left", fontsize=9)
        out.append(svg(fig))
        out.append("<p class='nota'>Linhas cheia e tracejada do geral quase sobrepostas = cortar não mudou nada. Anos com IC "
                   "mais baixo para todos (inclusive a régua) são anos difíceis, não defeito do modelo.</p>")
    return "".join(out)


# ------------------------------------------------------------------ seção 13: ablação
def sec_ablation(universe: str) -> str:
    out = ["<h2>13. Impacto de cada ativo</h2>",
           "<p><b>Pergunta.</b> Se um ativo sai do treino do modelo geral, quanto os outros pioram? "
           "<i>Diferença nos outros</i> = IC do modelo sem o ativo menos IC do completo, só nas linhas dos outros ativos. "
           "Negativo = o ativo ajudava. <i>Ruído</i> = quanto o IC muda só trocando a semente aleatória; "
           "diferença menor que o ruído não é efeito do ativo.</p>"]
    t = read(common.RESULTS_DIR / "ablacao_ativo.csv")
    if t is None:
        return "".join(out) + missing(f"python src/models/ablation.py --universe {universe}")
    t = t[t["target"] == "y_vol_h5"].copy()
    t["perda_no_proprio"] = t["ic_proprio_nao_visto"] - t["ic_proprio_visto"]
    # conta como impacto só o que passa do ruído E da diferença prática mínima (MIN_IC), como nas outras seções
    above = t[(t["acima_do_ruido"] == True) & (t["dif_ic_outros"].abs() >= MIN_IC)]  # noqa: E712
    helps, hurts = above[above["dif_ic_outros"] < 0], above[above["dif_ic_outros"] > 0]
    maxabs = t["dif_ic_outros"].abs().max()
    txt = f"<p class='leitura'><b>O que aconteceu:</b> a maior mudança no IC dos outros ativos foi de {num(maxabs, 4)}. "
    if len(above) == 0:
        txt += ("Nenhum ativo sozinho muda o modelo geral de forma relevante (todas as mudanças abaixo de 0,005 de IC): "
                "o ganho do modelo geral vem do conjunto, não de um ativo específico. ")
        n_noise = int((t["acima_do_ruido"] == True).sum())  # noqa: E712
        if n_noise:
            txt += (f"{n_noise} ativos passaram do ruído medido, mas por diferenças desprezíveis; o ruído, medido com "
                    "poucas sementes, tende a ser subestimado. ")
    else:
        txt += (f"{len(helps)} ativos ajudavam os outros de forma relevante"
                + (f" ({', '.join(helps['ativo'])})" if len(helps) else "")
                + f" e {len(hurts)} atrapalhavam" + (f" ({', '.join(hurts['ativo'])})" if len(hurts) else "") + ". ")
    big = t.reindex(t["dif_ic_outros"].abs().sort_values(ascending=False).index).head(3)
    txt += "Maiores mudanças: " + "; ".join(
        f"{r.ativo} {num(r.dif_ic_outros, 4)} (ruído {num(r.ruido, 4)})" for r in big.itertuples()) + ". "
    worst = t.sort_values("perda_no_proprio").head(3)
    txt += ("No próprio ativo, nunca visto no treino, as maiores perdas de IC foram: " + "; ".join(
        f"{r.ativo} {num(r.perda_no_proprio)}" for r in worst.itertuples()) + ".</p>")
    out.append(txt)
    plt = plt_setup()
    s = t.sort_values("dif_ic_outros")
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.18 * len(s))))
    ax.barh(s["ativo"], s["dif_ic_outros"], color=["#1f4e79" if a else "#bbb" for a in s["acima_do_ruido"]])
    ax.errorbar([0] * len(s), s["ativo"], xerr=s["ruido"], fmt="none", ecolor="#c00", lw=0.8, capsize=2)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("diferença de IC nos outros ativos ao tirar o ativo (negativo = ele ajudava)")
    ax.set_title("Ablação: barra azul = passou do ruído; traço vermelho = ruído em torno de zero (relevante só acima de 0,005)", loc="left", fontsize=9)
    out.append(svg(fig))
    out.append(table(t.sort_values("dif_ic_outros"),
                     {"ativo": "Ativo", "dif_ic_outros": "Diferença nos outros", "ruido": "Ruído",
                      "anos_pior": "Anos em que os outros pioraram", "ic_proprio_visto": "IC no próprio (visto)",
                      "ic_proprio_nao_visto": "IC no próprio (nunca visto)", "anos_no_treino": "Anos no treino"}, 4))
    c = read(common.RESULTS_DIR / "ablacao_classe.csv")
    if c is not None:
        c = c[c["target"] == "y_vol_h5"]
        out.append("<p><b>Tirando uma classe inteira:</b></p>")
        out.append(table(c, {"classe": "Classe", "dif_ic_outros": "Diferença nos outros", "ruido": "Ruído",
                             "anos_pior": "Anos em que os outros pioraram", "ic_proprio_visto": "IC na classe (vista)",
                             "ic_proprio_nao_visto": "IC na classe (nunca vista)"}, 4))
    return "".join(out)


# ------------------------------------------------------------------ seção 14: importância
def sec_importance(universe: str) -> str:
    out = ["<h2>14. Quais features o modelo geral mais usa</h2>",
           "<p><b>Como ler.</b> Importância = % da melhora total do modelo que veio de divisões feitas com cada feature, "
           "média entre os anos de teste. Mostra o que o modelo <i>usa</i>, não prova causa. Fórmulas em docs/features.md.</p>"]
    t = read(common.RESULTS_DIR / "importancia_geral_y_vol_h5.csv", index_col=0)
    if t is None:
        return "".join(out) + missing(f"python src/models/importance.py --universe {universe}")
    groups = t.groupby("grupo")["importancia_%"].sum().sort_values(ascending=False)
    top = t.head(3)
    out.append("<p class='leitura'><b>O que aconteceu:</b> as 3 mais usadas são " + "; ".join(
        f"{r['nome']} ({num(r['importancia_%'], 1)}%)" for _, r in top.iterrows()) + f". Por grupo, {groups.index[0]} soma "
        f"{num(groups.iloc[0], 1)}% do total.</p>")
    plt = plt_setup()
    s = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(s) + 0.6))
    ax.barh(s["nome"], s["importancia_%"], xerr=s["desvio_%"], color="#1f4e79", ecolor="#888", capsize=2)
    ax.set_xlabel("% da importância total (traço = desvio entre os anos)")
    out.append(svg(fig))
    out.append(table(t.reset_index().rename(columns={"index": "codigo"}),
                     {"codigo": "Código", "nome": "Nome técnico", "grupo": "Grupo", "importancia_%": "Importância (%)", "desvio_%": "Desvio (%)"}, 1))
    return "".join(out)


# ------------------------------------------------------------------ seção 15: mercado
def sec_market(universe: str) -> str:
    out = ["<h2>15. A variabilidade dos outros ativos agrega valor?</h2>",
           "<p><b>Pergunta.</b> Dar ao modelo a agitação média do mercado e da classe (do pregão anterior) melhora a "
           "previsão? <i>Ganho</i> = IC com as features de mercado menos IC sem, mesmas linhas, média dos anos.</p>"]
    t = read(common.RESULTS_DIR / "mercado_comparacao.csv")
    if t is None:
        return "".join(out) + missing(f"python src/models/market.py --universe {universe}")
    t = t[t["target"] == "y_vol_h5"].copy()
    t["modelo_"] = t["modelo"].map(LABEL).fillna(t["modelo"])
    out.append(table(t, {"modelo_": "Modelo", "ic_sem": "IC sem mercado", "ic_com": "IC com mercado",
                         "ganho": "Ganho", "ganho_std": "Desvio", "anos_melhor": "Anos melhor com mercado"}, 4))
    lines = []
    for r in t.itertuples():
        k, n = frac(r.anos_melhor)
        lines.append(f"{LABEL.get(r.modelo, r.modelo)}: {verdict(r.ganho, k, n)} (ganho {num(r.ganho, 4)}, {k} de {n} anos)")
    out.append("<p class='leitura'><b>O que aconteceu:</b> " + "; ".join(lines) + ".</p>")
    out.append("<p class='nota'>Leitura combinada: se o <i>por ativo</i> melhora e o <i>geral</i> não, a informação dos outros "
               "ativos existe e o modelo geral já a aproveita ao treinar com todos. Se nenhum melhora, a agitação dos outros "
               "ativos não acrescenta nada à do próprio ativo. Se piora, as features novas só acrescentaram ruído.</p>")
    imp = read(common.RESULTS_DIR / "importancia_geral_mercado_y_vol_h5.csv", index_col=0)
    if imp is not None:
        m = imp[imp["grupo"] == "mercado"]["importancia_%"].sum()
        out.append(f"<p>No modelo geral com mercado, as 4 features de mercado somam {num(m, 1)}% da importância.</p>")
    return "".join(out)


# ------------------------------------------------------------------ seção 16: contorno
def sec_boundary(universe: str) -> str:
    out = ["<h2>16. Condições de contorno</h2>",
           "<p>Em que situações o resultado vale. Cada tabela mostra o IC de cada modelo dentro de um recorte, nas mesmas "
           "linhas para todos. <i>Geral − régua</i> &gt; 0 = o modelo geral acrescenta algo à régua naquele recorte.</p>"]
    folder = common.RESULTS_DIR / "contorno"
    if not folder.exists():
        return "".join(out) + missing(f"python src/models/boundary.py --universe {universe}")
    plt = plt_setup()
    for name, col, title in [("regime", "regime", "Por regime de mercado"), ("classe", "classe", "Por classe"),
                             ("historico_inicio", "historico", "Por anos de histórico do ativo (inclui os primeiros anos)"),
                             ("ano", "ano", "Por ano de teste")]:
        t = read(folder / f"contorno_{name}.csv")
        if t is None:
            continue
        t = t[t["target"] == "y_vol_h5"].copy()
        t["geral_menos_regua"] = t["lgbm_geral"] - t["vol_passada"]
        t[col] = t[col].astype(str)
        cols = {col: title.split(" ", 1)[1].capitalize()} | {m: LABEL[m] for m in LABEL if m in t and m != "media_historica"}
        cols |= {"geral_menos_regua": "Geral − régua", "anos": "Anos", "linhas": "Linhas"}
        out.append(f"<h3>{title}</h3>")
        best, worst = t.loc[t["lgbm_geral"].idxmax()], t.loc[t["lgbm_geral"].idxmin()]
        neg = t[t["geral_menos_regua"] <= 0][col].tolist()
        out.append(f"<p class='leitura'>Geral melhor em <b>{best[col]}</b> (IC {num(best['lgbm_geral'])}) e pior em "
                   f"<b>{worst[col]}</b> (IC {num(worst['lgbm_geral'])}). "
                   + (f"Não ganhou da régua em: {', '.join(map(str, neg))}." if neg else "Ganhou da régua em todos os recortes.")
                   + "</p>")
        if name == "ano":
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.plot(t["ano"].astype(int), t["lgbm_geral"], marker="o", ms=3, color="#1f4e79", label="geral")
            ax.plot(t["ano"].astype(int), t["vol_passada"], marker="o", ms=3, color="#999", label="régua")
            ax.set_ylabel("IC"); ax.legend(frameon=False, fontsize=8)
            out.append(svg(fig))
        out.append(table(t, {k: v for k, v in cols.items() if k in t}))
    cov = read(folder / "cobertura_historico.csv")
    if cov is not None:
        cov = cov[cov["target"] == "y_vol_h5"]
        out.append("<h3>Cobertura: % das linhas de teste que cada modelo consegue prever</h3>")
        out.append(table(cov, {"historico": "Histórico"} | {m: LABEL[m] for m in LABEL if m in cov} | {"linhas_teste": "Linhas"}, 1))
        first = cov[cov["historico"].str.startswith("1)")]
        if len(first) and "lgbm_ativo" in first:
            out.append(f"<p class='leitura'>Nos primeiros 2 anos de um ativo, o modelo por ativo só consegue prever "
                       f"{num(first['lgbm_ativo'].iloc[0], 1)}% das linhas; o geral, {num(first['lgbm_geral'].iloc[0], 1)}%.</p>")
    edges = read(folder / "bordas.csv")
    if edges is not None:
        out.append("<h3>Bordas dos dados, por ativo</h3>")
        cols = {"ativo": "Ativo", "classe": "Classe", "inicio_painel": "Início no painel", "fim_painel": "Fim", "linhas": "Linhas",
                "aquecimento_pregoes": "Aquecimento (pregões)", "retornos_descartados": "Retornos descartados",
                "maior_buraco_dias": "Maior buraco (dias)", "sem_volume_%": "Sem volume (%)", "sem_amplitude_%": "Sem amplitude (%)"}
        out.append(table(edges, {k: v for k, v in cols.items() if k in edges}, 1))
    return "".join(out)


CSS = """body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111;background:#fff;line-height:1.45}
h1{font-size:20px}h2{font-size:17px;border-bottom:1px solid #444;padding-bottom:2px;margin-top:32px}h3{font-size:14px;margin-top:20px}
table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;margin:8px 0;display:block;overflow-x:auto}
th,td{padding:3px 8px;border-bottom:1px solid #ccc;text-align:left;white-space:nowrap}th{border-bottom:1px solid #000;border-top:1px solid #000}
td.n{text-align:right;font-variant-numeric:tabular-nums}.leitura{background:#f3f3f3;padding:6px 10px;border-left:3px solid #1f4e79}
.nota{font-size:13px;color:#444}.falta{color:#a00}code{font-size:12px}figure{margin:8px 0}figure svg{max-width:100%;height:auto}"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="todos_sem_alinhar")
    ap.add_argument("--cut", default="todos_sem_alinhar_desde2008", help="painel cortado (seção 12)")
    args = ap.parse_args(argv)
    common.use_universe(args.universe)
    body = [f"<h1>Resultados das pendências do professor — painel {html.escape(args.universe)}</h1>",
            "<p class='nota'>Alvo: volatilidade dos próximos 5 pregões (y_vol_h5), onde há sinal. As frases nas caixas cinzas seguem "
            "regras fixas (topo de src/models/report.py): consistente = mesmo sinal em pelo menos 70% dos anos; diferença menor que 0,005 de IC = praticamente igual. Confira "
            "sempre com a tabela.</p>",
            sec_panel(args.universe, args.cut), sec_ablation(args.universe), sec_importance(args.universe),
            sec_market(args.universe), sec_boundary(args.universe)]
    page = f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" \
           f"<title>Resultados das pendências</title><style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    out = common.RESULTS_DIR / "relatorio_pendencias.html"
    out.write_text(page, encoding="utf-8")
    print(f"-> {out.relative_to(common.ROOT)}  (abra no navegador)")


if __name__ == "__main__":
    main()
