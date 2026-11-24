# Features e alvos: como são calculados

Este documento explica cada coluna que entra no modelo (features, `f_*`) e cada coluna que ele tenta
prever (alvos, `y_*`). O código é `src/features.py` (Pessoa A) e `src/normalization.py`; as features de
mercado ficam em `src/models/market.py`. **Os nomes no código não mudam**: a tabela da seção 2 dá o nome
técnico de cada um para usar no relatório e no artigo.

Regra que vale para tudo: **uma feature do dia t só usa dados até o fechamento do dia t**. Os alvos usam
os dias depois de t. Há testes automáticos para isso (`tests/test_no_leakage.py`, `tests/test_models.py`).

## 1. A série de partida: retorno do dia e escala

Todo o resto sai de duas séries por ativo.

**Retorno do dia, x_t.** Para ativos de preço (índices, ações, câmbio, commodities, cripto, minis), é o
retorno logarítmico do preço de fechamento:

    x_t = ln(Close_t / Close_{t-1})

Exemplo: fechou ontem a 100,00 e hoje a 101,20 → x_t = ln(101,20 / 100,00) = ln(1,012) = 0,01193 (cerca de +1,2%).

Para **US10Y** (taxa de juros) e **VIX** (índice de volatilidade), que são níveis e não preços, x_t é a
diferença simples: x_t = Close_t − Close_{t−1}. Exemplo: o VIX foi de 18,0 para 19,5 → x_t = 1,5.

x_t fica **vazio** quando: o preço não é positivo (o petróleo WTI fechou negativo em 20/04/2020, e o log
de número negativo não existe); ou o intervalo desde o pregão anterior passa de 7 dias corridos (3 na
cripto, que negocia todo dia), porque aí o "retorno do dia" seria, na verdade, de várias semanas.

**Escala, s_t.** O desvio-padrão de x nos últimos 252 pregões (cerca de um ano), contando o dia t,
exigindo pelo menos 63 retornos válidos. É a "agitação normal" do ativo no último ano.

Exemplo: se os retornos diários do último ano variam tipicamente 1% para cima ou para baixo, s_t ≈ 0,010.

**Retorno normalizado, z_t** (normalização `vol_scaled`, a escolhida pela Pessoa A):

    z_t = x_t / s_t

Exemplo: x_t = 0,01193 e s_t = 0,010 → z_t = 1,19. Quer dizer "hoje o ativo andou 1,19 vez o tamanho de um
dia típico do último ano". Assim um dia de +1,2% no ouro (calmo) e um de +1,2% no bitcoin (agitado) viram
números diferentes, cada um na régua do próprio ativo, e o modelo pode juntar todos os ativos.

## 2. As 13 features do painel

| Código | Nome técnico | Fórmula (dados até t) | O que mede |
|---|---|---|---|
| `f_z_lag0` | Retorno normalizado do dia | z_t | o movimento de hoje |
| `f_z_lag1` … `f_z_lag4` | Retorno normalizado defasado (1 a 4 pregões) | z_{t−1}, …, z_{t−4} | os movimentos dos 4 dias anteriores |
| `f_z_std5` | Volatilidade realizada relativa de 5 pregões | desvio-padrão de z_{t−4..t} | agitação da última semana em relação ao último ano |
| `f_z_std21` | Volatilidade realizada relativa de 21 pregões | desvio-padrão de z_{t−20..t} | idem, último mês |
| `f_z_std63` | Volatilidade realizada relativa de 63 pregões | desvio-padrão de z_{t−62..t} | idem, último trimestre |
| `f_z_mean21` | Momentum ajustado ao risco de 21 pregões | média de z_{t−20..t} | tendência do último mês, medida em dias típicos |
| `f_z_mean63` | Momentum ajustado ao risco de 63 pregões | média de z_{t−62..t} | tendência do último trimestre |
| `f_rsi14` | Índice de Força Relativa (IFR) de 14 pregões | A / (A + B), ver 2.4 | se as últimas altas foram maiores que as quedas |
| `f_range` | Amplitude diária relativa | [(High−Low)/Close] ÷ média de 252 pregões | quanto o preço oscilou dentro do dia, comparado ao normal |
| `f_volrel` | Volume relativo de 21 pregões | Volume_t ÷ média do volume de 21 pregões | se negociou mais ou menos que o habitual |

Pregão = dia com negociação. As janelas contam pregões do próprio ativo (na cripto, dias corridos).

**Janela com falhas.** As janelas de 5, 21 e 63 pregões exigem 80% dos valores (4 de 5, 17 de 21, 51 de
63); assim um dia vazio isolado não apaga a feature da janela inteira.

### 2.1 Retornos defasados (`f_z_lag0` a `f_z_lag4`)

São os cinco últimos z, do mais recente (lag0 = hoje) ao mais antigo (lag4 = quatro pregões atrás).
Exemplo, com z de segunda a sexta = 0,3; −1,8; 0,5; 2,1; −0,4 e hoje sexta: lag0 = −0,4, lag1 = 2,1,
lag2 = 0,5, lag3 = −1,8, lag4 = 0,3. Dão ao modelo o "filme" da última semana. Um dia de |z| grande
costuma ser seguido de dias agitados (agrupamento de volatilidade), mesmo que o sinal do próximo dia seja
imprevisível.

### 2.2 Volatilidade realizada relativa (`f_z_std5`, `f_z_std21`, `f_z_std63`)

Desvio-padrão dos z da janela. Como z já está dividido pela escala de um ano, o resultado é
aproximadamente a **razão entre a volatilidade recente e a de um ano**:

    f_z_std21 ≈ (desvio-padrão dos retornos dos últimos 21 pregões) / s_t

Exemplo: no último mês os retornos variaram tipicamente 1,5%; no último ano, 1,0% → f_z_std21 ≈ 1,5 / 1,0
= 1,5 ("o mês está 50% mais agitado que o normal"). Valor 1 = agitação normal; abaixo de 1 = mais calmo.
São as features mais importantes para prever a volatilidade (ver `importance.py`): a volatilidade de
amanhã se parece com a de agora, como descrevem os modelos GARCH (Bollerslev, 1986).

### 2.3 Momentum ajustado ao risco (`f_z_mean21`, `f_z_mean63`)

Média dos z da janela. Exemplo com 21 pregões cuja soma de z dá 6,3 → f_z_mean21 = 6,3 / 21 = 0,30: em
média o ativo subiu 0,3 dia típico por pregão no último mês. Multiplicando por √21 ≈ 4,58 tem-se 1,37,
que é a tendência medida em "desvios" (parecido com uma estatística t): valores acima de 2 em módulo indicam
tendência forte. É o *time-series momentum* da literatura, já ajustado ao risco porque usa z.

### 2.4 Índice de Força Relativa (`f_rsi14`)

Com as variações do fechamento nos últimos 14 pregões, d = Close_t − Close_{t−1}:

    A = média das altas (as quedas entram como 0)
    B = média das quedas, em módulo (as altas entram como 0)
    f_rsi14 = A / (A + B)

Exemplo: em 14 pregões, as altas somam 10,5 e as quedas 3,5 → A = 10,5/14 = 0,75; B = 3,5/14 = 0,25;
f_rsi14 = 0,75 / (0,75 + 0,25) = 0,75.

Diferenças em relação ao IFR que aparece nas plataformas de corretora: (a) a escala aqui vai de **0 a 1**
(lá, 0 a 100; 0,75 aqui = 75 lá); (b) as médias são **simples** (versão de Cutler); a de Wilder, mais comum
nas plataformas, usa média exponencial, então os números diferem um pouco. Como A e B estão na mesma
unidade de preço, o resultado não depende do nível do preço e pode ser comparado entre ativos. Leitura
tradicional: perto de 1 = só subiu (sobrecomprado); perto de 0 = só caiu (sobrevendido); 0,5 = equilíbrio.

### 2.5 Amplitude diária relativa (`f_range`)

    amplitude_t = (High_t − Low_t) / Close_t
    f_range     = amplitude_t / média da amplitude nos últimos 252 pregões (mínimo 63)

Exemplo: máxima 102,0, mínima 99,5, fechamento 101,0 → amplitude = 2,5 / 101,0 = 0,0248 (2,5%). Se a
amplitude média do último ano é 1,6%, f_range = 2,48 / 1,6 = 1,55: o dia oscilou 55% mais que o normal.

A amplitude mede a agitação **dentro** do dia, que o fechamento sozinho não vê (um dia pode subir 3% e
voltar, fechando estável). É a ideia do estimador de volatilidade de Parkinson (1980), que usa máxima e
mínima. Fica vazia quando o dado de máxima e mínima não existe ou é inconsistente (máxima menor que a
mínima), o que acontece em índices antigos.

### 2.6 Volume relativo (`f_volrel`)

    f_volrel = Volume_t / média do volume dos últimos 21 pregões (contando t)

Exemplo: hoje negociou 3,0 milhões; a média do último mês é 2,0 milhões → f_volrel = 1,5. Volume fora do
normal costuma acompanhar notícia e agitação. Fica **vazia** em câmbio, VIX e US10Y, que não têm volume
no Yahoo, e nos trechos antigos de alguns índices: o `download_data.py` grava volume zero como ausente,
porque volume zero nunca é volume real.

Faltantes em `f_range` e `f_volrel` não tiram a linha do painel: o LightGBM aprende sozinho para que lado
mandar os valores vazios. As outras 11 features são obrigatórias (linha sem elas sai do painel).

## 3. Features de mercado (experimento "a variabilidade dos outros ativos agrega valor?")

Só entram com `tree.py --market` (código em `src/models/market.py`). Todas usam os valores dos **outros**
ativos (o próprio fica fora da média) e só até o **pregão anterior** a t, porque os mercados fecham em
horários diferentes: Tóquio fecha antes de Nova York no mesmo dia, e usar o fechamento de Nova York do dia
t para Tóquio seria usar informação que ainda não existia quando Tóquio fechou.

| Código | Nome técnico | Fórmula |
|---|---|---|
| `f_mkt_std5` | Volatilidade relativa do mercado, 5 pregões | média de `f_z_std5` dos outros ativos, no pregão anterior |
| `f_mkt_std21` | Volatilidade relativa do mercado, 21 pregões | média de `f_z_std21` dos outros ativos, no pregão anterior |
| `f_mkt_move` | Movimento absoluto do mercado | média de \|`f_z_lag0`\| dos outros ativos, no pregão anterior |
| `f_cls_std21` | Volatilidade relativa da classe, 21 pregões | média de `f_z_std21` dos outros ativos da mesma classe, no pregão anterior |

Exemplo para o IBOVESPA numa terça: pega a `f_z_std21` de segunda de cada um dos outros 31 ativos (se um
ativo não negociou na segunda, usa o último valor dele, de até 5 datas antes) e tira a média. Se der 1,4, o
mercado como um todo está 40% mais agitado que o normal. Exige pelo menos 5 outros ativos (mercado) ou 1
outro da classe (classe).

## 4. Alvos (o que o modelo prevê)

Todos olham para **depois** de t e são divididos pela escala s_t do dia t, para ficar na régua do ativo.

| Código | Nome técnico | Fórmula |
|---|---|---|
| `y_ret_h1` | Retorno normalizado do próximo pregão | x_{t+1} / s_t |
| `y_ret_h5` | Retorno normalizado acumulado de 5 pregões | (x_{t+1} + … + x_{t+5}) / (s_t · √5) |
| `y_dir_h1` | Direção do próximo pregão | 1 se x_{t+1} > 0; 0 se x_{t+1} < 0; vazio se x_{t+1} = 0 |
| `y_vol_h5` | Volatilidade realizada relativa dos próximos 5 pregões | desvio-padrão de x_{t+1..t+5} / s_t |

Exemplo de `y_vol_h5`: os próximos 5 retornos são +1,0%, −2,0%, +0,5%, −1,5%, +2,0%. Média = 0,0%;
desvios ao quadrado = 1,00; 4,00; 0,25; 2,25; 4,00 (em %²), soma 11,5; dividida por 4 (n − 1) = 2,875;
raiz = 1,70%. Com s_t = 1,0%, y_vol_h5 = 1,70: a próxima semana foi 70% mais agitada que um dia típico do
último ano.

O √5 em `y_ret_h5` deixa o retorno de 5 dias na mesma régua do de 1 dia (a variância de uma soma de 5
retornos independentes é 5 vezes a de um).

As colunas `r_ret_h1` e `r_vol_h5` são os mesmos alvos sem dividir pela escala, usados só para avaliação.

## 5. Bordas da série (onde o painel começa e termina)

- **Início de cada ativo:** a escala precisa de 63 retornos; a `f_z_std63` precisa de 51 valores de z.
  O primeiro dia com tudo pronto é o 114º pregão do arquivo bruto (113 pregões de aquecimento, cerca de 5
  meses). Conta: o 1º pregão não tem retorno; z existe a partir do 64º; mais 50 pregões até ter 51 z.
- **Fim:** os últimos 5 pregões não têm alvo de 5 dias e saem do painel.
- **Buracos e preços inválidos:** o retorno do dia fica vazio (seção 1) e as janelas seguem com 80% dos dados.
- `python src/models/boundary.py` lista, por ativo, essas bordas (início, aquecimento, retornos
  descartados, maior buraco, % sem volume e sem amplitude). Ver docs/modelo.md, seção 16.

## Referências

- BOLLERSLEV, T. Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, v. 31, n. 3, p. 307–327, 1986.
- MOSKOWITZ, T. J.; OOI, Y. H.; PEDERSEN, L. H. Time series momentum. *Journal of Financial Economics*, v. 104, n. 2, p. 228–250, 2012.
- PARKINSON, M. The extreme value method for estimating the variance of the rate of return. *The Journal of Business*, v. 53, n. 1, p. 61–65, 1980.
- WILDER, J. W. *New concepts in technical trading systems*. Greensboro: Trend Research, 1978.
