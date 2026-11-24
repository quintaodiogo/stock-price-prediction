# Modelo geral (Pessoa B: Laura Beatriz dos Santos Silva)

Baselines e LightGBM no walk-forward, e o experimento 1b do README (seção 4): o modelo treinado
em todos os ativos prevê melhor do que o modelo treinado só no ativo?

## 1. Como reproduzir

Requer **Python 3.10 ou mais novo** (o código usa `int | None`). No Mac, o LightGBM precisa do
`libomp` (`brew install libomp`).

```
python src/download_data.py
python src/features.py --universe sem_minis_cripto
python -m pytest tests                     # todos devem passar (1 pulado sem PyTorch)
python src/models/run_1b.py                # baselines + LightGBM nos 3 modos + comparação (~10 min)
```

Saída: `reports/results/comparacao_1b.csv` e um `run_<modelo>.json` por modelo, com o painel e os
parâmetros usados. As previsões (`preds_*.parquet`) e os modelos (`models/*.txt`) não vão para o git;
são regenerados pelos comandos acima.

## 2. Dados usados

| Item | Valor |
|---|---|
| Conjunto de ativos | `sem_minis_cripto`: 28 ativos (índices, commodities, câmbio, juros/VIX, ações da B3) |
| Normalização | `vol_scaled` (padrão): retorno / volatilidade móvel de 252 dias |
| Painel | 81.472 linhas, 2.960 datas comuns, 24/09/2008 a 14/10/2026 |
| Folds | 17 (teste de um ano, 2010 a 2026), embargo de 10 dias entre treino e teste |

Os minis (WIN, WDO) e a cripto ficaram de fora porque, com as datas comuns, encurtam o painel para
2017 ou 2021 e deixam poucos folds.

## 3. O que foi implementado (`src/models/`)

| Arquivo | Conteúdo |
|---|---|
| `common.py` | Carrega painel e folds, separa treino/teste, usa só colunas `f_*` como entrada, salva previsões no formato da seção 7 do README (`date, asset, fold, target, y_true, y_pred`) |
| `baselines.py` | `zero` (retorno 0, direção 0,5), `media_historica` (média do alvo por ativo no treino), `vol_passada` (vol. realizada dos últimos 20 pregões, só para `y_vol_h5`) |
| `tree.py` | LightGBM em três modos: `geral` (todos os ativos juntos), `classe` (um modelo por classe), `ativo` (um modelo por ativo). Opção `--exclude` tira ativos do treino (cenário "ativo não visto" da Pessoa C) |
| `run_1b.py` | Roda o que faltar e compara todos os modelos nas mesmas linhas |
| `edge_report.py` | Vantagem de cada modelo sobre a régua em vários painéis (decisão do painel, seção 12) |
| `ablation.py` | Tira um ativo (ou classe) por vez do treino do geral e mede o efeito (seção 13) |
| `importance.py` | Importância das features no modelo geral (seção 14) |
| `market.py` | Features de mercado e a comparação com e sem elas (seção 15) |
| `boundary.py` | Condições de contorno: bordas dos dados e IC por regime, classe, histórico e ano (seção 16) |
| `report.py` | Junta os resultados das seções 12 a 19 numa página HTML com gráficos, como ler cada tabela e o que aconteceu |
| `simple.py` | Modelos mais simples (regressão linear/logística e uma árvore de decisão) para comparar com o LightGBM (seção 18) |
| `window.py` | Treina o geral só com os últimos N anos antes de cada prova: quanto dado é preciso (seção 19) |
| `tests/test_models.py` | Vol. passada não usa o futuro; previsões no formato e no período certos; alvos nunca entram como feature; validação interna com embargo |

**LightGBM.** Hiperparâmetros fixos (`learning_rate` 0,03, `num_leaves` 15, `min_child_samples` 200,
reduzido em grupos pequenos, `feature_fraction` e `bagging_fraction` 0,8, `lambda_l2` 1, semente 42).
O número de árvores (até 1.000) sai de early stopping numa validação interna: os últimos 20% das datas
do treino de cada fold, com 10 dias de embargo antes dela. O modelo final é retreinado no treino inteiro
com esse número. O ano de teste nunca é usado para escolher nada. Direção usa `binary`; os demais, `l2`.

## 4. Resultados do experimento 1b

Média entre folds; desvio-padrão entre folds em `ic_std`. `vence_ativo_%` = em quantos folds o IC do
modelo foi maior que o do modelo por ativo. Comparação nas mesmas linhas (data, ativo, alvo) para todos
os modelos: **15 folds, 2012 a 2026**. Em 2010 e 2011 o treino por ativo tinha menos de 500 linhas e o
modelo por ativo não foi treinado, então esses anos ficaram de fora da comparação.

| Alvo | Modelo | IC | IC desvio | RMSE | AUC | Vence ativo |
|---|---|---|---|---|---|---|
| `y_vol_h5` | media_historica | -0,020 | 0,061 | 0,569 | | 0% |
| | vol_passada | 0,236 | 0,097 | 0,559 | | 20% |
| | lgbm_ativo | 0,295 | 0,059 | 0,535 | | — |
| | lgbm_classe | 0,339 | 0,058 | 0,521 | | **100%** |
| | **lgbm_geral** | **0,342** | 0,069 | **0,517** | | **87%** |
| `y_dir_h1` | zero | — | — | 0,5000 | 0,500 | — |
| | media_historica | 0,027 | 0,022 | 0,4995 | 0,516 | 47% |
| | lgbm_ativo | 0,031 | 0,026 | 0,5003 | 0,518 | — |
| | lgbm_classe | 0,033 | 0,024 | 0,4995 | 0,519 | 60% |
| | lgbm_geral | 0,028 | 0,026 | 0,4995 | 0,516 | 47% |
| `y_ret_h1` | zero | — | — | 1,0321 | | — |
| | media_historica | 0,007 | 0,021 | 1,0321 | | 27% |
| | lgbm_ativo | 0,016 | 0,026 | 1,0337 | | — |
| | lgbm_classe | 0,016 | 0,025 | 1,0331 | | 40% |
| | lgbm_geral | 0,020 | 0,015 | 1,0322 | | 53% |
| `y_ret_h5` | zero | — | — | 1,0118 | | — |
| | media_historica | 0,018 | 0,044 | 1,0113 | | 40% |
| | lgbm_ativo | 0,021 | 0,035 | 1,0166 | | — |
| | lgbm_classe | 0,011 | 0,047 | 1,0126 | | 40% |
| | lgbm_geral | 0,005 | 0,042 | 1,0107 | | 40% |

## 5. Conclusões

1. **Volatilidade: o modelo geral ganha do modelo por ativo.** IC 0,342 contra 0,295, menor RMSE, e
   vence em 13 dos 15 anos. Também supera com folga o baseline de vol. passada (IC 0,236). É a hipótese
   central do projeto, confirmada neste alvo.
2. **Geral e por classe empatam.** A diferença de IC (0,004) é muito menor que a variação entre folds
   (cerca de 0,06). Os dados não dizem se juntar tudo é melhor do que juntar por classe.
3. **Direção: nenhum modelo acrescenta algo à média histórica.** Todos ficam com AUC entre 0,516 e
   0,519, igual ao baseline que só usa a frequência de alta de cada ativo.
4. **Retorno: sem sinal.** ICs entre 0,005 e 0,02 com desvio do mesmo tamanho; nenhum modelo tem RMSE
   relevantemente menor que prever zero (diferenças de 0,001). Resultado negativo, esperado para
   retorno diário.

**Decisão sugerida para a Pessoa C:** seguir o plano com a volatilidade como alvo principal do
transfer learning; retorno e direção entram no relatório como resultado negativo.

## 6. Ressalvas

- ~~Hiperparâmetros fixos, iguais nos três modos.~~ Resolvido na seção 10: com ajuste por modo e por
  fold, a vantagem do modelo geral se mantém.
- Sem teste estatístico formal da diferença entre modelos; a contagem de folds vencidos é a evidência.
- O IC é calculado sobre todas as linhas do fold (ativos e datas juntos), não por data.
- Um único conjunto de ativos (`sem_minis_cripto`) e uma única normalização (`vol_scaled`).

## 7. Próximos passos

- Pessoa C: cenário "ativo não visto" com `python src/models/tree.py --mode geral --exclude <ATIVOS>`
  (retirar junto os irmãos próximos, ex.: `SP500 NASDAQ DOWJONES`).
- Pessoa B: etapas 1 a 4 concluídas (seções 4, 9 e 10). Pendente só o apoio à Pessoa C (fine-tuning
  a partir dos modelos salvos) e o relatório final.

## 8. O que deu errado no caminho

- A branch foi criada antes do PR #2 e ficou com o painel antigo; resolvido com `git merge main`
  (conflito simples no `requirements.txt`).
- Python 3.9 (padrão do macOS) não roda o código: `TypeError: unsupported operand type(s) for |`.
  Recriar o `.venv` com Python 3.12.
- LightGBM no Mac sem `libomp`: `Library not loaded: @rpath/libomp.dylib`. Resolvido com
  `brew install libomp`.
- `pytest` caindo com `Segmentation fault` no Mac: LightGBM e PyTorch trazem cada um o seu OpenMP e
  não convivem se o LightGBM carrega primeiro. `tests/conftest.py` importa o torch antes de tudo.
- Downloads renomeados pelo Mac (`modelo-3.md`, `test_models-2.py`): conferir o nome antes de commitar.

## 9. Rede sequencial (etapa 3)

`src/models/seq.py`: uma GRU que, em vez de olhar só as features de hoje, olha as features dos
**últimos 40 pregões** do ativo e prevê o alvo de hoje. É um modelo geral (todos os ativos juntos),
o mesmo papel do `lgbm_geral`, e começa pela volatilidade (`y_vol_h5`).

| Item | Escolha |
|---|---|
| Entrada | Janela de 40 pregões × as mesmas features `f_*` do painel, só do mesmo ativo e só até t |
| Padronização | Média e desvio de cada feature calculados **só no treino** do fold; ausente vira 0 |
| Rede | GRU de 1 camada, 32 unidades, dropout 0,1, uma saída linear |
| Treino | Adam (lr 0,001), lotes de 512, até 30 épocas; perda MSE (direção: entropia binária) |
| Parada | Early stopping pela perda na validação interna (mesma do `tree.py`), paciência 4; fica o peso da melhor época |
| Semente | 42 |

```
pip install torch
python src/models/seq.py                     # y_vol_h5, 17 folds (~10 min em CPU)
python src/models/run_1b.py                  # a comparação passa a incluir gru_geral
```

Testes em `tests/test_seq.py`: a janela nunca mistura ativos nem usa linha posterior à data da
previsão, e a padronização não muda quando os dados fora do treino mudam (verificado por mutação:
calcular o desvio com todas as linhas faz o teste falhar).

**Resultado com dados reais** (volatilidade, mesmas 15 dobras de 2012 a 2026 do experimento 1b):

| Modelo | IC | IC desvio | RMSE | Vence ativo |
|---|---|---|---|---|
| vol_passada | 0,236 | 0,097 | 0,559 | 20% |
| lgbm_ativo | 0,295 | 0,059 | 0,535 | — |
| lgbm_geral | 0,342 | 0,069 | 0,517 | 87% |
| **gru_geral** | **0,343** | 0,066 | 0,517 | 80% |

**Conclusão: a rede empata com o LightGBM geral.** A diferença de IC (0,0003) e de RMSE (0,0001) é
desprezível perto da variação entre folds (cerca de 0,07). Ver 40 dias de histórico não acrescentou
nada ao que as features do painel já resumem (desvios de 5, 21 e 63 dias, médias, defasagens). O
resultado reforça a conclusão do 1b: o ganho vem de **treinar com vários ativos**, não do tipo de
modelo, já que dois modelos bem diferentes chegam ao mesmo IC.

Consequências:
- O LightGBM continua como modelo principal: mesmo resultado, ~10× mais rápido e mais simples de
  usar no fine-tuning da Pessoa C.
- A GRU fica como segunda arquitetura, útil para mostrar que a conclusão não depende do modelo.
- Ressalva: a rede não foi ajustada (janela, tamanho, taxa de aprendizado). Testar janela maior ou a
  série `z` crua como entrada, em vez das features já agregadas, é o que mais poderia mudar o resultado.

## 10. Ajuste de hiperparâmetros (etapa 4)

`src/models/tune.py` ajusta o LightGBM **dentro de cada fold e de cada modo** (geral, classe,
ativo), sem nunca olhar o ano de teste:

1. o treino do fold é dividido em ajuste + validação interna (últimos 20% das datas, com embargo);
2. 9 combinações são treinadas no ajuste, com early stopping na validação: a configuração atual do
   `tree.py` mais 8 sorteadas (semente fixa) da grade abaixo;
3. fica a de menor perda na validação; o modelo final é retreinado no treino inteiro e prevê o teste.

| Hiperparâmetro | Valores |
|---|---|
| `num_leaves` | 7, 15, 31, 63 |
| `min_child_samples` | 50, 200, 500 (reduzido em grupos pequenos, como no `tree.py`) |
| `feature_fraction` | 0,6, 0,8, 1,0 |
| `lambda_l2` | 0, 1, 10 |
| `learning_rate` | 0,02, 0,05 |

Como os três modos recebem o mesmo ajuste, isso também responde à ressalva da seção 6
("hiperparâmetros fixos, iguais nos três modos"). As escolhas de cada fold ficam em
`reports/results/run_lgbm_<modo>_tuned.json`.

```
python src/models/tune.py                    # os 3 modos, y_vol_h5 (~15 min)
python src/models/run_1b.py --models vol_passada lgbm_ativo lgbm_geral lgbm_ativo_tuned \
    lgbm_classe_tuned lgbm_geral_tuned --ref lgbm_ativo_tuned --out comparacao_tuned.csv
```

`--folds 0 8` faz um teste rápido e salva como `..._tuned_teste`, sem sobrescrever o resultado completo.

**Resultado com dados reais** (volatilidade, mesmas 15 dobras de 2012 a 2026; `Vence ativo` aqui é
contra o `lgbm_ativo_tuned`, o modelo por ativo **já ajustado**):

| Modelo | IC | IC desvio | RMSE | Vence ativo ajustado |
|---|---|---|---|---|
| vol_passada | 0,236 | 0,097 | 0,559 | 20% |
| lgbm_ativo (sem ajuste) | 0,295 | 0,059 | 0,535 | 27% |
| lgbm_ativo_tuned | 0,299 | 0,057 | 0,533 | — |
| lgbm_classe_tuned | 0,338 | 0,061 | 0,520 | **100%** |
| lgbm_geral (sem ajuste) | 0,342 | 0,069 | 0,517 | 93% |
| **lgbm_geral_tuned** | **0,346** | 0,071 | **0,517** | 87% |

**Conclusões:**
1. **O ajuste quase não muda nada.** O IC sobe 0,004 no modelo por ativo e 0,003 no geral, bem menos
   que a variação entre folds (cerca de 0,06). A configuração original do `tree.py` já estava boa.
2. **A vantagem do modelo geral sobrevive ao ajuste.** Mesmo com o modelo por ativo ajustado, o geral
   ajustado vence em 13 dos 15 anos e o por classe em todos os 15. A diferença de IC entre geral e por
   ativo continua em torno de 0,05.
3. Com isso cai a ressalva da seção 6 sobre hiperparâmetros fixos: a conclusão do experimento 1b não
   era efeito de uma configuração que favorecia o modelo geral.
4. O `tree.py` sem ajuste continua sendo o modelo de referência para a Pessoa C: dá o mesmo
   resultado e roda em uma fração do tempo.

## 11. Com e sem minis e cripto

Os resultados das seções 4 a 10 usam `sem_minis_cripto` (28 ativos, desde 2008). Para medir o efeito de
incluir BTC, ETH, WIN e WDO, cada conjunto de ativos agora tem **pastas próprias**, sem sobrescrever os outros:
`data/processed/<conjunto>/`, `reports/results/<conjunto>/` e `models/<conjunto>/`.

| Conjunto | Ativos | Painel começa | Folds |
|---|---|---|---|
| `sem_minis_cripto` | 28 | 2008 | 17 (2010 a 2026) |
| `sem_minis` | 30 (+ BTC, ETH) | 2018 | 8 (2019 a 2026) |
| `todos` | 32 (+ BTC, ETH, WIN, WDO) | 2021 | 5 (2022 a 2026) |

(datas e folds de `docs/dados.md` e `docs/normalizacao.md`; conferir na saída do `prepare_universe.py`)

```
# 0. dados: cripto vem do Yahoo; minis precisam de data/intraday/WIN_M1.csv e WDO_M1.csv (MT5, Pessoa A)
python src/download_data.py
python src/minis_to_daily.py
# 1. um painel por conjunto, cada um na sua pasta
python src/models/prepare_universe.py --universe sem_minis_cripto
python src/models/prepare_universe.py --universe todos
# 2. os mesmos modelos em cada conjunto (vale para baselines, tree, seq, tune e run_1b)
python src/models/run_1b.py --universe sem_minis_cripto
python src/models/run_1b.py --universe todos
# 3. a comparação
python src/models/compare_universes.py --a sem_minis_cripto --b todos
```

`compare_universes.py` responde duas perguntas:

1. **Perdemos ou ganhamos ao incluir os ativos novos?** Cada modelo é comparado consigo mesmo nos dois
   conjuntos, **nas mesmas linhas**: mesma data, mesmo ativo (só os 28 que existem nos dois) e mesmo alvo.
   Assim a diferença vem só do treino: o conjunto maior tem mais ativos, mas bem menos anos de histórico.
   Saída: IC e RMSE por ano nos dois conjuntos, a diferença e em quantos anos o conjunto B venceu.
   `--strict` exige as mesmas linhas para todos os modelos ao mesmo tempo.
2. **Como o modelo se sai nos ativos novos?** IC de cada modelo em BTC, ETH, WIN e WDO, com o período e o
   número de dias de cada um.

Arquivos: `reports/results/comparacao_<a>_vs_<b>.csv` e `..._novos.csv`.

Cuidados ao ler:
- Em `todos`, o modelo **por ativo** quase não existe: ele precisa de 500 dias de treino por ativo, e o
  painel começa em 2021. Ele só aparece nos últimos folds, então as comparações com ele cobrem poucos anos.
  Isso também é um resultado: com pouco histórico, só o modelo geral ou por classe consegue ser treinado.
- A comparação da pergunta 1 cobre só os anos em que os dois conjuntos têm teste (a partir de 2021 ou 2022).
  Ela não mede a perda de 2010 a 2021, que simplesmente não existe no conjunto maior.
- Com 5 folds, uma diferença precisa ser grande e repetida na maioria dos anos para valer algo.
- Os minis têm a rolagem de contrato sem tratamento e buracos em 2023 (`docs/dados.md`, seção 3), e o WDO
  termina em 01/04/2026, o que encurta o painel `todos`.

**Resultado com dados reais** (`sem_minis_cripto` × `todos`; 5 anos de teste, 2022 a 2026; os mesmos 28 ativos e os mesmos dias nos dois):

*Pergunta 1: nos 28 ativos originais, incluir minis e cripto ajudou?* Volatilidade (`y_vol_h5`):

| Modelo | IC sem minis/cripto | IC com tudo | Diferença | Anos em que "todos" venceu |
|---|---|---|---|---|
| vol_passada | 0,200 | 0,200 | 0,000 | 40% |
| lgbm_classe | 0,298 | 0,231 | −0,067 | 20% |
| lgbm_geral | 0,314 | 0,247 | −0,067 | 60% |
| lgbm_ativo (só 2026) | 0,337 | 0,261 | −0,075 | 0% (1 ano) |

- **Piorou.** O modelo geral perde cerca de 0,07 de IC na volatilidade, e o RMSE sobe de 0,560 para 0,581. O
  por classe perde o mesmo e vence em só 1 dos 5 anos. A régua `vol_passada`, que não treina nada, não muda:
  a diferença vem do treino.
- O geral com tudo vence em 3 dos 5 anos, mas perde por mais nos outros dois. Com 5 anos e desvio de cerca de
  0,06 entre eles, a perda é consistente na média, mas não em todo ano.
- **A causa provável é o histórico, não os ativos novos em si.** Para incluir os minis, o treino passa a
  começar em 2021 em vez de 2008: em 2022 o modelo estuda 1 ano, não 14. Este teste não separa os dois
  efeitos (mais ativos × menos anos); ver "Próximo teste".
- Direção e retorno: diferenças de IC entre −0,03 e +0,07 em valores perto de zero, com AUC perto de 0,5 nos
  dois. É ruído, como nas seções 4 e 5.

*Pergunta 2: como os modelos se saem nos ativos novos?* Volatilidade, IC ao longo do tempo de cada ativo,
jan/2022 a mar/2026 (cerca de 617 dias):

| Ativo | lgbm_geral | lgbm_classe | vol_passada | media_historica |
|---|---|---|---|---|
| WIN | **0,375** | 0,316 | 0,241 | −0,085 |
| WDO | 0,175 | **0,211** | 0,159 | −0,275 |
| BTC | 0,080 | −0,135 | **0,188** | −0,156 |
| ETH | 0,058 | 0,073 | 0,090 | **0,122** |

- **Nos minis o modelo geral funciona bem**: no WIN ganha com folga da régua (0,375 × 0,241), e no WDO o
  por classe e o geral ficam acima dela. Faz sentido: os minis acompanham o Ibovespa e o dólar, que o modelo
  já conhece.
- **Na cripto, não**: no BTC e no ETH o modelo geral perde para a régua de agitação recente. A cripto se
  comporta diferente dos outros mercados (negocia todo dia, oscila muito mais), e o padrão aprendido nos
  outros ativos não se transfere bem.
- A coluna `lgbm_ativo` desta tabela foi omitida: em `todos` o modelo por ativo só existe em 2026 (cerca de
  40 dias por ativo), então o IC dele não é comparável com o dos outros, que cobrem 4 anos.
- Este IC é calculado ao longo do tempo de um único ativo; não é o mesmo número do IC das seções 4 a 10
  (vários ativos e dias juntos). Compare os números só dentro desta tabela.

**Decisão sugerida:** manter `sem_minis_cripto` como resultado principal (17 anos de teste) e apresentar
`todos` como análise complementar: incluir minis e cripto custa histórico e piora a volatilidade dos 28 ativos;
o modelo geral funciona nos minis e não funciona na cripto.

### 11.1 Manter minis e cripto sem perder histórico: painel sem alinhar

A perda acima vem de o painel ser **alinhado**: ele só guarda os dias em que *todos* os ativos negociaram,
então o ativo que começa mais tarde (os minis, em 2020) corta o histórico de todos. Com
`prepare_universe.py --no-align`, cada ativo entra com o **próprio** histórico: os 28 desde 2008, a cripto
desde 2014/2017 e os minis desde 2020, e o painel volta a ter 17 folds. Com a normalização `vol_scaled`, as
features e os alvos de um ativo dependem só dele, então os 28 ativos ficam **idênticos** com e sem os ativos
novos (há um teste para isso) e a única diferença é o modelo geral treinar também com eles.

```
python src/models/prepare_universe.py --universe sem_minis_cripto --no-align   # -> sem_minis_cripto_sem_alinhar
python src/models/prepare_universe.py --universe todos --no-align              # -> todos_sem_alinhar
python src/models/run_1b.py --universe sem_minis_cripto_sem_alinhar
python src/models/run_1b.py --universe todos_sem_alinhar
python src/models/compare_universes.py --a sem_minis_cripto_sem_alinhar --b todos_sem_alinhar
```

Cuidados:
- **Não compare painel alinhado com não alinhado.** O alinhado descarta os dias em que algum mercado estava
  fechado (feriados diferentes em cada país): sobram cerca de 164 pregões por ano em vez de cerca de 250, e
  "a volatilidade dos próximos 5 dias" vira outra conta. O `compare_universes.py` avisa quando os alvos não batem.
- Os números sem alinhar **não são comparáveis** com os das seções 4 a 10 (alinhadas), pelo mesmo motivo.
  Se o grupo adotar o painel sem alinhar, as comparações principais precisam ser refeitas nele.
- Por ativo e por classe não mudam com os ativos novos (eles ficam em classes próprias); só o geral muda.
- O alinhamento é decisão da Pessoa A (contrato do painel); usar `--no-align` como padrão deve ser combinado com o grupo.
- Cripto negocia todos os dias: sem alinhar, "5 dias" na cripto são 5 dias corridos, e nos outros, 5 pregões.

**Resultado com dados reais (sem alinhar)** (`sem_minis_cripto_sem_alinhar` × `todos_sem_alinhar`; 17 anos
de teste, 2010 a 2026; os mesmos 28 ativos e os mesmos dias nos dois):

*Pergunta 1: incluir minis e cripto mudou algo nos 28 ativos originais?* Volatilidade (`y_vol_h5`):

| Modelo | IC sem minis/cripto | IC com tudo | Diferença | RMSE (os dois) |
|---|---|---|---|---|
| vol_passada | 0,302 | 0,302 | 0 | 0,554 |
| lgbm_ativo | 0,364 | 0,364 | 0 | 0,512 |
| lgbm_classe | 0,388 | 0,388 | 0 | 0,511 |
| lgbm_geral | 0,378 | 0,378 | −0,0002 | 0,510 |

- **Não mudou nada.** Por ativo e por classe são idênticos, como esperado (os ativos novos ficam em classes
  próprias). O geral, o único que treina com os ativos novos, muda −0,0002 de IC, com o mesmo RMSE: efeito nulo.
- Direção e retorno: diferenças de no máximo 0,0008 de IC. Também nulas.
- **Conclusão: sem alinhar, minis e cripto entram sem custo para os outros ativos.** A perda de 0,07 do painel
  alinhado vinha do histórico cortado, não dos ativos novos.

*Pergunta 2: como os modelos se saem nos ativos novos, agora com todo o histórico deles?* Volatilidade, IC ao
longo do tempo de cada ativo:

| Ativo | Período | lgbm_geral | lgbm_classe | lgbm_ativo | vol_passada |
|---|---|---|---|---|---|
| BTC | 2015-01 a 2026-11 | **0,454** | 0,376 | 0,374 | 0,401 |
| ETH | 2018-03 a 2026-11 | 0,308 | **0,334** | 0,226 | 0,253 |
| WIN | 2021-04 a 2026-09 | **0,404** | 0,356 | 0,124 | 0,325 |
| WDO | 2021-03 a 2026-03 | 0,223 | 0,174 | **0,241** | 0,152 |

- **O modelo geral ganha da régua de agitação recente nos quatro ativos novos**, e é o melhor no BTC e no WIN.
- Isso muda a conclusão do painel alinhado sobre a cripto: lá o geral perdia para a régua no BTC e no ETH, mas
  com só 4 anos de teste e 1 ano de treino no início. Com o histórico completo, ele funciona também na cripto.
- O modelo por ativo é o pior no WIN (0,124): os minis têm pouco histórico (desde 2020) para um modelo sozinho.
- Direção e retorno: os modelos ficam entre −0,09 e +0,10, sem padrão (a média histórica chega a −0,14 no
  WDO). Ruído.

*Experimento 1b no painel sem alinhar com os 32 ativos* (`run_1b.py --universe todos_sem_alinhar`; 17 folds,
todos os ativos, mesmas linhas para todos os modelos). Volatilidade:

| Modelo | IC | IC desvio | RMSE | Vence o por ativo |
|---|---|---|---|---|
| vol_passada | 0,304 | 0,098 | 0,544 | 6% (1 de 17) |
| lgbm_ativo | 0,364 | 0,082 | 0,511 | — |
| lgbm_classe | **0,387** | 0,082 | 0,503 | **100% (17 de 17)** |
| lgbm_geral | 0,377 | 0,088 | **0,503** | 76% (13 de 17) |

- **A conclusão do 1b se mantém**: treinar com vários ativos juntos (por classe ou geral) vence o modelo por
  ativo na maioria dos anos; o por classe vence em todos.
- **A vantagem é menor que no painel alinhado** (geral − por ativo: 0,013 aqui, 0,047 na seção 4). *Não* é
  porque o modelo por ativo melhorou: a vantagem dele sobre a régua ficou igual (0,295 − 0,236 = 0,059 no
  alinhado; 0,364 − 0,304 = 0,060 aqui). O que caiu foi a vantagem do **geral** sobre a régua (0,342 − 0,236 =
  0,106 no alinhado; 0,377 − 0,304 = 0,073 aqui). Duas hipóteses, testadas na seção 12: (a) sem alinhar, os
  ativos com décadas de histórico (o S&P 500 desde 1927, o Nikkei desde 1965) dominam o treino do modelo
  geral, com dados de mercados muito diferentes dos atuais; (b) os anos de teste não são os mesmos (2012 a
  2026 no alinhado, 2010 a 2026 aqui). **Resultado (seção 12): as duas foram descartadas.** O mais provável é
  que as vantagens dos dois painéis simplesmente não sejam comparáveis, porque os alvos são diferentes (ver
  seção 12.1).
  *(Versão anterior deste item, errada: "o modelo por ativo tem mais dados e precisa menos da ajuda dos
  outros". Os números acima mostram que o por ativo não ganhou nada sobre a régua.)*
- Por classe e geral empatam (0,387 × 0,377, diferença bem menor que o desvio entre anos, cerca de 0,08), mas
  o por classe é o mais consistente.
- A régua de agitação recente fica mais forte (0,304 contra 0,236 no alinhado): os alvos sem buracos de
  calendário são mais fáceis de prever para todos.
- Direção e retorno: ruído, como antes; o geral fica até um pouco abaixo da média histórica na direção
  (AUC 0,508 × 0,516).

**Decisão sugerida:** se o grupo quiser minis e cripto, usar o painel **sem alinhar** (`--no-align`), com o
modelo por classe e o geral como principais. Antes de adotá-lo como padrão: (1) combinar com a Pessoa A, dona
do contrato do painel; (2) a Pessoa C refazer as comparações dela nesse painel; (3) não misturar números
alinhados e não alinhados no relatório.

**Teste feito:** seção 12 (cortar o histórico antigo). A vantagem do geral não voltou; ver 12.1.

## 12. Qual painel usar: histórico completo ou cortado

O professor confirmou que **cortar dados antigos é aceitável quando eles não representam o comportamento
atual dos ativos**. O teste: o mesmo painel sem alinhar, com e sem o histórico antigo, nas mesmas linhas de
teste.

`prepare_universe.py --start AAAA-MM-DD` corta o painel **depois** de calcular features e alvos. As janelas
móveis continuam usando os dias anteriores ao corte, então toda linha que sobra é idêntica à do painel
completo (há teste para isso); o corte só tira as linhas antigas do **treino**. Os anos de teste são os
mesmos (2010 a 2026 com corte em 2008).

```
python src/models/prepare_universe.py --universe todos --no-align --start 2008-01-01
python src/models/run_1b.py --universe todos_sem_alinhar_desde2008
python src/models/edge_report.py --panels todos_sem_alinhar todos_sem_alinhar_desde2008
python src/models/edge_report.py --panels todos_sem_alinhar todos_sem_alinhar_desde2008 --from-year 2012
```

O `edge_report.py` mostra, por painel e modelo: IC, RMSE, **vantagem sobre a régua** (IC do modelo menos IC
da régua, ano a ano) e em quantos anos ganhou dela; e, como os alvos são iguais nos dois painéis, a diferença
de IC de cada modelo entre os painéis, pareada por ano. O `--from-year 2012` repete tudo só nos anos de teste
do painel alinhado (hipótese b da seção 11.1).

Como decidir:
- se o geral cortado ganhar do geral completo na maioria dos anos, o histórico antigo atrapalha → **painel
  oficial: sem alinhar, desde 2008** (e a explicação da seção 11.1 é a hipótese a);
- se empatar, o histórico antigo não atrapalha nem ajuda → fica o painel completo (mais dados, menos escolhas
  arbitrárias), e a diferença da seção 11.1 vem dos anos de teste (hipótese b, confirmada com `--from-year 2012`);
- a data de corte não deve ser escolhida olhando o resultado do teste (isso seria ajustar ao teste). 2008 foi
  fixado antes, por ser o início do painel alinhado e incluir a crise de 2008 no treino.

### 12.1 Resultado com dados reais

Volatilidade (`y_vol_h5`), 17 anos de teste (2010 a 2026), mesmas linhas nos dois painéis.

| Painel | Modelo | IC | Desvio entre anos | RMSE | Vantagem sobre a régua | Vence a régua |
|---|---|---|---|---|---|---|
| completo | régua (vol. passada) | 0,300 | 0,098 | 0,542 | — | — |
| completo | por ativo | 0,361 | 0,082 | 0,509 | 0,061 | 16 de 17 |
| completo | por classe | 0,383 | 0,083 | 0,501 | 0,082 | 17 de 17 |
| completo | geral | 0,374 | 0,088 | 0,500 | 0,074 | 17 de 17 |
| desde 2008 | régua (vol. passada) | 0,300 | 0,098 | 0,542 | — | — |
| desde 2008 | por ativo | 0,343 | 0,080 | 0,515 | 0,042 | 13 de 17 |
| desde 2008 | por classe | 0,378 | 0,084 | 0,507 | 0,077 | 17 de 17 |
| desde 2008 | geral | 0,375 | 0,087 | 0,503 | 0,074 | 17 de 17 |

Diferença cortado − completo, ano a ano: geral 0,000 (cortado melhor em 8 de 17 anos); por classe −0,005
(7 de 17); por ativo −0,019 (5 de 17).

- **Geral:** cortar não muda nada (diferença 0,000; a vantagem sobre a régua é 0,074 nos dois painéis).
- **Por ativo:** cortar piora (−0,019, melhor em só 5 de 17 anos). Faz sentido: ele só tem os dados do próprio
  ativo, e o corte tira parte deles.
- **Decisão: painel oficial = `todos_sem_alinhar`, sem corte.** O histórico antigo não atrapalha o modelo
  geral e ajuda o por ativo; cortar seria uma escolha a mais sem ganho.

**Hipóteses da seção 11.1.**
- (a) "o histórico antigo domina o treino e atrapalha": **descartada** pelo resultado acima.
- (b) "os anos de teste são diferentes": **descartada**. Média do Geral − régua só em 2012 a 2026 (os anos do
  painel alinhado), com os valores da tabela por ano da seção 16.1:
  0,088 + 0,120 + 0,064 + 0,081 + 0,081 + 0,052 + 0,067 + 0,094 + 0,024 + 0,099 + 0,083 + 0,071 + 0,055 +
  0,068 + 0,064 = 1,111; 1,111 / 15 = **0,074**, igual à dos 17 anos.
- **O que sobra (explicação provável, não provada):** os dois painéis medem coisas diferentes. No alinhado,
  "5 pregões" são 5 datas em que todos os ativos negociaram, o que pode cobrir duas semanas reais, e até a
  régua muda (0,236 no alinhado, 0,300 aqui). Por isso as vantagens dos dois painéis não devem ser comparadas
  entre si; vale comparar modelos dentro de um mesmo painel.

**Números um pouco diferentes da seção 11.1** (régua 0,304 → 0,300; geral 0,377 → 0,374; por ativo 0,364 →
0,361): o painel foi refeito com dados até novembro de 2026. As conclusões não mudam.

## 13. Impacto de cada ativo (tirar um por vez)

`ablation.py` treina o modelo geral 32 vezes, cada vez sem um ativo, e compara com o modelo completo:

- **nos outros ativos**: IC do modelo sem X menos IC do completo, nas linhas dos outros 31 ativos. Negativo
  = X ajudava os outros (e colocar X de volta ganha esse mesmo valor). Só entram os anos em que X já
  estava no treino;
- **no próprio X**: IC ao longo do tempo em X, visto no treino × nunca visto (a pergunta da Pessoa C, aqui
  como subproduto);
- **ruído**: só trocar a semente aleatória do LightGBM já muda o IC um pouco. O script treina o modelo
  completo com mais 2 sementes e mede essa variação nas mesmas linhas. Efeito menor que o ruído
  (`acima_do_ruido = False`) não deve ser lido como impacto do ativo.

```
python src/models/ablation.py --universe todos_sem_alinhar                  # ~35 treinos; continua de onde parou
python src/models/ablation.py --universe todos_sem_alinhar --by classe      # tira uma classe inteira por vez
```

Só a volatilidade, por padrão: em retorno e direção não há sinal para perder (seção 5).
Expectativa honesta: com 31 outros ativos, tirar um só deve mudar pouco; a tabela por classe mostra efeitos
maiores, se existirem.

### 13.1 Resultado com dados reais

- **Nenhum ativo sozinho importa.** Tirar qualquer um dos 32 ativos muda o IC dos outros em no máximo 0,0012
  (HANGSENG +0,0012, WEGE3 −0,0011, SP500 +0,0011), mais de 4 vezes abaixo do limite prático de 0,005. O ganho
  do modelo geral vem do **conjunto** de ativos, não de um ativo específico.
- 13 ativos passaram do ruído medido, mas por diferenças desprezíveis; o ruído foi medido com só 2
  sementes extras e tende a ser subestimado. *(A primeira versão do relatório contou esses ativos como "ajudam" e
  "atrapalham"; o `report.py` foi corrigido para exigir também a diferença mínima de 0,005.)*
- Padrão pequeno, mas que se repete: os ativos cuja saída **melhora** os outros são índices de histórico muito
  longo (SP500 desde 1928: os outros melhoraram em 15 de 17 anos; também NIKKEI225 e HANGSENG). Eles pesam um
  pouco demais no treino, mas o efeito (≈ 0,001) não muda nenhuma conclusão.
- **Ativo nunca visto (interessa à Pessoa C):** prevendo um ativo que ficou fora do treino, o IC quase não cai:
  em 30 dos 32 ativos a queda é menor que 0,006. As exceções são **SUZB3** (0,415 → 0,385, −0,030) e **VIX**
  (0,322 → 0,298, −0,024). O VIX é um nível de volatilidade, diferente de um preço; o SUZB3 tem problema de
  dados (seção 16.1).

## 14. Como as features são calculadas e nomes técnicos

Ver `docs/features.md`: fórmula, nome técnico e exemplo numérico de cada feature e de cada alvo, e as bordas
da série. Os nomes no código não mudaram. Importância de cada feature no modelo geral:

```
python src/models/importance.py --universe todos_sem_alinhar
```

### 14.1 Resultado com dados reais (modelo geral, `y_vol_h5`)

| Feature | Nome técnico | Importância (%) | Desvio entre anos (%) |
|---|---|---|---|
| `f_z_std21` | Volatilidade realizada relativa, 21 pregões | 37,3 | 1,1 |
| `f_range` | Amplitude diária relativa | 13,0 | 2,9 |
| `f_z_mean21` | Momentum ajustado ao risco, 21 pregões | 11,8 | 1,2 |
| `f_z_std5` | Volatilidade realizada relativa, 5 pregões | 10,8 | 0,4 |
| `f_rsi14` | Índice de força relativa, 14 pregões | 5,3 | 0,3 |
| `f_z_mean63` | Momentum ajustado ao risco, 63 pregões | 4,8 | 0,5 |
| `f_z_std63` | Volatilidade realizada relativa, 63 pregões | 4,7 | 0,5 |
| `f_z_lag0` | Retorno normalizado (t) | 4,7 | 0,4 |
| `f_z_lag1` | Retorno normalizado (t−1) | 3,0 | 0,2 |
| `f_volrel` | Volume relativo, 21 pregões | 1,8 | 0,6 |
| `f_z_lag2` | Retorno normalizado (t−2) | 1,5 | 0,1 |
| `f_z_lag3` | Retorno normalizado (t−3) | 0,8 | 0,1 |
| `f_z_lag4` | Retorno normalizado (t−4) | 0,6 | 0,1 |

- A volatilidade recente do próprio ativo (5, 21 e 63 pregões) soma 52,8%: a agitação de amanhã se parece com a
  de agora.
- A **amplitude diária** é a segunda: a oscilação dentro do dia traz informação que o fechamento não mostra.
- O **momentum de 21 pregões** (11,8%) provavelmente reflete o efeito alavancagem: depois de quedas, a
  volatilidade tende a subir. É uma interpretação; para usar no artigo, precisa de citação.
- **Volume** quase não é usado (1,8%); ele falta em câmbio, VIX e juros e em trechos antigos dos índices.
- Os desvios entre anos são pequenos: o modelo usa as mesmas features em todos os anos de teste.

## 15. A variabilidade dos outros ativos agrega valor?

Quatro features novas com a agitação do **mercado** e da **classe** (média das features de volatilidade dos
outros ativos, sempre do pregão anterior; definições em `docs/features.md`, seção 3). O teste separa duas
perguntas:

- **por ativo com mercado × por ativo**: um modelo que só vê o próprio ativo ganha ao ver também a agitação
  dos outros? Se sim, a variabilidade dos outros ativos tem informação;
- **geral com mercado × geral**: o modelo geral já aproveita isso por treinar com todos? Se o ganho aqui for
  bem menor que no por ativo, a vantagem do modelo geral vem em parte de "conhecer o mercado".

```
python src/models/tree.py --mode ativo --market --universe todos_sem_alinhar
python src/models/tree.py --mode classe --market --universe todos_sem_alinhar
python src/models/tree.py --mode geral --market --universe todos_sem_alinhar
python src/models/market.py --universe todos_sem_alinhar
python src/models/importance.py --universe todos_sem_alinhar --model geral_mercado
```

### 15.1 Resultado com dados reais

| Modelo | IC sem mercado | IC com mercado | Ganho | Desvio | Anos melhor com mercado |
|---|---|---|---|---|---|
| por ativo | 0,3642 | 0,3500 | −0,0141 | 0,0154 | 1 de 17 |
| por classe | 0,3903 | 0,3814 | −0,0089 | 0,0081 | 2 de 17 |
| geral | 0,3811 | 0,3742 | −0,0069 | 0,0083 | 4 de 17 |

- **Não agrega.** As features de mercado pioraram os três modelos de forma consistente (ganho negativo e acima
  de 0,005 em módulo, piores em pelo menos 13 de 17 anos).
- **O que isso diz sobre a hipótese do projeto:** nem o modelo por ativo, que não vê os outros ativos, ganhou
  ao receber a agitação do mercado. Então a vantagem do modelo geral **não vem de "conhecer o mercado"**; vem
  de ter muito mais exemplos para aprender o mesmo padrão (como a volatilidade de qualquer ativo se comporta
  depois de dias agitados).
- Ressalva: as features usam o pregão anterior, por cautela contra vazamento (mercados fecham em horários
  diferentes). Uma versão com o mesmo dia não foi testada.
- (Os ICs "sem mercado" diferem um pouco dos da seção 12 porque aqui a comparação usa as linhas comuns a cada
  par de modelos.)

## 16. Condições de contorno

Entendemos "condições de contorno" como **as bordas e os limites em que os resultados valem**: onde os dados
começam e terminam, o que acontece nas bordas (aquecimento das janelas, fim da série, buracos), e em que
situações o modelo funciona ou não (mercado calmo ou agitado, classe de ativo, pouco ou muito histórico,
ano). Se o professor quis dizer outra coisa, esta seção se adapta.

```
python src/models/boundary.py --universe todos_sem_alinhar
```

Saídas em `reports/results/<painel>/contorno/`:
- `bordas.csv`: por ativo, início e fim no painel, pregões de aquecimento perdidos (113 por ativo, ver
  `docs/features.md`, seção 5), retornos descartados por buraco ou preço inválido, maior buraco, % sem volume
  e sem amplitude;
- `contorno_regime.csv`: IC por regime de mercado (terços da agitação média do mercado no dia: calmo,
  normal, agitado; e os 5% mais agitados);
- `contorno_classe.csv`, `contorno_ano.csv`: IC por classe e por ano de teste;
- `contorno_historico.csv` e `contorno_historico_inicio.csv`: IC por anos de histórico do ativo;
  `cobertura_historico.csv`: % das linhas de teste que cada modelo consegue prever. O modelo por ativo
  precisa de 500 linhas do próprio ativo para existir; nos primeiros anos de um ativo novo só o geral e a
  régua preveem. Essa é uma condição de contorno a favor do modelo geral.

### 16.1 Resultado com dados reais (modelo geral, `y_vol_h5`)

**Em todos os recortes o geral ganhou da régua**, inclusive nos 17 de 17 anos de teste.

Por regime de mercado (agitação média do mercado no dia):

| Regime | Régua | Por ativo | Por classe | Geral | Geral − régua | Anos | Linhas |
|---|---|---|---|---|---|---|---|
| calmo | 0,230 | 0,280 | 0,297 | 0,297 | 0,068 | 16 | 38.443 |
| normal | 0,254 | 0,311 | 0,337 | 0,325 | 0,071 | 17 | 44.238 |
| agitado | 0,276 | 0,370 | 0,397 | 0,389 | 0,113 | 15 | 40.154 |
| extremo (5% mais agitados) | 0,229 | 0,362 | 0,391 | 0,405 | 0,176 | 8 | 4.839 |

- **O modelo é mais útil quando o mercado está agitado**: a vantagem sobre a régua vai de 0,068 (calmo) a 0,176
  (extremo). O extremo tem só 8 anos e 4.839 linhas, então é o número menos certo.

Por classe:

| Classe | Régua | Por ativo | Por classe | Geral | Geral − régua | Anos |
|---|---|---|---|---|---|---|
| ações BR | 0,232 | 0,243 | 0,282 | 0,290 | 0,058 | 17 |
| câmbio | 0,268 | 0,361 | 0,385 | 0,361 | 0,093 | 17 |
| commodities | 0,255 | 0,279 | 0,275 | 0,280 | 0,026 | 17 |
| cripto | 0,282 | 0,306 | 0,346 | 0,362 | 0,080 | 10 |
| índices | 0,302 | 0,414 | 0,430 | 0,418 | 0,116 | 17 |
| juros (VIX, US10Y) | 0,301 | 0,444 | 0,431 | 0,358 | 0,057 | 17 |
| minis | 0,200 | 0,135 | 0,282 | 0,265 | 0,065 | 3 |

- O geral **não é o melhor em todas as classes**: em **juros** fica bem abaixo do por ativo (0,358 × 0,444) —
  são séries de nível, diferentes das de preço; em **câmbio** e **índices** o por classe é melhor; em
  **commodities** quase não passa da régua (+0,026). É melhor em **ações BR** e **cripto**.
- Nos **minis** o por ativo fica abaixo da própria régua (0,135 × 0,200): pouco histórico; juntar ativos resolve.

Por anos de histórico do ativo (só régua e geral, que preveem desde o início):

| Histórico | Régua | Geral | Geral − régua | Linhas | Cobertura do por ativo |
|---|---|---|---|---|---|
| menos de 2 anos | 0,201 | 0,264 | 0,063 | 2.098 | 2,7% |
| 2 a 5 anos | 0,310 | 0,371 | 0,061 | 4.985 | 88,7% |
| 5 a 10 anos | 0,275 | 0,326 | 0,051 | 12.349 | 99,4% |
| 10 anos ou mais | 0,300 | 0,377 | 0,077 | 106.119 | 99,7% |

- **Ativo novo:** nos 2 primeiros anos o modelo por ativo só consegue prever 2,7% das linhas (precisa de 500
  linhas próprias); o geral prevê todas, e ganha da régua (0,264 × 0,201). É um argumento direto a favor da
  hipótese do projeto.

Por ano de teste: o geral ganhou da régua em todos (vantagem entre 0,024 e 0,120). **2020** foi o ano mais fácil
(geral 0,600) e o de menor vantagem (régua 0,576): com a agitação alta o ano inteiro, "repetir a agitação
recente" já acerta muito. O pior ano foi **2023** (0,253).

**Bordas dos dados** (`bordas.csv`): todo ativo perde 113 pregões de aquecimento, como previsto em
`docs/features.md`, exceto **SUZB3 (625)** e **WEGE3 (162)**, que têm falhas no início. Pontos para a
Pessoa A:
- **USDBRL**: buraco de **576 dias** sem cotação;
- **WIN** (137 dias) e **WDO** (110 dias): buracos no arquivo intradiário dos minis;
- **SUZB3**: só 2.458 linhas desde 2007 (o esperado seriam cerca de 4.800) — explica a queda dele na seção 13.1;
- **EURUSD** e **USDJPY**: buracos de 18 dias;
- falta de amplitude (máxima e mínima) em trechos antigos: US10Y 48,5%, NIKKEI225 37,2%, SP500 33,9%; o modelo
  lida com isso (seção 2.6 de `docs/features.md`).

## 17. Como ler os resultados das seções 12 a 19

```
python src/models/report.py --universe todos_sem_alinhar --cut todos_sem_alinhar_desde2008
```

Gera `reports/results/todos_sem_alinhar/relatorio_pendencias.html`. Para cada pergunta: a tabela, um gráfico,
como ler e uma frase do que aconteceu, por regras fixas: diferença menor que 0,005 de IC = praticamente igual;
"consistente" = mesmo sinal em pelo menos 70% dos anos de teste; na ablação, só conta o que passa do ruído.
As frases são ponto de partida: conferir sempre com a tabela.

## 18. Modelos mais simples que o LightGBM

Pedido do professor: comparar com modelos mais simples, para justificar a escolha do LightGBM. Dois modelos,
cada um na versão certa para o tipo de alvo (`src/models/simple.py`):

| Modelo | Alvos numéricos (retornos, volatilidade) | Direção (sobe ou cai: classificação) |
|---|---|---|
| linear | regressão linear com penalização (Ridge) | regressão logística |
| árvore | uma árvore de regressão (CART) | uma árvore de classificação |

O LightGBM é uma soma de centenas de árvores como essa; a regressão linear supõe que cada feature tem um efeito
fixo e somado. Mesmas regras de sempre: mesmo walk-forward, mesmas 13 features, nada do ano de teste usado para
escolher nada. Um hiperparâmetro por modelo é escolhido na validação interna (igual ao `tree.py`): força da
penalização (Ridge: alpha em 0,1; 1; 10; 100; logística: C em 0,01; 0,1; 1) ou profundidade da árvore (3, 5, 7,
9, com pelo menos 1% das linhas em cada folha). Valores vazios (amplitude, volume) = mediana do treino + uma
coluna avisando que estava vazio; a regressão linear recebe as features padronizadas com média e desvio do treino.

```
python src/models/simple.py --model linear --universe todos_sem_alinhar
python src/models/simple.py --model arvore --universe todos_sem_alinhar
python src/models/simple.py --compare --universe todos_sem_alinhar
```

Como decidir: se o LightGBM melhora de forma consistente (≥ 0,005 de IC e em pelo menos 70% dos anos) sobre os
dois, a escolha está justificada. Se a regressão linear ficar praticamente igual na volatilidade, isso deve ser
dito: o ganho do projeto viria de juntar ativos, não da complexidade do modelo (coerente com a seção 9, em que a
rede neural empatou com o LightGBM).

### 18.1 Resultado com dados reais

Volatilidade (`y_vol_h5`), 17 anos de prova, mesmas linhas para todos os modelos:

| Modelo | IC | LightGBM − modelo | Anos com LightGBM melhor |
|---|---|---|---|
| média histórica | 0,011 | 0,370 | 17 de 17 |
| régua (vol. passada) | 0,308 | 0,073 | 17 de 17 |
| linear (Ridge) | 0,334 | 0,047 | 17 de 17 |
| uma árvore de decisão | 0,348 | 0,033 | 17 de 17 |
| **LightGBM geral** | **0,381** | — | — |

- **A escolha do LightGBM está justificada:** ganhou da regressão linear (+0,047) e da árvore única (+0,033)
  em todos os 17 anos. A regressão linear ganha da régua só por 0,026; o LightGBM, por 0,073, quase três vezes
  mais.
- **Isso afina uma conclusão anterior** (seção 9, "o ganho vem dos dados, não do tipo de modelo"): o modelo
  precisa ser **flexível** (não linear). Entre dois modelos flexíveis bem diferentes (LightGBM e rede neural
  GRU) o resultado empatou; um modelo linear ou uma árvore só ficam abaixo.
- **Retornos e direção:** todos perto de zero (IC entre −0,014 e 0,019) ou de 0,5 (AUC entre 0,502 e 0,516).
  Não há sinal para nenhum modelo. A média histórica tem AUC 0,516 na direção, um pouco acima dos outros; isso
  provavelmente vem de diferenças de tendência média entre ativos ao longo do tempo, e não de previsão do dia
  seguinte. Não usar como resultado.
- O IC do LightGBM aqui (0,381) difere do da seção 12.1 (0,374) porque cada tabela usa as linhas comuns aos
  modelos que compara.


## 19. Quanto dado é preciso para treinar

Pedido do professor: qual a menor quantidade de dados que podemos usar para treinar o modelo sem perder
desempenho, testando um corte na época da pandemia (2021). Dois testes:

1. **Corte fixo em 2021-01-01** (`prepare_universe.py --start`). O modelo só estuda dados a partir de 2021,
   então só pode ser avaliado de 2022 a 2026 (5 anos de prova), com 1 a 5 anos de treino. A comparação com o
   painel completo usa as mesmas linhas desses 5 anos. O modelo por ativo não existe em 2022 (precisa de 500
   dias do próprio ativo), por isso a comparação usa só a régua e o geral.
2. **Janela de treino** (`window.py`): em cada um dos 17 anos de prova, o geral estuda só os últimos N anos
   (N = 1, 2, 3, 5, 10). Responde à mesma pergunta em todos os anos, e não só depois de 2021.

```
python src/models/prepare_universe.py --universe todos --no-align --start 2021-01-01
python src/models/baselines.py --universe todos_sem_alinhar_desde2021
python src/models/tree.py --mode geral --targets y_vol_h5 --universe todos_sem_alinhar_desde2021
python src/models/edge_report.py --panels todos_sem_alinhar todos_sem_alinhar_desde2021 --models vol_passada lgbm_geral
python src/models/window.py --universe todos_sem_alinhar
```

Leitura: a menor janela "praticamente igual" (diferença menor que 0,005 de IC) a usar todo o passado é a
quantidade mínima de dados. Atenção: a janela curta perde as crises antigas (2008, 2020) do treino, então o
resultado em anos de crise pode piorar mais que a média.

### 19.1 Resultado com dados reais

**Janela de treino** (volatilidade, 17 anos de prova; diferença para o treino com todo o passado):

| Treino com | IC | Diferença | Anos pior | Leitura |
|---|---|---|---|---|
| todo o passado | 0,381 | — | — | referência |
| últimos 10 anos | 0,382 | +0,001 | 8 de 17 | praticamente igual |
| últimos 5 anos | 0,379 | −0,003 | 11 de 17 | praticamente igual |
| últimos 3 anos | 0,375 | −0,006 | 11 de 17 | empate (muda de sinal entre os anos) |
| últimos 2 anos | 0,370 | −0,011 | 12 de 17 | perde de forma consistente |
| último 1 ano | 0,339 | −0,043 | 17 de 17 | perde de forma consistente |
| régua (não treina) | 0,308 | −0,073 | — | piso |

**Corte em 2021** (só os anos de prova 2022 a 2026): geral 0,312 contra 0,318 com todo o passado (−0,006;
melhor em 2 de 5 anos: empate pela regra). A vantagem sobre a régua cai de 0,069 para 0,063 e o geral continua
ganhando dela nos 5 anos.

- **Resposta ao professor: cerca de 5 anos de histórico bastam.** Com 5 ou 10 anos, o modelo fica praticamente
  igual a estudar todo o passado. Com 3 anos começa a perder um pouco; com 1 ou 2 anos, perde de forma
  consistente (−0,043 com 1 ano).
- O corte em 2021 concorda: em 2022 o modelo tinha só 1 ano para estudar e em 2026, 5 anos; no conjunto,
  perde pouco (−0,006) e ainda ganha da régua em todos os anos.
- Por que 5 anos bastam: o modelo geral junta os 32 ativos; 5 anos são cerca de 40 mil linhas de treino
  (32 ativos × cerca de 250 pregões × 5), o bastante para aprender o padrão. O ganho de juntar ativos é,
  em parte, ganho de quantidade de dados.
- Isso também é coerente com a seção 12.1: cortar em 2008 (mais de 5 anos antes de cada prova) não mudou nada.

