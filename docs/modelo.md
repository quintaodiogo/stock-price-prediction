# Modelo geral (Pessoa B)

Baselines e LightGBM no walk-forward, e o experimento 1b do README (seção 4): o modelo treinado
em todos os ativos prevê melhor do que o modelo treinado só no ativo?

## 1. Como reproduzir

Requer **Python 3.10 ou mais novo** (o código usa `int | None`). No Mac, o LightGBM precisa do
`libomp` (`brew install libomp`).

```
python src/download_data.py
python src/features.py --universe sem_minis_cripto
python -m pytest tests                     # 26 testes, incluindo os da Pessoa B
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

**Resultado com dados reais (sem alinhar):** *a preencher depois de rodar.*

**Próximo teste (separar os dois efeitos):** rodar os 28 ativos com o treino começando também em 2021. Se a
perda de 0,07 aparecer ali também, a causa é o histórico mais curto, e não os ativos novos. O conjunto
`sem_minis` (com cripto, desde 2018) ajuda a separar o efeito da cripto do dos minis.
