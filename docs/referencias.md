# Referências e como vamos aplicá-las

Resumo da literatura que sustenta as decisões do projeto (modelo geral, normalização, avaliação) e o que
cada artigo muda no nosso trabalho.

> **Nível de leitura:** as descrições abaixo vêm dos resumos e trechos disponíveis nas páginas dos
> artigos, não de uma leitura integral. Antes de citar um número ou um detalhe de método em relatório,
> abrir o artigo e confirmar. Itens marcados com **[ler]** merecem leitura completa por quem for
> implementar a parte correspondente.

## Mapa rápido

| Decisão do projeto | Artigos de apoio | Responsável principal |
|---|---|---|
| Treinar um modelo geral com vários ativos | Sirignano & Cont (2019); revisão de transfer learning | Pessoa B |
| Transferir para ativos novos | Sirignano & Cont (2019); revisão de transfer learning; Volatilidade com CNN | Pessoa C |
| Normalizar os dados | Kim et al. (2022, RevIN); Moreira & Muir (2017); Moskowitz et al. (2012) | Pessoa A |
| Escolher alvos e expectativas | Gu, Kelly & Xiu (2020) | Todos |
| Validar sem se enganar | Bailey et al. (2017) | Pessoa C |

## 1. Modelo geral e transfer learning

### Sirignano & Cont (2019), *Universal features of price formation in financial markets*
[arXiv](https://arxiv.org/pdf/1803.06917) · *Quantitative Finance* 19(9), 1449–1459 · **[ler]**

- **O que mostram:** um modelo de deep learning treinado com dados de muitas ações (alta frequência,
  livro de ordens) supera modelos treinados em cada ação separadamente, e continua a funcionar em
  ações que não estavam no treino.
- **Limite para nós:** os dados são intradiários e de microestrutura; o nosso painel é diário e mistura
  classes de ativos. O resultado é um argumento a favor da hipótese, não uma garantia.
- **Como aplicamos:**
  - O cenário **"ativo não visto"** do README é o equivalente direto do teste deles: treinar sem o
    ativo alvo e avaliar nele.
  - O baseline de comparação é o modelo treinado só no ativo, como no artigo.
  - Como as ações do estudo são parecidas entre si, testamos também por **classe de ativo** (treinar
    sem todos os câmbios, testar nos câmbios), para ver se a transferência vale entre mercados
    diferentes.

### *Transfer learning for financial data predictions: a systematic review*
[arXiv](https://arxiv.org/pdf/2409.17183) · **[ler]**

- **O que é:** revisão sistemática das formas de transfer learning em previsão financeira.
- **Como aplicamos:** usar como mapa para escolher as estratégias de transferência na Pessoa C
  (zero-shot, fine-tuning, congelar camadas) e para ver o que já foi tentado e quais armadilhas de
  avaliação são comuns. A leitura deve vir antes de implementar `src/transfer/`.

### *Volatility Forecasting with 1-dimensional CNNs via transfer learning*
[arXiv](https://arxiv.org/pdf/2009.05508)

- **O que é:** transfer learning aplicado à previsão de volatilidade com redes convolucionais 1D.
- **Como aplicamos:** a volatilidade foi o alvo mais previsível nos nossos testes (IC ~0,25 contra AUC
  ~0,51 na direção). É por onde o modelo de rede sequencial deve começar na Pessoa B.

### Outros para consulta
- [A Survey of Forex and Stock Price Prediction Using Deep Learning](https://arxiv.org/pdf/2103.09750):
  panorama de modelos e armadilhas.
- [Cross-sectional Stock Price Prediction using Deep Learning for Actual Investment Management](https://arxiv.org/pdf/2002.06975):
  previsão cross-sectional com deep learning.
- [Learning Universal Multi-level Market Irrationality Factors](https://arxiv.org/html/2502.04737v1):
  fator de mercado universal; ideia possível para uma feature comum a todos os ativos.

## 2. Normalização e escala por volatilidade

### Kim et al. (2022), *Reversible Instance Normalization (RevIN)*
[código oficial](https://github.com/ts-kim/RevIN) · ICLR 2022 · **[ler]**

- **O que propõe:** normalizar cada janela de entrada pela sua própria média e desvio, passar pelo
  modelo e reverter a normalização na saída, para o modelo lidar com mudanças de distribuição ao longo
  do tempo.
- **Como aplicamos:** é a variante "por janela" na comparação de normalizações. Sendo reversível,
  encaixa em redes neurais (Pessoa B/C). Ver também
  [On the Role of Reversible Instance Normalization](https://arxiv.org/html/2603.11869).
- **Ressalva:** o artigo parte de séries com tendência de nível (energia, tráfego). Em retornos, que já
  não têm nível, o ganho pode ser menor. Só o teste dirá.

### Moreira & Muir (2017), *Volatility-Managed Portfolios*
[Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513) · *Journal of Finance* 72(4)

- **O que mostram:** reduzir a exposição quando a volatilidade recente é alta melhora o Sharpe.
- **Como aplicamos:**
  - Justifica **escalar retornos pela volatilidade passada** (o nosso z-score móvel sem subtrair a
    média).
  - Dá uma regra de backtest da direção: posição proporcional ao sinal e inversa à volatilidade.

### Moskowitz, Ooi & Pedersen (2012), *Time series momentum*
[PDF](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf) · *JFE* 104(2), 228–250

- **O que mostram:** em vários mercados e classes de ativos, o retorno dos últimos 12 meses prevê o
  seguinte, com retornos escalados por volatilidade.
- **Como aplicamos:** é o desenho mais parecido com o nosso painel (várias classes, escala por
  volatilidade). Usar **momentum de 1 a 12 meses** como features e como baseline simples a bater.

### Complemento
- [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/pdf/1904.04912):
  deep learning sobre momentum em vários ativos; referência para a rede sequencial.

## 3. Expectativas e validação

### Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*
[RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276) · 33(5), 2223–2273

- **O que mostram:** métodos não lineares (árvores, redes) ganham dos lineares na previsão de
  retornos, mas os autores destacam a pouca quantidade de dados e a relação sinal/ruído muito baixa.
- **Como aplicamos:**
  - **Expectativa realista:** um AUC de direção perto de 0,5 é normal neste problema; um resultado
    muito acima disso deve ser tratado como suspeito de vazamento até prova em contrário.
  - Redes pequenas e muito regularizadas; árvores continuam sendo um bom ponto de partida.
  - Avaliar por IC e por ganho económico, e não só por erro quadrático.

### Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*
[eScholarship](https://escholarship.org/uc/item/4w1110bb) · **[ler]** (Pessoa C)

- **O que mostram:** ao testar muitas configurações, a melhor no treino tende a parecer boa só por
  acaso; a probabilidade de sobreajuste cresce com o número de configurações testadas.
- **Como aplicamos:**
  - Registrar **todas** as configurações testadas (normalizações, modelos, hiperparâmetros) em
    `docs/`, não só a vencedora.
  - Escolher hiperparâmetros só com dados de treino/validação de cada fold.
  - Desconfiar de diferenças pequenas: no nosso teste de normalização, a maior parte das diferenças
    ficou dentro do ruído entre anos.

## 4. O que já testamos e como se liga à literatura

Experimento de normalização feito antes de a equipa se dividir (o script e os CSVs foram removidos do
repositório; ficam aqui os números). Configuração: gradient boosting, painel de 20 ativos do Yahoo
Finance, walk-forward anual 2010–2026 (17 anos de teste), embargo de 5 dias, avaliação sempre em
retorno cru e log da volatilidade. Comparou retorno logarítmico cru, z-score expansivo e z-score móvel
de 252 dias.

| Alvo | Cru | Z-score expansivo | Z-score móvel |
|---|---|---|---|
| AUC da direção (0,5 = acaso) | 0,509 | 0,507 | 0,512 |
| IC da volatilidade (5 dias) | 0,259 | 0,248 | 0,238 |
| RMSE da log-volatilidade | 0,497 | 0,495 | 0,498 |
| IC do retorno (1 dia) | 0,015 | −0,047 | −0,035 |

- Direção: AUC de 0,507 a 0,512 nas três normalizações, perto do acaso, o que é coerente com Gu et al.
- Volatilidade: IC de ~0,24 a 0,26; o retorno cru foi ligeiramente melhor, mas a diferença é pequena.
- O IC negativo do retorno nas variantes com z-score é provavelmente um artefacto de reconstruir a
  previsão com a média passada (hipótese ainda por verificar).
- Conclusão: com features só de retornos passados e árvores, a normalização muda pouco. Ela deve pesar
  mais em redes neurais, ainda não testadas.

### Limites da hipótese do modelo geral (teste pendente)

O método planeado (obter ativos, normalizar, modelo geral, transfer learning) é uma **hipótese**, não
um resultado demonstrado para o nosso caso:

- **Dados diferentes dos do artigo-base.** Sirignano & Cont usam dados intradiários de livro de ordens
  de ações americanas, parecidas entre si. O nosso painel é diário e mistura ouro, câmbio, cripto,
  juros e índices, que se comportam de forma bem distinta.
- **Sinal fraco.** Gu, Kelly & Xiu alertam para a baixa relação sinal/ruído em retornos. O nosso AUC
  de ~0,51 é coerente com isso.
- **Comparação essencial ainda não feita.** Só testámos a normalização, nunca o modelo geral contra o
  modelo por ativo.
- **Leitura incompleta.** Nenhum artigo foi lido na íntegra, e a revisão de transfer learning pode
  trazer resultados mistos que ainda não vimos.

**Teste a fazer antes de investir no resto do plano**, na mesma configuração de walk-forward:

1. modelo geral (todos os ativos juntos);
2. modelo por classe de ativo;
3. modelo por ativo;
4. baselines: retorno zero, e volatilidade passada ou GARCH para o alvo de volatilidade.

Se o modelo geral não ganhar nos alvos que importam, o plano muda antes de a equipa gastar semanas nele.

**Alternativas se o modelo geral não ganhar:**

- modelos por ativo ou por classe de ativo;
- focar em volatilidade (alvo que se mostrou previsível, IC ~0,25) e usá-la para dimensionar posições,
  como em Moreira & Muir;
- baselines clássicos: GARCH para volatilidade e momentum de 12 meses (Moskowitz et al.).

## 5. Plano de aplicação

| Passo | Ação | Base na literatura | Quem |
|---|---|---|---|
| 1 | Ler Sirignano & Cont, a revisão de transfer learning e Bailey et al. | seções 1 e 3 | B, C |
| 2 | Variante de normalização: retorno / volatilidade móvel, **sem subtrair a média** | Moreira & Muir; Moskowitz et al. | A |
| 3 | Variante por janela (RevIN) em rede neural | Kim et al. | A, B |
| 4 | Features de momentum de 1 a 12 meses | Moskowitz et al. | A |
| 5 | Modelo geral e cenário "ativo não visto" | Sirignano & Cont | B, C |
| 6 | Alvo de volatilidade com rede sequencial e transfer learning | CNN + transfer learning | B, C |
| 7 | Registro de todas as configurações e avaliação com cautela de sobreajuste | Bailey et al. | C |

## 6. Referências completas

1. Sirignano, J., Cont, R. (2019). Universal features of price formation in financial markets:
   perspectives from deep learning. *Quantitative Finance*, 19(9), 1449–1459.
2. Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial
   Studies*, 33(5), 2223–2273.
3. Kim, T. et al. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against
   Distribution Shift. *ICLR*.
4. Moreira, A., Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance*, 72(4), 1611–1644.
5. Moskowitz, T., Ooi, Y. H., Pedersen, L. H. (2012). Time series momentum. *Journal of Financial
   Economics*, 104(2), 228–250.
6. Bailey, D. H., Borwein, J., López de Prado, M., Zhu, Q. J. (2017). The probability of backtest
   overfitting. *Journal of Computational Finance*.
7. Lim, B., Zohren, S., Roberts, S. (2019). Enhancing Time Series Momentum Strategies Using Deep Neural
   Networks. arXiv:1904.04912. *(autores e ano a confirmar no artigo)*
