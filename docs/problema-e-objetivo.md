# Problema e objetivo do projeto

Documento de apoio para a apresentação de quinta-feira.

## 1. O problema

Prever o preço de um ativo financeiro é difícil por três motivos:

1. **O sinal é fraco e o ruído é grande.** Os preços reagem a notícias, decisões e fluxos que não estão
   nos dados históricos. Mesmo bons modelos acertam pouco além do acaso.
2. **Cada ativo precisa de pouco histórico útil.** Um modelo treinado só com a história de um ativo vê
   poucos dados, e isso favorece o sobreajuste (decorar o passado em vez de aprender algo que se repita).
3. **O mercado muda ao longo do tempo.** Um padrão que funcionou numa década pode desaparecer na
   seguinte, então o modelo precisa ser testado sempre no futuro, nunca em dados misturados.

A prática comum é treinar um modelo separado para cada ativo. Isso repete o mesmo trabalho 20 vezes e
desperdiça o que os ativos têm em comum: reação a choques, ciclos de volatilidade, tendências e reversões.

## 2. A ideia

Em vez de um modelo por ativo, treinar **um modelo geral** com vários ativos ao mesmo tempo e depois
**transferi-lo** (transfer learning) para cada ativo. A intuição é que o modelo aprende comportamentos
que se repetem entre mercados, e que esse conhecimento ajuda quando os dados de um ativo são poucos.

```
obter ativos  ->  normalizar  ->  treinar modelo geral  ->  transferir para cada ativo  ->  avaliar
```

## 3. Objetivo

**Objetivo geral.** Verificar se um modelo geral treinado em muitos ativos prevê melhor do que modelos
treinados em cada ativo separadamente, e se esse conhecimento se transfere para ativos que o modelo
nunca viu.

**Objetivos específicos.**

- Montar um conjunto de dados com 20 ativos de relevância global: índices de ações (S&P 500, Nasdaq,
  Dow Jones, FTSE 100, DAX, Nikkei 225, Hang Seng, Ibovespa), commodities (ouro, prata, petróleo,
  cobre), câmbio (EUR/USD, USD/JPY, GBP/USD, USD/BRL), cripto (BTC, ETH), juros dos EUA a 10 anos e VIX.
- Definir uma **normalização** que coloque todos os ativos na mesma escala sem vazar informação do futuro.
- Comparar, no mesmo teste: modelo geral, modelo por classe de ativo, modelo por ativo e métodos simples
  de referência.
- Medir a **transferência** em três situações: usar o modelo geral direto, ajustá-lo ao ativo
  (fine-tuning) e testá-lo num ativo que ficou fora do treino.
- Dizer com honestidade **quando funciona e quando não funciona**.

## 4. O que vamos prever

Três alvos, de dificuldade diferente:

| Alvo | O que é | Por que importa |
|---|---|---|
| Direção | O ativo sobe ou desce no próximo dia | Base de decisões de compra e venda |
| Retorno | De quanto é a variação | Mais difícil: o sinal é muito baixo |
| Volatilidade | Quanto o preço vai oscilar nos próximos dias | Gestão de risco; tende a ser mais previsível |

## 5. Como vamos avaliar

- **Sempre no futuro.** Treinar com o passado e testar logo depois (walk-forward), repetindo ao longo
  dos anos, com uma pausa de alguns dias entre treino e teste.
- **Comparar com referências simples.** "Amanhã igual a hoje", média histórica e um modelo só do ativo.
  Um resultado só conta se ganhar delas.
- **Não se enganar com métricas bonitas.** Prever o preço de amanhã com o de hoje já dá R² perto de 1;
  por isso avaliamos retornos e direção, não o nível do preço.
- **Registrar tudo o que foi testado**, e não só a melhor configuração, para evitar achar um resultado
  bom apenas por tentar muitas vezes.

## 6. O que já sabemos

**Da literatura** (detalhes em [referencias.md](referencias.md)):

- Sirignano & Cont (2019) encontraram que um modelo treinado em centenas de ações supera os modelos
  de cada ação e funciona em ações fora do treino. Os dados deles são intradiários, de ações
  parecidas entre si.
- Gu, Kelly & Xiu (2020) mostram que métodos não lineares ganham dos lineares, mas que o sinal em
  retornos é muito fraco frente ao ruído.
- Moreira & Muir (2017) e Moskowitz et al. (2012) apoiam escalar os retornos pela volatilidade.
- Kim et al. (2022, RevIN) propõem normalizar cada janela de dados e reverter na saída.

**Do nosso primeiro teste** (20 ativos, gradient boosting, avaliação de 2010 a 2026):

- Retorno cru, z-score expansivo e z-score móvel de 252 dias deram resultados muito parecidos.
- A direção ficou perto do acaso (AUC 0,507 a 0,512, onde 0,5 é o acaso).
- A volatilidade foi bem mais previsível (correlação de ordem ~0,25 com o valor real).

## 7. Limites e riscos

- O modelo geral é uma **hipótese**: ainda não comparamos modelo geral e modelo por ativo.
- Os resultados da literatura vêm de dados e mercados diferentes do nosso painel (diário, classes de
  ativos misturadas).
- Prever direção diária pode simplesmente não ser possível com dados só de preço. Isso também é um
  resultado válido.
- Ativos correlacionados (S&P, Nasdaq, Dow) podem inflar o teste de "ativo não visto" se aparecerem
  em lados diferentes do treino.
- Li só os resumos dos artigos; os detalhes precisam ser conferidos antes de serem citados como fato.

## 8. Divisão do trabalho

| Pessoa | Frente | Entrega |
|---|---|---|
| A | Dados e features | Download dos 20 ativos, normalização, painel de dados e partição temporal para teste |
| B | Modelo geral | Modelos de referência e modelo geral, previsões salvas |
| C | Transferência e avaliação | Zero-shot, fine-tuning, ativo não visto, tabelas e conclusões |

Detalhes e formatos de dados em [README.md](../README.md).

## 9. Roteiro sugerido para a apresentação

1. **O problema** (seção 1): por que prever preços é difícil e por que um modelo por ativo desperdiça
   informação.
2. **A ideia** (seção 2): o fluxo em cinco passos.
3. **Objetivo e pergunta central** (seção 3): o modelo geral supera o modelo por ativo e se transfere?
4. **O que vamos medir e como** (seções 4 e 5): três alvos, avaliação sempre no futuro, comparação
   com referências simples.
5. **O que já sabemos** (seção 6): literatura e o primeiro teste.
6. **Riscos e próximos passos** (seções 7 e 8): a hipótese ainda está por testar e o primeiro
   experimento é a comparação modelo geral contra modelo por ativo.

**Perguntas que podem surgir:**

- *Por que não usar só o preço?* Porque o nível do preço não é comparável entre ativos e engana as
  métricas; usamos retornos.
- *E se o modelo geral não ganhar?* O plano prevê alternativas: modelos por classe de ativo ou foco em
  volatilidade, onde há mais sinal.
- *Isso daria dinheiro?* Não afirmamos isso. O objetivo é medir previsibilidade e transferência, não
  montar uma estratégia de negociação.
