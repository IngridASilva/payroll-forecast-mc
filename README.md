# payroll-forecast-mc

![CI](https://github.com/IngridASilva/payroll-forecast-mc/actions/workflows/ci.yml/badge.svg)

Forecast de headcount e folha por simulação Monte Carlo, com bandas de
incerteza, cenários e backtest. Base sintética longitudinal:
[hr-synthetic-data-br](../hr-synthetic-data-br).

```bash
pip install -e ".[dev]"
forecast --base data_base --saida reports
```

---

## A pergunta

A projeção de folha que a maioria das empresas faz é uma linha: headcount
planejado vezes salário médio vezes um mais o dissídio. Produz um número e
nenhuma noção de quanto ele pode errar.

Este projeto produz uma banda, mede o erro contra o passado e mostra de onde a
incerteza realmente vem.

## Desenho

**Taxas lidas do histórico, não arbitradas.** Saída mensal, promoção mensal,
salto de promoção e mérito, estimados por grade em janela móvel de 24 meses.
Grades com menos de 30 pessoas-mês caem para a taxa agregada — sem esse recuo,
um único diretor que sai produz taxa de 8% ao mês no grade 8 e o forecast
estoura.

**Simulação pessoa a pessoa.** Cada rodada sorteia quem sai, quem é promovido
e quem recebe mérito. Vetorizada em NumPy: 500 rodadas de 12 meses sobre 1.700
pessoas rodam em segundos.

**Cenários.** Base, Contenção, Expansão e Mercado aquecido, com dissídio,
plano de quadro, multiplicador de saída e congelamento de promoção e mérito.

**Dissídio por data-base.** Cada diretoria tem seu sindicato e seu mês de
reajuste. Aplicar um reajuste único em janeiro para a empresa toda é o erro
mais comum em forecast de folha no Brasil, e distorce o perfil mensal mesmo
quando o total anual fecha.

## Backtest

Taxas estimadas apenas com dados até dez/2023, projeção de 2024 inteiro,
comparação com o realizado.

| | Só estocástica | Estocástica + paramétrica |
|---|---|---|
| MAPE do custo | 0,46% | **0,42%** |
| Viés | −0,41% | −0,36% |
| Erro acumulado 12M | −0,41% | −0,36% |
| Cobertura P10–P90 | 66,7% | **100%** |
| Largura média da banda | 1,19% | 3,55% |
| Erro médio de headcount | −2,4 pessoas | −2,6 pessoas |

Cobertura de 100% em 12 meses com banda nominal de 80% indica que a banda está
ligeiramente conservadora. Preferi assim: num orçamento, banda estreita demais
custa mais caro que banda larga demais.

## O achado central

**A incerteza sobre quem sai quase não importa. A incerteza sobre as premissas
importa muito.**

| Cenário | Banda estocástica | Banda total | Participação estocástica |
|---|---|---|---|
| Base | 1,25% | 3,73% | 33,5% |
| Contenção | 1,27% | 3,64% | 34,9% |
| Expansão | 1,35% | 3,68% | 36,7% |
| Mercado aquecido | 1,51% | 3,66% | 41,3% |

Com 1.700 pessoas, a lei dos grandes números faz o total de saídas mensais ser
notavelmente estável. Sortear indivíduo por indivíduo produz uma banda de
apenas 1,2% na folha anual.

Dois terços da incerteza real vêm de três premissas que ninguém sorteia porque
parecem decisões: quanto sairá o dissídio, se o plano de quadro será cumprido
e se o regime de turnover do próximo ano será parecido com o deste.

A consequência prática contraria a intuição de quem gosta de Monte Carlo:
**refinar o modelo de quem sai tem retorno quase nulo.** O que melhora o
forecast é negociar melhor a premissa de dissídio e medir a aderência
histórica ao plano de contratação.

As premissas são sorteadas uma vez por rodada, não por mês. Dissídio não muda
de mês para mês — ou o acordo saiu em 4,2% ou saiu em 5,5%, e essa escolha
vale o ano. Sortear mensalmente cria uma suavização artificial que estreita a
banda e passa uma confiança que o forecast não tem.

## De onde vem o crescimento da folha

Cenário Base, impacto acumulado sobre a massa mensal em 12 meses:

| Componente | Impacto | Participação |
|---|---|---|
| Dissídio | R$ 890.614 | 59% |
| Movimentação de quadro | R$ 431.242 | 28% |
| Mérito | R$ 149.754 | 10% |
| Promoção | R$ 42.474 | 3% |

Esta é a tabela que o controller pede e que quase nenhum relatório de RH
entrega. Ela responde por que a folha cresceu sem que ninguém tenha "decidido"
gastar mais: seis de cada dez reais vieram de negociação coletiva.

As quatro linhas fecham com a variação total e viram um gráfico de cascata no
Power BI.

## O erro que o backtest pegou

A primeira versão errou **4,7% de forma sistemática**, sempre para baixo, em
todos os 12 meses.

A decomposição do erro separou o problema: erro de quadro em 0,0%, erro de
salário médio em −0,3%. Headcount certo e salário médio certo, mas custo total
errado em 4,7% significa que o problema não estava na simulação — estava numa
rubrica de custo faltando.

Era hora extra. O motor projetava encargos e provisões sobre o salário nominal,
enquanto na folha real eles incidem sobre a remuneração bruta, que inclui as
horas extras. Corrigido, o MAPE caiu de 4,7% para 0,42%.

Vale registrar o que funcionou aqui: a decomposição do erro em quadro versus
salário apontou a causa em um passo. Um backtest que devolve só "MAPE de 4,7%"
não teria dito onde procurar.

Há um teste automatizado (`test_horas_extras_entram_no_custo`) que falha o
build se alguém reintroduzir o mesmo bug.

## Dashboard

O modelo semântico, as medidas DAX e o wireframe das telas estão especificados
em `powerbi/`. **O arquivo `.pbix` ainda não está neste repositório** — a
especificação veio primeiro de propósito, porque tela desenhada antes de ser
construída custa muito menos para corrigir.

| Arquivo | Conteúdo |
|---|---|
| `powerbi/simulador_whatif.md` | Simulador de três controles, medidas DAX e cascata de decomposição |

A grade what-if já está pré-computada em `reports/grade_whatif.parquet`:
1.512 combinações de dissídio, crescimento de quadro e regime de turnover.
É o que permite os três controles responderem a qualquer posição, em vez de
só aos quatro cenários nomeados.

## Estrutura

```
config/cenarios.yaml     premissas de negócio, fora do código
src/forecast/
  transitions.py         estimação das taxas com corte temporal
  simulate.py            motor Monte Carlo vetorizado + camada de custo
  backtest.py            avaliação, cobertura, decomposição do erro
  cli.py                 execução ponta a ponta
tests/                   8 testes, incluindo o do achado central
powerbi/                 simulador what-if e medidas
reports/                 fato_forecast.parquet, backtest, fontes de incerteza
```

## Saída para o Power BI

`reports/fato_forecast.parquet`, grão cenário × mês, com P10, P50 e P90 de
custo e headcount, além dos quatro componentes de impacto. Alimenta o gráfico
de leque e a cascata de decomposição.

## Ressalvas

Promoção é modelada por taxa histórica por grade, sem elegibilidade individual
por desempenho. Para forecast agregado de folha isso basta; para planejamento
de sucessão, não.

O motor não modela afastamentos, licenças nem verbas rescisórias no fluxo
mensal. Num orçamento real essas linhas entram, e a estrutura de
`custo_mensal` foi escrita para receber mais componentes sem reescrita.

As taxas assumem que o regime histórico continua. Depois de uma reestruturação
grande, reduza a janela de estimação para 12 meses e refaça o backtest antes
de confiar na banda.
