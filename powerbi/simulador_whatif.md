# Simulador what-if de folha

A tela que faz este projeto ser lembrado. Três controles, um gráfico de leque
e uma cascata.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Orçamento de folha · Jan a Dez/2025                    [Diretoria ▾]    │
├──────────────────────────────────────────────────────────────────────────┤
│  Dissídio         ├───●──────┤  4,8%                                     │
│  Crescimento HC   ├─────●────┤  5,0%                                     │
│  Turnover         ├──●───────┤  1,0x histórico                           │
├──────────────────────────────────────────────────────────────────────────┤
│ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐ │
│ │ FOLHA 12M (P50)    │ │ FAIXA P10–P90      │ │ vs ORÇADO              │ │
│ │   R$ 427,6 mi      │ │ 420,1 – 436,1 mi   │ │   +2,3%   ▲            │ │
│ └────────────────────┘ └────────────────────┘ └────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│  Projeção mensal com banda de confiança                                  │
│         ░░░░░░░░░░░░░░░░░░░  P90                                         │
│      ───────────────────────  P50                                        │
│         ░░░░░░░░░░░░░░░░░░░  P10                                         │
│   ····· realizado (até dez/24)                                           │
│   jan  fev  mar  abr  mai  jun  jul  ago  set  out  nov  dez             │
├──────────────────────────────────────────────────────────────────────────┤
│  De onde vem o crescimento                                               │
│   Base ──┐                                                               │
│          ├─ Dissídio +890 mil ──┐                                        │
│          │                      ├─ Quadro +431 mil ──┐                   │
│          │                      │                    ├─ Mérito +150 mil  │
│          │                      │                    │   └─ Promoção +42 │
│                                                              = Projetado │
└──────────────────────────────────────────────────────────────────────────┘
```

## Por que os controles ficam no topo

Mesma razão do Projeto 1: a premissa aparece antes do número que ela produz.
Um CFO que vê R$ 427,6 milhões sem saber que o valor assume 4,8% de dissídio
leva o número para a reunião como se fosse fato.

## Implementação

Três tabelas desconectadas de parâmetro:

```dax
p_dissidio    = GENERATESERIES(0.030, 0.080, 0.002)
p_crescimento = GENERATESERIES(-0.10, 0.20, 0.01)
p_turnover    = GENERATESERIES(0.5, 2.0, 0.1)
```

O forecast pré-calculado tem quatro cenários nomeados. Para o simulador
responder a qualquer combinação dos três controles, há duas rotas:

**Rota 1 — grade pré-computada (recomendada).** Rode o motor sobre uma grade
de combinações e materialize o resultado. Com 26 × 31 × 16 pontos são 12.896
cenários × 12 meses, o que dá 155 mil linhas: trivial para o VertiPaq e
instantâneo na tela. O custo é de máquina, não de usuário.

```bash
forecast --grade --passo-dissidio 0.002 --saida reports
```

**Rota 2 — aproximação em DAX.** Interpola linearmente entre os cenários
nomeados. Mais leve, porém erra nas bordas, porque o efeito do dissídio sobre
a folha não é exatamente linear quando há congelamento de promoção. Só use se
o modelo precisar rodar em capacidade compartilhada apertada.

## Medidas

```dax
Folha Projetada P50 =
CALCULATE (
    SUM ( fato_forecast[custo_total_p50] ),
    TREATAS ( VALUES ( p_dissidio[p_dissidio] ), fato_forecast[dissidio] ),
    TREATAS ( VALUES ( p_crescimento[p_crescimento] ), fato_forecast[crescimento] ),
    TREATAS ( VALUES ( p_turnover[p_turnover] ), fato_forecast[mult_saida] )
)
```

```dax
Banda de Incerteza =
VAR P10 = SUM ( fato_forecast[custo_total_p10] )
VAR P90 = SUM ( fato_forecast[custo_total_p90] )
VAR P50 = SUM ( fato_forecast[custo_total_p50] )
RETURN
    DIVIDE ( P90 - P10, P50 )
```

```dax
Aviso de Banda =
VAR Largura = [Banda de Incerteza]
RETURN
    "Faixa de " & FORMAT ( Largura, "0,0%" )
    & " entre os cenários pessimista e otimista. "
    & "Dois terços dessa faixa vêm das premissas, não da variação de quem sai."
```

A última medida existe para que a leitura correta acompanhe o número em vez de
depender de alguém ter lido a documentação.

## Cascata de decomposição

Use as quatro medidas de impacto como categorias de um gráfico de cascata,
com o total do período anterior como base e o projetado como total final. As
quatro fecham exatamente com a variação, o que torna o visual auditável — e é
o argumento que convence Finanças a confiar no resto do painel.
