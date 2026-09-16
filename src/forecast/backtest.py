"""
Backtest: o forecast teria acertado?

Sem esta etapa o projeto é uma opinião bem formatada. O teste é direto:
estimar as taxas apenas com dados até dez/2023, projetar 2024 inteiro e
comparar com o que de fato aconteceu.

Duas perguntas, e a segunda importa mais:

1. O erro do ponto central é aceitável? (MAPE)
2. O realizado caiu dentro da banda P10–P90?

A segunda é a que define se a banda pode ir para o orçamento. Um forecast com
MAPE de 2% e banda que nunca contém o realizado é pior que um com MAPE de 5%
e cobertura honesta, porque o primeiro produz falsa confiança.
"""

from __future__ import annotations

import pandas as pd


def folha_realizada(tabelas: dict[str, pd.DataFrame],
                    inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    folha = tabelas["fato_folha_mensal"]
    janela = folha[
        (folha["data_referencia"] >= inicio) & (folha["data_referencia"] <= fim)
    ]
    return janela.groupby(["id_mes", "data_referencia"]).agg(
        custo_realizado=("custo_total", "sum"),
        salario_realizado=("salario", "sum"),
        headcount_realizado=("matricula", "count"),
    ).reset_index()


def avaliar(previsto: pd.DataFrame, realizado: pd.DataFrame) -> pd.DataFrame:
    m = previsto.merge(realizado, on=["id_mes", "data_referencia"])
    m["erro_pct"] = m["custo_total_p50"] / m["custo_realizado"] - 1
    m["dentro_da_banda"] = (
        (m["custo_realizado"] >= m["custo_total_p10"])
        & (m["custo_realizado"] <= m["custo_total_p90"])
    )
    m["largura_banda_pct"] = (
        (m["custo_total_p90"] - m["custo_total_p10"]) / m["custo_total_p50"]
    )
    m["erro_headcount"] = m["headcount_p50"] - m["headcount_realizado"]
    return m


def resumo(avaliacao: pd.DataFrame) -> dict:
    return {
        "meses_avaliados": len(avaliacao),
        "mape_custo": round(float(avaliacao["erro_pct"].abs().mean()), 4),
        "vies_custo": round(float(avaliacao["erro_pct"].mean()), 4),
        "erro_acumulado_12m": round(float(
            avaliacao["custo_total_p50"].sum() / avaliacao["custo_realizado"].sum() - 1
        ), 4),
        "cobertura_p10_p90": round(float(avaliacao["dentro_da_banda"].mean()), 4),
        "largura_media_banda": round(float(avaliacao["largura_banda_pct"].mean()), 4),
        "erro_medio_headcount": round(float(avaliacao["erro_headcount"].mean()), 1),
    }


def decompor_erro(avaliacao: pd.DataFrame) -> pd.DataFrame:
    """Separa o erro de custo em erro de quadro e erro de salário médio.

    Errar a folha porque o headcount veio diferente é um problema de
    planejamento. Errar porque o salário médio veio diferente é um problema de
    premissa de reajuste. As duas conversas têm donos diferentes.
    """
    df = avaliacao.copy()
    df["salario_medio_previsto"] = df["salario_base_medio"] / df["headcount_p50"]
    df["salario_medio_realizado"] = df["salario_realizado"] / df["headcount_realizado"]
    df["erro_quadro_pct"] = df["headcount_p50"] / df["headcount_realizado"] - 1
    df["erro_salario_pct"] = (
        df["salario_medio_previsto"] / df["salario_medio_realizado"] - 1
    )
    return df[[
        "id_mes", "data_referencia", "erro_pct", "erro_quadro_pct",
        "erro_salario_pct", "dentro_da_banda",
    ]].round(4)
