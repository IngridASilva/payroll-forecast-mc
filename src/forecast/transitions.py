"""
Estimação das taxas de transição a partir do histórico.

O forecast não inventa premissa de movimentação: ele lê do que a empresa fez.
Quatro taxas, todas por grade, estimadas em janela móvel:

    saída mensal          desligamentos / pessoas-mês
    promoção mensal       promoções / pessoas-mês elegíveis
    salto de promoção     variação salarial mediana na promoção
    mérito                taxa de concessão e valor mediano

Janela padrão de 24 meses. Janela curta capta o regime atual mas tem
variância alta em grades pequenas; janela longa é estável mas carrega o
regime antigo. Vinte e quatro meses é o meio-termo defensável, e o parâmetro
está exposto porque a escolha certa depende do quanto a empresa mudou.

Grades com menos de 30 pessoas-mês na janela caem para a taxa agregada. Sem
esse recuo, um diretor que saiu produz taxa de saída de 8% ao mês no grade 8
e o forecast estoura.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MINIMO_PESSOAS_MES = 30


@dataclass
class Taxas:
    """Taxas mensais por grade, mais os agregados de recuo."""

    saida: dict[int, float]
    promocao: dict[int, float]
    salto_promocao: dict[int, float]
    merito_taxa: dict[int, float]
    merito_valor: dict[int, float]
    saida_agregada: float
    promocao_agregada: float
    salto_agregado: float
    diagnostico: pd.DataFrame

    def como_arrays(self, grades: np.ndarray) -> dict[str, np.ndarray]:
        """Converte para vetores alinhados ao array de grades da simulação."""
        def mapear(d: dict[int, float], padrao: float) -> np.ndarray:
            return np.array([d.get(int(g), padrao) for g in grades])

        return {
            "saida": mapear(self.saida, self.saida_agregada),
            "promocao": mapear(self.promocao, self.promocao_agregada),
            "salto": mapear(self.salto_promocao, self.salto_agregado),
            "merito_taxa": mapear(self.merito_taxa, 0.0),
            "merito_valor": mapear(self.merito_valor, 0.0),
        }


def carregar(dir_base: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(dir_base)
    return {
        n: pd.read_parquet(d / f"{n}.parquet")
        for n in ["fato_headcount_mensal", "fato_movimentacao",
                  "dim_colaborador", "dim_area", "dim_cargo", "fato_folha_mensal"]
    }


def estimar(
    tabelas: dict[str, pd.DataFrame],
    corte: pd.Timestamp,
    janela_meses: int = 24,
) -> Taxas:
    """Estima taxas usando apenas dados até `corte`.

    O corte é o que permite backtest honesto: nenhuma taxa pode ter sido
    estimada com informação posterior ao início do horizonte projetado.
    """
    snaps = tabelas["fato_headcount_mensal"]
    movs = tabelas["fato_movimentacao"]

    inicio = corte - pd.DateOffset(months=janela_meses)
    snaps_j = snaps[
        (snaps["data_referencia"] > inicio) & (snaps["data_referencia"] <= corte)
    ]
    movs_j = movs[
        (movs["data_evento"] > inicio) & (movs["data_evento"] <= corte)
    ]

    # Denominador comum: pessoas-mês por grade.
    expostos = snaps_j.groupby("grade").size().rename("pessoas_mes")

    desligamentos = (
        movs_j[movs_j["tipo_evento"] == "Desligamento"]
        .groupby("grade").size().rename("desligamentos")
    )
    promocoes_ev = movs_j[movs_j["tipo_evento"] == "Promoção"]
    promocoes = promocoes_ev.groupby("grade").size().rename("promocoes")

    # A promoção registra o grade de destino; o risco pertence ao de origem.
    promocoes.index = promocoes.index - 1
    promocoes = promocoes.groupby(level=0).sum()

    meritos_ev = movs_j[movs_j["tipo_evento"] == "Mérito"]
    meritos = meritos_ev.groupby("grade").size().rename("meritos")

    diag = pd.concat([expostos, desligamentos, promocoes, meritos], axis=1).fillna(0)
    diag["taxa_saida"] = diag["desligamentos"] / diag["pessoas_mes"]
    diag["taxa_promocao"] = diag["promocoes"] / diag["pessoas_mes"]

    # Mérito é anual: a taxa relevante é por ciclo, não por pessoa-mês.
    ciclos = max(janela_meses / 12, 1)
    diag["taxa_merito_ciclo"] = diag["meritos"] / (diag["pessoas_mes"] / 12) / ciclos

    diag["amostra_suficiente"] = diag["pessoas_mes"] >= MINIMO_PESSOAS_MES
    diag["origem"] = np.where(diag["amostra_suficiente"], "própria", "agregada")

    salto = promocoes_ev.groupby(promocoes_ev["grade"] - 1)["variacao_pct"].median()
    valor_merito = meritos_ev.groupby("grade")["variacao_pct"].median()

    def dicionario(serie: pd.Series, coluna_ok: pd.Series) -> dict[int, float]:
        return {
            int(g): float(v) for g, v in serie.items()
            if coluna_ok.get(g, False) and np.isfinite(v)
        }

    ok = diag["amostra_suficiente"]
    total_pm = diag["pessoas_mes"].sum()

    return Taxas(
        saida=dicionario(diag["taxa_saida"], ok),
        promocao=dicionario(diag["taxa_promocao"], ok),
        salto_promocao=dicionario(salto, ok),
        merito_taxa=dicionario(diag["taxa_merito_ciclo"], ok),
        merito_valor=dicionario(valor_merito, ok),
        saida_agregada=float(diag["desligamentos"].sum() / total_pm),
        promocao_agregada=float(diag["promocoes"].sum() / total_pm),
        salto_agregado=float(promocoes_ev["variacao_pct"].median()),
        diagnostico=diag.round(5),
    )


def estado_inicial(
    tabelas: dict[str, pd.DataFrame], corte: pd.Timestamp
) -> pd.DataFrame:
    """Foto do quadro ativo no mês de corte, enriquecida para a simulação."""
    snaps = tabelas["fato_headcount_mensal"]
    areas = tabelas["dim_area"]

    atual = snaps[snaps["data_referencia"] == corte].copy()
    atual = atual.merge(
        areas[["id_area", "diretoria", "mes_data_base", "criticidade"]],
        on="id_area", how="left",
    )
    return atual[[
        "matricula", "id_area", "diretoria", "criticidade", "familia_cargo",
        "grade", "salario", "modelo_trabalho", "idade", "meses_de_casa",
        "meses_sem_promocao", "mes_data_base", "horas_extras",
    ]].reset_index(drop=True)
