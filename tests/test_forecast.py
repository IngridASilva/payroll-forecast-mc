"""Testes de invariante do forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import avaliar, folha_realizada, resumo
from forecast.cli import medianas_de_faixa
from forecast.simulate import Cenario, IncertezaPremissas, Simulador, resumir
from forecast.transitions import carregar, estado_inicial, estimar

CORTE = pd.Timestamp("2024-12-01")
MESES = pd.date_range("2025-01-01", periods=6, freq="MS")


@pytest.fixture(scope="module")
def sim():
    t = carregar("data_base")
    return Simulador(
        estado_inicial(t, CORTE), estimar(t, CORTE, 24), medianas_de_faixa(t)
    ), t


def test_estimacao_nao_usa_o_futuro():
    """Taxa estimada com corte em 2023 não pode mudar se 2024 for removido."""
    t = carregar("data_base")
    corte = pd.Timestamp("2023-12-01")
    completo = estimar(t, corte, 24)

    truncado = {k: v.copy() for k, v in t.items()}
    for nome, col in [("fato_headcount_mensal", "data_referencia"),
                      ("fato_movimentacao", "data_evento")]:
        truncado[nome] = truncado[nome][truncado[nome][col] <= corte]
    parcial = estimar(truncado, corte, 24)

    assert completo.saida == parcial.saida
    assert completo.promocao == parcial.promocao


def test_reprodutibilidade(sim):
    s, _ = sim
    a = s.executar(Cenario("t", 0.05, 0.0), MESES, n_rodadas=10)
    b = s.executar(Cenario("t", 0.05, 0.0), MESES, n_rodadas=10)
    pd.testing.assert_frame_equal(a, b)


def test_custo_sempre_acima_da_remuneracao(sim):
    s, _ = sim
    r = resumir(s.executar(Cenario("t", 0.05, 0.0), MESES, n_rodadas=20))
    assert (r["custo_total_p50"] > r["remuneracao_bruta_medio"]).all()


def test_horas_extras_entram_no_custo(sim):
    """Guarda o bug que produziu 4,7% de erro sistemático no primeiro
    backtest: encargos e provisões incidem sobre a remuneração bruta."""
    s, _ = sim
    r = resumir(s.executar(Cenario("t", 0.05, 0.0), MESES, n_rodadas=10))
    assert (r["horas_extras_medio"] > 0).all()
    assert np.allclose(
        r["remuneracao_bruta_medio"],
        r["salario_base_medio"] + r["horas_extras_medio"],
        rtol=1e-6,
    )


def test_congelamento_reduz_a_folha(sim):
    s, _ = sim
    normal = resumir(s.executar(Cenario("n", 0.05, 0.0), MESES, n_rodadas=30))
    congelado = resumir(s.executar(
        Cenario("c", 0.05, 0.0, congelamento_promocoes=True,
                congelamento_merito=True),
        MESES, n_rodadas=30,
    ))
    assert congelado["custo_total_p50"].sum() < normal["custo_total_p50"].sum()


def test_crescimento_aumenta_o_quadro(sim):
    s, _ = sim
    baixo = resumir(s.executar(Cenario("b", 0.05, 0.00), MESES, n_rodadas=30))
    alto = resumir(s.executar(Cenario("a", 0.05, 0.15), MESES, n_rodadas=30))
    assert alto["headcount_p50"].iloc[-1] > baixo["headcount_p50"].iloc[-1]


def test_incerteza_parametrica_alarga_a_banda(sim):
    """O achado central do projeto, protegido por teste."""
    s, _ = sim
    c = Cenario("t", 0.05, 0.05)
    estoc = resumir(s.executar(c, MESES, n_rodadas=60))
    total = resumir(s.executar(c, MESES, n_rodadas=60,
                               incerteza=IncertezaPremissas()))

    def largura(df):
        return (df["custo_total_p90"].sum() - df["custo_total_p10"].sum()) \
            / df["custo_total_p50"].sum()

    assert largura(total) > largura(estoc) * 1.5


def test_backtest_dentro_do_tolerado():
    t = carregar("data_base")
    corte = pd.Timestamp("2023-12-01")
    meses = pd.date_range("2024-01-01", periods=12, freq="MS")
    s = Simulador(estado_inicial(t, corte), estimar(t, corte, 24),
                  medianas_de_faixa(t))
    prev = resumir(s.executar(Cenario("bt", 0.045, -0.03), meses, n_rodadas=100,
                              incerteza=IncertezaPremissas()))
    r = resumo(avaliar(prev, folha_realizada(t, meses[0], meses[-1])))
    assert r["mape_custo"] < 0.02, f"MAPE degradou: {r['mape_custo']}"
    assert r["cobertura_p10_p90"] >= 0.70, f"cobertura: {r['cobertura_p10_p90']}"
