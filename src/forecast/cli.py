"""Pipeline do forecast: estima taxas, faz backtest, projeta cenários."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtest import avaliar, decompor_erro, folha_realizada, resumo
from .simulate import Cenario, IncertezaPremissas, Simulador, resumir
from .transitions import carregar, estado_inicial, estimar

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def medianas_de_faixa(tabelas: dict[str, pd.DataFrame]) -> dict[tuple[str, int], float]:
    cargos = tabelas["dim_cargo"]
    return {
        (r["familia_cargo"], int(r["grade"])): float(r["faixa_mediana"])
        for _, r in cargos.iterrows()
    }


def montar_cenarios(cfg: dict) -> list[Cenario]:
    return [Cenario(**c) for c in cfg["cenarios"]]


def rodar_backtest(tabelas, cfg, medianas, saida: Path) -> None:
    """Estima até dez/2023 e projeta 2024, comparando com o realizado."""
    corte = pd.Timestamp("2023-12-01")
    meses = pd.date_range("2024-01-01", periods=12, freq="MS")

    taxas = estimar(tabelas, corte, cfg["janela_estimacao"])
    estado = estado_inicial(tabelas, corte)
    sim = Simulador(estado, taxas, medianas)

    # As premissas do backtest usam o que era sabido em dez/2023: dissídio
    # acordado e plano de quadro aprovado. Usar o realizado aqui seria trapaça.
    cenario = Cenario(
        nome="Backtest 2024", dissidio=0.045, crescimento_headcount=-0.03
    )
    real = folha_realizada(tabelas, meses[0], meses[-1])

    # Duas versões: uma só com o sorteio de quem sai e quem é promovido, outra
    # admitindo que as próprias premissas podem estar erradas. A comparação é
    # o resultado central deste projeto.
    aval_estoc = avaliar(
        resumir(sim.executar(cenario, meses, cfg["rodadas"])), real
    )
    aval_param = avaliar(
        resumir(sim.executar(
            cenario, meses, cfg["rodadas"], incerteza=IncertezaPremissas()
        )),
        real,
    )

    print("\n=== BACKTEST 2024 ===")
    comparacao = pd.DataFrame([
        {"fonte_de_incerteza": "só estocástica", **resumo(aval_estoc)},
        {"fonte_de_incerteza": "estocástica + paramétrica", **resumo(aval_param)},
    ])
    print(comparacao.T.to_string(header=False))

    aval = aval_param
    print("\nErro mês a mês:")
    print(aval[[
        "data_referencia", "custo_total_p50", "custo_realizado",
        "erro_pct", "dentro_da_banda",
    ]].round(4).to_string(index=False))
    print("\nDecomposição do erro:")
    print(decompor_erro(aval).to_string(index=False))

    aval.to_csv(saida / "backtest_2024.csv", index=False)
    decompor_erro(aval).to_csv(saida / "backtest_decomposicao.csv", index=False)


def rodar_grade(sim, meses, cfg, saida: Path, passo_dissidio: float,
                rodadas_por_ponto: int = 40) -> None:
    """Pré-computa uma grade de combinações para o simulador do Power BI.

    O relatório precisa responder a qualquer posição dos três controles, não
    só aos quatro cenários nomeados. Interpolar em DAX erra nas bordas, porque
    o efeito do dissídio não é linear quando há congelamento de promoção. A
    saída direta é materializar a grade: o custo é de máquina, não de usuário.

    Para manter o tempo razoável, a grade usa 40 rodadas por ponto em vez das
    500 do forecast principal. O erro de Monte Carlo com 40 rodadas fica na
    casa de 0,1% da folha, uma ordem de grandeza abaixo da largura da banda —
    que é o que de fato aparece na tela. Precisão além disso seria gasto de
    CPU invisível para o usuário.
    """
    dissidios = np.round(np.arange(0.030, 0.080 + 1e-9, passo_dissidio), 4)
    crescimentos = np.round(np.arange(-0.10, 0.20 + 1e-9, 0.05), 4)
    # Resolução fina em dissídio e crescimento, grossa em turnover. Não é
    # economia de CPU arbitrária: o próprio projeto mostrou que o regime de
    # saída responde por menos de um terço da incerteza, então gastar pontos
    # de grade nele compra pouca precisão onde o usuário olha.
    turnovers = np.round(np.arange(0.6, 1.8 + 1e-9, 0.6), 2)

    total = len(dissidios) * len(crescimentos) * len(turnovers)
    print(f"\n=== GRADE: {total} combinações x {len(meses)} meses ===")

    partes = []
    for i, d in enumerate(dissidios, 1):
        for cr in crescimentos:
            for tv in turnovers:
                cen = Cenario(
                    nome=f"d{d}_c{cr}_t{tv}", dissidio=float(d),
                    crescimento_headcount=float(cr), multiplicador_saida=float(tv),
                )
                r = resumir(sim.executar(
                    cen, meses, n_rodadas=rodadas_por_ponto,
                    incerteza=IncertezaPremissas()
                ))
                r["dissidio"] = d
                r["crescimento"] = cr
                r["mult_saida"] = tv
                partes.append(r)
        print(f"  dissídio {d:.3f} concluído ({i}/{len(dissidios)})")

    grade = pd.concat(partes, ignore_index=True).drop(columns=["cenario"])
    grade.to_parquet(saida / "grade_whatif.parquet", index=False)
    print(f"Grade gravada: {len(grade):,} linhas")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forecast de headcount e folha")
    ap.add_argument("--base", default="data_base")
    ap.add_argument("--config", default="config/cenarios.yaml")
    ap.add_argument("--saida", default="reports")
    ap.add_argument("--pular-backtest", action="store_true")
    ap.add_argument("--grade", action="store_true",
                    help="pré-computa a grade what-if para o Power BI")
    ap.add_argument("--passo-dissidio", type=float, default=0.01)
    ap.add_argument("--rodadas-grade", type=int, default=40)
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    tabelas = carregar(args.base)
    medianas = medianas_de_faixa(tabelas)

    if not args.pular_backtest:
        rodar_backtest(tabelas, cfg, medianas, saida)

    # ------------------------------------------------------------- projeção
    corte = pd.Timestamp(cfg["corte"])
    meses = pd.date_range(
        corte + pd.DateOffset(months=1), periods=cfg["horizonte_meses"], freq="MS"
    )
    taxas = estimar(tabelas, corte, cfg["janela_estimacao"])
    estado = estado_inicial(tabelas, corte)

    print("\n=== TAXAS ESTIMADAS (mensais, por grade) ===")
    print(taxas.diagnostico[[
        "pessoas_mes", "taxa_saida", "taxa_promocao", "origem",
    ]].to_string())

    sim = Simulador(estado, taxas, medianas)
    partes, comparacao_bandas = [], []
    for cenario in montar_cenarios(cfg):
        completo = resumir(sim.executar(
            cenario, meses, cfg["rodadas"], incerteza=IncertezaPremissas()
        ))
        partes.append(completo)

        somente_estoc = resumir(sim.executar(cenario, meses, cfg["rodadas"]))
        comparacao_bandas.append({
            "cenario": cenario.nome,
            "banda_estocastica": round(float(
                (somente_estoc["custo_total_p90"].sum()
                 - somente_estoc["custo_total_p10"].sum())
                / somente_estoc["custo_total_p50"].sum()
            ), 4),
            "banda_total": round(float(
                (completo["custo_total_p90"].sum()
                 - completo["custo_total_p10"].sum())
                / completo["custo_total_p50"].sum()
            ), 4),
        })
    forecast = pd.concat(partes, ignore_index=True)

    print(f"\n=== FORECAST {meses[0]:%b/%Y} a {meses[-1]:%b/%Y} ===")
    anual = forecast.groupby("cenario").agg(
        custo_12m_p50=("custo_total_p50", "sum"),
        custo_12m_p10=("custo_total_p10", "sum"),
        custo_12m_p90=("custo_total_p90", "sum"),
        headcount_final=("headcount_p50", "last"),
        saidas_12m=("saidas_medio", "sum"),
        admissoes_12m=("admissoes_medio", "sum"),
    ).round(0)
    anual["banda_pct"] = (
        (anual["custo_12m_p90"] - anual["custo_12m_p10"]) / anual["custo_12m_p50"]
    ).round(4)
    print(anual.to_string())

    print("\n=== DE ONDE VEM A INCERTEZA ===")
    cb = pd.DataFrame(comparacao_bandas)
    cb["participacao_estocastica"] = (
        cb["banda_estocastica"] / cb["banda_total"]
    ).round(3)
    print(cb.to_string(index=False))

    print("\n=== DE ONDE VEM O CRESCIMENTO DA FOLHA (cenário Base) ===")
    base = forecast[forecast["cenario"] == "Base"]
    decomp = pd.DataFrame([{
        "dissídio": base["impacto_dissidio_medio"].sum(),
        "mérito": base["impacto_merito_medio"].sum(),
        "promoção": base["impacto_promocao_medio"].sum(),
        "movimentação de quadro": base["impacto_quadro_medio"].sum(),
    }]).T.rename(columns={0: "impacto_mensal_acumulado_R$"})
    decomp["participacao"] = (
        decomp["impacto_mensal_acumulado_R$"].abs()
        / decomp["impacto_mensal_acumulado_R$"].abs().sum()
    ).round(4)
    print(decomp.round(2).to_string())

    if args.grade:
        rodar_grade(sim, meses, cfg, saida, args.passo_dissidio,
                    args.rodadas_grade)

    forecast.to_parquet(saida / "fato_forecast.parquet", index=False)
    taxas.diagnostico.to_csv(saida / "taxas_estimadas.csv")
    anual.to_csv(saida / "resumo_cenarios.csv")
    pd.DataFrame(comparacao_bandas).to_csv(saida / "fontes_incerteza.csv", index=False)
    print(f"\nArtefatos em {saida.resolve()}")


if __name__ == "__main__":
    main()
