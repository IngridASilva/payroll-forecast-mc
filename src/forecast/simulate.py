"""
Simulação Monte Carlo do quadro e da folha.

Por que Monte Carlo e não uma projeção determinística
-----------------------------------------------------
A projeção de folha que a maioria das empresas faz é uma linha: headcount
planejado vezes salário médio vezes um mais o dissídio. Ela produz um número
e nenhuma noção de quanto esse número pode errar.

O problema é que os dois maiores componentes da variação são estocásticos.
Quem sai e quem é promovido no mês que vem é sorteio, não decisão. Projetar a
média desses sorteios e apresentar o resultado como número único esconde a
incerteza exatamente onde ela é maior.

Aqui cada rodada sorteia saídas, promoções e méritos pessoa a pessoa, e o
resultado é uma distribuição. O que vai para o orçamento é a banda, não a
linha central.

Implementação vetorizada: dentro de uma rodada, todos os colaboradores são
processados de uma vez com operações NumPy. O laço externo percorre rodadas e
meses. Com 500 rodadas e 12 meses são 6.000 iterações sobre vetores de ~1.700
posições, o que roda em segundos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .transitions import Taxas

FAIXAS_ETARIAS = ["ate_29", "30_39", "40_49", "50_mais"]


@dataclass
class Cenario:
    """Premissas controláveis. É o que o usuário muda no simulador."""

    nome: str
    dissidio: float                  # reajuste coletivo anual
    crescimento_headcount: float     # variação anual planejada do quadro
    multiplicador_saida: float = 1.0   # choque sobre a taxa histórica
    congelamento_promocoes: bool = False
    congelamento_merito: bool = False
    salario_entrada_compa: float = 0.92


@dataclass
class IncertezaPremissas:
    """Desvio-padrão das premissas de negócio.

    Sorteadas uma vez por rodada, não por mês. Dissídio não muda de mês para
    mês: ou o acordo saiu em 4,2% ou saiu em 5,5%, e essa escolha vale o ano
    inteiro. Sortear por mês criaria uma suavização artificial que estreita a
    banda e passa uma confiança que o forecast não tem.
    """

    sd_dissidio: float = 0.008          # incerteza da negociação coletiva
    sd_crescimento: float = 0.030       # aderência ao plano de quadro
    sd_multiplicador_saida: float = 0.20  # regime de turnover do próximo ano
    ativa: bool = True


@dataclass
class ParametrosCusto:
    inss_patronal: float = 0.20
    rat: float = 0.02
    terceiros: float = 0.058
    fgts: float = 0.08
    vale_refeicao_mes: float = 882.0
    vale_transporte_mes: float = 320.0
    seguro_vida_mes: float = 34.0
    plano_saude: dict = field(default_factory=lambda: {
        "ate_29": 380.0, "30_39": 520.0, "40_49": 720.0, "50_mais": 1150.0
    })
    mes_ciclo_merito: int = 4

    @property
    def encargos(self) -> float:
        return self.inss_patronal + self.rat + self.terceiros + self.fgts


def _faixa_etaria_idx(idades: np.ndarray) -> np.ndarray:
    return np.digitize(idades, [30, 40, 50])


class Simulador:
    def __init__(
        self,
        estado: pd.DataFrame,
        taxas: Taxas,
        mediana_faixa: dict[tuple[str, int], float],
        custo: ParametrosCusto | None = None,
        seed: int = 42,
    ):
        self.estado = estado.reset_index(drop=True)
        self.taxas = taxas
        self.mediana_faixa = mediana_faixa
        self.custo = custo or ParametrosCusto()
        self.seed = seed

        # Vetores fixos da população inicial.
        self.familias = self.estado["familia_cargo"].to_numpy()
        self.areas = self.estado["id_area"].to_numpy()
        self.mes_base = self.estado["mes_data_base"].to_numpy()
        self.modelos = self.estado["modelo_trabalho"].to_numpy()
        self.grade_max_por_familia = (
            self.estado.groupby("familia_cargo")["grade"].max().to_dict()
        )

    # ------------------------------------------------------------------ custo
    def custo_mensal(self, salarios: np.ndarray, idades: np.ndarray,
                     modelos: np.ndarray, horas_extras: np.ndarray) -> dict[str, float]:
        c = self.custo
        base = salarios.sum()

        # Hora extra a 50% sobre o valor-hora, jornada de 220h. Encargos e
        # provisões incidem sobre a remuneração bruta, não sobre o salário
        # nominal. Esquecer esta rubrica foi o que produziu 4,7% de erro
        # sistemático na primeira versão do backtest — ver docs/achados.md.
        he_valor = (salarios / 220.0 * horas_extras * 1.5).sum()
        bruta = base + he_valor

        encargos = bruta * c.encargos
        prov_13 = bruta / 12 * (1 + c.encargos)
        prov_ferias = bruta * (4 / 3) / 12 * (1 + c.encargos)

        n = len(salarios)
        vr = c.vale_refeicao_mes * n
        vt = np.where(
            modelos == "Presencial", c.vale_transporte_mes,
            np.where(modelos == "Híbrido", c.vale_transporte_mes * 0.5, 0.0)
        ).sum()
        faixas = _faixa_etaria_idx(idades)
        tabela = np.array([c.plano_saude[f] for f in FAIXAS_ETARIAS])
        saude = tabela[faixas].sum()
        vida = c.seguro_vida_mes * n

        beneficios = vr + vt + saude + vida
        return {
            "headcount": n,
            "salario_base": base,
            "horas_extras": he_valor,
            "remuneracao_bruta": bruta,
            "encargos": encargos,
            "provisao_13o": prov_13,
            "provisao_ferias": prov_ferias,
            "beneficios": beneficios,
            "custo_total": bruta + encargos + prov_13 + prov_ferias + beneficios,
        }

    # ------------------------------------------------------------- uma rodada
    def _sortear_premissas(self, cenario: Cenario, rng: np.random.Generator,
                           inc: IncertezaPremissas) -> Cenario:
        if not inc.ativa:
            return cenario
        return Cenario(
            nome=cenario.nome,
            dissidio=max(0.0, rng.normal(cenario.dissidio, inc.sd_dissidio)),
            crescimento_headcount=rng.normal(
                cenario.crescimento_headcount, inc.sd_crescimento
            ),
            multiplicador_saida=max(
                0.1, rng.normal(cenario.multiplicador_saida, inc.sd_multiplicador_saida)
            ),
            congelamento_promocoes=cenario.congelamento_promocoes,
            congelamento_merito=cenario.congelamento_merito,
            salario_entrada_compa=cenario.salario_entrada_compa,
        )

    def _rodada(self, cenario: Cenario, meses: pd.DatetimeIndex,
                rng: np.random.Generator) -> list[dict]:
        sal = self.estado["salario"].to_numpy(dtype=float).copy()
        grade = self.estado["grade"].to_numpy(dtype=int).copy()
        idade = self.estado["idade"].to_numpy(dtype=float).copy()
        cargo_meses = self.estado["meses_sem_promocao"].to_numpy(dtype=float).copy()
        he = self.estado["horas_extras"].to_numpy(dtype=float).copy()
        fam = self.familias.copy()
        area = self.areas.copy()
        mesbase = self.mes_base.copy()
        modelo = self.modelos.copy()

        hc_inicial = len(sal)
        linhas = []

        for i, data in enumerate(meses, start=1):
            arrays = self.taxas.como_arrays(grade)

            # 1. Dissídio, no mês da data-base de cada sindicato.
            aplica = mesbase == data.month
            impacto_dissidio = (sal[aplica] * cenario.dissidio).sum()
            sal[aplica] *= 1 + cenario.dissidio

            # 2. Mérito, no ciclo anual.
            impacto_merito = 0.0
            if data.month == self.custo.mes_ciclo_merito and not cenario.congelamento_merito:
                recebe = rng.random(len(sal)) < arrays["merito_taxa"]
                impacto_merito = (sal[recebe] * arrays["merito_valor"][recebe]).sum()
                sal[recebe] *= 1 + arrays["merito_valor"][recebe]

            # 3. Promoções. Exige 12 meses no cargo e teto de grade da família.
            impacto_promocao = 0.0
            if not cenario.congelamento_promocoes:
                teto = np.array([self.grade_max_por_familia.get(f, 8) for f in fam])
                elegivel = (cargo_meses >= 12) & (grade < teto)
                sorteio = rng.random(len(sal)) < arrays["promocao"]
                promovidos = elegivel & sorteio
                impacto_promocao = (sal[promovidos] * arrays["salto"][promovidos]).sum()
                sal[promovidos] *= 1 + arrays["salto"][promovidos]
                grade[promovidos] += 1
                cargo_meses[promovidos] = 0

            # 4. Saídas.
            p_saida = arrays["saida"] * cenario.multiplicador_saida
            fica = rng.random(len(sal)) >= p_saida
            saidas = int((~fica).sum())
            massa_perdida = sal[~fica].sum()
            sal, grade, idade, he = sal[fica], grade[fica], idade[fica], he[fica]
            cargo_meses, fam, area = cargo_meses[fica], fam[fica], area[fica]
            mesbase, modelo = mesbase[fica], modelo[fica]

            # 5. Admissões até o quadro planejado.
            alvo = round(
                hc_inicial * (1 + cenario.crescimento_headcount) ** (i / 12)
            )
            vagas = max(alvo - len(sal), 0)
            massa_admitida = 0.0
            if vagas:
                idx = rng.integers(0, hc_inicial, vagas)
                nova_fam = self.familias[idx]
                nova_area = self.areas[idx]
                novo_grade = self.estado["grade"].to_numpy()[idx]
                medianas = np.array([
                    self.mediana_faixa.get((f, int(g)), np.median(sal))
                    for f, g in zip(nova_fam, novo_grade)
                ])
                novo_sal = medianas * cenario.salario_entrada_compa
                massa_admitida = novo_sal.sum()

                sal = np.concatenate([sal, novo_sal])
                grade = np.concatenate([grade, novo_grade])
                idade = np.concatenate([idade, self.estado["idade"].to_numpy()[idx]])
                cargo_meses = np.concatenate([cargo_meses, np.zeros(vagas)])
                fam = np.concatenate([fam, nova_fam])
                area = np.concatenate([area, nova_area])
                mesbase = np.concatenate([mesbase, self.mes_base[idx]])
                modelo = np.concatenate([modelo, self.modelos[idx]])
                he = np.concatenate([he, self.estado["horas_extras"].to_numpy()[idx]])

            cargo_meses += 1
            idade += 1 / 12

            custos = self.custo_mensal(sal, idade, modelo, he)
            custos.update({
                "data_referencia": data,
                "id_mes": int(data.strftime("%Y%m")),
                "saidas": saidas,
                "admissoes": vagas,
                "impacto_dissidio": impacto_dissidio,
                "impacto_merito": impacto_merito,
                "impacto_promocao": impacto_promocao,
                "impacto_quadro": massa_admitida - massa_perdida,
            })
            linhas.append(custos)

        return linhas

    # ------------------------------------------------------------------ saída
    def executar(self, cenario: Cenario, meses: pd.DatetimeIndex,
                 n_rodadas: int = 500,
                 incerteza: IncertezaPremissas | None = None) -> pd.DataFrame:
        """Duas fontes de incerteza, e só uma delas costuma importar.

        A estocástica é quem sai e quem é promovido, sorteado pessoa a pessoa.
        A paramétrica é o quanto as premissas de dissídio, quadro e turnover
        podem estar erradas. Com `incerteza=None` apenas a primeira é
        simulada, o que serve para medir a contribuição de cada uma.
        """
        inc = incerteza or IncertezaPremissas(ativa=False)
        rng = np.random.default_rng(self.seed)
        todas = []
        for r in range(n_rodadas):
            cen_r = self._sortear_premissas(cenario, rng, inc)
            for linha in self._rodada(cen_r, meses, rng):
                linha["dissidio_sorteado"] = cen_r.dissidio
                linha["crescimento_sorteado"] = cen_r.crescimento_headcount
                linha["rodada"] = r
                linha["cenario"] = cenario.nome
                todas.append(linha)
        return pd.DataFrame(todas)


def resumir(bruto: pd.DataFrame) -> pd.DataFrame:
    """Colapsa as rodadas em percentis. É o formato que vai para o Power BI."""
    metricas = [
        "headcount", "salario_base", "horas_extras", "remuneracao_bruta",
        "custo_total", "saidas", "admissoes",
        "impacto_dissidio", "impacto_merito", "impacto_promocao", "impacto_quadro",
    ]
    g = bruto.groupby(["cenario", "id_mes", "data_referencia"])
    saida = g[metricas].mean().add_suffix("_medio")
    for q, nome in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
        saida[f"custo_total_{nome}"] = g["custo_total"].quantile(q)
        saida[f"headcount_{nome}"] = g["headcount"].quantile(q)
    saida = saida.reset_index()
    num = saida.select_dtypes("number").columns
    return saida.round({c: 2 for c in num})
