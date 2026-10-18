"""Comparacao PAREADA de dois runs nos itens em comum.

Dois numeros agregados de runs diferentes nao se comparam: amostras e
sorteios do juiz diferem. O que se compara e' o item a item -- quais itens B
acerta e A erra, e vice-versa. McNemar exato sobre as discordancias e
bootstrap do delta dao a confianca; a `leitura` traduz os numeros.

Funcao pura sobre listas de resultados.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.stats import bootstrap_pareado, p_valor_de_mcnemar
from app.schemas.avaliacao import ComparacaoPareada, ResultadoDoItem, RunDeAvaliacao

_ALFA = 0.05


@dataclass(frozen=True)
class _Painel:
    """Os itens em comum e os acertos de cada lado neles."""

    comuns: list[str]
    acerto_a: dict[str, bool]
    acerto_b: dict[str, bool]

    @property
    def so_a(self) -> list[str]:
        return [i for i in self.comuns if self.acerto_a[i] and not self.acerto_b[i]]

    @property
    def so_b(self) -> list[str]:
        return [i for i in self.comuns if self.acerto_b[i] and not self.acerto_a[i]]

    @property
    def diferencas(self) -> list[int]:
        return [int(self.acerto_b[i]) - int(self.acerto_a[i]) for i in self.comuns]


def _painel(itens_a: list[ResultadoDoItem], itens_b: list[ResultadoDoItem]) -> _Painel:
    acerto_a = {r.item.id: r.acerto for r in itens_a if r.veredito is not None}
    acerto_b = {r.item.id: r.acerto for r in itens_b if r.veredito is not None}
    return _Painel(sorted(set(acerto_a) & set(acerto_b)), acerto_a, acerto_b)


def _alucinacoes_em(itens: list[ResultadoDoItem], comuns: list[str]) -> int:
    por_id = {r.item.id: r.alucinacao for r in itens}
    return sum(1 for i in comuns if por_id.get(i))


def _leitura(delta: float, p: float, mesma_versao: bool, em_comum: int) -> str:
    if em_comum == 0:
        return "Sem itens em comum: os runs usaram conjuntos diferentes; nada a comparar."
    ressalva = "" if mesma_versao else (
        " ATENCAO: versoes da medida diferentes (prompt do juiz, regras ou conjunto mudaram); "
        "leitura INDICATIVA, nao conta como ganho nem como regressao."
    )
    if p < _ALFA and delta > 0:
        return f"B supera A em {delta * 100:+.1f} p.p. nos itens em comum (McNemar p={p:.3f}).{ressalva}"
    if p < _ALFA and delta < 0:
        return f"B regride {delta * 100:+.1f} p.p. em relacao a A nos itens em comum (McNemar p={p:.3f}).{ressalva}"
    return f"Diferenca de {delta * 100:+.1f} p.p. dentro do ruido (McNemar p={p:.3f}): sem mudanca mensuravel.{ressalva}"


def comparar(
    run_a: RunDeAvaliacao, itens_a: list[ResultadoDoItem], run_b: RunDeAvaliacao, itens_b: list[ResultadoDoItem]
) -> ComparacaoPareada:
    painel = _painel(itens_a, itens_b)
    n = len(painel.comuns)
    acertos_a = sum(1 for i in painel.comuns if painel.acerto_a[i])
    acertos_b = sum(1 for i in painel.comuns if painel.acerto_b[i])
    delta = round((acertos_b - acertos_a) / n, 4) if n else 0.0
    ic = bootstrap_pareado(painel.diferencas)
    p = p_valor_de_mcnemar(len(painel.so_a), len(painel.so_b))
    mesma_versao = run_a.versao_da_medida == run_b.versao_da_medida
    return ComparacaoPareada(
        run_a=run_a.id, run_b=run_b.id, modelo_a=run_a.modelo, modelo_b=run_b.modelo,
        mesma_versao_da_medida=mesma_versao, itens_em_comum=n, acertos_a=acertos_a, acertos_b=acertos_b,
        so_a_acertou=len(painel.so_a), so_b_acertou=len(painel.so_b), delta_acuracia=delta,
        delta_ic95_bootstrap=(ic.inferior, ic.superior), p_mcnemar=p, melhoraram=painel.so_b, pioraram=painel.so_a,
        alucinacoes_a=_alucinacoes_em(itens_a, painel.comuns), alucinacoes_b=_alucinacoes_em(itens_b, painel.comuns),
        leitura=_leitura(delta, p, mesma_versao, n),
    )
