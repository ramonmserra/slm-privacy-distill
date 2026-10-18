"""Metricas de um run -- funcoes puras sobre a lista de resultados.

Separadas do engine pelo mesmo motivo do `verdict.py` do projeto de
referencia: e' o que mais muda a cada leitura dos resultados e o que menos
precisa de contexto para ser testado. Cada bloco de metricas e' uma funcao
pequena que preenche a sua parte do `MetricasDoRun`.
"""

from __future__ import annotations

from collections import Counter

from app.core.stats import intervalo_de_wilson, percentil
from app.schemas.avaliacao import MetricasDoRun, ResultadoDoItem


def decidir(resultado: ResultadoDoItem) -> ResultadoDoItem:
    """Aplica as REGRAS (codigo) sobre o veredito (juiz) e a verificacao
    deterministica. `acerto` = correto segundo o juiz. `alucinacao` =
    lastro negado pelo juiz OU artigo inexistente na prosa (que o juiz pode
    nao ver, mas o corpus ve)."""

    v = resultado.veredito
    resultado.acerto = bool(v and v.correto)
    resultado.alucinacao = bool(resultado.resposta.artigos_inexistentes) or (v is not None and v.grounded is False)
    if resultado.item.abstencao_esperada:
        resultado.abstencao_adequada = bool(v and v.abstencao and v.correto)
    return resultado


def _proporcao(numerador: int, denominador: int) -> float | None:
    return round(numerador / denominador, 4) if denominador else None


def _proporcao_com_ic(numerador: int, denominador: int) -> tuple[float | None, tuple[float, float] | None]:
    if not denominador:
        return None, None
    ic = intervalo_de_wilson(numerador, denominador)
    return _proporcao(numerador, denominador), (ic.inferior, ic.superior)


def calcular(resultados: list[ResultadoDoItem], custo_usd: float) -> MetricasDoRun:
    m = MetricasDoRun(itens=len(resultados), custo_usd=round(custo_usd, 6))
    respondidos = [r for r in resultados if r.resposta.erro is None]
    julgados = [r for r in respondidos if r.veredito is not None]
    m.respondidos = len(respondidos)
    m.julgados = len(julgados)
    _eficacia(m, julgados)
    _alucinacao(m, respondidos, julgados)
    _rubrica_e_abstencao(m, julgados)
    m.categorias_de_erro = dict(Counter(r.veredito.categoria.value for r in julgados if not r.acerto and r.veredito.categoria))
    m.por_tipo = _por_tipo(julgados)
    _custo_e_latencia(m, respondidos, custo_usd)
    return m


def _eficacia(m: MetricasDoRun, julgados: list[ResultadoDoItem]) -> None:
    m.acertos = sum(1 for r in julgados if r.acerto)
    m.acuracia, m.acuracia_ic95 = _proporcao_com_ic(m.acertos, len(julgados))


def _alucinacao(m: MetricasDoRun, respondidos: list[ResultadoDoItem], julgados: list[ResultadoDoItem]) -> None:
    m.alucinacoes = sum(1 for r in julgados if r.alucinacao)
    m.taxa_de_alucinacao, m.taxa_de_alucinacao_ic95 = _proporcao_com_ic(m.alucinacoes, len(julgados))
    m.itens_com_artigo_inexistente = sum(1 for r in respondidos if r.resposta.artigos_inexistentes)
    com_referencia = [r for r in respondidos if r.item.artigos_referencia]
    m.taxa_citou_referencia = _proporcao(sum(1 for r in com_referencia if r.resposta.citou_referencia), len(com_referencia))


def _rubrica_e_abstencao(m: MetricasDoRun, julgados: list[ResultadoDoItem]) -> None:
    com_rubrica = [r for r in julgados if r.veredito.criterios_total > 0]
    if com_rubrica:
        m.criterios_pct = round(sum(r.veredito.criterios_atendidos / r.veredito.criterios_total for r in com_rubrica) / len(com_rubrica), 4)
    esperadas = [r for r in julgados if r.item.abstencao_esperada]
    m.abstencoes_esperadas = len(esperadas)
    m.abstencoes_adequadas = sum(1 for r in esperadas if r.abstencao_adequada)


def _custo_e_latencia(m: MetricasDoRun, respondidos: list[ResultadoDoItem], custo_usd: float) -> None:
    latencias = [float(r.resposta.latencia_ms) for r in respondidos]
    m.latencia_p50_ms = percentil(latencias, 0.5)
    m.latencia_p95_ms = percentil(latencias, 0.95)
    m.tokens_entrada = sum(r.resposta.input_tokens for r in respondidos)
    m.tokens_saida = sum(r.resposta.output_tokens for r in respondidos)
    m.custo_usd_por_item = round(custo_usd / len(respondidos), 6) if respondidos else None


def _por_tipo(julgados: list[ResultadoDoItem]) -> dict[str, dict[str, float | int]]:
    saida: dict[str, dict[str, float | int]] = {}
    for tipo in sorted({r.item.tipo for r in julgados}):
        do_tipo = [r for r in julgados if r.item.tipo == tipo]
        acertos = sum(1 for r in do_tipo if r.acerto)
        saida[tipo] = {
            "itens": len(do_tipo),
            "acertos": acertos,
            "acuracia": round(acertos / len(do_tipo), 4),
            "alucinacoes": sum(1 for r in do_tipo if r.alucinacao),
        }
    return saida
