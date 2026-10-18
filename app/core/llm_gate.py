"""Portao de toda chamada de LLM: orcamento por job + disjuntor por provedor.

## Orcamento

Um job de sintese faz milhares de chamadas pagas. Sem teto, um prompt errado
descoberto no fim do job e' dinheiro perdido. O portao compara, ANTES de cada
chamada, o custo real ja acumulado pelo job (telemetria) mais a estimativa de
ENTRADA desta chamada contra o teto do job; estourou, levanta
`OrcamentoExcedido` sem sair para a rede.

Honestidade da conta: a SAIDA da chamada nao e' estimada (nao se sabe quanto
o modelo vai escrever), entao o teto pode ser ultrapassado pela saida da
ultima chamada que passou -- por um valor da ordem de centavos. E' o lado
certo do erro: subestimar a entrada deixaria passar varias chamadas.

## Disjuntor

Depois de N falhas de SATURACAO seguidas (429/5xx), as chamadas seguintes
falham na hora ate o tempo de espera passar. Falhar rapido evita a avalanche
retry -> mais 429 -> mais retry. So' erro de saturacao abre o circuito: erro
de credencial, schema ou prompt nao melhora com espera, e abrir por causa
deles derrubaria o processo inteiro por um bug de um call site.

Sem teto de tokens/minuto: um processo so', para um grupo de pesquisa, fica
longe do TPM da conta. Se um 429 recorrente aparecer, o lugar e' aqui.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.core.config import llm_circuit_cooldown_s, llm_circuit_failures
from app.core.erros import OrcamentoExcedido, SaturacaoDoProvedor
from app.core.llm_pricing import pricing_for
from app.core.llm_usage import current_job, tracker

logger = logging.getLogger(__name__)

# Aproximacao para estimar tokens de ENTRADA a partir do texto antes de
# chamar: boa o bastante para um teto de custo, nao para faturar.
CHARS_POR_TOKEN = 4


def e_saturacao(exc: BaseException) -> bool:
    """429 ou 5xx, detectado por FORMA (status_code) e nao por tipo, para nao
    amarrar este modulo aos SDKs de cada provedor."""

    codigo = getattr(exc, "status_code", None)
    if codigo is None:
        resposta = getattr(exc, "response", None)
        codigo = getattr(resposta, "status_code", None)
    if isinstance(codigo, int) and (codigo == 429 or 500 <= codigo <= 599):
        return True
    nome = type(exc).__name__.lower()
    return "ratelimit" in nome or "overloaded" in nome or "serviceunavailable" in nome


def _texto_das_mensagens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        conteudo = m.get("content", "")
        if isinstance(conteudo, str):
            total += len(conteudo)
        elif isinstance(conteudo, list):
            total += sum(len(bloco.get("text", "")) for bloco in conteudo if isinstance(bloco, dict))
    return total


def estimar_custo_de_entrada(model_id: str, messages: list[dict[str, Any]]) -> float:
    """USD estimados da ENTRADA desta chamada. Modelo sem preco estima zero --
    o relatorio ja marca o modelo como sem preco; o portao nao tem como
    inventar um numero."""

    preco = pricing_for(model_id)
    if preco is None:
        return 0.0
    tokens = _texto_das_mensagens(messages) / CHARS_POR_TOKEN
    return tokens * preco.input / 1_000_000


class _Disjuntor:
    def __init__(self, limiar: int, espera_s: float) -> None:
        self._limiar = limiar
        self._espera_s = espera_s
        self._falhas = 0
        self._aberto_ate = 0.0
        self._lock = threading.Lock()

    def exigir_passagem(self, provider: str) -> None:
        with self._lock:
            if self._aberto_ate == 0.0:
                return
            restante = self._aberto_ate - time.monotonic()
            if restante > 0:
                raise SaturacaoDoProvedor(
                    f"O provedor {provider!r} recusou {self._falhas} chamada(s) seguidas por "
                    f"saturacao; novas tentativas so' em {restante:.0f}s. Esta chamada nao foi enviada."
                )
            # Prazo vencido: meio-aberto -- a proxima chamada sonda o provedor.
            self._aberto_ate = 0.0
            self._falhas = 0

    def registrar_sucesso(self) -> None:
        with self._lock:
            self._falhas = 0

    def registrar_falha(self, provider: str) -> None:
        with self._lock:
            self._falhas += 1
            if self._falhas < self._limiar:
                return
            self._aberto_ate = time.monotonic() + self._espera_s
            logger.warning(
                "Disjuntor de %s ABERTO por %.0fs apos %d falha(s) de saturacao seguidas.",
                provider, self._espera_s, self._falhas,
            )


_MAX_PROVEDORES = 16
_disjuntores: "OrderedDict[str, _Disjuntor]" = OrderedDict()
_registro_lock = threading.Lock()


def _disjuntor(provider: str) -> _Disjuntor:
    disjuntor = _disjuntores.get(provider)
    if disjuntor is not None:
        return disjuntor
    with _registro_lock:
        disjuntor = _disjuntores.get(provider)
        if disjuntor is None:
            disjuntor = _Disjuntor(llm_circuit_failures(), llm_circuit_cooldown_s())
            _disjuntores[provider] = disjuntor
            while len(_disjuntores) > _MAX_PROVEDORES:
                _disjuntores.popitem(last=False)
    return disjuntor


def limpar_estado_para_teste() -> None:
    with _registro_lock:
        _disjuntores.clear()


def _exigir_orcamento(model_id: str, messages: list[dict[str, Any]]) -> None:
    job = current_job()
    if job is None or job.teto_usd <= 0:
        return
    acumulado = tracker.custo_do_job(job.job_id)
    previsto = acumulado + estimar_custo_de_entrada(model_id, messages)
    if previsto > job.teto_usd:
        raise OrcamentoExcedido(
            f"Job {job.job_id!r}: custo acumulado US$ {acumulado:.4f} + entrada estimada desta "
            f"chamada ultrapassaria o teto de US$ {job.teto_usd:.2f}. Chamada nao enviada. "
            "Suba LLM_BUDGET_USD_POR_JOB ou reduza o lote."
        )


@asynccontextmanager
async def portao_do_llm(model_id: str, *, messages: list[dict[str, Any]]) -> AsyncIterator[None]:
    """Envolve UMA chamada ao provedor. Ordem: disjuntor (gratis) antes do
    orcamento (le a telemetria), e nenhum dos dois faz I/O de rede."""

    provider = model_id.partition(":")[0]
    disjuntor = _disjuntor(provider)
    disjuntor.exigir_passagem(provider)
    _exigir_orcamento(model_id, messages)
    try:
        yield
    except Exception as exc:
        if e_saturacao(exc):
            disjuntor.registrar_falha(provider)
        raise
    else:
        disjuntor.registrar_sucesso()
