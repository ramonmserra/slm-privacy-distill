"""Execucao de LOTES concorrentes sob orcamento -- o laco que todo job com
LLM deste projeto repete (sintese, filtro, avaliacao).

Regras, iguais para todos os chamadores:

- concorrencia limitada por semaforo;
- `OrcamentoExcedido` em qualquer lote PARA o job: os lotes ainda nao
  iniciados nao saem para a rede, os que ja terminaram ficam;
- qualquer outra excecao conta como lote com erro e o job SEGUE -- um 5xx
  esporadico do provedor nao pode descartar os lotes que deram certo;
- o pos-processamento de cada lote roda sob um lock: dedup, contadores e
  persistencia sao estado compartilhado.

Extraido do gerador de sintese quando o filtro precisou do mesmo laco:
duas copias do mesmo laco e' como elas divergem.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

from app.core.erros import OrcamentoExcedido

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ResultadoDosLotes:
    total: int
    concluidos: int = 0
    com_erro: int = 0
    orcamento_excedido: str | None = None

    @property
    def todos_falharam(self) -> bool:
        return self.total > 0 and self.com_erro == self.total


class ExecutorDeLotes(Generic[T, R]):
    """`chamar(lote) -> R` faz a(s) chamada(s) de LLM de um lote (pode
    levantar); `consolidar(lote, resultado)` e' o pos-processamento, chamado
    sob o lock. `ao_progresso()` e' chamado apos cada lote (sucesso, erro ou
    orcamento) para o chamador persistir o estado."""

    def __init__(
        self,
        *,
        chamar: Callable[[list[T]], Awaitable[R]],
        consolidar: Callable[[list[T], R], None],
        ao_progresso: Callable[[], None] | None = None,
        rotulo: str = "job",
    ) -> None:
        self._chamar = chamar
        self._consolidar = consolidar
        self._ao_progresso = ao_progresso or (lambda: None)
        self._rotulo = rotulo

    async def executar(self, lotes: list[list[T]], concorrencia: int) -> ResultadoDosLotes:
        resultado = ResultadoDosLotes(total=len(lotes))
        semaforo = asyncio.Semaphore(max(1, concorrencia))
        parar = asyncio.Event()
        lock = asyncio.Lock()

        async def um_lote(lote: list[T]) -> None:
            if parar.is_set():
                return
            async with semaforo:
                if parar.is_set():
                    return
                try:
                    saida = await self._chamar(lote)
                except OrcamentoExcedido as exc:
                    parar.set()
                    async with lock:
                        resultado.orcamento_excedido = str(exc)
                        self._ao_progresso()
                    return
                except Exception:
                    logger.exception("%s: lote falhou", self._rotulo)
                    async with lock:
                        resultado.com_erro += 1
                        self._ao_progresso()
                    return
                async with lock:
                    self._consolidar(lote, saida)
                    resultado.concluidos += 1
                    self._ao_progresso()

        await asyncio.gather(*(um_lote(lote) for lote in lotes))
        return resultado


def dividir(itens: list[T], tamanho: int) -> list[list[T]]:
    tamanho = max(1, tamanho)
    return [itens[i:i + tamanho] for i in range(0, len(itens), tamanho)]
