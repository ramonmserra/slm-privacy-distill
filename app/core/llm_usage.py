"""Telemetria de consumo de LLM por ETAPA do pipeline e por JOB.

Toda chamada de LLM reporta aqui o `usage_metadata` devolvido pelo provedor
(tokens de entrada/saida/cache) e o modelo usado. O agregado alimenta:

- `GET /usage/llm`: quanto custou gerar, filtrar e avaliar -- a comparacao
  de custo professor x aluno que o TCC precisa;
- o portao de orcamento (`app.core.llm_gate`): custo acumulado do job
  corrente, para recusar a chamada que estouraria o teto.

Duas formas de atribuir etapa e job: explicita (`etapa=` no record) ou por
contexto (`etapa_scope`/`job_scope`, ContextVars) -- um engine abre o escopo
e tudo que roda dentro herda, sem cada call site precisar saber.

Persistencia: memoria do processo (fonte imediata) espelhada em
`DATA_DIR/llm_usage.json` apos cada registro. Write-through e' aceitavel
aqui: um job de sintese faz poucas chamadas por segundo, e perder telemetria
num restart custaria mais que a escrita.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from app.core.config import data_dir, llm_budget_usd_por_job
from app.core.llm_pricing import estimate_cost_usd, is_verified
from app.repositories._file_store import escrever_json_atomico, ler_json
from app.schemas.comum import Etapa

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobDeLlm:
    """Identidade e teto do job corrente. `teto_usd=0` = sem teto."""

    job_id: str
    teto_usd: float


_etapa_ctx: ContextVar[Etapa | None] = ContextVar("etapa_ctx", default=None)
_job_ctx: ContextVar[JobDeLlm | None] = ContextVar("job_ctx", default=None)


def current_etapa() -> Etapa | None:
    return _etapa_ctx.get()


def current_job() -> JobDeLlm | None:
    return _job_ctx.get()


@contextmanager
def etapa_scope(etapa: Etapa) -> Iterator[None]:
    token = _etapa_ctx.set(etapa)
    try:
        yield
    finally:
        _etapa_ctx.reset(token)


@contextmanager
def job_scope(job_id: str, teto_usd: float | None = None) -> Iterator[JobDeLlm]:
    """Abre o escopo de um job. `teto_usd=None` usa `LLM_BUDGET_USD_POR_JOB`."""

    teto = llm_budget_usd_por_job() if teto_usd is None else teto_usd
    job = JobDeLlm(job_id=job_id, teto_usd=teto)
    token = _job_ctx.set(job)
    try:
        yield job
    finally:
        _job_ctx.reset(token)


@dataclass
class Consumo:
    chamadas: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    custo_usd: float = 0.0
    # Chamadas cujo modelo nao tem preco: o custo acima esta' SUBESTIMADO e o
    # relatorio precisa dizer isso em vez de somar zero em silencio.
    chamadas_sem_preco: int = 0

    def somar(self, outro: "Consumo") -> None:
        self.chamadas += outro.chamadas
        self.input_tokens += outro.input_tokens
        self.output_tokens += outro.output_tokens
        self.cache_creation_input_tokens += outro.cache_creation_input_tokens
        self.cache_read_input_tokens += outro.cache_read_input_tokens
        self.custo_usd += outro.custo_usd
        self.chamadas_sem_preco += outro.chamadas_sem_preco


@dataclass
class _Estado:
    # etapa -> modelo -> consumo
    por_etapa: dict[str, dict[str, Consumo]] = field(default_factory=dict)
    # job_id -> consumo: o portao le o custo; o relatorio de um run le o resto
    por_job: dict[str, Consumo] = field(default_factory=dict)


class LlmUsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._estado = _Estado()
        self._carregado = False

    def _arquivo(self):
        return data_dir() / "llm_usage.json"

    def _carregar_locked(self) -> None:
        if self._carregado:
            return
        bruto = ler_json(self._arquivo(), default={})
        try:
            self._estado = _Estado(
                por_etapa={
                    etapa: {modelo: Consumo(**c) for modelo, c in modelos.items()}
                    for etapa, modelos in bruto.get("por_etapa", {}).items()
                },
                por_job={job: Consumo(**c) for job, c in bruto.get("por_job", {}).items()},
            )
        except (TypeError, AttributeError):
            logger.warning("llm_usage.json com formato inesperado -- comecando do zero.")
            self._estado = _Estado()
        self._carregado = True

    def _gravar_locked(self) -> None:
        payload = {
            "por_etapa": {
                etapa: {modelo: asdict(c) for modelo, c in modelos.items()}
                for etapa, modelos in self._estado.por_etapa.items()
            },
            "por_job": {job: asdict(c) for job, c in self._estado.por_job.items()},
        }
        try:
            escrever_json_atomico(self._arquivo(), payload)
        except OSError:
            # Telemetria e' best-effort: nunca derruba a chamada que descreve.
            logger.warning("Nao foi possivel gravar llm_usage.json", exc_info=True)

    def record(
        self,
        model_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        etapa: Etapa | None = None,
        job_id: str | None = None,
    ) -> float | None:
        """Registra uma chamada e devolve o custo estimado dela (None = sem
        preco). Etapa e job vem do argumento ou do contexto; sem nenhum dos
        dois a etapa e' `diagnostico`."""

        etapa_efetiva = etapa or current_etapa() or Etapa.DIAGNOSTICO
        job = current_job()
        job_efetivo = job_id or (job.job_id if job else None)
        custo = estimate_cost_usd(
            model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        delta = Consumo(
            chamadas=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            custo_usd=custo or 0.0,
            chamadas_sem_preco=0 if custo is not None else 1,
        )
        with self._lock:
            self._carregar_locked()
            modelos = self._estado.por_etapa.setdefault(etapa_efetiva.value, {})
            modelos.setdefault(model_id, Consumo()).somar(delta)
            if job_efetivo:
                self._estado.por_job.setdefault(job_efetivo, Consumo()).somar(delta)
            self._gravar_locked()
        return custo

    def custo_do_job(self, job_id: str) -> float:
        with self._lock:
            self._carregar_locked()
            consumo = self._estado.por_job.get(job_id)
            return consumo.custo_usd if consumo else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Agregado para `GET /usage/llm`: por etapa, por modelo, total e a
        lista de modelos sem preco ou com preco nao conferido."""

        with self._lock:
            self._carregar_locked()
            total = Consumo()
            sem_preco: set[str] = set()
            nao_conferidos: set[str] = set()
            por_etapa: dict[str, Any] = {}
            for etapa, modelos in self._estado.por_etapa.items():
                subtotal = Consumo()
                for modelo, consumo in modelos.items():
                    subtotal.somar(consumo)
                    if consumo.chamadas_sem_preco:
                        sem_preco.add(modelo)
                    elif not is_verified(modelo):
                        nao_conferidos.add(modelo)
                total.somar(subtotal)
                por_etapa[etapa] = {
                    "modelos": {m: asdict(c) for m, c in modelos.items()},
                    "subtotal": asdict(subtotal),
                }
            return {
                "por_etapa": por_etapa,
                "por_job": {j: asdict(c) for j, c in self._estado.por_job.items()},
                "total": asdict(total),
                "modelos_sem_preco": sorted(sem_preco),
                "modelos_com_preco_nao_conferido": sorted(nao_conferidos),
                "aviso": (
                    "custo_usd e' ESTIMATIVA por tabela publica, nao fatura; "
                    "modelos em modelos_sem_preco entram com custo zero."
                ),
            }

    def limpar_para_teste(self) -> None:
        with self._lock:
            self._estado = _Estado()
            self._carregado = True


tracker = LlmUsageTracker()


def record_langchain_usage(model_id: str, message: Any, etapa: Etapa | None = None) -> float | None:
    """Le o `usage_metadata` de uma `AIMessage` do LangChain e registra.

    Anthropic e OpenAI 5.6 reportam o cache em
    `input_token_details.cache_creation`/`cache_read`. Mensagem sem
    `usage_metadata` registra a chamada com zero tokens -- a CHAMADA
    aconteceu e precisa aparecer na contagem."""

    uso = getattr(message, "usage_metadata", None) or {}
    detalhes = uso.get("input_token_details") or {}
    return tracker.record(
        model_id,
        input_tokens=int(uso.get("input_tokens") or 0),
        output_tokens=int(uso.get("output_tokens") or 0),
        cache_creation_input_tokens=int(detalhes.get("cache_creation") or 0),
        cache_read_input_tokens=int(detalhes.get("cache_read") or 0),
        etapa=etapa,
    )
