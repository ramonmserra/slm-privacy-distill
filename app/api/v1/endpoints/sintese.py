"""Jobs de sintese: criar, acompanhar e inspecionar o que o professor gerou.

- `POST /sintese/jobs` cria o job e o executa em segundo plano no processo
  da API (uma instancia; o job morre com o processo -- aceitavel para a
  pesquisa, e o que foi gerado ate' ali ja esta' em disco).
- `GET /sintese/jobs/{id}` e' o polling: status, contadores, custo.
- `GET /sintese/jobs/{id}/exemplos` e `/descartados` sao a inspecao humana
  que valida a etapa: o grupo le os exemplos antes de gerar em volume.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.endpoints.corpus import corpus_ativo
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.sintese import ExemploDescartado, ExemploSintetico, JobDeSintese, ParametrosDeSintese
from app.services.sintese.gerador import GeradorDeSintese, novo_job
from app.services.sintese.llm_service import LlmSinteseService, SinteseLLMService

router = APIRouter()
_repositorio = SinteseRepository()
# Referencias das tasks em andamento: sem elas o GC pode recolher a task.
_tarefas: set[asyncio.Task] = set()


@lru_cache(maxsize=1)
def servico_de_sintese() -> SinteseLLMService:
    return LlmSinteseService()


@router.post("/jobs", response_model=JobDeSintese, status_code=202)
async def criar_job(parametros: ParametrosDeSintese) -> JobDeSintese:
    corpus = corpus_ativo(parametros.fonte)
    job = novo_job(parametros)
    _repositorio.salvar_job(job)
    gerador = GeradorDeSintese(servico_de_sintese(), corpus, _repositorio)
    tarefa = asyncio.create_task(gerador.executar(job))
    _tarefas.add(tarefa)
    tarefa.add_done_callback(_tarefas.discard)
    return job


@router.get("/jobs", response_model=list[JobDeSintese])
async def listar_jobs() -> list[JobDeSintese]:
    return _repositorio.listar_jobs()


def _job_ou_404(job_id: str) -> JobDeSintese:
    job = _repositorio.carregar_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} nao existe.")
    return job


@router.get("/jobs/{job_id}", response_model=JobDeSintese)
async def obter_job(job_id: str) -> JobDeSintese:
    return _job_ou_404(job_id)


@router.get("/jobs/{job_id}/exemplos", response_model=list[ExemploSintetico])
async def exemplos(
    job_id: str, limite: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0)
) -> list[ExemploSintetico]:
    _job_ou_404(job_id)
    return _repositorio.exemplos(job_id)[offset:offset + limite]


@router.get("/jobs/{job_id}/descartados", response_model=list[ExemploDescartado])
async def descartados(job_id: str) -> list[ExemploDescartado]:
    _job_ou_404(job_id)
    return _repositorio.descartados(job_id)
