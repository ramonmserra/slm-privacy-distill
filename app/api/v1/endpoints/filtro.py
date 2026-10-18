"""Jobs de filtro sobre exemplos de sintese: criar, acompanhar, inspecionar
decisoes (aprovadas e reprovadas, com motivos e veredito do juiz)."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.endpoints.corpus import corpus_ativo
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.filtro import DecisaoDoFiltro, JobDeFiltro, ParametrosDeFiltro
from app.services.filtro.engine import FiltroDeQualidade, novo_job
from app.services.filtro.llm_service import FiltroLLMService, LlmFiltroService

router = APIRouter()
_repositorio = FiltroRepository()
_sintese = SinteseRepository()
_tarefas: set[asyncio.Task] = set()


@lru_cache(maxsize=4)
def servico_de_filtro(temperatura: float) -> FiltroLLMService:
    return LlmFiltroService(temperatura_de_reamostragem=temperatura)


@router.post("/jobs", response_model=JobDeFiltro, status_code=202)
async def criar_job(parametros: ParametrosDeFiltro) -> JobDeFiltro:
    corpus = corpus_ativo()
    for job_sintese in parametros.jobs_de_sintese:
        if _sintese.carregar_job(job_sintese) is None:
            raise HTTPException(status_code=404, detail=f"Job de sintese {job_sintese!r} nao existe.")
    job = novo_job(parametros)
    _repositorio.salvar_job(job)
    engine = FiltroDeQualidade(servico_de_filtro(parametros.temperatura_de_reamostragem), corpus, _repositorio, _sintese)
    tarefa = asyncio.create_task(engine.executar(job))
    _tarefas.add(tarefa)
    tarefa.add_done_callback(_tarefas.discard)
    return job


@router.get("/jobs", response_model=list[JobDeFiltro])
async def listar_jobs() -> list[JobDeFiltro]:
    return _repositorio.listar_jobs()


def _job_ou_404(job_id: str) -> JobDeFiltro:
    job = _repositorio.carregar_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} nao existe.")
    return job


@router.get("/jobs/{job_id}", response_model=JobDeFiltro)
async def obter_job(job_id: str) -> JobDeFiltro:
    return _job_ou_404(job_id)


@router.get("/jobs/{job_id}/decisoes", response_model=list[DecisaoDoFiltro])
async def decisoes(
    job_id: str,
    aprovado: bool | None = Query(default=None),
    limite: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[DecisaoDoFiltro]:
    _job_ou_404(job_id)
    return _repositorio.decisoes(job_id, aprovado)[offset:offset + limite]
