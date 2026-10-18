"""Jobs de autodestilacao on-policy: criar, acompanhar, inspecionar amostras
e baixar os arquivos de treino (SFT e DPO)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.api.v1.endpoints.corpus import corpus_ativo
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.on_policy_repository import OnPolicyRepository
from app.schemas.on_policy import AmostraJulgada, JobOnPolicy, ParametrosOnPolicy
from app.services.on_policy.engine import EngineOnPolicy, novo_job
from app.services.on_policy.llm_service import LlmOnPolicyService, OnPolicyLLMService

router = APIRouter()
_repositorio = OnPolicyRepository()
_datasets = DatasetRepository()
_tarefas: set[asyncio.Task] = set()
_ARQUIVOS = ("sft", "dpo", "amostras")


def servico_on_policy(parametros: ParametrosOnPolicy) -> OnPolicyLLMService:
    return LlmOnPolicyService(parametros.aluno, temperatura=parametros.temperatura, max_tokens_resposta=parametros.max_tokens_resposta)


@router.post("/jobs", response_model=JobOnPolicy, status_code=202)
async def criar_job(parametros: ParametrosOnPolicy) -> JobOnPolicy:
    corpus = corpus_ativo()
    if _datasets.carregar(parametros.dataset_versao) is None:
        raise HTTPException(status_code=404, detail=f"Dataset {parametros.dataset_versao!r} nao existe.")
    job = novo_job(parametros)
    _repositorio.salvar_job(job)
    engine = EngineOnPolicy(servico_on_policy(parametros), corpus, _repositorio, _datasets)
    tarefa = asyncio.create_task(engine.executar(job))
    _tarefas.add(tarefa)
    tarefa.add_done_callback(_tarefas.discard)
    return job


@router.get("/jobs", response_model=list[JobOnPolicy])
async def listar() -> list[JobOnPolicy]:
    return _repositorio.listar_jobs()


def _job_ou_404(job_id: str) -> JobOnPolicy:
    job = _repositorio.carregar_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} nao existe.")
    return job


@router.get("/jobs/{job_id}", response_model=JobOnPolicy)
async def obter(job_id: str) -> JobOnPolicy:
    return _job_ou_404(job_id)


@router.get("/jobs/{job_id}/amostras", response_model=list[AmostraJulgada])
async def amostras(job_id: str, so_reprovadas: bool = Query(default=False), limite: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0)) -> list[AmostraJulgada]:
    _job_ou_404(job_id)
    todas = _repositorio.amostras(job_id)
    if so_reprovadas:
        todas = [a for a in todas if not a.aprovada]
    return todas[offset:offset + limite]


@router.get("/jobs/{job_id}/arquivo/{nome}")
async def arquivo(job_id: str, nome: str) -> FileResponse:
    _job_ou_404(job_id)
    if nome not in _ARQUIVOS:
        raise HTTPException(status_code=404, detail=f"arquivo deve ser um de {_ARQUIVOS}")
    caminho = _repositorio.caminho(job_id, nome)
    if not caminho.exists():
        raise HTTPException(status_code=404, detail="arquivo ainda nao gerado")
    return FileResponse(caminho, media_type="application/jsonl", filename=f"{job_id}-{nome}.jsonl")
