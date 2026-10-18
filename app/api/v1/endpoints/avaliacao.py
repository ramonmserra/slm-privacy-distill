"""Runs de avaliacao: criar (segundo plano), acompanhar, inspecionar itens e
comparar dois runs item a item."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.endpoints.corpus import corpus_ativo
from app.core.llm import UnknownProviderError
from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.repositories.dataset_repository import DatasetRepository
import datetime as dt

from app.schemas.avaliacao import (
    Anotacao,
    AnotacaoRegistrada,
    ComparacaoPareada,
    ParametrosDeAvaliacao,
    ResultadoDoItem,
    ResumoDeCalibracao,
    RunDeAvaliacao,
)
from app.services.avaliacao import calibracao, pareado
from app.services.avaliacao.engine import Avaliador, novo_run
from app.services.avaliacao.llm_service import AvaliacaoLLMService, LlmAvaliacaoService

router = APIRouter()
_repositorio = AvaliacaoRepository()
_datasets = DatasetRepository()
_tarefas: set[asyncio.Task] = set()


def servico_de_avaliacao(modelo: str, max_tokens_resposta: int) -> AvaliacaoLLMService:
    return LlmAvaliacaoService(modelo, max_tokens_resposta=max_tokens_resposta)


@router.post("/runs", response_model=RunDeAvaliacao, status_code=202)
async def criar_run(parametros: ParametrosDeAvaliacao) -> RunDeAvaliacao:
    corpus = corpus_ativo()
    try:
        run = novo_run(parametros)
    except UnknownProviderError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if parametros.conjunto.value == "held_out" and (not parametros.dataset_versao or _datasets.carregar(parametros.dataset_versao) is None):
        raise HTTPException(status_code=404, detail=f"Dataset {parametros.dataset_versao!r} nao existe (obrigatorio para held_out).")
    _repositorio.salvar_run(run)
    avaliador = Avaliador(servico_de_avaliacao(run.modelo, parametros.max_tokens_resposta), corpus, _repositorio, _datasets)
    tarefa = asyncio.create_task(avaliador.executar(run))
    _tarefas.add(tarefa)
    tarefa.add_done_callback(_tarefas.discard)
    return run


@router.get("/runs", response_model=list[RunDeAvaliacao])
async def listar_runs() -> list[RunDeAvaliacao]:
    return _repositorio.listar_runs()


def _run_ou_404(run_id: str) -> RunDeAvaliacao:
    run = _repositorio.carregar_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} nao existe.")
    return run


@router.get("/runs/{run_id}", response_model=RunDeAvaliacao)
async def obter_run(run_id: str) -> RunDeAvaliacao:
    return _run_ou_404(run_id)


@router.get("/runs/{run_id}/itens", response_model=list[ResultadoDoItem])
async def itens(
    run_id: str,
    so_erros: bool = Query(default=False, description="So' itens sem acerto ou com alucinacao"),
    limite: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[ResultadoDoItem]:
    _run_ou_404(run_id)
    todos = _repositorio.itens(run_id)
    if so_erros:
        todos = [r for r in todos if not r.acerto or r.alucinacao]
    return todos[offset:offset + limite]


@router.post("/runs/{run_id}/anotacoes", response_model=AnotacaoRegistrada, status_code=201)
async def anotar(run_id: str, anotacao: Anotacao) -> AnotacaoRegistrada:
    """Julgamento humano de um item (amostra de calibracao do juiz). Reanotar
    o mesmo item pelo mesmo anotador substitui a anotacao anterior."""

    _run_ou_404(run_id)
    if not any(r.item.id == anotacao.item_id for r in _repositorio.itens(run_id)):
        raise HTTPException(status_code=404, detail=f"Item {anotacao.item_id!r} nao existe neste run.")
    registrada = AnotacaoRegistrada(run_id=run_id, criado_em=dt.datetime.now(dt.timezone.utc), **anotacao.model_dump())
    _repositorio.acrescentar_anotacao(registrada)
    return registrada


@router.get("/runs/{run_id}/anotacoes", response_model=list[AnotacaoRegistrada])
async def anotacoes(run_id: str, anotador: str | None = Query(default=None)) -> list[AnotacaoRegistrada]:
    """Anotacoes do run (todas ou de um anotador), para a tela de anotacao
    pular o que a pessoa ja julgou."""

    _run_ou_404(run_id)
    todas = _repositorio.anotacoes(run_id)
    return [a for a in todas if anotador is None or a.anotador == anotador]


@router.get("/runs/{run_id}/calibracao", response_model=ResumoDeCalibracao)
async def calibracao_do_juiz(run_id: str) -> ResumoDeCalibracao:
    _run_ou_404(run_id)
    return calibracao.resumir(run_id, _repositorio.itens(run_id), _repositorio.anotacoes(run_id))


@router.get("/comparar", response_model=ComparacaoPareada)
async def comparar(a: str = Query(description="run de referencia"), b: str = Query(description="run candidato")) -> ComparacaoPareada:
    run_a, run_b = _run_ou_404(a), _run_ou_404(b)
    return pareado.comparar(run_a, _repositorio.itens(a), run_b, _repositorio.itens(b))
