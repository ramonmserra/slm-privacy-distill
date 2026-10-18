"""Registro dos runs de treino executados fora da API.

- `POST /treinos` recebe o `run.json` do script de treino. Exige que a
  versao de dataset exista aqui: um run sobre um dataset que a API nao
  conhece nao e' reproduzivel.
- `POST /treinos/{id}/verificar` consulta o servidor local do aluno e marca
  se o `modelo_local_id` ja esta servido -- e' o que a avaliacao vai exigir.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from app.repositories.dataset_repository import DatasetRepository
from app.repositories.treino_repository import TreinoRepository
from app.schemas.treino import RegistroDeRun, RunDeTreino, VerificacaoLocal
from app.services.treino import registro as servico

router = APIRouter()
_repositorio = TreinoRepository()
_datasets = DatasetRepository()


@router.post("", response_model=RunDeTreino, status_code=201)
async def registrar(registro: RegistroDeRun) -> RunDeTreino:
    if _datasets.carregar(registro.dataset_versao) is None:
        raise HTTPException(status_code=404, detail=f"Dataset {registro.dataset_versao!r} nao existe nesta API.")
    existente = _repositorio.por_modelo_local(registro.modelo_local_id)
    if existente is not None and existente.status == "concluido" and registro.status == "concluido":
        raise HTTPException(
            status_code=409,
            detail=f"Ja existe um run concluido servido como {registro.modelo_local_id!r} ({existente.id}). "
            "Use outro modelo_local_id: cada versao treinada e' um id novo.",
        )
    return servico.registrar(registro, _repositorio)


@router.get("", response_model=list[RunDeTreino])
async def listar() -> list[RunDeTreino]:
    return _repositorio.listar()


def _run_ou_404(run_id: str) -> RunDeTreino:
    run = _repositorio.carregar(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} nao existe.")
    return run


@router.get("/{run_id}", response_model=RunDeTreino)
async def obter(run_id: str) -> RunDeTreino:
    return _run_ou_404(run_id)


@router.post("/{run_id}/verificar", response_model=VerificacaoLocal)
async def verificar(run_id: str) -> VerificacaoLocal:
    run = _run_ou_404(run_id)
    try:
        return await servico.verificar(run, _repositorio)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Servidor local do aluno indisponivel: {exc}")
