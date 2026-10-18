"""Versoes de dataset: construir a partir de jobs de filtro/sintese, listar,
ler o manifesto, amostrar linhas e baixar os arquivos JSONL para o treino."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.repositories.dataset_repository import DatasetRepository
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.dataset import ManifestoDoDataset, ParametrosDeDataset, Split
from app.services.datasets.construtor import ConstrutorDeDataset, OrigemDesconhecida, OrigemVazia

router = APIRouter()
_repositorio = DatasetRepository()


def _construtor() -> ConstrutorDeDataset:
    return ConstrutorDeDataset(_repositorio, FiltroRepository(), SinteseRepository())


@router.post("", response_model=ManifestoDoDataset, status_code=201)
async def construir(parametros: ParametrosDeDataset) -> ManifestoDoDataset:
    try:
        return _construtor().construir(parametros)
    except OrigemDesconhecida as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except OrigemVazia as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("", response_model=list[ManifestoDoDataset])
async def listar() -> list[ManifestoDoDataset]:
    return _repositorio.listar()


def _manifesto_ou_404(versao: str) -> ManifestoDoDataset:
    manifesto = _repositorio.carregar(versao)
    if manifesto is None:
        raise HTTPException(status_code=404, detail=f"Dataset {versao!r} nao existe.")
    return manifesto


@router.get("/{versao}", response_model=ManifestoDoDataset)
async def manifesto(versao: str) -> ManifestoDoDataset:
    return _manifesto_ou_404(versao)


@router.get("/{versao}/amostra")
async def amostra(versao: str, split: Split = Split.TREINO, limite: int = Query(default=5, ge=1, le=100)) -> list[dict]:
    _manifesto_ou_404(versao)
    return _repositorio.linhas(versao, split)[:limite]


@router.get("/{versao}/arquivo/{split}")
async def arquivo(versao: str, split: Split) -> FileResponse:
    _manifesto_ou_404(versao)
    caminho = _repositorio.caminho_do_split(versao, split)
    if not caminho.exists():
        raise HTTPException(status_code=404, detail="arquivo do split nao encontrado")
    return FileResponse(caminho, media_type="application/jsonl", filename=f"{versao}-{split.value}.jsonl")
