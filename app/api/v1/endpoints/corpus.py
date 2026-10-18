"""Corpus juridico: ingestao da fonte versionada, consulta por id e busca.

- `POST /corpus/ingerir` le `recursos/fontes/<fonte>.txt`, chunka por
  dispositivo e persiste. Idempotente: ingerir de novo substitui.
- `GET /corpus` resume o corpus ativo.
- `GET /corpus/dispositivos/{id}` devolve um dispositivo e o artigo inteiro.
- `GET /corpus/buscar?q=` e' o retrieval (BM25) que a sintese usa.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.repositories.corpus_repository import CorpusRepository
from app.schemas.corpus import Dispositivo, ResultadoDeBusca, ResumoDoCorpus
from app.services.corpus import ingestao

router = APIRouter()
_repositorio = CorpusRepository()


def corpus_ativo(fonte: str = "LGPD") -> ingestao.Corpus:
    try:
        return ingestao.registro.obter(_repositorio, fonte)
    except ingestao.CorpusVazio as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class PedidoDeIngestao(BaseModel):
    fonte: str = "LGPD"


@router.post("/ingerir", response_model=ResumoDoCorpus)
async def ingerir(pedido: PedidoDeIngestao) -> ResumoDoCorpus:
    try:
        corpus = ingestao.ingerir(pedido.fonte, _repositorio)
    except ingestao.FonteDesconhecida as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ingestao.CorpusInvalido as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    ingestao.registro.definir(corpus)
    return corpus.resumo()


@router.get("", response_model=ResumoDoCorpus)
async def resumo() -> ResumoDoCorpus:
    return corpus_ativo().resumo()


class DispositivoComContexto(BaseModel):
    dispositivo: Dispositivo
    artigo_completo: str


@router.get("/dispositivos/{id_:path}", response_model=DispositivoComContexto)
async def dispositivo(id_: str) -> DispositivoComContexto:
    corpus = corpus_ativo()
    d = corpus.dispositivo(id_)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Dispositivo {id_!r} nao existe no corpus.")
    return DispositivoComContexto(dispositivo=d, artigo_completo=corpus.texto_do_artigo(d.id))


@router.get("/buscar", response_model=list[ResultadoDeBusca])
async def buscar(
    q: str = Query(min_length=2, max_length=500), k: int = Query(default=5, ge=1, le=20)
) -> list[ResultadoDeBusca]:
    return [
        ResultadoDeBusca(artigo_id=a.id, rotulo=a.rotulo, score=a.score, trecho=a.trecho)
        for a in corpus_ativo().buscar(q, k)
    ]
