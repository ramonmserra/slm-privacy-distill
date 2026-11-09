"""Relatorios para o TCC, derivados dos runs persistidos.

- `GET /relatorios/condicoes?runs=a,b,c&rotulos=a:C0,b:C1`: tabela das
  condicoes (JSON); `.md` devolve Markdown para colar no texto.
- `GET /relatorios/pareadas?referencia=a&candidatos=b,c`: cada candidato
  contra a referencia, item a item.
- `GET /relatorios/custos`: tokens e custo por etapa do pipeline.
- `GET /relatorios/cobertura?jobs=a,b`: quanto de cada artigo da lei os
  exemplos aceitos da sintese cobrem; `.md` devolve Markdown.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.api.v1.endpoints.corpus import corpus_ativo
from app.core.llm_usage import tracker
from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.avaliacao import ComparacaoPareada, RunDeAvaliacao
from app.services.avaliacao import pareado
from app.services.relatorios import cobertura, condicoes

router = APIRouter()
_avaliacao = AvaliacaoRepository()
_sintese = SinteseRepository()


def _runs(ids: str) -> list[RunDeAvaliacao]:
    runs: list[RunDeAvaliacao] = []
    for run_id in [i.strip() for i in ids.split(",") if i.strip()]:
        run = _avaliacao.carregar_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} nao existe.")
        runs.append(run)
    if not runs:
        raise HTTPException(status_code=422, detail="informe ao menos um run")
    return runs


def _rotulos(texto: str | None) -> dict[str, str]:
    rotulos: dict[str, str] = {}
    for par in (texto or "").split(","):
        if ":" in par:
            run_id, rotulo = par.split(":", 1)
            rotulos[run_id.strip()] = rotulo.strip()
    return rotulos


@router.get("/condicoes", response_model=condicoes.TabelaDeCondicoes)
async def tabela_de_condicoes(runs: str = Query(description="ids separados por virgula"), rotulos: str | None = Query(default=None, description="id:rotulo,id:rotulo")) -> condicoes.TabelaDeCondicoes:
    return condicoes.tabela(_runs(runs), _rotulos(rotulos))


@router.get("/condicoes.md", response_class=PlainTextResponse)
async def tabela_de_condicoes_markdown(runs: str = Query(), rotulos: str | None = Query(default=None)) -> str:
    return condicoes.markdown(condicoes.tabela(_runs(runs), _rotulos(rotulos)))


@router.get("/pareadas", response_model=list[ComparacaoPareada])
async def comparacoes_pareadas(referencia: str = Query(), candidatos: str = Query(description="ids separados por virgula")) -> list[ComparacaoPareada]:
    run_ref = _runs(referencia)[0]
    itens_ref = _avaliacao.itens(run_ref.id)
    return [pareado.comparar(run_ref, itens_ref, cand, _avaliacao.itens(cand.id)) for cand in _runs(candidatos)]


@router.get("/custos")
async def custos() -> dict:
    snap = tracker.snapshot()
    return {
        "por_etapa": {etapa: dados["subtotal"] for etapa, dados in snap["por_etapa"].items()},
        "total": snap["total"],
        "modelos_sem_preco": snap["modelos_sem_preco"],
        "aviso": snap["aviso"],
    }


def _cobertura(jobs: str, fonte: str) -> cobertura.RelatorioDeCobertura:
    ids = [i.strip() for i in jobs.split(",") if i.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="informe ao menos um job de sintese")
    for job_id in ids:
        if _sintese.carregar_job(job_id) is None:
            raise HTTPException(status_code=404, detail=f"Job de sintese {job_id!r} nao existe.")
    corpus = corpus_ativo(fonte)
    artigos = [a.id for a in corpus.artigos(corpus.fonte)]
    exemplos = [e for j in ids for e in _sintese.exemplos(j)]
    descartados = [d for j in ids for d in _sintese.descartados(j)]
    return cobertura.relatorio(artigos, exemplos, descartados, ids)


@router.get("/cobertura", response_model=cobertura.RelatorioDeCobertura)
async def cobertura_da_lei(jobs: str = Query(description="ids de jobs de sintese separados por virgula"), fonte: str = Query(default="LGPD")) -> cobertura.RelatorioDeCobertura:
    return _cobertura(jobs, fonte)


@router.get("/cobertura.md", response_class=PlainTextResponse)
async def cobertura_da_lei_markdown(jobs: str = Query(), fonte: str = Query(default="LGPD")) -> str:
    return cobertura.markdown(_cobertura(jobs, fonte))
