"""Roteador da v1. Toda rota exige credencial (dependencia no roteador, nao em
cada endpoint): uma rota nova nasce protegida."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.endpoints import avaliacao, corpus, datasets, filtro, on_policy, relatorios, sintese, treinos, usage
from app.core.auth import autenticar

api_router = APIRouter(dependencies=[Depends(autenticar)])
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])
api_router.include_router(corpus.router, prefix="/corpus", tags=["corpus"])
api_router.include_router(sintese.router, prefix="/sintese", tags=["sintese"])
api_router.include_router(filtro.router, prefix="/filtro", tags=["filtro"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(treinos.router, prefix="/treinos", tags=["treinos"])
api_router.include_router(avaliacao.router, prefix="/avaliacao", tags=["avaliacao"])
api_router.include_router(on_policy.router, prefix="/on-policy", tags=["on-policy"])
api_router.include_router(relatorios.router, prefix="/relatorios", tags=["relatorios"])
