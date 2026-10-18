"""Roteador da v1. Toda rota exige credencial (dependencia no roteador, nao em
cada endpoint): uma rota nova nasce protegida."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.endpoints import corpus, sintese, usage
from app.core.auth import autenticar

api_router = APIRouter(dependencies=[Depends(autenticar)])
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])
api_router.include_router(corpus.router, prefix="/corpus", tags=["corpus"])
api_router.include_router(sintese.router, prefix="/sintese", tags=["sintese"])
