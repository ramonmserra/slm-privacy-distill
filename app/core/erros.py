"""Excecoes de dominio e a traducao delas em respostas HTTP.

Handlers globais (e nao `try/except` em cada endpoint) porque a regra e' da
excecao, nao da rota: um endpoint novo que use o portao do LLM ja nasce
devolvendo o status certo.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class OrcamentoExcedido(RuntimeError):
    """O job atingiria o teto de custo configurado; a chamada NAO saiu."""


class SaturacaoDoProvedor(RuntimeError):
    """Disjuntor aberto: o provedor vinha recusando por saturacao e esta
    chamada nao chegou a sair. Nao houve custo; tentar agora so' agrava."""


class StructuredOutputError(RuntimeError):
    """O modelo respondeu, mas a saida nao validou contra o schema. Distinto
    de erro de rede/credencial: aqui a chamada FOI paga."""


def registrar_handlers(app: FastAPI) -> None:
    @app.exception_handler(OrcamentoExcedido)
    async def _orcamento(_request: Request, exc: OrcamentoExcedido) -> JSONResponse:
        # 409: o estado do job (custo acumulado) conflita com o limite pedido.
        return JSONResponse(
            {"detail": str(exc), "motivo": "orcamento_do_job_excedido"}, status_code=409
        )

    @app.exception_handler(SaturacaoDoProvedor)
    async def _saturacao(_request: Request, exc: SaturacaoDoProvedor) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc), "motivo": "provedor_saturado"},
            status_code=503,
            headers={"Retry-After": "30"},
        )

    @app.exception_handler(StructuredOutputError)
    async def _saida(_request: Request, exc: StructuredOutputError) -> JSONResponse:
        # 502: o upstream respondeu algo que nao da' para usar.
        return JSONResponse(
            {"detail": str(exc), "motivo": "saida_do_modelo_invalida"}, status_code=502
        )
