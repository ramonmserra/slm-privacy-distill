"""Consumo de LLM e diagnostico dos provedores.

- `GET /usage/llm`: agregado de tokens e custo por etapa/modelo/job.
- `GET /usage/providers`: que modelo atende cada papel, de onde veio a
  escolha e o que falta configurar -- transforma um erro de credencial num
  diagnostico acionavel.
- `POST /usage/ping/{papel}`: UMA chamada estruturada minima ao modelo do
  papel. E' a validacao ponta a ponta da Etapa 0: credencial, transporte,
  saida estruturada e telemetria. Custa centavos; o endpoint diz quanto.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import llm
from app.core.llm_usage import etapa_scope, tracker
from app.schemas.comum import Etapa

router = APIRouter()


class PapelDeModelo(BaseModel):
    papel: str
    provedor: str
    modelo: str
    origem_do_modelo: str = Field(description="`env` ({PROVEDOR}_MODEL_{PAPEL}) ou `default`")
    variavel_de_ambiente: str
    credencial_presente: bool


class DiagnosticoDeProvedores(BaseModel):
    papeis: list[PapelDeModelo]
    local_base_url: str
    avisos: list[str]


@router.get("/llm")
async def consumo_de_llm() -> dict:
    return tracker.snapshot()


@router.get("/providers", response_model=DiagnosticoDeProvedores)
async def provedores() -> DiagnosticoDeProvedores:
    papeis = []
    for tier in llm.ModelTier:
        provider = llm.provider_for_tier(tier)
        papeis.append(
            PapelDeModelo(
                papel=tier.value,
                provedor=provider,
                modelo=llm.resolve_llm_id(tier),
                origem_do_modelo=llm.tier_model_source(tier),
                variavel_de_ambiente=llm.model_env_var(provider, tier),
                credencial_presente=llm.llm_available(provider),
            )
        )
    return DiagnosticoDeProvedores(
        papeis=papeis, local_base_url=llm.local_base_url(), avisos=llm.avisos_de_configuracao()
    )


class RespostaDePing(BaseModel):
    """Saida estruturada minima que o modelo precisa devolver."""

    saudacao: str = Field(description="Uma saudacao curta em portugues")
    numero: int = Field(description="O numero 42")


class ResultadoDoPing(BaseModel):
    papel: str
    modelo: str
    latencia_ms: int
    resposta: RespostaDePing
    custo_estimado_usd: float | None


_PING_SYSTEM = "Voce responde exatamente no formato pedido, sem texto extra."
_PING_USER = "Responda com uma saudacao curta em portugues e com o numero 42."


@router.post("/ping/{papel}", response_model=ResultadoDoPing)
async def ping(papel: str) -> ResultadoDoPing:
    """Chamada REAL e paga ao modelo do papel (`teacher`, `judge` ou
    `student`). Serve para provar que a configuracao do `.env` funciona antes
    de gastar num job de verdade."""

    try:
        tier = llm.ModelTier(papel)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Papel {papel!r} desconhecido: {[t.value for t in llm.ModelTier]}")
    provider = llm.provider_for_tier(tier)
    if not llm.llm_available(provider):
        raise HTTPException(
            status_code=503,
            detail=f"Credencial do provedor {provider!r} ausente para o papel {papel!r}.",
        )
    cliente = llm.StructuredLlmClient(tier, max_tokens=256, etapa=Etapa.DIAGNOSTICO)
    antes = tracker.snapshot()["total"]["custo_usd"]
    inicio = time.perf_counter()
    with etapa_scope(Etapa.DIAGNOSTICO):
        resposta = await cliente.parse(RespostaDePing, system=_PING_SYSTEM, user=_PING_USER)
    latencia_ms = int((time.perf_counter() - inicio) * 1000)
    depois = tracker.snapshot()["total"]["custo_usd"]
    return ResultadoDoPing(
        papel=papel,
        modelo=cliente.model_id,
        latencia_ms=latencia_ms,
        resposta=resposta,
        custo_estimado_usd=round(depois - antes, 6),
    )
