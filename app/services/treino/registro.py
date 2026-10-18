"""Registro de runs e verificacao no servidor local.

A verificacao consulta `GET {LOCAL_LLM_BASE_URL}/models` (OpenAI-compatible;
o Ollama responde com os nomes criados). O nome no Ollama costuma vir com
tag (`<nome>:latest`); a comparacao aceita com e sem.
"""

from __future__ import annotations

import datetime as dt
import uuid

import httpx

from app.core import llm
from app.repositories.treino_repository import TreinoRepository
from app.schemas.treino import RegistroDeRun, RunDeTreino, VerificacaoLocal


def _agora() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def registrar(registro: RegistroDeRun, repositorio: TreinoRepository) -> RunDeTreino:
    run = RunDeTreino(id=uuid.uuid4().hex[:12], criado_em=_agora(), **registro.model_dump())
    repositorio.salvar(run)
    return run


async def modelos_servidos(base_url: str | None = None) -> list[str]:
    url = (base_url or llm.local_base_url()).rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=5.0) as client:
        resposta = await client.get(url, headers={"Authorization": f"Bearer {llm.credencial(llm.PROVIDER_LOCAL)}"})
    resposta.raise_for_status()
    dados = resposta.json().get("data", [])
    return sorted(str(m.get("id", "")) for m in dados if m.get("id"))


def esta_servido(modelo_local_id: str, servidos: list[str]) -> bool:
    _, nome = llm.split_llm_id(modelo_local_id)
    candidatos = {nome, f"{nome}:latest"}
    return any(s in candidatos or s.split(":")[0] == nome for s in servidos)


async def verificar(run: RunDeTreino, repositorio: TreinoRepository) -> VerificacaoLocal:
    servidos = await modelos_servidos()
    disponivel = esta_servido(run.modelo_local_id, servidos)
    run.disponivel_localmente = disponivel
    run.verificado_em = _agora()
    repositorio.salvar(run)
    return VerificacaoLocal(
        modelo_local_id=run.modelo_local_id, disponivel=disponivel, modelos_servidos=servidos, base_url=llm.local_base_url()
    )
