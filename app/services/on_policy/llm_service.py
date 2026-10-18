"""Camada de LLM do on-policy: o ALUNO amostra com temperatura; o JUIZ e' o
mesmo da avaliacao (`JuizLlm`), com a mesma rubrica e a mesma versao."""

from __future__ import annotations

from typing import Protocol

from app.core.llm import ModelTier, RespostaDeTexto, StructuredLlmClient
from app.schemas.avaliacao import VereditoDeAvaliacao
from app.schemas.comum import Etapa
from app.services.avaliacao.llm_service import CasoDeAvaliacao, JuizLlm
from app.services.datasets.prompt_do_aluno import PROMPT_DO_ALUNO


class OnPolicyLLMService(Protocol):
    async def amostrar(self, pergunta: str) -> RespostaDeTexto: ...

    async def julgar(self, casos: list[CasoDeAvaliacao]) -> list[VereditoDeAvaliacao]: ...


class LlmOnPolicyService:
    def __init__(self, aluno: str, *, temperatura: float, max_tokens_resposta: int = 1024) -> None:
        # Temperatura > 0 de proposito: a diversidade das amostras e' o que
        # da' ao juiz algo para escolher. Sem cache de prompt: o conteudo muda.
        self._aluno = StructuredLlmClient(
            ModelTier.STUDENT, max_tokens=max_tokens_resposta, etapa=Etapa.ON_POLICY, override=aluno, temperature=temperatura,
        )
        self._juiz = JuizLlm(etapa=Etapa.ON_POLICY)

    @property
    def modelo_aluno(self) -> str:
        return self._aluno.model_id

    @property
    def modelo_juiz(self) -> str:
        return self._juiz.modelo

    async def amostrar(self, pergunta: str) -> RespostaDeTexto:
        return await self._aluno.text_detalhado(system=PROMPT_DO_ALUNO, user=pergunta)

    async def julgar(self, casos: list[CasoDeAvaliacao]) -> list[VereditoDeAvaliacao]:
        return await self._juiz.julgar(casos)
