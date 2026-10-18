"""Contratos do DATASET: uma versao imutavel de exemplos divididos em treino
e held-out, exportada no formato de chat que o treino consome."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Split(StrEnum):
    TREINO = "treino"
    HELD_OUT = "held_out"


class ParametrosDeDataset(BaseModel):
    nome: str = Field(min_length=1, max_length=60, pattern=r"^[a-zA-Z0-9_.-]+$")
    jobs_de_filtro: list[str] = Field(default_factory=list, description="Entram so' os exemplos APROVADOS destes jobs")
    jobs_de_sintese_sem_filtro: list[str] = Field(
        default_factory=list,
        description="Entram TODOS os exemplos destes jobs de sintese (condicao experimental 'sem filtro')",
    )
    fracao_held_out: float = Field(default=0.15, ge=0.0, le=0.5)
    incluir_raciocinio: bool = Field(default=True, description="A resposta de treino traz o raciocinio antes da conclusao")

    @model_validator(mode="after")
    def _alguma_origem(self) -> "ParametrosDeDataset":
        if not self.jobs_de_filtro and not self.jobs_de_sintese_sem_filtro:
            raise ValueError("informe jobs_de_filtro ou jobs_de_sintese_sem_filtro")
        return self


class ContagemPorTipo(BaseModel):
    total: int = 0
    por_tipo: dict[str, int] = Field(default_factory=dict)


class ManifestoDoDataset(BaseModel):
    versao: str = Field(description="hash do conteudo + parametros; muda se qualquer exemplo mudar")
    nome: str
    criado_em: dt.datetime
    parametros: ParametrosDeDataset
    treino: ContagemPorTipo
    held_out: ContagemPorTipo
    duplicados_entre_origens: int = 0
    versao_do_prompt_do_aluno: str
    versoes_do_filtro: list[str] = Field(default_factory=list)
    versoes_do_prompt_de_sintese: list[str] = Field(default_factory=list)
    arquivos: dict[str, str] = Field(default_factory=dict)


class MensagemDeChat(BaseModel):
    role: str
    content: str


class LinhaDeTreino(BaseModel):
    """Uma linha do JSONL exportado. `messages` e' o que o treino le; o resto
    e' metadado para analise por tipo e para a avaliacao reconhecer o item."""

    id: str
    tipo: str
    dificuldade: str
    artigos_citados: list[str]
    messages: list[MensagemDeChat]
