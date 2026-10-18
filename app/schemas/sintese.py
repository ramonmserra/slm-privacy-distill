"""Contratos da SINTESE: o que o professor produz e como um job e' descrito."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field


class TipoDeTarefa(StrEnum):
    """Taxonomia fixada ANTES de gerar: da' cobertura e permite analise por
    tipo. `SEM_RESPOSTA` existe para medir abstencao -- a alucinacao mais
    barata de medir e' o modelo responder o que o corpus nao sustenta."""

    CONCEITO = "conceito"
    BASE_LEGAL = "base_legal"
    ANALISE_DE_CASO = "analise_de_caso"
    DEVERES_DOS_AGENTES = "deveres_dos_agentes"
    DIREITOS_DO_TITULAR = "direitos_do_titular"
    INCIDENTE = "incidente"
    IDENTIFICACAO_DE_DADO = "identificacao_de_dado"
    SEM_RESPOSTA = "sem_resposta"


class Dificuldade(StrEnum):
    BASICA = "basica"
    INTERMEDIARIA = "intermediaria"
    AVANCADA = "avancada"


class ExemploSintetico(BaseModel):
    """Um exemplo gerado pelo professor, ja normalizado e anotado pelo engine.
    Nada aqui e' confianca cega: `citacoes_validas` e' verificacao
    deterministica, e o filtro (Etapa 2) decide o que vai ao treino."""

    id: str = Field(description="hash da pergunta normalizada")
    job_id: str
    tipo: TipoDeTarefa
    dificuldade: Dificuldade
    pergunta: str
    raciocinio: str = Field(description="Cadeia curta de raciocinio juridico, citando ids de dispositivo")
    resposta: str
    artigos_citados: list[str] = Field(default_factory=list, description="Ids canonicos citados pelo professor")
    artigos_citados_invalidos: list[str] = Field(default_factory=list, description="Ids que nao existem no corpus")
    citacoes_validas: bool
    contexto_ids: list[str] = Field(description="Dispositivos entregues ao professor como contexto")
    modelo: str
    versao_do_prompt: str
    criado_em: dt.datetime


class ExemploDescartado(BaseModel):
    job_id: str
    motivo: str
    pergunta: str
    tipo: TipoDeTarefa | None = None
    detalhe: str = ""


class StatusDoJob(StrEnum):
    PENDENTE = "pendente"
    EXECUTANDO = "executando"
    CONCLUIDO = "concluido"
    ORCAMENTO_EXCEDIDO = "orcamento_excedido"
    FALHOU = "falhou"


class ParametrosDeSintese(BaseModel):
    quantidade: int = Field(default=50, ge=1, le=5000, description="Exemplos a PEDIR ao professor")
    tipos: list[TipoDeTarefa] = Field(default_factory=lambda: list(TipoDeTarefa))
    fonte: str = "LGPD"
    seed: int = Field(default=42, description="Sorteio dos artigos: mesma seed, mesmo plano")
    tamanho_do_lote: int = Field(default=5, ge=1, le=20, description="Pedidos por chamada ao professor")
    concorrencia: int = Field(default=3, ge=1, le=12)
    teto_usd: float | None = Field(default=None, ge=0, description="Teto deste job; None = LLM_BUDGET_USD_POR_JOB")


class JobDeSintese(BaseModel):
    id: str
    status: StatusDoJob
    parametros: ParametrosDeSintese
    criado_em: dt.datetime
    atualizado_em: dt.datetime
    pedidos: int = 0
    lotes: int = 0
    lotes_concluidos: int = 0
    lotes_com_erro: int = 0
    gerados: int = 0
    aceitos: int = 0
    descartados: int = 0
    descartes_por_motivo: dict[str, int] = Field(default_factory=dict)
    custo_usd: float = 0.0
    modelo: str = ""
    versao_do_prompt: str = ""
    erro: str | None = None
