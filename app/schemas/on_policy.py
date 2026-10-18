"""Contratos da AUTODESTILACAO ON-POLICY.

O aluno ja treinado gera N respostas para cada prompt do split de TREINO
de um dataset (nunca do held-out); o juiz -- o mesmo da avaliacao -- compara
cada resposta a referencia do professor. Com professor por API black-box,
nao ha logits a destilar: o que se aproveita e' a selecao.

Dois produtos, para duas receitas de treino:

- SFT on-policy (rejection sampling): a MELHOR resposta aprovada do proprio
  aluno vira o novo turno do assistente. O aluno aprende sobre a propria
  distribuicao, que e' o argumento da autodestilacao.
- DPO: pares (aprovada, reprovada) do mesmo prompt, quando existem.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.avaliacao import VereditoDeAvaliacao
from app.schemas.sintese import StatusDoJob


class ParametrosOnPolicy(BaseModel):
    dataset_versao: str = Field(min_length=1, description="Prompts vem do split TREINO desta versao")
    aluno: str = Field(pattern=r"^local:.+", description="Id do aluno ja treinado, ex.: local:<id do run>")
    amostras_por_prompt: int = Field(default=4, ge=2, le=8)
    temperatura: float = Field(default=0.8, ge=0.1, le=1.5)
    limite: int | None = Field(default=None, ge=1, description="So' os primeiros N prompts (ordem estavel)")
    exigir_lastro: bool = Field(default=True, description="Aprovar exige correto E grounded")
    fallback_para_o_professor: bool = Field(
        default=False,
        description="Prompt sem amostra aprovada mantem a resposta do professor no SFT (em vez de sair do dataset)",
    )
    max_tokens_resposta: int = Field(default=1024, ge=64, le=8192)
    concorrencia: int = Field(default=3, ge=1, le=12)
    tamanho_do_lote_do_juiz: int = Field(default=4, ge=1, le=10)
    teto_usd: float | None = Field(default=None, ge=0)


class AmostraJulgada(BaseModel):
    prompt_id: str
    indice: int
    texto: str
    veredito: VereditoDeAvaliacao | None = None
    aprovada: bool = False
    artigos_inexistentes: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    erro: str | None = None


class OrigemDoSft(StrEnum):
    ALUNO = "aluno"
    PROFESSOR = "professor"


class LinhaDoSftOnPolicy(BaseModel):
    id: str
    tipo: str
    origem: OrigemDoSft
    similaridade: float | None = None
    messages: list[dict[str, str]]


class ParDpo(BaseModel):
    id: str
    tipo: str
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
    motivo_da_rejeicao: str = ""


class JobOnPolicy(BaseModel):
    id: str
    status: StatusDoJob
    parametros: ParametrosOnPolicy
    aluno: str
    modelo_juiz: str = ""
    versao_da_medida: str = ""
    criado_em: dt.datetime
    atualizado_em: dt.datetime
    prompts: int = 0
    prompts_processados: int = 0
    amostras: int = 0
    amostras_aprovadas: int = 0
    prompts_com_aprovada: int = 0
    linhas_sft: int = 0
    linhas_sft_do_professor: int = 0
    pares_dpo: int = 0
    lotes_do_juiz: int = 0
    lotes_do_juiz_com_erro: int = 0
    custo_usd: float = 0.0
    erro: str | None = None
