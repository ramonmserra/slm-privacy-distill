"""Contratos do FILTRO: o que decide se um exemplo sintetico entra no treino.

Tres filtros independentes, cada um com motivo proprio de reprovacao:

- CITACAO (deterministico): ids citados existem; artigos mencionados na
  prosa existem; tipo que exige fundamentacao tem ao menos uma citacao;
- INCERTEZA do professor (self-consistency): k reamostragens da resposta
  com temperatura > 0; o JUIZ diz quantas sao equivalentes a original;
- QUALIDADE (juiz, familia diferente do professor): rubrica de correcao,
  lastro, completude, abstencao e dados pessoais.

Nada e' apagado: a decisao fica gravada com `aprovado`, `score_incerteza` e
`motivos`, e o dataset escolhe o que usar -- inclusive "sem filtro".
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from app.schemas.sintese import StatusDoJob, TipoDeTarefa

MOTIVO_CITACAO_INVALIDA = "citacao_invalida"
MOTIVO_ARTIGO_INEXISTENTE_NO_TEXTO = "artigo_inexistente_no_texto"
MOTIVO_SEM_CITACAO = "sem_citacao"
MOTIVO_INCERTEZA_ALTA = "incerteza_alta"
MOTIVO_INCORRETO = "incorreto_juridicamente"
MOTIVO_SEM_LASTRO = "sem_lastro_nas_citacoes"
MOTIVO_INCOMPLETO = "incompleto"
MOTIVO_ABSTENCAO_INADEQUADA = "abstencao_inadequada"
MOTIVO_DADO_PESSOAL_REAL = "dado_pessoal_real"
MOTIVO_JUIZ_INDISPONIVEL = "juiz_nao_avaliou"


class ParametrosDeFiltro(BaseModel):
    jobs_de_sintese: list[str] = Field(min_length=1)
    k_reamostragens: int = Field(default=3, ge=0, le=5, description="0 desliga a self-consistency")
    temperatura_de_reamostragem: float = Field(default=0.8, ge=0.1, le=1.5)
    limiar_de_incerteza: float = Field(
        default=0.34, ge=0.0, le=1.0,
        description="Reprova se score_incerteza (fracao de reamostragens NAO equivalentes) for maior que isto",
    )
    exigir_citacoes_validas: bool = True
    tamanho_do_lote: int = Field(default=5, ge=1, le=20)
    concorrencia: int = Field(default=3, ge=1, le=12)
    teto_usd: float | None = Field(default=None, ge=0)


class VereditoDeQualidade(BaseModel):
    """Saida estruturada do JUIZ para um exemplo. Campos booleanos explicitos
    (nao um dicionario) para a decodificacao restrita exigir cada criterio."""

    indice: int = Field(description="O `indice` do caso a que este veredito responde")
    correcao_juridica: bool = Field(description="A resposta esta juridicamente correta frente aos dispositivos em contexto")
    lastro_nas_citacoes: bool = Field(description="Toda afirmacao normativa da resposta e' sustentada por um dispositivo citado e presente no contexto")
    completude: bool = Field(description="A resposta cobre o que a pergunta pede, inclusive condicoes e excecoes relevantes")
    abstencao_adequada: bool | None = Field(
        default=None,
        description="So' para tipo sem_resposta: a resposta se abstem corretamente e explica o que falta, sem inventar; null nos demais tipos",
    )
    sem_dado_pessoal_real: bool = Field(description="Nao ha nome, documento ou identificador que pareca de pessoa ou empresa real")
    nota: int = Field(ge=1, le=5, description="Qualidade geral do exemplo como material de treino (5 = exemplar)")
    problemas: str = Field(default="", description="Em ate 300 caracteres, o que esta errado ou faltando; vazio se nada")
    reamostragens_equivalentes: list[bool] = Field(
        default_factory=list,
        description="Para cada resposta alternativa, na ordem: chega a MESMA conclusao juridica com as MESMAS bases que a original",
    )


class DecisaoDoFiltro(BaseModel):
    exemplo_id: str
    job_filtro_id: str
    job_sintese_id: str
    tipo: TipoDeTarefa
    aprovado: bool
    score_incerteza: float | None = Field(default=None, description="1 - fracao de reamostragens equivalentes; null se k=0 ou nao avaliado")
    motivos: list[str] = Field(default_factory=list)
    veredito: VereditoDeQualidade | None = None
    reamostragens: list[str] = Field(default_factory=list)
    modelo_juiz: str = ""
    modelo_professor: str = ""
    versao_do_filtro: str
    criado_em: dt.datetime


class JobDeFiltro(BaseModel):
    id: str
    status: StatusDoJob
    parametros: ParametrosDeFiltro
    criado_em: dt.datetime
    atualizado_em: dt.datetime
    exemplos: int = 0
    reprovados_sem_llm: int = 0
    lotes: int = 0
    lotes_concluidos: int = 0
    lotes_com_erro: int = 0
    avaliados: int = 0
    aprovados: int = 0
    reprovados: int = 0
    reprovacoes_por_motivo: dict[str, int] = Field(default_factory=dict)
    custo_usd: float = 0.0
    modelo_juiz: str = ""
    modelo_professor: str = ""
    versao_do_filtro: str = ""
    erro: str | None = None
