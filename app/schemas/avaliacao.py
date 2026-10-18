"""Contratos da AVALIACAO: o backtest de um modelo contra um conjunto, com
juiz e verificacao deterministica, e a comparacao pareada entre runs.

Principios herdados do Sandbox do projeto de referencia: o juiz e' o MESMO
para professor e aluno e nunca e' destilado; `grounded` (lastro) e' eixo
separado de `correto`; o veredito e' redigido a partir de numeros por
funcoes puras; dois runs so' se comparam com a mesma `versao_da_medida`.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.sintese import StatusDoJob


class Conjunto(StrEnum):
    BANCO_INICIAL = "banco_inicial"
    HELD_OUT = "held_out"


class CategoriaDeErro(StrEnum):
    NON_ANSWER = "non_answer"
    HALLUCINATION = "hallucination"
    WRONG_LEGAL_BASIS = "wrong_legal_basis"
    INCOMPLETE_ANSWER = "incomplete_answer"
    OUT_OF_SCOPE_ANSWERED = "out_of_scope_answered"
    FORMAT_ERROR = "format_error"


class ParametrosDeAvaliacao(BaseModel):
    modelo: str = Field(
        description="Papel (`teacher`, `judge`, `student`) ou id qualificado (`local:<id do run de treino>`, `openai:<modelo>`)"
    )
    conjunto: Conjunto = Conjunto.BANCO_INICIAL
    dataset_versao: str | None = Field(default=None, description="Obrigatorio para `held_out`")
    limite: int | None = Field(default=None, ge=1, description="Avalia so' os primeiros N itens (ordem estavel)")
    com_rag: bool = Field(default=False, description="Condicao de controle: o modelo recebe os k artigos mais proximos da pergunta")
    k_rag: int = Field(default=3, ge=1, le=8)
    max_tokens_resposta: int = Field(default=1024, ge=64, le=8192)
    concorrencia: int = Field(default=3, ge=1, le=12)
    tamanho_do_lote_do_juiz: int = Field(default=5, ge=1, le=20)
    teto_usd: float | None = Field(default=None, ge=0)


class ItemDeAvaliacao(BaseModel):
    id: str
    conjunto: Conjunto
    pergunta: str
    referencia: str = Field(description="Resposta do professor (held-out) ou criterios esperados (banco)")
    artigos_referencia: list[str] = Field(default_factory=list, description="Ids de ARTIGO que sustentam a referencia")
    tipo: str = Field(description="TipoDeTarefa no held-out; 'pergunta' ou 'consistencia' no banco")
    abstencao_esperada: bool = Field(default=False, description="A resposta correta e' abster-se / pedir informacao")
    rotulo: str = ""


class RespostaDoModelo(BaseModel):
    item_id: str
    texto: str
    latencia_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    erro: str | None = None
    artigos_mencionados: list[str] = Field(default_factory=list)
    artigos_inexistentes: list[str] = Field(default_factory=list, description="Mencionados na prosa e ausentes do corpus")
    citou_referencia: bool = Field(default=False, description="Mencionou ao menos um artigo da referencia")


class VereditoDeAvaliacao(BaseModel):
    """Saida estruturada do JUIZ para um item. `correto` e `grounded` sao
    eixos independentes: uma resposta pode coincidir com a referencia e
    afirmar fato sem lastro (adivinhou e acertou), ou divergir e estar
    inteiramente sustentada."""

    indice: int
    correto: bool = Field(description="A resposta chega a conclusao juridica da referencia / atende os criterios essenciais")
    criterios_atendidos: int = Field(ge=0, description="Quantos criterios esperados foram atendidos (0 se nao ha lista)")
    criterios_total: int = Field(ge=0, description="Quantos criterios esperados havia (0 se nao ha lista)")
    grounded: bool | None = Field(description="Toda afirmacao normativa esta' sustentada pelos dispositivos de referencia fornecidos; null se nao avaliavel")
    abstencao: bool = Field(description="O modelo se absteve / pediu informacao em vez de concluir")
    categoria: CategoriaDeErro | None = Field(default=None, description="Categoria do erro quando correto=false; null quando correto")
    similaridade: float = Field(ge=0, le=1, description="0-1: quao equivalente a resposta e' a referencia")
    justificativa: str = Field(default="", description="Ate 300 caracteres")


class ResultadoDoItem(BaseModel):
    item: ItemDeAvaliacao
    resposta: RespostaDoModelo
    veredito: VereditoDeAvaliacao | None = None
    acerto: bool = False
    alucinacao: bool = Field(default=False, description="grounded=false OU artigo inexistente na prosa")
    abstencao_adequada: bool | None = Field(default=None, description="So' nos itens com abstencao_esperada")


class MetricasDoRun(BaseModel):
    itens: int = 0
    respondidos: int = 0
    julgados: int = 0
    acertos: int = 0
    acuracia: float | None = None
    acuracia_ic95: tuple[float, float] | None = None
    alucinacoes: int = 0
    taxa_de_alucinacao: float | None = None
    taxa_de_alucinacao_ic95: tuple[float, float] | None = None
    itens_com_artigo_inexistente: int = 0
    taxa_citou_referencia: float | None = None
    criterios_pct: float | None = Field(default=None, description="Media de criterios atendidos / total, nos itens com rubrica")
    abstencoes_esperadas: int = 0
    abstencoes_adequadas: int = 0
    categorias_de_erro: dict[str, int] = Field(default_factory=dict)
    por_tipo: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    latencia_p50_ms: float | None = None
    latencia_p95_ms: float | None = None
    tokens_entrada: int = 0
    tokens_saida: int = 0
    custo_usd: float = 0.0
    custo_usd_por_item: float | None = None


class RunDeAvaliacao(BaseModel):
    id: str
    status: StatusDoJob
    parametros: ParametrosDeAvaliacao
    modelo: str = Field(description="Id qualificado efetivamente medido")
    modelo_juiz: str = ""
    versao_da_medida: str = Field(description="hash(prompt do juiz + prompt do aluno + regras + conjunto); runs so' se comparam se igual")
    criado_em: dt.datetime
    atualizado_em: dt.datetime
    itens: int = 0
    respondidos: int = 0
    lotes_do_juiz: int = 0
    lotes_do_juiz_concluidos: int = 0
    lotes_do_juiz_com_erro: int = 0
    metricas: MetricasDoRun = Field(default_factory=MetricasDoRun)
    erro: str | None = None


class Anotacao(BaseModel):
    """Julgamento HUMANO de um item de um run -- a amostra de calibracao do
    juiz. `correto` e `grounded` seguem a mesma definicao da rubrica do juiz
    para o kappa comparar a mesma pergunta."""

    item_id: str
    anotador: str = Field(min_length=1, max_length=60)
    correto: bool
    grounded: bool | None = None
    comentario: str = Field(default="", max_length=500)


class AnotacaoRegistrada(Anotacao):
    run_id: str
    criado_em: dt.datetime


class ConcordanciaPorAnotador(BaseModel):
    anotador: str
    itens: int
    kappa_correto: float | None
    kappa_grounded: float | None
    acuracia_do_juiz_vs_humano: float | None = Field(description="Fracao de itens em que juiz e humano concordam em `correto`")


class ResumoDeCalibracao(BaseModel):
    run_id: str
    itens_anotados: int
    anotadores: list[str]
    por_anotador: list[ConcordanciaPorAnotador]
    kappa_correto_maioria: float | None = Field(description="Juiz vs voto da maioria dos humanos (itens com >=2 anotadores)")
    kappa_entre_humanos: float | None = Field(description="Media dos kappas par a par entre anotadores, nos itens em comum")
    leitura: str


class ComparacaoPareada(BaseModel):
    run_a: str
    run_b: str
    modelo_a: str
    modelo_b: str
    mesma_versao_da_medida: bool
    itens_em_comum: int
    acertos_a: int
    acertos_b: int
    so_a_acertou: int
    so_b_acertou: int
    delta_acuracia: float = Field(description="acuracia_b - acuracia_a nos itens em comum")
    delta_ic95_bootstrap: tuple[float, float]
    p_mcnemar: float
    melhoraram: list[str] = Field(description="itens que B acerta e A erra")
    pioraram: list[str] = Field(description="itens que A acerta e B erra")
    alucinacoes_a: int = 0
    alucinacoes_b: int = 0
    leitura: str
