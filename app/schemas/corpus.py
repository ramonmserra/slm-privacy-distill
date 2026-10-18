"""Contratos do corpus juridico: a lei quebrada em DISPOSITIVOS com id
canonico. O id e' o que torna determinística a verificacao de citacao: um
exemplo sintetico que cita `LGPD:art.72` e' reprovado sem LLM, porque o id
nao existe."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TipoDeDispositivo(StrEnum):
    ARTIGO = "artigo"
    PARAGRAFO = "paragrafo"
    INCISO = "inciso"
    ALINEA = "alinea"


class Dispositivo(BaseModel):
    """Uma unidade normativa. `texto` e' SO' o texto daquela unidade; o artigo
    inteiro (com paragrafos e incisos) e' remontado por `Corpus.texto_do_artigo`."""

    id: str = Field(description="Canonico, ex.: LGPD:art.7:inc.IX, LGPD:art.5:inc.II, LGPD:art.48:par.1")
    fonte: str = Field(description="Sigla da norma, ex.: LGPD")
    artigo: str = Field(description="Numero do artigo como string (55-A tem sufixo)")
    tipo: TipoDeDispositivo
    rotulo: str = Field(description="Como aparece no texto: 'Art. 7º', '§ 1º', 'IX', 'c)'")
    texto: str
    pai_id: str | None = Field(default=None, description="Dispositivo que contem este (None para artigo)")
    capitulo: str = ""
    secao: str = ""
    ordem: int = Field(description="Posicao no texto, para remontar na ordem original")


class ResumoDoCorpus(BaseModel):
    fonte: str
    dispositivos: int
    artigos: int
    paragrafos: int
    incisos: int
    alineas: int
    caminho: str


class ResultadoDeBusca(BaseModel):
    artigo_id: str
    rotulo: str
    score: float
    trecho: str
