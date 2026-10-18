"""Chunking de texto legal brasileiro em DISPOSITIVOS com id canonico.

Entrada: as linhas do `.txt` limpo (uma por paragrafo HTML do Planalto).
Saida: lista de `Dispositivo` na ordem do texto, com hierarquia
artigo > paragrafo > inciso > alinea e o capitulo/secao correntes.

Ids (o contrato que o resto do projeto consome):

    LGPD:art.7            artigo (caput)
    LGPD:art.7:par.1      § 1º do art. 7
    LGPD:art.1:par.unico  paragrafo unico
    LGPD:art.7:inc.IX     inciso IX do caput
    LGPD:art.7:par.1:inc.II   inciso de um paragrafo
    LGPD:art.11:inc.II:ali.c  alinea de um inciso
    LGPD:art.55-A         artigo com sufixo

Uma linha sem marcador reconhecido e' CONTINUACAO do dispositivo corrente
(o Planalto quebra frases; e as alteracoes a outras leis dentro do art. 60
vem entre aspas). Nao se inventa dispositivo para o que nao tem marcador.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.corpus import Dispositivo, TipoDeDispositivo

_RE_CAPITULO = re.compile(r"^CAP[IÍ]TULO\s+([IVXL]+)\b\s*(.*)$")
_RE_SECAO = re.compile(r"^Se[cç][aã]o\s+([IVXL]+)\b\s*(.*)$")
_RE_ARTIGO = re.compile(r"^Art\.\s*(\d+(?:-[A-Z])?)\s*[ºo°]?\.?\s*(.*)$")
_RE_PARAGRAFO = re.compile(r"^§\s*(\d+)\s*[ºo°]?\.?\s*(.*)$")
_RE_PAR_UNICO = re.compile(r"^Par[aá]grafo\s+[uú]nico\.?\s*(.*)$")
_RE_INCISO = re.compile(r"^([IVXL]+(?:-[A-Z])?)\s*[-–]\s*(.*)$")
_RE_ALINEA = re.compile(r"^([a-z])\)\s*(.*)$")
_RE_COMENTARIO = re.compile(r"^#")


@dataclass
class _Estado:
    fonte: str
    capitulo: str = ""
    secao: str = ""
    artigo: str | None = None
    paragrafo_id: str | None = None   # paragrafo corrente (ou None = caput)
    inciso_id: str | None = None      # inciso corrente dentro do paragrafo/caput
    ordem: int = 0
    dispositivos: list[Dispositivo] = field(default_factory=list)

    def _emitir(self, id_: str, tipo: TipoDeDispositivo, rotulo: str, texto: str, pai: str | None) -> None:
        self.ordem += 1
        self.dispositivos.append(
            Dispositivo(
                id=id_, fonte=self.fonte, artigo=self.artigo or "", tipo=tipo, rotulo=rotulo,
                texto=texto.strip(), pai_id=pai, capitulo=self.capitulo, secao=self.secao, ordem=self.ordem,
            )
        )

    def artigo_id(self) -> str:
        return f"{self.fonte}:art.{self.artigo}"

    def novo_artigo(self, numero: str, texto: str, rotulo: str) -> None:
        self.artigo = numero
        self.paragrafo_id = None
        self.inciso_id = None
        self._emitir(self.artigo_id(), TipoDeDispositivo.ARTIGO, rotulo, texto, None)

    def novo_paragrafo(self, sufixo: str, texto: str, rotulo: str) -> None:
        if self.artigo is None:
            return
        self.paragrafo_id = f"{self.artigo_id()}:par.{sufixo}"
        self.inciso_id = None
        self._emitir(self.paragrafo_id, TipoDeDispositivo.PARAGRAFO, rotulo, texto, self.artigo_id())

    def novo_inciso(self, numeral: str, texto: str, rotulo: str) -> None:
        if self.artigo is None:
            return
        pai = self.paragrafo_id or self.artigo_id()
        self.inciso_id = f"{pai}:inc.{numeral}"
        self._emitir(self.inciso_id, TipoDeDispositivo.INCISO, rotulo, texto, pai)

    def nova_alinea(self, letra: str, texto: str, rotulo: str) -> None:
        if self.inciso_id is None:
            # Alinea sem inciso corrente: pendura no paragrafo/caput, que e' o
            # que o texto permite afirmar.
            pai = self.paragrafo_id or (self.artigo_id() if self.artigo else None)
            if pai is None:
                return
        else:
            pai = self.inciso_id
        self._emitir(f"{pai}:ali.{letra}", TipoDeDispositivo.ALINEA, rotulo, texto, pai)

    def continuar(self, texto: str) -> None:
        if not self.dispositivos:
            return
        ultimo = self.dispositivos[-1]
        ultimo.texto = f"{ultimo.texto} {texto.strip()}".strip()


def _titulo(numeral: str, resto: str, prefixo: str) -> str:
    return f"{prefixo} {numeral}" + (f" - {resto.strip()}" if resto.strip() else "")


def chunkar(linhas: list[str], fonte: str = "LGPD") -> list[Dispositivo]:
    estado = _Estado(fonte=fonte)
    titulo_pendente: str | None = None
    for bruta in linhas:
        linha = bruta.strip()
        if not linha or _RE_COMENTARIO.match(linha):
            continue
        if m := _RE_CAPITULO.match(linha):
            estado.capitulo = _titulo(m.group(1), m.group(2), "Capítulo")
            estado.secao = ""
            titulo_pendente = "capitulo" if not m.group(2).strip() else None
            continue
        if m := _RE_SECAO.match(linha):
            estado.secao = _titulo(m.group(1), m.group(2), "Seção")
            titulo_pendente = "secao" if not m.group(2).strip() else None
            continue
        if m := _RE_ARTIGO.match(linha):
            titulo_pendente = None
            estado.novo_artigo(m.group(1), m.group(2), f"Art. {m.group(1)}")
            continue
        if m := _RE_PARAGRAFO.match(linha):
            estado.novo_paragrafo(m.group(1), m.group(2), f"§ {m.group(1)}º")
            continue
        if m := _RE_PAR_UNICO.match(linha):
            estado.novo_paragrafo("unico", m.group(1), "Parágrafo único")
            continue
        if m := _RE_INCISO.match(linha):
            estado.novo_inciso(m.group(1), m.group(2), m.group(1))
            continue
        if m := _RE_ALINEA.match(linha):
            estado.nova_alinea(m.group(1), m.group(2), f"{m.group(1)})")
            continue
        # Titulo de capitulo/secao que veio na linha seguinte ao marcador
        # (o Planalto separa "CAPÍTULO IX" do nome em alguns casos).
        if titulo_pendente == "capitulo" and linha.isupper():
            estado.capitulo = f"{estado.capitulo} - {linha.rstrip('’')}"
            titulo_pendente = None
            continue
        if titulo_pendente == "secao":
            estado.secao = f"{estado.secao} - {linha}"
            titulo_pendente = None
            continue
        estado.continuar(linha)
    return estado.dispositivos
