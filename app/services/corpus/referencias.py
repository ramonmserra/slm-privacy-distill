"""Referencias legais em texto livre -> ids canonicos do corpus.

Dois usos, nos dois lados da pesquisa:

- normalizar o que o PROFESSOR cita (`artigos_citados`) e o que o banco de
  perguntas traz em `referencia` ("art. 5º, I, II e III; art. 12");
- extrair do texto de uma RESPOSTA os artigos mencionados ("conforme o art.
  18 da LGPD"), para a verificacao deterministica de citacao.

Gramatica aceita (a que aparece no banco inicial e em pareceres):

    art. 7º, IX; art. 10                -> art.7:inc.IX, art.10
    arts. 33 a 36                        -> art.33 ... art.36
    arts. 9º, 18 e 23                    -> art.9, art.18, art.23
    art. 52, II e § 1º                   -> art.52:inc.II, art.52:par.1
    art. 7º, §§ 3º e 4º                  -> art.7:par.3, art.7:par.4
    art. 11, II, “c”                     -> art.11:inc.II:ali.c
    arts. 7º, IV; 11, II, “c”; 13 e 16   -> art.7:inc.IV, art.11:inc.II:ali.c, art.13, art.16
    LGPD:art.48:par.1                    -> ja canonico, devolvido como esta

E' uma varredura SEQUENCIAL de tokens com um estado minimo (artigo corrente,
inciso corrente, "estou lendo paragrafos?"): numero em modo artigo e' outro
artigo; numeral romano e' inciso do artigo corrente; letra entre aspas e'
alinea do ultimo inciso; depois de `§` os numeros sao paragrafos ate o
proximo `art.`. Sempre devolve tambem o id do artigo-pai de cada item.
"""

from __future__ import annotations

import re

_FONTE_DEFAULT = "LGPD"

_RE_ID_CANONICO = re.compile(
    r"^[A-Z]+:art\.\d+(?:-[A-Z])?(?::(?:par\.(?:\d+|unico)|inc\.[IVXL]+(?:-[A-Z])?|ali\.[a-z]))*$"
)
_RE_TOKEN = re.compile(
    r"(?P<art>\barts?\b\.?)"
    r"|(?P<par_unico>\bpar[aá]grafo\s+[uú]nico\b)"
    r"|(?P<par>§+)"
    r"|(?P<num>\d+(?:-[A-Z])?)\s*[ºo°]?"
    r"|(?P<ali>[“\"']([a-z])[”\"'])"
    r"|(?P<rom>\b[IVXL]+(?:-[A-Z])?\b)"
    r"|(?P<range>\b(?:a|até)\b)"
    r"|(?P<sep>[;])",
    re.IGNORECASE,
)
_RE_MENCAO_NO_TEXTO = re.compile(r"\bart(?:igo)?s?\.?\s*(\d+(?:-[A-Z])?)\s*[ºo°]?", re.IGNORECASE)


def _pais(id_: str) -> list[str]:
    partes = id_.split(":")
    return [":".join(partes[:corte]) for corte in range(2, len(partes))]


class _Varredura:
    def __init__(self, fonte: str) -> None:
        self.fonte = fonte
        self.ids: list[str] = []
        self.artigo: str | None = None
        self.inciso: str | None = None
        self.lendo_paragrafos = False
        self.range_pendente = False

    def _add(self, id_: str) -> None:
        for candidato in _pais(id_) + [id_]:
            if candidato not in self.ids:
                self.ids.append(candidato)

    def novo_artigo(self, numero: str) -> None:
        if self.range_pendente and self.artigo and self.artigo.isdigit() and numero.isdigit():
            for n in range(int(self.artigo) + 1, int(numero) + 1):
                self._add(f"{self.fonte}:art.{n}")
        else:
            self._add(f"{self.fonte}:art.{numero}")
        self.artigo = numero
        self.inciso = None
        self.lendo_paragrafos = False
        self.range_pendente = False

    def numero(self, valor: str) -> None:
        if self.artigo is not None and self.lendo_paragrafos:
            self._add(f"{self.fonte}:art.{self.artigo}:par.{valor}")
            return
        self.novo_artigo(valor)

    def paragrafo_unico(self) -> None:
        if self.artigo is not None:
            self._add(f"{self.fonte}:art.{self.artigo}:par.unico")

    def romano(self, numeral: str) -> None:
        if self.artigo is None:
            return
        numeral = numeral.upper()
        self.inciso = f"{self.fonte}:art.{self.artigo}:inc.{numeral}"
        self._add(self.inciso)

    def alinea(self, letra: str) -> None:
        if self.inciso is not None:
            self._add(f"{self.inciso}:ali.{letra.lower()}")

    def separador(self) -> None:
        # ';' encerra o item; o artigo corrente continua valendo para um
        # segmento que comece por numero ("arts. 7º, IV; 11, II").
        self.inciso = None
        self.lendo_paragrafos = False


def normalizar_referencias(texto: str, fonte: str = _FONTE_DEFAULT) -> list[str]:
    """Ids canonicos (sem duplicatas, na ordem) de uma string de referencia.
    Strings ja canonicas passam direto, com os pais."""

    texto = texto.strip()
    if not texto:
        return []
    if _RE_ID_CANONICO.match(texto):
        return _pais(texto) + [texto]

    v = _Varredura(fonte)
    viu_art = False
    for m in _RE_TOKEN.finditer(texto):
        grupo = m.lastgroup
        if grupo == "art":
            viu_art = True
            v.lendo_paragrafos = False
        elif grupo == "num":
            if not viu_art:
                continue  # "Lei 13.709" ou numero solto antes do primeiro 'art.'
            v.numero(m.group("num"))
        elif grupo == "par":
            v.lendo_paragrafos = True
        elif grupo == "par_unico":
            v.paragrafo_unico()
        elif grupo == "rom":
            v.romano(m.group("rom"))
        elif grupo == "ali":
            v.alinea(m.group(6))
        elif grupo == "range":
            v.range_pendente = m.group("range").lower() in ("a", "até") and not v.lendo_paragrafos
        elif grupo == "sep":
            v.separador()
    return v.ids


def artigos_mencionados(texto: str, fonte: str = _FONTE_DEFAULT) -> list[str]:
    """Ids de ARTIGO mencionados num texto livre (resposta do modelo), sem
    duplicatas, na ordem. Nao tenta inciso/paragrafo: numa prosa, "II" pode
    ser qualquer coisa; o artigo e' o que se consegue afirmar com seguranca."""

    vistos: list[str] = []
    for m in _RE_MENCAO_NO_TEXTO.finditer(texto):
        id_ = f"{fonte}:art.{m.group(1)}"
        if id_ not in vistos:
            vistos.append(id_)
    return vistos


def artigo_de(id_: str) -> str:
    """`LGPD:art.7:inc.IX` -> `LGPD:art.7`."""

    partes = id_.split(":")
    return ":".join(partes[:2]) if len(partes) >= 2 else id_
