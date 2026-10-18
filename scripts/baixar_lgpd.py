"""Baixa o texto CONSOLIDADO da LGPD (Lei 13.709/2018) do Planalto e grava o
texto limpo em `recursos/fontes/lgpd.txt`, versionado no repositorio.

    python -m scripts.baixar_lgpd                    # baixa e grava
    python -m scripts.baixar_lgpd --de arquivo.htm   # usa um HTML ja baixado

Por que um arquivo versionado e nao download em runtime: o experimento
precisa ser reproduzivel -- o texto da lei muda (ANPD, alteracoes) e o
corpus de uma rodada tem de ser exatamente o da outra. O cabecalho do .txt
registra fonte e data.

O que a limpeza faz (e por que):

- descarta `<strike>` e `span` com `text-decoration: line-through`: e' assim
  que o Planalto marca a REDACAO ANTERIOR de dispositivo alterado. Sem isso,
  o inciso VIII do art. 7 apareceria duas vezes, uma delas revogada;
- descarta as anotacoes de alteracao `(Redacao dada pela Lei ...)`,
  `(Incluido pela ...)`, `(Vide ...)`, `(Convertida ...)` e o link
  `Vigencia` colado ao fim do dispositivo: sao metadado editorial, nao texto
  normativo, e confundiriam o professor e o chunking;
- descarta o dispositivo revogado de que sobrou so' o marcador (`§ 1º .`);
- junta linhas quebradas dentro do mesmo paragrafo HTML e normaliza espacos;
- corta o rodape da pagina a partir da assinatura ("Brasília, 14 de agosto").

O HTML baixado e' tratado como dado nao confiavel: so' e' lido pelo parser
da biblioteca padrao, nunca executado nem interpretado.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

URL_PLANALTO = "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm"
DESTINO = Path("recursos/fontes/lgpd.txt")

_ANOTACOES = re.compile(
    r"\((?:Reda[cç][aã]o dada|Inclu[ií]d[oa]|Revogad[oa]|Vide|Vig[eê]ncia|Renumerad[oa]|"
    r"Promulga[cç][aã]o|Convertid[oa])[^)]*\)",
    re.IGNORECASE,
)
# O link "Vigência" que o Planalto pendura ao fim do dispositivo alterado
# chega colado ao texto ("...Municípios. Vigência").
_VIGENCIA_SOLTA = re.compile(r"(^|[\s.;:,)])Vig[eê]ncia(?=\s|$)")
# Dispositivo revogado cuja redacao foi riscada: sobra so' o marcador.
_SO_MARCADOR = re.compile(
    r"^(?:§\s*\d+[ºo°]?|Par[aá]grafo [uú]nico\.?|[IVXL]+(?:-[A-Z])?\s*[-–]|[a-z]\))\s*\.?\s*;?\s*$"
)
_ARTIGO = re.compile(r"^Art\. ?\d")
# Erro de digitacao da PROPRIA pagina do Planalto: "Art. 5 7. (VETADO)." no
# lugar de "Art. 57.". Sem a correcao, o parser le "artigo 5" e o id
# LGPD:art.5 fica duplicado, sobrescrevendo o art. 5 verdadeiro.
_NUMERO_DE_ARTIGO_PARTIDO = re.compile(r"^(Art\.\s*\d)\s+(\d+\.)")


class _ExtratorDeTexto(HTMLParser):
    """Um item de texto por `<p>`, pulando o que esta riscado."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragrafos: list[str] = []
        self._atual: list[str] = []
        self._riscado = 0  # profundidade dentro de strike/line-through

    @staticmethod
    def _e_riscado(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in ("strike", "s", "del"):
            return True
        estilo = dict(attrs).get("style") or ""
        return "line-through" in estilo.replace(" ", "").lower()

    def handle_starttag(self, tag, attrs):
        if self._riscado or self._e_riscado(tag, attrs):
            self._riscado += 1
            return
        if tag == "p":
            self._fechar_paragrafo()
        if tag == "br":
            self._atual.append(" ")

    def handle_endtag(self, tag):
        if self._riscado:
            self._riscado -= 1
            return
        if tag == "p":
            self._fechar_paragrafo()

    def handle_data(self, data):
        if not self._riscado:
            self._atual.append(data)

    def _fechar_paragrafo(self) -> None:
        texto = html.unescape("".join(self._atual))
        texto = re.sub(r"[ \t\r\n\xa0]+", " ", texto).strip()
        self._atual = []
        if texto:
            self.paragrafos.append(texto)

    def close(self):
        super().close()
        self._fechar_paragrafo()


def limpar(paragrafos: list[str]) -> list[str]:
    limpos: list[str] = []
    for p in paragrafos:
        p = _ANOTACOES.sub("", p)
        p = _VIGENCIA_SOLTA.sub(r"\1", p)
        p = _NUMERO_DE_ARTIGO_PARTIDO.sub(r"\1\2", p)
        p = re.sub(r"\s+([;.,)])", r"\1", p)
        p = re.sub(r"\s{2,}", " ", p).strip()
        if not p or _SO_MARCADOR.match(p):
            continue
        limpos.append(p)
    return limpos


def extrair_texto(html_bruto: bytes) -> list[str]:
    try:
        texto = html_bruto.decode("windows-1252")
    except UnicodeDecodeError:
        texto = html_bruto.decode("utf-8", errors="replace")
    parser = _ExtratorDeTexto()
    parser.feed(texto)
    parser.close()
    return limpar(parser.paragrafos)


def recortar_corpo_da_lei(paragrafos: list[str]) -> list[str]:
    """Do primeiro `CAPÍTULO I` ate a assinatura presidencial: fora disso
    ficam o cabecalho da pagina e o rodape (scripts, DOU)."""

    inicio = next((i for i, p in enumerate(paragrafos) if re.match(r"^CAP[IÍ]TULO I\b", p)), 0)
    corpo = paragrafos[inicio:]
    fim = next((i for i, p in enumerate(corpo) if re.match(r"^Bras[ií]lia\s*,", p)), len(corpo))
    return corpo[:fim]


def baixar(url: str) -> bytes:
    import httpx

    resposta = httpx.get(
        url, headers={"User-Agent": "Mozilla/5.0 (pesquisa academica)"}, follow_redirects=True, timeout=60
    )
    resposta.raise_for_status()
    return resposta.content


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--de", type=Path, help="HTML ja baixado (em vez de acessar a rede)")
    ap.add_argument("--para", type=Path, default=DESTINO)
    args = ap.parse_args(argv)

    bruto = args.de.read_bytes() if args.de else baixar(URL_PLANALTO)
    paragrafos = recortar_corpo_da_lei(extrair_texto(bruto))
    artigos = sum(1 for p in paragrafos if _ARTIGO.match(p))
    if artigos < 60:
        print(f"ERRO: so' {artigos} artigos encontrados; a pagina mudou de formato?", file=sys.stderr)
        return 1

    cabecalho = [
        "# Lei nº 13.709, de 14 de agosto de 2018 (LGPD) -- texto consolidado",
        f"# Fonte: {URL_PLANALTO}",
        f"# Baixado em: {dt.date.today().isoformat()}",
        "# Limpeza: redacoes revogadas (riscadas) e anotacoes de alteracao removidas; ver scripts/baixar_lgpd.py",
        "",
    ]
    args.para.parent.mkdir(parents=True, exist_ok=True)
    args.para.write_text("\n".join(cabecalho + paragrafos) + "\n", encoding="utf-8")
    print(f"{artigos} artigos, {len(paragrafos)} paragrafos -> {args.para}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
