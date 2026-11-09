"""Baixa do Diario Oficial da Uniao a Resolucao CD/ANPD n. 4/2023 e grava o
ANEXO -- o Regulamento de Dosimetria e Aplicacao de Sancoes Administrativas
-- limpo em `recursos/fontes/dosimetria_anpd.txt`, versionado no repositorio.

    python -m scripts.baixar_dosimetria                    # baixa e grava
    python -m scripts.baixar_dosimetria --de pagina.html   # usa um HTML ja baixado

Por que o Regulamento entra no corpus: a LGPD delega a' ANPD a dosimetria
das sancoes (arts. 52 a 54). Perguntas sobre gravidade da infracao, multa,
agravantes e atenuantes nao tem resposta no texto da lei, e o professor so'
pode citar o que esta' no corpus. O Regulamento e' norma (resolucao com
artigos, paragrafos, incisos e alineas), entao passa pelo mesmo chunking e
pela mesma verificacao de citacao que a lei, com o prefixo `DOSIMETRIA`.

O que entra e o que fica de fora:

- entra so' o ANEXO (arts. 1 a 28 do Regulamento). Os arts. 1 a 4 da
  Resolucao em si aprovam o anexo e alteram a Resolucao CD/ANPD 1/2021 com
  texto entre aspas; chunkar isso criaria artigos falsos;
- os APENDICES I e II (metodologia do valor-base e valores minimos) sao
  publicados como IMAGENS no DOU e nao tem texto extraivel. O arquivo diz
  isso no cabecalho; uma resposta que precise deles deve dizer que o valor
  esta' em apendice nao transcrito;
- "Subseção" nao existe no chunker (a LGPD nao tem); vira comentario,
  preservado para leitura humana.

O HTML baixado e' tratado como dado nao confiavel: so' e' lido pelo parser
da biblioteca padrao, nunca executado nem interpretado.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from scripts.baixar_lgpd import _ExtratorDeTexto, baixar

URL_DOU = "https://www.in.gov.br/web/dou/-/resolucao-cd/anpd-n-4-de-24-de-fevereiro-de-2023-466146077"
DESTINO = Path("recursos/fontes/dosimetria_anpd.txt")
ARTIGOS_ESPERADOS = 28

_ARTIGO = re.compile(r"^Art\. ?\d")
_CABECALHO_DE_DIVISAO = re.compile(r"^(CAP[IÍ]TULO|Se[cç][aã]o|Subse[cç][aã]o)\s+[IVXL]+\s*$")
_MARCADOR = re.compile(r"^(?:Art\.|§|Par[aá]grafo [uú]nico|[IVXL]+(?:-[A-Z])?\s*[-–]|[a-z]\))")
_LIGATURAS = {"ﬁ": "fi", "ﬂ": "fl"}


def extrair_paragrafos(html_bruto: bytes) -> list[str]:
    parser = _ExtratorDeTexto()
    parser.feed(html_bruto.decode("utf-8", errors="replace"))
    parser.close()
    return parser.paragrafos


def recortar_anexo(paragrafos: list[str]) -> list[str]:
    """Do `CAPÍTULO I` que segue o titulo do ANEXO ate o ultimo artigo. O que
    vem antes e' a Resolucao; o que vem depois e' rodape do portal."""

    inicio_anexo = next(i for i, p in enumerate(paragrafos) if p.strip().upper() == "ANEXO")
    corpo = paragrafos[inicio_anexo + 1 :]
    inicio = next(i for i, p in enumerate(corpo) if re.match(r"^CAP[IÍ]TULO I\b", p))
    corpo = corpo[inicio:]
    fim = next((i for i, p in enumerate(corpo) if p.startswith("Este conteúdo não substitui")), len(corpo))
    return corpo[:fim]


def normalizar(paragrafos: list[str]) -> list[str]:
    """Junta 'CAPÍTULO I' + 'DISPOSIÇÕES GERAIS' numa linha (formato do
    chunker), transforma 'Subseção' em comentario e tira ligaduras."""

    saida: list[str] = []
    i = 0
    while i < len(paragrafos):
        p = paragrafos[i]
        for lig, plano in _LIGATURAS.items():
            p = p.replace(lig, plano)
        if _CABECALHO_DE_DIVISAO.match(p) and i + 1 < len(paragrafos) and not _MARCADOR.match(paragrafos[i + 1]):
            p = f"{p} {paragrafos[i + 1].strip()}"
            i += 1
        if p.lower().startswith("subseção"):
            p = f"# {p}"
        saida.append(p)
        i += 1
    return saida


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--de", type=Path, help="HTML ja baixado (em vez de acessar a rede)")
    ap.add_argument("--para", type=Path, default=DESTINO)
    args = ap.parse_args(argv)

    bruto = args.de.read_bytes() if args.de else baixar(URL_DOU)
    linhas = normalizar(recortar_anexo(extrair_paragrafos(bruto)))
    artigos = sum(1 for p in linhas if _ARTIGO.match(p))
    if artigos != ARTIGOS_ESPERADOS:
        print(f"ERRO: {artigos} artigos encontrados, esperados {ARTIGOS_ESPERADOS}; a pagina mudou?", file=sys.stderr)
        return 1

    cabecalho = [
        "# Resolução CD/ANPD nº 4, de 24 de fevereiro de 2023 -- ANEXO: Regulamento de Dosimetria e Aplicação de Sanções Administrativas",
        f"# Fonte: {URL_DOU}",
        f"# Baixado em: {dt.date.today().isoformat()}",
        "# Recorte: só o Anexo (arts. 1 a 28). Os arts. 1 a 4 da Resolução (aprovação do anexo e alterações da Resolução CD/ANPD 1/2021) ficaram de fora.",
        "# Os Apêndices I e II (metodologia do valor-base e valores mínimos da multa) são imagens no DOU e NÃO estão transcritos aqui; ver scripts/baixar_dosimetria.py",
        "",
    ]
    args.para.parent.mkdir(parents=True, exist_ok=True)
    args.para.write_text("\n".join(cabecalho + linhas) + "\n", encoding="utf-8")
    print(f"{artigos} artigos, {len(linhas)} linhas -> {args.para}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
