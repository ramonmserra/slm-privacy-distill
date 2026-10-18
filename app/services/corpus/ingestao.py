"""O CORPUS em memoria: dispositivos indexados por id, artigos remontados e o
indice BM25 -- a unica estrutura que sintese, filtro e avaliacao consultam.

`ingerir` le o `.txt` versionado em `recursos/fontes/`, chunka e persiste em
`DATA_DIR/corpus/<fonte>.jsonl`; `carregar` le o que foi persistido. A API
mantem um `Corpus` por processo (`corpus_ativo`), carregado na primeira
leitura.
"""

from __future__ import annotations

import logging
import threading
from functools import cached_property
from pathlib import Path

from app.repositories.corpus_repository import CorpusRepository
from app.schemas.corpus import Dispositivo, ResumoDoCorpus, TipoDeDispositivo
from app.services.corpus.chunking import chunkar
from app.services.corpus.referencias import artigo_de
from app.services.corpus.retrieval import Acerto, Documento, IndiceBM25

logger = logging.getLogger(__name__)

FONTES_VERSIONADAS: dict[str, Path] = {
    "LGPD": Path("recursos/fontes/lgpd.txt"),
}


class FonteDesconhecida(ValueError):
    pass


class CorpusVazio(RuntimeError):
    """Nenhuma fonte ingerida ainda: `POST /corpus/ingerir` primeiro."""


class CorpusInvalido(ValueError):
    """Dois dispositivos com o mesmo id. Um dicionario por id sobrescreveria
    um deles em silencio -- foi assim que um "Art. 5 7." da fonte apagou o
    art. 5 verdadeiro. Falha fechado: a fonte precisa ser corrigida."""


class Corpus:
    def __init__(self, fonte: str, dispositivos: list[Dispositivo], caminho: Path) -> None:
        self.fonte = fonte
        self.caminho = caminho
        self._dispositivos = sorted(dispositivos, key=lambda d: d.ordem)
        vistos: set[str] = set()
        duplicados = sorted({d.id for d in self._dispositivos if d.id in vistos or vistos.add(d.id)})
        if duplicados:
            raise CorpusInvalido(f"Ids duplicados no corpus {fonte}: {duplicados[:5]}")
        self._por_id = {d.id: d for d in self._dispositivos}
        self._filhos: dict[str, list[Dispositivo]] = {}
        for d in self._dispositivos:
            if d.pai_id:
                self._filhos.setdefault(d.pai_id, []).append(d)

    # -- consulta --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dispositivos)

    def existe(self, id_: str) -> bool:
        return id_ in self._por_id

    def dispositivo(self, id_: str) -> Dispositivo | None:
        return self._por_id.get(id_)

    def artigos(self) -> list[Dispositivo]:
        return [d for d in self._dispositivos if d.tipo == TipoDeDispositivo.ARTIGO]

    def ids(self) -> list[str]:
        return [d.id for d in self._dispositivos]

    def texto_do_artigo(self, artigo_id: str) -> str:
        """O artigo inteiro remontado na ordem do texto, com os rotulos --
        e' o que vai ao professor como contexto e ao juiz como rastro."""

        raiz = self._por_id.get(artigo_de(artigo_id))
        if raiz is None:
            return ""
        linhas = [f"{raiz.rotulo} {raiz.texto}".strip()]

        def descer(pai: Dispositivo, nivel: int) -> None:
            for filho in self._filhos.get(pai.id, []):
                linhas.append(f"{'  ' * nivel}{filho.rotulo} {filho.texto}".strip())
                descer(filho, nivel + 1)

        descer(raiz, 1)
        return "\n".join(linhas)

    def resumo(self) -> ResumoDoCorpus:
        contagem = {t: 0 for t in TipoDeDispositivo}
        for d in self._dispositivos:
            contagem[d.tipo] += 1
        return ResumoDoCorpus(
            fonte=self.fonte,
            dispositivos=len(self._dispositivos),
            artigos=contagem[TipoDeDispositivo.ARTIGO],
            paragrafos=contagem[TipoDeDispositivo.PARAGRAFO],
            incisos=contagem[TipoDeDispositivo.INCISO],
            alineas=contagem[TipoDeDispositivo.ALINEA],
            caminho=str(self.caminho),
        )

    # -- recuperacao (o "R" do RAG) --------------------------------------------

    @cached_property
    def indice(self) -> IndiceBM25:
        return IndiceBM25(
            [Documento(id=a.id, rotulo=a.rotulo, texto=self.texto_do_artigo(a.id)) for a in self.artigos()]
        )

    def buscar(self, consulta: str, k: int = 5) -> list[Acerto]:
        return self.indice.buscar(consulta, k)

    def relacionados(self, artigo_id: str, k: int = 2) -> list[Dispositivo]:
        """Artigos mais proximos de um artigo, pelo texto dele -- o contexto
        extra que o professor recebe para enxergar remissoes ("observado o
        art. 11") sem que ninguem precise cadastrar as remissoes a mao."""

        texto = self.texto_do_artigo(artigo_id)
        if not texto:
            return []
        acertos = [a for a in self.indice.buscar(texto, k + 1) if a.id != artigo_de(artigo_id)]
        return [self._por_id[a.id] for a in acertos[:k] if a.id in self._por_id]


def ingerir(fonte: str, repositorio: CorpusRepository, caminho: Path | None = None) -> Corpus:
    caminho = caminho or FONTES_VERSIONADAS.get(fonte.upper())
    if caminho is None:
        raise FonteDesconhecida(f"Fonte {fonte!r} desconhecida; conhecidas: {sorted(FONTES_VERSIONADAS)}.")
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} nao existe. Rode `python -m scripts.baixar_lgpd` antes.")
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    dispositivos = chunkar(linhas, fonte=fonte.upper())
    repositorio.salvar(fonte.upper(), dispositivos)
    logger.info("Corpus %s ingerido: %d dispositivos de %s", fonte.upper(), len(dispositivos), caminho)
    return Corpus(fonte.upper(), dispositivos, caminho)


class _Registro:
    """Um `Corpus` por processo, carregado do repositorio na primeira leitura."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._corpus: Corpus | None = None

    def obter(self, repositorio: CorpusRepository, fonte: str = "LGPD") -> Corpus:
        with self._lock:
            if self._corpus is None or self._corpus.fonte != fonte.upper():
                dispositivos = repositorio.carregar(fonte.upper())
                if not dispositivos:
                    raise CorpusVazio(f"Corpus {fonte.upper()} nao ingerido. Chame POST /corpus/ingerir.")
                self._corpus = Corpus(fonte.upper(), dispositivos, repositorio.caminho(fonte.upper()))
            return self._corpus

    def definir(self, corpus: Corpus) -> None:
        with self._lock:
            self._corpus = corpus

    def limpar(self) -> None:
        with self._lock:
            self._corpus = None


registro = _Registro()
