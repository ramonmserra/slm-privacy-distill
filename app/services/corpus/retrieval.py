"""Recuperacao lexical (BM25) sobre os ARTIGOS do corpus.

Implementacao propria de ~60 linhas em vez de uma dependencia: o corpus tem
dezenas de artigos, o calculo e' trivial, e nao ha embeddings pagos no
caminho. Deterministico por construcao -- mesma consulta, mesmo ranking --,
que e' o que a reprodutibilidade dos experimentos pede.

Tokenizacao para portugues juridico: minusculas, acentos removidos,
stopwords curtas fora. Sem stemming: "tratamento"/"tratamentos" ficam
separados, e isso e' aceitavel para um indice de apoio a sintese e ao
contexto do juiz (nao e' o retrieval de producao de um agente).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre e ou que se "
    "ao aos à às pelo pela pelos pelas este esta estes estas esse essa esses essas aquele aquela isto isso "
    "seu sua seus suas ser é são foi era ter tem têm há como mais menos muito já também não nem quando onde "
    "qual quais cujo cuja cujos cujas lei desta deste nesta neste art artigo inciso caput".split()
)
_RE_TOKEN = re.compile(r"[a-z0-9]+")


def tokenizar(texto: str) -> list[str]:
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return [t for t in _RE_TOKEN.findall(sem_acento) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Documento:
    id: str
    rotulo: str
    texto: str


@dataclass(frozen=True)
class Acerto:
    id: str
    rotulo: str
    score: float
    trecho: str


class IndiceBM25:
    """BM25 Okapi (k1=1.5, b=0.75) sobre documentos tokenizados uma vez."""

    def __init__(self, documentos: list[Documento], k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = documentos
        self._k1 = k1
        self._b = b
        self._tokens = [tokenizar(d.texto) for d in documentos]
        self._tf = [Counter(t) for t in self._tokens]
        self._tamanhos = [len(t) for t in self._tokens]
        self._media = (sum(self._tamanhos) / len(self._tamanhos)) if self._tamanhos else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(documentos)
        self._idf = {termo: math.log(1 + (n - f + 0.5) / (f + 0.5)) for termo, f in df.items()}

    def __len__(self) -> int:
        return len(self._docs)

    def buscar(self, consulta: str, k: int = 5) -> list[Acerto]:
        termos = tokenizar(consulta)
        if not termos or not self._docs:
            return []
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            norm = self._k1 * (1 - self._b + self._b * self._tamanhos[i] / (self._media or 1))
            for termo in termos:
                f = tf.get(termo, 0)
                if f:
                    s += self._idf.get(termo, 0.0) * f * (self._k1 + 1) / (f + norm)
            if s > 0:
                scores.append((s, i))
        # Desempate pelo indice: ranking estavel entre execucoes.
        scores.sort(key=lambda par: (-par[0], par[1]))
        return [
            Acerto(id=self._docs[i].id, rotulo=self._docs[i].rotulo, score=round(s, 4), trecho=self._docs[i].texto[:240])
            for s, i in scores[:k]
        ]
