"""Deduplicacao de perguntas e protecao contra VAZAMENTO do conjunto de
avaliacao para o treino.

Duas referencias, um mecanismo: as perguntas ja aceitas no job (duplicata)
e as perguntas do banco inicial de avaliacao (vazamento). Um exemplo
sintetico parecido demais com uma pergunta da prova faria o aluno decorar a
prova -- e a metrica viraria memorizacao. O descarte e' por construcao, nao
por disciplina.

Medida: Jaccard sobre o conjunto de tokens normalizados (minusculas, sem
acento, sem stopwords). Barata, deterministica e boa para parafrase
proxima; parafrase profunda fica para o filtro semantico, se um dia a
medicao mostrar que escapa.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.services.corpus.retrieval import tokenizar

BANCO_INICIAL = Path("recursos/avaliacao/banco_inicial_lgpd.jsonl")
LIMIAR_DEFAULT = 0.7


def hash_da_pergunta(pergunta: str) -> str:
    """Id estavel do exemplo: hash dos tokens normalizados, em ordem."""

    return hashlib.sha256(" ".join(tokenizar(pergunta)).encode("utf-8")).hexdigest()[:16]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Duplicata:
    origem: str
    similaridade: float
    referencia: str


class Deduplicador:
    def __init__(self, limiar: float = LIMIAR_DEFAULT) -> None:
        self._limiar = limiar
        self._referencias: list[tuple[str, set[str], str]] = []  # (origem, tokens, texto)
        self._hashes: set[str] = set()

    def adicionar(self, pergunta: str, origem: str) -> None:
        self._referencias.append((origem, set(tokenizar(pergunta)), pergunta))
        self._hashes.add(hash_da_pergunta(pergunta))

    def adicionar_varias(self, perguntas: list[str], origem: str) -> None:
        for p in perguntas:
            self.adicionar(p, origem)

    def verificar(self, pergunta: str) -> Duplicata | None:
        """A referencia mais parecida acima do limiar, ou None."""

        if hash_da_pergunta(pergunta) in self._hashes:
            for origem, _, texto in self._referencias:
                if hash_da_pergunta(texto) == hash_da_pergunta(pergunta):
                    return Duplicata(origem=origem, similaridade=1.0, referencia=texto)
        tokens = set(tokenizar(pergunta))
        melhor: Duplicata | None = None
        for origem, ref_tokens, texto in self._referencias:
            s = jaccard(tokens, ref_tokens)
            if s >= self._limiar and (melhor is None or s > melhor.similaridade):
                melhor = Duplicata(origem=origem, similaridade=round(s, 3), referencia=texto)
        return melhor


def carregar_banco_inicial(caminho: Path = BANCO_INICIAL) -> list[str]:
    if not caminho.exists():
        return []
    perguntas: list[str] = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            perguntas.append(json.loads(linha)["pergunta"])
    return perguntas
