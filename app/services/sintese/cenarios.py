"""CENARIOS de inspiracao para a sintese: situacoes praticas extraidas dos
guias orientativos publicos da ANPD e versionadas em
`recursos/fontes/anpd/cenarios.jsonl`.

O que um cenario E: um bloco de DADO anexado ao pedido para ambientar a
pergunta (setor, agente, tipo de dado, situacao) -- da' realismo e
diversidade sem depender da imaginacao do professor.

O que um cenario NAO E: fonte. O professor continua citando somente
dispositivos do contexto (a lei); o cenario nao entra no indice BM25, nao
vira id citavel, nao vai ao juiz como rastro e nao muda a verificacao
deterministica de citacoes. Guia orientativo nao vincula e muda de versao;
por isso ele inspira a pergunta e nao sustenta a resposta.

Cada linha do JSONL guarda `guia`, `pagina` e um `trecho` literal do PDF
para que qualquer pessoa confira a origem; a curadoria (reescrita como
situacao ficticia, com marcadores no lugar de nomes) esta descrita no
README da pasta.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas.sintese import TipoDeTarefa

logger = logging.getLogger(__name__)

CAMINHO_DEFAULT = Path("recursos/fontes/anpd/cenarios.jsonl")


class CenariosInvalidos(ValueError):
    """Arquivo de cenarios com id duplicado ou linha fora do contrato."""


class Cenario(BaseModel):
    id: str = Field(min_length=1, description="Ex.: anpd-cookies-003; vai gravado no exemplo como `cenario_id`")
    guia: str = Field(min_length=1, description="Apelido do guia, ex.: cookies, encarregado")
    titulo_do_guia: str
    ano: int | None = None
    pagina: int = Field(ge=1, description="Pagina do PDF de onde saiu o trecho")
    tipos: list[TipoDeTarefa] = Field(min_length=1, description="Tipos de tarefa que o cenario ambienta")
    cenario: str = Field(min_length=20, description="Situacao ficticia, com marcadores no lugar de nomes; sem conclusao juridica")
    trecho: str = Field(default="", description="Trecho literal do guia que motivou o cenario (rastreabilidade)")


def carregar_cenarios(caminho: Path = CAMINHO_DEFAULT) -> list[Cenario]:
    """Le o JSONL (linhas vazias e `#` ignoradas). Arquivo ausente = sintese
    sem cenarios, com aviso no log; id duplicado = falha fechado, porque o
    `cenario_id` gravado no exemplo deixaria de identificar a origem."""

    if not caminho.exists():
        logger.warning("Cenarios da ANPD nao encontrados em %s; a sintese roda sem cenarios.", caminho)
        return []
    cenarios: list[Cenario] = []
    vistos: set[str] = set()
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), 1):
        if not linha.strip() or linha.startswith("#"):
            continue
        try:
            cenario = Cenario.model_validate_json(linha)
        except ValueError as exc:
            raise CenariosInvalidos(f"{caminho}:{numero}: {exc}") from exc
        if cenario.id in vistos:
            raise CenariosInvalidos(f"{caminho}:{numero}: id duplicado {cenario.id!r}")
        vistos.add(cenario.id)
        cenarios.append(cenario)
    return cenarios


class SorteioDeCenarios:
    """Uma fila por tipo de tarefa, sem reposicao, reembaralhada quando
    acaba -- o mesmo criterio do sorteio de artigos no plano: mesma seed,
    mesma sequencia. Tipo sem cenario cadastrado recebe None."""

    def __init__(self, cenarios: list[Cenario], rng: random.Random) -> None:
        self._rng = rng
        self._por_tipo: dict[TipoDeTarefa, list[Cenario]] = {}
        for cenario in cenarios:
            for tipo in cenario.tipos:
                self._por_tipo.setdefault(tipo, []).append(cenario)
        self._filas: dict[TipoDeTarefa, list[Cenario]] = {}

    def proximo(self, tipo: TipoDeTarefa) -> Cenario | None:
        base = self._por_tipo.get(tipo)
        if not base:
            return None
        fila = self._filas.get(tipo)
        if not fila:
            fila = list(base)
            self._rng.shuffle(fila)
            self._filas[tipo] = fila
        return fila.pop()
