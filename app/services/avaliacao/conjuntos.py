"""Os conjuntos de avaliacao, carregados numa forma unica (`ItemDeAvaliacao`).

- BANCO_INICIAL: as 34 perguntas redigidas pelo grupo, com artigos de
  referencia e criterios esperados (`recursos/avaliacao/banco_inicial_lgpd.jsonl`).
  A referencia e' a RUBRICA, nao uma resposta-modelo.
- HELD_OUT: o split held-out de uma versao de dataset. A referencia e' a
  resposta do professor (sem o prefixo "Raciocinio:"), e o gold de citacao
  sao os artigos que ele citou.

A ordem e' estavel (arquivo), entao `limite` recorta sempre os mesmos itens
e dois runs no mesmo conjunto comparam os mesmos ids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.repositories.dataset_repository import DatasetRepository
from app.schemas.avaliacao import Conjunto, ItemDeAvaliacao
from app.schemas.dataset import Split
from app.schemas.sintese import TipoDeTarefa
from app.services.corpus.referencias import artigo_de, normalizar_referencias

BANCO_INICIAL = Path("recursos/avaliacao/banco_inicial_lgpd.jsonl")

# Itens do banco cuja resposta correta e' NAO concluir: o Teste C pede que o
# modelo liste as informacoes faltantes em vez de autorizar.
_BANCO_ABSTENCAO_ESPERADA = frozenset({"TC"})


class ConjuntoIndisponivel(ValueError):
    pass


def _artigos_de_referencia(referencia: str | None) -> list[str]:
    if not referencia:
        return []
    artigos: list[str] = []
    for id_ in normalizar_referencias(referencia):
        raiz = artigo_de(id_)
        if raiz not in artigos:
            artigos.append(raiz)
    return artigos


def carregar_banco_inicial(caminho: Path = BANCO_INICIAL) -> list[ItemDeAvaliacao]:
    if not caminho.exists():
        raise ConjuntoIndisponivel(f"{caminho} nao existe.")
    itens: list[ItemDeAvaliacao] = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        bruto = json.loads(linha)
        itens.append(
            ItemDeAvaliacao(
                id=bruto["id"], conjunto=Conjunto.BANCO_INICIAL, pergunta=bruto["pergunta"],
                referencia=bruto["criterios_esperados"], artigos_referencia=_artigos_de_referencia(bruto.get("referencia")),
                tipo=bruto.get("tipo", "pergunta"), abstencao_esperada=bruto["id"] in _BANCO_ABSTENCAO_ESPERADA,
                rotulo=bruto.get("titulo", ""),
            )
        )
    return itens


def _sem_prefixo_de_raciocinio(conteudo: str) -> str:
    marcador = "\n\nResposta:\n"
    if conteudo.startswith("Raciocinio:") and marcador in conteudo:
        return conteudo.split(marcador, 1)[1].strip()
    return conteudo.strip()


def carregar_held_out(dataset_versao: str, repositorio: DatasetRepository) -> list[ItemDeAvaliacao]:
    manifesto = repositorio.carregar(dataset_versao)
    if manifesto is None:
        raise ConjuntoIndisponivel(f"Dataset {dataset_versao!r} nao existe.")
    linhas = repositorio.linhas(dataset_versao, Split.HELD_OUT)
    if not linhas:
        raise ConjuntoIndisponivel(f"Dataset {dataset_versao!r} nao tem held-out.")
    itens: list[ItemDeAvaliacao] = []
    for l in linhas:
        mensagens = {m["role"]: m["content"] for m in l["messages"]}
        itens.append(
            ItemDeAvaliacao(
                id=l["id"], conjunto=Conjunto.HELD_OUT, pergunta=mensagens["user"],
                referencia=_sem_prefixo_de_raciocinio(mensagens["assistant"]),
                artigos_referencia=sorted({artigo_de(a) for a in l.get("artigos_citados", [])}),
                tipo=l.get("tipo", ""), abstencao_esperada=l.get("tipo") == TipoDeTarefa.SEM_RESPOSTA.value,
            )
        )
    return itens


def versao_do_conjunto(conjunto: Conjunto, dataset_versao: str | None, caminho_do_banco: Path = BANCO_INICIAL) -> str:
    """Identidade do conjunto para a versao da medida: hash do arquivo do
    banco, ou a versao do dataset (que ja e' hash do conteudo)."""

    if conjunto == Conjunto.HELD_OUT:
        return f"held_out:{dataset_versao}"
    return "banco_inicial:" + hashlib.sha256(caminho_do_banco.read_bytes()).hexdigest()[:12]
