"""Vocabulario compartilhado por toda a API."""

from __future__ import annotations

from enum import StrEnum


class Etapa(StrEnum):
    """Etapa do pipeline da pesquisa a que uma chamada de LLM pertence. E' o
    eixo do agregado de consumo (`GET /usage/llm`): a pergunta que o TCC
    responde e' "quanto custou GERAR, FILTRAR e AVALIAR", nao "quanto custou
    no total"."""

    CORPUS = "corpus"
    SINTESE = "sintese"
    FILTRO = "filtro"
    DATASET = "dataset"
    TREINO = "treino"
    AVALIACAO = "avaliacao"
    ON_POLICY = "on_policy"
    DIAGNOSTICO = "diagnostico"
