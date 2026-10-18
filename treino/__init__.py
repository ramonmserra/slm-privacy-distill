"""Receitas de TREINO -- rodam FORA da API (Colab T4 ou CPU local).

A API exporta o dataset (`/datasets/{versao}/arquivo/{split}`) e registra o
run (`POST /treinos`); o que acontece entre os dois e' este pacote. E' o
unico lugar do projeto onde nomes de modelo-base (Qwen/...) aparecem fora
de `app/core/llm.py`, de proposito: a receita e' externa a API.
"""
