"""Contratos do registro de TREINO. O treino roda fora da API; aqui fica o
que o torna reproduzivel e avaliavel: qual dataset, qual base, quais
hiperparametros, em que hardware, com que resultado, e sob qual id local o
modelo treinado passa a ser servido.

`RegistroDeRun` e' exatamente o `run.json` que `treino/sft_lora.py` grava
(mesmos nomes de campo): o corpo de `POST /treinos` e' o arquivo, sem
traducao manual.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class RegistroDeRun(BaseModel):
    dataset_versao: str = Field(min_length=1)
    modelo_base: str = Field(min_length=1, description="Ex.: Qwen/Qwen3-1.7B")
    perfil: str = ""
    receita: str = "sft_lora"
    hiperparametros: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    duracao_s: float = Field(default=0.0, ge=0)
    metricas: dict[str, Any] = Field(default_factory=dict)
    artefato: str = Field(default="", description="Caminho/URI do adapter, merged ou GGUF")
    modelo_local_id: str = Field(
        pattern=r"^local:[A-Za-z0-9][A-Za-z0-9._:-]*$",
        description="Id com que o modelo treinado e' servido: local:<base>-lgpd-<8 chars da versao do dataset>",
    )
    status: str = Field(pattern=r"^(registrado|dry_run|concluido|falhou)$")
    observacoes: str = ""
    iniciado_em: dt.datetime | None = None
    concluido_em: dt.datetime | None = None
    exemplos_de_treino: int = Field(default=0, ge=0)
    exemplos_de_held_out: int = Field(default=0, ge=0)
    prefixo_do_assistente: str = ""


class RunDeTreino(RegistroDeRun):
    id: str
    criado_em: dt.datetime
    disponivel_localmente: bool | None = Field(default=None, description="O servidor local serve `modelo_local_id`? null = nao verificado")
    verificado_em: dt.datetime | None = None


class VerificacaoLocal(BaseModel):
    modelo_local_id: str
    disponivel: bool
    modelos_servidos: list[str]
    base_url: str
