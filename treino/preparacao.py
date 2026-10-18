"""Preparacao do treino SEM dependencias pesadas: le o JSONL exportado pela
API, monta as mensagens no formato que o tokenizer do modelo-base espera,
escolhe o perfil de hardware e escreve o `run.json` que a API registra.

Separado de `sft_lora.py` para ser testavel sem torch/transformers -- a
parte que decide O QUE o modelo ve no treino e' esta, e e' a que mais
precisa de teste.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Qwen3 em modo "sem pensar": o template oficial abre o turno do assistente
# com um bloco <think> vazio. O treino REPETE esse prefixo na resposta para
# que o que o modelo aprende seja exatamente o que o Ollama vai pedir na
# inferencia (ver Modelfile.template). Para outro modelo-base, passe "".
PREFIXO_SEM_PENSAR_QWEN3 = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class Perfil:
    """Hiperparametros por hardware. `colab-t4`: 16 GB, so' fp16 (T4 nao tem
    bf16), LoRA no 1.7B com checkpointing. `cpu-local`: validacao da receita
    com o 0.6B, sequencias curtas, 1 epoca -- prova o pipeline, nao treina
    para valer (ver README)."""

    nome: str
    modelo_base: str
    epocas: float
    taxa_de_aprendizado: float
    tamanho_do_lote: int
    acumulacao: int
    comprimento_maximo: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    precisao: str               # fp16 | bf16 | fp32
    checkpointing: bool
    quatro_bits: bool           # bitsandbytes 4-bit (so' GPU)

    def como_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PERFIS: dict[str, Perfil] = {
    "colab-t4": Perfil(
        nome="colab-t4", modelo_base="Qwen/Qwen3-1.7B", epocas=2, taxa_de_aprendizado=2e-4,
        tamanho_do_lote=2, acumulacao=8, comprimento_maximo=1024, lora_r=16, lora_alpha=32,
        lora_dropout=0.05, precisao="fp16", checkpointing=True, quatro_bits=False,
    ),
    "colab-t4-4bit": Perfil(
        nome="colab-t4-4bit", modelo_base="Qwen/Qwen3-4B", epocas=2, taxa_de_aprendizado=2e-4,
        tamanho_do_lote=1, acumulacao=16, comprimento_maximo=1024, lora_r=16, lora_alpha=32,
        lora_dropout=0.05, precisao="fp16", checkpointing=True, quatro_bits=True,
    ),
    "cpu-local": Perfil(
        nome="cpu-local", modelo_base="Qwen/Qwen3-0.6B", epocas=1, taxa_de_aprendizado=2e-4,
        tamanho_do_lote=1, acumulacao=8, comprimento_maximo=512, lora_r=8, lora_alpha=16,
        lora_dropout=0.05, precisao="fp32", checkpointing=False, quatro_bits=False,
    ),
}

MODULOS_LORA_QWEN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def carregar_jsonl(caminho: Path) -> list[dict]:
    linhas: list[dict] = []
    with caminho.open(encoding="utf-8") as f:
        for numero, bruta in enumerate(f, start=1):
            bruta = bruta.strip()
            if not bruta:
                continue
            try:
                linhas.append(json.loads(bruta))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{caminho}: linha {numero} invalida: {exc}") from exc
    return linhas


def validar_linha(linha: dict) -> None:
    mensagens = linha.get("messages")
    if not isinstance(mensagens, list) or [m.get("role") for m in mensagens] != ["system", "user", "assistant"]:
        raise ValueError(f"linha {linha.get('id')!r}: esperado messages [system, user, assistant]")
    if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in mensagens):
        raise ValueError(f"linha {linha.get('id')!r}: mensagem vazia")


def montar_mensagens(linha: dict, prefixo_do_assistente: str) -> list[dict[str, str]]:
    """As tres mensagens, com o prefixo (ex.: bloco <think> vazio) anteposto
    a resposta. E' a UNICA transformacao entre o JSONL da API e o template
    de chat do tokenizer -- e por isso e' uma funcao pura testada."""

    validar_linha(linha)
    mensagens = [{"role": m["role"], "content": m["content"]} for m in linha["messages"]]
    if prefixo_do_assistente and not mensagens[2]["content"].startswith(prefixo_do_assistente):
        mensagens[2]["content"] = prefixo_do_assistente + mensagens[2]["content"]
    return mensagens


def preparar(linhas: list[dict], prefixo_do_assistente: str) -> list[dict[str, Any]]:
    return [{"id": l.get("id"), "tipo": l.get("tipo"), "messages": montar_mensagens(l, prefixo_do_assistente)} for l in linhas]


def detectar_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "plataforma": platform.platform(),
        "processador": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "gpu": None,
        "vram_gb": None,
    }
    try:  # torch e' opcional aqui: a deteccao roda tambem no dry-run sem ele
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except ImportError:
        info["torch"] = None
    return info


@dataclass
class RegistroDoRun:
    """Espelha o payload de `POST /treinos` da API: o `run.json` que o script
    grava e' o corpo da requisicao de registro, sem traducao manual."""

    dataset_versao: str
    modelo_base: str
    perfil: str
    receita: str = "sft_lora"
    hiperparametros: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=detectar_hardware)
    duracao_s: float = 0.0
    metricas: dict[str, Any] = field(default_factory=dict)
    artefato: str = ""
    modelo_local_id: str = ""
    status: str = "registrado"
    observacoes: str = ""
    iniciado_em: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    concluido_em: str | None = None
    exemplos_de_treino: int = 0
    exemplos_de_held_out: int = 0
    prefixo_do_assistente: str = ""

    def gravar(self, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def modelo_local_id_sugerido(modelo_base: str, dataset_versao: str) -> str:
    """`Qwen/Qwen3-1.7B` + versao `ab12cd34ef56` -> `local:qwen3-1.7b-lgpd-ab12cd34`.
    E' o nome que o Ollama vai receber e que a API vai avaliar."""

    base = modelo_base.split("/")[-1].lower()
    return f"local:{base}-lgpd-{dataset_versao[:8]}"
