"""Configuracao central: tudo o que e' parametrizavel vem do ambiente (`.env`),
lido por funcoes pequenas com default e validacao de faixa.

Funcoes, e nao um objeto Settings congelado no import: os testes mexem no
ambiente com `monkeypatch` e precisam ver o valor novo na leitura seguinte.
Cada leitura custa um `os.getenv` -- barato, e nenhuma delas esta no caminho
quente de uma chamada de LLM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega `.env` sem sobrescrever variaveis ja setadas no ambiente real --
# e' o que permite ao CI e ao Docker injetarem valores por cima do arquivo.
load_dotenv()

logger = logging.getLogger(__name__)

APP_ENV_TEST = "test"
APP_ENV_DEVELOPMENT = "development"
APP_ENV_PRODUCTION = "production"
_APP_ENVS = (APP_ENV_TEST, APP_ENV_DEVELOPMENT, APP_ENV_PRODUCTION)


def _env_int(nome: str, default: int, minimo: int, maximo: int) -> int:
    raw = os.getenv(nome, "").strip()
    if not raw:
        return default
    try:
        return max(minimo, min(maximo, int(raw)))
    except ValueError:
        logger.warning("%s=%r nao e' um inteiro valido; usando %d.", nome, raw, default)
        return default


def _env_float(nome: str, default: float, minimo: float, maximo: float) -> float:
    raw = os.getenv(nome, "").strip()
    if not raw:
        return default
    try:
        return max(minimo, min(maximo, float(raw)))
    except ValueError:
        logger.warning("%s=%r nao e' numerico; usando %s.", nome, raw, default)
        return default


def app_env() -> str:
    """Ambiente declarado (`APP_ENV`). Valor desconhecido vira `production`:
    errar para o lado restritivo e' o unico erro aceitavel numa variavel que
    decide se a autenticacao pode ser desligada."""

    raw = os.getenv("APP_ENV", APP_ENV_DEVELOPMENT).strip().lower()
    if raw not in _APP_ENVS:
        logger.warning("APP_ENV=%r desconhecido; tratando como %s.", raw, APP_ENV_PRODUCTION)
        return APP_ENV_PRODUCTION
    return raw


def data_dir() -> Path:
    """Raiz de datasets, telemetria e artefatos (`DATA_DIR`, default `data/`)."""

    return Path(os.getenv("DATA_DIR", "data").strip() or "data")


def log_format() -> str:
    """`text` (default) ou `json`."""

    return os.getenv("LOG_FORMAT", "text").strip().lower()


def llm_timeout_s() -> float:
    """Timeout de UMA chamada de chat model (s). O default do SDK e' 10 min --
    tempo demais para um lote que o operador esta esperando."""

    return _env_float("LLM_TIMEOUT_S", 120.0, 5.0, 900.0)


def llm_max_retries() -> int:
    """Retries do SDK por erro de rede/429. Nao confundir com o retry de
    VALIDACAO de saida de `ainvoke_structured`."""

    return _env_int("LLM_MAX_RETRIES", 2, 0, 5)


def llm_budget_usd_por_job() -> float:
    """Teto de custo estimado (USD) por job de LLM. 0 desliga."""

    return _env_float("LLM_BUDGET_USD_POR_JOB", 5.0, 0.0, 10_000.0)


def llm_circuit_failures() -> int:
    return _env_int("LLM_CIRCUIT_FAILURES", 3, 1, 100)


def llm_circuit_cooldown_s() -> float:
    return _env_float("LLM_CIRCUIT_COOLDOWN_S", 30.0, 1.0, 3600.0)
