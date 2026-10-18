"""Isolamento de ambiente para a suite inteira.

As variaveis do `.env` de quem roda (provedores, modelos, chaves, orcamento)
NAO podem vazar para os testes: uma suite que passa ou falha conforme o
`.env` da maquina vira ruido que o time aprende a ignorar. Aqui tudo e'
limpo antes de cada teste; quem precisa de um valor o define com
`monkeypatch`.
"""

from __future__ import annotations

import pytest

from app.core import llm_gate
from app.core.llm_usage import tracker

_ENV_DE_LLM = (
    "TEACHER_PROVIDER", "JUDGE_PROVIDER", "STUDENT_PROVIDER",
    "OPENAI_MODEL_TEACHER", "OPENAI_MODEL_JUDGE", "OPENAI_MODEL_STUDENT",
    "ANTHROPIC_MODEL_TEACHER", "ANTHROPIC_MODEL_JUDGE", "ANTHROPIC_MODEL_STUDENT",
    "LOCAL_MODEL_TEACHER", "LOCAL_MODEL_JUDGE", "LOCAL_MODEL_STUDENT",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_LLM_API_KEY", "LOCAL_LLM_BASE_URL",
    "LLM_PRICING_JSON", "LLM_BUDGET_USD_POR_JOB", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES",
    "LLM_CIRCUIT_FAILURES", "LLM_CIRCUIT_COOLDOWN_S",
)

_ENV_DE_INFRA = ("AUTH_MODE", "AUTH_API_KEYS", "APP_ENV", "DATA_DIR", "LOG_FORMAT")


@pytest.fixture(autouse=True)
def ambiente_isolado(monkeypatch, tmp_path):
    for nome in _ENV_DE_LLM + _ENV_DE_INFRA:
        monkeypatch.delenv(nome, raising=False)
    # `disabled` so' e' aceito em APP_ENV=test: a suite precisa declarar-se como tal.
    monkeypatch.setenv("AUTH_MODE", "disabled")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    tracker.limpar_para_teste()
    llm_gate.limpar_estado_para_teste()
    yield
    tracker.limpar_para_teste()
    llm_gate.limpar_estado_para_teste()
