"""Tabela de precos por modelo -- converte o VOLUME de tokens acumulado pela
telemetria em custo estimado (USD).

Tres decisoes de desenho, herdadas do projeto de referencia:

1. **Modelo desconhecido devolve `None`, nunca 0.** Um custo zerado para um
   modelo fora da tabela seria uma mentira silenciosa; `None` propaga como
   "nao sei" e o relatorio lista o modelo em `modelos_sem_preco`.
2. **Precos sao ESTIMATIVA, nao fatura.** Servem para comparar professor x
   aluno na mesma ordem de grandeza -- a pergunta do TCC.
3. **Sobrescreviveis por ambiente** (`LLM_PRICING_JSON`): preco de tabela
   muda mais rapido que release.

EXCECAO DELIBERADA a (1): o provedor `local` custa ZERO por token, e isso
e' fato, nao lacuna -- o custo do aluno local e' medido em tempo, RAM e
hardware pela telemetria de avaliacao, nao aqui.

PRECOS EM USD POR 1 MILHAO DE TOKENS. `gpt-5.6-luna` conferido em 2026-09-07
(projeto de referencia) contra developers.openai.com/api/docs/pricing; os
demais da OpenAI entraram `verified=False`. Anthropic conferidos em 2026-07.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25
ANTHROPIC_CACHE_READ_MULTIPLIER = 0.1

_ENV_OVERRIDE = "LLM_PRICING_JSON"
_PREFIXO_LOCAL = "local:"


@dataclass(frozen=True)
class ModelPricing:
    """`cache_write`/`cache_read` a `None` = derivar de `input` pelos
    multiplicadores da Anthropic. Modelo OpenAI declara os dois explicitos:
    uma mudanca de politica do provedor nao pode virar heranca silenciosa."""

    input: float
    output: float
    cache_write: float | None = None
    cache_read: float | None = None
    verified: bool = True

    def effective_cache_write(self) -> float:
        if self.cache_write is not None:
            return self.cache_write
        return self.input * ANTHROPIC_CACHE_WRITE_MULTIPLIER

    def effective_cache_read(self) -> float:
        if self.cache_read is not None:
            return self.cache_read
        return self.input * ANTHROPIC_CACHE_READ_MULTIPLIER


_PRECO_LOCAL = ModelPricing(input=0.0, output=0.0, cache_write=0.0, cache_read=0.0)

# Chaves provider-qualificadas, exatamente como `resolve_llm_id` devolve.
_PRICING: dict[str, ModelPricing] = {
    "anthropic:claude-haiku-4-5": ModelPricing(input=1.00, output=5.00),
    "anthropic:claude-sonnet-5": ModelPricing(input=3.00, output=15.00),
    "anthropic:claude-opus-4-5": ModelPricing(input=5.00, output=25.00),
    # Familia 5.6 COBRA a escrita de cache (1,25x o input). Tarifa de contexto
    # curto: acima de 272K tokens de entrada a requisicao inteira sai a 2x/1,5x
    # e esta tabela subestima -- os lotes deste projeto ficam longe disso.
    "openai:gpt-5.6-luna": ModelPricing(
        input=0.20, output=1.20, cache_write=0.25, cache_read=0.02, verified=True
    ),
    "openai:gpt-5": ModelPricing(input=1.25, output=10.00, cache_write=0.0, cache_read=0.125, verified=False),
    "openai:gpt-5-mini": ModelPricing(input=0.25, output=2.00, cache_write=0.0, cache_read=0.025, verified=False),
    "openai:gpt-5-nano": ModelPricing(input=0.05, output=0.40, cache_write=0.0, cache_read=0.005, verified=False),
    "openai:gpt-5.4-mini": ModelPricing(input=0.75, output=4.50, cache_write=0.0, cache_read=0.025, verified=True),
    "openai:gpt-5.4-nano": ModelPricing(input=0.20, output=1.25, cache_write=0.0, cache_read=0.005, verified=True),
}


@lru_cache(maxsize=4)
def _parse_override(raw: str) -> dict[str, ModelPricing]:
    # Memoizado pelo VALOR bruto: mudar o ambiente vale sem restart, e o
    # tracker nao reparseia o mesmo JSON a cada chamada.
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {
            model: ModelPricing(
                input=float(p["input"]),
                output=float(p["output"]),
                cache_write=None if p.get("cache_write") is None else float(p["cache_write"]),
                cache_read=None if p.get("cache_read") is None else float(p["cache_read"]),
                verified=bool(p.get("verified", True)),
            )
            for model, p in data.items()
        }
    except Exception:
        logger.warning("%s malformado -- ignorado, usando a tabela embutida.", _ENV_OVERRIDE, exc_info=True)
        return {}


def pricing_for(model_id: str) -> ModelPricing | None:
    """Preco do modelo (id provider-qualificado), ou None se desconhecido."""

    override = _parse_override(os.getenv(_ENV_OVERRIDE, "").strip())
    if model_id in override:
        return override[model_id]
    if model_id in _PRICING:
        return _PRICING[model_id]
    if model_id.startswith(_PREFIXO_LOCAL):
        return _PRECO_LOCAL
    return None


def estimate_cost_usd(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float | None:
    """Custo estimado em USD, ou None quando o modelo nao tem preco. Os quatro
    contadores tem precos diferentes -- somar tudo e multiplicar por um preco
    unico erraria por ate 12x entre input em cache e output."""

    price = pricing_for(model_id)
    if price is None:
        return None
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_creation_input_tokens * price.effective_cache_write()
        + cache_read_input_tokens * price.effective_cache_read()
    ) / 1_000_000


def is_verified(model_id: str) -> bool:
    price = pricing_for(model_id)
    return price is not None and price.verified
