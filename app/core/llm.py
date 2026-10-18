"""Camada unica de provedor de LLM -- o UNICO modulo que sabe que provedores
existem e que nomes de modelo eles usam.

INVARIANTE: nenhum nome de modelo (`gpt-*`, `claude-*`, `qwen*`) aparece fora
daqui. Um call site pede um PAPEL (`ModelTier`) ou repassa um override que
veio de fora (um id ja qualificado, ex.: `local:qwen3-1.7b-lgpd-v2`).
Verificavel:

    grep -rnE "gpt-|claude-|qwen" --include=*.py app | grep -v "app/core/llm"

## Os tres papeis da pesquisa

| Papel | O que faz | Provedor default |
|---|---|---|
| `TEACHER` | gera dados sinteticos e cadeias de raciocinio; corrige na rodada on-policy | `openai` |
| `JUDGE` | avalia professor E aluno; nunca e' destilado | `anthropic` |
| `STUDENT` | o SLM medido, servido localmente | `local` |

Cada papel tem o proprio provedor (`{PAPEL}_PROVIDER`) e o modelo vem de
`{PROVEDOR}_MODEL_{PAPEL}`. REGRA DA PESQUISA: professor e juiz de familias
diferentes (vies de autopreferencia); `avisos_de_configuracao` denuncia
quando coincidem.

## Provedor `local`

Qualquer servidor OpenAI-compatible (Ollama, vLLM, llama.cpp) em
`LOCAL_LLM_BASE_URL`. Trocar o aluno e' trocar `LOCAL_MODEL_STUDENT`; uma
versao treinada vira um id novo (`local:qwen3-1.7b-lgpd-v2`), e rollback e'
reescrever a string.

TRANSPORTE: LangChain (`init_chat_model`). Os contratos de dominio continuam
sendo os `Protocol` de cada servico, que nao mencionam provedor nem framework
e aceitam dublês em teste.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from app.core.config import llm_max_retries, llm_timeout_s
from app.core.erros import StructuredOutputError

if TYPE_CHECKING:  # evita importar langchain no import-time de quem so' tipa
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_LOCAL = "local"


class ModelTier(StrEnum):
    TEACHER = "teacher"
    JUDGE = "judge"
    STUDENT = "student"


_PROVIDER_ENV_BY_TIER: dict[ModelTier, str] = {
    ModelTier.TEACHER: "TEACHER_PROVIDER",
    ModelTier.JUDGE: "JUDGE_PROVIDER",
    ModelTier.STUDENT: "STUDENT_PROVIDER",
}

_DEFAULT_PROVIDER_BY_TIER: dict[ModelTier, str] = {
    ModelTier.TEACHER: PROVIDER_OPENAI,
    ModelTier.JUDGE: PROVIDER_ANTHROPIC,
    ModelTier.STUDENT: PROVIDER_LOCAL,
}

# DEFAULTS por provedor e papel -- o unico lugar do projeto onde nome de
# modelo e' escrito. `{PROVEDOR}_MODEL_{PAPEL}` no ambiente tem precedencia.
# Todo provedor atende todo papel: e' o que permite, por exemplo, medir o
# professor como "aluno" (condicao C0 do desenho experimental) ou usar um
# modelo local como juiz num teste de fumaca sem gastar API.
_MODEL_BY_PROVIDER_TIER: dict[str, dict[ModelTier, str]] = {
    PROVIDER_OPENAI: {
        ModelTier.TEACHER: "gpt-5.6-luna",
        ModelTier.JUDGE: "gpt-5.6-luna",
        ModelTier.STUDENT: "gpt-5-nano",
    },
    PROVIDER_ANTHROPIC: {
        ModelTier.TEACHER: "claude-sonnet-5",
        ModelTier.JUDGE: "claude-sonnet-5",
        ModelTier.STUDENT: "claude-haiku-4-5",
    },
    # Tags do Ollama. Aluno default: Qwen3-1.7B (segue instrucao, produz JSON,
    # GGUF Q4 roda em CPU; o Manaca-1B so' existe em versao base).
    PROVIDER_LOCAL: {
        ModelTier.TEACHER: "qwen3:1.7b",
        ModelTier.JUDGE: "qwen3:1.7b",
        ModelTier.STUDENT: "qwen3:1.7b",
    },
}

_CREDENTIAL_ENV: dict[str, str] = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    # Servidores locais nao autenticam; alguns exigem um token qualquer.
    PROVIDER_LOCAL: "LOCAL_LLM_API_KEY",
}
_LOCAL_API_KEY_DEFAULT = "ollama"
_ENV_LOCAL_BASE_URL = "LOCAL_LLM_BASE_URL"
_LOCAL_BASE_URL_DEFAULT = "http://localhost:11434/v1"

# Comportamentos POR MODELO observados contra a API (projeto de referencia).
# Entrada nova so' com o mesmo teste -- adivinhar troca um 400 explicito por
# comportamento silencioso.

# Recusa `temperature` (400): o parametro sai do payload, com aviso.
_MODELOS_SEM_TEMPERATURE = frozenset({"claude-sonnet-5"})

# So' aceita ferramenta e preserva `usage_metadata` pela Responses API.
_MODELOS_SO_COM_RESPONSES_API = frozenset({"gpt-5.6-luna"})

# Cobra a escrita do cache de prompt (1,25x o input). Chamador cujos prompts
# nao reusam prefixo pede a requisicao sem cache.
_MODELOS_QUE_COBRAM_ESCRITA_DE_CACHE = frozenset({"gpt-5.6-luna"})

# Modelo de raciocinio da OpenAI: o pensamento sai do MESMO teto de saida.
# Reserva somada ao `max_tokens` para o pedido do call site continuar sendo
# uma promessa sobre a RESPOSTA. Medido: taxonomias truncadas com 2.048 tokens
# todos de raciocinio; 8.192 cobre com folga.
_RESERVA_DE_RACIOCINIO: dict[str, int] = {"gpt-5.6-luna": 8192}

_avisados_sem_temperatura: set[str] = set()


class UnknownProviderError(ValueError):
    """Provedor fora de `_MODEL_BY_PROVIDER_TIER` -- typo em `*_PROVIDER` ou
    override sem o prefixo `provedor:`."""


def provedores() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_BY_PROVIDER_TIER))


def split_llm_id(model_id: str) -> tuple[str, str]:
    """`"local:qwen3:1.7b"` -> `("local", "qwen3:1.7b")`. Separa no PRIMEIRO
    `:`, porque tags do Ollama tem `:` dentro do nome. Id sem provedor
    conhecido levanta: aqui nao ha provedor legado a assumir, e um nome cru
    enviado ao provedor errado seria um 401 dificil de ler."""

    provider, sep, model = model_id.partition(":")
    provider = provider.strip().lower()
    if not sep or provider not in _MODEL_BY_PROVIDER_TIER:
        raise UnknownProviderError(
            f"Identificador de modelo {model_id!r} precisa do prefixo de provedor "
            f"({', '.join(provedores())}), ex.: 'local:qwen3:1.7b'."
        )
    return provider, model.strip()


def provider_for_tier(tier: ModelTier) -> str:
    """Provedor do papel: `{PAPEL}_PROVIDER` > default do papel. Valor
    desconhecido cai no default COM aviso -- um typo que degradasse em
    silencio faria o grupo achar que mediu um modelo e ter medido outro."""

    env = _PROVIDER_ENV_BY_TIER[tier]
    raw = os.getenv(env, "").strip().lower()
    if not raw:
        return _DEFAULT_PROVIDER_BY_TIER[tier]
    if raw not in _MODEL_BY_PROVIDER_TIER:
        logger.warning(
            "%s=%r nao e' um provedor conhecido (%s) -- usando %r.",
            env, raw, ", ".join(provedores()), _DEFAULT_PROVIDER_BY_TIER[tier],
        )
        return _DEFAULT_PROVIDER_BY_TIER[tier]
    return raw


def model_env_var(provider: str, tier: ModelTier) -> str:
    """Ex.: `OPENAI_MODEL_TEACHER`. Funcao, e nao constante, porque o
    diagnostico (`GET /usage/providers`) mostra ao operador qual variavel
    mexer."""

    return f"{provider.upper()}_MODEL_{tier.value.upper()}"


def model_for_tier(tier: ModelTier, provider: str | None = None) -> str:
    """Nome CRU do modelo do papel no provedor. Ordem: variavel de ambiente
    `{PROVEDOR}_MODEL_{PAPEL}` > tabela embutida."""

    resolved = provider or provider_for_tier(tier)
    if resolved not in _MODEL_BY_PROVIDER_TIER:
        raise UnknownProviderError(f"Provedor {resolved!r} desconhecido. Conhecidos: {provedores()}.")
    override = os.getenv(model_env_var(resolved, tier), "").strip()
    if override:
        return override
    return _MODEL_BY_PROVIDER_TIER[resolved][tier]


def tier_model_source(tier: ModelTier) -> str:
    """`"env"` se `{PROVEDOR}_MODEL_{PAPEL}` esta definido, senao `"default"`."""

    provider = provider_for_tier(tier)
    return "env" if os.getenv(model_env_var(provider, tier), "").strip() else "default"


def resolve_llm_id(tier: ModelTier, override: str | None = None) -> str:
    """Identificador ESTAVEL e provider-qualificado do modelo a usar.

    1. `override` qualificado (`"local:qwen3-1.7b-lgpd-v2"`): o experimento
       fixou um modelo exato; o papel nao interfere.
    2. papel -> provedor do papel -> modelo do papel.

    E' o que `init_chat_model` aceita, o que entra como chave na telemetria e
    o que um run de avaliacao grava como "modelo medido"."""

    if override:
        provider, model = split_llm_id(override)
        return f"{provider}:{model}"
    provider = provider_for_tier(tier)
    return f"{provider}:{model_for_tier(tier, provider)}"


def credencial(provider: str) -> str | None:
    """Credencial do provedor no ambiente, ou None. O provedor local devolve o
    token placeholder: ele nao autentica, mas o SDK exige algum valor."""

    env = _CREDENTIAL_ENV.get(provider)
    if env is None:
        raise UnknownProviderError(f"Provedor {provider!r} desconhecido.")
    valor = os.getenv(env, "").strip()
    if provider == PROVIDER_LOCAL:
        return valor or _LOCAL_API_KEY_DEFAULT
    return valor or None


def llm_available(provider: str) -> bool:
    return credencial(provider) is not None


def local_base_url() -> str:
    return os.getenv(_ENV_LOCAL_BASE_URL, "").strip() or _LOCAL_BASE_URL_DEFAULT


def aceita_temperature(model_id: str) -> bool:
    _, model = split_llm_id(model_id)
    return model not in _MODELOS_SEM_TEMPERATURE


def reserva_de_raciocinio(model_id: str) -> int:
    _, model = split_llm_id(model_id)
    return _RESERVA_DE_RACIOCINIO.get(model, 0)


def avisos_de_configuracao() -> list[str]:
    """Problemas de configuracao que NAO impedem o processo de subir mas
    comprometem a pesquisa. Expostos em `GET /usage/providers`."""

    avisos: list[str] = []
    professor = provider_for_tier(ModelTier.TEACHER)
    juiz = provider_for_tier(ModelTier.JUDGE)
    if professor == juiz:
        avisos.append(
            f"Professor e juiz no MESMO provedor ({professor}): vies de autopreferencia "
            "no julgamento. Aponte JUDGE_PROVIDER para outra familia."
        )
    for tier in ModelTier:
        provider = provider_for_tier(tier)
        if not llm_available(provider):
            avisos.append(
                f"Papel {tier.value}: credencial do provedor {provider!r} ausente "
                f"({_CREDENTIAL_ENV[provider]})."
            )
    return avisos


def build_chat_model(
    model_id: str, *, max_tokens: int, sem_cache_de_prompt: bool = False, **kwargs: Any
) -> "BaseChatModel":
    """Constroi o chat model do LangChain -- o unico ponto que traduz um
    identificador em cliente de provedor.

    Kwarg com valor None e' OMITIDO do payload: e' como um chamador diz "sem
    opiniao sobre este parametro, use o default do provedor" numa expressao
    so'. A chave vem explicita do ambiente: deixar o SDK ler `os.environ`
    esconderia qual variavel faltou quando a chamada falhar."""

    from langchain.chat_models import init_chat_model

    provider, model = split_llm_id(model_id)
    definidos = {k: v for k, v in kwargs.items() if v is not None}

    if "temperature" in definidos and not aceita_temperature(model_id):
        definidos.pop("temperature")
        if model_id not in _avisados_sem_temperatura:
            _avisados_sem_temperatura.add(model_id)
            logger.info(
                "%s nao aceita `temperature`; o parametro foi omitido e o modelo "
                "roda na amostragem default do provedor.", model_id,
            )

    if "api_key" not in definidos:
        chave = credencial(provider)
        if chave is None:
            raise RuntimeError(
                f"Credencial do provedor {provider!r} ausente: defina {_CREDENTIAL_ENV[provider]} no .env."
            )
        definidos["api_key"] = chave
    # Explicitos, nunca o default do SDK (10 min): um provedor pendurado
    # seguraria o lote inteiro por esse tempo.
    definidos.setdefault("timeout", llm_timeout_s())
    definidos.setdefault("max_retries", llm_max_retries())

    max_tokens += reserva_de_raciocinio(model_id)
    if model in _MODELOS_SO_COM_RESPONSES_API:
        definidos.setdefault("use_responses_api", True)
    if sem_cache_de_prompt and model in _MODELOS_QUE_COBRAM_ESCRITA_DE_CACHE:
        definidos.setdefault("prompt_cache_options", {"mode": "explicit"})

    # O servidor local fala o protocolo da OpenAI: mesmo transporte, outro
    # endereco. `model_provider="openai"` e' TRANSPORTE; a identidade do
    # projeto continua sendo `local:<modelo>`.
    transporte = PROVIDER_OPENAI if provider == PROVIDER_LOCAL else provider
    if provider == PROVIDER_LOCAL:
        definidos.setdefault("base_url", local_base_url())
    return init_chat_model(model, model_provider=transporte, max_tokens=max_tokens, **definidos)


def structured_model(chat_model: "BaseChatModel", schema: type[TSchema]):
    """Envelopa o modelo para devolver `schema` validado.

    - `method="json_schema"`: decodificacao restrita de verdade; o default do
      LangChain varia por provedor (`function_calling` em alguns) e com
      function calling o modelo pode omitir campos.
    - `include_raw=True`: sem isso a `AIMessage` se perde -- e com ela o
      `usage_metadata`, fonte de TODA a telemetria de custo.

    No provedor local, o suporte a `json_schema` depende do servidor (Ollama
    recente aceita `response_format` com schema). Falha de validacao aqui vira
    `StructuredOutputError` com retentativa -- e e' exatamente a medida "o
    aluno sabe produzir o formato" que a avaliacao registra."""

    return chat_model.with_structured_output(schema, method="json_schema", include_raw=True)


def system_message(text: str, model_id: str, *, cacheable: bool = False) -> dict[str, Any]:
    """Mensagem de sistema com prompt caching quando o provedor o exige
    explicitamente (Anthropic: `cache_control`). OpenAI cacheia por prefixo
    automaticamente, entao la' `cacheable` nao muda nada."""

    provider, _ = split_llm_id(model_id)
    if cacheable and provider == PROVIDER_ANTHROPIC:
        return {
            "role": "system",
            "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}],
        }
    return {"role": "system", "content": text}


def user_message(text: str, model_id: str, *, cacheable_prefix: str | None = None) -> dict[str, Any]:
    """Mensagem de usuario com um PREFIXO estavel cacheavel (ex.: o texto do
    artigo da LGPD reenviado em varias chamadas)."""

    if not cacheable_prefix:
        return {"role": "user", "content": text}
    provider, _ = split_llm_id(model_id)
    if provider == PROVIDER_ANTHROPIC:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": cacheable_prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": text},
            ],
        }
    return {"role": "user", "content": f"{cacheable_prefix}{text}"}


# Tentativas quando a saida NAO valida. Duas amostragens seguidas invalidas
# indicam problema de prompt ou schema -- insistir so' aumenta a conta.
_STRUCTURED_ATTEMPTS = 2


def _truncado_pelo_teto(raw: Any) -> str | None:
    """Descricao do truncamento quando a resposta parou pelo teto de saida
    (Responses API, Chat Completions ou Anthropic); None caso contrario."""

    meta = getattr(raw, "response_metadata", None) or {}
    incompleto = meta.get("incomplete_details") or {}
    motivo = incompleto.get("reason") if isinstance(incompleto, dict) else None
    if (
        motivo != "max_output_tokens"
        and meta.get("finish_reason") != "length"
        and meta.get("stop_reason") != "max_tokens"
    ):
        return None
    uso = getattr(raw, "usage_metadata", None) or {}
    saida = uso.get("output_tokens")
    raciocinio = (uso.get("output_token_details") or {}).get("reasoning")
    detalhe = f"{saida} tokens de saida" if saida is not None else "teto de saida"
    if raciocinio:
        detalhe += f", {raciocinio} deles de raciocinio"
    return detalhe


async def ainvoke_structured(
    chat_model: "BaseChatModel",
    schema: type[TSchema],
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    etapa: Any = None,
) -> TSchema:
    """Invoca pedindo saida estruturada, contabiliza o consumo e devolve o
    objeto validado.

    Concentra os passos que, soltos, se esquecem um de cada vez: fixar a
    decodificacao (`structured_model`), contabilizar o `usage_metadata` ANTES
    de a validacao falhar (a chamada ja foi paga), e transformar
    `parsing_error` numa excecao explicita. Retenta SO' `StructuredOutputError`
    -- erro de rede, credencial ou rate limit sobe na hora."""

    from app.core.llm_gate import portao_do_llm
    from app.core.llm_usage import record_langchain_usage

    modelo = structured_model(chat_model, schema)
    error: Any = None
    for tentativa in range(1, _STRUCTURED_ATTEMPTS + 1):
        async with portao_do_llm(model_id, messages=messages):
            result = await modelo.ainvoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else None
        if raw is not None:
            record_langchain_usage(model_id, raw, etapa=etapa)

        error = result.get("parsing_error") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if error is None and parsed is not None:
            return parsed

        truncado = _truncado_pelo_teto(raw)
        if truncado:
            raise StructuredOutputError(
                f"Saida de {model_id} nao validou contra {schema.__name__}: "
                f"resposta truncada pelo teto de saida ({truncado}). Erro: {error}"
            )
        if tentativa < _STRUCTURED_ATTEMPTS:
            logger.warning(
                "Saida de %s nao validou contra %s na tentativa %d/%d; reamostrando. Erro: %s",
                model_id, schema.__name__, tentativa, _STRUCTURED_ATTEMPTS, error,
            )
    raise StructuredOutputError(
        f"Saida de {model_id} nao validou contra {schema.__name__} em "
        f"{_STRUCTURED_ATTEMPTS} tentativas: {error}"
    )


class RespostaDeTexto(BaseModel):
    """Texto livre + o consumo DESTA chamada. A avaliacao mede tokens por
    item; o agregado da telemetria nao consegue atribuir por item quando
    varias chamadas correm em paralelo."""

    texto: str
    input_tokens: int = 0
    output_tokens: int = 0


def _texto_do_conteudo(content: Any) -> str:
    """Normaliza o `content` (string ou lista de blocos), concatenando so' os
    blocos de texto -- blocos de raciocinio nao viram resposta."""

    if isinstance(content, str):
        return content
    return "".join(
        bloco.get("text", "")
        for bloco in content
        if isinstance(bloco, dict) and bloco.get("type", "text") == "text"
    )


async def ainvoke_text_detalhado(
    chat_model: "BaseChatModel",
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    etapa: Any = None,
) -> RespostaDeTexto:
    """Invoca esperando texto livre, contabiliza o consumo e devolve texto e
    tokens da chamada."""

    from app.core.llm_gate import portao_do_llm
    from app.core.llm_usage import record_langchain_usage

    async with portao_do_llm(model_id, messages=messages):
        message = await chat_model.ainvoke(messages)
    record_langchain_usage(model_id, message, etapa=etapa)
    uso = getattr(message, "usage_metadata", None) or {}
    return RespostaDeTexto(
        texto=_texto_do_conteudo(message.content),
        input_tokens=int(uso.get("input_tokens") or 0),
        output_tokens=int(uso.get("output_tokens") or 0),
    )


async def ainvoke_text(
    chat_model: "BaseChatModel",
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    etapa: Any = None,
) -> str:
    """Invoca esperando texto livre e contabiliza o consumo."""

    resposta = await ainvoke_text_detalhado(chat_model, messages, model_id=model_id, etapa=etapa)
    return resposta.texto


class StructuredLlmClient:
    """Cliente de saida estruturada para os servicos em lote (sintese, filtro,
    avaliacao). Os tres compartilham a mesma estrutura -- cliente lazy por
    teto de tokens, system cacheavel, saida validada, consumo contabilizado --
    e repetida por servico cada copia e' uma chance de esquecer um passo."""

    # Amostragem DESLIGADA por default: extracao estruturada sobre evidencia
    # nao ganha nada amostrando e perde comparabilidade entre rodadas. A
    # self-consistency do filtro pede temperatura > 0 explicitamente.
    DEFAULT_TEMPERATURE = 0.0
    _MAX_MODELOS = 8

    def __init__(
        self,
        tier: ModelTier,
        *,
        max_tokens: int,
        etapa: Any = None,
        temperature: float | None = DEFAULT_TEMPERATURE,
        override: str | None = None,
        cache_de_prompt: bool = True,
    ) -> None:
        self.tier = tier
        self.override = override
        self.max_tokens = max_tokens
        self.etapa = etapa
        self.temperature = temperature
        self.cache_de_prompt = cache_de_prompt
        self._models: OrderedDict[tuple[str, int], "BaseChatModel"] = OrderedDict()

    @property
    def model_id(self) -> str:
        """Resolvido a cada leitura: mudar o `.env` e reiniciar basta."""

        return resolve_llm_id(self.tier, self.override)

    def _model(self, model_id: str, max_tokens: int) -> "BaseChatModel":
        chave = (model_id, max_tokens)
        modelo = self._models.get(chave)
        if modelo is not None:
            self._models.move_to_end(chave)
            return modelo
        modelo = build_chat_model(
            model_id,
            max_tokens=max_tokens,
            temperature=self.temperature,
            sem_cache_de_prompt=not self.cache_de_prompt,
        )
        self._models[chave] = modelo
        if len(self._models) > self._MAX_MODELOS:
            self._models.popitem(last=False)
        return modelo

    async def parse(
        self,
        schema: type[TSchema],
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        user_cacheable_prefix: str | None = None,
    ) -> TSchema:
        """Uma chamada com saida validada contra `schema`."""

        limit = max_tokens or self.max_tokens
        model_id = self.model_id
        return await ainvoke_structured(
            self._model(model_id, limit),
            schema,
            [
                system_message(system, model_id, cacheable=self.cache_de_prompt),
                user_message(user, model_id, cacheable_prefix=user_cacheable_prefix),
            ],
            model_id=model_id,
            etapa=self.etapa,
        )

    async def text_detalhado(self, *, system: str, user: str, max_tokens: int | None = None) -> RespostaDeTexto:
        """Uma chamada de texto livre com os tokens da chamada -- e' como o
        ALUNO e' medido (a resposta juridica em prosa), ja que exigir JSON
        dele mediria formato, nao conteudo."""

        limit = max_tokens or self.max_tokens
        model_id = self.model_id
        return await ainvoke_text_detalhado(
            self._model(model_id, limit),
            [system_message(system, model_id, cacheable=self.cache_de_prompt), user_message(user, model_id)],
            model_id=model_id,
            etapa=self.etapa,
        )

    async def text(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        return (await self.text_detalhado(system=system, user=user, max_tokens=max_tokens)).texto
