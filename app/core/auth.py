"""Autenticacao da API -- OBRIGATORIA, com opt-out explicito.

Nao ha default permissivo: `AUTH_MODE` ausente faz o servidor recusar toda
requisicao (503) e registrar um erro no boot dizendo o que configurar. Um
default que funciona sem autenticacao vira producao; ninguem "esquece de
ligar" o que nao sobe sem estar configurado.

Modos (`AUTH_MODE`):

- `api_key`: header `X-API-Key` conferido contra `AUTH_API_KEYS`
  (`chave[:nome]`, itens separados por virgula). `nome` identifica quem
  chamou nos logs; a chave em si nunca e' logada.
- `disabled`: EXPLICITO, aceito so' com `APP_ENV=test` -- a suite de testes
  exercita regra de negocio, nao o gate de credencial.

Single-tenant por desenho (projeto de pesquisa de um grupo): nao ha tenant,
papel nem RBAC. Se um dia houver, o lugar e' aqui, nao nos endpoints.
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import APP_ENV_TEST, app_env

logger = logging.getLogger(__name__)

MODE_DISABLED = "disabled"
MODE_API_KEY = "api_key"
MODE_UNSET = "unset"
MODE_INVALID = "invalid"
_MODOS = (MODE_DISABLED, MODE_API_KEY)

HEADER_API_KEY = "X-API-Key"

# Declarar o header como esquema de seguranca faz o Swagger (/docs) mostrar o
# botao Authorize. `auto_error=False`: a decisao 401/503 continua aqui embaixo.
_esquema_api_key = APIKeyHeader(name=HEADER_API_KEY, auto_error=False)


@dataclass(frozen=True)
class Principal:
    """Quem esta chamando, ja verificado. `subject` vai para os logs."""

    subject: str


def auth_mode() -> str:
    raw = os.getenv("AUTH_MODE", "").strip().lower()
    if not raw:
        return MODE_UNSET
    if raw not in _MODOS:
        return MODE_INVALID
    return raw


def disabled_proibido() -> bool:
    """`disabled` fora de `APP_ENV=test` e' erro de configuracao, nao escolha."""

    return auth_mode() == MODE_DISABLED and app_env() != APP_ENV_TEST


def _chaves_configuradas() -> dict[str, str]:
    """`AUTH_API_KEYS` -> {chave: nome}. Item sem nome recebe `api-key`."""

    chaves: dict[str, str] = {}
    for item in os.getenv("AUTH_API_KEYS", "").split(","):
        item = item.strip()
        if not item:
            continue
        chave, _, nome = item.partition(":")
        if chave:
            chaves[chave] = nome.strip() or "api-key"
    return chaves


def verificar_configuracao() -> None:
    """Chamado no boot: o operador descobre a lacuna ao subir o processo, nao
    no primeiro request. Nao levanta -- o servidor sobe e responde 503 a tudo,
    com a mensagem no log e no /health/ready."""

    modo = auth_mode()
    if modo == MODE_UNSET:
        logger.error(
            "AUTH_MODE nao configurado: a API vai recusar todas as requisicoes com 503. "
            "Defina AUTH_MODE=api_key (com AUTH_API_KEYS) ou, so' em APP_ENV=test, disabled."
        )
    elif modo == MODE_INVALID:
        logger.error("AUTH_MODE=%r invalido; aceitos: %s.", os.getenv("AUTH_MODE"), ", ".join(_MODOS))
    elif disabled_proibido():
        logger.error(
            "AUTH_MODE=disabled so' e' aceito com APP_ENV=test (APP_ENV atual: %s). "
            "A API vai recusar todas as requisicoes.", app_env(),
        )
    elif modo == MODE_API_KEY and not _chaves_configuradas():
        logger.error("AUTH_MODE=api_key sem nenhuma chave em AUTH_API_KEYS: nenhuma credencial sera aceita.")


def autenticar(apresentada: str | None = Security(_esquema_api_key)) -> Principal:
    """Dependencia do FastAPI: 503 se o servidor esta mal configurado, 401 se
    a credencial falta ou e' invalida."""

    modo = auth_mode()
    if modo in (MODE_UNSET, MODE_INVALID) or disabled_proibido():
        raise HTTPException(
            status_code=503,
            detail="Autenticacao nao configurada no servidor (AUTH_MODE). Veja o log do processo.",
        )
    if modo == MODE_DISABLED:
        return Principal(subject="anonimo-teste")

    apresentada = (apresentada or "").strip()
    if not apresentada:
        raise HTTPException(status_code=401, detail=f"Header {HEADER_API_KEY} ausente.")
    # Comparacao em tempo constante contra TODAS as chaves: um `in dict` vazaria
    # por tempo se a chave existe; aqui o custo nao depende de qual bateu.
    nome_encontrado: str | None = None
    for chave, nome in _chaves_configuradas().items():
        if hmac.compare_digest(chave.encode(), apresentada.encode()):
            nome_encontrado = nome
    if nome_encontrado is None:
        raise HTTPException(status_code=401, detail="Credencial invalida.")
    return Principal(subject=nome_encontrado)
