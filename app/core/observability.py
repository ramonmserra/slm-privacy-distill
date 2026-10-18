"""Observabilidade: request_id por requisicao e log estruturado opcional.

- `RequestContextMiddleware`: gera um `request_id` (ou herda `X-Request-Id`),
  devolve-o no header e registra a linha de acesso com duracao e status.
- `JsonLogFormatter` (`LOG_FORMAT=json`): cada log do processo sai com o
  `request_id` corrente -- um `logger.info` dentro de um job ja sai etiquetado.

ASGI puro (nao `BaseHTTPMiddleware`): a classe base cria task group e canal
por requisicao; aqui so' se le o `scope` e se envolve o `send`.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import log_format

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id_ctx", default=None)

logger = logging.getLogger("app.access")

HEADER_REQUEST_ID = "X-Request-Id"


def get_request_id() -> str | None:
    return _request_id_ctx.get()


def _sem_quebra_de_linha(valor: str) -> str:
    # Header de cliente vai para o log: uma quebra de linha embutida forjaria
    # uma linha de log.
    return valor.replace("\r", "").replace("\n", "")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        extras = getattr(record, "extras", None)
        if isinstance(extras, dict):
            payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configurar_logging() -> None:
    """Aplica o formato escolhido aos handlers do root logger. Chamado uma
    vez no boot, DEPOIS do basicConfig -- troca so' o formatter."""

    if log_format() != "json":
        return
    for handler in logging.getLogger().handlers:
        handler.setFormatter(JsonLogFormatter())


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = _sem_quebra_de_linha(
            request.headers.get(HEADER_REQUEST_ID) or uuid.uuid4().hex
        )[:64]
        token = _request_id_ctx.set(request_id)
        inicio = time.perf_counter()
        status = {"code": 500}

        async def send_com_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers[HEADER_REQUEST_ID] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_com_request_id)
        except Exception:
            duracao_ms = (time.perf_counter() - inicio) * 1000
            logger.exception(
                "%s %s -> 500 (%.1f ms)", request.method, request.url.path, duracao_ms,
            )
            raise
        finally:
            _request_id_ctx.reset(token)

        duracao_ms = (time.perf_counter() - inicio) * 1000
        if not request.url.path.startswith("/health"):
            logger.info(
                "%s %s -> %d (%.1f ms)",
                request.method, request.url.path, status["code"], duracao_ms,
                extra={"extras": {
                    "method": request.method, "path": request.url.path,
                    "status": status["code"], "duration_ms": round(duracao_ms, 1),
                    "request_id": request_id,
                }},
            )
