"""Ponto de entrada da API: monta o FastAPI, os middlewares e os routers da
v1. `main.py` na raiz so' reexporta `app` daqui."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core import llm
from app.core.auth import auth_mode, disabled_proibido, verificar_configuracao
from app.core.config import app_env, data_dir
from app.core.erros import registrar_handlers
from app.core.observability import RequestContextMiddleware, configurar_logging

# O uvicorn so' configura handlers para os loggers dele; sem isto todo
# `logger.info` do nosso codigo e' descartado pelo root logger (WARNING).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
configurar_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.services.jobs_interrompidos import marcar_jobs_interrompidos

    data_dir().mkdir(parents=True, exist_ok=True)
    for aviso in llm.avisos_de_configuracao():
        logger.warning("Configuracao de LLM: %s", aviso)
    # Jobs rodam como tasks deste processo: o que estava aberto quando o
    # processo anterior morreu nao vai terminar; o estado gravado nao pode
    # dizer o contrario para quem faz polling.
    marcar_jobs_interrompidos()
    yield


app = FastAPI(
    title="GENA -- Destilacao de SLM em Privacidade e Protecao de Dados",
    description=(
        "Plataforma da pesquisa: sintese de dados por LLM professor, filtragem, "
        "exportacao para treino e avaliacao do SLM aluno com LLM-as-judge."
    ),
    lifespan=lifespan,
)

# O operador descobre AUTH_MODE ausente ao subir o processo, nao no primeiro request.
verificar_configuracao()

# Ultimo add_middleware = camada mais externa: o request_id nasce antes de tudo.
app.add_middleware(RequestContextMiddleware)
registrar_handlers(app)
app.include_router(api_router, prefix="/api/v1")

# Front minimo (HTML + JS puro, sem build) para a parte HUMANA do fluxo:
# anotacao de calibracao, inspecao de exemplos/decisoes e painel de runs.
# Os arquivos sao publicos; cada chamada que a pagina faz a API leva a chave
# que a pessoa digitou, entao a protecao continua sendo a da API.
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "ui", html=True), name="ui")


@app.middleware("http")
async def _sem_cache_no_front(request, call_next):
    """O painel e' HTML+JS sem build nem hash no nome: sem este header o
    navegador guarda o `app.js` antigo e um botao novo "nao aparece"."""

    resposta = await call_next(request)
    if request.url.path.startswith("/ui"):
        resposta.headers["Cache-Control"] = "no-cache"
    return resposta


@app.get("/", include_in_schema=False)
async def raiz() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, str]:
    """Liveness: o processo responde. Nao checa dependencia."""

    return {"status": "ok"}


def _checar_data_dir() -> str:
    try:
        raiz = data_dir()
        raiz.mkdir(parents=True, exist_ok=True)
        sonda = raiz / ".health-probe"
        sonda.write_text("ok", encoding="utf-8")
        sonda.unlink()
        return "ok"
    except OSError as exc:
        logger.warning("health/ready: DATA_DIR falhou: %s", exc)
        return "falha"


async def _checar_servidor_local() -> str:
    """O endpoint OpenAI-compatible do aluno responde? Informativo: sem ele a
    API sobe (sintese e filtro nao precisam do aluno), mas a avaliacao nao."""

    url = llm.local_base_url().rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resposta = await client.get(url)
        return "ok" if resposta.status_code < 500 else "falha"
    except httpx.HTTPError:
        return "indisponivel"


@app.get("/health/ready", tags=["health"])
async def health_ready() -> JSONResponse:
    """Readiness: a instancia consegue ATENDER? 503 quando falta o que
    impede qualquer trabalho (autenticacao, DATA_DIR, credencial do
    professor); o corpo diz o que falta."""

    checks = {
        "auth_mode": auth_mode(),
        "app_env": app_env(),
        "data_dir": _checar_data_dir(),
        "credencial_professor": "ok" if llm.llm_available(llm.provider_for_tier(llm.ModelTier.TEACHER)) else "falha",
        "credencial_juiz": "ok" if llm.llm_available(llm.provider_for_tier(llm.ModelTier.JUDGE)) else "falha",
        "servidor_local_do_aluno": await _checar_servidor_local(),
        "avisos": llm.avisos_de_configuracao(),
    }
    auth_ok = checks["auth_mode"] in ("api_key", "disabled") and not disabled_proibido()
    pronto = auth_ok and checks["data_dir"] == "ok" and checks["credencial_professor"] == "ok"
    return JSONResponse(
        {"status": "ok" if pronto else "degraded", "checks": checks},
        status_code=200 if pronto else 503,
    )
