"""O front estatico e' servido em /ui sem credencial (a chave vai em cada
chamada que a pagina faz a API); a raiz redireciona; a API continua
protegida; o endpoint de anotacoes por anotador existe."""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import avaliacao as avaliacao_endpoint
from app.main import app
from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.schemas.avaliacao import (
    Conjunto,
    ItemDeAvaliacao,
    ParametrosDeAvaliacao,
    RespostaDoModelo,
    ResultadoDoItem,
    RunDeAvaliacao,
    VereditoDeAvaliacao,
)
from app.schemas.sintese import StatusDoJob

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = AvaliacaoRepository(tmp_path)
    repo.salvar_run(RunDeAvaliacao(id="r", status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeAvaliacao(modelo="student"), modelo="local:x",
                                   versao_da_medida="v", criado_em=AGORA, atualizado_em=AGORA))
    v = VereditoDeAvaliacao(indice=1, correto=True, criterios_atendidos=0, criterios_total=0, grounded=True, abstencao=False, similaridade=0.9)
    repo.acrescentar_itens("r", [ResultadoDoItem(item=ItemDeAvaliacao(id="i1", conjunto=Conjunto.BANCO_INICIAL, pergunta="P", referencia="R", tipo="pergunta"),
                                                 resposta=RespostaDoModelo(item_id="i1", texto="x", latencia_ms=1), veredito=v, acerto=True)])
    monkeypatch.setattr(avaliacao_endpoint, "_repositorio", repo)
    with TestClient(app) as c:
        yield c


def test_ui_e_publica_e_a_api_continua_protegida(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_API_KEYS", "k")
    pagina = client.get("/ui/")
    assert pagina.status_code == 200 and "text/html" in pagina.headers["content-type"]
    assert "Anotação" in pagina.text and 'src="app.js"' in pagina.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/app.css").status_code == 200
    assert client.get("/api/v1/avaliacao/runs").status_code == 401
    assert client.get("/api/v1/avaliacao/runs", headers={"X-API-Key": "k"}).status_code == 200


def test_raiz_redireciona_para_a_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/ui/"


def test_anotacoes_por_anotador(client):
    client.post("/api/v1/avaliacao/runs/r/anotacoes", json={"item_id": "i1", "anotador": "ana", "correto": True})
    client.post("/api/v1/avaliacao/runs/r/anotacoes", json={"item_id": "i1", "anotador": "bia", "correto": False})
    assert len(client.get("/api/v1/avaliacao/runs/r/anotacoes").json()) == 2
    so_ana = client.get("/api/v1/avaliacao/runs/r/anotacoes", params={"anotador": "ana"}).json()
    assert [a["anotador"] for a in so_ana] == ["ana"]
    assert client.get("/api/v1/avaliacao/runs/nao/anotacoes").status_code == 404


def test_js_compila():
    """`node --check` valida a sintaxe do front. Pula onde nao ha Node (o CI
    do GitHub tem)."""

    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node indisponivel")
    resultado = subprocess.run([node, "--check", "app/ui/app.js"], capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr
