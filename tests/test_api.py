"""Autenticacao, health e endpoints de consumo/diagnostico pelo TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import usage as usage_endpoint
from app.core import llm
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestAutenticacao:
    def test_disabled_em_test_libera(self, client):
        assert client.get("/api/v1/usage/llm").status_code == 200

    def test_disabled_fora_de_test_recusa_com_503(self, client, monkeypatch):
        monkeypatch.setenv("APP_ENV", "development")
        r = client.get("/api/v1/usage/llm")
        assert r.status_code == 503

    def test_auth_mode_ausente_recusa_com_503(self, client, monkeypatch):
        monkeypatch.delenv("AUTH_MODE")
        assert client.get("/api/v1/usage/llm").status_code == 503

    def test_api_key(self, client, monkeypatch):
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("AUTH_MODE", "api_key")
        monkeypatch.setenv("AUTH_API_KEYS", "segredo:ramon, outra")
        assert client.get("/api/v1/usage/llm").status_code == 401
        assert client.get("/api/v1/usage/llm", headers={"X-API-Key": "errada"}).status_code == 401
        assert client.get("/api/v1/usage/llm", headers={"X-API-Key": "segredo"}).status_code == 200
        assert client.get("/api/v1/usage/llm", headers={"X-API-Key": "outra"}).status_code == 200

    def test_openapi_declara_api_key_para_o_swagger(self, client):
        esquema = client.get("/openapi.json").json()
        seguranca = esquema["components"]["securitySchemes"]
        assert {"type": "apiKey", "in": "header", "name": "X-API-Key"}.items() <= next(iter(seguranca.values())).items()
        assert esquema["paths"]["/api/v1/usage/llm"]["get"]["security"]
        assert "security" not in esquema["paths"]["/health/live"]["get"]

    def test_health_nao_exige_credencial(self, client, monkeypatch):
        monkeypatch.setenv("AUTH_MODE", "api_key")
        assert client.get("/health/live").status_code == 200


class TestHealth:
    def test_live(self, client):
        assert client.get("/health/live").json() == {"status": "ok"}
        assert "X-Request-Id" in client.get("/health/live").headers

    def test_ready_degradado_sem_credencial_do_professor(self, client):
        r = client.get("/health/ready")
        assert r.status_code == 503
        corpo = r.json()
        assert corpo["status"] == "degraded"
        assert corpo["checks"]["credencial_professor"] == "falha"
        assert corpo["checks"]["servidor_local_do_aluno"] in ("ok", "indisponivel", "falha")

    def test_ready_ok_com_credencial(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["checks"]["data_dir"] == "ok"


class TestUsage:
    def test_providers_descreve_os_tres_papeis(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL_TEACHER", "gpt-5-mini")
        corpo = client.get("/api/v1/usage/providers").json()
        papeis = {p["papel"]: p for p in corpo["papeis"]}
        assert papeis["teacher"]["modelo"] == "openai:gpt-5-mini"
        assert papeis["teacher"]["origem_do_modelo"] == "env"
        assert papeis["teacher"]["variavel_de_ambiente"] == "OPENAI_MODEL_TEACHER"
        assert papeis["student"]["provedor"] == "local"
        assert papeis["student"]["credencial_presente"] is True
        assert any("OPENAI_API_KEY" in a for a in corpo["avisos"])

    def test_llm_vazio(self, client):
        corpo = client.get("/api/v1/usage/llm").json()
        assert corpo["total"]["chamadas"] == 0

    def test_ping_papel_desconhecido(self, client):
        assert client.post("/api/v1/usage/ping/professor").status_code == 404

    def test_ping_sem_credencial(self, client):
        assert client.post("/api/v1/usage/ping/teacher").status_code == 503

    def test_ping_com_duble(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        async def fake_parse(self, schema, **_kwargs):
            return schema(saudacao="Ola", numero=42)

        monkeypatch.setattr(llm.StructuredLlmClient, "parse", fake_parse)
        r = client.post("/api/v1/usage/ping/teacher")
        assert r.status_code == 200
        corpo = r.json()
        assert corpo["modelo"] == "openai:gpt-5.6-luna"
        assert corpo["resposta"] == {"saudacao": "Ola", "numero": 42}
        assert usage_endpoint.RespostaDePing(**corpo["resposta"]).numero == 42
