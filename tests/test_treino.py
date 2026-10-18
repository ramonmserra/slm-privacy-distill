"""Treino: preparacao dos dados (sem torch), dry-run do script, registro e
verificacao de runs na API."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import treinos as treinos_endpoint
from app.main import app
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.treino_repository import TreinoRepository
from app.schemas.dataset import ContagemPorTipo, ManifestoDoDataset, ParametrosDeDataset, Split
from app.schemas.treino import RegistroDeRun
from app.services.treino import registro as servico
from treino import sft_lora
from treino.preparacao import (
    PERFIS,
    PREFIXO_SEM_PENSAR_QWEN3,
    RegistroDoRun,
    carregar_jsonl,
    modelo_local_id_sugerido,
    montar_mensagens,
    preparar,
)

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)


def linha(id_: str, resposta="Resposta.") -> dict:
    return {"id": id_, "tipo": "conceito", "dificuldade": "basica", "artigos_citados": ["LGPD:art.7"],
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": f"P {id_}?"}, {"role": "assistant", "content": resposta}]}


def _jsonl(caminho: Path, linhas: list[dict]) -> Path:
    caminho.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas), encoding="utf-8")
    return caminho


class TestPreparacao:
    def test_perfis_coerentes(self):
        assert set(PERFIS) == {"colab-t4", "colab-t4-4bit", "cpu-local"}
        assert PERFIS["colab-t4"].precisao == "fp16"  # T4 nao tem bf16
        assert PERFIS["cpu-local"].precisao == "fp32" and not PERFIS["cpu-local"].quatro_bits
        assert PERFIS["cpu-local"].comprimento_maximo < PERFIS["colab-t4"].comprimento_maximo

    def test_prefixo_do_assistente_e_idempotente(self):
        m = montar_mensagens(linha("a", "Resposta."), PREFIXO_SEM_PENSAR_QWEN3)
        assert m[2]["content"] == PREFIXO_SEM_PENSAR_QWEN3 + "Resposta."
        assert montar_mensagens({"messages": m}, PREFIXO_SEM_PENSAR_QWEN3)[2]["content"] == m[2]["content"]
        assert montar_mensagens(linha("a"), "")[2]["content"] == "Resposta."
        assert [x["role"] for x in m] == ["system", "user", "assistant"]

    def test_linha_invalida(self):
        with pytest.raises(ValueError, match="system, user, assistant"):
            montar_mensagens({"id": "x", "messages": [{"role": "user", "content": "p"}]}, "")
        with pytest.raises(ValueError, match="vazia"):
            montar_mensagens(linha("x", resposta="   "), "")

    def test_carregar_jsonl_rejeita_linha_quebrada(self, tmp_path):
        caminho = tmp_path / "t.jsonl"
        caminho.write_text('{"a": 1}\n{nao json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="linha 2"):
            carregar_jsonl(caminho)

    def test_preparar_preserva_id_e_tipo(self):
        preparado = preparar([linha("a")], "")
        assert preparado[0]["id"] == "a" and preparado[0]["tipo"] == "conceito"

    def test_modelo_local_id_sugerido(self):
        assert modelo_local_id_sugerido("Qwen/Qwen3-1.7B", "ab12cd34ef56") == "local:qwen3-1.7b-lgpd-ab12cd34"

    def test_run_json_tem_os_campos_do_registro_da_api(self, tmp_path):
        run = RegistroDoRun(dataset_versao="v", modelo_base="Qwen/Qwen3-0.6B", perfil="cpu-local", modelo_local_id="local:x", status="concluido")
        run.gravar(tmp_path / "run.json")
        bruto = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert set(RegistroDeRun.model_fields) <= set(bruto)
        RegistroDeRun.model_validate(bruto)  # o arquivo e' o corpo de POST /treinos


class TestScriptDryRun:
    def test_dry_run_escreve_plano(self, tmp_path, capsys):
        treino = _jsonl(tmp_path / "treino.jsonl", [linha("a"), linha("b"), linha("c")])
        held = _jsonl(tmp_path / "held.jsonl", [linha("z")])
        codigo = sft_lora.main([
            "--perfil", "cpu-local", "--dataset", str(treino), "--held-out", str(held),
            "--dataset-versao", "ab12cd34ef56", "--saida", str(tmp_path / "saida"), "--dry-run", "--max-exemplos", "2",
        ])
        assert codigo == 0
        run = json.loads((tmp_path / "saida" / "run.json").read_text(encoding="utf-8"))
        assert run["status"] == "dry_run" and run["exemplos_de_treino"] == 2 and run["exemplos_de_held_out"] == 1
        assert run["modelo_base"] == "Qwen/Qwen3-0.6B" and run["modelo_local_id"] == "local:qwen3-0.6b-lgpd-ab12cd34"
        assert run["hiperparametros"]["lora_r"] == 8 and run["prefixo_do_assistente"] == PREFIXO_SEM_PENSAR_QWEN3
        assert "perfil=cpu-local" in capsys.readouterr().out

    def test_held_out_vazado_aborta(self, tmp_path):
        treino = _jsonl(tmp_path / "treino.jsonl", [linha("a")])
        held = _jsonl(tmp_path / "held.jsonl", [linha("a")])
        with pytest.raises(SystemExit, match="held-out"):
            sft_lora.main(["--perfil", "colab-t4", "--dataset", str(treino), "--held-out", str(held), "--dry-run"])

    def test_sobrescritas_de_perfil(self, tmp_path):
        treino = _jsonl(tmp_path / "treino.jsonl", [linha("a")])
        sft_lora.main(["--perfil", "colab-t4", "--dataset", str(treino), "--modelo-base", "Qwen/Qwen3-4B", "--epocas", "1",
                       "--saida", str(tmp_path / "s"), "--dry-run"])
        run = json.loads((tmp_path / "s" / "run.json").read_text(encoding="utf-8"))
        assert run["modelo_base"] == "Qwen/Qwen3-4B" and run["hiperparametros"]["epocas"] == 1.0


class TestServicoDeRegistro:
    def test_esta_servido_aceita_tag(self):
        assert servico.esta_servido("local:qwen3-1.7b-lgpd-ab12", ["qwen3-1.7b-lgpd-ab12:latest", "qwen3:1.7b"])
        assert servico.esta_servido("local:qwen3-1.7b-lgpd-ab12", ["qwen3-1.7b-lgpd-ab12"])
        assert not servico.esta_servido("local:qwen3-1.7b-lgpd-ab12", ["qwen3:1.7b"])


def _registro(**kw) -> dict:
    base = dict(dataset_versao="v1", modelo_base="Qwen/Qwen3-1.7B", perfil="colab-t4", modelo_local_id="local:qwen3-1.7b-lgpd-v1",
                status="concluido", duracao_s=3600.5, metricas={"train_loss": 0.9, "eval_loss": 1.1}, hardware={"gpu": "Tesla T4"})
    base.update(kw)
    return base


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        datasets = DatasetRepository(tmp_path)
        manifesto = ManifestoDoDataset(
            versao="v1", nome="v1", criado_em=AGORA, parametros=ParametrosDeDataset(nome="v1", jobs_de_filtro=["f"]),
            treino=ContagemPorTipo(total=1), held_out=ContagemPorTipo(), versao_do_prompt_do_aluno="p",
        )
        datasets.salvar(manifesto, {Split.TREINO: [], Split.HELD_OUT: []})
        monkeypatch.setattr(treinos_endpoint, "_datasets", datasets)
        monkeypatch.setattr(treinos_endpoint, "_repositorio", TreinoRepository(tmp_path))
        with TestClient(app) as c:
            yield c

    def test_registra_lista_e_obtem(self, client):
        r = client.post("/api/v1/treinos", json=_registro())
        assert r.status_code == 201, r.text
        run = r.json()
        assert run["disponivel_localmente"] is None and run["metricas"]["eval_loss"] == 1.1
        assert client.get("/api/v1/treinos").json()[0]["id"] == run["id"]
        assert client.get(f"/api/v1/treinos/{run['id']}").json()["modelo_local_id"] == "local:qwen3-1.7b-lgpd-v1"

    def test_dataset_desconhecido_e_id_repetido(self, client):
        assert client.post("/api/v1/treinos", json=_registro(dataset_versao="nao")).status_code == 404
        assert client.post("/api/v1/treinos", json=_registro()).status_code == 201
        assert client.post("/api/v1/treinos", json=_registro()).status_code == 409
        # Um dry_run ou falha com o mesmo id nao conflita: so' run concluido ocupa o nome.
        assert client.post("/api/v1/treinos", json=_registro(status="falhou")).status_code == 201

    def test_validacao_do_id_local_e_status(self, client):
        assert client.post("/api/v1/treinos", json=_registro(modelo_local_id="qwen3:1.7b")).status_code == 422
        assert client.post("/api/v1/treinos", json=_registro(status="rodando")).status_code == 422

    def test_verificar_consulta_o_servidor_local(self, client, monkeypatch):
        run_id = client.post("/api/v1/treinos", json=_registro()).json()["id"]

        async def fake_modelos(base_url=None):
            return ["qwen3-1.7b-lgpd-v1:latest", "qwen3:1.7b"]

        monkeypatch.setattr(servico, "modelos_servidos", fake_modelos)
        r = client.post(f"/api/v1/treinos/{run_id}/verificar")
        assert r.status_code == 200 and r.json()["disponivel"] is True
        assert client.get(f"/api/v1/treinos/{run_id}").json()["disponivel_localmente"] is True

    def test_verificar_sem_servidor_responde_503(self, client, monkeypatch):
        import httpx

        run_id = client.post("/api/v1/treinos", json=_registro()).json()["id"]

        async def caiu(base_url=None):
            raise httpx.ConnectError("recusado")

        monkeypatch.setattr(servico, "modelos_servidos", caiu)
        assert client.post(f"/api/v1/treinos/{run_id}/verificar").status_code == 503

    def test_run_inexistente(self, client):
        assert client.get("/api/v1/treinos/nao").status_code == 404
