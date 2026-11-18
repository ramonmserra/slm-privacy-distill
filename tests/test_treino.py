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
    TEMPLATE_DE_CHAT_GENERICO,
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


class TestPassosDeAquecimento:
    def test_tres_por_cento_do_total_no_perfil_do_colab(self):
        # 2592 exemplos / lote 2 = 1296 lotes; / acumulacao 8 = 162 passos; x 2 epocas = 324; 3% -> 10
        assert sft_lora.passos_de_aquecimento(2592, PERFIS["colab-t4"]) == 10

    def test_poucos_exemplos_nao_quebra(self):
        assert sft_lora.passos_de_aquecimento(8, PERFIS["colab-t4"]) == 1
        assert sft_lora.passos_de_aquecimento(1, PERFIS["cpu-local"]) == 1

    def test_e_inteiro(self):
        assert isinstance(sft_lora.passos_de_aquecimento(40, PERFIS["cpu-local"]), int)


class TestTemplateDeChatGenerico:
    MENSAGENS = [
        {"role": "system", "content": "instrucao"},
        {"role": "user", "content": "pergunta?"},
        {"role": "assistant", "content": "resposta."},
    ]

    def _render(self, mensagens, add_generation_prompt):
        jinja2 = pytest.importorskip("jinja2")
        return jinja2.Template(TEMPLATE_DE_CHAT_GENERICO).render(
            messages=mensagens, add_generation_prompt=add_generation_prompt, bos_token="<s>", eos_token="</s>")

    def test_conversa_completa(self):
        assert self._render(self.MENSAGENS, False) == "<s>instrucao\n\n### pergunta:\npergunta?\n\n### resposta:\nresposta.</s>"

    def test_prompt_e_prefixo_da_conversa_completa(self):
        # O TRL mascara a perda pelo comprimento do prompt: ele precisa ser
        # prefixo exato da conversa completa, senao a perda cai no lugar errado.
        prompt = self._render(self.MENSAGENS[:2], True)
        assert prompt.endswith("### resposta:\n") and self._render(self.MENSAGENS, False).startswith(prompt)


class TestNormalizarTokenizer:
    def test_lista_vira_additional_special_tokens(self, tmp_path):
        from treino import normalizar_tokenizer

        caminho = tmp_path / "tokenizer_config.json"
        caminho.write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer", "extra_special_tokens": ["<|im_start|>", "<|im_end|>"]}), encoding="utf-8")
        assert normalizar_tokenizer.main([str(tmp_path)]) == 0
        config = json.loads(caminho.read_text(encoding="utf-8"))
        assert "extra_special_tokens" not in config
        assert config["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"] and config["tokenizer_class"] == "Qwen2Tokenizer"

    def test_dicionario_ou_ausente_fica_como_esta(self):
        from treino.normalizar_tokenizer import normalizar

        for config in ({"extra_special_tokens": {"a": "<a>"}}, {"tokenizer_class": "X"}):
            antes = json.dumps(config)
            assert normalizar(config) is False and json.dumps(config) == antes


class TestVocabularioSentencePiece:
    def test_copia_quando_o_base_tem_tokenizer_model(self, tmp_path):
        origem = tmp_path / "cache" / "tokenizer.model"
        origem.parent.mkdir()
        origem.write_bytes(b"spm")
        destino = tmp_path / "merged"
        destino.mkdir()
        assert sft_lora.copiar_vocabulario_sentencepiece("org/modelo", destino, baixar=lambda repo, nome: str(origem)) is True
        assert (destino / "tokenizer.model").read_bytes() == b"spm"

    def test_segue_sem_o_arquivo_quando_o_base_nao_tem(self, tmp_path, capsys):
        def nao_existe(repo, nome):
            raise FileNotFoundError(nome)

        assert sft_lora.copiar_vocabulario_sentencepiece("Qwen/Qwen3-1.7B", tmp_path, baixar=nao_existe) is False
        assert not (tmp_path / "tokenizer.model").exists() and "nao copiado" in capsys.readouterr().out


class TestResumoDoRun:
    RUN = {
        "status": "concluido", "modelo_base": "Qwen/Qwen3-1.7B", "dataset_versao": "ab12cd34ef56", "duracao_s": 3629.4,
        "exemplos_de_treino": 2592, "metricas": {"train_loss": 0.7485, "eval_loss": 0.6286, "eval_mean_token_accuracy": 0.8287},
        "hiperparametros": {"lora_r": 16}, "hardware": {"gpu": "Tesla T4"},
    }

    def test_traz_tempo_metricas_e_parametros(self):
        from treino.resumo import resumir

        texto = resumir(self.RUN, {"modelo-q4_k_m.gguf": 1107408736})
        for trecho in ("ab12cd34ef56", "3629 s (1.01 h)", "Perda no held-out (final): 0.6286", "Acuracia por token no held-out: 0.8287",
                       "lora_r: 16", "gpu: Tesla T4", "modelo-q4_k_m.gguf: 1056.1 MB"):
            assert trecho in texto

    def test_grava_arquivo_ao_lado_do_run(self, tmp_path):
        from treino import resumo

        (tmp_path / "run.json").write_text(json.dumps(self.RUN), encoding="utf-8")
        assert resumo.main([str(tmp_path / "run.json"), str(tmp_path / "resumo.txt")]) == 0
        texto = (tmp_path / "resumo.txt").read_text(encoding="utf-8")
        assert "run.json" in texto and "resumo.txt" not in texto

    def test_run_sem_metricas_nao_quebra(self):
        from treino.resumo import resumir

        assert "Duracao total:       0 s" in resumir({"status": "falhou"})


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
