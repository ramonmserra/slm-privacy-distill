"""On-policy: selecao (funcao pura), engine com dublês de aluno e juiz,
endpoints e a preparacao dos pares DPO para o treino."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import on_policy as on_policy_endpoint
from app.core.erros import OrcamentoExcedido
from app.core.llm import RespostaDeTexto
from app.main import app
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.on_policy_repository import OnPolicyRepository
from app.schemas.avaliacao import CategoriaDeErro, Conjunto, VereditoDeAvaliacao
from app.schemas.dataset import ContagemPorTipo, ManifestoDoDataset, ParametrosDeDataset, Split
from app.schemas.on_policy import AmostraJulgada, OrigemDoSft, ParametrosOnPolicy
from app.schemas.sintese import StatusDoJob
from app.services.corpus import ingestao
from app.services.corpus.chunking import chunkar
from app.services.datasets.prompt_do_aluno import PROMPT_DO_ALUNO
from app.services.on_policy.engine import EngineOnPolicy, PromptDeTreino, aprovada, novo_job, selecionar
from treino import dpo_lora
from treino.preparacao import PREFIXO_SEM_PENSAR_QWEN3

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)
_LEI = ["Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses:", "IX - interesses legítimos;",
        "Art. 18. O titular tem direitos.", "Art. 48. O controlador deverá comunicar incidente."]


@pytest.fixture
def corpus():
    return ingestao.Corpus("LGPD", chunkar(_LEI), Path("fake.txt"))


def prompt(id_: str = "p1") -> PromptDeTreino:
    return PromptDeTreino(id=id_, conjunto=Conjunto.HELD_OUT, pergunta=f"Pergunta {id_}?", referencia="Ref.", resposta_do_professor="Ref.",
                          artigos_referencia=["LGPD:art.7"], tipo="conceito")


def amostra(indice: int, texto: str, correto=True, grounded=True, similaridade=0.9, inexistentes=(), julgada=True) -> AmostraJulgada:
    v = VereditoDeAvaliacao(indice=indice, correto=correto, criterios_atendidos=0, criterios_total=0, grounded=grounded, abstencao=False,
                            categoria=None if correto else CategoriaDeErro.WRONG_LEGAL_BASIS, similaridade=similaridade) if julgada else None
    a = AmostraJulgada(prompt_id="p1", indice=indice, texto=texto, veredito=v, artigos_inexistentes=list(inexistentes))
    a.aprovada = aprovada(a, exigir_lastro=True)
    return a


class TestSelecao:
    def test_aprovada(self):
        assert amostra(1, "a").aprovada
        assert not amostra(1, "a", correto=False).aprovada
        assert not amostra(1, "a", grounded=False).aprovada
        assert not amostra(1, "a", inexistentes=["LGPD:art.72"]).aprovada
        assert not amostra(1, "a", julgada=False).aprovada
        sem_lastro = AmostraJulgada(prompt_id="p", indice=1, texto="x", veredito=amostra(1, "x", grounded=False).veredito)
        assert aprovada(sem_lastro, exigir_lastro=False) is True

    def test_sft_escolhe_a_melhor_aprovada_e_dpo_pareia_com_a_pior(self):
        amostras = [amostra(1, "boa", similaridade=0.7), amostra(2, "otima", similaridade=0.95), amostra(3, "ruim", correto=False, similaridade=0.3),
                    amostra(4, "pessima", correto=False, similaridade=0.1)]
        linha, par = selecionar(prompt(), amostras, ParametrosOnPolicy(dataset_versao="v", aluno="local:x"))
        assert linha.origem == OrigemDoSft.ALUNO and linha.messages[2]["content"] == "otima" and linha.similaridade == 0.95
        assert linha.messages[0]["content"] == PROMPT_DO_ALUNO
        assert par.chosen[0]["content"] == "otima" and par.rejected[0]["content"] == "pessima" and par.motivo_da_rejeicao == "wrong_legal_basis"

    def test_sem_aprovada_sai_ou_cai_no_professor(self):
        amostras = [amostra(1, "ruim", correto=False)]
        sem, par = selecionar(prompt(), amostras, ParametrosOnPolicy(dataset_versao="v", aluno="local:x"))
        assert sem is None and par is None
        com, _ = selecionar(prompt(), amostras, ParametrosOnPolicy(dataset_versao="v", aluno="local:x", fallback_para_o_professor=True))
        assert com.origem == OrigemDoSft.PROFESSOR and com.messages[2]["content"] == "Ref."

    def test_so_aprovadas_nao_gera_par(self):
        _, par = selecionar(prompt(), [amostra(1, "a"), amostra(2, "b")], ParametrosOnPolicy(dataset_versao="v", aluno="local:x"))
        assert par is None


class ServicoDuble:
    modelo_juiz = "duble:juiz"
    modelo_aluno = "local:duble"

    def __init__(self, estourar_na_amostra: int | None = None, falhar_na_amostra: int | None = None, juiz_falha=False):
        self.n = 0
        self.estourar = estourar_na_amostra
        self.falhar = falhar_na_amostra
        self.juiz_falha = juiz_falha

    async def amostrar(self, pergunta):
        self.n += 1
        if self.n == self.estourar:
            raise OrcamentoExcedido("teto")
        if self.n == self.falhar:
            raise RuntimeError("aluno caiu")
        # Amostras impares sao "boas", pares sao "ruins" (e uma cita artigo inexistente).
        texto = f"Resposta boa {self.n} pelo art. 7º" if self.n % 2 else f"Resposta ruim {self.n} pelo art. 72"
        return RespostaDeTexto(texto=texto, input_tokens=10, output_tokens=5)

    async def julgar(self, casos):
        if self.juiz_falha:
            raise RuntimeError("juiz caiu")
        return [VereditoDeAvaliacao(indice=c.indice, correto="boa" in c.resposta, criterios_atendidos=0, criterios_total=0, grounded=True,
                                    abstencao=False, categoria=None if "boa" in c.resposta else CategoriaDeErro.INCOMPLETE_ANSWER,
                                    similaridade=0.9 if "boa" in c.resposta else 0.2) for c in casos]


@pytest.fixture
def ambiente(tmp_path, corpus):
    datasets = DatasetRepository(tmp_path)
    manifesto = ManifestoDoDataset(versao="v1", nome="v1", criado_em=AGORA, parametros=ParametrosDeDataset(nome="v1", jobs_de_filtro=["f"]),
                                   treino=ContagemPorTipo(total=3), held_out=ContagemPorTipo(total=1), versao_do_prompt_do_aluno="p")
    treino = [{"id": f"t{i}", "tipo": "conceito", "dificuldade": "basica", "artigos_citados": ["LGPD:art.7:inc.IX"],
               "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": f"Pergunta {i}?"}, {"role": "assistant", "content": f"Professor {i}."}]}
              for i in range(3)]
    held = [{**treino[0], "id": "h0"}]
    datasets.salvar(manifesto, {Split.TREINO: treino, Split.HELD_OUT: held})
    return corpus, OnPolicyRepository(tmp_path), datasets


class TestEngine:
    async def test_fluxo_completo(self, ambiente):
        corpus, repo, datasets = ambiente
        job = novo_job(ParametrosOnPolicy(dataset_versao="v1", aluno="local:aluno", amostras_por_prompt=2, concorrencia=1, teto_usd=0))
        job = await EngineOnPolicy(ServicoDuble(), corpus, repo, datasets).executar(job)
        assert job.status == StatusDoJob.CONCLUIDO and job.prompts == 3 and job.amostras == 6
        assert job.amostras_aprovadas == 3 and job.prompts_com_aprovada == 3
        assert job.linhas_sft == 3 and job.pares_dpo == 3 and job.linhas_sft_do_professor == 0
        sft = repo.linhas(job.id, "sft")
        assert all(l["origem"] == "aluno" and "boa" in l["messages"][2]["content"] for l in sft)
        dpo = repo.linhas(job.id, "dpo")
        assert all("ruim" in p["rejected"][0]["content"] for p in dpo)
        amostras = repo.amostras(job.id)
        assert len(amostras) == 6 and any(a.artigos_inexistentes == ["LGPD:art.72"] for a in amostras)
        # Nunca toca o held-out: so' ids do split de treino.
        assert {a.prompt_id for a in amostras} == {"t0", "t1", "t2"}

    async def test_orcamento_para_e_consolida_o_que_houver(self, ambiente):
        corpus, repo, datasets = ambiente
        job = novo_job(ParametrosOnPolicy(dataset_versao="v1", aluno="local:aluno", amostras_por_prompt=2, concorrencia=1, teto_usd=0))
        job = await EngineOnPolicy(ServicoDuble(estourar_na_amostra=3), corpus, repo, datasets).executar(job)
        assert job.status == StatusDoJob.ORCAMENTO_EXCEDIDO and job.amostras == 2 and job.lotes_do_juiz == 0

    async def test_amostra_com_erro_nao_derruba(self, ambiente):
        corpus, repo, datasets = ambiente
        job = novo_job(ParametrosOnPolicy(dataset_versao="v1", aluno="local:aluno", amostras_por_prompt=2, concorrencia=1, teto_usd=0))
        job = await EngineOnPolicy(ServicoDuble(falhar_na_amostra=2), corpus, repo, datasets).executar(job)
        assert job.status == StatusDoJob.CONCLUIDO
        assert sum(1 for a in repo.amostras(job.id) if a.erro) == 1

    async def test_juiz_falhando(self, ambiente):
        corpus, repo, datasets = ambiente
        job = novo_job(ParametrosOnPolicy(dataset_versao="v1", aluno="local:aluno", teto_usd=0))
        job = await EngineOnPolicy(ServicoDuble(juiz_falha=True), corpus, repo, datasets).executar(job)
        assert job.status == StatusDoJob.FALHOU and job.linhas_sft == 0

    async def test_dataset_inexistente(self, ambiente):
        corpus, repo, datasets = ambiente
        job = await EngineOnPolicy(ServicoDuble(), corpus, repo, datasets).executar(novo_job(ParametrosOnPolicy(dataset_versao="nao", aluno="local:x")))
        assert job.status == StatusDoJob.FALHOU


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch, ambiente):
        corpus, repo, datasets = ambiente
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        ingestao.registro.definir(corpus)
        monkeypatch.setattr(on_policy_endpoint, "_repositorio", repo)
        monkeypatch.setattr(on_policy_endpoint, "_datasets", datasets)
        monkeypatch.setattr(on_policy_endpoint, "servico_on_policy", lambda p: ServicoDuble())
        with TestClient(app) as c:
            yield c
        ingestao.registro.limpar()

    def test_cria_e_baixa_arquivos(self, client):
        r = client.post("/api/v1/on-policy/jobs", json={"dataset_versao": "v1", "aluno": "local:aluno", "amostras_por_prompt": 2, "teto_usd": 0})
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        fim = time.time() + 5
        while time.time() < fim:
            job = client.get(f"/api/v1/on-policy/jobs/{job_id}").json()
            if job["status"] not in ("pendente", "executando"):
                break
            time.sleep(0.05)
        assert job["status"] == "concluido" and job["pares_dpo"] == 3
        sft = client.get(f"/api/v1/on-policy/jobs/{job_id}/arquivo/sft")
        assert sft.status_code == 200 and sft.text.count("\n") == 3
        assert json.loads(client.get(f"/api/v1/on-policy/jobs/{job_id}/arquivo/dpo").text.splitlines()[0])["chosen"]
        assert len(client.get(f"/api/v1/on-policy/jobs/{job_id}/amostras", params={"so_reprovadas": True}).json()) == 3
        assert client.get(f"/api/v1/on-policy/jobs/{job_id}/arquivo/outro").status_code == 404

    def test_erros(self, client):
        assert client.post("/api/v1/on-policy/jobs", json={"dataset_versao": "nao", "aluno": "local:x"}).status_code == 404
        assert client.post("/api/v1/on-policy/jobs", json={"dataset_versao": "v1", "aluno": "openai:x"}).status_code == 422
        assert client.get("/api/v1/on-policy/jobs/nao").status_code == 404


class TestPreparacaoDpo:
    def _par(self, **kw):
        base = {"id": "p", "tipo": "conceito", "prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "P?"}],
                "chosen": [{"role": "assistant", "content": "boa"}], "rejected": [{"role": "assistant", "content": "ruim"}]}
        base.update(kw)
        return base

    def test_prefixo_nos_dois_lados(self):
        pares = dpo_lora.preparar_pares([self._par()], PREFIXO_SEM_PENSAR_QWEN3)
        assert pares[0]["chosen"][0]["content"] == PREFIXO_SEM_PENSAR_QWEN3 + "boa"
        assert pares[0]["rejected"][0]["content"] == PREFIXO_SEM_PENSAR_QWEN3 + "ruim"
        assert set(pares[0]) == {"prompt", "chosen", "rejected"}

    def test_validacao(self):
        with pytest.raises(ValueError, match="prompt"):
            dpo_lora.preparar_pares([self._par(prompt=[{"role": "user", "content": "x"}])], "")
        with pytest.raises(ValueError, match="rejected"):
            dpo_lora.preparar_pares([self._par(rejected=[{"role": "assistant", "content": " "}])], "")

    def test_dry_run(self, tmp_path):
        caminho = tmp_path / "dpo.jsonl"
        caminho.write_text(json.dumps(self._par()) + "\n", encoding="utf-8")
        assert dpo_lora.main(["--perfil", "colab-t4", "--pares", str(caminho), "--dataset-versao", "ab12cd34ef56", "--saida", str(tmp_path / "s"), "--dry-run"]) == 0
        run = json.loads((tmp_path / "s" / "run.json").read_text(encoding="utf-8"))
        assert run["receita"] == "dpo_lora" and run["modelo_local_id"] == "local:qwen3-1.7b-lgpd-dpo-ab12cd34" and run["hiperparametros"]["pares"] == 1
