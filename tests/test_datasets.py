"""Datasets: split por hash (estavel e disjunto), formato de chat, versao por
conteudo, origens com e sem filtro, endpoints."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import datasets as datasets_endpoint
from app.main import app
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.dataset import ParametrosDeDataset, Split
from app.schemas.filtro import DecisaoDoFiltro, JobDeFiltro, ParametrosDeFiltro
from app.schemas.sintese import Dificuldade, ExemploSintetico, JobDeSintese, ParametrosDeSintese, StatusDoJob, TipoDeTarefa
from app.services.datasets.construtor import ConstrutorDeDataset, OrigemDesconhecida, OrigemVazia
from app.services.datasets.prompt_do_aluno import PROMPT_DO_ALUNO, VERSAO_DO_PROMPT_DO_ALUNO, resposta_de_treino
from app.services.datasets.split import split_de

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)


def exemplo(id_: str, tipo=TipoDeTarefa.CONCEITO, job="s1") -> ExemploSintetico:
    return ExemploSintetico(
        id=id_, job_id=job, tipo=tipo, dificuldade=Dificuldade.BASICA, pergunta=f"Pergunta {id_}?",
        raciocinio=f"Raciocinio {id_}", resposta=f"Resposta {id_}", artigos_citados=["LGPD:art.7"],
        citacoes_validas=True, contexto_ids=["LGPD:art.7"], modelo="duble", versao_do_prompt="vs1", criado_em=AGORA,
    )


@pytest.fixture
def repos(tmp_path):
    sintese, filtro, datasets = SinteseRepository(tmp_path), FiltroRepository(tmp_path), DatasetRepository(tmp_path)
    sintese.salvar_job(JobDeSintese(id="s1", status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeSintese(), criado_em=AGORA, atualizado_em=AGORA))
    exemplos = [exemplo(f"e{i}", TipoDeTarefa.CONCEITO if i % 2 else TipoDeTarefa.INCIDENTE) for i in range(40)]
    sintese.acrescentar_exemplos("s1", exemplos)
    filtro.salvar_job(JobDeFiltro(id="f1", status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeFiltro(jobs_de_sintese=["s1"]), criado_em=AGORA, atualizado_em=AGORA))
    for i, e in enumerate(exemplos):
        filtro.acrescentar_decisao("f1", DecisaoDoFiltro(
            exemplo_id=e.id, job_filtro_id="f1", job_sintese_id="s1", tipo=e.tipo, aprovado=(i % 4 != 0),
            motivos=[] if i % 4 else ["incorreto_juridicamente"], versao_do_filtro="vf1", criado_em=AGORA,
        ))
    return sintese, filtro, datasets


class TestSplit:
    def test_estavel_e_respeita_fracao_aproximada(self):
        ids = [f"exemplo-{i}" for i in range(2000)]
        a = [split_de(i, 0.15) for i in ids]
        b = [split_de(i, 0.15) for i in ids]
        assert a == b
        fracao = sum(1 for s in a if s == Split.HELD_OUT) / len(a)
        assert 0.12 < fracao < 0.18

    def test_fracao_zero_e_tudo_treino(self):
        assert all(split_de(f"x{i}", 0.0) == Split.TREINO for i in range(50))

    def test_held_out_nao_depende_do_restante_do_dataset(self):
        """O mesmo id cai no mesmo lado em qualquer versao: e' o que impede o
        held-out da v1 de virar treino na v2."""

        assert split_de("abc", 0.3) == split_de("abc", 0.3)


class TestFormato:
    def test_resposta_de_treino(self):
        assert resposta_de_treino("R", "C", True) == "Raciocinio:\nR\n\nResposta:\nC"
        assert resposta_de_treino("R", "C", False) == "C"
        assert resposta_de_treino("", "C", True) == "C"

    def test_linha_de_chat(self):
        linha = ConstrutorDeDataset.linha(exemplo("e1"), incluir_raciocinio=True)
        assert [m.role for m in linha.messages] == ["system", "user", "assistant"]
        assert linha.messages[0].content == PROMPT_DO_ALUNO
        assert linha.messages[1].content == "Pergunta e1?"
        assert linha.messages[2].content.startswith("Raciocinio:")
        assert linha.tipo == "conceito" and linha.artigos_citados == ["LGPD:art.7"]


class TestConstrutor:
    def test_so_aprovados_do_filtro(self, repos):
        sintese, filtro, datasets = repos
        manifesto = ConstrutorDeDataset(datasets, filtro, sintese).construir(ParametrosDeDataset(nome="v1", jobs_de_filtro=["f1"]))
        assert manifesto.treino.total + manifesto.held_out.total == 30  # 40 - 10 reprovados
        assert set(manifesto.treino.por_tipo) <= {"conceito", "incidente"}
        assert manifesto.versoes_do_filtro == ["vf1"] and manifesto.versoes_do_prompt_de_sintese == ["vs1"]
        assert manifesto.versao_do_prompt_do_aluno == VERSAO_DO_PROMPT_DO_ALUNO
        treino = datasets.linhas(manifesto.versao, Split.TREINO)
        held = datasets.linhas(manifesto.versao, Split.HELD_OUT)
        assert len(treino) == manifesto.treino.total and len(held) == manifesto.held_out.total
        assert not ({l["id"] for l in treino} & {l["id"] for l in held})
        assert json.loads(open(datasets.caminho_do_split(manifesto.versao, Split.TREINO), encoding="utf-8").readline())["messages"][0]["role"] == "system"

    def test_sem_filtro_leva_todos_e_versao_difere(self, repos):
        sintese, filtro, datasets = repos
        construtor = ConstrutorDeDataset(datasets, filtro, sintese)
        com = construtor.construir(ParametrosDeDataset(nome="com", jobs_de_filtro=["f1"]))
        sem = construtor.construir(ParametrosDeDataset(nome="sem", jobs_de_sintese_sem_filtro=["s1"]))
        assert sem.treino.total + sem.held_out.total == 40
        assert sem.versao != com.versao
        # Held-out do "com" esta contido no held-out do "sem": mesmo id, mesmo lado.
        held_com = {l["id"] for l in datasets.linhas(com.versao, Split.HELD_OUT)}
        held_sem = {l["id"] for l in datasets.linhas(sem.versao, Split.HELD_OUT)}
        assert held_com <= held_sem

    def test_reconstruir_da_a_mesma_versao_e_duplicados_sao_contados(self, repos):
        sintese, filtro, datasets = repos
        construtor = ConstrutorDeDataset(datasets, filtro, sintese)
        a = construtor.construir(ParametrosDeDataset(nome="a", jobs_de_filtro=["f1"]))
        b = construtor.construir(ParametrosDeDataset(nome="b", jobs_de_filtro=["f1"], jobs_de_sintese_sem_filtro=["s1"]))
        c = construtor.construir(ParametrosDeDataset(nome="c", jobs_de_filtro=["f1"]))
        assert a.versao == c.versao
        assert b.duplicados_entre_origens == 30
        assert len(datasets.listar()) == 2

    def test_sem_raciocinio_muda_a_versao(self, repos):
        sintese, filtro, datasets = repos
        construtor = ConstrutorDeDataset(datasets, filtro, sintese)
        a = construtor.construir(ParametrosDeDataset(nome="a", jobs_de_filtro=["f1"]))
        b = construtor.construir(ParametrosDeDataset(nome="b", jobs_de_filtro=["f1"], incluir_raciocinio=False))
        assert a.versao != b.versao
        assert not datasets.linhas(b.versao, Split.TREINO)[0]["messages"][2]["content"].startswith("Raciocinio")

    def test_origens_invalidas(self, repos):
        sintese, filtro, datasets = repos
        construtor = ConstrutorDeDataset(datasets, filtro, sintese)
        with pytest.raises(OrigemDesconhecida):
            construtor.construir(ParametrosDeDataset(nome="x", jobs_de_filtro=["nao"]))
        filtro.salvar_job(JobDeFiltro(id="vazio", status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeFiltro(jobs_de_sintese=["s1"]), criado_em=AGORA, atualizado_em=AGORA))
        with pytest.raises(OrigemVazia):
            construtor.construir(ParametrosDeDataset(nome="x", jobs_de_filtro=["vazio"]))

    def test_parametros_exigem_origem(self):
        with pytest.raises(ValueError):
            ParametrosDeDataset(nome="x")
        with pytest.raises(ValueError):
            ParametrosDeDataset(nome="nome com espaco", jobs_de_filtro=["f"])


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch, repos):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setattr(datasets_endpoint, "_repositorio", DatasetRepository(tmp_path))
        with TestClient(app) as c:
            yield c

    def test_construir_listar_amostrar_baixar(self, client):
        r = client.post("/api/v1/datasets", json={"nome": "v1", "jobs_de_filtro": ["f1"]})
        assert r.status_code == 201, r.text
        versao = r.json()["versao"]
        assert client.get("/api/v1/datasets").json()[0]["versao"] == versao
        assert client.get(f"/api/v1/datasets/{versao}").json()["nome"] == "v1"
        amostra = client.get(f"/api/v1/datasets/{versao}/amostra", params={"split": "held_out", "limite": 2}).json()
        assert 0 < len(amostra) <= 2 and amostra[0]["messages"][0]["role"] == "system"
        arquivo = client.get(f"/api/v1/datasets/{versao}/arquivo/treino")
        assert arquivo.status_code == 200
        assert arquivo.text.count("\n") == r.json()["treino"]["total"]

    def test_erros(self, client):
        assert client.get("/api/v1/datasets/nao-existe").status_code == 404
        assert client.post("/api/v1/datasets", json={"nome": "x", "jobs_de_filtro": ["nao"]}).status_code == 404
        assert client.post("/api/v1/datasets", json={"nome": "x"}).status_code == 422
