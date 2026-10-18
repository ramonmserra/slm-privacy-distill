"""Calibracao humana do juiz (kappa) e relatorios das condicoes."""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import avaliacao as avaliacao_endpoint
from app.api.v1.endpoints import relatorios as relatorios_endpoint
from app.main import app
from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.schemas.avaliacao import (
    AnotacaoRegistrada,
    CategoriaDeErro,
    Conjunto,
    ItemDeAvaliacao,
    MetricasDoRun,
    ParametrosDeAvaliacao,
    RespostaDoModelo,
    ResultadoDoItem,
    RunDeAvaliacao,
    VereditoDeAvaliacao,
)
from app.schemas.sintese import StatusDoJob
from app.services.avaliacao import calibracao
from app.services.relatorios import condicoes

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)


def resultado(id_: str, acerto: bool, grounded: bool = True) -> ResultadoDoItem:
    v = VereditoDeAvaliacao(indice=1, correto=acerto, criterios_atendidos=0, criterios_total=0, grounded=grounded, abstencao=False,
                            categoria=None if acerto else CategoriaDeErro.INCOMPLETE_ANSWER, similaridade=0.5)
    return ResultadoDoItem(item=ItemDeAvaliacao(id=id_, conjunto=Conjunto.BANCO_INICIAL, pergunta="P", referencia="R", tipo="pergunta"),
                           resposta=RespostaDoModelo(item_id=id_, texto="x", latencia_ms=10), veredito=v, acerto=acerto, alucinacao=not grounded)


def anotacao(item: str, anotador: str, correto: bool, grounded: bool | None = None) -> AnotacaoRegistrada:
    return AnotacaoRegistrada(run_id="r", item_id=item, anotador=anotador, correto=correto, grounded=grounded, criado_em=AGORA)


class TestCalibracao:
    def test_concordancia_perfeita_e_maioria(self):
        itens = [resultado("1", True), resultado("2", False), resultado("3", True), resultado("4", False)]
        anotacoes = [anotacao(i, "ana", a) for i, a in (("1", True), ("2", False), ("3", True), ("4", False))]
        anotacoes += [anotacao(i, "bia", a) for i, a in (("1", True), ("2", False), ("3", False), ("4", False))]
        anotacoes += [anotacao(i, "caio", a) for i, a in (("1", True), ("2", False), ("3", True), ("4", False))]
        r = calibracao.resumir("r", itens, anotacoes)
        assert r.itens_anotados == 4 and r.anotadores == ["ana", "bia", "caio"]
        ana = next(p for p in r.por_anotador if p.anotador == "ana")
        assert ana.kappa_correto == 1.0 and ana.acuracia_do_juiz_vs_humano == 1.0 and ana.kappa_grounded is None
        assert r.kappa_correto_maioria == 1.0  # maioria (2 de 3) coincide com o juiz em todos
        assert 0 < r.kappa_entre_humanos < 1
        assert "quase perfeita" in r.leitura and "indicativo" in r.leitura

    def test_reanotar_substitui_e_empate_descarta(self):
        itens = [resultado("1", True), resultado("2", True)]
        anotacoes = [anotacao("1", "ana", False), anotacao("1", "ana", True), anotacao("2", "ana", True), anotacao("2", "bia", False)]
        r = calibracao.resumir("r", itens, anotacoes)
        ana = next(p for p in r.por_anotador if p.anotador == "ana")
        assert ana.acuracia_do_juiz_vs_humano == 1.0  # a reanotacao de "1" valeu
        assert r.kappa_correto_maioria is None  # item 1 tem 1 anotador; item 2 empatou
        assert "ao menos dois humanos" in r.leitura

    def test_kappa_grounded(self):
        itens = [resultado("1", True, grounded=True), resultado("2", True, grounded=False)]
        r = calibracao.resumir("r", itens, [anotacao("1", "ana", True, True), anotacao("2", "ana", True, False)])
        assert r.por_anotador[0].kappa_grounded == 1.0

    def test_sem_anotacoes(self):
        r = calibracao.resumir("r", [resultado("1", True)], [])
        assert r.itens_anotados == 0 and r.por_anotador == [] and r.kappa_correto_maioria is None


def _run(id_: str, modelo: str, acuracia: float, versao="v") -> RunDeAvaliacao:
    m = MetricasDoRun(itens=10, respondidos=10, julgados=10, acertos=int(acuracia * 10), acuracia=acuracia, acuracia_ic95=(acuracia - 0.2, acuracia + 0.2),
                      alucinacoes=1, taxa_de_alucinacao=0.1, taxa_de_alucinacao_ic95=(0.0, 0.4), abstencoes_esperadas=2, abstencoes_adequadas=1,
                      taxa_citou_referencia=0.8, latencia_p50_ms=120.0, custo_usd=0.05, custo_usd_por_item=0.005)
    return RunDeAvaliacao(id=id_, status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeAvaliacao(modelo=modelo), modelo=modelo,
                          versao_da_medida=versao, criado_em=AGORA, atualizado_em=AGORA, metricas=m)


class TestRelatorios:
    def test_tabela_e_markdown(self):
        t = condicoes.tabela([_run("a", "openai:prof", 0.9), _run("b", "local:aluno", 0.7)], {"a": "C0 professor", "b": "C3 aluno"})
        assert t.comparaveis and [l.rotulo for l in t.linhas] == ["C0 professor", "C3 aluno"]
        assert t.linhas[1].abstencoes_adequadas == "1/2" and t.linhas[1].custo_usd_por_item == 0.005
        md = condicoes.markdown(t)
        assert md.startswith("| Condição |") and "| C3 aluno | `local:aluno` |" in md and "70.0% [50–90]" in md
        assert "Wilson" in md

    def test_versoes_diferentes_avisam(self):
        t = condicoes.tabela([_run("a", "x", 0.5, "v1"), _run("b", "y", 0.5, "v2")])
        assert not t.comparaveis and "ATENCAO" in t.aviso and t.linhas[0].rotulo == "x"


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        repo = AvaliacaoRepository(tmp_path)
        run_a, run_b = _run("a", "openai:prof", 0.9), _run("b", "local:aluno", 0.7)
        repo.salvar_run(run_a)
        repo.salvar_run(run_b)
        repo.acrescentar_itens("a", [resultado("1", True), resultado("2", True), resultado("3", False)])
        repo.acrescentar_itens("b", [resultado("1", True), resultado("2", False), resultado("3", False)])
        monkeypatch.setattr(avaliacao_endpoint, "_repositorio", repo)
        monkeypatch.setattr(relatorios_endpoint, "_avaliacao", repo)
        with TestClient(app) as c:
            yield c

    def test_anotar_e_calibrar(self, client):
        r = client.post("/api/v1/avaliacao/runs/a/anotacoes", json={"item_id": "1", "anotador": "ana", "correto": True})
        assert r.status_code == 201 and r.json()["run_id"] == "a"
        client.post("/api/v1/avaliacao/runs/a/anotacoes", json={"item_id": "3", "anotador": "ana", "correto": False, "comentario": "ok"})
        assert client.post("/api/v1/avaliacao/runs/a/anotacoes", json={"item_id": "nao", "anotador": "ana", "correto": True}).status_code == 404
        cal = client.get("/api/v1/avaliacao/runs/a/calibracao").json()
        assert cal["itens_anotados"] == 2 and cal["por_anotador"][0]["kappa_correto"] == 1.0

    def test_relatorios(self, client):
        t = client.get("/api/v1/relatorios/condicoes", params={"runs": "a,b", "rotulos": "a:C0,b:C3"}).json()
        assert [l["rotulo"] for l in t["linhas"]] == ["C0", "C3"]
        md = client.get("/api/v1/relatorios/condicoes.md", params={"runs": "a,b"})
        assert md.status_code == 200 and md.text.startswith("| Condição |")
        pareadas = client.get("/api/v1/relatorios/pareadas", params={"referencia": "a", "candidatos": "b"}).json()
        assert len(pareadas) == 1 and pareadas[0]["itens_em_comum"] == 3 and pareadas[0]["pioraram"] == ["2"]
        custos = client.get("/api/v1/relatorios/custos").json()
        assert "total" in custos and "por_etapa" in custos
        assert client.get("/api/v1/relatorios/condicoes", params={"runs": "nao"}).status_code == 404
        assert client.get("/api/v1/relatorios/condicoes", params={"runs": ","}).status_code == 422
