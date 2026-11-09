"""Relatorio de cobertura da lei pelos exemplos da sintese."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import relatorios as relatorios_endpoint
from app.main import app
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.sintese import (
    Dificuldade,
    ExemploDescartado,
    ExemploSintetico,
    JobDeSintese,
    ParametrosDeSintese,
    StatusDoJob,
    TipoDeTarefa,
)
from app.services.corpus import ingestao
from app.services.corpus.chunking import chunkar
from app.services.relatorios import cobertura

AGORA = dt.datetime(2026, 11, 8, tzinfo=dt.timezone.utc)
ARTIGOS = ["LGPD:art.1", "LGPD:art.5", "LGPD:art.7", "LGPD:art.48"]


def exemplo(citados: list[str], contexto: list[str], alvo: str | None = None, job: str = "j1") -> ExemploSintetico:
    return ExemploSintetico(
        id=f"{job}-{len(citados)}-{'-'.join(contexto)}-{alvo}", job_id=job, tipo=TipoDeTarefa.CONCEITO,
        dificuldade=Dificuldade.BASICA, pergunta="P?", raciocinio="R", resposta="A", artigos_citados=citados,
        citacoes_validas=True, contexto_ids=contexto, alvo_id=alvo, modelo="m", versao_do_prompt="v", criado_em=AGORA,
    )


class TestRelatorio:
    def test_conta_alvo_citacao_e_cita_o_alvo(self):
        exemplos = [
            exemplo(["LGPD:art.1"], ["LGPD:art.1", "LGPD:art.5"], alvo="LGPD:art.1"),
            # Cita so' a definicao: nao citou o proprio alvo.
            exemplo(["LGPD:art.5:inc.I", "LGPD:art.5:inc.II"], ["LGPD:art.48", "LGPD:art.5"], alvo="LGPD:art.48"),
            exemplo(["LGPD:art.48:par.1", "LGPD:art.5"], ["LGPD:art.48", "LGPD:art.5"], alvo="LGPD:art.48"),
        ]
        descartados = [ExemploDescartado(job_id="j1", motivo="x", pergunta="P?", alvo_id="LGPD:art.7")]
        r = cobertura.relatorio(ARTIGOS, exemplos, descartados, ["j1"])

        por = {c.artigo_id: c for c in r.por_artigo}
        assert [c.artigo_id for c in r.por_artigo] == ARTIGOS
        assert por["LGPD:art.48"].alvo_aceitos == 2 and por["LGPD:art.48"].aceitos_que_citam_o_alvo == 1
        # Dois incisos do art. 5 no mesmo exemplo contam uma citacao so'.
        assert por["LGPD:art.5"].citacoes == 2 and por["LGPD:art.5"].alvo_aceitos == 0
        assert por["LGPD:art.7"].alvo_descartados == 1 and por["LGPD:art.7"].citacoes == 0
        assert r.exemplos == 3 and r.alvos_inferidos == 0
        assert r.artigos_como_alvo == 2 and r.artigos_citados == 3
        assert r.nunca_citados == ["LGPD:art.7"]
        assert r.taxa_cita_o_alvo == round(2 / 3, 4)
        assert r.concentracao_top5 == 1.0
        assert r.menos_citados[0].artigo_id == "LGPD:art.7"

    def test_sem_alvo_id_usa_o_primeiro_do_contexto_e_avisa(self):
        r = cobertura.relatorio(ARTIGOS, [exemplo(["LGPD:art.7"], ["LGPD:art.7", "LGPD:art.5"])], [], ["j1"])
        assert r.alvos_inferidos == 1 and "contexto_ids[0]" in r.aviso
        assert next(c for c in r.por_artigo if c.artigo_id == "LGPD:art.7").aceitos_que_citam_o_alvo == 1

    def test_descartados_antigos_sem_alvo_avisam(self):
        antigo = ExemploDescartado(job_id="j1", motivo="x", pergunta="P?")
        r = cobertura.relatorio(ARTIGOS, [], [antigo], ["j1"])
        assert "alvo_descartados fica zerado" in r.aviso
        assert all(c.alvo_descartados == 0 for c in r.por_artigo)

    def test_sem_exemplos_nao_divide_por_zero(self):
        r = cobertura.relatorio(ARTIGOS, [], [], ["j1"])
        assert r.taxa_cita_o_alvo is None and r.concentracao_top5 is None and r.nunca_citados == ARTIGOS

    def test_citacao_de_fonte_complementar_nao_entra(self):
        r = cobertura.relatorio(ARTIGOS, [exemplo(["DOSIMETRIA:art.3"], ["LGPD:art.1"], alvo="LGPD:art.1")], [], ["j1"])
        assert r.artigos_citados == 0 and r.concentracao_top5 is None

    def test_markdown(self):
        r = cobertura.relatorio(ARTIGOS, [exemplo(["LGPD:art.1"], ["LGPD:art.1"], alvo="LGPD:art.1")], [], ["j1"])
        md = cobertura.markdown(r)
        assert "**Artigos como alvo:** 1/4" in md and "| LGPD:art.1 | 1 | 0 | 1 | 1 |" in md


_LEI = """
Art. 1º Esta Lei dispõe sobre o tratamento de dados pessoais.
Art. 5º Para os fins desta Lei, considera-se:
I - dado pessoal: informação relacionada a pessoa natural identificada ou identificável;
Art. 7º O tratamento de dados pessoais somente poderá ser realizado mediante consentimento.
""".strip().splitlines()


class TestEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        repo = SinteseRepository(tmp_path)
        job = JobDeSintese(id="j1", status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeSintese(), criado_em=AGORA, atualizado_em=AGORA)
        repo.salvar_job(job)
        repo.acrescentar_exemplos("j1", [exemplo(["LGPD:art.5:inc.I"], ["LGPD:art.1", "LGPD:art.5"], alvo="LGPD:art.1")])
        corpus = ingestao.Corpus("LGPD", chunkar(_LEI), Path("fake.txt"))
        monkeypatch.setattr(relatorios_endpoint, "_sintese", repo)
        monkeypatch.setattr(relatorios_endpoint, "corpus_ativo", lambda fonte="LGPD": corpus)
        with TestClient(app) as c:
            yield c

    def test_json_e_markdown(self, client):
        r = client.get("/api/v1/relatorios/cobertura", params={"jobs": "j1"})
        assert r.status_code == 200
        corpo = r.json()
        assert corpo["artigos_no_corpus"] == 3 and corpo["artigos_como_alvo"] == 1 and corpo["taxa_cita_o_alvo"] == 0.0
        md = client.get("/api/v1/relatorios/cobertura.md", params={"jobs": "j1"})
        assert md.status_code == 200 and "| LGPD:art.5 | 0 | 0 | 0 | 1 |" in md.text

    def test_job_inexistente_e_lista_vazia(self, client):
        assert client.get("/api/v1/relatorios/cobertura", params={"jobs": "nao"}).status_code == 404
        assert client.get("/api/v1/relatorios/cobertura", params={"jobs": ","}).status_code == 422
