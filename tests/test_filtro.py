"""Filtro: pre-checagens deterministicas, decisao a partir da rubrica e da
self-consistency, laco de lotes e endpoints. Professor e juiz sao dublês."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import filtro as filtro_endpoint
from app.main import app
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas import filtro as F
from app.schemas.sintese import Dificuldade, ExemploSintetico, JobDeSintese, ParametrosDeSintese, StatusDoJob, TipoDeTarefa
from app.services.corpus import ingestao
from app.services.corpus.chunking import chunkar
from app.services.filtro.engine import FiltroDeQualidade, motivos_do_veredito, novo_job, pre_checagem, score_de_incerteza
from app.services.filtro.llm_service import VERSAO_DO_FILTRO, CasoDeQualidade, _build_juiz_prompt

_LEI = """
Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses:
I - mediante o fornecimento de consentimento pelo titular;
IX - quando necessário para atender aos interesses legítimos do controlador;
Art. 18. O titular dos dados pessoais tem direito a obter do controlador:
VI - eliminação dos dados pessoais tratados com o consentimento do titular;
Art. 48. O controlador deverá comunicar à autoridade nacional a ocorrência de incidente de segurança.
""".strip().splitlines()

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)


@pytest.fixture
def corpus():
    return ingestao.Corpus("LGPD", chunkar(_LEI), Path("fake.txt"))


def exemplo(id_: str, tipo=TipoDeTarefa.BASE_LEGAL, citados=("LGPD:art.7:inc.IX",), invalidos=(), resposta="Aplica-se o legitimo interesse.", job="s1") -> ExemploSintetico:
    return ExemploSintetico(
        id=id_, job_id=job, tipo=tipo, dificuldade=Dificuldade.BASICA,
        pergunta=f"Pergunta {id_} sobre o tratamento de dados com detalhes?", raciocinio="Pelo LGPD:art.7:inc.IX.",
        resposta=resposta, artigos_citados=list(citados), artigos_citados_invalidos=list(invalidos),
        citacoes_validas=not invalidos, contexto_ids=["LGPD:art.7", "LGPD:art.18"], modelo="duble", versao_do_prompt="v",
        criado_em=AGORA,
    )


def veredito(indice: int, equivalentes=(True, True, True), **kw) -> F.VereditoDeQualidade:
    base = dict(indice=indice, correcao_juridica=True, lastro_nas_citacoes=True, completude=True, abstencao_adequada=None,
                sem_dado_pessoal_real=True, nota=5, problemas="", reamostragens_equivalentes=list(equivalentes))
    base.update(kw)
    return F.VereditoDeQualidade(**base)


class TestPreChecagem:
    def test_limpo(self, corpus):
        assert pre_checagem(exemplo("a"), corpus, True) == []

    def test_citacao_invalida_e_sem_citacao(self, corpus):
        assert F.MOTIVO_CITACAO_INVALIDA in pre_checagem(exemplo("a", invalidos=["LGPD:art.72"]), corpus, True)
        assert F.MOTIVO_CITACAO_INVALIDA in pre_checagem(exemplo("a", citados=["LGPD:art.99"]), corpus, True)
        assert pre_checagem(exemplo("a", citados=[]), corpus, True) == [F.MOTIVO_SEM_CITACAO]
        assert pre_checagem(exemplo("a", citados=[], tipo=TipoDeTarefa.SEM_RESPOSTA), corpus, True) == []

    def test_sem_exigir_citacoes(self, corpus):
        assert pre_checagem(exemplo("a", invalidos=["LGPD:art.72"]), corpus, False) == []

    def test_artigo_inexistente_na_prosa(self, corpus):
        e = exemplo("a", resposta="Conforme o art. 72 da LGPD, aplica-se multa.")
        assert pre_checagem(e, corpus, True) == [F.MOTIVO_ARTIGO_INEXISTENTE_NO_TEXTO]


class TestDecisao:
    def test_score_de_incerteza(self):
        assert score_de_incerteza([]) is None
        assert score_de_incerteza([True, True, True]) == 0.0
        assert score_de_incerteza([True, False, False]) == 0.667

    def test_motivos_do_veredito(self):
        v = veredito(1, correcao_juridica=False, completude=False, sem_dado_pessoal_real=False)
        assert motivos_do_veredito(v, TipoDeTarefa.CONCEITO) == [F.MOTIVO_INCORRETO, F.MOTIVO_INCOMPLETO, F.MOTIVO_DADO_PESSOAL_REAL]
        assert motivos_do_veredito(veredito(1, abstencao_adequada=False), TipoDeTarefa.CONCEITO) == []
        assert motivos_do_veredito(veredito(1, abstencao_adequada=False), TipoDeTarefa.SEM_RESPOSTA) == [F.MOTIVO_ABSTENCAO_INADEQUADA]

    def test_prompt_do_juiz_e_cego_e_traz_o_rastro(self):
        caso = CasoDeQualidade(indice=1, tipo=TipoDeTarefa.CONCEITO, pergunta="P?", raciocinio="R", resposta="X",
                               artigos_citados=["LGPD:art.7"], contexto=[], reamostragens=["alt1", "alt2"])
        prompt = _build_juiz_prompt([caso])
        assert "modelo A" in prompt and "gpt" not in prompt.lower() and "claude" not in prompt.lower()
        assert "[Alternativa 1] alt1" in prompt and "(nenhum dispositivo citado)" in prompt
        assert len(VERSAO_DO_FILTRO) == 12


class ServicoDuble:
    modelo_juiz = "duble:juiz"
    modelo_professor = "duble:professor"

    def __init__(self, vereditos_por_pergunta=None, equivalentes=(True, True, True), falhar_reamostragem=False):
        self.vereditos_por_pergunta = vereditos_por_pergunta or {}
        self.equivalentes = list(equivalentes)
        self.falhar = falhar_reamostragem
        self.chamadas_de_reamostragem = 0
        self.casos_julgados: list[CasoDeQualidade] = []

    async def reamostrar(self, pedidos):
        self.chamadas_de_reamostragem += 1
        if self.falhar:
            raise RuntimeError("professor caiu")
        from app.services.filtro.llm_service import RespostaReamostrada
        return [RespostaReamostrada(indice=p.indice, resposta=f"alternativa {self.chamadas_de_reamostragem} para {p.indice}") for p in pedidos]

    async def julgar(self, casos):
        self.casos_julgados.extend(casos)
        saida = []
        for c in casos:
            fabrica = self.vereditos_por_pergunta.get(c.pergunta)
            saida.append(fabrica(c.indice) if fabrica else veredito(c.indice, self.equivalentes[: len(c.reamostragens)]))
        return saida


@pytest.fixture
def repos(tmp_path):
    sintese = SinteseRepository(tmp_path)
    filtro = FiltroRepository(tmp_path)
    return sintese, filtro


def _job_de_sintese(sintese: SinteseRepository, job_id: str, exemplos: list[ExemploSintetico]) -> None:
    job = JobDeSintese(id=job_id, status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeSintese(), criado_em=AGORA, atualizado_em=AGORA)
    sintese.salvar_job(job)
    sintese.acrescentar_exemplos(job_id, exemplos)


class TestEngine:
    async def test_fluxo_completo(self, corpus, repos):
        sintese, filtro = repos
        _job_de_sintese(sintese, "s1", [exemplo("a"), exemplo("b"), exemplo("c", invalidos=["LGPD:art.72"])])
        servico = ServicoDuble()
        engine = FiltroDeQualidade(servico, corpus, filtro, sintese)
        job = await engine.executar(novo_job(F.ParametrosDeFiltro(jobs_de_sintese=["s1"], k_reamostragens=2, tamanho_do_lote=5, teto_usd=0)))

        assert job.status == StatusDoJob.CONCLUIDO
        assert job.exemplos == 3 and job.reprovados_sem_llm == 1 and job.avaliados == 2
        assert job.aprovados == 2 and job.reprovados == 1
        assert job.reprovacoes_por_motivo == {F.MOTIVO_CITACAO_INVALIDA: 1}
        assert servico.chamadas_de_reamostragem == 2  # k=2 rodadas, 1 lote
        decisoes = filtro.decisoes(job.id)
        assert len(decisoes) == 3
        aprovada = next(d for d in decisoes if d.exemplo_id == "a")
        assert aprovada.aprovado and aprovada.score_incerteza == 0.0 and len(aprovada.reamostragens) == 2
        assert aprovada.veredito.nota == 5 and aprovada.versao_do_filtro == VERSAO_DO_FILTRO
        assert aprovada.modelo_juiz == "duble:juiz"
        # O juiz recebeu o rastro do artigo citado e as alternativas.
        caso = next(c for c in servico.casos_julgados if c.pergunta.startswith("Pergunta a"))
        assert caso.contexto[0].id == "LGPD:art.7" and caso.contexto[0].texto.startswith("Art. 7")
        assert caso.reamostragens == ["alternativa 1 para 1", "alternativa 2 para 1"]
        assert filtro.decisoes(job.id, aprovado=False)[0].exemplo_id == "c"

    async def test_incerteza_alta_reprova(self, corpus, repos):
        sintese, filtro = repos
        _job_de_sintese(sintese, "s1", [exemplo("a")])
        engine = FiltroDeQualidade(ServicoDuble(equivalentes=(True, False, False)), corpus, filtro, sintese)
        job = await engine.executar(novo_job(F.ParametrosDeFiltro(jobs_de_sintese=["s1"], k_reamostragens=3, teto_usd=0)))
        decisao = filtro.decisoes(job.id)[0]
        assert not decisao.aprovado and decisao.motivos == [F.MOTIVO_INCERTEZA_ALTA] and decisao.score_incerteza == 0.667

    async def test_k_zero_nao_reamostra(self, corpus, repos):
        sintese, filtro = repos
        _job_de_sintese(sintese, "s1", [exemplo("a")])
        servico = ServicoDuble()
        job = await FiltroDeQualidade(servico, corpus, filtro, sintese).executar(
            novo_job(F.ParametrosDeFiltro(jobs_de_sintese=["s1"], k_reamostragens=0, teto_usd=0))
        )
        assert servico.chamadas_de_reamostragem == 0
        assert filtro.decisoes(job.id)[0].score_incerteza is None and job.aprovados == 1

    async def test_rubrica_reprova_com_motivos(self, corpus, repos):
        sintese, filtro = repos
        _job_de_sintese(sintese, "s1", [exemplo("a")])
        servico = ServicoDuble({exemplo("a").pergunta: lambda i: veredito(i, lastro_nas_citacoes=False, nota=2, problemas="prazo inventado")})
        job = await FiltroDeQualidade(servico, corpus, filtro, sintese).executar(
            novo_job(F.ParametrosDeFiltro(jobs_de_sintese=["s1"], k_reamostragens=1, teto_usd=0))
        )
        d = filtro.decisoes(job.id)[0]
        assert d.motivos == [F.MOTIVO_SEM_LASTRO] and d.veredito.problemas == "prazo inventado"
        assert job.reprovacoes_por_motivo == {F.MOTIVO_SEM_LASTRO: 1}

    async def test_dedup_entre_jobs_de_sintese(self, corpus, repos):
        sintese, filtro = repos
        _job_de_sintese(sintese, "s1", [exemplo("a")])
        _job_de_sintese(sintese, "s2", [exemplo("a", job="s2"), exemplo("b", job="s2")])
        job = await FiltroDeQualidade(ServicoDuble(), corpus, filtro, sintese).executar(
            novo_job(F.ParametrosDeFiltro(jobs_de_sintese=["s1", "s2"], k_reamostragens=0, teto_usd=0))
        )
        assert job.exemplos == 2

    async def test_lote_com_erro(self, corpus, repos):
        sintese, filtro = repos
        _job_de_sintese(sintese, "s1", [exemplo("a")])
        job = await FiltroDeQualidade(ServicoDuble(falhar_reamostragem=True), corpus, filtro, sintese).executar(
            novo_job(F.ParametrosDeFiltro(jobs_de_sintese=["s1"], k_reamostragens=1, teto_usd=0))
        )
        assert job.status == StatusDoJob.FALHOU and job.lotes_com_erro == 1 and filtro.decisoes(job.id) == []


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch, corpus):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        ingestao.registro.definir(corpus)
        sintese = SinteseRepository(tmp_path)
        _job_de_sintese(sintese, "s1", [exemplo("a"), exemplo("b")])
        monkeypatch.setattr(filtro_endpoint, "_sintese", sintese)
        monkeypatch.setattr(filtro_endpoint, "_repositorio", FiltroRepository(tmp_path))
        monkeypatch.setattr(filtro_endpoint, "servico_de_filtro", lambda temperatura: ServicoDuble())
        with TestClient(app) as c:
            yield c
        ingestao.registro.limpar()

    def _esperar(self, client, job_id):
        fim = time.time() + 5
        while time.time() < fim:
            job = client.get(f"/api/v1/filtro/jobs/{job_id}").json()
            if job["status"] not in ("pendente", "executando"):
                return job
            time.sleep(0.05)
        raise AssertionError("job nao terminou")

    def test_cria_e_inspeciona(self, client):
        r = client.post("/api/v1/filtro/jobs", json={"jobs_de_sintese": ["s1"], "k_reamostragens": 1, "teto_usd": 0})
        assert r.status_code == 202, r.text
        job = self._esperar(client, r.json()["id"])
        assert job["status"] == "concluido" and job["aprovados"] == 2
        decisoes = client.get(f"/api/v1/filtro/jobs/{job['id']}/decisoes", params={"aprovado": True}).json()
        assert len(decisoes) == 2 and decisoes[0]["veredito"]["nota"] == 5
        assert client.get("/api/v1/filtro/jobs").json()[0]["id"] == job["id"]

    def test_job_de_sintese_inexistente(self, client):
        assert client.post("/api/v1/filtro/jobs", json={"jobs_de_sintese": ["nao-existe"]}).status_code == 404

    def test_parametros_invalidos(self, client):
        assert client.post("/api/v1/filtro/jobs", json={"jobs_de_sintese": []}).status_code == 422
