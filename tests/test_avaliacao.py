"""Avaliacao: conjuntos, verificacao deterministica, decisao e metricas,
comparacao pareada, engine com dublê e endpoints. Sem rede."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import avaliacao as avaliacao_endpoint
from app.core.erros import OrcamentoExcedido
from app.core.llm import RespostaDeTexto
from app.main import app
from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.avaliacao import (
    CategoriaDeErro,
    Conjunto,
    ItemDeAvaliacao,
    ParametrosDeAvaliacao,
    RespostaDoModelo,
    ResultadoDoItem,
    RunDeAvaliacao,
    VereditoDeAvaliacao,
)
from app.schemas.dataset import ContagemPorTipo, ManifestoDoDataset, ParametrosDeDataset, Split
from app.schemas.sintese import StatusDoJob
from app.services.avaliacao import conjuntos, metricas, pareado
from app.services.avaliacao.engine import Avaliador, novo_run, resolver_modelo, versao_da_medida
from app.services.avaliacao.llm_service import VERSAO_DO_JUIZ, CasoDeAvaliacao, _build_juiz_prompt, _prompt_do_usuario
from app.services.corpus import ingestao
from app.services.corpus.chunking import chunkar
from app.services.sintese.llm_service import TrechoDeContexto

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)
LGPD_TXT = Path("recursos/fontes/lgpd.txt")

_LEI = """
Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses:
I - mediante o fornecimento de consentimento pelo titular;
IX - quando necessário para atender aos interesses legítimos do controlador;
Art. 18. O titular dos dados pessoais tem direito a obter do controlador:
VI - eliminação dos dados pessoais tratados com o consentimento do titular;
Art. 20. O titular tem direito a solicitar a revisão de decisões tomadas unicamente com base em tratamento automatizado.
Art. 48. O controlador deverá comunicar à autoridade nacional a ocorrência de incidente de segurança.
""".strip().splitlines()


@pytest.fixture
def corpus():
    return ingestao.Corpus("LGPD", chunkar(_LEI), Path("fake.txt"))


def item(id_: str, artigos=("LGPD:art.7",), abstencao=False, tipo="conceito", conjunto=Conjunto.HELD_OUT) -> ItemDeAvaliacao:
    return ItemDeAvaliacao(id=id_, conjunto=conjunto, pergunta=f"Pergunta {id_}?", referencia="Referencia.",
                           artigos_referencia=list(artigos), tipo=tipo, abstencao_esperada=abstencao)


def veredito(indice=1, correto=True, grounded=True, abstencao=False, categoria=None, criterios=(0, 0)) -> VereditoDeAvaliacao:
    return VereditoDeAvaliacao(indice=indice, correto=correto, criterios_atendidos=criterios[0], criterios_total=criterios[1],
                               grounded=grounded, abstencao=abstencao, categoria=categoria, similaridade=0.9 if correto else 0.2)


def resultado(id_: str, *, acerto=True, grounded=True, inexistentes=(), abstencao_esperada=False, abstencao=False, tipo="conceito", erro=None, latencia=100) -> ResultadoDoItem:
    r = ResultadoDoItem(
        item=item(id_, abstencao=abstencao_esperada, tipo=tipo),
        resposta=RespostaDoModelo(item_id=id_, texto="x", latencia_ms=latencia, input_tokens=10, output_tokens=5, erro=erro,
                                  artigos_inexistentes=list(inexistentes), citou_referencia=True),
        veredito=None if erro else veredito(correto=acerto, grounded=grounded, abstencao=abstencao,
                                            categoria=None if acerto else CategoriaDeErro.INCOMPLETE_ANSWER),
    )
    return metricas.decidir(r)


class TestConjuntos:
    @pytest.mark.skipif(not conjuntos.BANCO_INICIAL.exists(), reason="banco inicial ausente")
    def test_banco_inicial(self):
        itens = conjuntos.carregar_banco_inicial()
        assert len(itens) == 34
        q52 = next(i for i in itens if i.id == "Q12.2")
        assert q52.artigos_referencia == ["LGPD:art.52"] and q52.conjunto == Conjunto.BANCO_INICIAL
        assert next(i for i in itens if i.id == "TC").abstencao_esperada is True
        assert next(i for i in itens if i.id == "TC").artigos_referencia == []
        assert next(i for i in itens if i.id == "Q7.1").artigos_referencia == ["LGPD:art.5", "LGPD:art.37", "LGPD:art.38", "LGPD:art.39", "LGPD:art.40", "LGPD:art.41"]

    def test_held_out(self, tmp_path):
        repo = DatasetRepository(tmp_path)
        manifesto = ManifestoDoDataset(versao="v1", nome="v1", criado_em=AGORA, parametros=ParametrosDeDataset(nome="v1", jobs_de_filtro=["f"]),
                                       treino=ContagemPorTipo(), held_out=ContagemPorTipo(total=1), versao_do_prompt_do_aluno="p")
        linha = {"id": "h1", "tipo": "sem_resposta", "dificuldade": "basica", "artigos_citados": ["LGPD:art.48:par.1"],
                 "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "P?"},
                              {"role": "assistant", "content": "Raciocinio:\nR\n\nResposta:\nA lei nao fixa prazo."}]}
        repo.salvar(manifesto, {Split.TREINO: [], Split.HELD_OUT: [linha]})
        itens = conjuntos.carregar_held_out("v1", repo)
        assert itens[0].referencia == "A lei nao fixa prazo." and itens[0].artigos_referencia == ["LGPD:art.48"]
        assert itens[0].abstencao_esperada is True
        with pytest.raises(conjuntos.ConjuntoIndisponivel):
            conjuntos.carregar_held_out("nao", repo)

    def test_versao_do_conjunto(self):
        assert conjuntos.versao_do_conjunto(Conjunto.HELD_OUT, "abc") == "held_out:abc"
        if conjuntos.BANCO_INICIAL.exists():
            assert conjuntos.versao_do_conjunto(Conjunto.BANCO_INICIAL, None).startswith("banco_inicial:")


class TestVerificacaoEDecisao:
    def test_verificar_citacoes(self, corpus):
        avaliador = Avaliador(None, corpus, AvaliacaoRepository(Path(".")), DatasetRepository(Path(".")))
        r = avaliador.verificar(item("a", artigos=["LGPD:art.7"]), RespostaDoModelo(item_id="a", texto="Pelo art. 7º e pelo art. 72 da LGPD.", latencia_ms=1))
        assert r.artigos_mencionados == ["LGPD:art.7", "LGPD:art.72"]
        assert r.artigos_inexistentes == ["LGPD:art.72"] and r.citou_referencia is True
        r2 = avaliador.verificar(item("a", artigos=["LGPD:art.18"]), RespostaDoModelo(item_id="a", texto="Sem artigo.", latencia_ms=1))
        assert r2.citou_referencia is False

    def test_decidir(self):
        assert resultado("a").acerto and not resultado("a").alucinacao
        assert resultado("b", grounded=False).alucinacao
        assert resultado("c", inexistentes=["LGPD:art.72"]).alucinacao
        assert resultado("d", abstencao_esperada=True, abstencao=True).abstencao_adequada is True
        assert resultado("e", abstencao_esperada=True, abstencao=False, acerto=False).abstencao_adequada is False
        assert resultado("f").abstencao_adequada is None

    def test_prompt_do_juiz_e_cego(self):
        caso = CasoDeAvaliacao(indice=1, item=item("a", conjunto=Conjunto.BANCO_INICIAL, abstencao=True), resposta="R", rastro=[])
        prompt = _build_juiz_prompt([caso])
        assert "MODELO A" in prompt and "CRITERIOS ESPERADOS" in prompt and "Abstencao esperada: SIM" in prompt
        assert "(nenhum dispositivo de referencia)" in prompt
        assert len(VERSAO_DO_JUIZ) == 12

    def test_prompt_do_usuario_com_e_sem_rag(self):
        i = item("a")
        assert _prompt_do_usuario(i, []) == "Pergunta a?"
        com = _prompt_do_usuario(i, [TrechoDeContexto(id="LGPD:art.7", texto="Art. 7 ...")])
        assert com.startswith("Dispositivos legais") and com.endswith("Pergunta: Pergunta a?")


class TestMetricas:
    def test_calcular(self):
        rs = [resultado("a"), resultado("b", acerto=False), resultado("c", grounded=False), resultado("d", erro="caiu", acerto=False),
              resultado("e", abstencao_esperada=True, abstencao=True, tipo="sem_resposta")]
        m = metricas.calcular(rs, custo_usd=0.02)
        assert m.itens == 5 and m.respondidos == 4 and m.julgados == 4
        assert m.acertos == 3 and m.acuracia == 0.75 and m.acuracia_ic95[0] < 0.75 < m.acuracia_ic95[1]
        assert m.alucinacoes == 1 and m.taxa_de_alucinacao == 0.25
        assert m.categorias_de_erro == {"incomplete_answer": 1}
        assert m.abstencoes_esperadas == 1 and m.abstencoes_adequadas == 1
        assert m.por_tipo["sem_resposta"]["acuracia"] == 1.0 and m.por_tipo["conceito"]["itens"] == 3
        assert m.latencia_p50_ms == 100.0 and m.tokens_entrada == 40 and m.custo_usd_por_item == 0.005
        assert m.taxa_citou_referencia == 1.0

    def test_vazio(self):
        m = metricas.calcular([], 0.0)
        assert m.acuracia is None and m.acuracia_ic95 is None and m.custo_usd_por_item is None


def _run(id_: str, modelo: str, versao="v") -> RunDeAvaliacao:
    return RunDeAvaliacao(id=id_, status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeAvaliacao(modelo=modelo), modelo=modelo,
                          versao_da_medida=versao, criado_em=AGORA, atualizado_em=AGORA)


class TestPareado:
    def test_comparacao(self):
        a = [resultado("1"), resultado("2"), resultado("3", acerto=False), resultado("4", acerto=False), resultado("5", acerto=False)]
        b = [resultado("1"), resultado("2", acerto=False), resultado("3"), resultado("4"), resultado("5"), resultado("6")]
        c = pareado.comparar(_run("A", "local:x"), a, _run("B", "local:y"), b)
        assert c.itens_em_comum == 5 and c.acertos_a == 2 and c.acertos_b == 4
        assert c.so_a_acertou == 1 and c.so_b_acertou == 3 and c.melhoraram == ["3", "4", "5"] and c.pioraram == ["2"]
        assert c.delta_acuracia == 0.4 and 0 < c.p_mcnemar <= 1 and c.mesma_versao_da_medida
        assert "p.p." in c.leitura

    def test_versoes_diferentes_sao_indicativas(self):
        c = pareado.comparar(_run("A", "x", "v1"), [resultado("1")], _run("B", "y", "v2"), [resultado("1")])
        assert not c.mesma_versao_da_medida and "INDICATIVA" in c.leitura

    def test_sem_itens_em_comum(self):
        c = pareado.comparar(_run("A", "x"), [resultado("1")], _run("B", "y"), [resultado("2")])
        assert c.itens_em_comum == 0 and "Sem itens em comum" in c.leitura


class TestResolucao:
    def test_resolver_modelo(self):
        assert resolver_modelo("teacher") == "openai:gpt-5.6-luna"
        assert resolver_modelo("local:qwen3-1.7b-lgpd-abc") == "local:qwen3-1.7b-lgpd-abc"
        from app.core.llm import UnknownProviderError
        with pytest.raises(UnknownProviderError):
            resolver_modelo("qwen3:1.7b")

    @pytest.mark.skipif(not conjuntos.BANCO_INICIAL.exists(), reason="banco inicial ausente")
    def test_versao_da_medida_muda_com_rag_e_conjunto(self):
        base = versao_da_medida(ParametrosDeAvaliacao(modelo="student"))
        assert base == versao_da_medida(ParametrosDeAvaliacao(modelo="teacher"))  # o modelo medido NAO entra
        assert base != versao_da_medida(ParametrosDeAvaliacao(modelo="student", com_rag=True))
        assert base != versao_da_medida(ParametrosDeAvaliacao(modelo="student", conjunto=Conjunto.HELD_OUT, dataset_versao="v"))


class ServicoDuble:
    modelo_juiz = "duble:juiz"

    def __init__(self, respostas=None, falhar_item: str | None = None, estourar_no_item: str | None = None, juiz_falha=False):
        self.respostas = respostas or {}
        self.falhar_item = falhar_item
        self.estourar = estourar_no_item
        self.juiz_falha = juiz_falha
        self.contextos: dict[str, list] = {}
        self.casos: list[CasoDeAvaliacao] = []

    async def responder(self, item, contexto_rag):
        self.contextos[item.id] = contexto_rag
        if item.id == self.falhar_item:
            raise RuntimeError("provedor caiu")
        if item.id == self.estourar:
            raise OrcamentoExcedido("teto")
        return RespostaDeTexto(texto=self.respostas.get(item.id, f"Resposta a {item.id} pelo art. 7º."), input_tokens=50, output_tokens=20)

    async def julgar(self, casos):
        self.casos.extend(casos)
        if self.juiz_falha:
            raise RuntimeError("juiz caiu")
        saida = []
        for c in casos:
            errado = "errada" in c.resposta
            saida.append(veredito(c.indice, correto=not errado, grounded="art. 72" not in c.resposta,
                                  categoria=CategoriaDeErro.WRONG_LEGAL_BASIS if errado else None))
        return saida


@pytest.fixture
def ambiente(tmp_path, corpus):
    datasets = DatasetRepository(tmp_path)
    manifesto = ManifestoDoDataset(versao="v1", nome="v1", criado_em=AGORA, parametros=ParametrosDeDataset(nome="v1", jobs_de_filtro=["f"]),
                                   treino=ContagemPorTipo(), held_out=ContagemPorTipo(total=4), versao_do_prompt_do_aluno="p")
    linhas = []
    for i in range(4):
        linhas.append({"id": f"h{i}", "tipo": "conceito" if i % 2 else "incidente", "dificuldade": "basica", "artigos_citados": ["LGPD:art.7:inc.IX"],
                       "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": f"Pergunta {i} sobre dados?"},
                                    {"role": "assistant", "content": f"Resposta {i}."}]})
    datasets.salvar(manifesto, {Split.TREINO: [], Split.HELD_OUT: linhas})
    return corpus, AvaliacaoRepository(tmp_path), datasets


class TestEngine:
    async def test_fluxo_completo_no_held_out(self, ambiente):
        corpus, repo, datasets = ambiente
        servico = ServicoDuble(respostas={"h1": "Resposta errada pelo art. 72."})
        run = novo_run(ParametrosDeAvaliacao(modelo="local:aluno-v1", conjunto=Conjunto.HELD_OUT, dataset_versao="v1", teto_usd=0))
        run = await Avaliador(servico, corpus, repo, datasets).executar(run)

        assert run.status == StatusDoJob.CONCLUIDO and run.itens == 4 and run.respondidos == 4
        assert run.modelo == "local:aluno-v1" and run.modelo_juiz == "duble:juiz"
        m = run.metricas
        assert m.acertos == 3 and m.acuracia == 0.75 and m.alucinacoes == 1 and m.itens_com_artigo_inexistente == 1
        assert m.categorias_de_erro == {"wrong_legal_basis": 1}
        assert m.taxa_citou_referencia == 0.75 and m.tokens_entrada == 200
        itens = repo.itens(run.id)
        assert len(itens) == 4 and [r.item.id for r in itens] == ["h0", "h1", "h2", "h3"]
        errado = next(r for r in itens if r.item.id == "h1")
        assert not errado.acerto and errado.alucinacao and errado.resposta.artigos_inexistentes == ["LGPD:art.72"]
        # O juiz recebeu o rastro do artigo de referencia e a referencia sem o prefixo de raciocinio.
        caso = next(c for c in servico.casos if c.item.id == "h0")
        assert caso.rastro[0].id == "LGPD:art.7" and caso.item.referencia == "Resposta 0."
        assert servico.contextos["h0"] == []  # sem RAG
        assert repo.carregar_run(run.id).metricas.acuracia == 0.75

    async def test_com_rag_entrega_artigos_recuperados(self, ambiente):
        corpus, repo, datasets = ambiente
        servico = ServicoDuble()
        run = novo_run(ParametrosDeAvaliacao(modelo="student", conjunto=Conjunto.HELD_OUT, dataset_versao="v1", com_rag=True, k_rag=2, limite=1, teto_usd=0))
        await Avaliador(servico, corpus, repo, datasets).executar(run)
        assert len(servico.contextos["h0"]) == 2 and all(t.texto.startswith("Art.") for t in servico.contextos["h0"])

    async def test_item_com_erro_fica_fora_do_denominador(self, ambiente):
        corpus, repo, datasets = ambiente
        run = novo_run(ParametrosDeAvaliacao(modelo="student", conjunto=Conjunto.HELD_OUT, dataset_versao="v1", teto_usd=0))
        run = await Avaliador(ServicoDuble(falhar_item="h2"), corpus, repo, datasets).executar(run)
        assert run.status == StatusDoJob.CONCLUIDO and run.metricas.respondidos == 3 and run.metricas.julgados == 3
        assert next(r for r in repo.itens(run.id) if r.item.id == "h2").resposta.erro.startswith("RuntimeError")

    async def test_orcamento_excedido_para_antes_do_juiz(self, ambiente):
        corpus, repo, datasets = ambiente
        servico = ServicoDuble(estourar_no_item="h1")
        run = novo_run(ParametrosDeAvaliacao(modelo="student", conjunto=Conjunto.HELD_OUT, dataset_versao="v1", concorrencia=1, teto_usd=0))
        run = await Avaliador(servico, corpus, repo, datasets).executar(run)
        assert run.status == StatusDoJob.ORCAMENTO_EXCEDIDO and servico.casos == [] and run.metricas.julgados == 0

    async def test_juiz_falhando_marca_falhou(self, ambiente):
        corpus, repo, datasets = ambiente
        run = novo_run(ParametrosDeAvaliacao(modelo="student", conjunto=Conjunto.HELD_OUT, dataset_versao="v1", teto_usd=0))
        run = await Avaliador(ServicoDuble(juiz_falha=True), corpus, repo, datasets).executar(run)
        assert run.status == StatusDoJob.FALHOU and run.metricas.respondidos == 4 and run.metricas.julgados == 0

    async def test_held_out_sem_dataset(self, ambiente):
        corpus, repo, datasets = ambiente
        run = novo_run(ParametrosDeAvaliacao(modelo="student", conjunto=Conjunto.HELD_OUT, dataset_versao="nao", teto_usd=0))
        run = await Avaliador(ServicoDuble(), corpus, repo, datasets).executar(run)
        assert run.status == StatusDoJob.FALHOU and "nao existe" in run.erro


class TestEndpoints:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch, ambiente):
        corpus, repo, datasets = ambiente
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        ingestao.registro.definir(corpus)
        monkeypatch.setattr(avaliacao_endpoint, "_repositorio", repo)
        monkeypatch.setattr(avaliacao_endpoint, "_datasets", datasets)
        monkeypatch.setattr(avaliacao_endpoint, "servico_de_avaliacao", lambda modelo, max_tokens: ServicoDuble(respostas={"h3": "errada"}))
        with TestClient(app) as c:
            yield c
        ingestao.registro.limpar()

    def _esperar(self, client, run_id):
        fim = time.time() + 5
        while time.time() < fim:
            run = client.get(f"/api/v1/avaliacao/runs/{run_id}").json()
            if run["status"] not in ("pendente", "executando"):
                return run
            time.sleep(0.05)
        raise AssertionError("run nao terminou")

    def test_cria_acompanha_inspeciona_e_compara(self, client):
        corpo = {"modelo": "local:aluno-v1", "conjunto": "held_out", "dataset_versao": "v1", "teto_usd": 0}
        a = client.post("/api/v1/avaliacao/runs", json=corpo)
        assert a.status_code == 202, a.text
        run_a = self._esperar(client, a.json()["id"])
        assert run_a["status"] == "concluido" and run_a["metricas"]["acertos"] == 3
        erros = client.get(f"/api/v1/avaliacao/runs/{run_a['id']}/itens", params={"so_erros": True}).json()
        assert [e["item"]["id"] for e in erros] == ["h3"]

        b = client.post("/api/v1/avaliacao/runs", json={**corpo, "modelo": "teacher"})
        run_b = self._esperar(client, b.json()["id"])
        comparacao = client.get("/api/v1/avaliacao/comparar", params={"a": run_a["id"], "b": run_b["id"]}).json()
        assert comparacao["itens_em_comum"] == 4 and comparacao["mesma_versao_da_medida"] is True
        assert comparacao["modelo_b"] == "openai:gpt-5.6-luna"
        assert client.get("/api/v1/avaliacao/runs").json()[0]["id"] in (run_a["id"], run_b["id"])

    def test_erros_de_entrada(self, client):
        assert client.post("/api/v1/avaliacao/runs", json={"modelo": "qwen3:1.7b"}).status_code == 422
        assert client.post("/api/v1/avaliacao/runs", json={"modelo": "student", "conjunto": "held_out", "dataset_versao": "nao"}).status_code == 404
        assert client.get("/api/v1/avaliacao/runs/nao").status_code == 404
        assert client.get("/api/v1/avaliacao/comparar", params={"a": "x", "b": "y"}).status_code == 404
