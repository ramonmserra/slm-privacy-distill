"""Corpus: chunking por dispositivo, parser de referencias, BM25 e endpoints.
Os testes sobre o texto REAL da LGPD usam o arquivo versionado em
`recursos/fontes/lgpd.txt`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.corpus_repository import CorpusRepository
from app.schemas.corpus import TipoDeDispositivo
from app.services.corpus import ingestao
from app.services.corpus.chunking import chunkar
from app.services.corpus.referencias import artigo_de, artigos_mencionados, normalizar_referencias
from app.services.corpus.retrieval import Documento, IndiceBM25, tokenizar

LGPD_TXT = Path("recursos/fontes/lgpd.txt")
BANCO = Path("recursos/avaliacao/banco_inicial_lgpd.jsonl")

_TRECHO = """
# comentario
CAPÍTULO I DISPOSIÇÕES PRELIMINARES
Art. 1º Esta Lei dispõe sobre o tratamento.
Parágrafo único. As normas gerais são de interesse nacional.
CAPÍTULO II
DO TRATAMENTO
Seção I Dos Requisitos
Art. 7º O tratamento somente poderá ser realizado:
I - mediante consentimento;
II - para o cumprimento de obrigação legal
ou regulatória;
§ 1º O paragrafo um.
§ 2º O paragrafo dois:
I - inciso do paragrafo;
a) alinea do inciso;
Art. 55-A Fica criada a ANPD.
I-A – inciso com sufixo;
""".strip().splitlines()


class TestChunking:
    def test_hierarquia_e_ids(self):
        disp = chunkar(_TRECHO)
        ids = [d.id for d in disp]
        assert ids == [
            "LGPD:art.1", "LGPD:art.1:par.unico",
            "LGPD:art.7", "LGPD:art.7:inc.I", "LGPD:art.7:inc.II",
            "LGPD:art.7:par.1", "LGPD:art.7:par.2", "LGPD:art.7:par.2:inc.I", "LGPD:art.7:par.2:inc.I:ali.a",
            "LGPD:art.55-A", "LGPD:art.55-A:inc.I-A",
        ]

    def test_continuacao_de_linha_e_pais(self):
        por_id = {d.id: d for d in chunkar(_TRECHO)}
        assert por_id["LGPD:art.7:inc.II"].texto == "para o cumprimento de obrigação legal ou regulatória;"
        assert por_id["LGPD:art.7:inc.II"].pai_id == "LGPD:art.7"
        assert por_id["LGPD:art.7:par.2:inc.I:ali.a"].pai_id == "LGPD:art.7:par.2:inc.I"
        assert por_id["LGPD:art.1"].pai_id is None

    def test_capitulo_e_secao(self):
        por_id = {d.id: d for d in chunkar(_TRECHO)}
        assert por_id["LGPD:art.1"].capitulo == "Capítulo I - DISPOSIÇÕES PRELIMINARES"
        assert por_id["LGPD:art.7"].capitulo == "Capítulo II - DO TRATAMENTO"
        assert por_id["LGPD:art.7"].secao == "Seção I - Dos Requisitos"
        assert por_id["LGPD:art.1"].secao == ""

    def test_id_duplicado_e_recusado_pelo_corpus(self):
        linhas = ["Art. 5º Para os fins desta Lei:", "Art. 5 7. (VETADO)."]
        with pytest.raises(ingestao.CorpusInvalido, match="LGPD:art.5"):
            ingestao.Corpus("LGPD", chunkar(linhas), Path("x"))

    def test_tipos(self):
        tipos = {d.id: d.tipo for d in chunkar(_TRECHO)}
        assert tipos["LGPD:art.7"] == TipoDeDispositivo.ARTIGO
        assert tipos["LGPD:art.7:par.1"] == TipoDeDispositivo.PARAGRAFO
        assert tipos["LGPD:art.7:inc.I"] == TipoDeDispositivo.INCISO
        assert tipos["LGPD:art.7:par.2:inc.I:ali.a"] == TipoDeDispositivo.ALINEA


class TestReferencias:
    @pytest.mark.parametrize(
        "texto, esperado",
        [
            ("art. 7º, IX; art. 10", ["LGPD:art.7", "LGPD:art.7:inc.IX", "LGPD:art.10"]),
            ("arts. 33 a 36.", ["LGPD:art.33", "LGPD:art.34", "LGPD:art.35", "LGPD:art.36"]),
            ("arts. 9º, 18 e 23.", ["LGPD:art.9", "LGPD:art.18", "LGPD:art.23"]),
            ("art. 52, II e § 1º.", ["LGPD:art.52", "LGPD:art.52:inc.II", "LGPD:art.52:par.1"]),
            ("art. 7º, §§ 3º e 4º", ["LGPD:art.7", "LGPD:art.7:par.3", "LGPD:art.7:par.4"]),
            ("art. 11, II, “c”", ["LGPD:art.11", "LGPD:art.11:inc.II", "LGPD:art.11:inc.II:ali.c"]),
            ("art. 5º, I, II e III; art. 12.", ["LGPD:art.5", "LGPD:art.5:inc.I", "LGPD:art.5:inc.II", "LGPD:art.5:inc.III", "LGPD:art.12"]),
            ("arts. 6º, 10, 20, 38 e 46, conforme o caso.", ["LGPD:art.6", "LGPD:art.10", "LGPD:art.20", "LGPD:art.38", "LGPD:art.46"]),
            ("art. 1º, parágrafo único", ["LGPD:art.1", "LGPD:art.1:par.unico"]),
            ("LGPD:art.48:par.1", ["LGPD:art.48", "LGPD:art.48:par.1"]),
            ("", []),
            ("Lei nº 13.709, sem artigo", []),
        ],
    )
    def test_normaliza(self, texto, esperado):
        assert normalizar_referencias(texto) == esperado

    def test_segmentos_herdam_o_prefixo_art(self):
        assert normalizar_referencias("arts. 7º, IV; 11, II, “c”; 13 e 16, conforme o caso.") == [
            "LGPD:art.7", "LGPD:art.7:inc.IV",
            "LGPD:art.11", "LGPD:art.11:inc.II", "LGPD:art.11:inc.II:ali.c",
            "LGPD:art.13", "LGPD:art.16",
        ]

    def test_artigos_mencionados_em_prosa(self):
        prosa = "Conforme o art. 18 da LGPD e o artigo 20, o titular pode pedir revisão (art. 18, VI)."
        assert artigos_mencionados(prosa) == ["LGPD:art.18", "LGPD:art.20"]

    def test_artigo_de(self):
        assert artigo_de("LGPD:art.7:inc.IX") == "LGPD:art.7"
        assert artigo_de("LGPD:art.7") == "LGPD:art.7"


class TestBM25:
    def test_tokenizacao_sem_acento_e_stopwords(self):
        assert tokenizar("O tratamento de Dados Pessoais é lícito") == ["tratamento", "dados", "pessoais", "licito"]

    def test_ranking_estavel_e_relevante(self):
        indice = IndiceBM25([
            Documento("a", "Art. 1", "consentimento do titular para o tratamento"),
            Documento("b", "Art. 2", "incidente de segurança comunicado à autoridade nacional"),
            Documento("c", "Art. 3", "consentimento consentimento consentimento específico"),
        ])
        acertos = indice.buscar("consentimento", k=2)
        assert [a.id for a in acertos] == ["c", "a"]
        assert indice.buscar("palavra-inexistente") == []
        assert indice.buscar("") == []


@pytest.fixture(scope="module")
def corpus():
    linhas = LGPD_TXT.read_text(encoding="utf-8").splitlines()
    return ingestao.Corpus("LGPD", chunkar(linhas), LGPD_TXT)


@pytest.mark.skipif(not LGPD_TXT.exists(), reason="texto da LGPD nao baixado")
class TestLgpdReal:

    def test_estrutura_minima(self, corpus):
        resumo = corpus.resumo()
        assert resumo.artigos >= 75
        assert corpus.dispositivo("LGPD:art.5").texto.startswith("Para os fins desta Lei")
        assert corpus.dispositivo("LGPD:art.57").texto == "(VETADO)."
        for id_ in ("LGPD:art.7:inc.IX", "LGPD:art.11:inc.II:ali.c", "LGPD:art.55-A", "LGPD:art.48:par.1", "LGPD:art.65:inc.I-A"):
            assert corpus.existe(id_), id_

    def test_sem_restos_da_limpeza(self):
        texto = LGPD_TXT.read_text(encoding="utf-8")
        assert "Vigência" not in texto
        assert "Redação dada" not in texto
        assert "Brasília" not in texto

    def test_texto_do_artigo_remonta_a_hierarquia(self, corpus):
        texto = corpus.texto_do_artigo("LGPD:art.7:inc.IX")
        assert texto.startswith("Art. 7 O tratamento de dados pessoais")
        assert "IX quando necessário para atender aos interesses legítimos" in texto
        assert "§ 3º" in texto

    def test_referencias_do_banco_inicial_existem_no_corpus(self, corpus):
        for linha in BANCO.read_text(encoding="utf-8").splitlines():
            item = json.loads(linha)
            if not item["referencia"]:
                continue
            ids = normalizar_referencias(item["referencia"])
            assert ids, item["id"]
            ausentes = [i for i in ids if not corpus.existe(i)]
            assert ausentes == [], (item["id"], ausentes)
            artigos = sorted({int(artigo_de(i).split("art.")[1].split("-")[0]) for i in ids})
            assert artigos == sorted(set(item["artigos_principais"])), item["id"]

    def test_busca_encontra_o_artigo_certo(self, corpus):
        assert corpus.buscar("comunicação de incidente de segurança à autoridade nacional", k=1)[0].id == "LGPD:art.48"
        assert corpus.buscar("decisões tomadas unicamente com base em tratamento automatizado", k=1)[0].id == "LGPD:art.20"

    def test_relacionados_excluem_o_proprio(self, corpus):
        relacionados = corpus.relacionados("LGPD:art.7", k=2)
        assert len(relacionados) == 2
        assert all(d.id != "LGPD:art.7" for d in relacionados)


@pytest.mark.skipif(not LGPD_TXT.exists(), reason="texto da LGPD nao baixado")
class TestEndpointsDoCorpus:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        ingestao.registro.limpar()
        with TestClient(app) as c:
            yield c
        ingestao.registro.limpar()

    def test_sem_ingestao_responde_409(self, client):
        assert client.get("/api/v1/corpus").status_code == 409

    def test_ingerir_consultar_buscar(self, client, tmp_path):
        r = client.post("/api/v1/corpus/ingerir", json={"fonte": "LGPD"})
        assert r.status_code == 200, r.text
        assert r.json()["artigos"] >= 75
        assert (tmp_path / "corpus" / "lgpd.jsonl").exists()

        d = client.get("/api/v1/corpus/dispositivos/LGPD:art.7:inc.IX").json()
        assert d["dispositivo"]["pai_id"] == "LGPD:art.7"
        assert d["artigo_completo"].startswith("Art. 7")
        assert client.get("/api/v1/corpus/dispositivos/LGPD:art.999").status_code == 404

        busca = client.get("/api/v1/corpus/buscar", params={"q": "relatório de impacto à proteção de dados", "k": 3}).json()
        assert busca[0]["artigo_id"] in ("LGPD:art.38", "LGPD:art.5")

    def test_fonte_desconhecida(self, client):
        assert client.post("/api/v1/corpus/ingerir", json={"fonte": "CDC"}).status_code == 404

    def test_recarrega_do_disco_apos_restart(self, client):
        client.post("/api/v1/corpus/ingerir", json={"fonte": "LGPD"})
        ingestao.registro.limpar()  # simula outro processo
        assert client.get("/api/v1/corpus").status_code == 200
        assert CorpusRepository().carregar("LGPD")
