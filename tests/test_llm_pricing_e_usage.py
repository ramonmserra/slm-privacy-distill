"""Precos e telemetria de consumo: o custo que o TCC reporta sai daqui."""

from __future__ import annotations

from pathlib import Path

from app.core import llm_pricing
from app.core.llm_usage import LlmUsageTracker, etapa_scope, job_scope, record_langchain_usage, tracker
from app.schemas.comum import Etapa


class TestPricing:
    def test_luna_por_contador(self):
        custo = llm_pricing.estimate_cost_usd(
            "openai:gpt-5.6-luna", input_tokens=1_000_000, output_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000,
        )
        assert custo == 0.20 + 1.20 + 0.25 + 0.02

    def test_anthropic_deriva_cache_dos_multiplicadores(self):
        custo = llm_pricing.estimate_cost_usd(
            "anthropic:claude-haiku-4-5", cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000
        )
        assert custo == 1.0 * 1.25 + 1.0 * 0.1

    def test_local_custa_zero_e_e_conferido(self):
        assert llm_pricing.estimate_cost_usd("local:qualquer-coisa", input_tokens=10_000) == 0.0
        assert llm_pricing.is_verified("local:qualquer-coisa")

    def test_desconhecido_devolve_none_nunca_zero(self):
        assert llm_pricing.estimate_cost_usd("openai:modelo-inexistente", input_tokens=10) is None
        assert not llm_pricing.is_verified("openai:modelo-inexistente")

    def test_override_por_ambiente(self, monkeypatch):
        monkeypatch.setenv("LLM_PRICING_JSON", '{"openai:novo": {"input": 2.0, "output": 4.0}}')
        assert llm_pricing.estimate_cost_usd("openai:novo", input_tokens=1_000_000) == 2.0

    def test_override_malformado_e_ignorado(self, monkeypatch):
        monkeypatch.setenv("LLM_PRICING_JSON", "{nao e json")
        assert llm_pricing.pricing_for("openai:gpt-5.6-luna") is not None


class _Msg:
    def __init__(self, **uso):
        self.usage_metadata = uso


class TestTracker:
    def test_agrega_por_etapa_e_por_job(self):
        with etapa_scope(Etapa.SINTESE), job_scope("job-1", teto_usd=0):
            record_langchain_usage("openai:gpt-5.6-luna", _Msg(input_tokens=1000, output_tokens=100))
            record_langchain_usage("openai:gpt-5.6-luna", _Msg(input_tokens=1000, output_tokens=100))
        with etapa_scope(Etapa.AVALIACAO):
            record_langchain_usage("local:qwen3:1.7b", _Msg(input_tokens=500, output_tokens=50))

        snap = tracker.snapshot()
        assert snap["por_etapa"]["sintese"]["subtotal"]["chamadas"] == 2
        assert snap["por_etapa"]["avaliacao"]["modelos"]["local:qwen3:1.7b"]["input_tokens"] == 500
        assert snap["por_job"]["job-1"]["chamadas"] == 2
        assert tracker.custo_do_job("job-1") > 0
        assert snap["total"]["chamadas"] == 3

    def test_sem_escopo_cai_em_diagnostico(self):
        record_langchain_usage("openai:gpt-5.6-luna", _Msg(input_tokens=1))
        assert "diagnostico" in tracker.snapshot()["por_etapa"]

    def test_cache_do_provedor_e_lido_dos_detalhes(self):
        record_langchain_usage(
            "openai:gpt-5.6-luna",
            _Msg(input_tokens=100, output_tokens=1, input_token_details={"cache_read": 60, "cache_creation": 40}),
        )
        consumo = tracker.snapshot()["por_etapa"]["diagnostico"]["modelos"]["openai:gpt-5.6-luna"]
        assert consumo["cache_read_input_tokens"] == 60
        assert consumo["cache_creation_input_tokens"] == 40

    def test_modelo_sem_preco_e_denunciado(self):
        record_langchain_usage("openai:sem-preco", _Msg(input_tokens=10))
        snap = tracker.snapshot()
        assert snap["modelos_sem_preco"] == ["openai:sem-preco"]
        assert snap["total"]["chamadas_sem_preco"] == 1

    def test_preco_nao_conferido_e_listado(self):
        record_langchain_usage("openai:gpt-5-nano", _Msg(input_tokens=10))
        assert tracker.snapshot()["modelos_com_preco_nao_conferido"] == ["openai:gpt-5-nano"]

    def test_persiste_e_recarrega(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        primeiro = LlmUsageTracker()
        primeiro.record("openai:gpt-5.6-luna", input_tokens=7, etapa=Etapa.FILTRO, job_id="j")
        assert Path(tmp_path, "llm_usage.json").exists()

        segundo = LlmUsageTracker()
        assert segundo.snapshot()["por_etapa"]["filtro"]["subtotal"]["input_tokens"] == 7
        assert segundo.custo_do_job("j") == primeiro.custo_do_job("j")

    def test_mensagem_sem_usage_conta_a_chamada(self):
        class SemUso:
            pass

        record_langchain_usage("openai:gpt-5.6-luna", SemUso())
        assert tracker.snapshot()["total"]["chamadas"] == 1
