"""Portao do LLM: orcamento por job e disjuntor por provedor."""

from __future__ import annotations

import pytest

from app.core import llm_gate
from app.core.erros import OrcamentoExcedido, SaturacaoDoProvedor
from app.core.llm_usage import job_scope, tracker

_MSG = [{"role": "user", "content": "x" * 4000}]  # ~1000 tokens


class _Erro429(Exception):
    status_code = 429


class _Erro400(Exception):
    status_code = 400


class TestOrcamento:
    async def test_sem_job_passa(self):
        async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
            pass

    async def test_teto_zero_desliga(self):
        with job_scope("j", teto_usd=0):
            async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
                pass

    async def test_recusa_antes_de_chamar_quando_estouraria(self):
        with job_scope("j", teto_usd=0.001):
            tracker.record("openai:gpt-5.6-luna", input_tokens=5000, job_id="j")  # 0.001 USD
            with pytest.raises(OrcamentoExcedido, match="teto de US\\$ 0.00"):
                async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
                    raise AssertionError("a chamada nao deveria ter saido")

    async def test_teto_do_ambiente_quando_nao_informado(self, monkeypatch):
        monkeypatch.setenv("LLM_BUDGET_USD_POR_JOB", "0.0001")
        with job_scope("j") as job:
            assert job.teto_usd == 0.0001
            with pytest.raises(OrcamentoExcedido):
                async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
                    pass

    def test_estimativa_de_entrada(self):
        # 4000 chars / 4 = 1000 tokens * 0.20/M
        assert llm_gate.estimar_custo_de_entrada("openai:gpt-5.6-luna", _MSG) == pytest.approx(0.0002)
        assert llm_gate.estimar_custo_de_entrada("local:qwen3:1.7b", _MSG) == 0.0
        assert llm_gate.estimar_custo_de_entrada("openai:sem-preco", _MSG) == 0.0

    def test_estimativa_le_blocos_de_conteudo(self):
        mensagens = [{"role": "system", "content": [{"type": "text", "text": "a" * 400}]}]
        assert llm_gate.estimar_custo_de_entrada("openai:gpt-5.6-luna", mensagens) == pytest.approx(0.00002)


class TestDisjuntor:
    async def _falhar(self, exc):
        with pytest.raises(type(exc)):
            async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
                raise exc

    async def test_abre_apos_tres_saturacoes_seguidas(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUIT_FAILURES", "3")
        for _ in range(3):
            await self._falhar(_Erro429())
        with pytest.raises(SaturacaoDoProvedor):
            async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
                raise AssertionError("nao deveria sair para a rede")

    async def test_erro_permanente_nao_abre(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUIT_FAILURES", "2")
        for _ in range(5):
            await self._falhar(_Erro400())
        async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
            pass

    async def test_sucesso_zera_a_sequencia(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUIT_FAILURES", "2")
        await self._falhar(_Erro429())
        async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
            pass
        await self._falhar(_Erro429())
        async with llm_gate.portao_do_llm("openai:gpt-5.6-luna", messages=_MSG):
            pass

    async def test_disjuntor_e_por_provedor(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUIT_FAILURES", "1")
        await self._falhar(_Erro429())
        async with llm_gate.portao_do_llm("anthropic:claude-sonnet-5", messages=_MSG):
            pass

    def test_deteccao_por_forma(self):
        class Resp:
            status_code = 503

        class ComResposta(Exception):
            response = Resp()

        class RateLimitError(Exception):
            pass

        assert llm_gate.e_saturacao(ComResposta())
        assert llm_gate.e_saturacao(RateLimitError())
        assert not llm_gate.e_saturacao(_Erro400())
        assert not llm_gate.e_saturacao(ValueError("x"))
