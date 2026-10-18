"""`app.core.llm`: resolucao de papel -> provedor -> modelo, montagem do chat
model por provedor e o laco de saida estruturada. Sem rede: `init_chat_model`
e' capturado, e o chat model e' um dublê."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.core import llm
from app.core.erros import StructuredOutputError
from app.core.llm import ModelTier
from app.core.llm_usage import tracker


class TestResolucaoDePapel:
    def test_defaults_dos_tres_papeis(self):
        assert llm.resolve_llm_id(ModelTier.TEACHER) == "openai:gpt-5.6-luna"
        assert llm.resolve_llm_id(ModelTier.JUDGE) == "anthropic:claude-sonnet-5"
        assert llm.resolve_llm_id(ModelTier.STUDENT) == "local:qwen3:1.7b"

    def test_provedor_do_papel_vem_do_ambiente(self, monkeypatch):
        monkeypatch.setenv("JUDGE_PROVIDER", "openai")
        assert llm.resolve_llm_id(ModelTier.JUDGE) == "openai:gpt-5.6-luna"

    def test_modelo_do_papel_vem_do_ambiente(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL_TEACHER", "gpt-5-mini")
        assert llm.resolve_llm_id(ModelTier.TEACHER) == "openai:gpt-5-mini"
        assert llm.tier_model_source(ModelTier.TEACHER) == "env"
        assert llm.tier_model_source(ModelTier.JUDGE) == "default"

    def test_provedor_desconhecido_cai_no_default_com_aviso(self, monkeypatch, caplog):
        monkeypatch.setenv("TEACHER_PROVIDER", "openia")
        with caplog.at_level("WARNING"):
            assert llm.provider_for_tier(ModelTier.TEACHER) == "openai"
        assert "nao e' um provedor conhecido" in caplog.text

    def test_override_qualificado_vence_o_papel(self):
        assert llm.resolve_llm_id(ModelTier.STUDENT, "local:qwen3-1.7b-lgpd-v2") == "local:qwen3-1.7b-lgpd-v2"

    def test_override_sem_provedor_e_recusado(self):
        with pytest.raises(llm.UnknownProviderError):
            llm.resolve_llm_id(ModelTier.STUDENT, "qwen3:1.7b")

    def test_tag_do_ollama_com_dois_pontos_e_preservada(self):
        assert llm.split_llm_id("local:qwen3:1.7b") == ("local", "qwen3:1.7b")

    def test_variavel_de_modelo_por_provedor_e_papel(self):
        assert llm.model_env_var("openai", ModelTier.TEACHER) == "OPENAI_MODEL_TEACHER"


class TestCredenciais:
    def test_local_nao_exige_credencial(self):
        assert llm.llm_available("local")
        assert llm.credencial("local") == "ollama"

    def test_openai_sem_chave(self, monkeypatch):
        assert not llm.llm_available("openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        assert llm.llm_available("openai")

    def test_aviso_quando_professor_e_juiz_sao_da_mesma_familia(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
        assert llm.avisos_de_configuracao() == []
        monkeypatch.setenv("JUDGE_PROVIDER", "openai")
        avisos = llm.avisos_de_configuracao()
        assert len(avisos) == 1 and "MESMO provedor" in avisos[0]

    def test_aviso_de_credencial_ausente(self):
        avisos = llm.avisos_de_configuracao()
        assert any("teacher" in a and "OPENAI_API_KEY" in a for a in avisos)
        assert any("judge" in a and "ANTHROPIC_API_KEY" in a for a in avisos)


@pytest.fixture
def init_capturado(monkeypatch):
    chamadas: list[dict] = []

    def fake_init(model, *, model_provider, **kwargs):
        chamadas.append({"model": model, "model_provider": model_provider, **kwargs})
        return object()

    import langchain.chat_models as mod

    monkeypatch.setattr(mod, "init_chat_model", fake_init)
    return chamadas


class TestBuildChatModel:
    def test_luna_usa_responses_api_e_reserva_raciocinio(self, init_capturado, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        llm.build_chat_model("openai:gpt-5.6-luna", max_tokens=1000, temperature=0.0)
        c = init_capturado[0]
        assert c["model_provider"] == "openai" and c["model"] == "gpt-5.6-luna"
        assert c["use_responses_api"] is True
        assert c["max_tokens"] == 1000 + 8192
        assert c["temperature"] == 0.0
        assert c["api_key"] == "sk-x"
        assert "prompt_cache_options" not in c

    def test_luna_sem_cache_de_prompt(self, init_capturado, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        llm.build_chat_model("openai:gpt-5.6-luna", max_tokens=10, sem_cache_de_prompt=True)
        assert init_capturado[0]["prompt_cache_options"] == {"mode": "explicit"}

    def test_local_fala_openai_com_base_url(self, init_capturado, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://servidor:8000/v1")
        llm.build_chat_model("local:qwen3:1.7b", max_tokens=512)
        c = init_capturado[0]
        assert c["model_provider"] == "openai"
        assert c["model"] == "qwen3:1.7b"
        assert c["base_url"] == "http://servidor:8000/v1"
        assert c["api_key"] == "ollama"
        assert c["max_tokens"] == 512

    def test_sonnet_descarta_temperature(self, init_capturado, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
        llm.build_chat_model("anthropic:claude-sonnet-5", max_tokens=10, temperature=0.0)
        assert "temperature" not in init_capturado[0]

    def test_kwarg_none_e_omitido(self, init_capturado, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        llm.build_chat_model("openai:gpt-5-nano", max_tokens=10, temperature=None)
        assert "temperature" not in init_capturado[0]

    def test_timeout_e_retries_explicitos(self, init_capturado, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("LLM_TIMEOUT_S", "33")
        monkeypatch.setenv("LLM_MAX_RETRIES", "1")
        llm.build_chat_model("openai:gpt-5-nano", max_tokens=10)
        assert init_capturado[0]["timeout"] == 33.0
        assert init_capturado[0]["max_retries"] == 1

    def test_sem_credencial_falha_cedo_dizendo_a_variavel(self, init_capturado):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm.build_chat_model("openai:gpt-5-nano", max_tokens=10)
        assert init_capturado == []


class TestMensagens:
    def test_system_cacheavel_so_na_anthropic(self):
        a = llm.system_message("x", "anthropic:claude-sonnet-5", cacheable=True)
        assert a["content"][0]["cache_control"] == {"type": "ephemeral"}
        o = llm.system_message("x", "openai:gpt-5.6-luna", cacheable=True)
        assert o == {"role": "system", "content": "x"}

    def test_prefixo_cacheavel(self):
        o = llm.user_message("pergunta", "openai:gpt-5.6-luna", cacheable_prefix="ARTIGO\n")
        assert o["content"] == "ARTIGO\npergunta"
        a = llm.user_message("pergunta", "anthropic:claude-sonnet-5", cacheable_prefix="ARTIGO\n")
        assert [b["text"] for b in a["content"]] == ["ARTIGO\n", "pergunta"]


class Saida(BaseModel):
    valor: int


class _Mensagem:
    def __init__(self, usage=None, meta=None):
        self.usage_metadata = usage or {"input_tokens": 100, "output_tokens": 20}
        self.response_metadata = meta or {}
        self.content = "ok"


class _Runnable:
    def __init__(self, resultados):
        self._resultados = list(resultados)
        self.chamadas = 0

    async def ainvoke(self, _messages):
        self.chamadas += 1
        return self._resultados.pop(0)


class _ChatModel:
    def __init__(self, runnable):
        self._runnable = runnable

    def with_structured_output(self, schema, **kwargs):
        assert kwargs == {"method": "json_schema", "include_raw": True}
        return self._runnable


class TestAinvokeStructured:
    async def test_caminho_feliz_contabiliza_consumo(self):
        runnable = _Runnable([{"raw": _Mensagem(), "parsed": Saida(valor=1), "parsing_error": None}])
        saida = await llm.ainvoke_structured(
            _ChatModel(runnable), Saida, [{"role": "user", "content": "x"}], model_id="openai:gpt-5.6-luna"
        )
        assert saida.valor == 1
        total = tracker.snapshot()["total"]
        assert total["chamadas"] == 1 and total["input_tokens"] == 100

    async def test_retenta_uma_vez_e_contabiliza_as_duas(self):
        runnable = _Runnable([
            {"raw": _Mensagem(), "parsed": None, "parsing_error": ValueError("virgula")},
            {"raw": _Mensagem(), "parsed": Saida(valor=2), "parsing_error": None},
        ])
        saida = await llm.ainvoke_structured(
            _ChatModel(runnable), Saida, [{"role": "user", "content": "x"}], model_id="openai:gpt-5.6-luna"
        )
        assert saida.valor == 2
        assert runnable.chamadas == 2
        assert tracker.snapshot()["total"]["chamadas"] == 2

    async def test_duas_falhas_levantam(self):
        runnable = _Runnable([
            {"raw": _Mensagem(), "parsed": None, "parsing_error": ValueError("a")},
            {"raw": _Mensagem(), "parsed": None, "parsing_error": ValueError("b")},
        ])
        with pytest.raises(StructuredOutputError, match="2 tentativas"):
            await llm.ainvoke_structured(
                _ChatModel(runnable), Saida, [{"role": "user", "content": "x"}], model_id="openai:gpt-5.6-luna"
            )

    async def test_truncamento_nao_reamostra(self):
        truncada = _Mensagem(
            usage={"input_tokens": 10, "output_tokens": 256, "output_token_details": {"reasoning": 256}},
            meta={"incomplete_details": {"reason": "max_output_tokens"}},
        )
        runnable = _Runnable([
            {"raw": truncada, "parsed": None, "parsing_error": ValueError("cortado")},
            {"raw": _Mensagem(), "parsed": Saida(valor=9), "parsing_error": None},
        ])
        with pytest.raises(StructuredOutputError, match="truncada"):
            await llm.ainvoke_structured(
                _ChatModel(runnable), Saida, [{"role": "user", "content": "x"}], model_id="openai:gpt-5.6-luna"
            )
        assert runnable.chamadas == 1


class TestAinvokeText:
    async def test_concatena_so_blocos_de_texto(self):
        class Modelo:
            async def ainvoke(self, _m):
                msg = _Mensagem()
                msg.content = [{"type": "reasoning", "text": "pensando"}, {"type": "text", "text": "resposta"}]
                return msg

        texto = await llm.ainvoke_text(Modelo(), [{"role": "user", "content": "x"}], model_id="local:qwen3:1.7b")
        assert texto == "resposta"
        assert tracker.snapshot()["total"]["custo_usd"] == 0.0
