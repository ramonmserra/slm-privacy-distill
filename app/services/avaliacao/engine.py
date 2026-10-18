"""Engine da avaliacao: resolve o modelo, carrega o conjunto, faz o modelo
responder item a item (concorrente), verifica citacoes por codigo, manda
lotes ao juiz, decide por regra e calcula as metricas.

Ordem por item:

1. RESPOSTA do modelo medido (temperatura 0, prompt do aluno; com RAG na
   condicao de controle). Erro de provedor num item e' registrado no item
   (nao derruba o run): o item fica "sem resposta" e fora do denominador.
2. VERIFICACAO deterministica: artigos mencionados na prosa existem no
   corpus? algum artigo da referencia foi citado?
3. JULGAMENTO em lote (juiz cego, rastro = artigos de referencia).
4. DECISAO (codigo): `acerto`, `alucinacao`, `abstencao_adequada`.
5. METRICAS com intervalo; `versao_da_medida` gravada no run.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import time
import uuid

from app.core import llm
from app.core.erros import OrcamentoExcedido
from app.core.llm_usage import etapa_scope, job_scope, tracker
from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.avaliacao import (
    Conjunto,
    ItemDeAvaliacao,
    ParametrosDeAvaliacao,
    RespostaDoModelo,
    ResultadoDoItem,
    RunDeAvaliacao,
    VereditoDeAvaliacao,
)
from app.schemas.comum import Etapa
from app.schemas.sintese import StatusDoJob
from app.services.avaliacao import conjuntos, metricas
from app.services.avaliacao.llm_service import VERSAO_DO_JUIZ, AvaliacaoLLMService, CasoDeAvaliacao
from app.services.corpus.ingestao import Corpus
from app.services.corpus.referencias import artigos_mencionados
from app.services.lotes import ExecutorDeLotes, dividir
from app.services.sintese.llm_service import TrechoDeContexto

logger = logging.getLogger(__name__)


def _agora() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def resolver_modelo(modelo: str) -> str:
    """Papel (`teacher`/`judge`/`student`) ou id qualificado -> id qualificado."""

    try:
        return llm.resolve_llm_id(llm.ModelTier(modelo))
    except ValueError:
        return llm.resolve_llm_id(llm.ModelTier.STUDENT, override=modelo)


def versao_da_medida(parametros: ParametrosDeAvaliacao) -> str:
    """Juiz + prompt do aluno + regras + conjunto + RAG ligado/desligado:
    qualquer mudanca aqui torna dois runs incomparaveis."""

    material = f"{VERSAO_DO_JUIZ}|{conjuntos.versao_do_conjunto(parametros.conjunto, parametros.dataset_versao)}|rag={parametros.com_rag}:{parametros.k_rag if parametros.com_rag else 0}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def novo_run(parametros: ParametrosDeAvaliacao) -> RunDeAvaliacao:
    agora = _agora()
    return RunDeAvaliacao(
        id=uuid.uuid4().hex[:12], status=StatusDoJob.PENDENTE, parametros=parametros,
        modelo=resolver_modelo(parametros.modelo), versao_da_medida=versao_da_medida(parametros),
        criado_em=agora, atualizado_em=agora,
    )


class Avaliador:
    def __init__(
        self,
        servico: AvaliacaoLLMService,
        corpus: Corpus,
        repositorio: AvaliacaoRepository,
        datasets: DatasetRepository,
    ) -> None:
        self._servico = servico
        self._corpus = corpus
        self._repositorio = repositorio
        self._datasets = datasets

    def carregar_itens(self, parametros: ParametrosDeAvaliacao) -> list[ItemDeAvaliacao]:
        if parametros.conjunto == Conjunto.HELD_OUT:
            if not parametros.dataset_versao:
                raise conjuntos.ConjuntoIndisponivel("`dataset_versao` e' obrigatorio para o conjunto held_out.")
            itens = conjuntos.carregar_held_out(parametros.dataset_versao, self._datasets)
        else:
            itens = conjuntos.carregar_banco_inicial()
        return itens[: parametros.limite] if parametros.limite else itens

    def _rastro(self, item: ItemDeAvaliacao) -> list[TrechoDeContexto]:
        return [TrechoDeContexto(id=a, texto=self._corpus.texto_do_artigo(a)) for a in item.artigos_referencia if self._corpus.existe(a)]

    def _contexto_rag(self, item: ItemDeAvaliacao, k: int) -> list[TrechoDeContexto]:
        return [TrechoDeContexto(id=a.id, texto=self._corpus.texto_do_artigo(a.id)) for a in self._corpus.buscar(item.pergunta, k)]

    def verificar(self, item: ItemDeAvaliacao, resposta: RespostaDoModelo) -> RespostaDoModelo:
        mencionados = artigos_mencionados(resposta.texto, fonte=self._corpus.fonte)
        resposta.artigos_mencionados = mencionados
        resposta.artigos_inexistentes = [a for a in mencionados if not self._corpus.existe(a)]
        resposta.citou_referencia = any(a in item.artigos_referencia for a in mencionados)
        return resposta

    async def _responder(self, item: ItemDeAvaliacao, parametros: ParametrosDeAvaliacao) -> RespostaDoModelo:
        contexto = self._contexto_rag(item, parametros.k_rag) if parametros.com_rag else []
        inicio = time.perf_counter()
        try:
            saida = await self._servico.responder(item, contexto)
        except OrcamentoExcedido:
            raise
        except Exception as exc:
            logger.exception("Avaliacao: item %s falhou", item.id)
            return RespostaDoModelo(item_id=item.id, texto="", latencia_ms=int((time.perf_counter() - inicio) * 1000), erro=f"{type(exc).__name__}: {exc}"[:300])
        resposta = RespostaDoModelo(
            item_id=item.id, texto=saida.texto, latencia_ms=int((time.perf_counter() - inicio) * 1000),
            input_tokens=saida.input_tokens, output_tokens=saida.output_tokens,
        )
        return self.verificar(item, resposta)

    async def executar(self, run: RunDeAvaliacao) -> RunDeAvaliacao:
        params = run.parametros
        try:
            itens = self.carregar_itens(params)
        except conjuntos.ConjuntoIndisponivel as exc:
            run.status = StatusDoJob.FALHOU
            run.erro = str(exc)
            self._salvar(run)
            return run
        run.itens = len(itens)
        run.status = StatusDoJob.EXECUTANDO
        run.modelo_juiz = getattr(self._servico, "modelo_juiz", "")
        self._salvar(run)

        resultados: list[ResultadoDoItem] = []
        with etapa_scope(Etapa.AVALIACAO), job_scope(run.id, params.teto_usd):
            orcamento = await self._fase_de_respostas(run, itens, resultados)
            if orcamento is None:
                orcamento = await self._fase_de_julgamento(run, resultados)

        for r in resultados:
            metricas.decidir(r)
        self._repositorio.acrescentar_itens(run.id, resultados)
        run.metricas = metricas.calcular(resultados, tracker.custo_do_job(run.id))
        if orcamento:
            run.status = StatusDoJob.ORCAMENTO_EXCEDIDO
            run.erro = orcamento
        elif run.lotes_do_juiz and run.lotes_do_juiz_com_erro == run.lotes_do_juiz:
            run.status = StatusDoJob.FALHOU
            run.erro = "o juiz falhou em todos os lotes; veja o log do processo"
        else:
            run.status = StatusDoJob.CONCLUIDO
        self._salvar(run)
        return run

    async def _fase_de_respostas(self, run: RunDeAvaliacao, itens: list[ItemDeAvaliacao], resultados: list[ResultadoDoItem]) -> str | None:
        """Respostas concorrentes; devolve a mensagem de orcamento excedido
        (parando cedo) ou None."""

        semaforo = asyncio.Semaphore(run.parametros.concorrencia)
        parar = asyncio.Event()
        excedido: list[str] = []

        async def um(item: ItemDeAvaliacao) -> None:
            if parar.is_set():
                return
            async with semaforo:
                if parar.is_set():
                    return
                try:
                    resposta = await self._responder(item, run.parametros)
                except OrcamentoExcedido as exc:
                    parar.set()
                    excedido.append(str(exc))
                    return
                resultados.append(ResultadoDoItem(item=item, resposta=resposta))
                run.respondidos += 1
                if run.respondidos % 10 == 0:
                    self._salvar(run)

        await asyncio.gather(*(um(i) for i in itens))
        resultados.sort(key=lambda r: itens.index(r.item))
        self._salvar(run)
        return excedido[0] if excedido else None

    async def _fase_de_julgamento(self, run: RunDeAvaliacao, resultados: list[ResultadoDoItem]) -> str | None:
        julgaveis = [r for r in resultados if r.resposta.erro is None]
        lotes = dividir(julgaveis, run.parametros.tamanho_do_lote_do_juiz)
        run.lotes_do_juiz = len(lotes)

        async def chamar(lote: list[ResultadoDoItem]) -> list[VereditoDeAvaliacao]:
            casos = [CasoDeAvaliacao(indice=i + 1, item=r.item, resposta=r.resposta.texto, rastro=self._rastro(r.item)) for i, r in enumerate(lote)]
            return await self._servico.julgar(casos)

        def consolidar(lote: list[ResultadoDoItem], vereditos: list[VereditoDeAvaliacao]) -> None:
            por_indice = {v.indice: v for v in vereditos}
            for i, r in enumerate(lote):
                r.veredito = por_indice.get(i + 1)

        executor = ExecutorDeLotes(chamar=chamar, consolidar=consolidar, ao_progresso=lambda: self._salvar(run), rotulo=f"Avaliacao {run.id}")
        resultado = await executor.executar(lotes, run.parametros.concorrencia)
        run.lotes_do_juiz_concluidos = resultado.concluidos
        run.lotes_do_juiz_com_erro = resultado.com_erro
        return resultado.orcamento_excedido

    def _salvar(self, run: RunDeAvaliacao) -> None:
        run.atualizado_em = _agora()
        self._repositorio.salvar_run(run)
