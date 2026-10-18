"""Engine do filtro: pre-checagens deterministicas, reamostragem do
professor, julgamento e decisao.

Ordem por exemplo:

1. CITACOES (sem LLM). Reprovado aqui nao vai ao LLM: nao se paga para
   julgar o que ja esta reprovado por regra.
2. REAMOSTRAGEM (professor, k vezes, temperatura > 0) em lote.
3. JULGAMENTO (juiz) em lote, com o rastro dos dispositivos citados e as
   reamostragens.
4. DECISAO (sem LLM): `aprovado` = todos os criterios da rubrica verdadeiros
   E score_incerteza <= limiar. `score_incerteza` = 1 - fracao de
   reamostragens equivalentes.

O juiz pontua; quem decide e' a regra em codigo, com motivos nomeados. O
laco de lotes e' o compartilhado em `app.services.lotes`.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from app.core.llm_usage import etapa_scope, job_scope, tracker
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.comum import Etapa
from app.schemas.filtro import (
    MOTIVO_ABSTENCAO_INADEQUADA,
    MOTIVO_ARTIGO_INEXISTENTE_NO_TEXTO,
    MOTIVO_CITACAO_INVALIDA,
    MOTIVO_DADO_PESSOAL_REAL,
    MOTIVO_INCERTEZA_ALTA,
    MOTIVO_INCOMPLETO,
    MOTIVO_INCORRETO,
    MOTIVO_JUIZ_INDISPONIVEL,
    MOTIVO_SEM_CITACAO,
    MOTIVO_SEM_LASTRO,
    DecisaoDoFiltro,
    JobDeFiltro,
    ParametrosDeFiltro,
    VereditoDeQualidade,
)
from app.schemas.sintese import ExemploSintetico, StatusDoJob, TipoDeTarefa
from app.services.corpus.ingestao import Corpus
from app.services.corpus.referencias import artigos_mencionados
from app.services.filtro.llm_service import (
    VERSAO_DO_FILTRO,
    CasoDeQualidade,
    FiltroLLMService,
    PedidoDeReamostragem,
    RespostaReamostrada,
)
from app.services.lotes import ExecutorDeLotes, dividir
from app.services.sintese.llm_service import TrechoDeContexto

logger = logging.getLogger(__name__)


def _agora() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def novo_job(parametros: ParametrosDeFiltro) -> JobDeFiltro:
    agora = _agora()
    return JobDeFiltro(
        id=uuid.uuid4().hex[:12], status=StatusDoJob.PENDENTE, parametros=parametros,
        criado_em=agora, atualizado_em=agora, versao_do_filtro=VERSAO_DO_FILTRO,
    )


def pre_checagem(exemplo: ExemploSintetico, corpus: Corpus, exigir_citacoes_validas: bool) -> list[str]:
    """Motivos de reprovacao DETERMINISTICOS; lista vazia = segue ao LLM."""

    motivos: list[str] = []
    if exigir_citacoes_validas:
        invalidos = [i for i in exemplo.artigos_citados if not corpus.existe(i)] + exemplo.artigos_citados_invalidos
        if invalidos:
            motivos.append(MOTIVO_CITACAO_INVALIDA)
        if exemplo.tipo != TipoDeTarefa.SEM_RESPOSTA and not exemplo.artigos_citados:
            motivos.append(MOTIVO_SEM_CITACAO)
    mencionados = artigos_mencionados(f"{exemplo.raciocinio}\n{exemplo.resposta}", fonte=corpus.fonte)
    if any(not corpus.existe(a) for a in mencionados):
        motivos.append(MOTIVO_ARTIGO_INEXISTENTE_NO_TEXTO)
    return motivos


def motivos_do_veredito(veredito: VereditoDeQualidade, tipo: TipoDeTarefa) -> list[str]:
    motivos: list[str] = []
    if not veredito.correcao_juridica:
        motivos.append(MOTIVO_INCORRETO)
    if not veredito.lastro_nas_citacoes:
        motivos.append(MOTIVO_SEM_LASTRO)
    if not veredito.completude:
        motivos.append(MOTIVO_INCOMPLETO)
    if tipo == TipoDeTarefa.SEM_RESPOSTA and veredito.abstencao_adequada is False:
        motivos.append(MOTIVO_ABSTENCAO_INADEQUADA)
    if not veredito.sem_dado_pessoal_real:
        motivos.append(MOTIVO_DADO_PESSOAL_REAL)
    return motivos


def score_de_incerteza(equivalentes: list[bool]) -> float | None:
    if not equivalentes:
        return None
    return round(1 - sum(1 for e in equivalentes if e) / len(equivalentes), 3)


class FiltroDeQualidade:
    def __init__(
        self,
        servico: FiltroLLMService,
        corpus: Corpus,
        repositorio: FiltroRepository,
        sintese: SinteseRepository,
    ) -> None:
        self._servico = servico
        self._corpus = corpus
        self._repositorio = repositorio
        self._sintese = sintese

    def _carregar_exemplos(self, jobs: list[str]) -> list[ExemploSintetico]:
        vistos: set[str] = set()
        exemplos: list[ExemploSintetico] = []
        for job_id in jobs:
            for e in self._sintese.exemplos(job_id):
                if e.id not in vistos:
                    vistos.add(e.id)
                    exemplos.append(e)
        return exemplos

    def _contexto_dos_citados(self, exemplo: ExemploSintetico) -> list[TrechoDeContexto]:
        artigos: list[str] = []
        for id_ in exemplo.artigos_citados:
            raiz = ":".join(id_.split(":")[:2])
            if raiz not in artigos:
                artigos.append(raiz)
        return [TrechoDeContexto(id=a, texto=self._corpus.texto_do_artigo(a)) for a in artigos if self._corpus.existe(a)]

    async def executar(self, job: JobDeFiltro) -> JobDeFiltro:
        params = job.parametros
        exemplos = self._carregar_exemplos(params.jobs_de_sintese)
        job.exemplos = len(exemplos)
        job.status = StatusDoJob.EXECUTANDO
        job.modelo_juiz = getattr(self._servico, "modelo_juiz", "")
        job.modelo_professor = getattr(self._servico, "modelo_professor", "")

        para_llm: list[ExemploSintetico] = []
        for e in exemplos:
            motivos = pre_checagem(e, self._corpus, params.exigir_citacoes_validas)
            if motivos:
                self._registrar(job, e, aprovado=False, motivos=motivos, veredito=None, reamostragens=[], score=None)
                job.reprovados_sem_llm += 1
            else:
                para_llm.append(e)

        lotes = dividir(para_llm, params.tamanho_do_lote)
        job.lotes = len(lotes)
        self._salvar(job)

        async def chamar(lote: list[ExemploSintetico]) -> tuple[list[list[str]], list[VereditoDeQualidade]]:
            reamostragens = await self._reamostrar(lote, params.k_reamostragens)
            casos = [
                CasoDeQualidade(
                    indice=i + 1, tipo=e.tipo, pergunta=e.pergunta, raciocinio=e.raciocinio, resposta=e.resposta,
                    artigos_citados=e.artigos_citados, contexto=self._contexto_dos_citados(e),
                    reamostragens=reamostragens[i],
                )
                for i, e in enumerate(lote)
            ]
            return reamostragens, await self._servico.julgar(casos)

        def consolidar(lote: list[ExemploSintetico], saida: tuple[list[list[str]], list[VereditoDeQualidade]]) -> None:
            reamostragens, vereditos = saida
            por_indice = {v.indice: v for v in vereditos}
            for i, e in enumerate(lote):
                veredito = por_indice.get(i + 1)
                if veredito is None:
                    self._registrar(job, e, aprovado=False, motivos=[MOTIVO_JUIZ_INDISPONIVEL], veredito=None, reamostragens=reamostragens[i], score=None)
                    continue
                score = score_de_incerteza(veredito.reamostragens_equivalentes[: len(reamostragens[i])])
                motivos = motivos_do_veredito(veredito, e.tipo)
                if score is not None and score > params.limiar_de_incerteza:
                    motivos.append(MOTIVO_INCERTEZA_ALTA)
                self._registrar(job, e, aprovado=not motivos, motivos=motivos, veredito=veredito, reamostragens=reamostragens[i], score=score)
                job.avaliados += 1

        executor = ExecutorDeLotes(chamar=chamar, consolidar=consolidar, ao_progresso=lambda: self._salvar(job), rotulo=f"Filtro {job.id}")
        with etapa_scope(Etapa.FILTRO), job_scope(job.id, params.teto_usd):
            resultado = await executor.executar(lotes, params.concorrencia)

        job.lotes_concluidos = resultado.concluidos
        job.lotes_com_erro = resultado.com_erro
        if resultado.orcamento_excedido:
            job.status = StatusDoJob.ORCAMENTO_EXCEDIDO
            job.erro = resultado.orcamento_excedido
        elif resultado.todos_falharam:
            job.status = StatusDoJob.FALHOU
            job.erro = "todos os lotes falharam; veja o log do processo"
        else:
            job.status = StatusDoJob.CONCLUIDO
        self._salvar(job)
        return job

    async def _reamostrar(self, lote: list[ExemploSintetico], k: int) -> list[list[str]]:
        """k respostas alternativas por exemplo, na ordem do lote. Cada rodada
        e' UMA chamada ao professor com o lote inteiro."""

        por_exemplo: list[list[str]] = [[] for _ in lote]
        if k <= 0:
            return por_exemplo
        pedidos = [
            PedidoDeReamostragem(indice=i + 1, pergunta=e.pergunta, contexto=self._contexto_do_pedido(e))
            for i, e in enumerate(lote)
        ]
        for _ in range(k):
            respostas: list[RespostaReamostrada] = await self._servico.reamostrar(pedidos)
            por_indice = {r.indice: r.resposta for r in respostas}
            for i in range(len(lote)):
                if i + 1 in por_indice:
                    por_exemplo[i].append(por_indice[i + 1])
        return por_exemplo

    def _contexto_do_pedido(self, exemplo: ExemploSintetico) -> list[TrechoDeContexto]:
        """O MESMO contexto que o professor viu na sintese (`contexto_ids`):
        reamostrar com outro contexto mediria outra coisa."""

        return [
            TrechoDeContexto(id=i, texto=self._corpus.texto_do_artigo(i))
            for i in exemplo.contexto_ids
            if self._corpus.existe(i)
        ]

    def _registrar(
        self,
        job: JobDeFiltro,
        exemplo: ExemploSintetico,
        *,
        aprovado: bool,
        motivos: list[str],
        veredito: VereditoDeQualidade | None,
        reamostragens: list[str],
        score: float | None,
    ) -> None:
        self._repositorio.acrescentar_decisao(
            job.id,
            DecisaoDoFiltro(
                exemplo_id=exemplo.id, job_filtro_id=job.id, job_sintese_id=exemplo.job_id, tipo=exemplo.tipo,
                aprovado=aprovado, score_incerteza=score, motivos=motivos, veredito=veredito,
                reamostragens=reamostragens, modelo_juiz=job.modelo_juiz, modelo_professor=job.modelo_professor,
                versao_do_filtro=VERSAO_DO_FILTRO, criado_em=_agora(),
            ),
        )
        if aprovado:
            job.aprovados += 1
        else:
            job.reprovados += 1
            for m in motivos:
                job.reprovacoes_por_motivo[m] = job.reprovacoes_por_motivo.get(m, 0) + 1

    def _salvar(self, job: JobDeFiltro) -> None:
        job.custo_usd = round(tracker.custo_do_job(job.id), 6)
        job.atualizado_em = _agora()
        self._repositorio.salvar_job(job)
