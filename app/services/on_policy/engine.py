"""Engine do on-policy.

Por prompt do split de TREINO (nunca do held-out, por construcao -- o
engine so' le `treino.jsonl`):

1. o ALUNO gera N amostras com temperatura;
2. verificacao deterministica: artigos mencionados existem no corpus;
3. o JUIZ compara cada amostra a referencia do PROFESSOR (a resposta do
   dataset) com o rastro dos artigos citados -- mesma rubrica da avaliacao;
4. regra (codigo): amostra APROVADA = correta e, se exigido, com lastro e
   sem artigo inexistente;
5. SFT: a aprovada de maior similaridade vira o turno do assistente; sem
   aprovada, o prompt sai (ou mantem a resposta do professor, se pedido);
6. DPO: (melhor aprovada, pior reprovada) do mesmo prompt.

O aluno nunca ve a referencia; o juiz nunca sabe que esta' julgando o aluno.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid

from app.core.erros import OrcamentoExcedido
from app.core.llm_usage import etapa_scope, job_scope, tracker
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.on_policy_repository import OnPolicyRepository
from app.schemas.avaliacao import Conjunto, ItemDeAvaliacao, VereditoDeAvaliacao
from app.schemas.comum import Etapa
from app.schemas.dataset import Split
from app.schemas.on_policy import AmostraJulgada, JobOnPolicy, LinhaDoSftOnPolicy, OrigemDoSft, ParametrosOnPolicy, ParDpo
from app.schemas.sintese import StatusDoJob, TipoDeTarefa
from app.services.avaliacao.conjuntos import ConjuntoIndisponivel
from app.services.avaliacao.llm_service import VERSAO_DO_JUIZ, CasoDeAvaliacao
from app.services.corpus.ingestao import Corpus
from app.services.corpus.referencias import artigo_de, artigos_mencionados
from app.services.datasets.prompt_do_aluno import PROMPT_DO_ALUNO
from app.services.lotes import ExecutorDeLotes, dividir
from app.services.on_policy.llm_service import OnPolicyLLMService
from app.services.sintese.llm_service import TrechoDeContexto

logger = logging.getLogger(__name__)


def _agora() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def novo_job(parametros: ParametrosOnPolicy) -> JobOnPolicy:
    agora = _agora()
    return JobOnPolicy(
        id=uuid.uuid4().hex[:12], status=StatusDoJob.PENDENTE, parametros=parametros, aluno=parametros.aluno,
        versao_da_medida=VERSAO_DO_JUIZ, criado_em=agora, atualizado_em=agora,
    )


class PromptDeTreino(ItemDeAvaliacao):
    """Um prompt do split de treino com a referencia do professor."""

    resposta_do_professor: str


def prompts_de_treino(dataset_versao: str, datasets: DatasetRepository, limite: int | None) -> list[PromptDeTreino]:
    if datasets.carregar(dataset_versao) is None:
        raise ConjuntoIndisponivel(f"Dataset {dataset_versao!r} nao existe.")
    linhas = datasets.linhas(dataset_versao, Split.TREINO)
    if not linhas:
        raise ConjuntoIndisponivel(f"Dataset {dataset_versao!r} nao tem split de treino.")
    prompts: list[PromptDeTreino] = []
    for l in linhas[:limite] if limite else linhas:
        mensagens = {m["role"]: m["content"] for m in l["messages"]}
        prompts.append(PromptDeTreino(
            id=l["id"], conjunto=Conjunto.HELD_OUT, pergunta=mensagens["user"], referencia=mensagens["assistant"],
            resposta_do_professor=mensagens["assistant"], artigos_referencia=sorted({artigo_de(a) for a in l.get("artigos_citados", [])}),
            tipo=l.get("tipo", ""), abstencao_esperada=l.get("tipo") == TipoDeTarefa.SEM_RESPOSTA.value,
        ))
    return prompts


def aprovada(amostra: AmostraJulgada, exigir_lastro: bool) -> bool:
    v = amostra.veredito
    if v is None or not v.correto or amostra.artigos_inexistentes:
        return False
    return not exigir_lastro or v.grounded is not False


def selecionar(prompt: PromptDeTreino, amostras: list[AmostraJulgada], parametros: ParametrosOnPolicy) -> tuple[LinhaDoSftOnPolicy | None, ParDpo | None]:
    """Funcao pura: da lista julgada de um prompt, a linha de SFT e o par
    DPO (ou None para cada)."""

    boas = sorted((a for a in amostras if a.aprovada), key=lambda a: -(a.veredito.similaridade if a.veredito else 0))
    ruins = sorted((a for a in amostras if a.veredito is not None and not a.aprovada), key=lambda a: a.veredito.similaridade)
    prompt_msgs = [{"role": "system", "content": PROMPT_DO_ALUNO}, {"role": "user", "content": prompt.pergunta}]
    linha: LinhaDoSftOnPolicy | None = None
    if boas:
        linha = LinhaDoSftOnPolicy(id=prompt.id, tipo=prompt.tipo, origem=OrigemDoSft.ALUNO, similaridade=boas[0].veredito.similaridade,
                                   messages=prompt_msgs + [{"role": "assistant", "content": boas[0].texto}])
    elif parametros.fallback_para_o_professor:
        linha = LinhaDoSftOnPolicy(id=prompt.id, tipo=prompt.tipo, origem=OrigemDoSft.PROFESSOR,
                                   messages=prompt_msgs + [{"role": "assistant", "content": prompt.resposta_do_professor}])
    par: ParDpo | None = None
    if boas and ruins:
        pior = ruins[0]
        motivo = pior.veredito.categoria.value if pior.veredito.categoria else ("sem_lastro" if pior.veredito.grounded is False else "artigo_inexistente")
        par = ParDpo(id=prompt.id, tipo=prompt.tipo, prompt=prompt_msgs,
                     chosen=[{"role": "assistant", "content": boas[0].texto}], rejected=[{"role": "assistant", "content": pior.texto}],
                     motivo_da_rejeicao=motivo)
    return linha, par


class EngineOnPolicy:
    def __init__(self, servico: OnPolicyLLMService, corpus: Corpus, repositorio: OnPolicyRepository, datasets: DatasetRepository) -> None:
        self._servico = servico
        self._corpus = corpus
        self._repositorio = repositorio
        self._datasets = datasets

    def _rastro(self, prompt: PromptDeTreino) -> list[TrechoDeContexto]:
        return [TrechoDeContexto(id=a, texto=self._corpus.texto_do_artigo(a)) for a in prompt.artigos_referencia if self._corpus.existe(a)]

    async def _amostrar(self, prompt: PromptDeTreino, indice: int) -> AmostraJulgada:
        try:
            saida = await self._servico.amostrar(prompt.pergunta)
        except OrcamentoExcedido:
            raise
        except Exception as exc:
            logger.exception("On-policy: amostra %s/%d falhou", prompt.id, indice)
            return AmostraJulgada(prompt_id=prompt.id, indice=indice, texto="", erro=f"{type(exc).__name__}: {exc}"[:300])
        mencionados = artigos_mencionados(saida.texto, fonte=self._corpus.fonte)
        return AmostraJulgada(
            prompt_id=prompt.id, indice=indice, texto=saida.texto, input_tokens=saida.input_tokens, output_tokens=saida.output_tokens,
            artigos_inexistentes=[a for a in mencionados if not self._corpus.existe(a)],
        )

    async def executar(self, job: JobOnPolicy) -> JobOnPolicy:
        params = job.parametros
        try:
            prompts = prompts_de_treino(params.dataset_versao, self._datasets, params.limite)
        except ConjuntoIndisponivel as exc:
            job.status, job.erro = StatusDoJob.FALHOU, str(exc)
            self._salvar(job)
            return job
        job.prompts = len(prompts)
        job.status = StatusDoJob.EXECUTANDO
        job.modelo_juiz = getattr(self._servico, "modelo_juiz", "")
        self._salvar(job)

        amostras_por_prompt: dict[str, list[AmostraJulgada]] = {p.id: [] for p in prompts}
        with etapa_scope(Etapa.ON_POLICY), job_scope(job.id, params.teto_usd):
            orcamento = await self._fase_de_amostragem(job, prompts, amostras_por_prompt)
            if orcamento is None:
                orcamento = await self._fase_de_julgamento(job, prompts, amostras_por_prompt)

        self._consolidar(job, prompts, amostras_por_prompt)
        if orcamento:
            job.status, job.erro = StatusDoJob.ORCAMENTO_EXCEDIDO, orcamento
        elif job.lotes_do_juiz and job.lotes_do_juiz_com_erro == job.lotes_do_juiz:
            job.status, job.erro = StatusDoJob.FALHOU, "o juiz falhou em todos os lotes; veja o log do processo"
        else:
            job.status = StatusDoJob.CONCLUIDO
        self._salvar(job)
        return job

    async def _fase_de_amostragem(self, job: JobOnPolicy, prompts: list[PromptDeTreino], por_prompt: dict[str, list[AmostraJulgada]]) -> str | None:
        semaforo = asyncio.Semaphore(job.parametros.concorrencia)
        parar = asyncio.Event()
        excedido: list[str] = []

        async def um(prompt: PromptDeTreino, indice: int) -> None:
            if parar.is_set():
                return
            async with semaforo:
                if parar.is_set():
                    return
                try:
                    amostra = await self._amostrar(prompt, indice)
                except OrcamentoExcedido as exc:
                    parar.set()
                    excedido.append(str(exc))
                    return
                por_prompt[prompt.id].append(amostra)
                job.amostras += 1

        inicio = time.perf_counter()
        await asyncio.gather(*(um(p, i + 1) for p in prompts for i in range(job.parametros.amostras_por_prompt)))
        logger.info("On-policy %s: %d amostras em %.0fs", job.id, job.amostras, time.perf_counter() - inicio)
        job.prompts_processados = sum(1 for p in prompts if por_prompt[p.id])
        self._salvar(job)
        return excedido[0] if excedido else None

    async def _fase_de_julgamento(self, job: JobOnPolicy, prompts: list[PromptDeTreino], por_prompt: dict[str, list[AmostraJulgada]]) -> str | None:
        por_id = {p.id: p for p in prompts}
        julgaveis = [a for p in prompts for a in por_prompt[p.id] if a.erro is None and a.texto.strip()]
        lotes = dividir(julgaveis, job.parametros.tamanho_do_lote_do_juiz)
        job.lotes_do_juiz = len(lotes)

        async def chamar(lote: list[AmostraJulgada]) -> list[VereditoDeAvaliacao]:
            casos = [CasoDeAvaliacao(indice=i + 1, item=por_id[a.prompt_id], resposta=a.texto, rastro=self._rastro(por_id[a.prompt_id])) for i, a in enumerate(lote)]
            return await self._servico.julgar(casos)

        def consolidar(lote: list[AmostraJulgada], vereditos: list[VereditoDeAvaliacao]) -> None:
            por_indice = {v.indice: v for v in vereditos}
            for i, a in enumerate(lote):
                a.veredito = por_indice.get(i + 1)
                a.aprovada = aprovada(a, job.parametros.exigir_lastro)

        executor = ExecutorDeLotes(chamar=chamar, consolidar=consolidar, ao_progresso=lambda: self._salvar(job), rotulo=f"On-policy {job.id}")
        resultado = await executor.executar(lotes, job.parametros.concorrencia)
        job.lotes_do_juiz_com_erro = resultado.com_erro
        return resultado.orcamento_excedido

    def _consolidar(self, job: JobOnPolicy, prompts: list[PromptDeTreino], por_prompt: dict[str, list[AmostraJulgada]]) -> None:
        linhas: list[LinhaDoSftOnPolicy] = []
        pares: list[ParDpo] = []
        for p in prompts:
            amostras = por_prompt[p.id]
            self._repositorio.acrescentar_amostras(job.id, amostras)
            job.amostras_aprovadas += sum(1 for a in amostras if a.aprovada)
            if any(a.aprovada for a in amostras):
                job.prompts_com_aprovada += 1
            linha, par = selecionar(p, amostras, job.parametros)
            if linha:
                linhas.append(linha)
                if linha.origem == OrigemDoSft.PROFESSOR:
                    job.linhas_sft_do_professor += 1
            if par:
                pares.append(par)
        self._repositorio.gravar_sft(job.id, linhas)
        self._repositorio.gravar_dpo(job.id, pares)
        job.linhas_sft = len(linhas)
        job.pares_dpo = len(pares)

    def _salvar(self, job: JobOnPolicy) -> None:
        job.custo_usd = round(tracker.custo_do_job(job.id), 6)
        job.atualizado_em = _agora()
        self._repositorio.salvar_job(job)
