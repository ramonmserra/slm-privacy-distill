"""Engine da sintese: planeja os pedidos, chama o professor em lotes
concorrentes, verifica e deduplica o que volta e persiste tudo.

O que e' DETERMINISTICO aqui (e por isso mora no engine, nao no prompt):

- o PLANO: artigos sorteados com seed fixa, tipos e dificuldades em ciclo --
  mesma seed, mesmo plano, mesma cobertura;
- o CONTEXTO (RAG): artigo-alvo + artigos relacionados por BM25;
- a VERIFICACAO de citacoes: todo id citado e' normalizado e conferido no
  corpus; `citacoes_validas` e' calculado, nunca declarado pelo modelo;
- a DEDUPLICACAO e a protecao contra vazamento do banco inicial;
- a varredura anti-injecao da saida.

O professor so' escreve pergunta, raciocinio e resposta. Tudo o que decide
o destino de um exemplo e' codigo.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random
import uuid

from app.core.erros import OrcamentoExcedido
from app.core.llm_usage import etapa_scope, job_scope, tracker
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.comum import Etapa
from app.schemas.sintese import (
    Dificuldade,
    ExemploDescartado,
    ExemploSintetico,
    JobDeSintese,
    ParametrosDeSintese,
    StatusDoJob,
    TipoDeTarefa,
)
from app.services.corpus.ingestao import Corpus
from app.services.corpus.referencias import normalizar_referencias
from app.services.injecao import marca_de_injecao
from app.services.sintese.dedup import Deduplicador, carregar_banco_inicial, hash_da_pergunta
from app.services.sintese.llm_service import (
    VERSAO_DO_PROMPT,
    ExemploGerado,
    PedidoDeExemplo,
    SinteseLLMService,
    TrechoDeContexto,
)

logger = logging.getLogger(__name__)

RELACIONADOS_POR_PEDIDO = 2
PERGUNTA_MINIMA_CHARS = 20

MOTIVO_INDICE_DESCONHECIDO = "indice_desconhecido"
MOTIVO_INJECAO = "marca_de_injecao"
MOTIVO_PERGUNTA_CURTA = "pergunta_curta"
MOTIVO_DUPLICATA = "duplicata_no_job"
MOTIVO_VAZAMENTO = "vazamento_do_banco_inicial"
ORIGEM_BANCO_INICIAL = "banco_inicial"
ORIGEM_JOB = "job"


def _agora() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def novo_job(parametros: ParametrosDeSintese) -> JobDeSintese:
    agora = _agora()
    return JobDeSintese(
        id=uuid.uuid4().hex[:12], status=StatusDoJob.PENDENTE, parametros=parametros,
        criado_em=agora, atualizado_em=agora, versao_do_prompt=VERSAO_DO_PROMPT,
    )


class GeradorDeSintese:
    def __init__(
        self,
        servico: SinteseLLMService,
        corpus: Corpus,
        repositorio: SinteseRepository,
        banco_inicial: list[str] | None = None,
    ) -> None:
        self._servico = servico
        self._corpus = corpus
        self._repositorio = repositorio
        self._banco_inicial = carregar_banco_inicial() if banco_inicial is None else banco_inicial

    # -- plano --------------------------------------------------------------------

    def planejar(self, parametros: ParametrosDeSintese) -> list[PedidoDeExemplo]:
        """Um pedido por exemplo desejado. Artigos em sorteio sem reposicao
        (reembaralhado quando acabam), tipos e dificuldades em ciclo."""

        rng = random.Random(parametros.seed)
        artigos = self._corpus.artigos()
        if not artigos:
            return []
        fila: list = []
        tipos = parametros.tipos or list(TipoDeTarefa)
        dificuldades = list(Dificuldade)
        pedidos: list[PedidoDeExemplo] = []
        for i in range(parametros.quantidade):
            if not fila:
                fila = list(artigos)
                rng.shuffle(fila)
            alvo = fila.pop()
            contexto = [TrechoDeContexto(id=alvo.id, texto=self._corpus.texto_do_artigo(alvo.id))]
            for rel in self._corpus.relacionados(alvo.id, k=RELACIONADOS_POR_PEDIDO):
                contexto.append(TrechoDeContexto(id=rel.id, texto=self._corpus.texto_do_artigo(rel.id)))
            pedidos.append(
                PedidoDeExemplo(
                    indice=i + 1, tipo=tipos[i % len(tipos)], dificuldade=dificuldades[i % len(dificuldades)],
                    alvo_id=alvo.id, contexto=contexto,
                )
            )
        return pedidos

    # -- execucao ---------------------------------------------------------------------

    async def executar(self, job: JobDeSintese) -> JobDeSintese:
        pedidos = self.planejar(job.parametros)
        tamanho = job.parametros.tamanho_do_lote
        lotes = [pedidos[i:i + tamanho] for i in range(0, len(pedidos), tamanho)]
        job.pedidos = len(pedidos)
        job.lotes = len(lotes)
        job.status = StatusDoJob.EXECUTANDO
        job.modelo = getattr(self._servico, "model", "")
        self._salvar(job)

        dedup = Deduplicador()
        dedup.adicionar_varias(self._banco_inicial, ORIGEM_BANCO_INICIAL)
        semaforo = asyncio.Semaphore(job.parametros.concorrencia)
        parar = asyncio.Event()
        # Um lock serializa a pos-processagem: dedup e contadores sao estado
        # compartilhado entre lotes concorrentes.
        lock = asyncio.Lock()

        async def processar_lote(lote: list[PedidoDeExemplo]) -> None:
            if parar.is_set():
                return
            async with semaforo:
                if parar.is_set():
                    return
                try:
                    # Lotes reindexados 1..N: o professor ve indices locais.
                    locais = [p.model_copy(update={"indice": i + 1}) for i, p in enumerate(lote)]
                    gerados = await self._servico.gerar(locais)
                except OrcamentoExcedido as exc:
                    parar.set()
                    async with lock:
                        job.status = StatusDoJob.ORCAMENTO_EXCEDIDO
                        job.erro = str(exc)
                        self._salvar(job)
                    return
                except Exception:
                    logger.exception("Sintese %s: lote falhou", job.id)
                    async with lock:
                        job.lotes_com_erro += 1
                        self._salvar(job)
                    return
                async with lock:
                    aceitos, descartados = self._processar(job, locais, gerados, dedup)
                    self._repositorio.acrescentar_exemplos(job.id, aceitos)
                    self._repositorio.acrescentar_descartados(job.id, descartados)
                    job.lotes_concluidos += 1
                    job.gerados += len(gerados)
                    job.aceitos += len(aceitos)
                    job.descartados += len(descartados)
                    for d in descartados:
                        job.descartes_por_motivo[d.motivo] = job.descartes_por_motivo.get(d.motivo, 0) + 1
                    self._salvar(job)

        with etapa_scope(Etapa.SINTESE), job_scope(job.id, job.parametros.teto_usd):
            await asyncio.gather(*(processar_lote(lote) for lote in lotes))

        if job.status != StatusDoJob.ORCAMENTO_EXCEDIDO:
            if job.lotes and job.lotes_com_erro == job.lotes:
                job.status = StatusDoJob.FALHOU
                job.erro = "todos os lotes falharam; veja o log do processo"
            else:
                job.status = StatusDoJob.CONCLUIDO
        self._salvar(job)
        return job

    # -- pos-processamento -------------------------------------------------------

    def _processar(
        self,
        job: JobDeSintese,
        pedidos: list[PedidoDeExemplo],
        gerados: list[ExemploGerado],
        dedup: Deduplicador,
    ) -> tuple[list[ExemploSintetico], list[ExemploDescartado]]:
        por_indice = {p.indice: p for p in pedidos}
        aceitos: list[ExemploSintetico] = []
        descartados: list[ExemploDescartado] = []
        for g in gerados:
            pedido = por_indice.get(g.indice)
            if pedido is None:
                descartados.append(ExemploDescartado(job_id=job.id, motivo=MOTIVO_INDICE_DESCONHECIDO, pergunta=g.pergunta, detalhe=f"indice={g.indice}"))
                continue
            marca = marca_de_injecao(f"{g.pergunta}\n{g.raciocinio}\n{g.resposta}")
            if marca:
                descartados.append(ExemploDescartado(job_id=job.id, motivo=MOTIVO_INJECAO, pergunta=g.pergunta, tipo=pedido.tipo, detalhe=marca))
                continue
            if len(g.pergunta.strip()) < PERGUNTA_MINIMA_CHARS:
                descartados.append(ExemploDescartado(job_id=job.id, motivo=MOTIVO_PERGUNTA_CURTA, pergunta=g.pergunta, tipo=pedido.tipo))
                continue
            duplicata = dedup.verificar(g.pergunta)
            if duplicata is not None:
                motivo = MOTIVO_VAZAMENTO if duplicata.origem == ORIGEM_BANCO_INICIAL else MOTIVO_DUPLICATA
                descartados.append(ExemploDescartado(
                    job_id=job.id, motivo=motivo, pergunta=g.pergunta, tipo=pedido.tipo,
                    detalhe=f"similaridade={duplicata.similaridade} com: {duplicata.referencia[:120]}",
                ))
                continue
            dedup.adicionar(g.pergunta, ORIGEM_JOB)
            validos, invalidos = self._normalizar_citacoes(g.artigos_citados)
            aceitos.append(ExemploSintetico(
                id=hash_da_pergunta(g.pergunta), job_id=job.id, tipo=pedido.tipo, dificuldade=g.dificuldade,
                pergunta=g.pergunta.strip(), raciocinio=g.raciocinio.strip(), resposta=g.resposta.strip(),
                artigos_citados=validos, artigos_citados_invalidos=invalidos,
                citacoes_validas=not invalidos and (bool(validos) or pedido.tipo == TipoDeTarefa.SEM_RESPOSTA),
                contexto_ids=[t.id for t in pedido.contexto], modelo=job.modelo,
                versao_do_prompt=VERSAO_DO_PROMPT, criado_em=_agora(),
            ))
        return aceitos, descartados

    def _normalizar_citacoes(self, citadas: list[str]) -> tuple[list[str], list[str]]:
        """Ids canonicos ou referencias livres -> (existem no corpus, nao existem)."""

        validos: list[str] = []
        invalidos: list[str] = []
        for bruta in citadas:
            ids = normalizar_referencias(bruta) or [bruta.strip()]
            # So' o id mais especifico da referencia conta como citacao; os
            # pais sao implicitos.
            especifico = ids[-1]
            destino = validos if self._corpus.existe(especifico) else invalidos
            if especifico not in destino:
                destino.append(especifico)
        return validos, invalidos

    def _salvar(self, job: JobDeSintese) -> None:
        job.custo_usd = round(tracker.custo_do_job(job.id), 6)
        job.atualizado_em = _agora()
        self._repositorio.salvar_job(job)
