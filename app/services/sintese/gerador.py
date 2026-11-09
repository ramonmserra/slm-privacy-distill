"""Engine da sintese: planeja os pedidos, chama o professor em lotes
concorrentes, verifica e deduplica o que volta e persiste tudo.

O que e' DETERMINISTICO aqui (e por isso mora no engine, nao no prompt):

- o PLANO: artigos sorteados com seed fixa, tipos e dificuldades em ciclo --
  mesma seed, mesmo plano, mesma cobertura;
- o CONTEXTO (RAG): artigo-alvo + artigos-ancora do tipo de tarefa (ver
  `taxonomia`: as definicoes para classificar dado, as hipoteses legais
  para base legal...) + REMISSOES do alvo (os artigos que o texto dele
  cita: art. 44 remete ao 46, art. 42 ao 43) + artigos relacionados por
  BM25. O alvo e' sempre da fonte principal (a lei); dispositivos de
  fonte complementar (Regulamento de Dosimetria) entram por remissao ou
  BM25 e sao citaveis como qualquer outro;
- o CENARIO de inspiracao (guias da ANPD): sorteado por tipo com seed
  derivada da seed do job; ambienta a pergunta e nunca vira citacao;
- a VERIFICACAO de citacoes: todo id citado e' normalizado e conferido no
  corpus; `citacoes_validas` e' calculado, nunca declarado pelo modelo;
  citar artigo que nao estava no contexto entregue descarta o exemplo (o
  juiz so' ve o contexto, entao nao poderia conferir);
- a DEDUPLICACAO e a protecao contra vazamento do banco inicial;
- a varredura anti-injecao da saida e o descarte de texto que fala 'do
  trecho fornecido' -- o aluno respondera' sem contexto nenhum.

O professor so' escreve pergunta, raciocinio e resposta. Tudo o que decide
o destino de um exemplo e' codigo. O laco de lotes (concorrencia, orcamento,
erro por lote) e' o compartilhado em `app.services.lotes`.
"""

from __future__ import annotations

import datetime as dt
import logging
import random
import re
import uuid

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
from app.services.corpus.referencias import artigo_de, artigos_mencionados, normalizar_referencias
from app.services.injecao import marca_de_injecao
from app.services.lotes import ExecutorDeLotes, dividir
from app.services.sintese.cenarios import Cenario, SorteioDeCenarios, carregar_cenarios
from app.services.sintese.dedup import Deduplicador, carregar_banco_inicial, hash_da_pergunta
from app.services.sintese.taxonomia import TAXONOMIA
from app.services.sintese.llm_service import (
    VERSAO_DO_PROMPT,
    ExemploGerado,
    PedidoDeExemplo,
    SinteseLLMService,
    TrechoDeContexto,
)

logger = logging.getLogger(__name__)

RELACIONADOS_POR_PEDIDO = 2
REMISSOES_POR_PEDIDO = 3
PERGUNTA_MINIMA_CHARS = 20

MOTIVO_INDICE_DESCONHECIDO = "indice_desconhecido"
MOTIVO_INJECAO = "marca_de_injecao"
MOTIVO_PERGUNTA_CURTA = "pergunta_curta"
MOTIVO_DUPLICATA = "duplicata_no_job"
MOTIVO_VAZAMENTO = "vazamento_do_banco_inicial"
MOTIVO_CITACAO_FORA_DO_CONTEXTO = "citacao_fora_do_contexto"
MOTIVO_REFERENCIA_AO_CONTEXTO = "referencia_ao_contexto"
ORIGEM_BANCO_INICIAL = "banco_inicial"
ORIGEM_JOB = "job"

# Frases que so' fazem sentido para quem viu o prompt. O exemplo vai treinar
# um modelo que respondera' SEM contexto; 'o trecho fornecido nao contem'
# vira, no aluno, uma desculpa sem referente. Deliberadamente NAO pega 'no
# contexto de uma relacao de consumo', que e' portugues normal.
_RE_REFERENCIA_AO_CONTEXTO = re.compile(
    r"\b(?:trechos?|textos?|dispositivos?|artigos?|normas?)\s+(?:fornecid\w*|apresentad\w*|dispon[ií]ve\w*|acima)"
    r"|\bcontexto\s+(?:fornecido|apresentado|dado|acima)"
    r"|\bcom base (?:apenas|somente|exclusivamente) n[oa]s?\s+(?:trechos?|textos?|dispositivos?)"
    r"|\bapenas com (?:esses|estes|os)\s+(?:trechos?|textos?|dispositivos?)",
    re.IGNORECASE,
)


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
        cenarios: list[Cenario] | None = None,
    ) -> None:
        self._servico = servico
        self._corpus = corpus
        self._repositorio = repositorio
        self._banco_inicial = carregar_banco_inicial() if banco_inicial is None else banco_inicial
        self._cenarios = carregar_cenarios() if cenarios is None else cenarios

    # -- plano --------------------------------------------------------------------

    def planejar(self, parametros: ParametrosDeSintese) -> list[PedidoDeExemplo]:
        """Um pedido por exemplo desejado. Artigos em sorteio sem reposicao
        (reembaralhado quando acabam), tipos e dificuldades em ciclo; o
        cenario sai de um gerador proprio (seed derivada) para que ligar ou
        desligar cenarios nao mude quais artigos cada pedido recebe."""

        rng = random.Random(parametros.seed)
        sorteio = SorteioDeCenarios(self._cenarios, random.Random(f"{parametros.seed}:cenarios")) if parametros.com_cenarios else None
        artigos = self._corpus.artigos(self._corpus.fonte)
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
            tipo = tipos[i % len(tipos)]
            pedidos.append(
                PedidoDeExemplo(
                    indice=i + 1, tipo=tipo, dificuldade=dificuldades[i % len(dificuldades)],
                    alvo_id=alvo.id, contexto=self._contexto(alvo.id, tipo),
                    cenario=sorteio.proximo(tipo) if sorteio else None,
                )
            )
        return pedidos

    def _contexto(self, alvo_id: str, tipo: TipoDeTarefa) -> list[TrechoDeContexto]:
        """Alvo primeiro; depois as ancoras do tipo, as remissoes do alvo (os
        artigos que o proprio texto dele cita: 'observado o art. 46') e os
        relacionados por BM25; sem repetir e so' o que existe no corpus.
        Ancora e remissao vem antes do relacionado porque sao o que a tarefa
        e a lei exigem; o relacionado e' o que o indice achou parecido."""

        ids = [alvo_id]
        texto_do_alvo = self._corpus.texto_do_artigo(alvo_id)
        candidatos = list(TAXONOMIA[tipo].ancoras)
        candidatos += artigos_mencionados(texto_do_alvo, fonte=self._corpus.fonte)[:REMISSOES_POR_PEDIDO]
        candidatos += [r.id for r in self._corpus.relacionados(alvo_id, k=RELACIONADOS_POR_PEDIDO)]
        for id_ in candidatos:
            if id_ not in ids and self._corpus.existe(id_):
                ids.append(id_)
        return [TrechoDeContexto(id=i, texto=self._corpus.texto_do_artigo(i)) for i in ids]

    # -- execucao ---------------------------------------------------------------------

    async def executar(self, job: JobDeSintese) -> JobDeSintese:
        pedidos = self.planejar(job.parametros)
        lotes = dividir(pedidos, job.parametros.tamanho_do_lote)
        job.pedidos = len(pedidos)
        job.lotes = len(lotes)
        job.status = StatusDoJob.EXECUTANDO
        job.modelo = getattr(self._servico, "model", "")
        self._salvar(job)

        dedup = Deduplicador()
        dedup.adicionar_varias(self._banco_inicial, ORIGEM_BANCO_INICIAL)

        async def chamar(lote: list[PedidoDeExemplo]) -> tuple[list[PedidoDeExemplo], list[ExemploGerado]]:
            # Lotes reindexados 1..N: o professor ve indices locais.
            locais = [p.model_copy(update={"indice": i + 1}) for i, p in enumerate(lote)]
            return locais, await self._servico.gerar(locais)

        def consolidar(_lote: list[PedidoDeExemplo], saida: tuple[list[PedidoDeExemplo], list[ExemploGerado]]) -> None:
            locais, gerados = saida
            aceitos, descartados = self._processar(job, locais, gerados, dedup)
            self._repositorio.acrescentar_exemplos(job.id, aceitos)
            self._repositorio.acrescentar_descartados(job.id, descartados)
            job.gerados += len(gerados)
            job.aceitos += len(aceitos)
            job.descartados += len(descartados)
            for d in descartados:
                job.descartes_por_motivo[d.motivo] = job.descartes_por_motivo.get(d.motivo, 0) + 1

        executor = ExecutorDeLotes(chamar=chamar, consolidar=consolidar, rotulo=f"Sintese {job.id}")
        with etapa_scope(Etapa.SINTESE), job_scope(job.id, job.parametros.teto_usd):
            resultado = await executor.executar(lotes, job.parametros.concorrencia)

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
            validos, invalidos = self._normalizar_citacoes(g.artigos_citados)
            motivo = self._motivo_de_descarte(g, pedido, dedup, validos)
            if motivo is not None:
                descartados.append(ExemploDescartado(
                    job_id=job.id, motivo=motivo[0], pergunta=g.pergunta,
                    tipo=pedido.tipo if pedido else None, detalhe=motivo[1],
                    alvo_id=pedido.alvo_id if pedido else None,
                ))
                continue
            dedup.adicionar(g.pergunta, ORIGEM_JOB)
            aceitos.append(ExemploSintetico(
                id=hash_da_pergunta(g.pergunta), job_id=job.id, tipo=pedido.tipo, dificuldade=g.dificuldade,
                pergunta=g.pergunta.strip(), raciocinio=g.raciocinio.strip(), resposta=g.resposta.strip(),
                artigos_citados=validos, artigos_citados_invalidos=invalidos,
                citacoes_validas=not invalidos and (bool(validos) or pedido.tipo == TipoDeTarefa.SEM_RESPOSTA),
                contexto_ids=[t.id for t in pedido.contexto], alvo_id=pedido.alvo_id, cenario_id=pedido.cenario.id if pedido.cenario else None,
                modelo=job.modelo,
                versao_do_prompt=VERSAO_DO_PROMPT, criado_em=_agora(),
            ))
        return aceitos, descartados

    @staticmethod
    def _motivo_de_descarte(
        g: ExemploGerado, pedido: PedidoDeExemplo | None, dedup: Deduplicador, citados: list[str]
    ) -> tuple[str, str] | None:
        """(motivo, detalhe) do descarte, ou None se o exemplo segue.
        `citados` sao os ids ja normalizados e existentes no corpus."""

        if pedido is None:
            return MOTIVO_INDICE_DESCONHECIDO, f"indice={g.indice}"
        texto = f"{g.pergunta}\n{g.raciocinio}\n{g.resposta}"
        marca = marca_de_injecao(texto)
        if marca:
            return MOTIVO_INJECAO, marca
        if len(g.pergunta.strip()) < PERGUNTA_MINIMA_CHARS:
            return MOTIVO_PERGUNTA_CURTA, ""
        referencia = _RE_REFERENCIA_AO_CONTEXTO.search(texto)
        if referencia:
            return MOTIVO_REFERENCIA_AO_CONTEXTO, referencia.group(0)
        entregues = {artigo_de(t.id) for t in pedido.contexto}
        fora = [c for c in citados if artigo_de(c) not in entregues]
        if fora:
            return MOTIVO_CITACAO_FORA_DO_CONTEXTO, ", ".join(fora)
        duplicata = dedup.verificar(g.pergunta)
        if duplicata is not None:
            motivo = MOTIVO_VAZAMENTO if duplicata.origem == ORIGEM_BANCO_INICIAL else MOTIVO_DUPLICATA
            return motivo, f"similaridade={duplicata.similaridade} com: {duplicata.referencia[:120]}"
        return None

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
