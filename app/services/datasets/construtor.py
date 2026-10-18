"""Constroi uma VERSAO de dataset a partir de decisoes de filtro (aprovados)
e/ou de jobs de sintese inteiros (condicao "sem filtro").

A versao e' o hash do conteudo ordenado + parametros: qualquer exemplo a
mais, a menos ou alterado muda a versao. Reconstruir com as mesmas origens
da' a mesma versao e nao duplica nada.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

from app.repositories.dataset_repository import DatasetRepository
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.dataset import ContagemPorTipo, LinhaDeTreino, ManifestoDoDataset, MensagemDeChat, ParametrosDeDataset, Split
from app.schemas.sintese import ExemploSintetico
from app.services.datasets.prompt_do_aluno import PROMPT_DO_ALUNO, VERSAO_DO_PROMPT_DO_ALUNO, resposta_de_treino
from app.services.datasets.split import split_de


class OrigemVazia(ValueError):
    """Nenhum exemplo encontrado nas origens informadas."""


class OrigemDesconhecida(ValueError):
    pass


class ConstrutorDeDataset:
    def __init__(self, repositorio: DatasetRepository, filtro: FiltroRepository, sintese: SinteseRepository) -> None:
        self._repositorio = repositorio
        self._filtro = filtro
        self._sintese = sintese

    def _coletar(self, parametros: ParametrosDeDataset) -> tuple[list[ExemploSintetico], int, list[str], list[str]]:
        exemplos: dict[str, ExemploSintetico] = {}
        duplicados = 0
        versoes_do_filtro: list[str] = []
        versoes_de_sintese: list[str] = []
        indice_de_sintese: dict[str, dict[str, ExemploSintetico]] = {}

        def exemplos_do_job(job_id: str) -> dict[str, ExemploSintetico]:
            if job_id not in indice_de_sintese:
                if self._sintese.carregar_job(job_id) is None:
                    raise OrigemDesconhecida(f"Job de sintese {job_id!r} nao existe.")
                indice_de_sintese[job_id] = {e.id: e for e in self._sintese.exemplos(job_id)}
            return indice_de_sintese[job_id]

        def acrescentar(e: ExemploSintetico) -> None:
            nonlocal duplicados
            if e.id in exemplos:
                duplicados += 1
                return
            exemplos[e.id] = e
            if e.versao_do_prompt not in versoes_de_sintese:
                versoes_de_sintese.append(e.versao_do_prompt)

        for job_filtro in parametros.jobs_de_filtro:
            if self._filtro.carregar_job(job_filtro) is None:
                raise OrigemDesconhecida(f"Job de filtro {job_filtro!r} nao existe.")
            for decisao in self._filtro.decisoes(job_filtro, aprovado=True):
                if decisao.versao_do_filtro not in versoes_do_filtro:
                    versoes_do_filtro.append(decisao.versao_do_filtro)
                exemplo = exemplos_do_job(decisao.job_sintese_id).get(decisao.exemplo_id)
                if exemplo is not None:
                    acrescentar(exemplo)
        for job_sintese in parametros.jobs_de_sintese_sem_filtro:
            for exemplo in exemplos_do_job(job_sintese).values():
                acrescentar(exemplo)
        return list(exemplos.values()), duplicados, versoes_do_filtro, versoes_de_sintese

    @staticmethod
    def linha(exemplo: ExemploSintetico, incluir_raciocinio: bool) -> LinhaDeTreino:
        return LinhaDeTreino(
            id=exemplo.id, tipo=exemplo.tipo.value, dificuldade=exemplo.dificuldade.value,
            artigos_citados=exemplo.artigos_citados,
            messages=[
                MensagemDeChat(role="system", content=PROMPT_DO_ALUNO),
                MensagemDeChat(role="user", content=exemplo.pergunta),
                MensagemDeChat(role="assistant", content=resposta_de_treino(exemplo.raciocinio, exemplo.resposta, incluir_raciocinio)),
            ],
        )

    @staticmethod
    def _versao(linhas: dict[Split, list[LinhaDeTreino]], parametros: ParametrosDeDataset) -> str:
        h = hashlib.sha256()
        h.update(parametros.model_dump_json(exclude={"nome"}).encode("utf-8"))
        h.update(VERSAO_DO_PROMPT_DO_ALUNO.encode("utf-8"))
        for split in Split:
            for linha in sorted(linhas[split], key=lambda l: l.id):
                h.update(split.value.encode("utf-8"))
                h.update(linha.model_dump_json().encode("utf-8"))
        return h.hexdigest()[:12]

    def construir(self, parametros: ParametrosDeDataset) -> ManifestoDoDataset:
        exemplos, duplicados, versoes_do_filtro, versoes_de_sintese = self._coletar(parametros)
        if not exemplos:
            raise OrigemVazia("nenhum exemplo nas origens informadas (jobs sem exemplos aprovados?)")

        linhas: dict[Split, list[LinhaDeTreino]] = {Split.TREINO: [], Split.HELD_OUT: []}
        contagens: dict[Split, ContagemPorTipo] = {Split.TREINO: ContagemPorTipo(), Split.HELD_OUT: ContagemPorTipo()}
        for exemplo in sorted(exemplos, key=lambda e: e.id):
            split = split_de(exemplo.id, parametros.fracao_held_out)
            linhas[split].append(self.linha(exemplo, parametros.incluir_raciocinio))
            contagens[split].total += 1
            contagens[split].por_tipo[exemplo.tipo.value] = contagens[split].por_tipo.get(exemplo.tipo.value, 0) + 1

        versao = self._versao(linhas, parametros)
        manifesto = ManifestoDoDataset(
            versao=versao, nome=parametros.nome, criado_em=dt.datetime.now(dt.timezone.utc), parametros=parametros,
            treino=contagens[Split.TREINO], held_out=contagens[Split.HELD_OUT], duplicados_entre_origens=duplicados,
            versao_do_prompt_do_aluno=VERSAO_DO_PROMPT_DO_ALUNO, versoes_do_filtro=versoes_do_filtro,
            versoes_do_prompt_de_sintese=versoes_de_sintese,
        )
        manifesto.arquivos = {split.value: str(self._repositorio.caminho_do_split(versao, split)) for split in Split}
        self._repositorio.salvar(manifesto, {s: [json.loads(l.model_dump_json()) for l in linhas[s]] for s in Split})
        return manifesto
