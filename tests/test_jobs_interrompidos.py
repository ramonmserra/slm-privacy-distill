"""Jobs abertos no boot sao marcados como falhos, nos tres repositorios."""

from __future__ import annotations

import datetime as dt

from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.avaliacao import ParametrosDeAvaliacao, RunDeAvaliacao
from app.schemas.filtro import JobDeFiltro, ParametrosDeFiltro
from app.schemas.sintese import JobDeSintese, ParametrosDeSintese, StatusDoJob
from app.services.jobs_interrompidos import MOTIVO, marcar_jobs_interrompidos

AGORA = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)


def test_marca_so_os_abertos(tmp_path):
    sintese, filtro, avaliacao = SinteseRepository(tmp_path), FiltroRepository(tmp_path), AvaliacaoRepository(tmp_path)
    sintese.salvar_job(JobDeSintese(id="s-exec", status=StatusDoJob.EXECUTANDO, parametros=ParametrosDeSintese(), criado_em=AGORA, atualizado_em=AGORA))
    sintese.salvar_job(JobDeSintese(id="s-ok", status=StatusDoJob.CONCLUIDO, parametros=ParametrosDeSintese(), criado_em=AGORA, atualizado_em=AGORA, aceitos=3))
    filtro.salvar_job(JobDeFiltro(id="f-pend", status=StatusDoJob.PENDENTE, parametros=ParametrosDeFiltro(jobs_de_sintese=["s"]), criado_em=AGORA, atualizado_em=AGORA))
    avaliacao.salvar_run(RunDeAvaliacao(id="a-exec", status=StatusDoJob.EXECUTANDO, parametros=ParametrosDeAvaliacao(modelo="student"), modelo="local:x", versao_da_medida="v", criado_em=AGORA, atualizado_em=AGORA, respondidos=5))

    assert marcar_jobs_interrompidos(sintese, filtro, avaliacao) == 3

    assert sintese.carregar_job("s-exec").status == StatusDoJob.FALHOU and sintese.carregar_job("s-exec").erro == MOTIVO
    assert sintese.carregar_job("s-ok").status == StatusDoJob.CONCLUIDO and sintese.carregar_job("s-ok").aceitos == 3
    assert filtro.carregar_job("f-pend").status == StatusDoJob.FALHOU
    run = avaliacao.carregar_run("a-exec")
    assert run.status == StatusDoJob.FALHOU and run.respondidos == 5  # o progresso gravado fica
    assert marcar_jobs_interrompidos(sintese, filtro, avaliacao) == 0


def test_sem_dados_nao_falha(tmp_path):
    assert marcar_jobs_interrompidos(SinteseRepository(tmp_path), FiltroRepository(tmp_path), AvaliacaoRepository(tmp_path)) == 0
