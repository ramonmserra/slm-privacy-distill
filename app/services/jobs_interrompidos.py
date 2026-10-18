"""Jobs que morreram com o processo.

Sintese, filtro e avaliacao rodam como tasks do processo da API: um restart
no meio deixa o job gravado como `executando` para sempre, e quem faz
polling nunca recebe um desfecho. No boot, todo job ainda `pendente` ou
`executando` e' marcado como `falhou` com o motivo -- o que ja foi gravado
em disco (exemplos, decisoes, itens) fica; so' o estado deixa de mentir.

E' o lugar unico dessa regra: os tres repositorios tem o mesmo par
`listar_*`/`salvar_*`, e repetir a varredura em cada um e' como elas
divergem.
"""

from __future__ import annotations

import datetime as dt
import logging

from app.repositories.avaliacao_repository import AvaliacaoRepository
from app.repositories.filtro_repository import FiltroRepository
from app.repositories.on_policy_repository import OnPolicyRepository
from app.repositories.sintese_repository import SinteseRepository
from app.schemas.sintese import StatusDoJob

logger = logging.getLogger(__name__)

MOTIVO = "interrompido por reinicio do processo da API; o que foi gravado ate' entao permanece"
_ABERTOS = (StatusDoJob.PENDENTE, StatusDoJob.EXECUTANDO)


def marcar_jobs_interrompidos(
    sintese: SinteseRepository | None = None,
    filtro: FiltroRepository | None = None,
    avaliacao: AvaliacaoRepository | None = None,
    on_policy: OnPolicyRepository | None = None,
) -> int:
    """Devolve quantos jobs foram marcados."""

    sintese = sintese or SinteseRepository()
    filtro = filtro or FiltroRepository()
    avaliacao = avaliacao or AvaliacaoRepository()
    on_policy = on_policy or OnPolicyRepository()
    agora = dt.datetime.now(dt.timezone.utc)
    marcados = 0
    for rotulo, listar, salvar in (
        ("sintese", sintese.listar_jobs, sintese.salvar_job),
        ("filtro", filtro.listar_jobs, filtro.salvar_job),
        ("avaliacao", avaliacao.listar_runs, avaliacao.salvar_run),
        ("on_policy", on_policy.listar_jobs, on_policy.salvar_job),
    ):
        for job in listar():
            if job.status not in _ABERTOS:
                continue
            job.status = StatusDoJob.FALHOU
            job.erro = MOTIVO
            job.atualizado_em = agora
            salvar(job)
            marcados += 1
            logger.warning("Job de %s %s estava %s ao subir: marcado como falhou.", rotulo, job.id, "aberto")
    return marcados
