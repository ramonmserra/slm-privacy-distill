"""Persistencia do filtro: `DATA_DIR/filtro/jobs/<id>.json` e
`DATA_DIR/filtro/<id>/decisoes.jsonl` (uma decisao por exemplo, aprovada ou
nao)."""

from __future__ import annotations

from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import acrescentar_jsonl, escrever_json_atomico, ler_json, ler_jsonl
from app.schemas.filtro import DecisaoDoFiltro, JobDeFiltro


class FiltroRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "filtro"

    def _arquivo_do_job(self, job_id: str) -> Path:
        return self._dir() / "jobs" / f"{job_id}.json"

    def _caminho_das_decisoes(self, job_id: str) -> Path:
        return self._dir() / job_id / "decisoes.jsonl"

    def salvar_job(self, job: JobDeFiltro) -> None:
        escrever_json_atomico(self._arquivo_do_job(job.id), job.model_dump(mode="json"))

    def carregar_job(self, job_id: str) -> JobDeFiltro | None:
        bruto = ler_json(self._arquivo_do_job(job_id), default=None)
        return JobDeFiltro.model_validate(bruto) if bruto else None

    def listar_jobs(self) -> list[JobDeFiltro]:
        pasta = self._dir() / "jobs"
        if not pasta.exists():
            return []
        jobs = [JobDeFiltro.model_validate(b) for p in pasta.glob("*.json") if (b := ler_json(p, default=None))]
        return sorted(jobs, key=lambda j: j.criado_em, reverse=True)

    def acrescentar_decisao(self, job_id: str, decisao: DecisaoDoFiltro) -> None:
        acrescentar_jsonl(self._caminho_das_decisoes(job_id), decisao.model_dump(mode="json"))

    def decisoes(self, job_id: str, aprovado: bool | None = None) -> list[DecisaoDoFiltro]:
        todas = [DecisaoDoFiltro.model_validate(r) for r in ler_jsonl(self._caminho_das_decisoes(job_id))]
        if aprovado is None:
            return todas
        return [d for d in todas if d.aprovado == aprovado]
