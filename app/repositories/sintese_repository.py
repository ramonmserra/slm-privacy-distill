"""Persistencia da sintese: `DATA_DIR/sintese/jobs/<id>.json` para o job e
`DATA_DIR/sintese/<id>/{exemplos,descartados}.jsonl` para o conteudo.
Nada e' apagado: um exemplo descartado fica no arquivo de descartados com o
motivo -- e' o que permite o experimento "treinar com e sem filtro"."""

from __future__ import annotations

from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import acrescentar_jsonl, escrever_json_atomico, ler_json, ler_jsonl
from app.schemas.sintese import ExemploDescartado, ExemploSintetico, JobDeSintese


class SinteseRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "sintese"

    def _arquivo_do_job(self, job_id: str) -> Path:
        return self._dir() / "jobs" / f"{job_id}.json"

    def caminho_dos_exemplos(self, job_id: str) -> Path:
        return self._dir() / job_id / "exemplos.jsonl"

    def _caminho_dos_descartados(self, job_id: str) -> Path:
        return self._dir() / job_id / "descartados.jsonl"

    def salvar_job(self, job: JobDeSintese) -> None:
        escrever_json_atomico(self._arquivo_do_job(job.id), job.model_dump(mode="json"))

    def carregar_job(self, job_id: str) -> JobDeSintese | None:
        bruto = ler_json(self._arquivo_do_job(job_id), default=None)
        return JobDeSintese.model_validate(bruto) if bruto else None

    def listar_jobs(self) -> list[JobDeSintese]:
        pasta = self._dir() / "jobs"
        if not pasta.exists():
            return []
        jobs = [JobDeSintese.model_validate(ler_json(p, default={})) for p in pasta.glob("*.json") if ler_json(p, default=None)]
        return sorted(jobs, key=lambda j: j.criado_em, reverse=True)

    def acrescentar_exemplos(self, job_id: str, exemplos: list[ExemploSintetico]) -> None:
        for e in exemplos:
            acrescentar_jsonl(self.caminho_dos_exemplos(job_id), e.model_dump(mode="json"))

    def acrescentar_descartados(self, job_id: str, descartados: list[ExemploDescartado]) -> None:
        for d in descartados:
            acrescentar_jsonl(self._caminho_dos_descartados(job_id), d.model_dump(mode="json"))

    def exemplos(self, job_id: str) -> list[ExemploSintetico]:
        return [ExemploSintetico.model_validate(r) for r in ler_jsonl(self.caminho_dos_exemplos(job_id))]

    def descartados(self, job_id: str) -> list[ExemploDescartado]:
        return [ExemploDescartado.model_validate(r) for r in ler_jsonl(self._caminho_dos_descartados(job_id))]
