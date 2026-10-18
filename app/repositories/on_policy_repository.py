"""Persistencia do on-policy: `DATA_DIR/on_policy/jobs/<id>.json` e, por job,
`amostras.jsonl` (toda amostra, aprovada ou nao, com veredito), `sft.jsonl`
(formato de chat, pronto para `treino/sft_lora.py`) e `dpo.jsonl`
(prompt/chosen/rejected, pronto para `treino/dpo_lora.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import acrescentar_jsonl, escrever_json_atomico, ler_json, ler_jsonl
from app.schemas.on_policy import AmostraJulgada, JobOnPolicy, LinhaDoSftOnPolicy, ParDpo


class OnPolicyRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "on_policy"

    def _arquivo_do_job(self, job_id: str) -> Path:
        return self._dir() / "jobs" / f"{job_id}.json"

    def caminho(self, job_id: str, nome: str) -> Path:
        return self._dir() / job_id / f"{nome}.jsonl"

    def salvar_job(self, job: JobOnPolicy) -> None:
        escrever_json_atomico(self._arquivo_do_job(job.id), job.model_dump(mode="json"))

    def carregar_job(self, job_id: str) -> JobOnPolicy | None:
        bruto = ler_json(self._arquivo_do_job(job_id), default=None)
        return JobOnPolicy.model_validate(bruto) if bruto else None

    def listar_jobs(self) -> list[JobOnPolicy]:
        pasta = self._dir() / "jobs"
        if not pasta.exists():
            return []
        jobs = [JobOnPolicy.model_validate(b) for p in pasta.glob("*.json") if (b := ler_json(p, default=None))]
        return sorted(jobs, key=lambda j: j.criado_em, reverse=True)

    def acrescentar_amostras(self, job_id: str, amostras: list[AmostraJulgada]) -> None:
        for a in amostras:
            acrescentar_jsonl(self.caminho(job_id, "amostras"), a.model_dump(mode="json"))

    def amostras(self, job_id: str) -> list[AmostraJulgada]:
        return [AmostraJulgada.model_validate(r) for r in ler_jsonl(self.caminho(job_id, "amostras"))]

    def gravar_sft(self, job_id: str, linhas: list[LinhaDoSftOnPolicy]) -> None:
        self._gravar(self.caminho(job_id, "sft"), [l.model_dump(mode="json") for l in linhas])

    def gravar_dpo(self, job_id: str, pares: list[ParDpo]) -> None:
        self._gravar(self.caminho(job_id, "dpo"), [p.model_dump(mode="json") for p in pares])

    @staticmethod
    def _gravar(caminho: Path, registros: list[dict]) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        tmp = caminho.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")
        tmp.replace(caminho)

    def linhas(self, job_id: str, nome: str) -> list[dict]:
        return ler_jsonl(self.caminho(job_id, nome))
