"""Persistencia da avaliacao: `DATA_DIR/avaliacao/runs/<id>.json` (o run com
as metricas) e `DATA_DIR/avaliacao/<id>/itens.jsonl` (um resultado por item:
resposta, verificacao deterministica, veredito)."""

from __future__ import annotations

from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import acrescentar_jsonl, escrever_json_atomico, ler_json, ler_jsonl
from app.schemas.avaliacao import AnotacaoRegistrada, ResultadoDoItem, RunDeAvaliacao


class AvaliacaoRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "avaliacao"

    def _arquivo_do_run(self, run_id: str) -> Path:
        return self._dir() / "runs" / f"{run_id}.json"

    def _itens(self, run_id: str) -> Path:
        return self._dir() / run_id / "itens.jsonl"

    def salvar_run(self, run: RunDeAvaliacao) -> None:
        escrever_json_atomico(self._arquivo_do_run(run.id), run.model_dump(mode="json"))

    def carregar_run(self, run_id: str) -> RunDeAvaliacao | None:
        bruto = ler_json(self._arquivo_do_run(run_id), default=None)
        return RunDeAvaliacao.model_validate(bruto) if bruto else None

    def listar_runs(self) -> list[RunDeAvaliacao]:
        pasta = self._dir() / "runs"
        if not pasta.exists():
            return []
        runs = [RunDeAvaliacao.model_validate(b) for p in pasta.glob("*.json") if (b := ler_json(p, default=None))]
        return sorted(runs, key=lambda r: r.criado_em, reverse=True)

    def acrescentar_itens(self, run_id: str, resultados: list[ResultadoDoItem]) -> None:
        for r in resultados:
            acrescentar_jsonl(self._itens(run_id), r.model_dump(mode="json"))

    def itens(self, run_id: str) -> list[ResultadoDoItem]:
        return [ResultadoDoItem.model_validate(r) for r in ler_jsonl(self._itens(run_id))]

    def _anotacoes(self, run_id: str) -> Path:
        return self._dir() / run_id / "anotacoes.jsonl"

    def acrescentar_anotacao(self, anotacao: AnotacaoRegistrada) -> None:
        acrescentar_jsonl(self._anotacoes(anotacao.run_id), anotacao.model_dump(mode="json"))

    def anotacoes(self, run_id: str) -> list[AnotacaoRegistrada]:
        return [AnotacaoRegistrada.model_validate(r) for r in ler_jsonl(self._anotacoes(run_id))]
