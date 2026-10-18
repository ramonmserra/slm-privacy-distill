"""Persistencia dos runs de treino: `DATA_DIR/treinos/<id>.json`."""

from __future__ import annotations

from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import escrever_json_atomico, ler_json
from app.schemas.treino import RunDeTreino


class TreinoRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "treinos"

    def salvar(self, run: RunDeTreino) -> None:
        escrever_json_atomico(self._dir() / f"{run.id}.json", run.model_dump(mode="json"))

    def carregar(self, run_id: str) -> RunDeTreino | None:
        bruto = ler_json(self._dir() / f"{run_id}.json", default=None)
        return RunDeTreino.model_validate(bruto) if bruto else None

    def listar(self) -> list[RunDeTreino]:
        if not self._dir().exists():
            return []
        runs = [RunDeTreino.model_validate(b) for p in self._dir().glob("*.json") if (b := ler_json(p, default=None))]
        return sorted(runs, key=lambda r: r.criado_em, reverse=True)

    def por_modelo_local(self, modelo_local_id: str) -> RunDeTreino | None:
        return next((r for r in self.listar() if r.modelo_local_id == modelo_local_id), None)
