"""Persistencia do corpus: um JSONL por fonte em `DATA_DIR/corpus/`."""

from __future__ import annotations

from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import escrever_json_atomico, ler_jsonl
from app.schemas.corpus import Dispositivo


class CorpusRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "corpus"

    def caminho(self, fonte: str) -> Path:
        return self._dir() / f"{fonte.lower()}.jsonl"

    def salvar(self, fonte: str, dispositivos: list[Dispositivo]) -> None:
        # Reescrita INTEIRA e atomica: o corpus e' um snapshot da fonte, nao
        # um log; ingerir de novo substitui.
        caminho = self.caminho(fonte)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        linhas = "\n".join(d.model_dump_json() for d in dispositivos) + "\n"
        tmp = caminho.with_suffix(".tmp")
        tmp.write_text(linhas, encoding="utf-8")
        tmp.replace(caminho)
        escrever_json_atomico(
            self._dir() / f"{fonte.lower()}.manifesto.json",
            {"fonte": fonte, "dispositivos": len(dispositivos)},
        )

    def carregar(self, fonte: str) -> list[Dispositivo]:
        return [Dispositivo.model_validate(r) for r in ler_jsonl(self.caminho(fonte))]
