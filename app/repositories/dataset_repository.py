"""Persistencia de datasets: `DATA_DIR/datasets/<versao>/{treino,held_out}.jsonl`
e `manifesto.json`. Uma versao e' IMUTAVEL: reconstruir a mesma versao
reescreve arquivos identicos."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import data_dir
from app.repositories._file_store import escrever_json_atomico, ler_json, ler_jsonl
from app.schemas.dataset import ManifestoDoDataset, Split


class DatasetRepository:
    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _dir(self) -> Path:
        return (self._raiz or data_dir()) / "datasets"

    def caminho_do_split(self, versao: str, split: Split) -> Path:
        return self._dir() / versao / f"{split.value}.jsonl"

    def _manifesto(self, versao: str) -> Path:
        return self._dir() / versao / "manifesto.json"

    def salvar(self, manifesto: ManifestoDoDataset, linhas: dict[Split, list[dict]]) -> None:
        pasta = self._dir() / manifesto.versao
        pasta.mkdir(parents=True, exist_ok=True)
        for split, registros in linhas.items():
            caminho = self.caminho_do_split(manifesto.versao, split)
            tmp = caminho.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")
            tmp.replace(caminho)
        escrever_json_atomico(self._manifesto(manifesto.versao), manifesto.model_dump(mode="json"))

    def carregar(self, versao: str) -> ManifestoDoDataset | None:
        bruto = ler_json(self._manifesto(versao), default=None)
        return ManifestoDoDataset.model_validate(bruto) if bruto else None

    def listar(self) -> list[ManifestoDoDataset]:
        if not self._dir().exists():
            return []
        manifestos = [self.carregar(p.name) for p in self._dir().iterdir() if p.is_dir()]
        return sorted((m for m in manifestos if m), key=lambda m: m.criado_em, reverse=True)

    def linhas(self, versao: str, split: Split) -> list[dict]:
        return ler_jsonl(self.caminho_do_split(versao, split))
