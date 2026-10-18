"""Escrita atomica de JSON em disco -- a mecanica que todos os repositorios
em arquivo compartilham.

`tmp` + `os.replace`: um restart no meio de um write nao deixa arquivo pela
metade. O retry do `os.replace` existe por causa do Windows: antivirus e
indexador seguram o destino por milissegundos e o `replace` falha com
PermissionError transitorio (achado real no projeto de referencia).

LIMITACAO CONHECIDA: pressupoe UMA instancia do processo escrevendo. Com
`uvicorn --workers N` duas copias reescreveriam o mesmo arquivo. Este projeto
roda uma instancia; se isso mudar, a persistencia migra para SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_REPLACE_ATTEMPTS = 6
_REPLACE_INITIAL_DELAY_S = 0.05


def _replace_with_retry(tmp: Path, path: Path) -> None:
    delay = _REPLACE_INITIAL_DELAY_S
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(delay)
            delay *= 2


def escrever_json_atomico(path: Path, payload: Any) -> None:
    """Grava `payload` em `path` por inteiro ou nao grava nada."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _replace_with_retry(tmp, path)


def ler_json(path: Path, default: Any) -> Any:
    """Le o JSON de `path`; arquivo ausente ou ilegivel devolve `default` (com
    aviso no segundo caso -- dado corrompido nao pode impedir o processo de
    subir, mas tambem nao pode sumir em silencio)."""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("Arquivo ilegivel (%s) -- ignorado.", path)
        return default


def acrescentar_jsonl(path: Path, registro: dict) -> None:
    """Uma linha JSON ao fim do arquivo. Datasets crescem assim: append e'
    barato e um crash perde no maximo a ultima linha."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def ler_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    registros: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for numero, linha in enumerate(f, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError:
                logger.warning("%s: linha %d invalida -- pulada.", path, numero)
    return registros
