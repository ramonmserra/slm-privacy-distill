"""Deixa o `tokenizer_config.json` do modelo fundido legivel pelo transformers
4.x que o conversor do llama.cpp instala.

    python -m treino.normalizar_tokenizer saida/merged

O transformers 5 (o do treino no Colab) grava `extra_special_tokens` como
LISTA; o 4.x espera um dicionario e quebra com "'list' object has no attribute
'keys'" ao carregar o tokenizer. A lista e' o que o 4.x chama de
`additional_special_tokens`, entao basta renomear a chave. O vocabulario em
si (tokenizer.json) nao e' tocado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def normalizar(config: dict) -> bool:
    """Altera `config` no lugar; devolve True se mudou algo."""

    extras = config.get("extra_special_tokens")
    if not isinstance(extras, list):
        return False
    del config["extra_special_tokens"]
    config.setdefault("additional_special_tokens", extras)
    return True


def main(argv: list[str] | None = None) -> int:
    argumentos = sys.argv[1:] if argv is None else argv
    if len(argumentos) != 1:
        print("uso: python -m treino.normalizar_tokenizer <pasta do modelo>", file=sys.stderr)
        return 2
    caminho = Path(argumentos[0]) / "tokenizer_config.json"
    config = json.loads(caminho.read_text(encoding="utf-8"))
    if normalizar(config):
        caminho.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{caminho}: extra_special_tokens (lista) -> additional_special_tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
