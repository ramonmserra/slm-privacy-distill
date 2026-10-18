"""Split treino/held-out por HASH do id do exemplo.

Por que hash e nao sorteio: o mesmo exemplo cai SEMPRE no mesmo lado, em
qualquer versao de dataset, com qualquer fracao fixa. Um exemplo que foi
held-out na v1 nunca aparece no treino da v2 -- a garantia e' por
construcao, nao por disciplina ou por lembrar de excluir.

A fracao e' aproximada por grupo (tipo, artigo): o hash nao estratifica. Com
centenas de exemplos por tipo a diferenca fica em poucos pontos
percentuais, e o manifesto reporta a contagem real por tipo em cada split.
"""

from __future__ import annotations

import hashlib

from app.schemas.dataset import Split

_RESOLUCAO = 10_000


def split_de(exemplo_id: str, fracao_held_out: float) -> Split:
    if fracao_held_out <= 0:
        return Split.TREINO
    posicao = int(hashlib.sha256(exemplo_id.encode("utf-8")).hexdigest()[:8], 16) % _RESOLUCAO
    return Split.HELD_OUT if posicao < int(fracao_held_out * _RESOLUCAO) else Split.TREINO
