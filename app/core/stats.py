"""Estatistica da avaliacao -- funcoes puras, sem dependencia externa.

- Wilson: intervalo de confianca de uma proporcao. Com 34 itens do banco
  inicial o intervalo e' largo (+-15 p.p.), e e' exatamente por isso que ele
  acompanha TODA proporcao reportada: um numero sem intervalo convida a
  ler ruido como diferenca.
- McNemar exato: dois modelos nos MESMOS itens; so' os itens em que eles
  divergem informam. E' o teste pareado certo para acerto/erro.
- Bootstrap pareado do delta de acuracia: intervalo para "quantos pontos o
  modelo B ganha de A", reamostrando itens.
- Kappa de Cohen: concordancia juiz x humano na amostra de calibracao.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Intervalo:
    inferior: float
    superior: float


def intervalo_de_wilson(acertos: int, total: int, z: float = Z_95) -> Intervalo:
    if total <= 0:
        return Intervalo(0.0, 0.0)
    p = acertos / total
    denominador = 1 + z * z / total
    centro = (p + z * z / (2 * total)) / denominador
    margem = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominador
    return Intervalo(round(max(0.0, centro - margem), 4), round(min(1.0, centro + margem), 4))


def p_valor_de_mcnemar(so_a: int, so_b: int) -> float:
    """p-valor bilateral EXATO (binomial) para a hipotese de que A e B erram
    os mesmos itens: `so_a` = itens que so' A acertou, `so_b` = so' B.
    Discordancias zero -> p = 1 (nada a testar)."""

    n = so_a + so_b
    if n == 0:
        return 1.0
    k = min(so_a, so_b)
    cauda = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return round(min(1.0, 2 * cauda), 6)


def bootstrap_pareado(diferencas: list[int], reamostras: int = 2000, seed: int = 42) -> Intervalo:
    """IC 95% da MEDIA das diferencas por item (ex.: acerto_B - acerto_A, em
    {-1, 0, 1}) por bootstrap percentil. Deterministico pela seed."""

    if not diferencas:
        return Intervalo(0.0, 0.0)
    rng = random.Random(seed)
    n = len(diferencas)
    medias = sorted(sum(rng.choice(diferencas) for _ in range(n)) / n for _ in range(reamostras))
    return Intervalo(round(medias[int(0.025 * reamostras)], 4), round(medias[min(reamostras - 1, int(0.975 * reamostras))], 4))


def kappa_de_cohen(a: list[bool], b: list[bool]) -> float | None:
    """Concordancia entre dois avaliadores binarios nos mesmos itens. None se
    vazio; 1.0 se concordam em tudo com margens degeneradas."""

    if not a or len(a) != len(b):
        return None
    n = len(a)
    observada = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    esperada = pa * pb + (1 - pa) * (1 - pb)
    if esperada >= 1.0:
        return 1.0
    return round((observada - esperada) / (1 - esperada), 4)


def percentil(valores: list[float], p: float) -> float | None:
    if not valores:
        return None
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * p
    i = math.floor(k)
    j = min(i + 1, len(ordenados) - 1)
    return round(ordenados[i] + (ordenados[j] - ordenados[i]) * (k - i), 2)
