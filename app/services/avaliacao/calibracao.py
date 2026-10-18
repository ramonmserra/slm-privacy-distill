"""Calibracao do juiz contra anotacao HUMANA -- funcoes puras.

Sem isto a metrica de alucinacao nao convence banca: "o juiz disse" nao e'
evidencia. Com >= 100 itens anotados pelos tres autores, o kappa juiz x
humanos (e humanos x humanos, como teto) diz quanto a regua automatica
reproduz o julgamento humano.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from app.core.stats import kappa_de_cohen
from app.schemas.avaliacao import AnotacaoRegistrada, ConcordanciaPorAnotador, ResultadoDoItem, ResumoDeCalibracao

# (anotador, item_id) -> anotacao vigente
_Vigentes = dict[tuple[str, str], AnotacaoRegistrada]


def _vigentes(anotacoes: list[AnotacaoRegistrada]) -> _Vigentes:
    """A ULTIMA anotacao de cada (anotador, item) vale: reanotar corrige."""

    vigentes: _Vigentes = {}
    for a in anotacoes:
        vigentes[(a.anotador, a.item_id)] = a
    return vigentes


def _do_anotador(vigentes: _Vigentes, anotador: str) -> list[AnotacaoRegistrada]:
    return [a for (quem, _), a in vigentes.items() if quem == anotador]


def _concordancia(anotador: str, anotacoes: list[AnotacaoRegistrada], juiz_correto: dict[str, bool], juiz_grounded: dict[str, bool | None]) -> ConcordanciaPorAnotador:
    pares = [a for a in anotacoes if a.item_id in juiz_correto]
    humano = [a.correto for a in pares]
    juiz = [juiz_correto[a.item_id] for a in pares]
    com_grounded = [(a.grounded, juiz_grounded.get(a.item_id)) for a in pares if a.grounded is not None and juiz_grounded.get(a.item_id) is not None]
    return ConcordanciaPorAnotador(
        anotador=anotador, itens=len(pares),
        kappa_correto=kappa_de_cohen(juiz, humano),
        kappa_grounded=kappa_de_cohen([j for _, j in com_grounded], [h for h, _ in com_grounded]) if com_grounded else None,
        acuracia_do_juiz_vs_humano=round(sum(1 for h, j in zip(humano, juiz) if h == j) / len(pares), 4) if pares else None,
    )


def _maioria(vigentes: _Vigentes, juiz: dict[str, bool]) -> dict[str, bool]:
    """Voto da maioria por item, so' onde ha >= 2 anotadores e o juiz julgou.
    Empate descarta o item: nao ha maioria."""

    votos: dict[str, Counter] = {}
    for (_, item_id), a in vigentes.items():
        if item_id in juiz:
            votos.setdefault(item_id, Counter())[a.correto] += 1
    saida: dict[str, bool] = {}
    for item_id, contagem in votos.items():
        if sum(contagem.values()) < 2:
            continue
        (valor, n), *resto = contagem.most_common()
        if resto and resto[0][1] == n:
            continue
        saida[item_id] = valor
    return saida


def _kappa_entre_humanos(vigentes: _Vigentes, anotadores: list[str]) -> float | None:
    kappas: list[float] = []
    for a, b in combinations(anotadores, 2):
        itens_a = {x.item_id: x.correto for x in _do_anotador(vigentes, a)}
        itens_b = {x.item_id: x.correto for x in _do_anotador(vigentes, b)}
        comuns = sorted(set(itens_a) & set(itens_b))
        k = kappa_de_cohen([itens_a[i] for i in comuns], [itens_b[i] for i in comuns]) if comuns else None
        if k is not None:
            kappas.append(k)
    return round(sum(kappas) / len(kappas), 4) if kappas else None


def _leitura(kappa: float | None, n: int) -> str:
    if kappa is None:
        return "Sem itens com anotacao de ao menos dois humanos; anote mais itens para calcular o kappa contra a maioria."
    faixa = "quase perfeita" if kappa >= 0.81 else "substancial" if kappa >= 0.61 else "moderada" if kappa >= 0.41 else "fraca" if kappa >= 0.21 else "pobre"
    aviso = " Amostra pequena: trate como indicativo." if n < 50 else ""
    return f"Concordancia juiz x maioria humana {faixa} (kappa={kappa:.2f}, n={n}; escala de Landis & Koch).{aviso}"


def resumir(run_id: str, itens: list[ResultadoDoItem], anotacoes: list[AnotacaoRegistrada]) -> ResumoDeCalibracao:
    juiz_correto = {r.item.id: r.acerto for r in itens if r.veredito is not None}
    juiz_grounded = {r.item.id: r.veredito.grounded for r in itens if r.veredito is not None}
    vigentes = _vigentes(anotacoes)
    anotadores = sorted({a.anotador for a in vigentes.values()})
    maioria = _maioria(vigentes, juiz_correto)
    kappa_maioria = kappa_de_cohen([juiz_correto[i] for i in maioria], [maioria[i] for i in maioria]) if maioria else None
    return ResumoDeCalibracao(
        run_id=run_id,
        itens_anotados=len({a.item_id for a in vigentes.values()}),
        anotadores=anotadores,
        por_anotador=[_concordancia(a, _do_anotador(vigentes, a), juiz_correto, juiz_grounded) for a in anotadores],
        kappa_correto_maioria=kappa_maioria,
        kappa_entre_humanos=_kappa_entre_humanos(vigentes, anotadores),
        leitura=_leitura(kappa_maioria, len(maioria)),
    )
