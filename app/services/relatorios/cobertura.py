"""Cobertura da lei pelos exemplos sinteticos: quanto de cada artigo o
aluno vai de fato ver. Funcoes puras sobre exemplos e descartados.

O plano da sintese sorteia todos os artigos, mas isso so' garante que cada
um foi PEDIDO. Este relatorio mede o que sobrou: quantos exemplos aceitos
tiveram o artigo como alvo, quantas vezes ele foi citado e se o exemplo
citou o proprio alvo (sem isso, a pergunta tratou de outra coisa -- em
geral das definicoes do art. 5o que vao junto no contexto).

Exemplos gravados antes do campo `alvo_id` existir usam `contexto_ids[0]`,
onde o gerador sempre coloca o alvo; `alvos_inferidos` conta quantos.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from app.schemas.sintese import ExemploDescartado, ExemploSintetico
from app.services.corpus.referencias import artigo_de

MENOS_CITADOS = 10
CONCENTRACAO_TOP = 5


class CoberturaDoArtigo(BaseModel):
    artigo_id: str
    alvo_aceitos: int = Field(description="Exemplos aceitos com este artigo como alvo")
    alvo_descartados: int = Field(description="Descartados com este artigo como alvo (so' os que gravaram alvo_id)")
    aceitos_que_citam_o_alvo: int
    citacoes: int = Field(description="Exemplos aceitos que citam o artigo ou algum dispositivo dele")


class RelatorioDeCobertura(BaseModel):
    jobs: list[str]
    exemplos: int
    alvos_inferidos: int = Field(description="Exemplos sem alvo_id; alvo lido de contexto_ids[0]")
    artigos_no_corpus: int
    artigos_como_alvo: int
    artigos_citados: int
    taxa_cita_o_alvo: float | None = Field(description="Aceitos que citam o proprio alvo / aceitos")
    concentracao_top5: float | None = Field(description="Fracao das citacoes que vai aos 5 artigos mais citados")
    nunca_citados: list[str]
    menos_citados: list[CoberturaDoArtigo]
    por_artigo: list[CoberturaDoArtigo] = Field(description="Na ordem da lei")
    aviso: str


def _alvo(exemplo: ExemploSintetico) -> tuple[str | None, bool]:
    """(alvo, inferido)."""

    if exemplo.alvo_id:
        return exemplo.alvo_id, False
    return (exemplo.contexto_ids[0] if exemplo.contexto_ids else None), True


def relatorio(
    artigos: list[str],
    exemplos: list[ExemploSintetico],
    descartados: list[ExemploDescartado],
    jobs: list[str],
) -> RelatorioDeCobertura:
    """`artigos`: ids dos artigos da fonte principal, na ordem da lei."""

    alvo_aceitos: Counter[str] = Counter()
    alvo_descartados: Counter[str] = Counter()
    citam_o_alvo: Counter[str] = Counter()
    citacoes: Counter[str] = Counter()
    inferidos = 0

    for e in exemplos:
        alvo, inferido = _alvo(e)
        inferidos += inferido
        citados = {artigo_de(c) for c in e.artigos_citados}
        citacoes.update(citados)
        if alvo is None:
            continue
        alvo = artigo_de(alvo)
        alvo_aceitos[alvo] += 1
        if alvo in citados:
            citam_o_alvo[alvo] += 1
    for d in descartados:
        if d.alvo_id:
            alvo_descartados[artigo_de(d.alvo_id)] += 1

    por_artigo = [
        CoberturaDoArtigo(
            artigo_id=a, alvo_aceitos=alvo_aceitos[a], alvo_descartados=alvo_descartados[a],
            aceitos_que_citam_o_alvo=citam_o_alvo[a], citacoes=citacoes[a],
        )
        for a in artigos
    ]
    total_de_citacoes = sum(citacoes[a] for a in artigos)
    top = sorted((citacoes[a] for a in artigos), reverse=True)[:CONCENTRACAO_TOP]
    com_alvo = sum(alvo_aceitos.values())

    aviso = (
        "Cobertura medida nos exemplos aceitos pela sintese, antes do filtro. Citacao conta uma vez por exemplo "
        "e por artigo (incisos e paragrafos somam no artigo). Artigos de fontes complementares nao entram."
    )
    if inferidos:
        aviso += f" {inferidos} exemplo(s) sem alvo_id: alvo lido de contexto_ids[0]."
    if descartados and not any(d.alvo_id for d in descartados):
        aviso += " Descartados sem alvo_id: alvo_descartados fica zerado."

    return RelatorioDeCobertura(
        jobs=jobs, exemplos=len(exemplos), alvos_inferidos=inferidos, artigos_no_corpus=len(artigos),
        artigos_como_alvo=sum(1 for a in artigos if alvo_aceitos[a]),
        artigos_citados=sum(1 for a in artigos if citacoes[a]),
        taxa_cita_o_alvo=round(sum(citam_o_alvo.values()) / com_alvo, 4) if com_alvo else None,
        concentracao_top5=round(sum(top) / total_de_citacoes, 4) if total_de_citacoes else None,
        nunca_citados=[a for a in artigos if not citacoes[a]],
        menos_citados=sorted(por_artigo, key=lambda c: (c.citacoes, artigos.index(c.artigo_id)))[:MENOS_CITADOS],
        por_artigo=por_artigo, aviso=aviso,
    )


def _pct(valor: float | None) -> str:
    return "-" if valor is None else f"{valor * 100:.1f}%"


def markdown(r: RelatorioDeCobertura) -> str:
    resumo = [
        f"**Jobs:** {', '.join(r.jobs)} · **Exemplos aceitos:** {r.exemplos}",
        f"**Artigos como alvo:** {r.artigos_como_alvo}/{r.artigos_no_corpus} · "
        f"**Artigos citados:** {r.artigos_citados}/{r.artigos_no_corpus} · "
        f"**Citam o próprio alvo:** {_pct(r.taxa_cita_o_alvo)} · "
        f"**Citações nos 5 mais citados:** {_pct(r.concentracao_top5)}",
    ]
    cabecalho = "| Artigo | Alvo (aceitos) | Alvo (descartados) | Citam o alvo | Citações |"
    separador = "|---|---:|---:|---:|---:|"
    corpo = [
        f"| {c.artigo_id} | {c.alvo_aceitos} | {c.alvo_descartados} | {c.aceitos_que_citam_o_alvo} | {c.citacoes} |"
        for c in r.por_artigo
    ]
    return "\n\n".join(resumo) + "\n\n" + "\n".join([cabecalho, separador, *corpo]) + f"\n\n_{r.aviso}_\n"
