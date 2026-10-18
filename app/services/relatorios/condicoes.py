"""Tabela das condicoes (C0..C6) a partir dos runs de avaliacao, em JSON e
em Markdown pronto para o texto do TCC. Funcoes puras sobre `RunDeAvaliacao`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.avaliacao import RunDeAvaliacao


class LinhaDaTabela(BaseModel):
    run_id: str
    rotulo: str = Field(description="Nome da condicao dado pelo operador, ou o modelo")
    modelo: str
    conjunto: str
    com_rag: bool
    itens_julgados: int
    acuracia: float | None
    acuracia_ic95: tuple[float, float] | None
    taxa_de_alucinacao: float | None
    taxa_de_alucinacao_ic95: tuple[float, float] | None
    abstencoes_adequadas: str
    taxa_citou_referencia: float | None
    latencia_p50_ms: float | None
    custo_usd_por_item: float | None
    versao_da_medida: str


class TabelaDeCondicoes(BaseModel):
    linhas: list[LinhaDaTabela]
    versoes_da_medida: list[str]
    comparaveis: bool = Field(description="Todos os runs na mesma versao da medida")
    aviso: str


def linha(run: RunDeAvaliacao, rotulo: str | None = None) -> LinhaDaTabela:
    m = run.metricas
    return LinhaDaTabela(
        run_id=run.id, rotulo=rotulo or run.modelo, modelo=run.modelo, conjunto=run.parametros.conjunto.value,
        com_rag=run.parametros.com_rag, itens_julgados=m.julgados, acuracia=m.acuracia, acuracia_ic95=m.acuracia_ic95,
        taxa_de_alucinacao=m.taxa_de_alucinacao, taxa_de_alucinacao_ic95=m.taxa_de_alucinacao_ic95,
        abstencoes_adequadas=f"{m.abstencoes_adequadas}/{m.abstencoes_esperadas}" if m.abstencoes_esperadas else "-",
        taxa_citou_referencia=m.taxa_citou_referencia, latencia_p50_ms=m.latencia_p50_ms,
        custo_usd_por_item=m.custo_usd_por_item, versao_da_medida=run.versao_da_medida,
    )


def tabela(runs: list[RunDeAvaliacao], rotulos: dict[str, str] | None = None) -> TabelaDeCondicoes:
    rotulos = rotulos or {}
    linhas = [linha(r, rotulos.get(r.id)) for r in runs]
    versoes = sorted({r.versao_da_medida for r in runs})
    comparaveis = len(versoes) <= 1
    aviso = (
        "Intervalos de Wilson a 95%. Acuracia = veredito 'correto' do juiz; alucinacao = lastro negado pelo juiz ou "
        "artigo inexistente na prosa. Custo de modelo local = 0 por token; o custo do aluno e' tempo e hardware."
    )
    if not comparaveis:
        aviso += " ATENCAO: runs em versoes da medida diferentes; compare so' dentro da mesma versao."
    return TabelaDeCondicoes(linhas=linhas, versoes_da_medida=versoes, comparaveis=comparaveis, aviso=aviso)


def _pct(valor: float | None, ic: tuple[float, float] | None) -> str:
    if valor is None:
        return "-"
    texto = f"{valor * 100:.1f}%"
    if ic:
        texto += f" [{ic[0] * 100:.0f}–{ic[1] * 100:.0f}]"
    return texto


def markdown(t: TabelaDeCondicoes) -> str:
    cabecalho = "| Condição | Modelo | Conjunto | RAG | n | Acurácia [IC95] | Alucinação [IC95] | Abstenção adequada | Citou referência | Latência p50 (ms) | Custo/item (US$) |"
    separador = "|---|---|---|---|---:|---|---|---|---|---:|---:|"
    corpo = [
        f"| {l.rotulo} | `{l.modelo}` | {l.conjunto} | {'sim' if l.com_rag else 'não'} | {l.itens_julgados} | "
        f"{_pct(l.acuracia, l.acuracia_ic95)} | {_pct(l.taxa_de_alucinacao, l.taxa_de_alucinacao_ic95)} | {l.abstencoes_adequadas} | "
        f"{_pct(l.taxa_citou_referencia, None)} | {l.latencia_p50_ms if l.latencia_p50_ms is not None else '-'} | "
        f"{f'{l.custo_usd_por_item:.4f}' if l.custo_usd_por_item is not None else '-'} |"
        for l in t.linhas
    ]
    rodape = f"\n_{t.aviso}_\n"
    return "\n".join([cabecalho, separador, *corpo]) + rodape
