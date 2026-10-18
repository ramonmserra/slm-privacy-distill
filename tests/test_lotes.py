"""`app.services.lotes`: o laco compartilhado de lotes sob orcamento."""

from __future__ import annotations

from app.core.erros import OrcamentoExcedido
from app.services.lotes import ExecutorDeLotes, dividir


def test_dividir():
    assert dividir([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert dividir([], 3) == []
    assert dividir([1, 2], 0) == [[1], [2]]


async def test_sucesso_erro_e_progresso():
    consolidados: list[list[int]] = []
    progresso = {"n": 0}

    async def chamar(lote):
        if lote == [3, 4]:
            raise RuntimeError("falha")
        return sum(lote)

    def consolidar(lote, saida):
        consolidados.append(lote)
        assert saida == sum(lote)

    executor = ExecutorDeLotes(chamar=chamar, consolidar=consolidar, ao_progresso=lambda: progresso.__setitem__("n", progresso["n"] + 1))
    resultado = await executor.executar([[1, 2], [3, 4], [5]], concorrencia=2)
    assert resultado.concluidos == 2 and resultado.com_erro == 1
    assert resultado.orcamento_excedido is None and not resultado.todos_falharam
    assert sorted(consolidados) == [[1, 2], [5]]
    assert progresso["n"] == 3


async def test_orcamento_para_os_lotes_seguintes():
    chamados: list[list[int]] = []

    async def chamar(lote):
        chamados.append(lote)
        if lote == [1]:
            raise OrcamentoExcedido("teto")
        return None

    executor = ExecutorDeLotes(chamar=chamar, consolidar=lambda lote, s: None)
    resultado = await executor.executar([[1], [2], [3]], concorrencia=1)
    assert resultado.orcamento_excedido == "teto"
    assert chamados == [[1]]


async def test_todos_falharam():
    async def chamar(lote):
        raise RuntimeError("x")

    executor = ExecutorDeLotes(chamar=chamar, consolidar=lambda lote, s: None)
    resultado = await executor.executar([[1], [2]], concorrencia=3)
    assert resultado.todos_falharam
