"""`app.core.stats`: funcoes puras conferidas contra valores conhecidos."""

from __future__ import annotations

import pytest

from app.core.stats import bootstrap_pareado, intervalo_de_wilson, kappa_de_cohen, percentil, p_valor_de_mcnemar


class TestWilson:
    def test_valores_conhecidos(self):
        ic = intervalo_de_wilson(17, 34)  # 50% em n=34: ~[0.34, 0.66]
        assert ic.inferior == pytest.approx(0.3414, abs=0.002)
        assert ic.superior == pytest.approx(0.6586, abs=0.002)

    def test_extremos_e_vazio(self):
        assert intervalo_de_wilson(0, 10).inferior == 0.0
        assert intervalo_de_wilson(10, 10).superior == 1.0
        assert intervalo_de_wilson(0, 0) == intervalo_de_wilson(0, 0)
        assert (intervalo_de_wilson(0, 0).inferior, intervalo_de_wilson(0, 0).superior) == (0.0, 0.0)

    def test_mais_itens_intervalo_mais_estreito(self):
        pequeno = intervalo_de_wilson(17, 34)
        grande = intervalo_de_wilson(170, 340)
        assert (grande.superior - grande.inferior) < (pequeno.superior - pequeno.inferior)


class TestMcNemar:
    def test_sem_discordancia(self):
        assert p_valor_de_mcnemar(0, 0) == 1.0

    def test_valores_conhecidos(self):
        # 0 vs 5: 2 * (1/32) = 0.0625; 1 vs 9: 2 * (1 + 10)/1024 = 0.0215
        assert p_valor_de_mcnemar(0, 5) == pytest.approx(0.0625)
        assert p_valor_de_mcnemar(1, 9) == pytest.approx(0.021484, abs=1e-6)
        assert p_valor_de_mcnemar(5, 5) == 1.0
        assert p_valor_de_mcnemar(5, 5) == p_valor_de_mcnemar(5, 5)

    def test_simetrico(self):
        assert p_valor_de_mcnemar(2, 8) == p_valor_de_mcnemar(8, 2)


class TestBootstrap:
    def test_deterministico_e_contem_a_media(self):
        diferencas = [1] * 10 + [0] * 20 + [-1] * 5
        a = bootstrap_pareado(diferencas)
        assert a == bootstrap_pareado(diferencas)
        media = sum(diferencas) / len(diferencas)
        assert a.inferior <= media <= a.superior

    def test_vazio(self):
        ic = bootstrap_pareado([])
        assert (ic.inferior, ic.superior) == (0.0, 0.0)


class TestKappa:
    def test_concordancia_total_e_nenhuma(self):
        assert kappa_de_cohen([True, False, True], [True, False, True]) == 1.0
        assert kappa_de_cohen([True, True, False, False], [False, False, True, True]) == -1.0
        assert kappa_de_cohen([], []) is None
        assert kappa_de_cohen([True], [True, False]) is None

    def test_concordancia_ao_acaso_da_zero(self):
        a = [True, True, False, False]
        b = [True, False, True, False]
        assert kappa_de_cohen(a, b) == 0.0


def test_percentil():
    assert percentil([], 0.5) is None
    assert percentil([10.0], 0.95) == 10.0
    assert percentil([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentil([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85, abs=0.01)
